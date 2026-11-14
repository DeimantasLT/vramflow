#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""`vramflow-func ingest` funkciniai testai (planas docs/planai/2026-11-11-production-grade.md §P3, user story P3).

Medžiaga — tik sintetinė (scripts/linux/62-make-av-clip.sh; nėra — sugeneruojama), siuntimas — tik loopback
(tests/perf/ts_burst.py: UDP / RTP pagal PCR, pliūpsniai, sulaikymai, pagreitintas laikrodis; ffmpeg — HLS segmentuotojas ir SRT).
Matavimas — `verify --mode ingest` (tiesioginis MXL skaitytojas: kadro žymė ↔ 1 mėginio impulsas, mėginio tikslumu).

ingest_av_offset[fmt/dec]   — 1080i50 ir 1080p25, NVDEC (G) ir programinis (C), UDP realiu laiku: kiekvieno impulso |A/V| ≤ 1 laukas
                              (20 ms; 1080p25 — 1 kadras, 40 ms; išmatuota — rezultate), video ≥ 99,8 % grain'ų, 0 pavėluotų, exit 0.
ingest_bursty_no_drift      — 2 s medžiagos per 50 ms, 120 s, --latency-ms 3000: A/V pokytis tarp t ≈ 10 s ir t ≈ 110 s ≤ 1 ms,
                              drift_slips == 0, relock == 0, 0 pavėluotų.
ingest_marker_continuity    — 15 s klipas kilpa 40 s: žymės iš eilės, išskyrus sandūras (po vieną šuolį ir relock_cont sandūrai),
                              indeksai be spragų.
ingest_pts_order_not_arrival — siuntėjas kas 5 s sulaiko paketus 300 ms: atitikimas žymė → indeksas nepakinta (0 map_changes).
ingest_pts_wrap             — PTS persiverčia per 2^33: indeksai be spragos ir pasikartojimo, wraps == 1, be perfiksavimo.
ingest_drift_accelerated    — siuntėjas 0,5 % greitesnis 120 s (--max-comp-ppm 6000): drift_slips ≥ 1, |A/V| ≤ 1 laukas visada,
                              video trūkiai ≤ 1 kadras, garse nė vieno tylos tarpo / trūkio, audio_resyncs == 0.
ingest_hls / ingest_srt / ingest_rtp — tie patys A/V ir tolydumo kriterijai per HLS (ffmpeg segmentuotojas + http.server
                              127.0.0.1), SRT (ffmpeg listener; SKIP, jei FFmpeg be srt), RTP (PT 33).
ingest_two_formats          — 1080i50 + 1080p25 viename procese: abiem — A/V ir ≥ 99,8 %.
ingest_hevc                 — HEVC 1080p25 (jei FFmpeg turi libx265; kitaip SKIP): tie patys kriterijai.
ingest_reconnect            — šaltinis dingsta 30 s ir grįžta (siuntėjas iš naujo): input_reconnects ≥ 1, relock_fresh ≥ 1, po grįžimo
                              |A/V| ≤ 1 laukas.
ingest_overload_skip        — dirbtinai lėtas dekoderis: kadrai, kurie nebespėja, praleidžiami iki rakto kadro (skip_to_key ≥ 1),
                              vaizdas atsistato kiekvieną GOP (ne visiškas užtemimas).
ingest_progressive_chroma   — 1080p25: grain'ai bitas į bitą su programiniu dekoderiu (--ref) ir su FFmpeg bilinear 4:2:2 (PSNR inf).
ingest_bad_url / ingest_no_video / ingest_format_change / ingest_unsupported — neigiami ir kraštiniai (US 8, 11, 10, 9).
Aplinka: INGEST_CLIPS (numatyta ~/media-test/av), INGEST_QUICK=1 — sutrumpintos trukmės (kūrimui; ne priėmimui).
"""
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time

import functional as F

REPO = pathlib.Path(__file__).resolve().parents[2]
PERF = REPO / "tests" / "perf"
CLIPS = pathlib.Path(os.environ.get("INGEST_CLIPS", str(pathlib.Path.home() / "media-test/av")))
QUICK = os.environ.get("INGEST_QUICK") == "1"
BASE_PORT = 26000


def dur(s):
    return max(15, s // 3) if QUICK else s


def ensure_clips():
    need = ["av_1080i50.ts", "av_1080p25.ts", "fmtchange.ts", "wrap_1080i50.ts", "audio_only_90.ts", "bad_422.ts", "bad_10bit.ts", "bad_bff.ts",
            "fmt_1080i50_short.ts", "ref_crc_1080p25.json"]
    if all((CLIPS / n).exists() for n in need):
        return None
    r = subprocess.run(["bash", str(REPO / "scripts/linux/62-make-av-clip.sh"), str(CLIPS)], capture_output=True, text=True,
                       env=dict(F.ENV, FUNC_BIN=F.FUNC_BIN))
    if r.returncode != 0 or not all((CLIPS / n).exists() for n in need):
        return f"klipų sugeneruoti nepavyko: {r.stderr[-400:]}"
    return None


class Sender:
    """ts_burst.py (UDP / RTP) arba bet kokia komanda (ffmpeg) kaip procesas."""

    def __init__(self, out, tag, cmd):
        self.log = pathlib.Path(out) / f"{tag}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(cmd, stdout=open(self.log, "w"), stderr=subprocess.STDOUT, env=F.ENV)

    @classmethod
    def ts(cls, out, tag, clip, port, *extra):
        return cls(out, tag, [sys.executable, str(PERF / "ts_burst.py"), str(clip), "--port", str(port), *map(str, extra)])

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()

    def text(self):
        return self.log.read_text(errors="replace") if self.log.exists() else ""


def run_case(out, tid, inputs, senders, seconds, ingest_args=(), verify_flows=None, verify_args=(), warm=6, impl="gpu",
             audio=True, video=True):
    """Paleidžia ingest ir verify --mode ingest; grąžina (ingest santrauka, verify santrauka, ingest exit, ingest stderr)."""
    d = F.fresh_domain(tid.replace("[", "-").replace("]", "").replace("/", "-"))
    storage = "host" if impl == "cpu" else "cuda-linear"
    v = F.vcommon(impl, storage)
    args = []
    for u in inputs:
        args += ["--input", u]
    ing = vf = None
    snd = []
    try:
        ing = F.FProc(out, tid, "ingest", d, *args, "--duration", seconds + warm + 8, "--warmup", warm, *v, *ingest_args)
        for s in senders:
            snd.append(s())
        nflows = len(inputs) if verify_flows is None else verify_flows[1]
        base = 0 if verify_flows is None else verify_flows[0]
        # verify langas = seconds: --duration skaičiuojama nuo starto, --warmup jos pradžią praleidžia
        vf = F.FProc(out, tid, "verify", d, "--mode", "ingest", "--flows", nflows, "--flow-base", base, "--duration", seconds + warm,
                     "--warmup", warm, "--lag-fields", 3, "--audio", 1 if audio else 0, "--video", 1 if video else 0,
                     "--input-wait-ms", 30000, *F.vimpl(impl, storage), *verify_args)
        rv = vf.wait(timeout=seconds + warm + 90)
        ri = ing.wait(timeout=seconds + warm + 90)
    finally:
        for pr in (vf, ing):
            if pr is not None:
                pr.term()
        for s in snd:
            s.stop()
    err = ing.err_path.read_text(errors="replace") if ing is not None and ing.err_path.exists() else ""
    return F.vsummary(ing, "ingest"), F.vsummary(vf, "verify-ingest"), ri, err, snd


def av_stats(sv, n=0):
    ing = ((sv or {}).get("ingest") or [{}])
    x = ing[n] if n < len(ing) else {}
    pairs = x.get("pairs") or []
    offs = [p[1] / 1000.0 for p in pairs]
    return {"pairs": len(pairs), "mean_ms": round(sum(offs) / len(offs), 3) if offs else None,
            "max_abs_ms": round(max(abs(o) for o in offs), 3) if offs else None, "offs": offs, "raw": pairs,
            "jumps": x.get("jumps") or [], "map_changes": x.get("map_changes") or [], "disc": x.get("disc") or [], "zero": x.get("zero") or []}


def ratio_ok(fv):
    total = sum(fv.get(k, 0) for k in ("present", "missing", "invalid", "too_late"))
    return (fv.get("present", 0) / total) if total else 0.0


def flow(s, n=0):
    fl = (s or {}).get("flows") or []
    return fl[n] if n < len(fl) else {}


def base_checks(si, sv, ri, limit_ms, seconds, n=0, want_rc=(0,), src=None):
    """Bendri kriterijai (US 1–2): A/V kiekvienam impulsui, ≥ 99,8 % grain'ų, 0 pavėluotų, exit. n — verify flow'as, src —
    ingest šaltinis (numatyta = n)."""
    fi, fv = flow(si, n if src is None else src), flow(sv, n)
    a = av_stats(sv, n)
    ok_ratio = ratio_ok(fv)
    problems = []
    if ri not in want_rc:
        problems.append(f"ingest exit {ri}")
    if a["pairs"] < max(3, int(seconds * 0.8)):
        problems.append(f"A/V porų {a['pairs']} (< {int(seconds * 0.8)})")
    if a["max_abs_ms"] is None or a["max_abs_ms"] > limit_ms:
        problems.append(f"|A/V| maks. {a['max_abs_ms']} ms > {limit_ms}")
    if ok_ratio < 0.998:
        problems.append(f"video {ok_ratio:.4%} < 99,8 %")
    if fi.get("late_frames", 0) != 0:
        problems.append(f"late_frames {fi.get('late_frames')}")
    if fv.get("parity_mismatch", 0):
        problems.append(f"parity_mismatch {fv.get('parity_mismatch')}")
    return problems, a, ok_ratio


def summary_line(si, sv, a, ok_ratio, n=0, src=None):
    fi = flow(si, n if src is None else src)
    keys = ("fields_ok", "frames_decoded", "late_frames", "missing_frames", "repeat_frames", "slip_drop_frames", "skip_to_key", "drift_slips",
            "relock_cont", "relock_fresh", "audio_resyncs", "audio_late_samples", "input_reconnects", "decode_errors")
    fv = flow(sv, n)
    return (f"A/V vid. {a['mean_ms']} ms, maks. |{a['max_abs_ms']}| ms ({a['pairs']} impulsų); video {ok_ratio:.4%}; "
            f"ingest {{{', '.join(f'{k}={fi.get(k)}' for k in keys if k in fi)}}}; verify jumps={fv.get('marker_jumps', 0)} "
            f"map_changes={fv.get('map_changes', 0)} audio_zero_runs={fv.get('audio_zero_runs')} audio_disc={fv.get('audio_disc')}")


CASES = [("1080i50", "nvdec", "gpu", 20.0), ("1080i50", "sw", "cpu", 20.0), ("1080p25", "nvdec", "gpu", 40.0), ("1080p25", "sw", "cpu", 40.0)]


def ingest_av_offset(out):
    res = []
    if (why := ensure_clips()):
        return [F.result("ingest_av_offset", "FAIL", why=why)]
    for k, (fmt, dec, impl, limit) in enumerate(CASES):
        tid = f"ingest_av_offset[{fmt}/{dec}]"
        port = BASE_PORT + k
        T = dur(60)
        si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda p=port, f=fmt: Sender.ts(out, f"{tid}/snd", CLIPS / f"av_{f}.ts", p)],
                                      T, ("--decoder", dec), impl=impl)
        problems, a, r = base_checks(si, sv, ri, limit, T)
        res.append(F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") + summary_line(si, sv, a, r),
                            av_mean_ms=a["mean_ms"], av_max_abs_ms=a["max_abs_ms"], ingest=si))
    return res


def ingest_hevc(out):
    """HEVC 1080p25 (libx265 klipas, jei FFmpeg turi libx265): tas pats kelias (nvdec = hevc + CUDA hwaccel) — A/V ≤ 1 kadras,
    ≥ 99,8 % grain'ų, 0 pavėluotų. Klipo nėra — SKIP su priežastimi."""
    if (why := ensure_clips()):
        return [F.result("ingest_hevc", "FAIL", why=why)]
    clip = CLIPS / "hevc_1080p25.ts"
    if not clip.exists():
        return [F.result("ingest_hevc", "SKIP", why="nėra hevc_1080p25.ts (FFmpeg be libx265)")]
    tid, port = "ingest_hevc", BASE_PORT + 25
    T = 8  # klipas 20 s: ingest (T + 3 + 8 s) baigiasi anksčiau už siuntėją
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda: Sender.ts(out, f"{tid}/snd", clip, port)], T, warm=3)
    problems, a, r = base_checks(si, sv, ri, 40.0, T)
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + " | " + summary_line(si, sv, a, r), ingest=si)]


def window_offset(a, t_rel, rate, half=3.0):
    """A/V vidurkis lange aplink t_rel (s nuo pirmo impulso)."""
    if not a["raw"]:
        return None
    i0 = a["raw"][0][0]
    xs = [p[1] / 1000.0 for p in a["raw"] if abs((p[0] - i0) / rate - t_rel) <= half]
    return sum(xs) / len(xs) if xs else None


def ingest_bursty_no_drift(out):
    if (why := ensure_clips()):
        return [F.result("ingest_bursty_no_drift", "FAIL", why=why)]
    tid, port = "ingest_bursty_no_drift", BASE_PORT + 10
    T = dur(120)
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"],
                                  [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port, "--burst-s", 2, "--burst-ms", 50)], T,
                                  ("--latency-ms", 3000), warm=8)
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    fi = flow(si)
    w10, w110 = window_offset(a, 10, 50), window_offset(a, T - 10, 50)
    delta = None if w10 is None or w110 is None else round(abs(w110 - w10), 3)
    if delta is None or delta > 1.0:
        problems.append(f"A/V pokytis 10 s → {T - 10} s = {delta} ms > 1")
    if fi.get("drift_slips", 0) != 0:
        problems.append(f"drift_slips {fi.get('drift_slips')}")
    if fi.get("relock_fresh", 0) + fi.get("relock_cont", 0) != 0:
        problems.append(f"relock {fi.get('relock_fresh')}+{fi.get('relock_cont')}")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | A/V ties 10 s {w10}, ties {T - 10} s {w110} ms (Δ {delta}) | " + summary_line(si, sv, a, r),
                     av_delta_ms=delta, ingest=si)]


def ingest_marker_continuity(out):
    if (why := ensure_clips()):
        return [F.result("ingest_marker_continuity", "FAIL", why=why)]
    tid, port = "ingest_marker_continuity", BASE_PORT + 11
    T = 40
    si, sv, ri, err, snd = run_case(out, tid, [f"udp://127.0.0.1:{port}"],
                                    [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "fmt_1080i50_short.ts", port, "--loop")], T, warm=6)
    fi, fv = flow(si), flow(sv)
    a = av_stats(sv)
    splices = fi.get("relock_cont", 0)
    problems = []
    # 15 s klipas, langas 40 s: 2–3 sandūros; kiekviena — lygiai vienas žymių šuolis ir vienas atvaizdžio pokytis
    if not 2 <= splices <= 3:
        problems.append(f"relock_cont {splices} (laukta 2–3 sandūros)")
    if fi.get("relock_fresh", 0):
        problems.append(f"relock_fresh {fi.get('relock_fresh')}")
    if fv.get("marker_jumps", 0) > splices or fv.get("map_changes", 0) > splices:
        problems.append(f"žymių šuoliai {fv.get('marker_jumps')} / atvaizdžio pokyčiai {fv.get('map_changes')} > sandūrų {splices}")
    if fv.get("missing", 0) + fv.get("invalid", 0) != 0:
        problems.append(f"indeksai be turinio: missing {fv.get('missing')} invalid {fv.get('invalid')}")
    if a["max_abs_ms"] is None or a["max_abs_ms"] > 20:
        problems.append(f"|A/V| {a['max_abs_ms']} ms")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | sandūros {splices}, šuoliai {a['jumps'][:6]} | " + summary_line(si, sv, a, ratio_ok(fv)), ingest=si)]


def ingest_pts_order_not_arrival(out):
    if (why := ensure_clips()):
        return [F.result("ingest_pts_order_not_arrival", "FAIL", why=why)]
    tid, port = "ingest_pts_order_not_arrival", BASE_PORT + 12
    T = dur(60)
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"],
                                  [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port, "--stall-every-s", 5, "--stall-ms", 300)], T)
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    fv = flow(sv)
    if fv.get("map_changes", 0) or fv.get("marker_jumps", 0):
        problems.append(f"žymė → indeksas pakito: map_changes {fv.get('map_changes')} marker_jumps {fv.get('marker_jumps')}")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + " | " + summary_line(si, sv, a, r), ingest=si)]


def ingest_drift_accelerated(out):
    if (why := ensure_clips()):
        return [F.result("ingest_drift_accelerated", "FAIL", why=why)]
    tid, port = "ingest_drift_accelerated", BASE_PORT + 13
    T = dur(120)
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"],
                                  [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port, "--speed", 1.005)], T,
                                  ("--max-comp-ppm", 6000), warm=6)
    fi, fv = flow(si), flow(sv)
    a = av_stats(sv)
    problems = []
    if fi.get("drift_slips", 0) < 1:
        problems.append(f"drift_slips {fi.get('drift_slips')} < 1")
    if a["max_abs_ms"] is None or a["max_abs_ms"] > 20:
        problems.append(f"|A/V| maks. {a['max_abs_ms']} ms > 20")
    # interlaced: praleistas vienas kadras (laukų pora) — žymės žingsnis +3, pakartotas vienas kadras — −1 (2F+1 → 2F); daugiau
    # nei vienas kadras iš eilės (žingsnis ≥ 5 arba du pakartojimai per 4 laukus) — trūkis > 1 kadro
    bad_steps = [j for j in a["jumps"] if j[1] not in (-1, 1, 3)]
    reps = [j[0] for j in a["jumps"] if j[1] == -1]
    bad_steps += [(x, "pakartota 2 kadrai") for x, y in zip(reps, reps[1:]) if y - x <= 4]
    if bad_steps:
        problems.append(f"video trūkiai > 1 kadro: {bad_steps[:5]}")
    if fv.get("missing", 0) + fv.get("invalid", 0):
        problems.append(f"spragos: missing {fv.get('missing')} invalid {fv.get('invalid')}")
    if fv.get("audio_zero_runs", 1) or fv.get("audio_disc", 1):
        problems.append(f"garso trūkiai: tylos {fv.get('audio_zero_runs')} trūkiai {fv.get('audio_disc')} {a['disc'][:5]}")
    if fi.get("audio_resyncs", 0):
        problems.append(f"audio_resyncs {fi.get('audio_resyncs')}")
    ppm = fi.get("drift_ppm_abs_x1000", 0) / 1000.0 * (-1 if fi.get("drift_ppm_negative") else 1)
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | drift_ppm {ppm:.0f} | " + summary_line(si, sv, a, ratio_ok(fv)), drift_ppm=ppm, ingest=si)]


def ingest_pts_wrap(out):
    """US 5: PTS persiverčia per 2^33 (klipas su -output_ts_offset, persivertimas ~20-ą s): indeksai tęsiasi be spragos ir be
    pasikartojimo, A/V ≤ 1 laukas, wraps == 1, perfiksavimų nėra (relock_cont persivertimui nepadidėja)."""
    if (why := ensure_clips()):
        return [F.result("ingest_pts_wrap", "FAIL", why=why)]
    tid, port = "ingest_pts_wrap", BASE_PORT + 24
    T = 24  # klipas 40 s: ingest (T + 4 + 8) turi baigtis anksčiau už siuntėją (kitaip pabaigoje — tyla ir reconnect)
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "wrap_1080i50.ts", port)], T,
                                  warm=4)
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    fi, fv = flow(si), flow(sv)
    if fi.get("wraps", 0) != 1:
        problems.append(f"wraps {fi.get('wraps')} (laukta 1)")
    if fi.get("relock_cont", 0) + fi.get("relock_fresh", 0) + fi.get("discontinuities", 0):
        problems.append(f"perfiksavimai cont {fi.get('relock_cont')} fresh {fi.get('relock_fresh')} nutrūkimai {fi.get('discontinuities')}")
    if fv.get("marker_jumps", 0) or fv.get("map_changes", 0):
        problems.append(f"indeksai: šuoliai {fv.get('marker_jumps')} atvaizdžio pokyčiai {fv.get('map_changes')}")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + " | " + summary_line(si, sv, a, r), ingest=si)]

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ingest_hls(out):
    if (why := ensure_clips()):
        return [F.result("ingest_hls", "FAIL", why=why)]
    tid = "ingest_hls"
    T = dur(60)
    work = pathlib.Path(out) / tid / "hls"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    hport = free_port()
    seg = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-i", str(CLIPS / "av_1080i50.ts"), "-c", "copy",
                            "-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+omit_endlist",
                            str(work / "live.m3u8")], stdout=subprocess.DEVNULL, stderr=open(work / "seg.log", "w"))
    http = subprocess.Popen([sys.executable, "-m", "http.server", str(hport), "--bind", "127.0.0.1", "--directory", str(work)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.monotonic()
        while not (work / "live.m3u8").exists() and time.monotonic() - t0 < 20:
            time.sleep(0.2)
        time.sleep(4.5)  # bent 2 segmentai grojaraštyje
        si, sv, ri, err, _ = run_case(out, tid, [f"http://127.0.0.1:{hport}/live.m3u8"], [], T, warm=10)
    finally:
        for p in (seg, http):
            p.terminate()
            p.wait()
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    fi = flow(si)
    lat = (si or {}).get("proc", {}).get("latency_ms")
    if lat is None or lat < 3 * 2000:
        problems.append(f"latencija {lat} ms < 3 segmentai")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | latencija {lat} ms | " + summary_line(si, sv, a, r),
                     ingest=si, relock=fi.get("relock_fresh"))]


def ingest_srt(out):
    if (why := ensure_clips()):
        return [F.result("ingest_srt", "FAIL", why=why)]
    tid = "ingest_srt"
    protos = subprocess.run(["ffmpeg", "-hide_banner", "-protocols"], capture_output=True, text=True).stdout.split()
    if "srt" not in protos:
        return [F.result(tid, "SKIP", why="sisteminis FFmpeg be srt protokolo")]
    T = dur(60)
    port = free_port()
    url = f"srt://127.0.0.1:{port}"
    snd = lambda: Sender(out, f"{tid}/snd", ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-i", str(CLIPS / "av_1080i50.ts"),
                                             "-c", "copy", "-f", "mpegts", f"{url}?mode=caller&latency=120000"])
    si, sv, ri, err, _ = run_case(out, tid, [f"{url}?mode=listener&latency=120000"], [snd], T)
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + " | " + summary_line(si, sv, a, r), ingest=si)]


def ingest_rtp(out):
    if (why := ensure_clips()):
        return [F.result("ingest_rtp", "FAIL", why=why)]
    tid, port = "ingest_rtp", BASE_PORT + 14
    T = dur(60)
    si, sv, ri, err, _ = run_case(out, tid, [f"rtp://127.0.0.1:{port}"], [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port, "--rtp")], T)
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + " | " + summary_line(si, sv, a, r), ingest=si)]


def ingest_two_formats(out):
    if (why := ensure_clips()):
        return [F.result("ingest_two_formats", "FAIL", why=why)]
    tid = "ingest_two_formats"
    T = dur(45)
    p0, p1 = BASE_PORT + 15, BASE_PORT + 16
    d = F.fresh_domain(tid)
    v = F.vcommon("gpu", "cuda-linear")
    ing = None
    snd = []
    vfs = []
    try:
        ing = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{p0}", "--input", f"udp://127.0.0.1:{p1}", "--duration", T + 14,
                      "--warmup", 6, *v)
        snd = [Sender.ts(out, f"{tid}/snd0", CLIPS / "av_1080i50.ts", p0), Sender.ts(out, f"{tid}/snd1", CLIPS / "av_1080p25.ts", p1)]
        for n in (0, 1):  # verify — vienas formatas procesui
            vfs.append(F.FProc(out, f"{tid}/v{n}", "verify", d, "--mode", "ingest", "--flows", 1, "--flow-base", n, "--duration", T + 6, "--warmup", 6,
                               "--lag-fields", 3, "--audio", 1, "--input-wait-ms", 30000, "--impl", "gpu", "--storage", "cuda-linear"))
        for vf in vfs:
            vf.wait(timeout=T + 90)
        ri = ing.wait(timeout=T + 90)
    finally:
        for pr in [*vfs, ing]:
            if pr is not None:
                pr.term()
        for s in snd:
            s.stop()
    si = F.vsummary(ing, "ingest")
    res = []
    for n, (fmt, limit) in enumerate((("1080i50", 20.0), ("1080p25", 40.0))):
        sv = F.vsummary(vfs[n], "verify-ingest") if n < len(vfs) else None
        problems, a, r = base_checks(si, sv, ri, limit, T, n=0, src=n)  # kiekvienas verify — vienas flow'as; ingest — šaltinis n
        res.append(F.result(f"{tid}[{fmt}]", "FAIL" if problems else "PASS", why="; ".join(problems) + " | " + summary_line(si, sv, a, r, n=0, src=n)))
    return res


def ingest_reconnect(out):
    if (why := ensure_clips()):
        return [F.result("ingest_reconnect", "FAIL", why=why)]
    tid, port = "ingest_reconnect", BASE_PORT + 17
    UP1, GAP, UP2 = 20, 30, 25
    T = UP1 + GAP + UP2
    d = F.fresh_domain(tid)
    v = F.vcommon("gpu", "cuda-linear")
    ing = vf = None
    s1 = s2 = None
    try:
        ing = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{port}", "--duration", T + 12, "--warmup", 6, *v)
        s1 = Sender.ts(out, f"{tid}/snd1", CLIPS / "av_1080i50.ts", port)
        vf = F.FProc(out, tid, "verify", d, "--mode", "ingest", "--flows", 1, "--duration", T + 6, "--warmup", 6, "--lag-fields", 3, "--audio", 1,
                     "--input-wait-ms", 30000, "--impl", "gpu", "--storage", "cuda-linear")
        time.sleep(UP1)
        s1.stop()
        time.sleep(GAP)
        s2 = Sender.ts(out, f"{tid}/snd2", CLIPS / "av_1080i50.ts", port)
        vf.wait(timeout=T + 90)
        ri = ing.wait(timeout=T + 90)
    finally:
        for pr in (vf, ing):
            if pr is not None:
                pr.term()
        for s in (s1, s2):
            if s is not None:
                s.stop()
    si, sv = F.vsummary(ing, "ingest"), F.vsummary(vf, "verify-ingest")
    fi = flow(si)
    a = av_stats(sv)
    i0 = a["raw"][0][0] if a["raw"] else 0
    after = [p[1] / 1000.0 for p in a["raw"] if (p[0] - i0) / 50 > UP1 + GAP - 6]
    problems = []
    if fi.get("input_reconnects", 0) < 1:
        problems.append(f"input_reconnects {fi.get('input_reconnects')}")
    if fi.get("relock_fresh", 0) < 1:
        problems.append(f"relock_fresh {fi.get('relock_fresh')}")
    if len(after) < UP2 * 0.6:
        problems.append(f"po grįžimo A/V porų {len(after)}")
    if not after or max(abs(x) for x in after) > 20:
        problems.append(f"po grįžimo |A/V| {max((abs(x) for x in after), default=None)} ms")
    if ri not in (0, 4):
        problems.append(f"exit {ri}")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | po grįžimo A/V maks. {max((abs(x) for x in after), default=None)} ms ({len(after)}) | "
                     + summary_line(si, sv, a, ratio_ok(flow(sv))), ingest=si)]


def ingest_overload_skip(out):
    if (why := ensure_clips()):
        return [F.result("ingest_overload_skip", "FAIL", why=why)]
    tid, port = "ingest_overload_skip", BASE_PORT + 18
    T = 25
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080p25.ts", port)], T,
                                  ("--test-decode-delay-ms", 48))
    fi, fv = flow(si), flow(sv)
    problems = []
    if fi.get("skip_to_key", 0) < 1:
        problems.append(f"skip_to_key {fi.get('skip_to_key')}")
    frac = ratio_ok(fv)
    # 48 ms paketui (> 40 ms kadras): be praleidimo dekoderis atsiliktų vis daugiau ir po ~1 s visi kadrai būtų pavėlavę (0 %)
    if frac < 0.25:
        problems.append(f"vaizdo dalis {frac:.1%} < 25 % (sugriuvo)")
    tail = fv.get("present", 0)
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | vaizdo dalis {frac:.1%}, present {tail}, skip_to_key {fi.get('skip_to_key')}, "
                     f"skip_dropped_pkts {fi.get('skip_dropped_pkts')}, late_frames {fi.get('late_frames')}", ingest=si)]


def ingest_progressive_chroma(out):
    if (why := ensure_clips()):
        return [F.result("ingest_progressive_chroma", "FAIL", why=why)]
    tid, port = "ingest_progressive_chroma", BASE_PORT + 19
    dump = pathlib.Path(out) / tid / "dump"
    shutil.rmtree(dump, ignore_errors=True)
    dump.mkdir(parents=True)
    T = 20
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080p25.ts", port)], T,
                                  verify_args=("--ref", CLIPS / "ref_crc_1080p25.json", "--every", 5, "--dump-dir", dump, "--dump-count", 3))
    fv = flow(sv)
    problems = []
    if fv.get("dec_bitexact", 0) < 50 or fv.get("dec_mismatch", 1) != 0 or fv.get("ref_marker_unknown", 0):
        problems.append(f"bitų tikslumas: bitexact {fv.get('dec_bitexact')} mismatch {fv.get('dec_mismatch')} unknown {fv.get('ref_marker_unknown')}")
    psnrs = []
    sys.path.insert(0, str(PERF))
    import chroma_psnr  # tests/perf/chroma_psnr.py — tas pats FFmpeg psnr palyginimas
    for f in sorted(dump.glob("ingest_0_m*.v210"))[:3]:
        m = int(f.stem.split("_m")[1])
        k = m - 1000  # klipo kadras (barkodas nuo --first-field 1000)
        ref = dump / f"ref_{m}.yuv"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-c:v", "h264", "-i", str(CLIPS / "av_1080p25.ts"),
                        "-vf", f"select=eq(n\\,{k}),scale=flags=bilinear,format=yuv422p10le", "-frames:v", "1", "-f", "rawvideo", str(ref)],
                       check=False)
        p = chroma_psnr.psnr("ffmpeg", 1920, 1080, f, ref) if ref.exists() else None
        psnrs.append((m, p))
        if not p or min(p.get("u", 0), p.get("v", 0)) < 60 or p.get("y", 0) < 99:
            problems.append(f"PSNR prieš FFmpeg bilinear (žymė {m}): {p}")
    if not psnrs:
        problems.append("nėra išsaugotų grain'ų")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | bitexact {fv.get('dec_bitexact')} mismatch {fv.get('dec_mismatch')} | PSNR {psnrs}", ingest=si)]


def flows_in(domain):
    p = pathlib.Path(domain)
    return sorted(x.name for x in p.glob("*.mxl-flow")) if p.exists() else []


def ingest_bad_url(out):
    tid = "ingest_bad_url"
    d = F.fresh_domain(tid)
    url = f"udp://127.0.0.1:{BASE_PORT + 20}"
    t0 = time.monotonic()
    p = F.FProc(out, tid, "ingest", d, "--input", url, "--input-wait-s", 5, *F.vcommon("gpu", "cuda-linear"))
    rc = p.wait(timeout=30)
    dt = time.monotonic() - t0
    s = F.vsummary(p, "ingest") or {}
    fatal = s.get("fatal", "")
    left = flows_in(d)
    ok = rc not in (0, None) and dt <= 6.0 and url in fatal and ("nėra duomenų" in fatal or "nepavyko" in fatal) and not left
    res = [F.result(tid, "PASS" if ok else "FAIL", why=f"exit {rc} per {dt:.1f} s, fatal {fatal!r}, flow'ai domene {left}")]
    # US 7 (antra dalis): protokolas, kurio sisteminis FFmpeg neturi — aiškus „<schema> protokolas nepalaikomas“, ne bendra klaida.
    # Šiame FFmpeg srt yra, todėl kelias tikrinamas neegzistuojančia schema (tas pats avio_enum_protocols patikrinimas).
    url2 = "xsrt://127.0.0.1:9000"
    p2 = F.FProc(out, f"{tid}/proto", "ingest", d, "--input", url2, "--input-wait-s", 5, *F.vcommon("gpu", "cuda-linear"))
    rc2 = p2.wait(timeout=30)
    fatal2 = (F.vsummary(p2, "ingest") or {}).get("fatal", "")
    ok2 = rc2 not in (0, None) and "xsrt protokolas nepalaikomas" in fatal2 and url2 in fatal2 and not flows_in(d)
    res.append(F.result(f"{tid}[protocol]", "PASS" if ok2 else "FAIL", why=f"exit {rc2}, fatal {fatal2!r}"))
    return res


def ingest_no_video(out):
    if (why := ensure_clips()):
        return [F.result("ingest_no_video", "FAIL", why=why)]
    tid, port = "ingest_no_video", BASE_PORT + 21
    T = dur(60)  # US 11: 60 s; klipas 90 s — visas langas (ir ingest pabaiga) — kol siuntėjas dar siunčia
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "audio_only_90.ts", port)], T,
                                  video=False)
    fv = flow(sv)
    problems = []
    if ri != 0:
        problems.append(f"exit {ri}")
    if fv.get("audio_blocks", 0) < T * 50 * 0.95:
        problems.append(f"audio_blocks {fv.get('audio_blocks')}")
    if fv.get("audio_zero_runs", 1) or fv.get("audio_disc", 1) or fv.get("audio_missing", 0):
        problems.append(f"garso spragos: tylos {fv.get('audio_zero_runs')} trūkiai {fv.get('audio_disc')} missing {fv.get('audio_missing')}")
    vid = [n for n in (si or {}).get("flows", []) if n.get("fields_ok")]
    if vid:
        problems.append("yra video grain'ų")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | verify {fv} | ingest audio_blocks {flow(si).get('audio_blocks')}")]


def ingest_format_change(out):
    if (why := ensure_clips()):
        return [F.result("ingest_format_change", "FAIL", why=why)]
    tid = "ingest_format_change"
    p0, p1 = BASE_PORT + 22, BASE_PORT + 23
    T = 30
    d = F.fresh_domain(tid)
    v = F.vcommon("gpu", "cuda-linear")
    ing = vf = None
    snd = []
    try:
        ing = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{p0}", "--input", f"udp://127.0.0.1:{p1}", "--duration", T + 10,
                      "--warmup", 6, *v)
        snd = [Sender.ts(out, f"{tid}/snd0", CLIPS / "fmtchange.ts", p0), Sender.ts(out, f"{tid}/snd1", CLIPS / "av_1080i50.ts", p1)]
        vf = F.FProc(out, tid, "verify", d, "--mode", "ingest", "--flows", 1, "--flow-base", 1, "--duration", T + 6, "--warmup", 6, "--lag-fields", 3,
                     "--audio", 1, "--input-wait-ms", 30000, "--impl", "gpu", "--storage", "cuda-linear")
        vf.wait(timeout=T + 90)
        ri = ing.wait(timeout=T + 90)
    finally:
        for pr in (vf, ing):
            if pr is not None:
                pr.term()
        for s in snd:
            s.stop()
    si, sv = F.vsummary(ing, "ingest"), F.vsummary(vf, "verify-ingest")
    fv = flow(sv)
    err = ing.err_path.read_text(errors="replace") if ing is not None and ing.err_path.exists() else ""
    problems = []
    if ri != 6:
        problems.append(f"exit {ri} (laukta 6)")
    if flow(si, 0).get("format_changed", 0) != 1 or "formatas pasikeitė" not in err:
        problems.append("nėra aiškaus pranešimo apie formato pasikeitimą")
    if ratio_ok(fv) < 0.998 or fv.get("marker_jumps", 0):
        problems.append(f"antras šaltinis: {ratio_ok(fv):.4%}, šuoliai {fv.get('marker_jumps')}")
    if flow(si, 1).get("late_frames", 0) or flow(si, 1).get("missing_frames", 0):
        problems.append(f"antras šaltinis: late {flow(si, 1).get('late_frames')} missing {flow(si, 1).get('missing_frames')}")
    line = next((ln for ln in err.splitlines() if "formatas pasikeitė" in ln), "")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | {line.strip()[:200]} | antras: {ratio_ok(fv):.4%}")]


def ingest_unsupported(out):
    if (why := ensure_clips()):
        return [F.result("ingest_unsupported", "FAIL", why=why)]
    res = []
    for k, (clip, want) in enumerate((("bad_422.ts", "4:2:2"), ("bad_10bit.ts", "10 bitų"), ("bad_bff.ts", "BFF"))):
        tid = f"ingest_unsupported[{clip[:-3]}]"
        port = BASE_PORT + 30 + k
        d = F.fresh_domain(tid.replace("[", "-").replace("]", ""))
        s = Sender.ts(out, f"{tid}/snd", CLIPS / clip, port, "--loop")
        try:
            p = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{port}", "--input-wait-s", 10, *F.vcommon("gpu", "cuda-linear"))
            rc = p.wait(timeout=40)
        finally:
            s.stop()
        fatal = (F.vsummary(p, "ingest") or {}).get("fatal", "")
        left = flows_in(d)
        ok = rc == 2 and want in fatal and "nepalaikomas" in fatal and not left
        res.append(F.result(tid, "PASS" if ok else "FAIL", why=f"exit {rc}, fatal {fatal!r}, flow'ai {left}"))
    return res


TESTS = {"ingest_av_offset": ingest_av_offset, "ingest_hevc": ingest_hevc, "ingest_bursty_no_drift": ingest_bursty_no_drift,
         "ingest_marker_continuity": ingest_marker_continuity, "ingest_pts_order_not_arrival": ingest_pts_order_not_arrival,
         "ingest_drift_accelerated": ingest_drift_accelerated, "ingest_pts_wrap": ingest_pts_wrap, "ingest_hls": ingest_hls, "ingest_srt": ingest_srt, "ingest_rtp": ingest_rtp,
         "ingest_two_formats": ingest_two_formats, "ingest_reconnect": ingest_reconnect, "ingest_overload_skip": ingest_overload_skip,
         "ingest_progressive_chroma": ingest_progressive_chroma, "ingest_bad_url": ingest_bad_url, "ingest_no_video": ingest_no_video,
         "ingest_format_change": ingest_format_change, "ingest_unsupported": ingest_unsupported}
