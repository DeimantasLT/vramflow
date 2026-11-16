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
    need.append("av_1080i50_long.ts")
    need += ["intra_refresh_1080p25.ts", "open_gop_1080p25.ts"]
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


def ts_edges(path):
    """TS failo video ir garso pateikimo kraštai, s: v0 / a0 — mažiausias PTS, v1 / a1 — didžiausias PTS + trukmė (ffprobe)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "packet=stream_index,pts_time,duration_time", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    e = {}
    for ln in r.stdout.splitlines():
        x = ln.split(",")
        if len(x) < 3 or x[1] in ("", "N/A"):
            continue
        k = "v" if x[0] == "0" else "a"
        t, d = float(x[1]), float(x[2]) if x[2] not in ("", "N/A") else 0.0
        e[k + "0"] = min(e.get(k + "0", t), t)
        e[k + "1"] = max(e.get(k + "1", t + d), t + d)
    return e


def ingest_relock_audio_resync(out):
    """M2 (round 10): sandūra, po kurios video PTS pusę kadro nuo DTS tinklelio (perfiksavimo žingsnis lygiai Tf/2 = 20 ms, langas
    [Tf/2, 20 ms]) — 1080i50 ir 1080p25. Garsas ir video pjaunami ir sulyginami tame pačiame taške: verify — ilgiausia tyla
    ≤ sandūros turinio tarpo (A garso pabaiga / B garso pradžia ne ties video riba, matuojama ffprobe) + vieno garso bloko (grain'o:
    960 / 1920 mėginių), |A/V| ≤ ribos (20 / 40 ms) visame lange, indeksai be turinio ≤ 2."""
    if (why := ensure_clips()):
        return [F.result("ingest_relock_audio_resync", "FAIL", why=why)]
    import re
    res = []
    for k, (fmt, limit, block) in enumerate((("1080i50", 20.0, 960), ("1080p25", 40.0, 1920))):
        tid = f"ingest_relock_audio_resync[{fmt}]"
        work = pathlib.Path(out) / tid
        work.mkdir(parents=True, exist_ok=True)
        a_ts, b_ts, clip = work / "a.ts", work / "b.ts", work / "splice.ts"
        ff = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(CLIPS / f"av_{fmt}.ts"), "-t", "15", "-c", "copy"]
        subprocess.run(ff + ["-f", "mpegts", str(a_ts)], check=True)
        # B: +100 s PTS šuolis (nutrūkimas → tęsinys) ir video PTS (ne DTS) bei viso garso laikai +20 ms: tęsinys skaičiuojamas pagal
        # DTS, o kvantuojamas pagal PTS — lygiai pusė kadro nuo tinklelio, žingsnis ±20 ms; garsas kartu, todėl turinio A/V nepakinta
        subprocess.run(ff + ["-output_ts_offset", "100", "-bsf:v", "setts=pts=PTS+1800", "-bsf:a", "setts=pts=PTS+1800:dts=DTS+1800",
                             "-f", "mpegts", str(b_ts)], check=True)
        clip.write_bytes(a_ts.read_bytes() + b_ts.read_bytes())
        # Turinio tarpas sandūroje: A garsas baigiasi / B garsas prasideda ne ties video riba (AAC kadrai, -t kirpimas) — tiek tylos
        # teisėta ir prieš perfiksavimą (kilpos sandūrose irgi); riba — šis tarpas + vienas garso blokas
        ea, eb = ts_edges(a_ts), ts_edges(b_ts)
        content_gap = max(0, round(((eb["a0"] - eb["v0"]) - (ea["a1"] - ea["v1"])) * 48000))
        port = BASE_PORT + 55 + k
        T = 22
        si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda p=port, c=clip, t=tid: Sender.ts(out, f"{t}/snd", c, p, "--pcr-continuous")], T, warm=6)
        fi, fv = flow(si), flow(sv)
        a = av_stats(sv)
        problems = []
        line = next((ln for ln in err.splitlines() if "perfiksavimas" in ln), "")
        m = re.search(r"poslinkis ([+-][0-9.]+) ms", line)
        step = abs(float(m.group(1))) if m else None
        if step is None or not 10.0 <= step <= 20.0:
            problems.append(f"sandūros žingsnis {step} ms ne lange [Tf/2 … 20 ms] (testo sąranka)")
        if fv.get("audio_zero_max", 10**9) > content_gap + block:
            problems.append(f"ilgiausia tyla {fv.get('audio_zero_max')} mėginių > turinio tarpas {content_gap} + blokas {block}")
        if a["max_abs_ms"] is None or a["max_abs_ms"] > limit or a["pairs"] < int(T * 0.8):
            problems.append(f"|A/V| maks. {a['max_abs_ms']} ms (riba {limit}), porų {a['pairs']}")
        if fv.get("missing", 0) + fv.get("invalid", 0) > 2:
            problems.append(f"indeksai be turinio: missing {fv.get('missing')} invalid {fv.get('invalid')}")
        if fi.get("late_frames", 0):
            problems.append(f"late_frames {fi.get('late_frames')}")
        res.append(F.result(tid, "FAIL" if problems else "PASS",
                            why="; ".join(problems) + f" | {line.strip()[-160:]} | tyla maks. {fv.get('audio_zero_max')} mėginių, "
                            f"(turinio tarpas {content_gap}), relock_flushed_pkts {fi.get('relock_flushed_pkts')}, audio_relock_resyncs {fi.get('audio_relock_resyncs')} | " +
                            summary_line(si, sv, a, ratio_ok(fv))))
        for f in (a_ts, b_ts, clip):
            f.unlink(missing_ok=True)
    return res


def ingest_codec_change_reconnect(out):
    """L5 (round 10): po prisijungimo iš naujo tas pats rastras, bet kitas kodekas (HEVC 1080p25 → H.264 1080p25 atviru GOP, prisijungta
    vidury — IDR neateina): tai ne formato pasikeitimas — dekoderis atidaromas iš naujo šiam kodekui, įėjimo taško paieška seka
    DABARTINĖS sesijos kodeką (anksčiau — pirmos sesijos: H.264 recovery point / I pjūvis po HEVC sesijos nebuvo ieškomi ir vaizdas
    negrįždavo), šaltinis tęsiasi: format_changed == 0, žurnale — kodeko pakeitimas, po jo kadrai dekoduojami ir |A/V| ≤ 1 kadras."""
    tid = "ingest_codec_change_reconnect"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    if not (CLIPS / "hevc_1080p25.ts").exists():
        return [F.result(tid, "SKIP", why="nėra hevc_1080p25.ts (FFmpeg be libx265)")]
    port = BASE_PORT + 57
    T, FIRST, GAP = 34, 12, 4
    tb = [sys.executable, str(PERF / "ts_burst.py")]
    q = lambda xs: " ".join(f"'{x}'" for x in xs)
    cmd = (f"{q(tb + [str(CLIPS / 'hevc_1080p25.ts'), '--port', str(port), '--duration', str(FIRST)])}; sleep {GAP}; "
           f"exec {q(tb + [str(CLIPS / 'open_gop_1080p25.ts'), '--port', str(port), '--start-s', '2.3'])}")
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda: Sender(out, f"{tid}/snd", ["bash", "-c", cmd])], T, warm=6)
    fi = flow(si)
    a = av_stats(sv)
    problems = []
    if ri is None or ri < 0 or ri == 6 or si is None:
        problems.append(f"exit {ri} (laukta ne 6 ir ne signalas)")
    if fi.get("format_changed", 0):
        problems.append("format_changed — šaltinis sustabdytas")
    if fi.get("input_reconnects", 0) < 1:
        problems.append(f"input_reconnects {fi.get('input_reconnects')}")
    line = next((ln for ln in err.splitlines() if "kodekas pasikeitė" in ln), "")
    if "h264" not in line:
        problems.append("žurnale nėra kodeko pakeitimo hevc → h264")
    if fi.get("no_entry_point", 0):
        problems.append(f"no_entry_point {fi.get('no_entry_point')} (H.264 įėjimo taškas po HEVC sesijos neieškotas)")
    i0 = a["raw"][0][0] if a["raw"] else 0
    after = [p[1] / 1000.0 for p in a["raw"] if (p[0] - i0) / 25 > FIRST + GAP + 4]
    if len(after) < 5 or max(abs(x) for x in after) > 40:
        problems.append(f"po kodeko pakeitimo A/V porų {len(after)}, maks. {max((abs(x) for x in after), default=None)} ms")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | {line.strip()[-120:]} | po pakeitimo porų {len(after)} | " + summary_line(si, sv, a, ratio_ok(flow(sv))))]


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


def ingest_drift_default(out):
    """Round 9 (C): realistinis dreifas NUMATYTOMIS parinktimis (--max-comp-ppm 1000): siuntėjas ±500 ppm, 240 s — keli video
    slydimai (≥ 2), 0 perfiksavimų, garse nė vieno tylos tarpo / trūkio, audio_resyncs == 0, |A/V| ≤ 1 laukas visada. (Ekstremalus
    ingest_drift_accelerated — 0,5 % su --max-comp-ppm 6000 — lieka nepakeistas.)"""
    if (why := ensure_clips()):
        return [F.result("ingest_drift_default", "FAIL", why=why)]
    res = []
    for k, ppm in enumerate((500, -500)):
        tid, port = f"ingest_drift_default[{ppm:+d}ppm]", BASE_PORT + 44 + k
        T = dur(240)
        si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"],
                                      [lambda p=port, s=1 + ppm * 1e-6, t=tid: Sender.ts(out, f"{t}/snd", CLIPS / "av_1080i50_long.ts", p, "--speed", s)],
                                      T, (), warm=6)
        fi, fv = flow(si), flow(sv)
        a = av_stats(sv)
        problems = []
        if ri not in (0,):
            problems.append(f"ingest exit {ri}")
        if fi.get("drift_slips", 0) < 2:
            problems.append(f"drift_slips {fi.get('drift_slips')} < 2")
        if fi.get("relock_fresh", 0) + fi.get("relock_cont", 0) + fi.get("discontinuities", 0):
            problems.append(f"perfiksavimai {fi.get('relock_fresh')}+{fi.get('relock_cont')}, nutrūkimai {fi.get('discontinuities')}")
        if a["max_abs_ms"] is None or a["max_abs_ms"] > 20:
            problems.append(f"|A/V| maks. {a['max_abs_ms']} ms > 20")
        if fv.get("audio_zero_runs", 1) or fv.get("audio_disc", 1):
            problems.append(f"garso trūkiai: tylos {fv.get('audio_zero_runs')} trūkiai {fv.get('audio_disc')} {a['disc'][:5]}")
        if fi.get("audio_resyncs", 0):
            problems.append(f"audio_resyncs {fi.get('audio_resyncs')}")
        bad_steps = [j for j in a["jumps"] if j[1] not in (-1, 1, 3)]
        if bad_steps:
            problems.append(f"video trūkiai > 1 kadro: {bad_steps[:5]}")
        sgn = -1 if fi.get("drift_ppm_negative") else 1
        est = sgn * fi.get("drift_ppm_abs_x1000", 0) / 1000.0
        res.append(F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | drift_ppm {est:.0f} | " +
                            summary_line(si, sv, a, ratio_ok(fv)), drift_ppm=est))
    return res


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


class StreamBody:
    """HTTP atsakymas iš failo: pirmi nbytes baitų (0 — visas failas) repeat kartų, rate baitų/s (0 — tinklo greičiu)."""

    def __init__(self, path, nbytes=0, rate=0, repeat=1, ctype="video/mp2t"):
        self.path, self.nbytes, self.rate, self.repeat, self.ctype = str(path), nbytes, rate, repeat, ctype

    def serve(self, h):
        size = os.path.getsize(self.path)
        n = min(size, self.nbytes) if self.nbytes else size
        h.send_response(200)
        h.send_header("Content-Type", self.ctype)
        h.send_header("Content-Length", str(n * self.repeat))
        h.end_headers()
        t0, sent = time.monotonic(), 0
        try:
            for _ in range(self.repeat):
                with open(self.path, "rb") as f:
                    left = n
                    while left > 0:
                        b = f.read(min(left, 188 * 700))
                        if not b:
                            break
                        h.wfile.write(b)
                        left -= len(b)
                        sent += len(b)
                        if self.rate:
                            ahead = sent / self.rate - (time.monotonic() - t0)
                            if ahead > 0:
                                time.sleep(ahead)
        except (BrokenPipeError, ConnectionResetError):
            pass


class PlaylistServer:
    """Loopback HTTP (127.0.0.1, laisvas prievadas): virtualūs keliai (routes — bytes, keičiami testo metu) ir statiniai failai iš
    katalogo; hits — užklausų skaičius keliui."""

    def __init__(self, directory=None):
        import http.server
        import threading
        self.routes, self.hits = {}, {}
        srv = self

        class H(http.server.SimpleHTTPRequestHandler):
            def __init__(h, *a, **kw):
                super().__init__(*a, directory=str(directory or REPO / "nonexistent"), **kw)

            def log_message(h, *a):
                pass

            def do_GET(h):
                path = h.path.split("?")[0]
                srv.hits[path] = srv.hits.get(path, 0) + 1
                body = srv.routes.get(path)
                if callable(body):
                    body = body()
                if body is None:
                    return super().do_GET()
                if hasattr(body, "serve"):  # StreamBody ir kiti atsakymai su savo elgsena (gabalais, nukreipimas)
                    return body.serve(h)
                h.send_response(200)
                h.send_header("Content-Type", "application/vnd.apple.mpegurl")
                h.send_header("Content-Length", str(len(body)))
                h.end_headers()
                h.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def url(self, p):
        return f"http://127.0.0.1:{self.port}{p}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FifoCanary:
    """Vardinis kanalas: atidarymas skaitymui (bet kuo) atblokuoja rašytoją — opened() == True. Failas niekada neturi turinio."""

    def __init__(self, path):
        import threading
        self.path = str(path)
        if os.path.exists(self.path):
            os.unlink(self.path)
        os.mkfifo(self.path)
        self.releasing = False
        self.hit = False

        def w():
            with open(self.path, "wb"):
                self.hit = not self.releasing

        self.t = threading.Thread(target=w, daemon=True)
        self.t.start()

    def opened(self):
        return self.hit

    def close(self):
        if self.t.is_alive():
            self.releasing = True
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
            self.t.join(timeout=5)
            os.close(fd)
        os.unlink(self.path)


class TcpCanary:
    """Loopback TCP klausytojas: connections — kiek kartų prisijungta."""

    def __init__(self):
        import threading
        self.s = socket.socket()
        self.s.bind(("127.0.0.1", 0))
        self.s.listen(8)
        self.s.settimeout(0.2)
        self.port = self.s.getsockname()[1]
        self.connections = 0
        self.run = True

        def acc():
            while self.run:
                try:
                    c, _ = self.s.accept()
                    self.connections += 1
                    c.close()
                except OSError:
                    pass

        self.t = threading.Thread(target=acc, daemon=True)
        self.t.start()

    def close(self):
        self.run = False
        self.t.join(timeout=5)
        self.s.close()


def two_source_http(out, tid, srv, path, port, T, swap_at, swap, ingest_args=(), impl="gpu", watch=None):
    """UDP šaltinis 0 (matuojamas verify) + HTTP šaltinis 1 (srv kelias path). Po swap_at s nuo paleidimo kviečiama swap() (serverio
    elgsenos pakeitimas vidury darbo); watch(pid) — kas 0,1 s (pvz. RSS). Grąžina (ingest santrauka, verify santrauka, exit, stderr)."""
    d = F.fresh_domain(tid.replace("[", "-").replace("]", ""))
    ing = vf = snd = None
    storage = "host" if impl == "cpu" else "cuda-linear"
    try:
        ing = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{port}", "--input", srv.url(path), "--duration", T + 14,
                      "--warmup", 6, *F.vcommon(impl, storage), *ingest_args)
        snd = Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port)
        vf = F.FProc(out, tid, "verify", d, "--mode", "ingest", "--flows", 1, "--flow-base", 0, "--duration", T + 6, "--warmup", 6,
                     "--lag-fields", 3, "--audio", 1, "--input-wait-ms", 30000, *F.vimpl(impl, storage))
        t0 = time.monotonic()
        swapped = False
        while ing.p.poll() is None and time.monotonic() - t0 < T + 90:
            if not swapped and time.monotonic() - t0 >= swap_at:
                swap()
                swapped = True
            if watch is not None:
                watch(ing.p.pid)
            if vf.p.poll() is not None and ing.p.poll() is None and time.monotonic() - t0 > T + 30:
                break
            time.sleep(0.1)
        vf.wait(timeout=T + 90)
        ri = ing.wait(timeout=60)
    finally:
        for pr in (vf, ing):
            if pr is not None:
                pr.term()
        if snd is not None:
            snd.stop()
    err = ing.err_path.read_text(errors="replace") if ing is not None and ing.err_path.exists() else ""
    return F.vsummary(ing, "ingest"), F.vsummary(vf, "verify-ingest"), ri, err


def hls_toctou_case(out):
    """H1 (round 10): HTTP šaltinio patikra mato MPEG-TS, o libav paties atidarymas po prisijungimo iš naujo gauna įdėtą master
    grojaraštį (rodo į save). Procesas gyvena iki galo, UDP šaltinis — be praradimų, HTTP šaltinio M3U8 atsakymas atmetamas."""
    tid = "ingest_hls_security[toctou]"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    T = 30
    srv = PlaylistServer()
    ts = StreamBody(CLIPS / "av_1080i50.ts", nbytes=188 * 7 * 9500, rate=1_250_000)  # ~10 s realiu laiku
    master = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\ntoctou\n"
    srv.routes["/toctou"] = ts
    n = [0]

    def after():  # pirmas GET (patikra) — TS, visi kiti (libav atidarymas, kiti bandymai) — master
        n[0] += 1
        return StreamBody(CLIPS / "av_1080i50.ts", nbytes=188 * 7 * 2000) if n[0] == 1 else master

    try:
        si, sv, ri, err = two_source_http(out, tid, srv, "/toctou", BASE_PORT + 50, T, 6, lambda: srv.routes.update({"/toctou": after}))
    finally:
        srv.close()
    problems, a, r = base_checks(si, sv, ri, 20.0, T, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    if flow(si, 1).get("input_reconnects", 0) < 1:
        problems.append(f"HTTP input_reconnects {flow(si, 1).get('input_reconnects')}")
    if n[0] < 2:
        problems.append(f"master grojaraštis neužklaustas (GET po pakeitimo: {n[0]})")
    # prisijungimai iš naujo su atidėjimu 1, 2, 4 … s — per ~30 s keliolika užklausų; daugiau — libav HLS kilpa be sargo
    if n[0] > 100:
        problems.append(f"{n[0]} GET po pakeitimo — libav HLS demuxeris kilpoje be sargo")
    line = next((ln for ln in err.splitlines() if "M3U8" in ln or "įdėtas master" in ln), "")
    if not line:
        problems.append("žurnale nėra M3U8 / įdėto master atmetimo")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | GET po pakeitimo {n[0]} | {line.strip()[:200]} | " + summary_line(si, sv, a, r))]


def conn_alive(sock):
    """Ar kliento TCP ryšys dar atviras (FIN negautas): skaitomas lizdas su 0 baitų MSG_PEEK — klientas uždarė."""
    import select
    try:
        r, _, _ = select.select([sock], [], [], 0)
        return not r or sock.recv(1, socket.MSG_PEEK) != b""
    except OSError:
        return False


class ChunkedHeadBody:
    """HTTP MPEG-TS, kurio pirmi head baitų siunčiami chunk dydžio gabalais su pauze (kiekvienas libav skaitymas gauna vieną gabalą),
    likusieji — rate baitų/s. Užklausa, atėjusi, kol ankstesnio atsakymo klientas dar skaito (ryšys atviras), — pakartotinė to paties
    URL užklausa (libav avio_seek → http_seek su Range: bytes=0-): jai — master grojaraštis (rodo į save); reseeks — jų skaičius.
    „Accept-Ranges: bytes“ — libav http laiko srautą ieškomu (http.c n6.1:1184-1187), tada AVIO buferis 32 KiB (aviobuf.c:985-989
    padvigubina tik neieškomam), ir antras buferio pildymas rašo nuo pradžios (aviobuf.c:563-564) — peržiūros pradžios buferyje nebelieka."""

    def __init__(self, path, nbytes, rate, head=8192, chunk=200, pause=0.015):
        import threading
        self.path, self.nbytes, self.rate, self.head, self.chunk, self.pause = str(path), nbytes, rate, head, chunk, pause
        self.lock = threading.Lock()
        self.live, self.gets, self.reseeks = set(), 0, 0

    def serve(self, h):
        with self.lock:
            self.gets += 1
            again = any(conn_alive(c) for c in self.live)
            if again:
                self.reseeks += 1
            else:
                self.live.add(h.connection)
        if again:
            body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\npeek.ts\n"
            h.send_response(200)
            h.send_header("Content-Type", "application/vnd.apple.mpegurl")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return
        try:
            h.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            data = open(self.path, "rb").read()[:self.nbytes]
            h.send_response(200)
            h.send_header("Content-Type", "video/mp2t")
            h.send_header("Content-Length", str(len(data)))
            h.send_header("Accept-Ranges", "bytes")  # libav http: ieškomas (is_streamed = 0) → AVIO buferis 32 KiB, ne 64 KiB
            h.end_headers()
            time.sleep(self.pause)
            p = 0
            while p < min(self.head, len(data)):
                h.wfile.write(data[p:p + self.chunk])
                p += self.chunk
                time.sleep(self.pause)
            t0 = time.monotonic()
            while p < len(data):
                h.wfile.write(data[p:p + 188 * 700])
                p += 188 * 700
                ahead = (p - self.head) / self.rate - (time.monotonic() - t0)
                if ahead > 0:
                    time.sleep(ahead)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.live.discard(h.connection)


def hls_peek_case(out):
    """A (round 11): sargo peržiūra (pirmi 4 KiB) ne M3U8 atsakymo neturi grąžinti tinklo konteksto atgal (avio_seek už buferio ribų →
    nauja HTTP užklausa, kurios atsakymas niekada netikrinamas). Serveris pirmus 8 KiB siunčia 200 B gabalais (peržiūra netelpa
    buferyje), pakartotinei užklausai atsako master grojaraščiu. Kriterijai: pakartotinių užklausų 0, HTTP šaltinis dekoduoja (peržiūros
    baitai ir likusi dalis perduodami libav), procesas gyvena, UDP šaltinis — be praradimų."""
    tid = "ingest_hls_security[peek_reseek]"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    T = 30
    srv = PlaylistServer()
    body = ChunkedHeadBody(CLIPS / "av_1080i50.ts", nbytes=188 * 7 * 9500, rate=1_250_000)  # ~10 s realiu laiku
    srv.routes["/peek.ts"] = body
    try:
        si, sv, ri, err = two_source_http(out, tid, srv, "/peek.ts", BASE_PORT + 58, T, 10**9, lambda: None)
    finally:
        srv.close()
    problems, a, r = base_checks(si, sv, ri, 20.0, T, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    if body.reseeks:
        problems.append(f"pakartotinių užklausų po peržiūros {body.reseeks} (laukta 0)")
    if flow(si, 1).get("frames_decoded", 0) < 25:
        problems.append(f"HTTP šaltinis dekodavo {flow(si, 1).get('frames_decoded')} kadrų")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | GET {body.gets}, pakartotinių {body.reseeks}, HTTP frames_decoded "
                     f"{flow(si, 1).get('frames_decoded')} | " + summary_line(si, sv, a, r))]


class Redirect:
    """HTTP 302 į to (kitas serveris); hits — kiek kartų nukreipta."""

    def __init__(self, to):
        self.to, self.hits = to, 0

    def serve(self, h):
        self.hits += 1
        h.send_response(302)
        h.send_header("Location", self.to)
        h.send_header("Content-Length", "0")
        h.end_headers()


def hls_live_swap_case(out, tid, port, extra, ingest_args=(), path="/live.m3u8"):
    """Gyvas HLS šaltinis (ffmpeg segmentuotojas) + UDP šaltinis 0. Po 18 s grojaraštis = dabartinis + extra(srv) (pvz. naujas
    segmentas, kuris nukreipia ar grąžina M3U8); extra None — be pakeitimo. path — šaltinio URL kelias (serveris jį normalizuoja).
    Grąžina (si, sv, ri, err, srv)."""
    work = pathlib.Path(out) / tid.replace("[", "-").replace("]", "") / "live"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    seg = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(CLIPS / "av_1080i50.ts"),
                            "-c", "copy", "-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+omit_endlist",
                            str(work / "live.m3u8")], stdout=subprocess.DEVNULL, stderr=open(work.parent / "seg.log", "w"))
    srv = PlaylistServer(work)
    T, SWAP = 30, 12
    try:
        t0 = time.monotonic()
        while not (work / "live.m3u8").exists() and time.monotonic() - t0 < 20:
            time.sleep(0.2)
        time.sleep(4.5)

        def swap():
            srv.routes["/live.m3u8"] = (work / "live.m3u8").read_bytes() + extra(srv)

        si, sv, ri, err = two_source_http(out, tid, srv, path, port, T, 6 + SWAP if extra is not None else 10**9, swap, ingest_args)
    finally:
        seg.terminate()
        seg.wait()
        srv.close()
    return si, sv, ri, err, srv


def hls_redirect_case(out):
    """D (round 11): --hls-same-host — segmento URL tame pačiame serveryje (grojaraščio tekstas tvarkingas) atsako 302 į kitą
    loopback prievadą (kitas klipas, 1080p25). Sargas lygina atidaryto konteksto location su leistinu serveriu: atmesta, to segmento
    medžiaga nenaudojama (1080p25 niekada nezonduojamas), procesas gyvena, UDP šaltinis — be praradimų. Nukreipimo užklausą libav
    jau būna padaręs (http.c n6.1:419-438; max_redirects parinkties 6.1 neturi) — tai dokumentuota, ne tikrinama."""
    tid = "ingest_hls_security[redirect]"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    other = PlaylistServer()
    other.routes["/seg.ts"] = StreamBody(CLIPS / "av_1080p25.ts", nbytes=188 * 7 * 2000)
    red = Redirect(other.url("/seg.ts"))

    def extra(srv):
        srv.routes["/redir.ts"] = red
        return b"#EXTINF:2.0,\nredir.ts\n"

    try:
        si, sv, ri, err, _ = hls_live_swap_case(out, tid, BASE_PORT + 62, extra, ("--hls-same-host",))
    finally:
        other.close()
    problems, a, r = base_checks(si, sv, ri, 20.0, 30, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    line = next((ln for ln in err.splitlines() if "--hls-same-host" in ln and "sesija nutraukiama" in ln), "")
    if not line:
        problems.append("žurnale nėra --hls-same-host atmetimo (sesija nutraukiama)")
    if red.hits < 1:
        problems.append("nukreipiantis segmentas neužklaustas (testo sąranka)")
    if "1080p25" in err:
        problems.append("kito serverio medžiaga (1080p25) pasiekė libav zondavimą")
    if flow(si, 1).get("input_reconnects", 0) < 1:
        problems.append(f"HLS input_reconnects {flow(si, 1).get('input_reconnects')}")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | nukreipta {red.hits} k., kito serverio GET {other.hits.get('/seg.ts', 0)} | {line.strip()[:220]} | " +
                     summary_line(si, sv, a, r))]


def hls_authority_case(out):
    """H1 (round 12): --hls-same-host — perkrautame grojaraštyje segmentai „http:127.0.0.1:<kitas>/x://127.0.0.1:<leistinas>/a.ts“ ir
    „http:127.0.0.1:<kitas>/?u=http://…“: libav av_url_split (utils.c n6.1:377-383) jungtųsi į <kitas> (host'as po pirmo ':'), nors
    „://“ kelyje rodo leistiną serverį. Sargas autoritetą skaičiuoja kaip libav ir „http:“ be „//“ atmeta: kitas prievadas (TCP
    kanarėlė) niekada neprisijungiamas, procesas gyvena, UDP šaltinis — be praradimų."""
    tid = "ingest_hls_security[authority_split]"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    tcp = TcpCanary()

    def extra(srv):
        return (b"#EXTINF:2.0,\nhttp:127.0.0.1:%d/x://127.0.0.1:%d/a.ts\n#EXTINF:2.0,\nhttp:127.0.0.1:%d/?u=http://127.0.0.1:%d/b.ts\n"
                % (tcp.port, srv.port, tcp.port, srv.port))

    try:
        si, sv, ri, err, _ = hls_live_swap_case(out, tid, BASE_PORT + 64, extra, ("--hls-same-host",))
    finally:
        tcp.close()
    problems, a, r = base_checks(si, sv, ri, 20.0, 30, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    if tcp.connections:
        problems.append(f"kitas prievadas (TCP kanarėlė) prisijungtas {tcp.connections} k. — libav jungėsi pagal av_url_split host'ą")
    line = next((ln for ln in err.splitlines() if "av_url_split" in ln and "sesija nutraukiama" in ln), "")
    if not line:
        problems.append("žurnale nėra av_url_split atmetimo (sesija nutraukiama)")
    if flow(si, 1).get("input_reconnects", 0) < 1:
        problems.append(f"HLS input_reconnects {flow(si, 1).get('input_reconnects')}")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | kanarėlės prisijungimų {tcp.connections} | {line.strip()[:220]} | " + summary_line(si, sv, a, r))]


class HugeLengthBody:
    """HTTP atsakymas su „Content-Length: 9223372036854775808“ (2^63: libav http filesize strtoull, http.c n6.1:1181, kaip int64 —
    neigiamas) ir „Accept-Ranges: bytes“ (ieškomas, http.c:1184-1187). avio_size atsarginis kelias (aviobuf.c:383-389) tada ieškotų
    SEEK_END ir atgal — naujos užklausos su Range (http.c:1938-1960), kurių atsakymų sargas nemato. Siunčiami pirmi nbytes failo baitų
    (rate B/s, 0 — tinklo greičiu); gets — užklausų, ranged — užklausų su Range poslinkiu > 0."""

    def __init__(self, path, nbytes, rate=0):
        import threading
        self.path, self.nbytes, self.rate = str(path), nbytes, rate
        self.lock = threading.Lock()
        self.gets, self.ranged = 0, 0

    def serve(self, h):
        import re
        m = re.match(r"bytes=(\d+)-", h.headers.get("Range", ""))
        with self.lock:
            self.gets += 1
            if m and int(m.group(1)) > 0:
                self.ranged += 1
        try:
            data = open(self.path, "rb").read()[:self.nbytes]
            h.send_response(200)
            h.send_header("Content-Type", "video/mp2t")
            h.send_header("Content-Length", "9223372036854775808")
            h.send_header("Accept-Ranges", "bytes")
            h.end_headers()
            t0, p = time.monotonic(), 0
            while p < len(data):
                h.wfile.write(data[p:p + 188 * 700])
                p += 188 * 700
                if self.rate:
                    ahead = p / self.rate - (time.monotonic() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
        except (BrokenPipeError, ConnectionResetError):
            pass


def huge_length_cases(out):
    """M1 (round 12): serveris praneša Content-Length 2^63 su „Accept-Ranges: bytes“. Sargo peek_io AVSEEK_SIZE kvietė avio_size(vidinis),
    kurio atsarginis kelias (dydis < 0 → SEEK_END ir atgal) http kontekste — naujos užklausos (Range: bytes=2^63−1- ir atgal). Dabar
    dydis imamas tik tiesiai (inner->seek AVSEEK_SIZE), < 0 — ENOSYS, be atsarginio kelio. [ts]: tiesioginis MPEG-TS per HTTP
    (avio_size — demux.c n6.1:1906); [init]: HLS init sekcija be BYTERANGE (avio_size — hls.c:1386) ir jos segmentas. Kriterijai: užklausų su
    Range poslinkiu > 0 — 0; TS: ≤ 2 užklausos sesijai (patikra + sargo atidarymas); init: ≤ 1 sesijai; procesas gyvena, UDP — be praradimų."""
    res = []
    if (why := ensure_clips()):
        return [F.result("ingest_hls_security[huge_length_ts]", "FAIL", why=why)]
    tid = "ingest_hls_security[huge_length_ts]"
    srv = PlaylistServer()
    body = HugeLengthBody(CLIPS / "av_1080i50.ts", nbytes=188 * 7 * 9500, rate=1_250_000)  # ~10 s realiu laiku
    srv.routes["/huge.ts"] = body
    try:
        si, sv, ri, err = two_source_http(out, tid, srv, "/huge.ts", BASE_PORT + 66, 30, 10**9, lambda: None)
    finally:
        srv.close()
    problems, a, r = base_checks(si, sv, ri, 20.0, 30, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    sessions = flow(si, 1).get("input_reconnects", 0) + 1
    if body.ranged:
        problems.append(f"užklausų su Range poslinkiu > 0: {body.ranged} (laukta 0)")
    if body.gets > 2 * sessions:
        problems.append(f"GET {body.gets} > 2 × {sessions} sesijų")
    if flow(si, 1).get("frames_decoded", 0) < 25:
        problems.append(f"HTTP šaltinis dekodavo {flow(si, 1).get('frames_decoded')} kadrų")
    res.append(F.result(tid, "FAIL" if problems else "PASS",
                        why="; ".join(problems) + f" | GET {body.gets}, su Range > 0 {body.ranged}, sesijų {sessions}, HTTP frames_decoded "
                        f"{flow(si, 1).get('frames_decoded')} | " + summary_line(si, sv, a, r)))

    tid = "ingest_hls_security[huge_length_init]"
    init = HugeLengthBody(CLIPS / "av_1080i50.ts", nbytes=188 * 7 * 20)
    seg = HugeLengthBody(CLIPS / "av_1080i50.ts", nbytes=188 * 7 * 2400)

    def extra(s):
        s.routes["/init.ts"] = init
        s.routes["/mseg.ts"] = seg
        return b'#EXT-X-MAP:URI="init.ts"\n#EXTINF:2.0,\nmseg.ts\n'

    si, sv, ri, err, _ = hls_live_swap_case(out, tid, BASE_PORT + 67, extra)
    problems, a, r = base_checks(si, sv, ri, 20.0, 30, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    sessions = flow(si, 1).get("input_reconnects", 0) + 1
    if init.gets < 1:
        problems.append("init.ts neužklaustas (testo sąranka)")
    if init.ranged or seg.ranged:
        problems.append(f"užklausų su Range poslinkiu > 0: init {init.ranged}, segmento {seg.ranged} (laukta 0)")
    if init.gets > sessions or seg.gets > sessions:
        problems.append(f"GET init {init.gets}, segmento {seg.gets} > {sessions} sesijų")
    res.append(F.result(tid, "FAIL" if problems else "PASS",
                        why="; ".join(problems) + f" | GET init {init.gets} (Range > 0: {init.ranged}), segmento {seg.gets} (Range > 0: "
                        f"{seg.ranged}), sesijų {sessions} | " + summary_line(si, sv, a, r)))
    return res


def hls_dot_location_case(out):
    """L1 (round 12): šaltinio URL su taškų segmentu („/x/../live.m3u8“). libav viršutinio media grojaraščio perkrovimus atidaro
    normalizuotu URL (ff_make_absolute_url(NULL, location), hls.c n6.1:324, 613-621) — „/live.m3u8“; sargas jį turi laikyti tuo pačiu
    grojaraščiu (ne „M3U8 ne grojaraščio URL“). Kriterijai: 0 prisijungimų iš naujo, žurnale nėra atmetimo, HLS šaltinis dekoduoja."""
    tid = "ingest_hls_security[dot_location]"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    si, sv, ri, err, srv = hls_live_swap_case(out, tid, BASE_PORT + 69, None, path="/x/../live.m3u8")
    problems, a, r = base_checks(si, sv, ri, 20.0, 30, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    line = next((ln for ln in err.splitlines() if "sesija nutraukiama" in ln), "")
    if line:
        problems.append("sargas atmetė perkrovimą")
    if flow(si, 1).get("input_reconnects", 0) != 0:
        problems.append(f"HLS input_reconnects {flow(si, 1).get('input_reconnects')} (laukta 0)")
    if flow(si, 1).get("frames_decoded", 0) < 250:
        problems.append(f"HLS šaltinis dekodavo {flow(si, 1).get('frames_decoded')} kadrų")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | HLS input_reconnects {flow(si, 1).get('input_reconnects')}, frames_decoded "
                     f"{flow(si, 1).get('frames_decoded')} | {line.strip()[:220]} | " + summary_line(si, sv, a, r))]


def hls_segment_m3u8_case(out):
    """E (round 11): media grojaraščio naujas segmentas atsako media grojaraščiu (M3U8 ne grojaraščio URL). Sargas M3U8 priima tik iš
    žinomų grojaraščių URL (viršutinis ir jo location, master variantai) — čia atmesta; procesas gyvena, UDP šaltinis — be praradimų."""
    tid = "ingest_hls_security[segment_m3u8]"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]

    def extra(srv):
        srv.routes["/nested.ts"] = b"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:2.0,\nlive0.ts\n"
        return b"#EXTINF:2.0,\nnested.ts\n"

    si, sv, ri, err, srv = hls_live_swap_case(out, tid, BASE_PORT + 63, extra)
    problems, a, r = base_checks(si, sv, ri, 20.0, 30, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    line = next((ln for ln in err.splitlines() if "ne grojaraščio URL" in ln and "sesija nutraukiama" in ln), "")
    if not line:
        problems.append("žurnale nėra „M3U8 … ne grojaraščio URL“ atmetimo (sesija nutraukiama)")
    if srv.hits.get("/nested.ts", 0) < 1:
        problems.append("segmentas nested.ts neužklaustas (testo sąranka)")
    if flow(si, 1).get("input_reconnects", 0) < 1:
        problems.append(f"HLS input_reconnects {flow(si, 1).get('input_reconnects')}")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | nested.ts GET {srv.hits.get('/nested.ts', 0)} | {line.strip()[:220]} | " + summary_line(si, sv, a, r))]



def rss_kb(pid):
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1])
    except OSError:
        pass
    return 0


def hls_reload_cases(out):
    """H2 / H3 (round 10): gyvas HLS šaltinis (+ UDP šaltinis 0); vidury darbo media grojaraščio perkrovimas (libav paties užklausa,
    ne patikra) tampa priešiškas: > 4 KiB komentarų ir tada master eilutė į save; 100 000 variantų (> 1 MiB); 50 000 segmentų
    (< 1 MiB). Kiekvienas atmetamas visame tekste, procesas gyvena, UDP šaltinis — be praradimų, RSS po pakeitimo auga ≤ 64 MB."""
    if (why := ensure_clips()):
        return [F.result("ingest_hls_security[reload]", "FAIL", why=why)]
    work = pathlib.Path(out) / "ingest_hls_security" / "live"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    seg = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(CLIPS / "av_1080i50.ts"),
                            "-c", "copy", "-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+omit_endlist",
                            str(work / "live.m3u8")], stdout=subprocess.DEVNULL, stderr=open(work.parent / "seg.log", "w"))
    srv = PlaylistServer(work)
    hdr = b"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n"
    pad = b"#" + b"x" * 6000 + b"\n"  # > 4 KiB: anksčiau sargas matė tik pradžią (media antraštę)
    cases = (("padding", hdr + pad + b"#EXT-X-STREAM-INF:BANDWIDTH=1\nlive.m3u8\n", "įdėtas master"),
             ("variants100k", hdr + pad + b"".join(b"#EXT-X-STREAM-INF:BANDWIDTH=1\nv%d.m3u8\n" % k for k in range(100000)), "didesnis nei"),
             ("segments50k", hdr + b"#EXTINF:2.0,\ns.ts\n" * 50000, "nuorodų"),
             # B (round 11): tik CR skyrikliai — libav ff_get_line eilutę baigia ir '\r' (aviobuf.c n6.1:825), sargas turi matyti tą patį
             ("cr_master", (hdr + b"#EXT-X-STREAM-INF:BANDWIDTH=1\nlive.m3u8\n").replace(b"\n", b"\r"), "įdėtas master"),
             # F (round 11): libav kiekvieną #EXT-X-MAP eilutę kaupia iki sesijos pabaigos (hls.c n6.1:456, 251-269) — 2000 eilučių po ~0,4 KiB
             # (< 1 MiB grojaraštis, perkraunamas kas 2 s) atmetamos
             ("map_many", hdr + b"".join(b'#EXT-X-MAP:URI="i%d_' % k + b"m" * 400 + b'.mp4"\n' for k in range(2000)) + b"#EXTINF:2.0,\ns.ts\n",
              "EXT-X-MAP"),
             # H2 (round 12): libav kiekvieną #EXT-X-MEDIA eilutę (ir be URI) kiekvieno perkrovimo metu prideda prie c->renditions
             # (hls.c n6.1:824-828, 523-527), atlaisvina tik uždarant (303-311) — ~40 000 eilučių (< 1 MiB) kas 2 s atmetamos
             ("media_many", hdr + b"#EXT-X-MEDIA:TYPE=AUDIO\n" * 40000 + b"#EXTINF:2.0,\ns.ts\n", "EXT-X-MEDIA"),
             # M2 (round 12): libav kiekvienam segmentui kopijuoja galiojančio rakto absoliutų URL (hls.c n6.1:948-975) — ~4 KiB rakto URI
             # ir 5000 trumpų segmentų (~0,1 MiB grojaraštis) — ~20 MiB libav atminties kas perkrovimą; atmetama (> 16 MiB grojaraščiui)
             ("key_long", hdr + b'#EXT-X-KEY:METHOD=AES-128,URI="' + b"k" * 4000 + b'"\n' + b"#EXTINF:2.0,\ns.ts\n" * 5000, "segmentų sąmata"))
    ports = (51, 52, 53, 59, 60, 61, 65, 68)  # BASE_PORT + …; 54 — ingest_queue_byte_cap
    res = []
    T, SWAP = 30, 12
    try:
        t0 = time.monotonic()
        while not (work / "live.m3u8").exists() and time.monotonic() - t0 < 20:
            time.sleep(0.2)
        time.sleep(4.5)
        for k, (name, body, want) in enumerate(cases):
            tid = f"ingest_hls_security[{name}]"
            srv.routes.pop("/live.m3u8", None)  # statinis failas (segmentuotojas)
            rss = {"before": 0, "max_after": 0, "swapped": False}

            def swap(b=body):
                rss["before"] = rss["last"]
                rss["swapped"] = True
                srv.routes["/live.m3u8"] = b

            def watch(pid):
                v = rss_kb(pid)
                rss["last"] = v
                if rss["swapped"]:
                    rss["max_after"] = max(rss["max_after"], v)

            rss["last"] = 0
            si, sv, ri, err = two_source_http(out, tid, srv, "/live.m3u8", BASE_PORT + ports[k], T, 6 + SWAP, swap, watch=watch)
            problems, a, r = base_checks(si, sv, ri, 20.0, T, want_rc=(0, 3, 4, 5, 6))
            if ri is None or ri < 0 or si is None:
                problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
            # sargo (libav užklausos) atmetimas — ne vėlesnės patikros prisijungiant iš naujo
            line = next((ln for ln in err.splitlines() if want in ln and "sesija nutraukiama" in ln), "")
            if not line:
                problems.append(f"žurnale nėra sargo atmetimo „{want}“ (sesija nutraukiama)")
            if flow(si, 1).get("input_reconnects", 0) < 1:
                problems.append(f"HLS input_reconnects {flow(si, 1).get('input_reconnects')}")
            grow = (rss["max_after"] - rss["before"]) / 1024
            if rss["before"] == 0 or grow > 64:
                problems.append(f"RSS po pakeitimo +{grow:.0f} MB (riba 64 MB; prieš {rss['before'] // 1024} MB)")
            res.append(F.result(tid, "FAIL" if problems else "PASS",
                                why="; ".join(problems) + f" | RSS prieš {rss['before'] // 1024} MB, maks. po +{grow:.0f} MB | {line.strip()[:200]} | " +
                                summary_line(si, sv, a, r)))
    finally:
        seg.terminate()
        seg.wait()
        srv.close()
    return res


def ingest_queue_byte_cap(out):
    """M1 (round 10): HLS serveris tinklo greičiu atiduoda vieną ilgą didelės spartos segmentą (~750 MB, 10 min. medžiagos):
    paketų eilės ribojamos baitais (--max-queue-mb 32), perteklius išmetamas ir skaičiuojamas (queue_drops, žurnalo eilutė), RSS
    po segmento pradžios auga ≤ 2 × riba + 64 MB = 128 MB; procesas gyvena, UDP šaltinis — be praradimų."""
    tid = "ingest_queue_byte_cap"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    work = pathlib.Path(out) / tid / "live"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    seg = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(CLIPS / "av_1080i50.ts"),
                            "-c", "copy", "-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+omit_endlist",
                            str(work / "live.m3u8")], stdout=subprocess.DEVNULL, stderr=open(work.parent / "seg.log", "w"))
    big = work.parent / "big.ts"  # 600 s ištisiniais laikais (-stream_loop su -c copy laikus tęsia; kartojant baitus — nutrūkimai)
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-stream_loop", "3", "-i", str(CLIPS / "av_1080i50.ts"),
                    "-c", "copy", "-f", "mpegts", str(big)], check=True)
    srv = PlaylistServer(work)
    srv.routes["/big.ts"] = StreamBody(big)
    T, SWAP, CAP = 30, 12, 32
    rss = {"before": 0, "max_after": 0, "swapped": False, "last": 0}

    def swap():  # grojaraštis + naujas „segmentas“ big.ts (libav jį parsisiunčia tinklo greičiu)
        rss["before"] = rss["last"]
        rss["swapped"] = True
        srv.routes["/live.m3u8"] = (work / "live.m3u8").read_bytes() + b"#EXTINF:2.0,\nbig.ts\n"

    def watch(pid):
        rss["last"] = rss_kb(pid)
        if rss["swapped"]:
            rss["max_after"] = max(rss["max_after"], rss["last"])

    try:
        t0 = time.monotonic()
        while not (work / "live.m3u8").exists() and time.monotonic() - t0 < 20:
            time.sleep(0.2)
        time.sleep(4.5)
        si, sv, ri, err = two_source_http(out, tid, srv, "/live.m3u8", BASE_PORT + 54, T, 6 + SWAP, swap, ("--max-queue-mb", CAP), watch=watch)
    finally:
        seg.terminate()
        seg.wait()
        srv.close()
        big.unlink(missing_ok=True)
    problems, a, r = base_checks(si, sv, ri, 20.0, T, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    grow = (rss["max_after"] - rss["before"]) / 1024
    if rss["before"] == 0 or grow > 2 * CAP + 64:
        problems.append(f"RSS po segmento +{grow:.0f} MB > {2 * CAP + 64} MB")
    if flow(si, 1).get("queue_drops", 0) < 1:
        problems.append(f"HLS queue_drops {flow(si, 1).get('queue_drops')}")
    line = next((ln for ln in err.splitlines() if "eilės riba" in ln), "")
    if not line:
        problems.append("žurnale nėra eilės ribos eilutės")
    if srv.hits.get("/big.ts", 0) < 1:
        problems.append("big.ts neparsisiųstas")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | RSS prieš {rss['before'] // 1024} MB, maks. po +{grow:.0f} MB, queue_drops "
                     f"{flow(si, 1).get('queue_drops')} | {line.strip()[:200]} | " + summary_line(si, sv, a, r))]


def ingest_hls_security(out):
    """A1 (round 9): grojaraščio pateikti URL — tik http / https; įdėti master grojaraščiai (A→B→A) ir milžiniškas
    #EXT-X-TARGETDURATION atmetami su aiškia klaida (procesas baigiasi tvarkingai, ne signalu). file:// kanarėlė (FIFO) niekada
    neatidaroma, loopback TCP klausytojas — niekada neprisijungiamas."""
    res = []
    work = pathlib.Path(out) / "ingest_hls_security"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    fifo = FifoCanary(work / "canary.m3u8")
    tcp = TcpCanary()
    srv = PlaylistServer()
    media = lambda refs, td=2: ("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:%d\n#EXT-X-MEDIA-SEQUENCE:0\n" % td +
                                "".join(f"#EXTINF:2.0,\n{r}\n" for r in refs)).encode()
    master = lambda *vs: ("#EXTM3U\n" + "".join(f"#EXT-X-STREAM-INF:BANDWIDTH=1000000\n{v}\n" for v in vs)).encode()
    srv.routes.update({
        "/good.m3u8": media(["seg0.ts"]),
        "/file_variant.m3u8": master(f"file://{fifo.path}", "good.m3u8"),
        "/file_segment.m3u8": media([f"file://{fifo.path}"]),
        "/key_file.m3u8": ("#EXTM3U\n#EXT-X-TARGETDURATION:2\n" + f"#EXT-X-KEY:METHOD=AES-128,URI=\"file://{fifo.path}\"\n#EXTINF:2.0,\nseg0.ts\n").encode(),
        "/tcp_segment.m3u8": media([f"tcp://127.0.0.1:{tcp.port}"]),
        "/a.m3u8": master("b.m3u8"),
        "/b.m3u8": master("a.m3u8"),
        "/huge.m3u8": media(["seg0.ts"], td=100000),
        # M2 (round 12): 5 variantai po ~14 MiB libav segmentų sąmatos (4000 baitų rakto URI × 3500 segmentų; kiekvienas < 16 MiB
        # grojaraščiui), kartu > 64 MiB sesijai — atmetama patikroje prieš libav
        "/sess.m3u8": master(*[f"sv{k}.m3u8" for k in range(5)]),
        **{f"/sv{k}.m3u8": ("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-KEY:METHOD=AES-128,URI=\"" + "k" * 4000 + "\"\n" +
                            "#EXTINF:2.0,\ns.ts\n" * 3500).encode() for k in range(5)},
    })
    cases = (("file_variant", "neleidžiama"), ("file_segment", "neleidžiama"), ("key_file", "neleidžiama"), ("tcp_segment", "neleidžiama"),
             ("a", "įdėtas master"), ("huge", "TARGETDURATION"), ("sess", "visiems grojaraščiams"))
    try:
        res += hls_toctou_case(out)
        res += hls_peek_case(out)
        res += hls_redirect_case(out)
        res += hls_authority_case(out)
        res += huge_length_cases(out)
        res += hls_dot_location_case(out)
        res += hls_segment_m3u8_case(out)
        res += hls_reload_cases(out)
        for name, want in cases:
            tid = f"ingest_hls_security[{name}]"
            d = F.fresh_domain(tid.replace("[", "-").replace("]", ""))
            t0 = time.monotonic()
            p = F.FProc(out, tid, "ingest", d, "--input", srv.url(f"/{name}.m3u8"), "--input-wait-s", 8, *F.vcommon("gpu", "cuda-linear"))
            rc = p.wait(timeout=40)
            if rc is None:
                p.term()
            dt = time.monotonic() - t0
            fatal = (F.vsummary(p, "ingest") or {}).get("fatal", "")
            problems = []
            if rc is None or rc <= 0:
                problems.append(f"exit {rc} (laukta tvarkinga klaida > 0, ne signalas / kabojimas)")
            if want not in fatal:
                problems.append(f"fatal be „{want}“")
            if fifo.opened():
                problems.append("file:// kanarėlė ATIDARYTA")
            if tcp.connections:
                problems.append(f"TCP kanarėlė prisijungta {tcp.connections} k.")
            if flows_in(d):
                problems.append(f"flow'ai domene {flows_in(d)}")
            res.append(F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | exit {rc} per {dt:.1f} s, fatal {fatal[:220]!r}"))
    finally:
        srv.close()
        tcp.close()
        fifo.close()
    return res


def ingest_hls_recursion_live(out):
    """A1: du šaltiniai (UDP + HLS); HLS grojaraštis vidury darbo tampa rekursyvus (A→B→A), segmentuotojas sustoja. Procesas
    gyvena iki galo (ne signalas), UDP šaltinis — be praradimų (A/V, ≥ 99,8 %, 0 pavėluotų), HLS šaltinis bando iš naujo ir
    praneša apie įdėtą master grojaraštį."""
    if (why := ensure_clips()):
        return [F.result("ingest_hls_recursion_live", "FAIL", why=why)]
    tid = "ingest_hls_recursion_live"
    T, SWAP = 40, 12
    work = pathlib.Path(out) / tid / "hls"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    port = BASE_PORT + 40
    seg = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-i", str(CLIPS / "av_1080i50.ts"), "-c", "copy",
                            "-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+omit_endlist",
                            str(work / "live.m3u8")], stdout=subprocess.DEVNULL, stderr=open(work / "seg.log", "w"))
    srv = PlaylistServer(work)
    d = F.fresh_domain(tid)
    ing = vf = snd = None
    try:
        t0 = time.monotonic()
        while not (work / "live.m3u8").exists() and time.monotonic() - t0 < 20:
            time.sleep(0.2)
        time.sleep(4.5)
        ing = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{port}", "--input", srv.url("/live.m3u8"), "--duration", T + 14,
                      "--warmup", 6, *F.vcommon("gpu", "cuda-linear"))
        snd = Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port)
        vf = F.FProc(out, tid, "verify", d, "--mode", "ingest", "--flows", 1, "--flow-base", 0, "--duration", T + 6, "--warmup", 6,
                     "--lag-fields", 3, "--audio", 1, "--input-wait-ms", 30000, "--impl", "gpu", "--storage", "cuda-linear")
        time.sleep(6 + SWAP)
        srv.routes["/live.m3u8"] = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nloop.m3u8\n"
        srv.routes["/loop.m3u8"] = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nlive.m3u8\n"
        seg.terminate()
        vf.wait(timeout=T + 90)
        ri = ing.wait(timeout=T + 90)
    finally:
        for pr in (vf, ing):
            if pr is not None:
                pr.term()
        if snd is not None:
            snd.stop()
        if seg.poll() is None:
            seg.terminate()
        seg.wait()
        srv.close()
    si, sv = F.vsummary(ing, "ingest"), F.vsummary(vf, "verify-ingest")
    err = ing.err_path.read_text(errors="replace") if ing is not None and ing.err_path.exists() else ""
    problems, a, r = base_checks(si, sv, ri, 20.0, T, want_rc=(0, 3, 4, 5, 6))
    if ri is None or ri < 0 or si is None:
        problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
    if "įdėtas master" not in err:
        problems.append("HLS šaltinis nepranešė apie įdėtą master grojaraštį")
    if flow(si, 1).get("input_reconnects", 0) < 1:
        problems.append(f"HLS input_reconnects {flow(si, 1).get('input_reconnects')}")
    line = next((ln for ln in err.splitlines() if "įdėtas master" in ln), "")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | {line.strip()[:200]} | " + summary_line(si, sv, a, r))]


def ingest_secrets_redacted(out):
    """A1: SRT su passphrase — srautas veikia (A/V, ≥ 99,8 %), o slaptažodžio nėra nei stderr, nei santraukoje, nei --stats-file."""
    if (why := ensure_clips()):
        return [F.result("ingest_secrets_redacted", "FAIL", why=why)]
    tid = "ingest_secrets_redacted"
    protos = subprocess.run(["ffmpeg", "-hide_banner", "-protocols"], capture_output=True, text=True).stdout.split()
    if "srt" not in protos:
        return [F.result(tid, "SKIP", why="sisteminis FFmpeg be srt protokolo")]
    T = dur(30)
    secret = "Zx9SecretPass42q"
    port = free_port()
    url = f"srt://127.0.0.1:{port}"
    stats = pathlib.Path(out) / tid / "stats.jsonl"
    stats.parent.mkdir(parents=True, exist_ok=True)
    if stats.exists():
        stats.unlink()
    snd = lambda: Sender(out, f"{tid}/snd", ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-i", str(CLIPS / "av_1080i50.ts"),
                                             "-c", "copy", "-f", "mpegts", f"{url}?mode=caller&latency=120000&passphrase={secret}&pbkeylen=16"])
    si, sv, ri, err, _ = run_case(out, tid, [f"{url}?mode=listener&latency=120000&passphrase={secret}&pbkeylen=16"], [snd], T,
                                  ("--stats-file", stats))
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    leaks = []
    for f in [pathlib.Path(out) / tid / "ingest.err", pathlib.Path(out) / tid / "summary-ingest.json", stats]:
        if not f.exists():
            leaks.append(f"{f.name}: nėra")
        elif secret in f.read_text(errors="replace"):
            leaks.append(f.name)
    if leaks:
        problems.append(f"slaptažodis matomas / failo nėra: {leaks}")
    if "passphrase=***" not in err:
        problems.append("stderr be išvalyto URL (passphrase=***)")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + " | " + summary_line(si, sv, a, r))]


def ingest_hls_token_redacted(out):
    """L1 (round 10): HLS segmentų URL su žetonu (?token=SECRET, kelyje — tarpas) — libav HLS „Opening '…' for reading“ INFO eilutės
    eina per av_log valymo kabliuką: SECRET nėra nei stderr, nei santraukoje, nei --stats-file, o „Opening“ eilutės su token=*** yra."""
    tid = "ingest_hls_token_redacted"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    secret = "Tk7SecretSeg93"
    T = dur(20)
    work = pathlib.Path(out) / tid / "hls"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    stats = pathlib.Path(out) / tid / "stats.jsonl"
    stats.unlink(missing_ok=True)
    seg = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-i", str(CLIPS / "av_1080i50.ts"), "-c", "copy",
                            "-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+omit_endlist",
                            "-hls_segment_filename", str(work / "live seg%d.ts"), str(work / "live.m3u8")],
                           stdout=subprocess.DEVNULL, stderr=open(work / "seg.log", "w"))
    srv = PlaylistServer(work)

    def playlist():  # segmentų eilutės — su žetonu (serveris užklausą atmeta, failą randa pagal kelią)
        t = (work / "live.m3u8").read_text() if (work / "live.m3u8").exists() else ""
        return "".join(ln.replace(" ", "%20") + f"?token={secret}\n" if ln.endswith(".ts") else ln + "\n" for ln in t.splitlines()).encode()

    srv.routes["/live.m3u8"] = playlist
    try:
        t0 = time.monotonic()
        while not (work / "live.m3u8").exists() and time.monotonic() - t0 < 20:
            time.sleep(0.2)
        time.sleep(4.5)
        si, sv, ri, err, _ = run_case(out, tid, [srv.url("/live.m3u8")], [], T, ("--stats-file", stats), warm=10)
    finally:
        seg.terminate()
        seg.wait()
        srv.close()
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    leaks = []
    for f in [pathlib.Path(out) / tid / "ingest.err", pathlib.Path(out) / tid / "summary-ingest.json", stats]:
        if not f.exists():
            leaks.append(f"{f.name}: nėra")
        elif secret in f.read_text(errors="replace"):
            leaks.append(f.name)
    if leaks:
        problems.append(f"žetonas matomas / failo nėra: {leaks}")
    opening = [ln for ln in err.splitlines() if "Opening" in ln and "token=***" in ln]
    if len(opening) < 3:
        problems.append(f"„Opening … token=***“ eilučių {len(opening)} (< 3: valymo kabliukas nepatikrintas)")
    return [F.result(tid, "FAIL" if problems else "PASS",
                     why="; ".join(problems) + f" | {opening[0].strip()[:160] if opening else ''} | " + summary_line(si, sv, a, r))]


def ingest_srt_passphrase_file(out):
    """L1 (round 10): SRT slaptažodis iš failo (--srt-passphrase-from file:…), ne iš argv: srautas veikia (A/V, ≥ 99,8 %), slaptažodžio
    nėra nei proceso /proc/<pid>/cmdline, nei stderr, santraukoje ar --stats-file; stderr — passphrase=*** (įdėtas URL išvalytas)."""
    tid = "ingest_srt_passphrase_file"
    if (why := ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    protos = subprocess.run(["ffmpeg", "-hide_banner", "-protocols"], capture_output=True, text=True).stdout.split()
    if "srt" not in protos:
        return [F.result(tid, "SKIP", why="sisteminis FFmpeg be srt protokolo")]
    T = dur(20)
    secret = "Qw8FilePass55z"
    port = free_port()
    url = f"srt://127.0.0.1:{port}"
    d = pathlib.Path(out) / tid
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "srt.pass"
    pf.write_text(secret + "\n")
    os.chmod(pf, 0o600)
    stats = d / "stats.jsonl"
    stats.unlink(missing_ok=True)
    cmdlines = []

    class Snd(Sender):  # siuntėjas (ffmpeg) slaptažodį gauna URL — tai siuntėjas, ne ingest; ingest cmdline tikrinama atskirai
        pass

    snd = lambda: Snd(out, f"{tid}/snd", ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-i", str(CLIPS / "av_1080i50.ts"),
                                          "-c", "copy", "-f", "mpegts", f"{url}?mode=caller&latency=120000&passphrase={secret}&pbkeylen=16"])

    def grab():  # ingest proceso komandinė eilutė (pid — naujausias vramflow-func ingest su šiuo prievadu)
        import glob
        for c in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                b = open(c, "rb").read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if "ingest" in b and f":{port}" in b and "vramflow-func" in b:
                cmdlines.append(b)
        return Snd(out, f"{tid}/noop", ["true"])

    si, sv, ri, err, _ = run_case(out, tid, [f"{url}?mode=listener&latency=120000&pbkeylen=16"], [snd, grab], T,
                                  ("--stats-file", stats, "--srt-passphrase-from", f"file:{pf}"))
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    leaks = [f.name for f in (d / "ingest.err", d / "summary-ingest.json", stats) if not f.exists() or secret in f.read_text(errors="replace")]
    if leaks:
        problems.append(f"slaptažodis matomas / failo nėra: {leaks}")
    if not cmdlines:
        problems.append("ingest /proc/<pid>/cmdline nerasta")
    elif any(secret in c for c in cmdlines):
        problems.append("slaptažodis ingest komandinėje eilutėje")
    if "passphrase=***" not in err:
        problems.append("stderr be išvalyto URL (passphrase=***)")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | cmdline tikrinta {len(cmdlines)} | " + summary_line(si, sv, a, r))]


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


def ingest_audio_absent_reconnect(out):
    """A2 (round 9): po prisijungimo iš naujo garso dekoderio atidaryti nepavyksta ([undecodable] — paslėptas
    --test-audio-reopen-fail: kiekvienas atidarymas po pirmo nepavyksta, kaip nedekoduojamas kodekas) arba garso srauto nebėra
    ([missing] — antras siuntėjas siunčia tik video). Procesas nekrenta (anksčiau — avcodec_send_packet(nullptr)), video tęsiasi po
    grįžimo, garsas — tyla (audio_absent ≥ 1, tylos mėginių ≥ antros sesijos trukmė)."""
    if (why := ensure_clips()):
        return [F.result("ingest_audio_absent_reconnect", "FAIL", why=why)]
    vonly = pathlib.Path(out) / "ingest_audio_absent_reconnect" / "video_only.ts"
    vonly.parent.mkdir(parents=True, exist_ok=True)
    if not vonly.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(CLIPS / "av_1080i50.ts"), "-map", "0:v", "-c", "copy",
                        "-t", "60", "-f", "mpegts", str(vonly)], check=False)
    res = []
    for k, (case, clip2, extra) in enumerate((("undecodable", CLIPS / "av_1080i50.ts", ("--test-audio-reopen-fail", 1)),
                                              ("missing", vonly, ()))):
        tid, port = f"ingest_audio_absent_reconnect[{case}]", BASE_PORT + 41 + k
        UP1, GAP, UP2 = 15, 6, 20
        T = UP1 + GAP + UP2
        d = F.fresh_domain(tid.replace("[", "-").replace("]", ""))
        ing = s1 = s2 = None
        try:
            ing = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{port}", "--duration", T + 10, "--warmup", 4,
                          *F.vcommon("gpu", "cuda-linear"), *extra)
            s1 = Sender.ts(out, f"{tid}/snd1", CLIPS / "av_1080i50.ts", port)
            time.sleep(UP1)
            s1.stop()
            time.sleep(GAP)
            s2 = Sender.ts(out, f"{tid}/snd2", clip2, port)
            ri = ing.wait(timeout=T + 60)
        finally:
            if ing is not None:
                ing.term()
            for s in (s1, s2):
                if s is not None:
                    s.stop()
        si = F.vsummary(ing, "ingest")
        fi = flow(si)
        err = ing.err_path.read_text(errors="replace") if ing.err_path.exists() else ""
        problems = []
        if ri is None or ri < 0 or si is None:
            problems.append(f"procesas: exit {ri}, santrauka {'yra' if si else 'nėra'}")
        if fi.get("input_reconnects", 0) < 1:
            problems.append(f"input_reconnects {fi.get('input_reconnects')}")
        if fi.get("audio_absent", 0) < 1 or "garso nėra" not in err:
            problems.append(f"audio_absent {fi.get('audio_absent')} / žurnale nėra „garso nėra“")
        # video tęsiasi po grįžimo: dekoduota ≥ 90 % abiejų sesijų kadrų (25 kadrai/s)
        if fi.get("frames_decoded", 0) < (UP1 + UP2 - 8) * 25 * 0.9:
            problems.append(f"frames_decoded {fi.get('frames_decoded')} (video po grįžimo?)")
        if fi.get("audio_silent_samples", 0) < (UP2 - 6) * 48000:
            problems.append(f"audio_silent_samples {fi.get('audio_silent_samples')} (garsas po grįžimo — ne tyla?)")
        line = next((ln for ln in err.splitlines() if "garso nėra" in ln), "")
        res.append(F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | {line.strip()[:200]} | exit {ri}, "
                            f"frames_decoded {fi.get('frames_decoded')}, audio_absent {fi.get('audio_absent')}, "
                            f"audio_absent_pkts {fi.get('audio_absent_pkts')}, silent {fi.get('audio_silent_samples')}"))
    return res


def ingest_live_stats_fallback(out):
    """B8: NVDEC neatsidaro (paslėptas --test-hw-decoder-fail) → programinis dekoderis su žurnalo eilute ir decoder_fallbacks == 1,
    srautas veikia (A/V, ≥ 99,8 %); šaltinio skaitikliai — gyvai --stats-file eilutėse (flows[n].fn), ne tik pabaigoje:
    Σ eilučių frames_decoded == santraukos, ne nulis bent 5 eilutėse prieš paskutinę."""
    if (why := ensure_clips()):
        return [F.result("ingest_live_stats_fallback", "FAIL", why=why)]
    import json
    tid, port = "ingest_live_stats_fallback", BASE_PORT + 43
    T = dur(20)
    stats = pathlib.Path(out) / tid / "stats.jsonl"
    stats.parent.mkdir(parents=True, exist_ok=True)
    if stats.exists():
        stats.unlink()
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"], [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port)], T,
                                  ("--stats-file", stats, "--test-hw-decoder-fail", 1))
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    fi = flow(si)
    if fi.get("decoder_fallbacks", 0) != 1 or "perjungiama į programinį" not in err:
        problems.append(f"decoder_fallbacks {fi.get('decoder_fallbacks')} / žurnale nėra perjungimo")
    lines = []
    for ln in (stats.read_text(errors="replace").splitlines() if stats.exists() else []):
        try:
            j = json.loads(ln)
        except ValueError:
            continue
        if "flows" in j:
            lines.append(((j["flows"][0].get("fn") or {}).get("frames_decoded")))
    total = sum(x or 0 for x in lines)
    live = sum(1 for x in lines[:-1] if x)
    # Σ eilučių ≤ santrauka: kadrai, dekoduoti po paskutinio tick'o (variklis jau sustojo), eilutėse nebūna — ≤ 1 s (25 kadrai)
    gap = (fi.get("frames_decoded") or 0) - total
    if not lines or not 0 <= gap <= 25 or live < 5:
        problems.append(f"stats eilutės: {len(lines)}, Σ frames_decoded {total} (santrauka {fi.get('frames_decoded')}), gyvų eilučių {live}")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | eilučių {len(lines)}, gyvų {live}, Σ {total} | " +
                     summary_line(si, sv, a, r))]


def ingest_hls_by_content(out):
    """B8: HLS atpažįstamas pagal turinį — grojaraštis URL be .m3u8 galūnės (/live): latencija ≥ 3 segmentai, A/V ir ≥ 99,8 %."""
    if (why := ensure_clips()):
        return [F.result("ingest_hls_by_content", "FAIL", why=why)]
    tid = "ingest_hls_by_content"
    T = dur(30)
    work = pathlib.Path(out) / tid / "hls"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    seg = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-i", str(CLIPS / "av_1080i50.ts"), "-c", "copy",
                            "-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+omit_endlist",
                            str(work / "live.m3u8")], stdout=subprocess.DEVNULL, stderr=open(work / "seg.log", "w"))
    srv = PlaylistServer(work)
    srv.routes["/live"] = lambda: (work / "live.m3u8").read_bytes() if (work / "live.m3u8").exists() else b""
    try:
        t0 = time.monotonic()
        while not (work / "live.m3u8").exists() and time.monotonic() - t0 < 20:
            time.sleep(0.2)
        time.sleep(4.5)
        si, sv, ri, err, _ = run_case(out, tid, [srv.url("/live")], [], T, warm=10)
    finally:
        seg.terminate()
        seg.wait()
        srv.close()
    problems, a, r = base_checks(si, sv, ri, 20.0, T)
    lat = (si or {}).get("proc", {}).get("latency_ms")
    if lat is None or lat < 3 * 2000:
        problems.append(f"latencija {lat} ms < 3 segmentai (HLS neatpažintas?)")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | latencija {lat} ms | " + summary_line(si, sv, a, r))]


def ingest_forced_relock(out):
    """B4: siuntėjas nuo 20 s visus paketus vėlina +600 ms visam laikui (latencijos pokytis > drift_limit) → apsaugos
    perfiksavimas: įvykis žurnale („perfiksavimas“ su priežastimi ir dydžiu), santraukoje ir gyvoje statistikoje (relocks),
    eilėje laukę video paketai išmesti / perlaikyti kartu su garsu, garsas perjungtas su perėjimu; po perfiksavimo |A/V| ≤ 1 laukas."""
    if (why := ensure_clips()):
        return [F.result("ingest_forced_relock", "FAIL", why=why)]
    import json
    tid, port = "ingest_forced_relock", BASE_PORT + 49
    T, SHIFT = 75, 20
    stats = pathlib.Path(out) / tid / "stats.jsonl"
    stats.parent.mkdir(parents=True, exist_ok=True)
    if stats.exists():
        stats.unlink()
    si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"],
                                  [lambda: Sender.ts(out, f"{tid}/snd", CLIPS / "av_1080i50.ts", port, "--shift-at-s", SHIFT, "--shift-ms", 600)], T,
                                  ("--stats-file", stats), warm=6)
    fi = flow(si)
    a = av_stats(sv)
    problems = []
    if fi.get("relock_fresh", 0) < 1:
        problems.append(f"relock_fresh {fi.get('relock_fresh')}")
    line = next((ln for ln in err.splitlines() if "perfiksavimas" in ln), "")
    if not line or "ms" not in line:
        problems.append("žurnale nėra perfiksavimo eilutės su dydžiu")
    live = 0
    for ln in (stats.read_text(errors="replace").splitlines() if stats.exists() else []):
        try:
            j = json.loads(ln)
        except ValueError:
            continue
        if "flows" in j:
            live += (j["flows"][0].get("fn") or {}).get("relocks", 0)
    if live < 1:
        problems.append("gyvoje statistikoje relocks == 0")
    i0 = a["raw"][0][0] if a["raw"] else 0
    after = [p[1] / 1000.0 for p in a["raw"] if (p[0] - i0) / 50 > SHIFT + 40]
    if len(after) < 8 or max(abs(x) for x in after) > 20:
        problems.append(f"po perfiksavimo A/V porų {len(after)}, maks. {max((abs(x) for x in after), default=None)} ms")
    if fi.get("late_frames", 0):
        problems.append(f"late_frames {fi.get('late_frames')}")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | {line.strip()[:220]} | gyvai relocks {live} | " +
                     summary_line(si, sv, a, ratio_ok(flow(sv))))]


def ingest_no_idr_entry(out):
    """B7: srautai be IDR vidury — x264 periodinis intra atnaujinimas (intra-refresh, IDR tik pirmas kadras) ir atviras GOP
    (open-gop, I kadrai be IDR); siuntėjas pradeda vidury klipo (--start-s), todėl IDR niekada neateina. Dekodavimas pradedamas nuo
    recovery point / I pjūvio: vaizdas atsiranda per ≤ 3 s, žurnale — įėjimo taškas ir išmestų paketų skaičius, A/V ≤ 1 kadras."""
    if (why := ensure_clips()):
        return [F.result("ingest_no_idr_entry", "FAIL", why=why)]
    res = []
    for k, (name, clip) in enumerate((("intra_refresh", "intra_refresh_1080p25.ts"), ("open_gop", "open_gop_1080p25.ts"))):
        tid, port = f"ingest_no_idr_entry[{name}]", BASE_PORT + 47 + k
        if not (CLIPS / clip).exists():
            res.append(F.result(tid, "FAIL", why=f"nėra {clip}"))
            continue
        # klipas 30 s, siunčiama nuo 2,3 s: ingest (T + 8 + 8 s) baigiasi anksčiau už siuntėją. Langas — nuo 8 s: intra atnaujinimo
        # sraute vaizdas teisėtai atsiranda tik pasibaigus atnaujinimo ciklui (recovery_frame_cnt ≈ 2 s) po zondavimo ir latencijos
        T = 10
        si, sv, ri, err, _ = run_case(out, tid, [f"udp://127.0.0.1:{port}"],
                                      [lambda p=port, c=clip, t=tid: Sender.ts(out, f"{t}/snd", CLIPS / c, p, "--start-s", 2.3)], T, warm=8)
        # intra atnaujinimas: per atnaujinimo ciklą po pirmo kadro dekoderis dalies kadrų dar neišveda (išmatuota: 2 ir 28 — iki ~1 s,
        # prieš matavimo langą) — tai missing_frames ir griežtas exit 4; lange vaizdas vis tiek turi būti ≥ 99,8 %, A/V ≤ 1 kadras
        problems, a, r = base_checks(si, sv, ri, 40.0, T, want_rc=(0, 4) if name == "intra_refresh" else (0,))
        fi = flow(si)
        if fi.get("missing_frames", 0) > 50:
            problems.append(f"missing_frames {fi.get('missing_frames')} > 50 (> ~2 s be vaizdo po įėjimo)")
        line = next((ln for ln in err.splitlines() if "įėjimo taško" in ln), "")
        if not line or "NĖRA" in line:
            problems.append(f"įėjimo taškas: {line.strip()[:160]!r}")
        res.append(F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | {line.strip()[:200]} | " + summary_line(si, sv, a, r)))
    return res


def ingest_reconnect_reprobe(out):
    """B6: po prisijungimo iš naujo pirmas zondavimas nepavyksta (siuntėjas kelias sekundes siunčia tik garsą — video srauto
    nėra) — tai NE formato pasikeitimas: bandoma iš naujo su atidėjimu, šaltinis grįžta, kai vėl ateina A/V (format_changed == 0,
    exit ne 6, po grįžimo kadrai dekoduojami ir A/V ≤ 1 laukas)."""
    if (why := ensure_clips()):
        return [F.result("ingest_reconnect_reprobe", "FAIL", why=why)]
    tid, port = "ingest_reconnect_reprobe", BASE_PORT + 46
    UP1, GAP, BAD, UP2 = 15, 4, 8, 25
    T = UP1 + GAP + BAD + UP2
    d = F.fresh_domain(tid)
    v = F.vcommon("gpu", "cuda-linear")
    ing = vf = None
    snd = []
    try:
        ing = F.FProc(out, tid, "ingest", d, "--input", f"udp://127.0.0.1:{port}", "--duration", T + 12, "--warmup", 6, *v)
        snd.append(Sender.ts(out, f"{tid}/snd1", CLIPS / "av_1080i50.ts", port))
        vf = F.FProc(out, tid, "verify", d, "--mode", "ingest", "--flows", 1, "--duration", T + 6, "--warmup", 6, "--lag-fields", 3, "--audio", 1,
                     "--input-wait-ms", 30000, "--impl", "gpu", "--storage", "cuda-linear")
        time.sleep(UP1)
        snd[-1].stop()
        time.sleep(GAP)
        snd.append(Sender.ts(out, f"{tid}/snd2", CLIPS / "audio_only_90.ts", port))
        time.sleep(BAD)
        snd[-1].stop()
        snd.append(Sender.ts(out, f"{tid}/snd3", CLIPS / "av_1080i50.ts", port))
        vf.wait(timeout=T + 90)
        ri = ing.wait(timeout=T + 90)
    finally:
        for pr in (vf, ing):
            if pr is not None:
                pr.term()
        for s in snd:
            s.stop()
    si, sv = F.vsummary(ing, "ingest"), F.vsummary(vf, "verify-ingest")
    fi = flow(si)
    err = ing.err_path.read_text(errors="replace") if ing is not None and ing.err_path.exists() else ""
    a = av_stats(sv)
    i0 = a["raw"][0][0] if a["raw"] else 0
    after = [p[1] / 1000.0 for p in a["raw"] if (p[0] - i0) / 50 > UP1 + GAP + BAD + 4]
    problems = []
    if fi.get("format_changed", 0) or ri == 6:
        problems.append(f"format_changed {fi.get('format_changed')}, exit {ri} (zondavimo nesėkmė palaikyta formato pasikeitimu)")
    if fi.get("input_reconnects", 0) < 2:
        problems.append(f"input_reconnects {fi.get('input_reconnects')} < 2")
    if len(after) < (UP2 - 8) * 0.6 or not after or max(abs(x) for x in after) > 20:
        problems.append(f"po grįžimo A/V porų {len(after)}, maks. {max((abs(x) for x in after), default=None)} ms")
    line = next((ln for ln in err.splitlines() if "nėra video" in ln or "formatas pasikeitė" in ln), "")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + f" | {line.strip()[:200]} | " +
                     summary_line(si, sv, a, ratio_ok(flow(sv))))]


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
         "ingest_format_change": ingest_format_change, "ingest_unsupported": ingest_unsupported,
         "ingest_hls_security": ingest_hls_security, "ingest_hls_recursion_live": ingest_hls_recursion_live,
         "ingest_secrets_redacted": ingest_secrets_redacted, "ingest_audio_absent_reconnect": ingest_audio_absent_reconnect,
         "ingest_live_stats_fallback": ingest_live_stats_fallback, "ingest_hls_by_content": ingest_hls_by_content}
TESTS["ingest_drift_default"] = ingest_drift_default
TESTS["ingest_reconnect_reprobe"] = ingest_reconnect_reprobe
TESTS["ingest_no_idr_entry"] = ingest_no_idr_entry
TESTS["ingest_forced_relock"] = ingest_forced_relock
TESTS["ingest_relock_audio_resync"] = ingest_relock_audio_resync
TESTS["ingest_codec_change_reconnect"] = ingest_codec_change_reconnect
TESTS["ingest_hls_token_redacted"] = ingest_hls_token_redacted
TESTS["ingest_srt_passphrase_file"] = ingest_srt_passphrase_file
TESTS["ingest_queue_byte_cap"] = ingest_queue_byte_cap
