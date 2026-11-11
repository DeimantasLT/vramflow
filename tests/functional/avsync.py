#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""A/V sinchronizacijos (lip-sync) funkciniai testai enc funkcijai (kodo auditas 2026-11-11, radinys 2). Kviečiami iš
functional.py (--only favsync_drop,favsync_restart).

favsync_drop    — enc su paslėpta testine parinktimi `--test-drop-every 37 --test-drop-after 8`: nuo 8-os sekundės kas 37-as
                  kadras atmetamas tuo pačiu keliu kaip pilnos eilės atveju (skaitiklis frames_dropped_queue). Išėjimo TS
                  įrašomas ir matuojamas tests/perf/av_offset.py: A/V poslinkis iki pirmos video spragos ir po jos
                  ≤ 1 AAC kadras (21,3 ms), audio paketų PTS monotoniški, dekodavimo klaidų 0, TS be sync / CC klaidų.
favsync_restart — gen (+ audio) nužudomas kill -9 ir po 3 s paleidžiamas vėl (realus „trūksta įėjimo“ kelias, kaip
                  fstart_restart); tos pačios A/V sąlygos prieš spragą ir po jos.
Variantai: kiekvienam formatui (FAVSYNC_FORMATS, numatyta 1080i50,1080p25) — C (cpu + x264) ir G (gpu + nvenc);
FAVSYNC_VARIANTS=C,G.

Testai nepriklauso nuo enc vidaus: tik CLI, exit kodai, summary skaitikliai ir įrašytas TS.
Pastaba: matavimas periodinis (1 s), todėl sveikų sekundžių poslinkis nematomas; favsync_drop spragos (po 40 ms) kaupiasi ir
sveikos sekundės nesudaro (8 atmetimai × 40 ms = 320 ms).
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import functional as F  # noqa: E402

AV_OFFSET = pathlib.Path(__file__).resolve().parents[1] / "perf" / "av_offset.py"
AAC_FRAME_MS = 1024 / 48.0
VARIANTS = {"C": ("cpu", "host", "x264"), "G": ("gpu", "cuda-linear", "nvenc")}


def formats():
    return [x for x in os.environ.get("FAVSYNC_FORMATS", "1080i50,1080p25").split(",") if x]


def variants():
    return [(t, *VARIANTS[t]) for t in os.environ.get("FAVSYNC_VARIANTS", "C,G").split(",") if t]


def windows(av):
    """Poslinkiai iki pirmos video spragos (be pirmos pusės sekundės) ir po jos (+ 0,5 s) — visi, ir tarp kitų spragų."""
    pairs = av.get("pairs") or []
    gaps = av.get("video_gaps") or []
    if not pairs or not gaps:
        return None, None
    t0, g_first = pairs[0]["t"], gaps[0]["t"]

    def summ(sel):
        offs = [p["offset_ms"] for p in pairs if sel(p["t"])]
        return {"n": len(offs), "max_abs_ms": max((abs(x) for x in offs), default=None),
                "mean_ms": round(sum(offs) / len(offs), 3) if offs else None}

    return summ(lambda t: t0 + 0.5 < t < g_first - 0.1), summ(lambda t: t > g_first + 0.5)


def av_verdict(av, before, after, min_before=3, min_after=4):
    sink = av.get("sink") or {}
    checks = {
        "pora_pries": before is not None and before["n"] >= min_before and before["max_abs_ms"] <= AAC_FRAME_MS,
        "pora_po": after is not None and after["n"] >= min_after and after["max_abs_ms"] <= AAC_FRAME_MS,
        "audio_pts_monotoniski": av.get("audio_pts_backward", 1) == 0 and av.get("audio_frames_monotonic") is True,
        "dekodavimo_klaidu_0": av.get("decode_errors", 1) == 0,
        "ts_be_klaidu": sink.get("sync_errors", 1) == 0 and sink.get("cc_errors", 1) == 0,
    }
    return all(checks.values()), checks


def run_case(out, tid, sub, fmt, impl, storage, enc, port, D, enc_extra, restart_gen=False):
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage) + ["--format", fmt]
    gen_args = ["--flows", 1, "--audio", 1, "--duration", D + 40, *v]
    g = F.FProc(out, f"{sub}/a", "gen", d, *gen_args)
    g2 = e = cap = None
    re_ = None
    av_json = pathlib.Path(out) / sub / "av.json"
    ts = pathlib.Path(out) / sub / "out.ts"
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
            return F.result(tid, "FAIL", why="gen flow'ai neatsirado")
        # įrašas startuoja prieš enc: TS prasideda nuo pirmo IDR (be dekodavimo klaidų dėl prisijungimo vidury GOP)
        cap = subprocess.Popen([sys.executable, str(AV_OFFSET), str(ts), "--capture-port", str(port), "--seconds", str(D + 3),
                                "--json", str(av_json)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        ea = ["--encoder", enc, "--audio", 1, "--port-base", port] + (["--x264-preset", "ultrafast"] if enc == "x264" else [])
        e = F.FProc(out, sub, "enc", d, "--flows", 1, "--duration", D, "--warmup", 2, *ea, *enc_extra, *v)
        if restart_gen:
            time.sleep(8)
            g.kill9()
            time.sleep(3)
            g2 = F.FProc(out, f"{sub}/b", "gen", d, *gen_args)
        re_ = e.wait(timeout=D + 60)
        cap.wait(timeout=D + 120)
    finally:
        if cap is not None and cap.poll() is None:
            cap.kill()
        for pr in (e, g2, g):
            if pr is not None:
                pr.term()
    se = F.vsummary(e, "enc")
    fl = ((se or {}).get("flows") or [{}])[0]
    try:
        av = json.loads(av_json.read_text())
    except (OSError, ValueError):
        return F.result(tid, "FAIL", why=f"av_offset rezultato nėra (enc exit={re_})", enc=se)
    before, after = windows(av)
    ok, checks = av_verdict(av, before, after)
    return ok, se, fl, re_, av, before, after, checks


def favsync_drop(out):
    res = []
    D = 22
    for fmt in formats():
        for tag, impl, storage, enc in variants():
            tid = f"favsync_drop[{fmt}/{tag}]"
            sub = f"favsync-drop-{fmt}-{tag}"
            port = 24000 + 10 * len(res)
            r = run_case(out, tid, sub, fmt, impl, storage, enc, port, D, ["--test-drop-every", 37, "--test-drop-after", 8])
            if isinstance(r, dict):
                res.append(r)
                continue
            ok, se, fl, re_, av, before, after, checks = r
            dropped = fl.get("frames_dropped_queue", 0)
            ok = ok and re_ == 4 and dropped >= 5 and len(av.get("video_gaps") or []) >= 5
            res.append(F.result(tid, "PASS" if ok else "FAIL",
                                why=f"prieš spragą {before}, po {after} (riba {AAC_FRAME_MS:.1f} ms); patikros {checks}; "
                                    f"enc exit={re_} frames_dropped_queue={dropped} video spragų {len(av.get('video_gaps') or [])} "
                                    f"audio spragų {len(av.get('audio_gaps') or [])} dekodavimo klaidos {av.get('decode_error_lines')}",
                                enc=se, av={k: av.get(k) for k in ("offset_mean_ms", "offset_max_abs_ms", "video_gaps", "audio_gaps",
                                                                   "audio_pts_backward", "decode_errors", "sink", "pairs")}))
    return res


def favsync_restart(out):
    res = []
    D = 26
    for fmt in formats():
        for tag, impl, storage, enc in variants():
            tid = f"favsync_restart[{fmt}/{tag}]"
            sub = f"favsync-restart-{fmt}-{tag}"
            port = 24500 + 10 * len(res)
            r = run_case(out, tid, sub, fmt, impl, storage, enc, port, D, [], restart_gen=True)
            if isinstance(r, dict):
                res.append(r)
                continue
            ok, se, fl, re_, av, before, after, checks = r
            missing = fl.get("missing_frames", 0)
            ok = ok and re_ == 4 and missing > 0 and fl.get("reattached", 0) >= 1
            res.append(F.result(tid, "PASS" if ok else "FAIL",
                                why=f"prieš spragą {before}, po {after} (riba {AAC_FRAME_MS:.1f} ms); patikros {checks}; "
                                    f"enc exit={re_} missing_frames={missing} reattached={fl.get('reattached')} "
                                    f"video spragos {av.get('video_gaps')} audio spragos {av.get('audio_gaps')} "
                                    f"dekodavimo klaidos {av.get('decode_error_lines')}",
                                enc=se, av={k: av.get(k) for k in ("offset_mean_ms", "offset_max_abs_ms", "video_gaps", "audio_gaps",
                                                                   "audio_pts_backward", "decode_errors", "sink", "pairs")}))
    return res


TESTS = {"favsync_drop": favsync_drop, "favsync_restart": favsync_restart}
