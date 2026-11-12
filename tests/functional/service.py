#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Tarnybos gyvavimo ciklo testai (planas docs/planai/2026-11-12-tick-variklis.md §6 D8, §9 V19, §14 K-T8, K-T11). Kviečiami iš
functional.py (--only fsvc_run,fsvc_loss,fsvc_second_signal,fsvc_fail_unit). Variantas C (CPU, host) — be GPU laiko triukšmo.

fsvc_run           — logo BE --duration (numatyta 0 — tarnyba) veikia 40 s; SIGTERM → exit 0 per ≤ P + 600 ms, santrauka
                     parašyta (strict_exit false, ≥ 1900 laukų).
fsvc_loss          — tarnyboje (be --duration) aukštupis nužudomas kill -9: SIGTERM → exit 0, santraukoje any_loss true ir
                     missing_fields > 0; tas pats su --duration 20 (griežtas, auto) → exit 4.
fsvc_second_signal — du SIGTERM 20 ms atstumu: logo ir enc baigiasi per ≤ 200 ms po antrojo, logo flow'ų domene nelieka.
fsvc_fail_unit     — logo N = 4, flow'o 1 išėjimas sugenda ilgam (--test-out-error open --test-out-error-flow 1): tik jis
                     sustabdomas (unit_failed, jo išėjimo flow'as dingsta iš domeno bėgimo metu), flow'ai 0, 2, 3 — visi laukai
                     be praradimų; griežtai (--duration 12) exit 2, tarnyboje (SIGTERM) exit 0.
"""
import json
import pathlib
import signal
import subprocess
import sys
import threading
import time

import functional as F  # noqa: E402

V = ("cpu", "host")
P_MS = 20.0


def flow_dir(d, role, n):
    return pathlib.Path(d) / f"{F.role_flow_id(role, n)}.mxl-flow"


def term_timed(pr, signals=1, gap_s=0.0, sig=signal.SIGTERM):
    """signals × sig (gap_s tarpu); grąžina (exit, ms nuo PASKUTINIO signalo iki pabaigos). Laukiama blokuojančiu waitpid
    atskiroje gijoje: Popen.wait(timeout=…) apklausia kas ≤ 50 ms, todėl matavimas būdavo suapvalintas į viršų iki +50 ms
    (A3: 215,7 ms buvo ~170–215 ms)."""
    done = {}

    def reap():
        try:
            done["rc"] = pr.p.wait()  # be timeout — os.waitpid blokuoja, grįžta iškart po pabaigos
        finally:
            done["t"] = time.monotonic()

    th = threading.Thread(target=reap, daemon=True)
    th.start()
    for k in range(signals):
        pr.p.send_signal(sig)
        t = time.monotonic()
        if k + 1 < signals:
            time.sleep(gap_s)
    th.join(timeout=30)
    if th.is_alive():
        pr.p.kill()
        th.join(timeout=10)
    return done.get("rc"), round((done.get("t", time.monotonic()) - t) * 1000.0, 1)


def fields_ok(s):
    return sum(f.get("fields_ok", 0) for f in (s or {}).get("flows") or [])


def fsvc_run(out):
    v = F.vcommon(*V)
    d = F.fresh_domain("fsvc-run")
    g = F.FProc(out, "fsvc-run/gen", "gen", d, "--flows", 1, *v)  # be --duration: tarnyba
    lg = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
            return [F.result("fsvc_run", "FAIL", why="gen flow'as neatsirado")]
        lg = F.FProc(out, "fsvc-run", "logo", d, "--flows", 1, "--warmup", 1, *v)
        time.sleep(40)
        alive = lg.p.poll() is None
        rc, ms = term_timed(lg)
    finally:
        g.term()
    s = lg.summary() or {}
    ok = alive and rc == 0 and ms <= P_MS + 600 and s.get("strict_exit") is False and fields_ok(s) >= 1900
    return [F.result("fsvc_run", "PASS" if ok else "FAIL",
                     why=f"gyvas po 40 s={alive} exit={rc} SIGTERM → pabaiga {ms} ms (riba {P_MS + 600:.0f}) "
                         f"strict_exit={s.get('strict_exit')} any_loss={s.get('any_loss')} fields_ok={fields_ok(s)}")]


def loss_case(out, tag, extra):
    v = F.vcommon(*V)
    d = F.fresh_domain(f"fsvc-loss-{tag}")
    g = F.FProc(out, f"fsvc-loss-{tag}/gen", "gen", d, "--flows", 1, *v)
    lg = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
            return None, None, {"err": "gen flow'as neatsirado"}
        lg = F.FProc(out, f"fsvc-loss-{tag}", "logo", d, "--flows", 1, "--warmup", 1, *extra, *v)
        time.sleep(6)
        g.kill9()  # aukštupio praradimas
        time.sleep(4)
        if "--duration" in extra:
            rc = lg.wait(timeout=60)
            ms = None
        else:
            rc, ms = term_timed(lg)
    finally:
        if g.p.poll() is None:
            g.term()
    return rc, ms, lg.summary() or {}


def fsvc_loss(out):
    res = []
    for tag, extra, want in (("svc", [], 0), ("dur", ["--duration", 20], 4)):
        rc, ms, s = loss_case(out, tag, extra)
        miss = sum(f.get("missing_fields", 0) for f in s.get("flows") or [])
        ok = rc == want and s.get("any_loss") is True and miss > 0 and s.get("strict_exit") is (want == 4)
        res.append(F.result(f"fsvc_loss[{'tarnyba' if tag == 'svc' else '--duration 20'}]", "PASS" if ok else "FAIL",
                            why=f"exit={rc} (laukta {want}) strict_exit={s.get('strict_exit')} any_loss={s.get('any_loss')} "
                                f"missing_fields={miss} {s.get('err', '')}"))
    return res


def fsvc_second_signal(out):
    res = []
    v = F.vcommon(*V)
    d = F.fresh_domain("fsvc-2sig")
    g = F.FProc(out, "fsvc-2sig/gen", "gen", d, "--flows", 2, "--audio", 1, *v)
    sink = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(2)] + [F.role_flow_id(3, n) for n in range(2)]):
            return [F.result("fsvc_second_signal", "FAIL", why="gen flow'ai neatsirado")]
        lg = F.FProc(out, "fsvc-2sig/logo", "logo", d, "--flows", 2, "--warmup", 1, *v)
        port = 27500
        sink = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "perf/ts_sink.py"),
                                 "--port-base", str(port), "--flows", "2", "--seconds", "20", "--out",
                                 str(pathlib.Path(out) / "fsvc-2sig" / "sink.json")], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        # enc su x264 fast: sustabdant išleidžiama eilė ir koderio lookahead (~30 kadrų, prieš pataisą — 1,1 s po antro
        # signalo); antras signalas abu nutraukia
        en = F.FProc(out, "fsvc-2sig/enc", "enc", d, "--flows", 2, "--audio", 1, "--encoder", "x264", "--x264-preset", "fast",
                     "--port-base", port, "--warmup", 1, *v)
        if not F.wait_flows(d, [F.role_flow_id(2, n) for n in range(2)], 30):
            return [F.result("fsvc_second_signal", "FAIL", why="logo išėjimai neatsirado")]
        time.sleep(6)
        for name, pr in (("logo", lg), ("enc", en)):
            rc, ms = term_timed(pr, signals=2, gap_s=0.020)
            left = [n for n in range(2) if flow_dir(d, 2, n).exists()] if name == "logo" else []
            ok = ms <= 200 and not left and rc in (0, 4)
            res.append(F.result(f"fsvc_second_signal[{name}]", "PASS" if ok else "FAIL",
                                why=f"exit={rc} antras SIGTERM → pabaiga {ms} ms (riba 200); likę logo flow'ai {left}"))
    finally:
        if sink is not None and sink.poll() is None:
            sink.kill()
        g.term()
    return res


def fsvc_fail_unit(out):
    res = []
    for tag, extra, want in (("strict", ["--duration", 12], 2), ("svc", [], 0)):
        v = F.vcommon(*V)
        d = F.fresh_domain(f"fsvc-fail-{tag}")
        g = F.FProc(out, f"fsvc-fail-{tag}/gen", "gen", d, "--flows", 4, *v)
        lg = None
        gone_during = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(4)]):
                res.append(F.result(f"fsvc_fail_unit[{tag}]", "FAIL", why="gen flow'ai neatsirado"))
                continue
            lg = F.FProc(out, f"fsvc-fail-{tag}", "logo", d, "--flows", 4, "--warmup", 1, "--test-out-error", "open",
                         "--test-out-error-after", 3, "--test-out-error-flow", 1, *extra, *v)
            F.wait_flows(d, [F.role_flow_id(2, n) for n in range(4)], 30)
            time.sleep(7)
            # sugedusio vieneto išėjimas atlaisvintas bėgimo metu (žemupys mato flow dingusį), kitų — yra
            gone_during = (not flow_dir(d, 2, 1).exists(), all(flow_dir(d, 2, n).exists() for n in (0, 2, 3)))
            rc = lg.wait(timeout=60) if extra else term_timed(lg)[0]
        finally:
            g.term()
            if lg is not None and lg.p.poll() is None:
                lg.term()
        s = lg.summary() or {}
        fl = s.get("flows") or []
        bad = []
        if rc != want:
            bad.append(f"exit {rc} (laukta {want})")
        if gone_during != (True, True):
            bad.append(f"flow'ai bėgimo metu (1 dingęs, kiti yra) = {gone_during}")
        if len(fl) < 4 or fl[1].get("unit_failed") != 1:
            bad.append(f"flow 1 unit_failed={fl[1].get('unit_failed') if len(fl) > 1 else None}")
        for n in (0, 2, 3):
            f = fl[n] if n < len(fl) else {}
            lost = {k: f[k] for k in ("late_fields", "lapped", "too_late", "in_invalid", "missing_fields", "late_skips",
                                      "output_errors") if f.get(k)}
            if lost or f.get("unit_failed"):
                bad.append(f"sveiko #{n} praradimai {lost} unit_failed={f.get('unit_failed')}")
        ok_f = [fl[n].get("fields_ok", 0) for n in (0, 2, 3)] if len(fl) >= 4 else []
        if not ok_f or min(ok_f) < 300 or len(set(ok_f)) != 1:  # tarnyboje ~8 s, griežtai 12 s; visi sveiki — po lygiai
            bad.append(f"sveikų fields_ok {ok_f}")
        res.append(F.result(f"fsvc_fail_unit[{tag}]", "FAIL" if bad else "PASS",
                            why=f"{'; '.join(bad) or 'tik flow 1 sustabdytas, kiti be praradimų'} | exit={rc} "
                                f"failed_units={(s.get('proc') or {}).get('failed_units')} fatal={s.get('fatal')!r} sveikų fields_ok={ok_f}"))
    return res


TESTS = {"fsvc_run": fsvc_run, "fsvc_loss": fsvc_loss, "fsvc_second_signal": fsvc_second_signal, "fsvc_fail_unit": fsvc_fail_unit}
