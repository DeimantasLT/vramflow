#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Tarnybos gyvavimo ciklo testai (planas docs/planai/2026-11-12-tick-variklis.md §6 D8, §9 V19, §14 K-T8, K-T11). Kviečiami iš
functional.py (--only fsvc_run,fsvc_loss,fsvc_second_signal,fsvc_fail_unit). Variantas C (CPU, host) — be GPU laiko triukšmo.

fsvc_run           — logo BE --duration (numatyta 0 — tarnyba) veikia 40 s; SIGTERM → exit 0 per ≤ P + 600 ms, santrauka
                     parašyta (strict_exit false, ≥ 1900 laukų).
fsvc_loss          — tarnyboje (be --duration) aukštupis nužudomas kill -9: SIGTERM → exit 0, santraukoje any_loss true ir
                     missing_fields > 0; tas pats su --duration 20 (griežtas, auto) → exit 4.
fsvc_second_signal — du SIGTERM 20 ms atstumu: logo C / G, enc C (x264) / G (nvenc), composite C ir dec C (įėjimas tylus 3 s) baigiasi per
                     ≤ 200 ms po antrojo, exit 0 (tarnyba; anksčiau priimta ir 4), funkcijos išėjimo flow'ų domene nelieka.
fsvc_signals       — SIGHUP ir SIGQUIT kaip SIGTERM: logo baigiasi tvarkingai (exit 0 per ≤ P + 600 ms, santrauka parašyta,
                     išėjimų flow'ų domene nelieka; anksčiau — nužudomas be atlaisvinimo); SIGPIPE ignoruojamas (procesas gyvas).
fenc_drain         — enc C (x264 faster, lookahead): įprasta pabaiga (--duration) su --drain-ms 1 — koderis išleidžiamas visas
                     (packets_video == frames_encoded; anksčiau --drain-ms nukirsdavo ir įprastą pabaigą, nesuskaičiuojant);
                     SIGTERM su --drain-ms 1 — nukirpti kadrai suskaičiuoti (encoder_flush_dropped_frames == frames − packets).
fdec_reconnect     — gen | enc | dec C: enc nužudomas (kill -9) 22 s (> buvęs UDP timeout 20 s), tada paleidžiamas iš naujo:
                     dec įvykis „jungiamasi iš naujo“ (stderr, su laiku), input_reconnects ≥ 1, gyvoje statistikoje flow'as
                     Unhealthy tylos metu ir Healthy po grįžimo, fields_ok po grįžimo auga; du SIGTERM → exit 0 per ≤ 200 ms.
fsvc_fail_unit     — logo N = 4, flow'o 1 išėjimas sugenda ilgam (--test-out-error open --test-out-error-flow 1): tik jis
                     sustabdomas (unit_failed, jo išėjimo flow'as dingsta iš domeno bėgimo metu), flow'ai 0, 2, 3 — visi laukai
                     be praradimų; atkūrimo bandymai su atidėjimu (unit_retries / unit_failures > 1); sustabdant vis dar
                     sugedęs → exit 2 ABIEM režimais (griežtai --duration 12 ir tarnyboje SIGTERM; anksčiau tarnyboje — 0);
                     operatoriaus eilutė „unit_failed“ stderr'e pasirodo ≤ 1 intervalą (1 s) po pirmos išėjimo klaidos;
                     [recover] — klaida praeina (--test-out-error-count 100): vienetas atkuriamas (unit_recovered), exit 0,
                     flow'as 1 vėl commit'ina, jo išėjimo flow'as domene pabaigoje yra.
fsvc_stats         — gyva statistika (planas V21): logo N = 4 su --stats-file; flow'o 0 aukštupis nužudomas (kill -9) ir
                     paleidžiamas iš naujo. Σ ok per eilutes == santraukos fields_ok kiekvienam flow'ui (tiksliai); flow'o 0
                     būsena Unhealthy per ≤ 2 intervalus po kill -9, vėl Healthy per ≤ 4 intervalus po gamintojo grįžimo;
                     sveiki 1…3 — niekada Unhealthy; įvykiai input_detached / input_reattached stderr'e su laiku.
"""
import re
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


def ts_sink(out, sub, port, flows, seconds):
    return subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "perf/ts_sink.py"),
                             "--port-base", str(port), "--flows", str(flows), "--seconds", str(seconds), "--out",
                             str(pathlib.Path(out) / sub / "sink.json")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# (vardas, impl, storage): funkcija, kurios sustabdymas matuojamas; jos išėjimų rolė (None — UDP)
SIG_CASES = (("logo", "cpu", "host"), ("enc", "cpu", "host"), ("logo", "gpu", "cuda-linear"), ("enc", "gpu", "cuda-linear"),
             ("composite", "cpu", "host"), ("dec", "cpu", "host"))
SIG_PORT = 27500


def sig_case(out, func, impl, storage, k):
    """Funkcija func veikia tarnybos režimu 6 s; du SIGTERM 20 ms atstumu → (exit, ms, likę išėjimų flow'ai, klaida)."""
    tag = "C" if impl == "cpu" else "G"
    sub = f"fsvc-2sig/{func}-{tag}"
    v = F.vcommon(impl, storage)
    d = F.fresh_domain(f"fsvc-2sig-{func}-{tag}")
    port = SIG_PORT + 10 * k
    others, target, role = [], None, None
    try:
        g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 2, "--audio", 1, *v)
        others.append(g)
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(2)] + [F.role_flow_id(3, n) for n in range(2)]):
            return None, None, None, "gen flow'ai neatsirado"
        if func == "logo":
            target, role = F.FProc(out, sub, "logo", d, "--flows", 2, "--warmup", 1, *v), 2
        elif func == "enc":
            # x264 fast: sustabdant išleidžiama eilė ir koderio lookahead (~30 kadrų, prieš pataisą — 1,1 s po antro signalo);
            # antras signalas abu nutraukia
            others.append(ts_sink(out, sub, port, 2, 20))
            ea = ["--encoder", "x264", "--x264-preset", "fast"] if impl == "cpu" else ["--encoder", "nvenc"]
            target = F.FProc(out, sub, "enc", d, "--flows", 2, "--audio", 1, *ea, "--port-base", port, "--warmup", 1, *v)
        elif func == "composite":
            x = F.FProc(out, f"{sub}/gfx", "gfxgen", d, "--flows", 2, "--layers", 1, "--layer-geom", "logo", "--mode", "pace", *v)
            others.append(x)
            if not F.wait_flows(d, [F.role_flow_id(5, n * 8) for n in range(2)]):
                return None, None, None, "gfxgen flow'ai neatsirado"
            target, role = F.FProc(out, sub, "composite", d, "--flows", 2, "--layers", 1, "--layer-xy", "1488,64", "--opacity", "1.0",
                                   "--base", "gen", "--out-format", "v210", "--warmup", 1, *v), 6
        elif func == "dec":
            target, role = F.FProc(out, sub, "dec", d, "--flows", 1, "--port-base", port, "--warmup", 1, *v), 4
            time.sleep(1)
            others.append(F.FProc(out, f"{sub}/enc", "enc", d, "--flows", 1, "--encoder", "x264", "--x264-preset", "ultrafast",
                                  "--port-base", port, *v))
        if role is not None and not F.wait_flows(d, [F.role_flow_id(role, n) for n in range(1 if func == "dec" else 2)], 30):
            return None, None, None, f"{func} išėjimai neatsirado"
        time.sleep(6)
        if func == "dec":  # sustabdoma, kai įėjimas tylus (anksčiau: UDP skaitymas be pertraukimo — join iki 20 s)
            others[-1].kill9()
            time.sleep(3)
        rc, ms = term_timed(target, signals=2, gap_s=0.020)
        left = [n for n in range(2) if role is not None and flow_dir(d, role, n).exists()]
        return rc, ms, left, None
    finally:
        for pr in others:
            if isinstance(pr, subprocess.Popen):
                if pr.poll() is None:
                    pr.kill()
            else:
                pr.term()
        if target is not None and target.p.poll() is None:
            target.term()


def fsvc_second_signal(out):
    res = []
    for k, (func, impl, storage) in enumerate(SIG_CASES):
        tid = f"fsvc_second_signal[{func}-{'C' if impl == 'cpu' else 'G'}]"
        rc, ms, left, err = sig_case(out, func, impl, storage, k)
        if err:
            res.append(F.result(tid, "FAIL", why=err))
            continue
        ok = ms <= 200 and not left and rc == 0
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"exit={rc} (laukta 0) antras SIGTERM → pabaiga {ms} ms (riba 200); likę išėjimų flow'ai {left}"))
    return res


def fsvc_signals(out):
    res = []
    for name, sig in (("SIGHUP", signal.SIGHUP), ("SIGQUIT", signal.SIGQUIT), ("SIGPIPE", signal.SIGPIPE)):
        v = F.vcommon(*V)
        sub = f"fsvc-sig-{name.lower()}"
        d = F.fresh_domain(sub)
        g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 1, *v)
        lg = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
                res.append(F.result(f"fsvc_signals[{name}]", "FAIL", why="gen flow'as neatsirado"))
                continue
            lg = F.FProc(out, sub, "logo", d, "--flows", 1, "--warmup", 1, *v)
            F.wait_flows(d, [F.role_flow_id(2, 0)], 30)
            time.sleep(4)
            if sig == signal.SIGPIPE:
                lg.p.send_signal(sig)
                time.sleep(1)
                alive = lg.p.poll() is None
                rc, ms = term_timed(lg)
            else:
                alive = None
                rc, ms = term_timed(lg, sig=sig)
            left = flow_dir(d, 2, 0).exists()
        finally:
            g.term()
            if lg is not None and lg.p.poll() is None:
                lg.term()
        s = lg.summary() or {}
        ok = rc == 0 and ms <= P_MS + 600 and not left and s.get("exit_code") == 0 and (alive is None or alive)
        res.append(F.result(f"fsvc_signals[{name}]", "PASS" if ok else "FAIL",
                            why=f"exit={rc} pabaiga {ms} ms (riba {P_MS + 600:.0f}) santrauka={'yra' if s else 'nėra'} "
                                f"išėjimas liko={left}{'' if alive is None else f' gyvas po SIGPIPE={alive}'}"))
    return res


def enc_streams(pr):
    s = pr.summary() or {}
    return s, (s.get("streams") or [{}])[0], (s.get("flows") or [{}])[0]


def fenc_drain(out):
    res = []
    v = F.vcommon(*V)
    for name, extra in (("pabaiga", ["--duration", 8]), ("SIGTERM", [])):
        sub = f"fenc-drain-{'end' if extra else 'term'}"
        d = F.fresh_domain(sub)
        port = SIG_PORT + 300 + (0 if extra else 10)
        g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 1, "--audio", 1, *v)
        sink = en = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, 0), F.role_flow_id(3, 0)]):
                res.append(F.result(f"fenc_drain[{name}]", "FAIL", why="gen flow'ai neatsirado"))
                continue
            sink = ts_sink(out, sub, port, 1, 16)
            en = F.FProc(out, sub, "enc", d, "--flows", 1, "--audio", 1, "--encoder", "x264", "--x264-preset", "faster",
                         "--port-base", port, "--drain-ms", 1, "--warmup", 1, *extra, *v)
            if extra:
                rc = en.wait(timeout=60)
            else:
                time.sleep(8)
                rc, _ = term_timed(en)
        finally:
            if sink is not None and sink.poll() is None:
                sink.kill()
            g.term()
            if en is not None and en.p.poll() is None:
                en.term()
        s, st, f0 = enc_streams(en)
        fe, pv, cut = st.get("frames_encoded", 0), st.get("packets_video", 0), f0.get("encoder_flush_dropped_frames", 0)
        if extra:
            ok = rc == 0 and fe > 150 and pv == fe and not cut
            want = "packets_video == frames_encoded, nukirpta 0"
        else:
            ok = rc == 0 and fe > 150 and cut > 0 and cut == fe - pv
            want = "encoder_flush_dropped_frames == frames_encoded − packets_video > 0"
        res.append(F.result(f"fenc_drain[{name}]", "PASS" if ok else "FAIL",
                            why=f"exit={rc} frames_encoded={fe} packets_video={pv} encoder_flush_dropped_frames={cut} "
                                f"(laukta {want})"))
    return res


def fdec_reconnect(out):
    """dec įėjimo tyla > 20 s ir grįžimas (žr. modulio aprašą)."""
    v = F.vcommon(*V)
    sub = "fdec-reconnect"
    d = F.fresh_domain(sub)
    port = SIG_PORT + 200
    stats = pathlib.Path(out) / sub / "stats.jsonl"
    g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 1, *v)
    e = e2 = dc = watch = None
    ea = ["--encoder", "x264", "--x264-preset", "ultrafast", "--port-base", port]
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
            return [F.result("fdec_reconnect", "FAIL", why="gen flow'as neatsirado")]
        dc = F.FProc(out, sub, "dec", d, "--flows", 1, "--port-base", port, "--warmup", 1, "--stats-file", stats, *v)
        watch = LogWatch(dc.err_path, {"reconnect": r"srautas 0: .*jungiamasi iš naujo"})
        time.sleep(1)
        e = F.FProc(out, f"{sub}/enc", "enc", d, "--flows", 1, *ea, *v)
        time.sleep(8)
        t_kill = time.clock_gettime_ns(time.CLOCK_TAI)
        e.kill9()
        time.sleep(22)
        t_back = time.clock_gettime_ns(time.CLOCK_TAI)
        e2 = F.FProc(out, f"{sub}/enc2", "enc", d, "--flows", 1, *ea, *v)
        time.sleep(10)
        rc, ms = term_timed(dc, signals=2, gap_s=0.020)
    finally:
        for pr in (e2, g):
            if pr is not None:
                pr.term()
        seen = watch.close() if watch is not None else {}
    s = dc.summary() or {}
    f0 = (s.get("flows") or [{}])[0]
    lines, events = stats_lines(stats)
    bad = []
    if rc != 0 or ms is None or ms > 200:
        bad.append(f"du SIGTERM: exit {rc}, {ms} ms (laukta 0, ≤ 200)")
    if not f0.get("input_reconnects"):
        bad.append(f"input_reconnects={f0.get('input_reconnects')}")
    if "reconnect" not in seen or not (t_kill < seen["reconnect"] < t_back):
        bad.append(f"eilutės „jungiamasi iš naujo“ tylos metu nėra ({seen})")
    if lines is None:
        bad.append(f"stats: {events}")
        lines = []
    st = lambda lo, hi: [x["flows"][0]["state"] for x in lines if lo <= x["tai"] < hi and x.get("flows")]  # noqa: E731
    quiet = st(t_kill + 3_000_000_000, t_back)
    after = st(t_back + 4_000_000_000, t_back + 9_000_000_000)
    ok_after = sum(x["flows"][0]["ok"] for x in lines if x["tai"] >= t_back + 4_000_000_000 and x.get("flows"))
    if not quiet or any(q != "Unhealthy" for q in quiet):
        bad.append(f"tylos metu būsenos {quiet} (laukta Unhealthy)")
    if not after or after[-1] != "Healthy" or ok_after < 100:
        bad.append(f"po grįžimo būsenos {after}, ok {ok_after} (laukta Healthy, ≥ 100)")
    return [F.result("fdec_reconnect", "FAIL" if bad else "PASS",
                     why=f"{'; '.join(bad) or 'įėjimas atkurtas'} | exit={rc} {ms} ms input_reconnects={f0.get('input_reconnects')} "
                         f"fields_ok={f0.get('fields_ok')} tyloje {quiet[:3]}… po grįžimo {after} ok={ok_after}")]


class LogWatch:
    """Seka proceso stderr failą realiu laiku: kada (TAI ns) pirmą kartą pasirodė eilutė su kiekvienu šablonu."""

    def __init__(self, path, patterns):
        self.path, self.patterns, self.seen = pathlib.Path(path), patterns, {}
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                txt = self.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                txt = ""
            now = time.clock_gettime_ns(time.CLOCK_TAI)
            for k, rx in self.patterns.items():
                if k not in self.seen and re.search(rx, txt):
                    self.seen[k] = now
            time.sleep(0.02)

    def close(self):
        self.stop.set()
        self.th.join(timeout=5)
        return self.seen


def stats_lines(path):
    """--stats-file → (eilutės, įvykiai) — JSON eilutės; įvykių eilutės turi raktą „event“."""
    lines, events = [], []
    try:
        for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                j = json.loads(ln)
                (events if "event" in j else lines).append(j)
    except (OSError, ValueError) as e:
        return None, str(e)
    return lines, events


def fail_case(out, tag, extra, count):
    v = F.vcommon(*V)
    d = F.fresh_domain(f"fsvc-fail-{tag}")
    g = F.FProc(out, f"fsvc-fail-{tag}/gen", "gen", d, "--flows", 4, *v)
    lg = watch = None
    gone_during = None
    present_end = None
    stats = pathlib.Path(out) / f"fsvc-fail-{tag}" / "stats.jsonl"
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(4)]):
            return None, {"err": "gen flow'ai neatsirado"}
        lg = F.FProc(out, f"fsvc-fail-{tag}", "logo", d, "--flows", 4, "--warmup", 1, "--test-out-error", "open",
                     "--test-out-error-after", 3, "--test-out-error-flow", 1, "--test-out-error-count", count,
                     "--stats-file", stats, *extra, *v)
        watch = LogWatch(lg.err_path, {"output_error": r"output_error \S+/out0", "unit_failed": r"unit_failed \S+:",
                                       "unit_recovered": r"unit_recovered \S+:"})
        F.wait_flows(d, [F.role_flow_id(2, n) for n in range(4)], 30)
        time.sleep(7)
        # sugedusio vieneto išėjimas atlaisvintas bėgimo metu (žemupys mato flow dingusį), kitų — yra. Atkūrimo bandymai
        # flow'ą trumpam (~60 ms) sukuria iš naujo — imama 5 kartus per 1 s, dingęs turi būti ≥ 4 kartus.
        samples = []
        for _ in range(5):
            samples.append((not flow_dir(d, 2, 1).exists(), all(flow_dir(d, 2, n).exists() for n in (0, 2, 3))))
            time.sleep(0.2)
        gone_during = (sum(a for a, _ in samples) >= 4, all(b for _, b in samples))
        if count:  # atkūrimo atvejis: palaukti atidėjimo (1 s + 2 s) ir patikrinti, kad flow'as grįžo
            time.sleep(4)
        present_end = flow_dir(d, 2, 1).exists()
        rc = lg.wait(timeout=60) if "--duration" in extra else term_timed(lg)[0]
    finally:
        g.term()
        if lg is not None and lg.p.poll() is None:
            lg.term()
        seen = watch.close() if watch is not None else {}
    return rc, {"summary": lg.summary() or {}, "gone_during": gone_during, "present_end": present_end, "seen": seen,
                "stats": stats_lines(stats)}


def fsvc_fail_unit(out):
    res = []
    for tag, extra, want in (("strict", ["--duration", 12], 2), ("svc", [], 2)):
        rc, r = fail_case(out, tag, extra, 0)
        if rc is None:
            res.append(F.result(f"fsvc_fail_unit[{tag}]", "FAIL", why=r["err"]))
            continue
        s, gone_during, seen = r["summary"], r["gone_during"], r["seen"]
        fl = s.get("flows") or []
        bad = []
        if rc != want:
            bad.append(f"exit {rc} (laukta {want}: sustabdant vienetas vis dar sugedęs)")
        if gone_during != (True, True):
            bad.append(f"flow'ai bėgimo metu (1 dingęs, kiti yra) = {gone_during}")
        if len(fl) < 4 or fl[1].get("unit_failed") != 1:
            bad.append(f"flow 1 unit_failed={fl[1].get('unit_failed') if len(fl) > 1 else None}")
        pr = s.get("proc") or {}
        if pr.get("unit_failures", 0) + pr.get("unit_retries", 0) < 2:  # bandyta atkurti (gedimai po bandymų kartojasi)
            bad.append(f"atkūrimo bandymų nėra: proc={pr}")
        if "unit_failed" not in seen or "output_error" not in seen:
            bad.append(f"operatoriaus eilučių nėra: {sorted(seen)}")
        elif seen["unit_failed"] - seen["output_error"] > 1_000_000_000:
            bad.append(f"unit_failed eilutė {(seen['unit_failed'] - seen['output_error']) / 1e6:.0f} ms po pirmos klaidos (> 1 intervalas)")
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
                                f"proc={s.get('proc')} fatal={s.get('fatal')!r} sveikų fields_ok={ok_f} "
                                f"eilutės po klaidos: { {k: round((v - seen.get('output_error', v)) / 1e6) for k, v in seen.items()} } ms"))
    # [recover]: klaida praeina po 100 indeksų — vienetas atkuriamas su atidėjimu (1 s, tada 2 s), tarnyboje exit 0
    rc, r = fail_case(out, "recover", [], 100)
    if rc is None:
        res.append(F.result("fsvc_fail_unit[recover]", "FAIL", why=r["err"]))
        return res
    s, seen = r["summary"], r["seen"]
    fl = s.get("flows") or []
    pr = s.get("proc") or {}
    bad = []
    if rc != 0:
        bad.append(f"exit {rc} (laukta 0: vienetas atsigavo)")
    if len(fl) < 4 or fl[1].get("unit_failed") or not fl[1].get("unit_recoveries"):
        bad.append(f"flow 1: {fl[1] if len(fl) > 1 else None}")
    if not pr.get("unit_recoveries"):
        bad.append(f"unit_recoveries nėra: {pr}")
    if "unit_recovered" not in seen:
        bad.append(f"eilutės unit_recovered nėra: {sorted(seen)}")
    if not r["present_end"]:
        bad.append("flow'o 1 išėjimas po atkūrimo domene nesukurtas")
    if r["gone_during"] is None or not r["gone_during"][1]:
        bad.append(f"sveikų flow'ų būsena {r['gone_during']}")
    ok_f = [fl[n].get("fields_ok", 0) for n in range(4)] if len(fl) >= 4 else []
    if not ok_f or ok_f[1] < 100 or len(set(ok_f[n] for n in (0, 2, 3))) != 1:
        bad.append(f"fields_ok {ok_f}")
    for n in (0, 2, 3):
        f = fl[n] if n < len(fl) else {}
        lost = {k: f[k] for k in ("late_fields", "lapped", "too_late", "in_invalid", "missing_fields", "late_skips", "output_errors")
                if f.get(k)}
        if lost or f.get("unit_failed"):
            bad.append(f"sveiko #{n} praradimai {lost}")
    res.append(F.result("fsvc_fail_unit[recover]", "FAIL" if bad else "PASS",
                        why=f"{'; '.join(bad) or 'flow 1 atkurtas, kiti be praradimų'} | exit={rc} proc={pr} fields_ok={ok_f}"))
    return res


def fsvc_stats(out):
    """Gyva statistika (V21): žr. modulio aprašą."""
    v = F.vcommon(*V)
    d = F.fresh_domain("fsvc-stats")
    stats = pathlib.Path(out) / "fsvc-stats" / "stats.jsonl"
    ga = F.FProc(out, "fsvc-stats/genA", "gen", d, "--flows", 3, "--flow-base", 1, *v)
    gb = F.FProc(out, "fsvc-stats/genB", "gen", d, "--flows", 1, *v)
    gb2 = lg = watch = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(4)]):
            return [F.result("fsvc_stats", "FAIL", why="gen flow'ai neatsirado")]
        lg = F.FProc(out, "fsvc-stats", "logo", d, "--flows", 4, "--warmup", 1, "--stats-file", stats, "--stats-interval", 1, *v)
        watch = LogWatch(lg.err_path, {"detached": r"input_detached \S+/in0", "reattached": r"input_reattached \S+/in0"})
        time.sleep(8)
        t_kill = time.clock_gettime_ns(time.CLOCK_TAI)
        gb.kill9()
        time.sleep(6)
        t_back = time.clock_gettime_ns(time.CLOCK_TAI)
        gb2 = F.FProc(out, "fsvc-stats/genB2", "gen", d, "--flows", 1, *v)
        time.sleep(8)
        rc, _ = term_timed(lg)
    finally:
        for pr in (gb2, ga):
            if pr is not None:
                pr.term()
        seen = watch.close() if watch is not None else {}
    s = lg.summary() or {}
    lines, events = stats_lines(stats)
    if lines is None:
        return [F.result("fsvc_stats", "FAIL", why=f"stats failas: {events}")]
    bad = []
    fl = s.get("flows") or []
    if rc != 0:
        bad.append(f"exit {rc}")
    if len(lines) < 20 or any(x.get("v") != 1 or len(x.get("flows") or []) != 4 for x in lines):
        bad.append(f"eilučių {len(lines)} (≥ 20, v == 1, 4 flow'ai)")
    for n in range(4):
        tot = sum(x["flows"][n]["ok"] for x in lines if len(x.get("flows") or []) > n)
        want = fl[n].get("fields_ok", 0) if n < len(fl) else None
        if tot != want:
            bad.append(f"flow {n}: Σ ok {tot} ≠ fields_ok {want}")
    after_kill = [x for x in lines if x["tai"] >= t_kill]
    st0 = [x["flows"][0]["state"] for x in after_kill]
    if "Unhealthy" not in st0[:2]:
        bad.append(f"flow 0 po kill -9: {st0[:3]} (Unhealthy laukta per ≤ 2 intervalus)")
    after_back = [x["flows"][0]["state"] for x in lines if x["tai"] >= t_back]
    if "Healthy" not in after_back[:4]:
        bad.append(f"flow 0 po grįžimo: {after_back[:5]} (Healthy laukta per ≤ 4 intervalus)")
    for n in (1, 2, 3):
        un = [i for i, x in enumerate(lines[1:], 1) if x["flows"][n]["state"] == "Unhealthy"]
        if un:
            bad.append(f"sveikas {n} Unhealthy eilutėse {un[:5]}")
    if "detached" not in seen or seen["detached"] - t_kill > 1_000_000_000:
        bad.append(f"input_detached eilutė: {seen.get('detached', 0) - t_kill if 'detached' in seen else None} ns po kill")
    if "reattached" not in seen:
        bad.append("input_reattached eilutės nėra")
    gpu = [x.get("gpu") for x in lines]
    return [F.result("fsvc_stats", "FAIL" if bad else "PASS",
                     why=f"{'; '.join(bad) or 'Σ ok == fields_ok, būsenos pagal gedimą'} | eilučių {len(lines)}, įvykių "
                         f"{len(events)}; flow 0 po kill {st0[:3]}, po grįžimo {after_back[:4]}; gpu {gpu[-1]}",
                     example=lines[len(lines) // 2] if lines else None)]


TESTS = {"fsvc_run": fsvc_run, "fsvc_loss": fsvc_loss, "fsvc_second_signal": fsvc_second_signal, "fsvc_fail_unit": fsvc_fail_unit,
         "fsvc_stats": fsvc_stats, "fdec_reconnect": fdec_reconnect, "fsvc_signals": fsvc_signals, "fenc_drain": fenc_drain}
