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
         "fsvc_stats": fsvc_stats}
