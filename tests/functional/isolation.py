#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Gedimų izoliacijos zondai (planas docs/planai/2026-11-12-tick-variklis.md §9, V0b). Kviečiami iš functional.py
(--only fiso_kill,fiso_stall,fiso_composite,fiso_enc).

Klausimas: ką sveikiems flow'ams daro vieno aukštupio rašytojo mirtis (kill -9) ar užstrigimas (SIGSTOP / SIGCONT)? N = 8 flow'ų;
vieno flow'o šaltinis — atskiras procesas (kitas `--flow-base`), likę N − 1 — kito proceso.

fiso_kill      — gen A (N − 1 flow'ų) + gen B (1 flow'as) → logo N → verify N; praėjus 10 s po logo starto B nužudomas (kill -9).
fiso_stall     — tas pats, bet B sustabdomas (SIGSTOP) 5 s ir tęsiamas (SIGCONT).
fiso_composite — gen (N bazių) + gfxgen A (N − 1) + gfxgen B (1 — vieno išėjimo sluoksnio šaltinis) → composite N → verify N;
                 B nužudomas.
fiso_enc       — gen A (N − 1) + gen B (1) → enc N (+ ts_sink); B nužudomas.
Variantai: C (cpu + host, x264) ir G (gpu + cuda-linear, nvenc); FISO_VARIANTS=C,G, FISO_N=8.

Kiekvienas zondas paleidžiamas du kartus: kontrolinis (be gedimo) ir su gedimu — kitaip tokie patys (ta pati trukmė ir `--warmup`,
todėl `out_lateness` histograma abiem apima tą patį langą: nuo 9-os sekundės, t. y. 1 s iki gedimo ir viskas po jo). Kontrolinis
— „prieš“ stulpelis (be gedimo), antrasis — „po“. Santraukos skaitikliai kaupiami visam paleidimui, bet iki gedimo sveikame
paleidime jie 0 (kontrolinis tai patikrina).

Tikrinama savybė: funkcija sveikiems flow'ams nepriduria SAVO praradimų, kai kitas įėjimas miršta ar užstringa. Ką prarado ar
pavėlavo pats sveikas aukštupis (gen A: `late_skips` + `late_commits` — commit'ai vėlesni nei --deadline-ms), funkcija
atkurti negali: sveikų flow'ų praradimai tikrinami ≤ šio skaičiaus (kiekvienam flow'ui), viskas virš jo — FAIL. Aukštupio
sutrikimas rašomas atskira INFO eilute (didžiausias gen A commit'o vėlavimas su gedimu / be jo ir vėlaus tick'o laikas nuo
SIGSTOP / SIGCONT / kill momento).

Būsena: fiso_kill ir fiso_stall — VARTAI logo funkcijai nuo V10 (izoliacija variklyje; tvirtinimai — logo_gate()); fiso_composite
— vartai nuo V11 (ta pati taisyklė, aukštupiai — gen ir gfxgen A); fiso_enc — kol kas INFO (vartais taps V12). FAIL ir tada, jei neveikia pati infrastruktūra (flow'ai neatsirado,
nėra santraukos). Struktūruoti skaičiai — `before` / `after` laukuose (functional.json). „Prieš“ skaičiai —
docs/rezultatai/2026-11-12-izoliacija-pries.md.

Testai tikrina tik iš išorės: procesai, signalai, exit kodai, santraukų skaitikliai, verify išvestis.
"""
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

import functional as F  # noqa: E402

VARIANTS = {"C": ("cpu", "host", "x264"), "G": ("gpu", "cuda-linear", "nvenc")}
N = int(os.environ.get("FISO_N", "8"))
FAULT_AT = float(os.environ.get("FISO_FAULT_AT", "10"))  # s po funkcijos starto
STALL_S = 5.0
RUN_S = 24  # funkcijos trukmė
LOGO_RUN_S = 27  # logo: po SIGCONT (ties 15 s) lieka ≥ 10 s patikrai „0 praradimų po atsigavimo“
POST_S = 10  # logo stall: verify paveiktam flow'ui po SIGCONT
WR_MS = 5.0  # Wr = P/4 prie 50 Hz (planas §6 D2)
WARM_S = FAULT_AT - 1  # histograma nuo 1 s iki gedimo
VERIFY_S = 20
ENC_PORT = int(os.environ.get("FISO_ENC_PORT", "25000"))
# Funkcijos santraukos srautų (flow'ų) skaitikliai, kuriuos registruojame (praradimai ir būsena).
FN_KEYS = ("fields_ok", "late_fields", "late_skips", "warmup_skips", "clock_skips", "lapped", "too_late", "in_invalid", "read_errors",
           "missing_fields", "detached", "flow_invalid", "reopened", "reattached", "stalls")
ENC_KEYS = ("frames_encoded", "late_frames", "late_skips_frames", "warmup_skips_frames", "clock_skips_frames", "missing_frames",
            "lapped", "too_late", "in_invalid", "frames_dropped_queue", "encode_errors", "mux_errors", "reattached")
VER_KEYS = ("samples", "present", "missing", "invalid", "mismatch_fields", "too_late", "read_errors", "reattached", "in_reattached",
            "marker_bad", "verify_skipped", "check_dropped")


def variants():
    want = [x for x in os.environ.get("FISO_VARIANTS", "C,G").split(",") if x]
    return [(t, *VARIANTS[t]) for t in want]


def hist_of(s, name):
    h = ((s or {}).get("hist") or {}).get(name) or {}
    return {k: h.get(k) for k in ("count", "p50_ms", "p99_ms", "max_ms")}


def flow_view(s, keys, idx):
    fl = (s or {}).get("flows") or []
    return {k: sum(fl[n].get(k, 0) for n in idx if n < len(fl)) for k in keys}


def split_summary(s, keys, n_total):
    """Sveiki (0 … N − 2) — sumos, paveiktas (N − 1) — atskirai."""
    return {"healthy": flow_view(s, keys, range(n_total - 1)), "affected": flow_view(s, keys, [n_total - 1])}


def sgn(pr):
    return pr.p.poll()


def kill_at(t0, at):
    d = t0 + at - time.monotonic()
    if d > 0:
        time.sleep(d)


def tai_ns():
    return time.clock_gettime_ns(time.CLOCK_TAI)


FAULT_TAI = {}  # paskutinio apply_fault momentai TAI ns: {"kill"|"stop"|"cont": ns}


def apply_fault(fault, victim, t0):
    """kill: kill -9 ties FAULT_AT; stall: SIGSTOP ties FAULT_AT, SIGCONT po STALL_S. Grąžina faktinį gedimo laiką (s nuo t0);
    momentai TAI laiku — FAULT_TAI (aukštupio vėlaus tick'o laikui palyginti)."""
    FAULT_TAI.clear()
    if fault == "none":
        return None
    kill_at(t0, FAULT_AT)
    t_fault = time.monotonic() - t0
    if fault == "kill":
        FAULT_TAI["kill"] = tai_ns()
        victim.kill9()
    elif fault == "stall":
        FAULT_TAI["stop"] = tai_ns()
        victim.p.send_signal(signal.SIGSTOP)
        time.sleep(STALL_S)
        FAULT_TAI["cont"] = tai_ns()
        victim.p.send_signal(signal.SIGCONT)
    return round(t_fault, 2)


LATE_RE = re.compile(r"\[gen\] vėlus commit: indeksas (\d+), ([\d.]+) ms po T\(i\), TAI (\d+)")


def upstream(pr):
    """Sveiko aukštupio (gen A) sutrikimai: praleisti / vėlūs indeksai, didžiausias commit'o vėlavimas, vėlių tick'ų laikas (ms)
    nuo gedimo momentų."""
    s = pr.summary() if pr is not None else None
    proc = (s or {}).get("proc") or {}
    late = []
    try:
        for m in LATE_RE.finditer(pr.err_path.read_text(encoding="utf-8", errors="replace")):
            tai = int(m.group(3))
            late.append({"index": int(m.group(1)), "late_ms": float(m.group(2)),
                         **{f"vs_{k}_ms": round((tai - v) / 1e6, 1) for k, v in FAULT_TAI.items()}})
    except OSError:
        pass
    return {"late_skips": proc.get("late_skips", 0), "late_commits": proc.get("late_commits", 0),
            "max_lateness_ms": hist_of(s, "commit_lateness").get("max_ms"), "late_ticks": late[:10]}


def stop_all(procs):
    for pr in procs:
        if pr is not None and pr.p.poll() is None:
            try:
                pr.p.send_signal(signal.SIGCONT)  # sustabdytas procesas SIGTERM'o nematytų
            except ProcessLookupError:
                pass
            pr.term()


def common_record(fault, t_fault):
    return {"fault": fault, "fault_at_s": t_fault, "stall_s": STALL_S if fault == "stall" else None}


# --- logo ----------------------------------------------------------------------------------------------------------------

def run_logo(out, tag, impl, storage, fault):
    sub = f"fiso-logo-{tag}-{fault}"
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    ga = F.FProc(out, f"{sub}/genA", "gen", d, "--flows", N - 1, "--duration", RUN_S + 60, *v)
    gb = F.FProc(out, f"{sub}/genB", "gen", d, "--flows", 1, "--flow-base", N - 1, "--duration", RUN_S + 60, *v)
    lg = vf = vp = None
    rl = rv = rp = None
    t_fault = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)]):
            return {"err": "gen flow'ai neatsirado"}
        t0 = time.monotonic()
        lg = F.FProc(out, sub, "logo", d, "--flows", N, "--duration", LOGO_RUN_S, "--warmup", WARM_S, *v)
        if not F.wait_flows(d, [F.role_flow_id(2, n) for n in range(N)], 30):
            return {"err": "logo išėjimo flow'ai neatsirado"}
        vf = F.FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 5, "--duration", VERIFY_S, "--warmup", 1,
                     *F.vimpl(impl, storage))
        t_fault = apply_fault(fault, gb, t0)
        if fault == "stall":  # po SIGCONT: paveiktas flow'as POST_S s — be praradimų (1 s įšilimas atsigavimui)
            vp = F.FProc(out, f"{sub}/post", "verify", d, "--mode", "logo", "--flows", 1, "--flow-base", N - 1, "--every", 1,
                         "--duration", POST_S, "--warmup", 1, *F.vimpl(impl, storage))
            rp = vp.wait(timeout=POST_S + 60)
        rv = vf.wait(timeout=VERIFY_S + 60)
        rl = lg.wait(timeout=LOGO_RUN_S + 60)
    finally:
        stop_all([vp, vf, lg, gb, ga])
    sl, sv = F.vsummary(lg, "logo"), F.vsummary(vf, "verify-logo")
    if sl is None or sv is None:
        return {"err": f"nėra santraukų (logo={sl is not None}, verify={sv is not None})"}
    rec = {**common_record(fault, t_fault), "exit": {"logo": rl, "verify": rv}, "fn": split_summary(sl, FN_KEYS, N),
           "out_lateness": hist_of(sl, "out_lateness"), "verify": split_summary(sv, VER_KEYS, N),
           "verify_proc": (sv.get("proc") or {}), "fatal": sl.get("fatal"), "upstream": upstream(ga),
           "fn_flows": [flow_view(sl, FN_KEYS, [n]) for n in range(N - 1)],
           "verify_flows": [flow_view(sv, VER_KEYS, [n]) for n in range(N - 1)]}
    if vp is not None:
        sp = F.vsummary(vp, "verify-logo")
        rec["post"] = {"exit": rp, **(flow_view(sp, VER_KEYS, [0]) if sp else {"err": "nėra post santraukos"})}
    return rec


# --- composite -----------------------------------------------------------------------------------------------------------

def run_composite(out, tag, impl, storage, fault):
    sub = f"fiso-comp-{tag}-{fault}"
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    xy = ["--layer-xy", "1488,64"]
    g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", N, "--duration", RUN_S + 60, *v)
    xa = F.FProc(out, f"{sub}/gfxA", "gfxgen", d, "--flows", N - 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace",
                 "--duration", RUN_S + 60, *v)
    xb = F.FProc(out, f"{sub}/gfxB", "gfxgen", d, "--flows", 1, "--flow-base", N - 1, "--layers", 1, "--layer-geom", "logo",
                 "--mode", "pace", "--duration", RUN_S + 60, *v)
    c = va = None
    rc = rv = None
    t_fault = None
    try:
        ids = [F.role_flow_id(1, n) for n in range(N)] + [F.role_flow_id(5, n * 8) for n in range(N)]
        if not F.wait_flows(d, ids):
            return {"err": "gen / gfxgen flow'ai neatsirado"}
        t0 = time.monotonic()
        c = F.FProc(out, sub, "composite", d, "--flows", N, "--layers", 1, *xy, "--opacity", "1.0", "--base", "gen",
                    "--out-format", "v210", "--deadline-ms", F.COMP_DEADLINE_MS, "--duration", RUN_S, "--warmup", WARM_S, *v)
        if not F.wait_flows(d, [F.role_flow_id(6, n) for n in range(N)], 30):
            return {"err": "composite išėjimo flow'ai neatsirado"}
        va = F.FProc(out, sub, "verify", d, "--mode", "composite", "--flows", N, "--layers", 1, "--layer-geom", "logo",
                     "--gfx-mode", "pace", "--base", "gen", "--out-format", "v210", "--every", 5, "--duration", VERIFY_S,
                     "--warmup", 1, *xy, "--opacity", "1.0", *F.vimpl(impl, storage))
        t_fault = apply_fault(fault, xb, t0)
        rv = va.wait(timeout=VERIFY_S + 60)
        rc = c.wait(timeout=RUN_S + 60)
    finally:
        stop_all([va, c, xb, xa, g])
    sc, sv = F.vsummary(c, "composite"), F.vsummary(va, "verify-composite")
    if sc is None or sv is None:
        return {"err": f"nėra santraukų (composite={sc is not None}, verify={sv is not None})"}
    ug, ux = upstream(g), upstream(xa)  # sveiki aukštupiai: gen (bazės) ir gfxgen A (sluoksniai)
    up = {"late_skips": ug["late_skips"] + ux["late_skips"], "late_commits": ug["late_commits"] + ux["late_commits"],
          "max_lateness_ms": ug["max_lateness_ms"], "late_ticks": ug["late_ticks"], "gfxgen": ux}
    return {**common_record(fault, t_fault), "exit": {"composite": rc, "verify": rv}, "fn": split_summary(sc, FN_KEYS, N),
            "out_lateness": hist_of(sc, "out_lateness"), "verify": split_summary(sv, VER_KEYS, N),
            "verify_proc": (sv.get("proc") or {}), "fatal": sc.get("fatal"), "upstream": up,
            "fn_flows": [flow_view(sc, FN_KEYS, [n]) for n in range(N - 1)],
            "verify_flows": [flow_view(sv, VER_KEYS, [n]) for n in range(N - 1)]}


# --- enc -----------------------------------------------------------------------------------------------------------------

def run_enc(out, tag, impl, storage, enc, fault):
    sub = f"fiso-enc-{tag}-{fault}"
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    sink_json = pathlib.Path(out) / sub / "sink.json"
    ga = F.FProc(out, f"{sub}/genA", "gen", d, "--flows", N - 1, "--duration", RUN_S + 60, *v)
    gb = F.FProc(out, f"{sub}/genB", "gen", d, "--flows", 1, "--flow-base", N - 1, "--duration", RUN_S + 60, *v)
    e = sink = None
    re_ = None
    t_fault = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)]):
            return {"err": "gen flow'ai neatsirado"}
        sink = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "perf/ts_sink.py"),
                                 "--port-base", str(ENC_PORT), "--flows", str(N), "--seconds", str(RUN_S + 4), "--out",
                                 str(sink_json)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ea = ["--encoder", enc, "--port-base", ENC_PORT] + (["--x264-preset", "ultrafast"] if enc == "x264" else [])
        t0 = time.monotonic()
        e = F.FProc(out, sub, "enc", d, "--flows", N, "--duration", RUN_S, "--warmup", WARM_S, *ea, *v)
        t_fault = apply_fault(fault, gb, t0)
        re_ = e.wait(timeout=RUN_S + 60)
        sink.wait(timeout=60)
    finally:
        if sink is not None and sink.poll() is None:
            sink.kill()
        stop_all([e, gb, ga])
    se = F.vsummary(e, "enc")
    if se is None:
        return {"err": "nėra enc santraukos"}
    try:
        sk = json.loads(sink_json.read_text())
    except (OSError, ValueError):
        sk = {}
    sflows = sk.get("flows") or []
    return {**common_record(fault, t_fault), "exit": {"enc": re_}, "fn": split_summary(se, ENC_KEYS, N),
            "out_lateness": hist_of(se, "frame_lateness"), "fatal": se.get("fatal"),
            "sink": {"healthy_video_pes_min": min((f.get("video_pes", 0) for f in sflows[:N - 1]), default=None),
                     "healthy_cc_errors": sum(f.get("cc_errors", 0) for f in sflows[:N - 1]),
                     "healthy_sync_errors": sum(f.get("sync_errors", 0) for f in sflows[:N - 1]),
                     "affected_video_pes": sflows[N - 1].get("video_pes") if len(sflows) >= N else None}}


# --- bendras apvalkalas --------------------------------------------------------------------------------------------------

_CONTROL = {}  # (zondas, variantas) → kontrolinis paleidimas, bendras fiso_kill ir fiso_stall


def describe(r):
    if r.get("err"):
        return r["err"]
    fh, fa = r["fn"]["healthy"], r["fn"]["affected"]
    pick = lambda d: {k: v for k, v in d.items() if v and k not in ("fields_ok", "frames_encoded")}  # noqa: E731
    lat = r["out_lateness"]
    s = (f"exit={r['exit']} sveiki: ok={fh.get('fields_ok', fh.get('frames_encoded'))} nenuliniai={pick(fh) or '-'} | "
         f"paveiktas: nenuliniai={pick(fa) or '-'} | lateness p50={lat['p50_ms']} p99={lat['p99_ms']} max={lat['max_ms']} ms")
    if "verify" in r:
        vh, va = r["verify"]["healthy"], r["verify"]["affected"]
        s += (f" | verify sveiki: samples={vh.get('samples')} missing={vh.get('missing')} invalid={vh.get('invalid')} "
              f"mismatch={vh.get('mismatch_fields')}; paveiktas: samples={va.get('samples')} present={va.get('present')} "
              f"missing={va.get('missing')} invalid={va.get('invalid')} mismatch={va.get('mismatch_fields')} "
              f"reattached={va.get('reattached')} too_late={va.get('too_late')}")
    return s


FN_LOSS = ("late_fields", "late_skips", "lapped", "too_late", "in_invalid", "read_errors", "missing_fields", "flow_invalid")


def upstream_budget(r):
    """Kiek indeksų kiekvienam sveikam flow'ui prarado / pavėlavo (> --deadline-ms) pats sveikas aukštupis."""
    u = r.get("upstream") or {}
    return u.get("late_skips", 0) + u.get("late_commits", 0)


def logo_gate(fault, ctrl, r):  # noqa: C901 — taip pat composite (V11)
    """V10 vartai (planas §9 V10) su priskyrimu: kiekvienam sveikam flow'ui funkcijos praradimai (FN_LOSS) ir verify missing +
    invalid ≤ aukštupio biudžetas (upstream_budget; be aukštupio sutrikimų — 0), mismatch == 0 visada; out_lateness p99 po gedimo
    ≤ p99 be gedimo + Wr + 1 ms; paveiktam — missing_fields > 0, fiso_stall: stalls ≥ 1 ir po SIGCONT POST_S s be praradimų.
    Grąžina neatitikimų sąrašą (tuščias — PASS)."""
    bad = []
    fa = r["fn"]["affected"]
    budget = upstream_budget(r)
    for n, f in enumerate(r.get("fn_flows") or []):
        lost = sum(f.get(k, 0) for k in FN_LOSS)
        if lost > budget:
            bad.append(f"sveiko #{n} funkcijos praradimai {lost} > aukštupio {budget} ({ {k: f[k] for k in FN_LOSS if f.get(k)} })")
    for n, v in enumerate(r.get("verify_flows") or []):
        lost = v.get("missing", 0) + v.get("invalid", 0)
        if lost > budget:
            bad.append(f"verify sveiko #{n} missing + invalid {lost} > aukštupio {budget}")
        if v.get("mismatch_fields", 0):
            bad.append(f"verify sveiko #{n} mismatch={v['mismatch_fields']}")
    p99a, p99b = r["out_lateness"].get("p99_ms"), ctrl["out_lateness"].get("p99_ms")
    if p99a is None or p99b is None or p99a > p99b + WR_MS + 1.0:
        bad.append(f"out_lateness p99 {p99b} → {p99a} ms > +{WR_MS + 1.0} ms")
    if not fa.get("missing_fields", 0):
        bad.append("paveikto missing_fields == 0")
    if fault == "stall":
        if fa.get("stalls", 0) < 1:
            bad.append(f"paveikto stalls={fa.get('stalls', 0)} < 1")
        post = r.get("post") or {}
        if post.get("err") or post.get("exit") != 0 or post.get("samples", 0) == 0 or any(
                post.get(k, 0) for k in ("missing", "invalid", "mismatch_fields")):
            bad.append(f"po SIGCONT {POST_S} s: {post}")
    return bad


def probe(out, name, runner, faults):
    """Kiekvienam variantui: kontrolinis (be gedimo) + su gedimu. logo (run_logo) — vartai (logo_gate), kiti — INFO."""
    res = []
    for tag, impl, storage, enc in variants():
        tid = f"{name}[{tag}]"
        args = (out, tag, impl, storage) + ((enc,) if runner is run_enc else ())
        key = (runner.__name__, tag)
        if key not in _CONTROL:
            _CONTROL[key] = runner(*args, "none")
        ctrl = _CONTROL[key]
        for fault in faults:
            r = runner(*args, fault)
            if ctrl.get("err") or r.get("err"):
                res.append(F.result(tid, "FAIL", why=f"kontrolinis: {describe(ctrl)}; su gedimu ({fault}): {describe(r)}"))
                continue
            why = f"PRIEŠ (be gedimo): {describe(ctrl)} || PO ({fault} ties {r['fault_at_s']} s): {describe(r)}"
            if "upstream" in r:  # sveiko aukštupio sutrikimai — atskira INFO eilutė su skaičiais
                res.append(F.result(f"{tid}:upstream", "INFO", why=f"sveikas aukštupis (gen A): be gedimo {ctrl.get('upstream')} | "
                                                                  f"su gedimu {r['upstream']}"))
            if runner in (run_logo, run_composite):
                bad = logo_gate(fault, ctrl, r)
                if r.get("post"):
                    why += f" | po SIGCONT: {r['post']}"
                res.append(F.result(tid, "FAIL" if bad else "PASS", why=(f"{'; '.join(bad)} || " if bad else "") + why,
                                    before=ctrl, after=r))
                continue
            res.append(F.result(tid, "INFO", why=why, before=ctrl, after=r))
    return res


def fiso_kill(out):
    return probe(out, "fiso_kill", run_logo, ["kill"])


def fiso_stall(out):
    return probe(out, "fiso_stall", run_logo, ["stall"])


def fiso_composite(out):
    return probe(out, "fiso_composite", run_composite, ["kill"])


def fiso_enc(out):
    return probe(out, "fiso_enc", run_enc, ["kill"])


TESTS = {"fiso_kill": fiso_kill, "fiso_stall": fiso_stall, "fiso_composite": fiso_composite, "fiso_enc": fiso_enc}
