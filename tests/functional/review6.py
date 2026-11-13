#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Šeštos peržiūros testai (2026-11-13, f61e5b6..f578fc3; kviečiami iš functional.py --only <vardas>).

fcomp_layer_kill   — composite (bazė gen, 1 sluoksnis): sluoksnio rašytojas nužudomas viduryje ir paleidžiamas iš naujo.
                     Išėjime 0 spragų (gap_*, missing_fields, late_fields == 0, exit 0 griežtame režime); verify --mode composite
                     su --layer-absent (nužudymo → paleidimo langas: etalonas BE sluoksnio) ir --layer-either (perėjimai):
                     0 neatitikimų, 0 missing / invalid, tikrinta ir be sluoksnio (samples_absent > 0), ir su juo po grįžimo
                     (sluoksnis vėl naudojamas — griežtas „yra“ langas pabaigoje). C / H / G.
fcomp_layer_idle   — sluoksnio rašytojas sukurtas ir ritmas eina, bet niekada nesubmit'ina (gfxgen --test-no-submit: tik INVALID
                     grain'ai): tas pats — 0 spragų, kiekvienas tikrintas grain'as == etalonas be sluoksnio. C / H / G.
fcomp_layer_none   — `--base none` (v210a), sluoksnis nužudomas: išėjimas — skaidri juoda (key 0) pagal etaloną be sluoksnio, ne
                     spraga. C.
fcomp_base_kill    — bazės (gen) rašytojas nužudomas: išėjimas — spraga pagal politiką (--on-loss invalid: gap_invalid > 0,
                     missing_fields > 0), composite gyvas. C / H / G.
"""
import json
import time

import functional as F  # noqa: E402

TESTS = {}
RATE = 50  # 1080i50: 50 grain'ų / s


def tai_index(t=None):
    """MXL indeksas TAI laikui (CLOCK_TAI; kaip mxlGetCurrentIndex 50/1)."""
    t = time.clock_gettime(time.CLOCK_TAI) if t is None else t
    return int(t * RATE)


def sleep_until_tai(t):
    while True:
        d = t - time.clock_gettime(time.CLOCK_TAI)
        if d <= 0:
            return
        time.sleep(min(d, 0.05))


def comp_args(base, ofmt, dur, v):
    return ["--flows", 1, "--layers", 1, "--layer-xy", "1488,64", "--opacity", "0.7", "--base", base, "--out-format", ofmt,
            "--deadline-ms", F.COMP_DEADLINE_MS, "--duration", dur, "--warmup", 1, *v]


def verify_args(base, ofmt, dur, v, windows):
    w = []
    for kind, k, a, b in windows:
        w += [f"--layer-{kind}", f"{k}:{a}-{b}"]
    return ["--mode", "composite", "--flows", 1, "--layers", 1, "--layer-geom", "logo", "--gfx-mode", "pace", "--base", base,
            "--out-format", ofmt, "--every", 2, "--duration", dur, "--warmup", 1, "--layer-xy", "1488,64", "--opacity", "0.7",
            *w, *v]


def vsum(pr):
    f = pr.out / "summary-verify-composite.json"
    return json.loads(f.read_text()) if f.exists() else None


GAP_KEYS = ("gap_invalid", "gap_frozen", "gap_black", "missing_fields", "late_fields", "lapped", "too_late", "in_invalid",
            "read_errors", "flow_invalid", "late_skips", "output_errors")


def judge(tid, rc, s, vs, want_absent_min, need_present_samples=True):
    f0 = F.flow_ctr(s, 0)
    pv = (vs or {}).get("proc", {})
    fv = ((vs or {}).get("flows") or [{}])[0]
    gaps = {k: f0.get(k, 0) for k in GAP_KEYS if f0.get(k, 0)}
    bad = []
    if rc != 0:
        bad.append(f"composite exit {rc}")
    if s is None or vs is None:
        bad.append("nėra santraukos")
    if gaps:
        bad.append(f"spragos {gaps}")
    if f0.get("layer_absent", 0) < want_absent_min:
        bad.append(f"layer_absent {f0.get('layer_absent', 0)} < {want_absent_min}")
    if pv.get("verification_errors", 1) != 0:
        bad.append(f"verification_errors {pv.get('verification_errors')}")
    if fv.get("missing", 0) or fv.get("invalid", 0) or fv.get("too_late", 0):
        bad.append(f"verify missing={fv.get('missing', 0)} invalid={fv.get('invalid', 0)} too_late={fv.get('too_late', 0)}")
    if fv.get("samples_absent", 0) < 20:
        bad.append(f"samples_absent {fv.get('samples_absent', 0)} < 20")
    if need_present_samples and fv.get("samples", 0) - fv.get("samples_absent", 0) - fv.get("samples_either_absent", 0) < 20:
        bad.append("per mažai tikrinta su sluoksniu")
    return F.result(tid, "FAIL" if bad else "PASS",
                    why=("; ".join(bad) if bad else "ok") + f" | layer_absent={f0.get('layer_absent')} fields_ok={f0.get('fields_ok')} "
                    f"verify samples={fv.get('samples')} absent={fv.get('samples_absent')} either={fv.get('samples_either_absent')} "
                    f"mismatch={fv.get('mismatch_fields', 0)}", composite=f0, verify=fv)


def layer_kill_run(out, tid, sub, impl, storage, base, ofmt, restart=True):
    """Sluoksnis nužudomas ties T0 + 6 s (ir paleidžiamas iš naujo ties T0 + 11 s); verify langai — iš anksto suplanuoti."""
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    gx = ["--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace", "--duration", 80, *v]
    g = F.FProc(out, sub, "gen", d, "--flows", 1, "--duration", 80, *v) if base == "gen" else None
    x = F.FProc(out, f"{sub}/a", "gfxgen", d, *gx)
    x2 = c = va = None
    rc = None
    try:
        ids = [F.role_flow_id(5, 0)] + ([F.role_flow_id(1, 0)] if g else [])
        if not F.wait_flows(d, ids):
            return F.result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado")
        c = F.FProc(out, sub, "composite", d, *comp_args(base, ofmt, 26, v))
        if not F.wait_flows(d, [F.role_flow_id(6, 0)], 20):
            return F.result(tid, "FAIL", why="composite išėjimas neatsirado")
        t0 = time.clock_gettime(time.CLOCK_TAI)
        t_kill, t_back = t0 + 6, t0 + 11
        i_kill, i_back = tai_index(t_kill), tai_index(t_back)
        # Langai: [i_kill − 5, i_kill + 10] — perėjimas; (i_kill + 10, i_back) — sluoksnio nėra; po paleidimo 5 s — perėjimas
        # (gfxgen startas + prisijungimas ≤ 100 ms + kadro riba); toliau — sluoksnis yra (griežtai).
        win = [("either", 0, i_kill - 5, i_kill + 10)]
        win += [("absent", 0, i_kill + 11, (i_back - 1) if restart else i_kill + 10_000)]
        if restart:
            win += [("either", 0, i_back, i_back + 5 * RATE)]
        va = F.FProc(out, f"{sub}/verify", "verify", d, *verify_args(base, ofmt, 18, v, win))
        sleep_until_tai(t_kill)
        x.kill9()
        if restart:
            sleep_until_tai(t_back)
            x2 = F.FProc(out, f"{sub}/b", "gfxgen", d, *gx)
        va.wait(timeout=60)
        rc = c.wait(timeout=60)
    finally:
        for pr in (va, c, x2, x, g):
            if pr is not None and pr.p.poll() is None:
                pr.term()
    return judge(tid, rc, c.summary() if c else None, vsum(va) if va else None, want_absent_min=4 * RATE,
                 need_present_samples=restart or base == "gen")


def fcomp_layer_kill(out):
    return [layer_kill_run(out, f"fcomp_layer_kill[{tag}]", f"fclk-{tag}", impl, storage, "gen", "v210")
            for tag, impl, storage in F.comp_variants()]


def fcomp_layer_none(out):
    return [layer_kill_run(out, "fcomp_layer_none[C]", "fcln-C", "cpu", "host", "none", "v210a", restart=False)]


def fcomp_layer_idle(out):
    res = []
    for tag, impl, storage in F.comp_variants():
        tid, sub = f"fcomp_layer_idle[{tag}]", f"fcli-{tag}"
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        g = F.FProc(out, sub, "gen", d, "--flows", 1, "--duration", 60, *v)
        x = F.FProc(out, sub, "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace", "--test-no-submit",
                    "--duration", 60, *v)
        c = va = None
        rc = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, 0), F.role_flow_id(5, 0)]):
                res.append(F.result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                continue
            c = F.FProc(out, sub, "composite", d, *comp_args("gen", "v210", 14, v))
            if not F.wait_flows(d, [F.role_flow_id(6, 0)], 20):
                res.append(F.result(tid, "FAIL", why="composite išėjimas neatsirado"))
                continue
            va = F.FProc(out, f"{sub}/verify", "verify", d, *verify_args("gen", "v210", 8, v, [("absent", 0, 0, 1 << 62)]))
            va.wait(timeout=60)
            rc = c.wait(timeout=60)
        finally:
            for pr in (va, c, x, g):
                if pr is not None and pr.p.poll() is None:
                    pr.term()
        s = c.summary() if c else None
        res.append(judge(tid, rc, s, vsum(va) if va else None, want_absent_min=10 * RATE, need_present_samples=False))
    return res


def fcomp_base_kill(out):
    res = []
    for tag, impl, storage in F.comp_variants():
        tid, sub = f"fcomp_base_kill[{tag}]", f"fcbk-{tag}"
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        g = F.FProc(out, sub, "gen", d, "--flows", 1, "--duration", 60, *v)
        x = F.FProc(out, sub, "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace", "--duration", 60, *v)
        c = None
        rc = None
        alive = False
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, 0), F.role_flow_id(5, 0)]):
                res.append(F.result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                continue
            c = F.FProc(out, sub, "composite", d, *comp_args("gen", "v210", 12, v))
            F.wait_flows(d, [F.role_flow_id(6, 0)], 20)
            time.sleep(4)
            g.kill9()
            time.sleep(3)
            alive = c.p.poll() is None
            rc = c.wait(timeout=60)
        finally:
            for pr in (c, x, g):
                if pr is not None and pr.p.poll() is None:
                    pr.term()
        f0 = F.flow_ctr(c.summary() if c else None, 0)
        # griežtas režimas (--duration > 0): bazės praradimas — exit 4; spragos commit'inamos pagal --on-loss invalid
        ok = alive and rc == 4 and f0.get("gap_invalid", 0) >= 2 * RATE and f0.get("missing_fields", 0) >= 2 * RATE
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"gyvas={alive} exit={rc} gap_invalid={f0.get('gap_invalid')} missing_fields={f0.get('missing_fields')} "
                                f"layer_absent={f0.get('layer_absent')}", composite=f0))
    return res


TESTS["fcomp_layer_kill"] = fcomp_layer_kill
TESTS["fcomp_layer_idle"] = fcomp_layer_idle
TESTS["fcomp_layer_none"] = fcomp_layer_none
TESTS["fcomp_base_kill"] = fcomp_base_kill
