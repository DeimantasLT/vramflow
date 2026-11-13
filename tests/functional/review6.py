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
import signal
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
            # --every 3 (ne 2): imtys ir lyginiuose, ir NElyginiuose (apatiniuose) laukuose — sluoksnio rinkinys turi sutapti abiem
            # kadro laukams (peržiūra D3; su --every 2 flow'ui 0 buvo tikrinami tik viršutiniai laukai)
            "--out-format", ofmt, "--every", 3, "--duration", dur, "--warmup", 1, "--layer-xy", "1488,64", "--opacity", "0.7",
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


# --- Punktas 2: INVALID grain'ai neneša pasenusio vaizdo; garso spragos — tyla ------------------------------------------

def flogo_invalid_payload(out):
    """gen | logo (--on-loss invalid) | verify --mode logo --every 1: gen nužudomas 3 s ir paleidžiamas iš naujo. Skaitytojas,
    kuris INVALID žymės nepaiso (verify tikrina ir INVALID grain'o turinį): visi INVALID grain'ai — juodi (invalid_not_black ==
    0, invalid_black > 0), anksčiau — pasenęs ring'o kadras (logo vaizdas prieš ~200 ms). C / H / G."""
    res = []
    for tag, impl, storage in F.comp_variants():
        tid, sub = f"flogo_invalid_payload[{tag}]", f"flip-{tag}"
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        g = F.FProc(out, f"{sub}/a", "gen", d, "--flows", 1, "--duration", 60, *v)
        g2 = lg = vf = None
        rv = rl = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
                res.append(F.result(tid, "FAIL", why="gen flow'as neatsirado"))
                continue
            lg = F.FProc(out, sub, "logo", d, "--flows", 1, "--duration", 18, "--warmup", 1, *v)
            if not F.wait_flows(d, [F.role_flow_id(2, 0)], 20):
                res.append(F.result(tid, "FAIL", why="logo išėjimas neatsirado"))
                continue
            vf = F.FProc(out, f"{sub}/verify", "verify", d, "--mode", "logo", "--flows", 1, "--every", 1, "--duration", 12,
                         "--warmup", 1, *F.vimpl(impl, storage))
            time.sleep(4)
            g.kill9()
            time.sleep(3)
            g2 = F.FProc(out, f"{sub}/b", "gen", d, "--flows", 1, "--duration", 60, *v)
            rv = vf.wait(timeout=60)
            rl = lg.wait(timeout=60)
        finally:
            for pr in (vf, lg, g2, g):
                if pr is not None and pr.p.poll() is None:
                    pr.term()
        sv = F.vsummary(vf, "verify-logo") if vf else None
        f0 = ((sv or {}).get("flows") or [{}])[0]
        fl = F.flow_ctr(F.vsummary(lg, "logo") if lg else None, 0)
        bad = []
        if f0.get("invalid", 0) < 2 * RATE:
            bad.append(f"verify invalid={f0.get('invalid', 0)} < {2 * RATE} (gedimo nebuvo?)")
        if f0.get("invalid_not_black", 0):
            bad.append(f"INVALID grain'ų ne juodu turiniu: {f0.get('invalid_not_black')} (pirmas indeksas {f0.get('invalid_not_black_first')})")
        if f0.get("invalid_black", 0) < 2 * RATE:
            bad.append(f"invalid_black={f0.get('invalid_black', 0)} < {2 * RATE}")
        if f0.get("mismatch_fields", 0):
            bad.append(f"mismatch={f0.get('mismatch_fields')}")
        res.append(F.result(tid, "FAIL" if bad else "PASS",
                            why=("; ".join(bad) or "ok") + f" | verify exit={rv} invalid={f0.get('invalid')} black={f0.get('invalid_black')} "
                                f"not_black={f0.get('invalid_not_black', 0)} samples={f0.get('samples')} | logo exit={rl} "
                                f"gap_invalid={fl.get('gap_invalid')} missing_fields={fl.get('missing_fields')}", verify=f0, logo=fl))
    return res


def fgen_audio_gap(out):
    """gen --audio 1 sustabdomas (SIGSTOP 100 ms) 4 kartus: variklio praleisti indeksai — garso blokai commit'inami tyla
    (gap_silence == praleistų indeksų, nes kiekvienam praleistam — tylos blokas), skaitytojas (verify --mode audio, žymių nėra)
    gedimo metu skaito tik tylą: silence > 0, audio_bad == 0 (anksčiau — pasenęs ring'o garsas, silence == 0). C."""
    tid, sub = "fgen_audio_gap[C]", "fgag-C"
    d = F.fresh_domain(sub)
    v = F.vcommon("cpu", "host")
    g = F.FProc(out, sub, "gen", d, "--flows", 1, "--audio", 1, "--duration", 16, *v)
    va = None
    rv = rg = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(3, 0)]):
            return [F.result(tid, "FAIL", why="gen garso flow'as neatsirado")]
        va = F.FProc(out, f"{sub}/verify", "verify", d, "--mode", "audio", "--flows", 1, "--lag-fields", 8, "--duration", 11,
                     "--warmup", 1)
        time.sleep(3)
        for _ in range(4):
            g.p.send_signal(signal.SIGSTOP)
            time.sleep(0.1)
            g.p.send_signal(signal.SIGCONT)
            time.sleep(2)
        rv = va.wait(timeout=60)
        rg = g.wait(timeout=60)
    finally:
        for pr in (va, g):
            if pr is not None and pr.p.poll() is None:
                pr.term()
    sv = F.vsummary(va, "verify-audio") if va else None
    f0 = ((sv or {}).get("flows") or [{}])[0]
    sg = F.vsummary(g, "gen") or {}
    fg = F.flow_ctr(sg, 0)
    pg = sg.get("proc") or {}
    # Visi variklio praleisti indeksai (ir įšilimo / laikrodžio — garso blokas be turinio vis tiek tyla)
    skips = pg.get("late_skips", 0) + pg.get("warmup_skips", 0) + pg.get("clock_skips", 0)
    bad = []
    if skips < 4:
        bad.append(f"gen praleistų indeksų {skips} < 4 (SIGSTOP nesukėlė praleidimų?)")
    if fg.get("gap_silence", 0) != skips:
        bad.append(f"gen gap_silence={fg.get('gap_silence', 0)} ≠ praleistų indeksų {skips}")
    if f0.get("audio_bad", 0):
        bad.append(f"verify audio_bad={f0.get('audio_bad')} (pirmas indeksas {f0.get('audio_bad_first')})")
    if f0.get("silence", 0) < 1 or f0.get("silence", 0) > fg.get("gap_silence", 0):
        bad.append(f"verify silence={f0.get('silence', 0)} (laukta 1…{fg.get('gap_silence', 0)})")
    if f0.get("audio_ok", 0) < 400:
        bad.append(f"verify audio_ok={f0.get('audio_ok', 0)} < 400")
    return [F.result(tid, "FAIL" if bad else "PASS",
                     why=("; ".join(bad) or "ok") + f" | gen exit={rg} late_skips={skips} gap_silence={fg.get('gap_silence')} | verify "
                         f"exit={rv} ok={f0.get('audio_ok')} silence={f0.get('silence')} bad={f0.get('audio_bad', 0)} "
                         f"missing={f0.get('missing', 0)}", verify=f0, gen=fg)]


TESTS["flogo_invalid_payload"] = flogo_invalid_payload
TESTS["fgen_audio_gap"] = fgen_audio_gap


# --- Punktas 3: aukštupio spraga — ne šios funkcijos praradimas -----------------------------------------------------------

LOSS3 = ("late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips", "missing_fields", "flow_invalid",
         "output_errors")


def chain3(out, sub, impl, storage, kill_source=False):
    """gen → logo → composite (bazė — logo išėjimas, --in-role 2; 1 sluoksnis gfxgen). Paleidžiama ATVIRKŠTINE tvarka:
    composite, logo, gfxgen, gen (logo --start immediate: iki gen — INVALID gap_startup; composite juos skaito). Griežtas
    exit (--duration > 0). kill_source — gen nužudomas viduryje (2 s) ir paleidžiamas iš naujo."""
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    DUR = 22
    c = F.FProc(out, sub, "composite", d, "--flows", 1, "--layers", 1, "--layer-xy", "1488,64", "--base", "gen", "--in-role", 2,
                "--out-format", "v210", "--duration", DUR, "--warmup", 1, *v)
    time.sleep(1)
    lg = F.FProc(out, sub, "logo", d, "--flows", 1, "--duration", DUR + 3, "--warmup", 1, *v)
    time.sleep(2)
    x = F.FProc(out, sub, "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace", "--duration", 60, *v)
    time.sleep(1)
    g = F.FProc(out, f"{sub}/a", "gen", d, "--flows", 1, "--duration", 60, *v)
    g2 = None
    rc = rl = None
    try:
        if kill_source:
            time.sleep(8)
            g.kill9()
            time.sleep(2)
            g2 = F.FProc(out, f"{sub}/b", "gen", d, "--flows", 1, "--duration", 60, *v)
        rc = c.wait(timeout=DUR + 60)
        rl = lg.wait(timeout=DUR + 60)
    finally:
        for pr in (c, lg, x, g2, g):
            if pr is not None and pr.p.poll() is None:
                pr.term()
    return {"rc": rc, "rl": rl, "c": F.flow_ctr(F.vsummary(c, "composite"), 0), "l": F.flow_ctr(F.vsummary(lg, "logo"), 0),
            "cs": F.vsummary(c, "composite") or {}}


def fchain_reverse(out):
    """3 grandinės stadijos paleistos atvirkštine tvarka, švarus bėgimas, griežtas exit: visos baigiasi exit 0. composite
    skaito logo paleidimo INVALID grain'us — gap_upstream > 0, ne praradimas (anksčiau — in_invalid ir exit 4); logo —
    gap_startup (ne praradimas). C / G."""
    res = []
    for tag, impl, storage in [v for v in F.comp_variants() if v[0] in ("C", "G")]:
        tid = f"fchain_reverse[{tag}]"
        r = chain3(out, f"fchrev-{tag}", impl, storage)
        c, lo = r["c"], r["l"]
        bad = []
        if r["rc"] != 0 or r["rl"] != 0:
            bad.append(f"exit composite={r['rc']} logo={r['rl']}")
        lc = {k: c[k] for k in LOSS3 if c.get(k)}
        ll = {k: lo[k] for k in LOSS3 if lo.get(k)}
        if lc or ll:
            bad.append(f"praradimai composite={lc} logo={ll}")
        if not c.get("gap_upstream", 0):
            bad.append("composite gap_upstream == 0 (logo paleidimo INVALID grain'ų nematė?)")
        if not lo.get("gap_startup", 0):
            bad.append("logo gap_startup == 0")
        if r["cs"].get("any_loss"):
            bad.append("composite any_loss")
        res.append(F.result(tid, "FAIL" if bad else "PASS",
                            why=("; ".join(bad) or "ok") + f" | composite exit={r['rc']} gap_upstream={c.get('gap_upstream')} "
                                f"fields_ok={c.get('fields_ok')} | logo exit={r['rl']} gap_startup={lo.get('gap_startup')}",
                            composite=c, logo=lo))
    return res


def fchain_source_outage(out):
    """Ta pati grandinė, gen (šaltinis) nužudomas 2 s: praradimą skaičiuoja tik pirma stadija po gedimo (logo:
    missing_fields > 0, griežtas exit 4); composite — tik gap_upstream (> 0), 0 praradimų, exit 0. C / G."""
    res = []
    for tag, impl, storage in [v for v in F.comp_variants() if v[0] in ("C", "G")]:
        tid = f"fchain_source_outage[{tag}]"
        r = chain3(out, f"fchout-{tag}", impl, storage, kill_source=True)
        c, lo = r["c"], r["l"]
        bad = []
        lc = {k: c[k] for k in LOSS3 if c.get(k)}
        if r["rc"] != 0 or lc:
            bad.append(f"composite exit={r['rc']} praradimai={lc} (laukta 0, exit 0)")
        if c.get("gap_upstream", 0) < 50:
            bad.append(f"composite gap_upstream={c.get('gap_upstream', 0)} < 50")
        if r["rl"] != 4 or lo.get("missing_fields", 0) < 50:
            bad.append(f"logo exit={r['rl']} missing_fields={lo.get('missing_fields', 0)} (laukta 4 ir ≥ 50)")
        res.append(F.result(tid, "FAIL" if bad else "PASS",
                            why=("; ".join(bad) or "ok") + f" | composite exit={r['rc']} gap_upstream={c.get('gap_upstream')} | "
                                f"logo exit={r['rl']} missing_fields={lo.get('missing_fields')} gap_invalid={lo.get('gap_invalid')}",
                            composite=c, logo=lo))
    return res


TESTS["fchain_reverse"] = fchain_reverse
TESTS["fchain_source_outage"] = fchain_source_outage
