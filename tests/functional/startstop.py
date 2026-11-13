#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Paleidimo ir sustabdymo funkciniai testai (planas §P6). Kviečiami iš functional.py (--only fstart_order,fstart_restart,…).

fstart_order    — logo ir verify paleisti 20 s (> buvusios 15 s ribos) PRIEŠ gen: abu laukia (eilutė „laukiu flow“), po gen
                  starto dirba normaliai: logo be praradimų, verify 0 teisingumo klaidų, abu exit 0.
                  V18 (--start immediate, numatyta): logo išėjimo flow'ai atsiranda per ≤ 1 s nuo logo starto (prieš gen); iki gen
                  — gap_startup > 0 (spragos pagal --on-loss), praradimų 0 (missing_fields == 0, gap_invalid == 0). Logo
                  --duration skaičiuojama nuo starto, todėl PRE + D + 6. fstart_order[wait-all:C] — ankstesnė elgsena: iki
                  gen išėjimo flow'ų nėra, gap_startup == 0.
fstart_restart  — gen nužudomas (kill -9) ir po 5 s paleidžiamas vėl: logo gyvas, reattached == 1, spraga suskaičiuota ir
                  KIEKVIENAS jos indeksas commit'intas INVALID (V17 K3: gap_invalid == missing_fields > 0; anksčiau — tik
                  missing_fields > 0, išėjime indeksų nebuvo), verify missing == 0, kitų praradimų 0, išėjimas vėl juda;
                  verify 0 teisingumo klaidų.
fstart_wait     — `--input-wait-ms 1500` be gamintojo → exit 4 per ≤ 2 s (V18; anksčiau < 10 s), aiškus tekstas (riba, parinktis,
                  flow ID) — ir su --start immediate (numatyta), kai išėjimai jau sukurti;
                  `--input-wait-ms 0` (neribotai) po 17 s vis dar laukia, SIGTERM → tvarkingai exit 0.
fstart_owner    — antras gen tame pačiame domene su tuo pačiu --flow-base → exit != 0, tekste pirmojo run_id; pirmasis
                  nepaliestas (flow head juda).
fstart_domain   — domain.py: make_domain unikalūs; remove_domain su gyvu gen → False ir flow failai vietoje; po gen
                  pabaigos → True (domenas ir nuomos katalogas pašalinti).
fstart_parallel — du capacity.py --func logo --n 2 lygiagrečiai (anksčiau — tas pats /dev/shm/mxlf-logo-2): abu PASS.
fstart_longpath — domeno kelias > 48 simbolių su cuda-linear (reader'io `payload.sock` kelias netilptų į 107 simbolius, o
                  vartotojas lauktų be galo) → gen iškart exit 1 su aiškiu tekstu; make_domain ilgą žymą sutrumpina.
fstart_logo_restart — pats logo (rašanti funkcija viduryje grandinės) nužudomas kill -9 ir paleidžiamas vėl: startuoja (savo
                  pasenusius išėjimus pašalina; be to cuda-linear restartas sukdavosi ant MXL_ERR_CONFLICT), išėjimas vėl
                  juda, o verify prie naujo flow prisijungia pats (reattached >= 1) ir neranda teisingumo klaidų.
Variantai: FSTART_VARIANTS=C,G (numatyta).

Šie testai tikrina elgseną iš išorės (procesai, exit kodai, summary skaitikliai `reattached`, `missing_fields`, `fields_ok`,
`run_id`, stderr tekstas) — nepriklauso nuo to, kuris ciklas ją įgyvendina, ir turi likti žali po tick variklio suvienodinimo.
"""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf"))
import domain  # noqa: E402  tests/perf/domain.py

import functional as F  # noqa: E402

VARIANTS = {"C": ("cpu", "host"), "H": ("gpu", "host"), "G": ("gpu", "cuda-linear")}
LOSS = ("late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips")


def variants():
    want = [x for x in os.environ.get("FSTART_VARIANTS", "C,G").split(",") if x]
    return [(t, *VARIANTS[t]) for t in want]


def err_text(pr):
    return pr.err_path.read_text(errors="replace") if pr is not None and pr.err_path.exists() else ""


def logo_losses(s):
    return {k: sum(f.get(k, 0) for f in (s or {}).get("flows") or []) for k in LOSS}


def fstart_order(out):
    res = []
    N, D, PRE = 2, 10, 20
    runs = [(tag, impl, storage, "immediate") for tag, impl, storage in variants()]
    runs += [(tag, impl, storage, "wait-all") for tag, impl, storage in variants() if tag == "C"]
    for tag, impl, storage, start in runs:
        imm = start == "immediate"
        tid = f"fstart_order[{tag}]" if imm else f"fstart_order[wait-all:{tag}]"
        sub = f"fstart-order-{tag}" + ("" if imm else "-waitall")
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        lg = vf = g = None
        alive = False
        out_ids = [F.role_flow_id(2, n) for n in range(N)]
        t_out = early = None
        try:
            t0 = time.monotonic()
            # immediate: --duration skaičiuojama nuo starto (variklis tiksi iškart) — PRE + D + 6; wait-all — nuo įėjimų
            lg = F.FProc(out, sub, "logo", d, "--flows", N, "--duration", (PRE + D + 6) if imm else D + 6, "--warmup", 2,
                         "--start", start, *v)
            if F.wait_flows(d, out_ids, 3.0):
                t_out = round(time.monotonic() - t0, 2)
            vf = F.FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 5, "--duration", D, "--warmup", 2,
                         *F.vimpl(impl, storage))
            time.sleep(max(0.0, PRE - (time.monotonic() - t0)))
            alive = lg.p.poll() is None and vf.p.poll() is None
            early = all((pathlib.Path(d) / f"{i}.mxl-flow" / "flow_def.json").exists() for i in out_ids)  # prieš gen
            g = F.FProc(out, sub, "gen", d, "--flows", N, "--duration", D + PRE + 30, *v)
            rv = vf.wait(timeout=D + 60)
            rl = lg.wait(timeout=PRE + D + 60)
        finally:
            for pr in (vf, lg, g):
                if pr is not None:
                    pr.term()
        sl, sv = F.vsummary(lg, "logo"), F.vsummary(vf, "verify-logo")
        fl = (sl or {}).get("flows") or []
        lost = sum(logo_losses(sl).values()) + sum(f.get("missing_fields", 0) + f.get("gap_invalid", 0) for f in fl)
        okf = [f.get("fields_ok", 0) for f in fl]
        gs = [f.get("gap_startup", 0) for f in fl]
        pv = (sv or {}).get("proc") or {}
        waited = "laukiu flow" in err_text(lg)
        common_ok = (alive and waited and rl == 0 and rv == 0 and lost == 0 and len(okf) == N and min(okf) >= 0.9 * D * 50
                     and pv.get("verification_errors", 1) == 0 and pv.get("samples_total", 0) > 0)
        if imm:  # V18: išėjimai per ≤ 1 s (prieš gen), paleidimo spragos suskaičiuotos (≈ PRE · 50), ne praradimas
            mode_ok = t_out is not None and t_out <= 1.0 and early and len(gs) == N and min(gs) >= (PRE - 2) * 50
        else:  # wait-all: iki gen išėjimų nėra
            mode_ok = t_out is None and not early and all(x == 0 for x in gs)
        res.append(F.result(tid, "PASS" if common_ok and mode_ok else "FAIL",
                            why=f"--start {start}: išėjimo flow'ai po {t_out} s (prieš gen: {early}) gap_startup={gs} | gyvi po {PRE} s={alive} "
                                f"'laukiu flow'={waited} logo exit={rl} praradimai={lost} fields_ok={okf} "
                                f"verify exit={rv} verification_errors={pv.get('verification_errors')} samples={pv.get('samples_total')}",
                            logo=sl, verify=sv))
    return res


def fstart_restart(out):
    res = []
    N, D = 1, 24
    for tag, impl, storage in variants():
        tid = f"fstart_restart[{tag}]"
        sub = f"fstart-restart-{tag}"
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        gen_args = ["--flows", N, "--duration", 80, *v]
        g = F.FProc(out, f"{sub}/a", "gen", d, *gen_args)
        g2 = lg = vf = None
        alive = False
        h1 = h2 = None
        rl = rv = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)]):
                res.append(F.result(tid, "FAIL", why="gen flow'ai neatsirado"))
                continue
            lg = F.FProc(out, sub, "logo", d, "--flows", N, "--duration", D, "--warmup", 1, *v)
            F.wait_flows(d, [F.role_flow_id(2, n) for n in range(N)], 20)
            vf = F.FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 5, "--duration", D - 4, "--warmup", 1,
                         *F.vimpl(impl, storage))
            time.sleep(4)
            g.kill9()
            time.sleep(5)
            alive = lg.p.poll() is None and vf.p.poll() is None
            g2 = F.FProc(out, f"{sub}/b", "gen", d, *gen_args)
            time.sleep(4)
            h1 = F.flow_head(d, F.role_flow_id(2, 0))
            time.sleep(2)
            h2 = F.flow_head(d, F.role_flow_id(2, 0))
            rl = lg.wait(timeout=D + 60)
            rv = vf.wait(timeout=D + 60)
        finally:
            for pr in (vf, lg, g2, g):
                if pr is not None:
                    pr.term()
        sl, sv = F.vsummary(lg, "logo"), F.vsummary(vf, "verify-logo")
        fl = (sl or {}).get("flows") or [{}]
        reatt = [f.get("reattached", 0) for f in fl]
        missing = [f.get("missing_fields", 0) for f in fl]
        gap_inv = [f.get("gap_invalid", 0) for f in fl]
        fvs = (sv or {}).get("flows") or [{}]
        v_missing = sum(f.get("missing", 0) for f in fvs)
        lost = logo_losses(sl)
        pv = (sv or {}).get("proc") or {}
        advancing = h1 is not None and h2 is not None and h2 - h1 >= 80  # 2 s = 100 laukų
        ok = (alive and rl in (0, 4) and sl is not None and all(r == 1 for r in reatt) and all(m > 0 for m in missing)
              and gap_inv == missing and v_missing == 0 and sum(lost.values()) == 0 and advancing and pv.get("verification_errors", 1) == 0 and pv.get("samples_total", 0) > 0)
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"gyvi per outage={alive} logo exit={rl} reattached={reatt} missing_fields={missing} gap_invalid={gap_inv} "
                                f"verify missing={v_missing} kiti={lost} "
                                f"head po paleidimo {h1}->{h2} verify exit={rv} verification_errors={pv.get('verification_errors')} "
                                f"capacity_errors={pv.get('capacity_errors')} samples={pv.get('samples_total')}",
                            logo=sl, verify=sv))
    return res


def fstart_wait(out):
    res = []
    d = F.fresh_domain("fstart-wait")
    v = F.vcommon("cpu", "host")
    t0 = time.monotonic()
    p = F.FProc(out, "fstart-wait/finite", "logo", d, "--flows", 1, "--duration", 5, "--input-wait-ms", 1500, *v)
    rc = p.wait(timeout=60)
    dt = time.monotonic() - t0
    fatal = (F.vsummary(p, "logo") or {}).get("fatal", "")
    fid = F.role_flow_id(1, 0)
    ok = rc == 4 and 1.4 <= dt <= 2.0 and "1500 ms" in fatal and "--input-wait-ms" in fatal and fid in fatal  # V18: ≤ 2 s
    res.append(F.result("fstart_wait[finite]", "PASS" if ok else "FAIL", why=f"exit={rc} per {dt:.1f} s fatal={fatal!r}"))
    # V18: --start immediate — variklis tiksi nuo starto, --duration skaičiuojama nuo jo; „vis dar laukia po 17 s“ tikrinama
    # tarnybos režime (be --duration), kaip veiks operatorius (anksčiau — --duration 5, kuri prasidėdavo tik prisijungus)
    q = F.FProc(out, "fstart-wait/unlimited", "logo", d, "--flows", 1, "--input-wait-ms", 0, *v)
    time.sleep(17)
    alive = q.p.poll() is None
    t1 = time.monotonic()
    rq = q.term()
    stop_s = time.monotonic() - t1
    txt = err_text(q)
    ok = alive and rq == 0 and stop_s < 3 and "laukiu flow" in txt
    res.append(F.result("fstart_wait[unlimited]", "PASS" if ok else "FAIL",
                        why=f"gyvas po 17 s={alive} SIGTERM exit={rq} per {stop_s:.2f} s 'laukiu flow'={'laukiu flow' in txt}"))
    return res


def fstart_owner(out):
    d = F.fresh_domain("fstart-owner")
    v = F.vcommon("cpu", "host")
    a = F.FProc(out, "fstart-owner/a", "gen", d, "--flows", 1, "--duration", 40, "--run-id", "fstart-first", *v)
    b = None
    rb = None
    h1 = h2 = None
    try:
        F.wait_flows(d, [F.role_flow_id(1, 0)])
        time.sleep(1)
        b = F.FProc(out, "fstart-owner/b", "gen", d, "--flows", 1, "--duration", 5, *v)
        rb = b.wait(timeout=30)
        h1 = F.flow_head(d, F.role_flow_id(1, 0))
        time.sleep(1)
        h2 = F.flow_head(d, F.role_flow_id(1, 0))
    finally:
        for pr in (b, a):
            if pr is not None:
                pr.term()
    sb = F.vsummary(b, "gen") or {}
    sa = F.vsummary(a, "gen") or {}
    fatal = sb.get("fatal", "")
    ok = rb not in (0, None) and "fstart-first" in fatal and h1 is not None and h2 is not None and h2 > h1 and sa.get("run_id") == "fstart-first"
    return [F.result("fstart_owner", "PASS" if ok else "FAIL",
                     why=f"antro gen exit={rb} fatal={fatal!r}; pirmojo head {h1}->{h2}; pirmojo run_id={sa.get('run_id')!r}")]


def fstart_domain(out):
    a, b = domain.make_domain("fstart-dom"), domain.make_domain("fstart-dom")
    distinct = a != b and a.is_dir() and b.is_dir()
    domain.remove_domain(b)
    g = F.FProc(out, "fstart-domain", "gen", a, "--flows", 1, "--duration", 30, *F.vcommon("cpu", "host"))
    removed_live = None
    files_kept = False
    try:
        F.wait_flows(a, [F.role_flow_id(1, 0)])
        time.sleep(1)
        owners = domain.live_owners(a)
        removed_live = domain.remove_domain(a)
        files_kept = (a / f"{F.role_flow_id(1, 0)}.mxl-flow" / "flow_def.json").exists()
    finally:
        g.term()
    removed_after = domain.remove_domain(a)
    gone = not a.exists() and not domain.lease_dir(a).exists()
    ok = distinct and len(owners) == 1 and removed_live is False and files_kept and removed_after is True and gone
    return [F.result("fstart_domain", "PASS" if ok else "FAIL",
                     why=f"unikalūs={distinct} savininkai={owners} remove su gyvu gen={removed_live} failai liko={files_kept} "
                         f"remove po pabaigos={removed_after} pašalinta={gone}")]


def fstart_parallel(out):
    cap = pathlib.Path(__file__).resolve().parents[1] / "perf" / "capacity.py"
    args = ["--func", "logo", "--n", "2", "--seconds", "10", "--warmup", "2", "--impl", "cpu", "--storage", "host", "--gpu-probe", "0",
            "--stop-after-fails", "1"]
    env = dict(F.ENV, FUNC_BIN=F.FUNC_BIN)
    ps = []
    for k in range(2):
        o = out / f"fstart-parallel/{k}"
        o.mkdir(parents=True, exist_ok=True)
        ps.append((o, subprocess.Popen([sys.executable, str(cap), str(o), *args], stdout=open(o / "capacity.out", "w"),
                                       stderr=subprocess.STDOUT, env=env)))
    rcs = [p.wait(timeout=300) for _, p in ps]
    verdict = []
    for o, _ in ps:
        f = o / "capacity-logo-C.json"
        r = json.loads(f.read_text())[0] if f.exists() else {}
        verdict.append({"pass": r.get("pass"), "pass_tol": r.get("pass_tol"), "losses": r.get("losses"),
                        "verif": r.get("verification_errors"), "fatal": r.get("fatal")})
    ok = all(v["pass_tol"] for v in verdict)
    return [F.result("fstart_parallel", "PASS" if ok else "FAIL", why=f"capacity exit={rcs} rezultatai={verdict}")]


def fstart_longpath(out):
    """Rasta pilname funkciniame paleidime: unikalus domeno vardas pailgino kelią, cuda-linear reader'is nebegalėjo prisijungti
    prie `<domenas>/<uuid>.mxl-flow/payload.sock` (sun_path 107 simboliai) ir vartotojas tyliai laukė."""
    res = []
    d = domain.make_domain("x" * 60)  # žyma per ilga — make_domain privalo sutrumpinti
    short_ok = len(str(d)) <= domain.MAX_DOMAIN_PATH and d.is_dir()
    domain.remove_domain(d)
    res.append(F.result("fstart_longpath[make_domain]", "PASS" if short_ok else "FAIL",
                        why=f"kelias {len(str(d))} simb. (riba {getattr(domain, 'MAX_DOMAIN_PATH', None)})"))
    long_dom = domain.SHM / ("vf-fstart-longpath-" + "y" * 40)  # 9 + 59 = 68 simboliai > 48
    long_dom.mkdir(exist_ok=True)
    F.DOMAINS.append(long_dom)
    p = F.FProc(out, "fstart-longpath/G", "gen", long_dom, "--flows", 1, "--duration", 3, "--impl", "gpu", "--storage", "cuda-linear")
    rc = p.wait(timeout=60)
    txt = err_text(p)
    q = F.FProc(out, "fstart-longpath/C", "gen", long_dom, "--flows", 1, "--duration", 2, "--impl", "cpu", "--storage", "host")
    rq = q.wait(timeout=60)  # host flow'ams lizdo nėra — ilgas kelias leidžiamas
    ok = rc == 1 and "payload.sock" in txt and "48" in txt and rq == 0
    res.append(F.result("fstart_longpath[func]", "PASS" if ok else "FAIL", why=f"G exit={rc} stderr={txt.strip()[-200:]!r}; C exit={rq}"))
    return res


def fstart_logo_restart(out):
    res = []
    N, D = 1, 22
    for tag, impl, storage in variants():
        tid = f"fstart_logo_restart[{tag}]"
        sub = f"fstart-logo-restart-{tag}"
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        logo_args = ["--flows", N, "--duration", 60, "--warmup", 1, *v]
        g = F.FProc(out, sub, "gen", d, "--flows", N, "--duration", 80, *v)
        l1 = l2 = vf = None
        alive = False
        h1 = h2 = None
        rv = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)]):
                res.append(F.result(tid, "FAIL", why="gen flow'ai neatsirado"))
                continue
            l1 = F.FProc(out, f"{sub}/a", "logo", d, *logo_args)
            F.wait_flows(d, [F.role_flow_id(2, n) for n in range(N)], 20)
            vf = F.FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 5, "--duration", D, "--warmup", 1,
                         *F.vimpl(impl, storage))
            time.sleep(4)
            l1.kill9()
            time.sleep(2)
            l2 = F.FProc(out, f"{sub}/b", "logo", d, *logo_args)
            time.sleep(5)
            alive = l2.p.poll() is None and vf.p.poll() is None
            h1 = F.flow_head(d, F.role_flow_id(2, 0))
            time.sleep(2)
            h2 = F.flow_head(d, F.role_flow_id(2, 0))
            rv = vf.wait(timeout=D + 60)
        finally:
            r2 = l2.term() if l2 is not None else None
            for pr in (vf, l1, g):
                if pr is not None:
                    pr.term()
        s2, sv = F.vsummary(l2, "logo"), F.vsummary(vf, "verify-logo")
        f2 = ((s2 or {}).get("flows") or [{}])[0]
        fv = ((sv or {}).get("flows") or [{}])[0]
        pv = (sv or {}).get("proc") or {}
        advancing = h1 is not None and h2 is not None and h2 - h1 >= 80  # 2 s = 100 laukų
        ok = (alive and advancing and r2 == 0 and f2.get("fields_ok", 0) > 0 and sum(logo_losses(s2).values()) == 0
              and fv.get("reattached", 0) >= 1 and pv.get("verification_errors", 1) == 0 and pv.get("samples_total", 0) > 0)
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"antras logo gyvas={alive} exit={r2} fatal={(s2 or {}).get('fatal')!r} fields_ok={f2.get('fields_ok')} "
                                f"praradimai={logo_losses(s2)} head {h1}->{h2}; verify exit={rv} reattached={fv.get('reattached')} "
                                f"verification_errors={pv.get('verification_errors')} samples={pv.get('samples_total')}",
                            logo=s2, verify=sv))
    return res


TESTS = {"fstart_order": fstart_order, "fstart_restart": fstart_restart, "fstart_wait": fstart_wait, "fstart_owner": fstart_owner,
         "fstart_domain": fstart_domain, "fstart_parallel": fstart_parallel, "fstart_longpath": fstart_longpath,
         "fstart_logo_restart": fstart_logo_restart}


def fout_error(out):
    """Išėjimo klaidos nėra tylios (peržiūros radinys: po migracijos open / commit klaida baigdavo indeksą kaip „laikrodis
    atgal“, procesas — exit 0 su 0 fields_ok). Injekcija — paslėpta --test-out-error (FaultyOutput), C variantas:
    gen: ilgalaikė commit klaida → exit 2, fatal su išėjimo vardu, output_errors == 3, clock_backward nėra;
    logo: vienkartinė open klaida → exit 4 (praradimas), output_errors == 1, fields_ok > 0, clock_backward nėra;
    logo: ilgalaikė open klaida → exit 2, fatal su išėjimo vardu."""
    res = []
    v = F.vcommon("cpu", "host")
    d = F.fresh_domain("fout-gen")
    g = F.FProc(out, "fout/gen", "gen", d, "--flows", 1, "--duration", 8, "--test-out-error", "commit",
                "--test-out-error-after", 2, *v)
    rg = g.wait(timeout=60)
    sg = g.summary() or {}
    fg = (sg.get("flows") or [{}])[0]
    fat = sg.get("fatal") or ""
    ok = rg == 2 and "išėjimas" in fat and "iš eilės" in fat and fg.get("output_errors") == 3 and not (sg.get("proc") or {}).get(
        "clock_backward")
    res.append(F.result("fout_error[gen commit ilgalaikė]", "PASS" if ok else "FAIL",
                        why=f"exit={rg} fatal={fat!r} output_errors={fg.get('output_errors')} proc={sg.get('proc')}"))
    for name, extra, want_rc in (("logo open vienkartinė", ["--test-out-error-count", 1], 4), ("logo open ilgalaikė", [], 2)):
        d = F.fresh_domain("fout-logo")
        gen = F.FProc(out, f"fout/{name}/gen", "gen", d, "--flows", 1, "--duration", 30, *v)
        lg = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
                res.append(F.result(f"fout_error[{name}]", "FAIL", why="gen flow'as neatsirado"))
                continue
            lg = F.FProc(out, f"fout/{name}", "logo", d, "--flows", 1, "--duration", 6, "--warmup", 1, "--test-out-error", "open",
                         "--test-out-error-after", 2, *extra, *v)
            rl = lg.wait(timeout=60)
        finally:
            gen.term()
            if lg is not None and lg.p.poll() is None:
                lg.term()
        sl = lg.summary() or {}
        fl = (sl.get("flows") or [{}])[0]
        fat = sl.get("fatal") or ""
        if want_rc == 4:
            ok = rl == 4 and fl.get("output_errors") == 1 and fl.get("fields_ok", 0) > 100 and not fat
        else:
            ok = rl == 2 and "išėjimas" in fat and "iš eilės" in fat and fl.get("output_errors") == 3
        ok = ok and not (sl.get("proc") or {}).get("clock_backward")
        res.append(F.result(f"fout_error[{name}]", "PASS" if ok else "FAIL",
                            why=f"exit={rl} fatal={fat!r} output_errors={fl.get('output_errors')} fields_ok={fl.get('fields_ok')} "
                                f"proc={sl.get('proc')}"))
    return res


TESTS["fout_error"] = fout_error


def fformat_change(out):
    """composite, --on-format-change exit (numatyta). C variantas.
    [layer] sluoksnio gamintojas grįžta kitu rastru (logo → full): sluoksnis tampa skaidrus (layer_absent > 0), operatoriaus
            įvykis (layer_format_mismatch ≥ 1, eilutė stderr), composite dirba toliau ir baigiasi exit 0, 0 bazės praradimų.
            Pakeistas lūkestis (C1, peržiūra 2026-11-13 D1): anksčiau — exit 6 ir reopen_mismatch == 1 (sluoksnio formato pokytis
            nuimdavo programą).
    [base]  bazės gamintojas (gen) grįžta kitu formatu (1080i50 → 1080p50): kaip anksčiau — exit 6, reopen_mismatch == 1, fatal."""
    v = F.vcommon("cpu", "host")
    res = []
    for case in ("layer", "base"):
        d = F.fresh_domain(f"ffmtchg-{case}")
        sub = f"ffmtchg-{case}"
        g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 1, "--duration", 60, *v)
        x = F.FProc(out, f"{sub}/gfx1", "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace",
                    "--duration", 60, *v)
        c = x2 = g2 = None
        rc = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, 0), F.role_flow_id(5, 0)]):
                res.append(F.result(f"fformat_change[{case}]", "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                continue
            c = F.FProc(out, sub, "composite", d, "--flows", 1, "--layers", 1, "--layer-xy", "0,0", "--base", "gen",
                        "--out-format", "v210", "--deadline-ms", F.COMP_DEADLINE_MS, "--duration", 14, "--warmup", 1, *v)
            time.sleep(4)
            if case == "layer":
                x.kill9()
                x2 = F.FProc(out, f"{sub}/gfx2", "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "full", "--mode", "pace",
                             "--duration", 60, *v)
            else:
                g.kill9()
                g2 = F.FProc(out, f"{sub}/gen2", "gen", d, "--flows", 1, "--duration", 60, "--format", "1080p50", *v)
            rc = c.wait(timeout=40)
        finally:
            for pr in (c, x2, x, g2, g):
                if pr is not None and pr.p.poll() is None:
                    pr.term()
        s = (c.summary() if c is not None else None) or {}
        f0 = (s.get("flows") or [{}])[0]
        fat = s.get("fatal") or ""
        err = c.err_path.read_text(errors="replace") if c is not None and c.err_path.exists() else ""
        if case == "layer":
            ok = (rc == 0 and f0.get("layer_format_mismatch", 0) >= 1 and f0.get("layer_absent", 0) > 0 and not fat
                  and not f0.get("missing_fields") and not f0.get("reopen_mismatch") and "laukiama tinkamo formato" in err)
            why = (f"exit={rc} layer_format_mismatch={f0.get('layer_format_mismatch')} layer_absent={f0.get('layer_absent')} "
                   f"missing_fields={f0.get('missing_fields')} fields_ok={f0.get('fields_ok')} fatal={fat[:120]!r}")
        else:
            ok = rc == 6 and f0.get("reopen_mismatch") == 1 and fat != ""
            why = f"exit={rc} reopen_mismatch={f0.get('reopen_mismatch')} fatal={fat[:160]!r}"
        res.append(F.result(f"fformat_change[{case}]", "PASS" if ok else "FAIL", why=why))
    return res


TESTS["fformat_change"] = fformat_change


def fduration0(out):
    """--duration 0 — veikti iki sustabdymo (planas D8 / K-T8; prieš migraciją — 0 tick'ų): gen ir logo po 3 s gyvi, SIGTERM →
    exit 0, santraukoje ≥ 100 laukų; `--help` tai sako. C variantas."""
    res = []
    hl = subprocess.run([F.FUNC_BIN, "logo", "--help"], capture_output=True, text=True, env=F.ENV)
    res.append(F.result("fduration0[--help]", "PASS" if "0 — veikti iki sustabdymo" in " ".join(hl.stdout.split()) else "FAIL",
                        why=f"exit {hl.returncode}"))
    v = F.vcommon("cpu", "host")
    d = F.fresh_domain("fdur0")
    g = F.FProc(out, "fdur0/gen", "gen", d, "--flows", 1, "--duration", 0, *v)
    lg = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, 0)]):
            return res + [F.result("fduration0", "FAIL", why="gen flow'as neatsirado")]
        lg = F.FProc(out, "fdur0", "logo", d, "--flows", 1, "--duration", 0, "--warmup", 1, *v)
        time.sleep(4)
        alive = g.p.poll() is None and lg.p.poll() is None
    finally:
        rl = lg.term() if lg is not None else None
        rg = g.term()
    for name, pr, rc in (("gen", g, rg), ("logo", lg, rl)):
        s = (pr.summary() if pr is not None else None) or {}
        span = s.get("last_index", 0) - s.get("first_index", 0) + 1 if s.get("first_index") is not None else 0
        ok = alive and rc == 0 and span >= 100
        res.append(F.result(f"fduration0[{name}]", "PASS" if ok else "FAIL", why=f"gyvi po 4 s={alive} exit={rc} span={span}"))
    return res


TESTS["fduration0"] = fduration0
