#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Paleidimo ir sustabdymo funkciniai testai (planas §P6). Kviečiami iš functional.py (--only fstart_order,fstart_restart,…).

fstart_order    — logo ir verify paleisti 20 s (> buvusios 15 s ribos) PRIEŠ gen: abu laukia (eilutė „laukiu flow“), po gen
                  starto dirba normaliai: logo be praradimų, verify 0 teisingumo klaidų, abu exit 0.
fstart_restart  — gen nužudomas (kill -9) ir po 5 s paleidžiamas vėl: logo gyvas, reattached == 1, spraga suskaičiuota
                  (missing_fields > 0), kitų praradimų 0, išėjimas vėl juda; verify 0 teisingumo klaidų.
fstart_wait     — `--input-wait-ms 1500` be gamintojo → exit 4, aiškus tekstas (riba, parinktis, flow ID);
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
    for tag, impl, storage in variants():
        tid = f"fstart_order[{tag}]"
        sub = f"fstart-order-{tag}"
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        lg = vf = g = None
        alive = False
        try:
            lg = F.FProc(out, sub, "logo", d, "--flows", N, "--duration", D + 6, "--warmup", 2, *v)
            vf = F.FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 5, "--duration", D, "--warmup", 2,
                         *F.vimpl(impl, storage))
            time.sleep(PRE)
            alive = lg.p.poll() is None and vf.p.poll() is None
            g = F.FProc(out, sub, "gen", d, "--flows", N, "--duration", D + PRE + 30, *v)
            rv = vf.wait(timeout=D + 60)
            rl = lg.wait(timeout=D + 60)
        finally:
            for pr in (vf, lg, g):
                if pr is not None:
                    pr.term()
        sl, sv = F.vsummary(lg, "logo"), F.vsummary(vf, "verify-logo")
        lost = sum(logo_losses(sl).values())
        okf = [f.get("fields_ok", 0) for f in (sl or {}).get("flows") or []]
        pv = (sv or {}).get("proc") or {}
        waited = "laukiu flow" in err_text(lg)
        ok = (alive and waited and rl == 0 and rv == 0 and lost == 0 and len(okf) == N and min(okf) >= 0.9 * D * 50
              and pv.get("verification_errors", 1) == 0 and pv.get("samples_total", 0) > 0)
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"gyvi po {PRE} s={alive} 'laukiu flow'={waited} logo exit={rl} praradimai={lost} fields_ok={okf} "
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
        lost = logo_losses(sl)
        pv = (sv or {}).get("proc") or {}
        advancing = h1 is not None and h2 is not None and h2 - h1 >= 80  # 2 s = 100 laukų
        ok = (alive and rl in (0, 4) and sl is not None and all(r == 1 for r in reatt) and all(m > 0 for m in missing)
              and sum(lost.values()) == 0 and advancing and pv.get("verification_errors", 1) == 0 and pv.get("samples_total", 0) > 0)
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"gyvi per outage={alive} logo exit={rl} reattached={reatt} missing_fields={missing} kiti={lost} "
                                f"head po paleidimo {h1}->{h2} verify exit={rv} verification_errors={pv.get('verification_errors')} "
                                f"capacity_errors={pv.get('capacity_errors')} samples={pv.get('samples_total')}",
                            logo=sl, verify=sv))
    return res


def fstart_wait(out):
    res = []
    d = F.fresh_domain("fstart-wait")
    v = ["--impl", "cpu", "--storage", "host"]
    t0 = time.monotonic()
    p = F.FProc(out, "fstart-wait/finite", "logo", d, "--flows", 1, "--duration", 5, "--input-wait-ms", 1500, *v)
    rc = p.wait(timeout=60)
    dt = time.monotonic() - t0
    fatal = (F.vsummary(p, "logo") or {}).get("fatal", "")
    fid = F.role_flow_id(1, 0)
    ok = rc == 4 and 1.4 <= dt < 10 and "1500 ms" in fatal and "--input-wait-ms" in fatal and fid in fatal
    res.append(F.result("fstart_wait[finite]", "PASS" if ok else "FAIL", why=f"exit={rc} per {dt:.1f} s fatal={fatal!r}"))
    q = F.FProc(out, "fstart-wait/unlimited", "logo", d, "--flows", 1, "--duration", 5, "--input-wait-ms", 0, *v)
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
    v = ["--impl", "cpu", "--storage", "host"]
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
    g = F.FProc(out, "fstart-domain", "gen", a, "--flows", 1, "--duration", 30, "--impl", "cpu", "--storage", "host")
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
