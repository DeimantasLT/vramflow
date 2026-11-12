#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Tiksliniai migracijos peržiūros testai (kodas 4567b1e; kviečiami iš functional.py --only fsplit,flate_stall,fcrashloop).

fsplit      — vieno indekso vienetai commit'inami skirtingose fazėse (A ir B / C): N = 4, vieno flow'o gamintojas „trūkčioja“
              (SIGSTOP 20 ms kas 200 ms — grain'ai ateina po A ribos, dalis po kito tick'o atidarymo → pernešimas į C), kito —
              nužudomas ir paleidžiamas iš naujo, trečio — SIGSTOP 2 s / SIGCONT. verify --every 1 visiems flow'ams visą laiką:
              0 neatitikimų (mismatch_fields), verify missing + invalid ≤ funkcijos pačios suskaičiuoti praradimai tam flow'ui,
              sveiko flow'o praradimų 0; held_ms histograma netuščia (fazės iš tikrųjų skilo). logo C / H / G, composite H / G
              (--deadline-ms 30: C fazė yra).
flate_stall — vėluojantis ir užstrigęs flow'ai TAME PAČIAME indekse (aligned_cycles: abu SIGSTOP ties T(k) − 0,5 ms, vėluojantis
              tęsiamas po 18,5 ms, užstrigęs — po 150 ms; 10 ciklų): C fazės kelias — paruoštas pernešto indekso vienetas laikomas
              ≤ Wr nuo savo paruošimo, ne iki C ribos (1fa6d8d). Sveikų flow'ų praradimų 0, kiekvieno sveiko maks. vėlavimas
              ≤ 2 · Wr + 10 ms, vėluojančio — 0 neatitikimų ir commit ≤ jo atėjimas + Wr + 3 ms. logo C / G.
fcrashloop  — gen B nužudomas ir paleidžiamas iš naujo kas 2 s 20 s: sveikų flow'ų praradimų 0 (išskyrus aukštupio — gen A —
              eilutes gedimo lange), logo procesas be nuotėkių: /proc/<pid>/fd skaičius po ≤ prieš + 2, RSS augimas ≤ 16 MB,
              G — GPU atminties augimas: proceso paties (nvidia-smi --query-compute-apps) ≤ 16 MB, o kur tvarkyklė jo nerodo
              (WDDM) — viso GPU memory.used medianos (5 matavimai) ≤ 64 MB. logo C / G.
Testai tikrina tik iš išorės (procesai, signalai, exit kodai, santraukos, /proc, nvidia-smi).
"""
import os
import pathlib
import signal
import subprocess
import threading
import time

import functional as F  # noqa: E402
import isolation as I  # noqa: E402  upstream(), window_budget(), FN_KEYS

VARIANTS = {"C": ("cpu", "host"), "H": ("gpu", "host"), "G": ("gpu", "cuda-linear")}
N = 4
DUR = 20
LOSS = ("late_fields", "late_skips", "lapped", "too_late", "in_invalid", "read_errors", "missing_fields", "flow_invalid", "output_errors")


def jitter(pr, stop_evt, pulse_s=0.020, every_s=0.200):
    """Gamintojo „trūkčiojimas“: SIGSTOP pulse_s kas every_s (grain'ai ateina pavėlavę, ne prarasti)."""
    while not stop_evt.is_set() and pr.p.poll() is None:
        try:
            pr.p.send_signal(signal.SIGSTOP)
            time.sleep(pulse_s)
            pr.p.send_signal(signal.SIGCONT)
        except ProcessLookupError:
            return
        stop_evt.wait(every_s)
    try:
        pr.p.send_signal(signal.SIGCONT)
    except ProcessLookupError:
        pass


def fn_loss(f):
    return sum(f.get(k, 0) for k in LOSS)


def proc_mem(pid):
    """(fd skaičius, VmRSS kB) arba (None, None)."""
    try:
        fds = len(os.listdir(f"/proc/{pid}/fd"))
        rss = next(int(ln.split()[1]) for ln in open(f"/proc/{pid}/status") if ln.startswith("VmRSS:"))
        return fds, rss
    except (OSError, StopIteration, ValueError):
        return None, None


def smi_bin():
    return "nvidia-smi" if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0 else "/usr/lib/wsl/lib/nvidia-smi"


def gpu_used_mb():
    try:
        r = subprocess.run([smi_bin(), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10)
        return int(r.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def proc_vram_mb(pid, samples=5):
    """(MB, šaltinis): proceso PATIES GPU atmintis (nvidia-smi --query-compute-apps used_memory), kai tvarkyklė ją rodo; WDDM
    (WSL2 nešiojamas) jos nerodo („Not available in WDDM driver model“, /proc/<pid>/maps CUDA rezervuoja VA iš anksto ir
    nekinta) — tada viso GPU memory.used mediana iš `samples` matavimų per ~1 s (tas pats procesų rinkinys gyvas abiem
    momentais). Šaltinis rašomas į rezultatą."""
    vals = []
    try:
        r = subprocess.run([smi_bin(), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        for ln in r.stdout.splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) == 2 and p[0] == str(pid) and p[1].isdigit():
                return int(p[1]), "process"
    except (OSError, subprocess.TimeoutExpired):
        pass
    for _ in range(samples):
        v = gpu_used_mb()
        if v is not None:
            vals.append(v)
        time.sleep(0.2)
    return (sorted(vals)[len(vals) // 2] if vals else None), "gpu-median"


def start_sources(out, sub, d, func, v, roles):
    """roles: flow'o numeris → gamintojo žyma; grąžina {žyma: FProc}. composite — ir gfxgen sluoksniai (vienas procesas visiems)."""
    procs = {}
    for tag, flows in roles.items():
        procs[tag] = F.FProc(out, f"{sub}/gen-{tag}", "gen", d, "--flows", len(flows), "--flow-base", flows[0], "--duration", DUR + 60, *v)
    if func == "composite":
        procs["gfx"] = F.FProc(out, f"{sub}/gfx", "gfxgen", d, "--flows", N, "--layers", 1, "--layer-geom", "logo", "--mode", "pace",
                               "--duration", DUR + 60, *v)
    return procs


class NiceVerify(F.FProc):
    """verify — stebėtojas, ne tikrinama sistema: žemesnis prioritetas (nice 10), kad CPU etalonas (--every 1, composite —
    dešimtys ms laukui) neatimtų CPU iš funkcijos; nespėjus — check_dropped (informacinis), ne klaida."""

    def __init__(self, out, tag, domain, *args):
        self.out = pathlib.Path(out) / tag
        self.out.mkdir(parents=True, exist_ok=True)
        self.sub = "verify"
        self.err_path = self.out / "verify.err"
        self.p = subprocess.Popen(["nice", "-n", "10", F.FUNC_BIN, "verify", "--domain", str(domain), "--out", str(self.out),
                                   *map(str, args)], stdout=subprocess.DEVNULL, stderr=open(self.err_path, "w"), env=F.ENV)


def run_func(out, sub, d, func, v, impl, storage):
    common = ["--flows", N, "--duration", DUR, "--warmup", 2, *v]
    if func == "logo":
        fn = F.FProc(out, sub, "logo", d, "--deadline-ms", 30, *common)
        vf = NiceVerify(out, sub, d, "--mode", "logo", "--flows", N, "--every", 1, "--duration", DUR - 4, "--warmup", 1,
                     *F.vimpl(impl, storage))
    else:
        xy = ["--layer-xy", "1488,64"]
        fn = F.FProc(out, sub, "composite", d, "--layers", 1, *xy, "--opacity", "1.0", "--base", "gen", "--out-format", "v210",
                     "--deadline-ms", 30, *common)
        vf = NiceVerify(out, sub, d, "--mode", "composite", "--flows", N, "--layers", 1, "--layer-geom", "logo", "--gfx-mode",
                     "pace", "--base", "gen", "--out-format", "v210", "--every", 1, "--duration", DUR - 4, "--warmup", 1, *xy,
                     "--opacity", "1.0", *F.vimpl(impl, storage))
    return fn, vf


FIELD_NS = 20_000_000  # 1080i50
STOP_LEAD_NS = 500_000  # abu gamintojai sustabdomi 0,5 ms iki T(k): grain'as k dar neįrašytas
JITTER_STOP_S = 0.0185  # vėluojantis tęsiamas ties T(k) + 18 ms: grain'as k ateina po b = T + 17,5 (C fazė, D = 30)
STALL_STOP_S = 0.150   # užstrigęs — 7,5 lauko (≥ 3 praleidimai → Stalled, tada atsigauna)
ALIGNED_CYCLES = 10


def aligned_cycles(jit, stl):
    """10 ciklų kas 1 s: abu gamintojai SIGSTOP ties T(k) − 0,5 ms (TAI), vėluojantis SIGCONT po 18,5 ms, užstrigęs — po 150 ms.
    Indekse k abu vienetai pernešami į C: vėluojantis paruoštas ~T + 18,5…21, užstrigusio brolio grain'o nėra. Teisinga C
    taisyklė (1fa6d8d): paruoštas laikomas ≤ Wr nuo t_r — commit ~atėjimas + 6 ms; be jos — iki C ribos T + 30."""
    for _ in range(ALIGNED_CYCLES):
        now = time.clock_gettime_ns(time.CLOCK_TAI)
        k = now // FIELD_NS + 50  # ~1 s į priekį
        at = k * FIELD_NS - STOP_LEAD_NS
        while True:
            left = at - time.clock_gettime_ns(time.CLOCK_TAI)
            if left <= 0:
                break
            time.sleep(left / 1e9 if left > 3_000_000 else 0)
        for pr in (jit, stl):
            pr.p.send_signal(signal.SIGSTOP)
        time.sleep(JITTER_STOP_S)
        jit.p.send_signal(signal.SIGCONT)
        time.sleep(STALL_STOP_S - JITTER_STOP_S)
        stl.p.send_signal(signal.SIGCONT)


def scenario(out, name, func, tag, impl, storage, faults, aligned=False):
    """faults: {"jitter": flow, "stall": flow, "restart": flow}; kiekvienas paveiktas flow'as — savo gen procesas. aligned —
    vietoj atsitiktinio „trūkčiojimo“ ir 2 s sustabdymo: aligned_cycles (flate_stall)."""
    sub = f"{name}-{func}-{tag}"
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    special = set(faults.values())
    # sveiki — po vieną gen kiekvienai ištisinei atkarpai (A0, A1, …): paveikti flow'ai gali būti bet kurioje vietoje
    roles, run = {}, []
    for n in range(N + 1):
        if n < N and n not in special:
            run.append(n)
        elif run:
            roles[f"A{len(roles)}"], run = run, []
    for k, n in faults.items():
        roles[k] = [n]
    procs = start_sources(out, sub, d, func, v, roles)
    fn = vf = None
    evt = threading.Event()
    th = []
    rc = {}
    try:
        ids = [F.role_flow_id(1, n) for n in range(N)] + ([F.role_flow_id(5, n * 8) for n in range(N)] if func == "composite" else [])
        if not F.wait_flows(d, ids):
            return {"err": "šaltinių flow'ai neatsirado"}
        fn, vf = run_func(out, sub, d, func, v, impl, storage)
        time.sleep(4)
        if aligned:  # flate_stall: vėluojantis ir užstrigęs — tame pačiame indekse (C fazės kelias, žr. aligned_cycles)
            aligned_cycles(procs["jitter"], procs["stall"])
        elif "jitter" in faults:
            th.append(threading.Thread(target=jitter, args=(procs["jitter"], evt), daemon=True))
            th[-1].start()
        if "stall" in faults and not aligned:
            procs["stall"].p.send_signal(signal.SIGSTOP)
            time.sleep(2)
            procs["stall"].p.send_signal(signal.SIGCONT)
        if "restart" in faults:
            time.sleep(1)
            procs["restart"].kill9()
            time.sleep(1)
            n = faults["restart"]
            procs["restart"] = F.FProc(out, f"{sub}/gen-restart2", "gen", d, "--flows", 1, "--flow-base", n, "--duration", DUR + 60, *v)
        rc["verify"] = vf.wait(timeout=DUR + 60)
        rc[func] = fn.wait(timeout=DUR + 60)
    finally:
        evt.set()
        for t in th:
            t.join(timeout=5)
        I.stop_all([vf, fn] + list(procs.values()))
    sf = F.vsummary(fn, func)
    sv = F.vsummary(vf, "verify-composite" if func == "composite" else "verify-logo")
    if sf is None or sv is None:
        return {"err": f"nėra santraukų ({func}={sf is not None}, verify={sv is not None})"}
    return {"rc": rc, "fn": sf, "verify": sv, "upstream": I.upstream([p for k, p in procs.items() if k.startswith("A")]),
            "special": special, "jitter_up": I.upstream([procs["jitter"]]) if "jitter" in procs else None}


def check_split(r, func):
    bad = []
    sf, sv = r["fn"], r["verify"]
    ff, vfl = sf.get("flows") or [], sv.get("flows") or []
    # Sveiko aukštupio (gen A…) paties praradimai (eilutės „vėlus commit“ > --deadline-ms ir „praleista“) — ne funkcijos;
    # kitų procesų GPU įvykiai (pvz. perkrauto gen CUDA kontekstas H variante) gali pavėlinti ir sveiką gamintoją.
    up = r.get("upstream") or {}
    budget = len(up.get("late_index") or []) + sum(b - a + 1 for a, b in up.get("skipped") or [])
    for n in range(N):
        f, vv = (ff[n] if n < len(ff) else {}), (vfl[n] if n < len(vfl) else {})
        if vv.get("mismatch_fields", 0):
            bad.append(f"#{n} mismatch_fields={vv['mismatch_fields']}")
        if not vv.get("samples", 0):
            bad.append(f"#{n} verify samples == 0")
        vl = vv.get("missing", 0) + vv.get("invalid", 0)
        if vl > fn_loss(f) + (0 if n not in r["special"] else 0):
            bad.append(f"#{n} verify missing + invalid {vl} > funkcijos praradimai {fn_loss(f)}")
        if n not in r["special"] and fn_loss(f) > budget:
            bad.append(f"sveiko #{n} praradimai { {k: f[k] for k in LOSS if f.get(k)} } > aukštupio {budget}")
    held = ((sf.get("hist") or {}).get("held_ms") or {}).get("count", 0)
    if not held:
        bad.append("held_ms tuščia — fazės neskilo (scenarijus nepatikrino skilusio indekso)")
    return bad, held


SPLIT_HEALTHY = {("logo", "C"): 0, ("logo", "H"): 1, ("logo", "G"): 2, ("composite", "H"): 3, ("composite", "G"): 1}


def fsplit(out):
    res = []
    for func, tags in (("logo", ("C", "H", "G")), ("composite", ("H", "G"))):
        for tag in tags:
            impl, storage = VARIANTS[tag]
            tid = f"fsplit[{func}-{tag}]"
            # sveiko flow'o vieta sukama per variantus (0, 1, 2, 3, 0): variklio tvarka negali paslėpti sugedusių už sveiko
            healthy = SPLIT_HEALTHY[(func, tag)]
            r = scenario(out, "fsplit", func, tag, impl, storage,
                         dict(zip(("jitter", "restart", "stall"), [n for n in range(N) if n != healthy])))
            if r.get("err"):
                res.append(F.result(tid, "FAIL", why=r["err"]))
                continue
            bad, held = check_split(r, func)
            res.append(F.result(tid, "PASS" if not bad else "FAIL",
                                why=f"exit={r['rc']} held_ms count={held}; {'; '.join(bad) or 'verify --every 1: 0 neatitikimų'}"))
    return res


def flate_stall(out):
    res = []
    for tag in ("C", "G"):
        impl, storage = VARIANTS[tag]
        tid = f"flate_stall[logo-{tag}]"
        # C: sveiki 1 ir 3 (du gen, paveikti 0 ir 2); G: sveiki 0 ir 2 (paveikti 1 ir 3) — ne „paveikti visada paskutiniai“
        faults = {"C": {"jitter": 0, "stall": 2}, "G": {"jitter": 1, "stall": 3}}[tag]
        r = scenario(out, "flatestall", "logo", tag, impl, storage, faults, aligned=True)
        if r.get("err"):
            res.append(F.result(tid, "FAIL", why=r["err"]))
            continue
        bad, held = check_split(r, "logo")
        ff = r["fn"].get("flows") or []
        lim = 2 * I.WR_MS + 10.0
        for n in range(N):
            if n in r["special"] or n >= len(ff):
                continue
            m = ff[n].get("lateness_max_us")
            if m is None or m / 1000.0 > lim:
                bad.append(f"sveiko #{n} maks. vėlavimas {None if m is None else m / 1000.0} ms > {lim} ms")
        # C fazės taisyklė (1fa6d8d): vėluojančio grain'as, atėjęs po b kartu su užstrigusiu broliu, commit'inamas ≤ jo atėjimas
        # + Wr + 3 ms (apdorojimas, 1 ms laukimo gabalai; išmatuota +5,9 ms), ne ties C riba T + 30 (be taisyklės — +8,5…10,7). Atėjimas — vėluojančio gen paties commit'o
        # vėlavimo maksimumas (commit_lateness_s); jis turi būti po b (kitaip scenarijus C kelio nepatikrino).
        j = faults["jitter"]
        up_s = list(((r.get("jitter_up") or {}).get("lat_s") or {}).values())
        arr = max(up_s) if up_s else None
        mj = (ff[j].get("lateness_max_us") or 0) / 1000.0 if j < len(ff) else None
        jlim = None if arr is None else arr + I.WR_MS + 3.0
        if arr is None or arr <= 17.5:
            bad.append(f"vėluojančio gen maks. commit'as {arr} ms ≤ b = 17,5 ms (C kelias nepatikrintas)")
        elif mj is None or mj > jlim:
            bad.append(f"vėluojančio #{j} maks. vėlavimas {mj} ms > atėjimas {arr:.2f} + Wr + 3 = {jlim:.2f} ms (laukta C ribos?)")
        res.append(F.result(tid, "PASS" if not bad else "FAIL",
                            why=f"exit={r['rc']} held_ms count={held}; vėluojančio #{j}: atėjimas {arr} ms, commit maks. {mj} ms; "
                                f"{'; '.join(bad) or 'sveiki be praradimų, vėluojantis ≤ atėjimas + Wr'}"))
    return res


def fcrashloop(out):
    res = []
    for tag in ("C", "G"):
        impl, storage = VARIANTS[tag]
        tid = f"fcrashloop[logo-{tag}]"
        sub = f"fcrash-{tag}"
        d = F.fresh_domain(sub)
        v = F.vcommon(impl, storage)
        ga = F.FProc(out, f"{sub}/genA", "gen", d, "--flows", N - 1, "--duration", 120, *v)
        gb = F.FProc(out, f"{sub}/genB0", "gen", d, "--flows", 1, "--flow-base", N - 1, "--duration", 120, *v)
        lg = None
        before = after = (None, None)
        vram0 = vram1 = None
        vsrc = vsrc1 = None
        rl = None
        t_fault = None
        try:
            if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)]):
                res.append(F.result(tid, "FAIL", why="gen flow'ai neatsirado"))
                continue
            lg = F.FProc(out, sub, "logo", d, "--flows", N, "--duration", 34, "--warmup", 2, *v)
            if not F.wait_flows(d, [F.role_flow_id(2, n) for n in range(N)], 30):
                res.append(F.result(tid, "FAIL", why="logo išėjimai neatsirado"))
                continue
            time.sleep(5)
            before, (vram0, vsrc) = proc_mem(lg.p.pid), proc_vram_mb(lg.p.pid)
            t_fault = I.tai_ns()
            for k in range(10):  # 20 s: kill -9 ir paleidimas iš naujo kas 2 s
                gb.kill9()
                time.sleep(0.3)
                gb = F.FProc(out, f"{sub}/genB{k + 1}", "gen", d, "--flows", 1, "--flow-base", N - 1, "--duration", 120, *v)
                time.sleep(1.7)
            time.sleep(4)  # paskutinis prisijungimas iš naujo ir nusistovėjimas
            after, (vram1, vsrc1) = proc_mem(lg.p.pid), proc_vram_mb(lg.p.pid)
            rl = lg.wait(timeout=60)
        finally:
            I.stop_all([lg, gb, ga])
        sl = F.vsummary(lg, "logo") or {}
        ff = sl.get("flows") or []
        up = I.upstream(ga)
        budget = I.window_budget(up, t_fault, sl.get("last_index")) or 0
        bad = []
        for n in range(N - 1):
            f = ff[n] if n < len(ff) else {}
            if fn_loss(f) > budget:
                bad.append(f"sveiko #{n} praradimai { {k: f[k] for k in LOSS if f.get(k)} } > aukštupio {budget}")
        fa = ff[N - 1] if len(ff) >= N else {}
        if fa.get("reattached", 0) < 8:
            bad.append(f"paveiktas reattached={fa.get('reattached', 0)} < 8")
        if None in before or None in after:
            bad.append(f"/proc nepasiekiamas ({before} → {after})")
        else:
            if after[0] > before[0] + 2:
                bad.append(f"fd {before[0]} → {after[0]}")
            if after[1] - before[1] > 16 * 1024:
                bad.append(f"RSS {before[1]} → {after[1]} kB")
        # proceso paties skaičius (jei tvarkyklė rodo) — ≤ 16 MB; WDDM — viso GPU mediana, ≤ 64 MB (vienas nutekėjęs 10 laukų
        # ring'as ~28 MB, 10 perkrovimų — ~280 MB)
        vlim = 16 if vsrc == "process" else 64
        if tag == "G" and (vram0 is None or vram1 is None or vsrc != vsrc1):
            bad.append(f"GPU atmintis neišmatuota ({vram0} → {vram1}, {vsrc} / {vsrc1})")
        elif tag == "G" and vram1 - vram0 > vlim:
            bad.append(f"GPU atmintis ({vsrc}) {vram0} → {vram1} MB > +{vlim}")
        res.append(F.result(tid, "PASS" if not bad else "FAIL",
                            why=f"exit={rl} fd {before[0]}→{after[0]} RSS {before[1]}→{after[1]} kB GPU ({vsrc}) {vram0}→{vram1} MB "
                                f"reattached={fa.get('reattached')} aukštupio biudžetas={budget}; {'; '.join(bad) or 'be nuotėkių ir sveikų praradimų'}"))
    return res


TESTS = {"fsplit": fsplit, "flate_stall": flate_stall, "fcrashloop": fcrashloop}
