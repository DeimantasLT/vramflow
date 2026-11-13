#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Funkciniai gedimų testai (architektūra §6.1: F5a, F5b, F6a–d, F7, F8). Kiekvienas procesas — atskiras.

Naudojimas: functional.py <out_dir> [--only F5a,F6,fgfx1,fcomp1,ffmt_logo]  (vardai be didžiųjų/mažųjų skirtumo)
A/V sinchronizacija: favsync_drop, favsync_restart (tests/functional/avsync.py).
Formatų testai (P2): ffmt_neg, ffmt_logo, ffmt_comp, ffmt_enc, ffmt_dec; FFMT_FORMATS=1080i50,1080p25,1080p50 (numatyta),
FFMT_DURATION=20 (s), FCOMP_VARIANTS=C,H,G.
Aplinka: VERIFY_BIN (senieji testai), FUNC_BIN + MXL_INFO (fgfx1 ir kiti vramflow-func testai).
Rezultatas: <out_dir>/functional.json; išėjimas 0 tik jei nė vienas testas nėra FAIL.
"""
import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf"))
import domain  # noqa: E402  tests/perf/domain.py — unikalūs domenai, trynimas tik be gyvų savininkų
import keepwarm  # noqa: E402  tests/perf/keepwarm.py — GPU virš P8 GPU variantų metu (VRAMFLOW_GPU_KEEPWARM=0 išjungia)

BIN = os.environ.get("VERIFY_BIN", str(pathlib.Path.home() / "build/vramflow/release/vramflow-verify"))
FUNC_BIN = os.environ.get("FUNC_BIN", str(pathlib.Path.home() / "build/vramflow/func/vramflow-func"))
MXL_INFO = os.environ.get("MXL_INFO", "/root/src/mxl-pr653/install/Linux-Clang-Release/bin/mxl-info")
LIB = os.environ.get("MXL_LIB", str(pathlib.Path.home() / "src/mxl-pr653/install/Linux-Clang-Release/lib"))
ENV = dict(os.environ, LD_LIBRARY_PATH=LIB)
# GPU „šildymas“ (tests/perf/keepwarm.py): įjungiamas GPU variantams, išjungiamas CPU (C) variantams — per gpu_mode()
# (vcommon jį kviečia kiekvieno varianto pradžioje); kiekvienas rezultatas įrašo, ar jis veikė (laukas keepwarm).
KW = keepwarm.KeepWarm(FUNC_BIN, ENV)


def gpu_mode(impl):
    """Prieš varianto procesus: impl gpu (H, G) — keepwarm veikia, cpu (C) — ne."""
    KW.want(impl == "gpu")


class Proc:
    def __init__(self, role, domain, out, *args):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.err = open(self.out / f"{role}.err", "w")
        extra = ["--host-crosscheck-every", "10"] if role == "reader" else []  # CPU kryžminė patikra visada įjungta
        self.p = subprocess.Popen([BIN, role, "--domain", str(domain), "--out", str(self.out), *extra, *map(str, args)],
                                  stdout=subprocess.PIPE, stderr=self.err, env=ENV, text=True)
        self.role = role

    def wait(self, timeout=None):
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill9(self):
        self.p.send_signal(signal.SIGKILL)
        self.p.wait()

    def term(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
        return self.p.wait(timeout=30)

    def summary(self):
        f = self.out / f"summary-{self.role}.json"
        return json.loads(f.read_text()) if f.exists() else None


def counters(s):
    return s["flows"][0]["counters"] if s else {}


def verify_errors(s):
    return s["totals"]["verification_errors"] if s else None


DOMAINS = []  # šio paleidimo domenai — pašalinami pabaigoje (tik be gyvų savininkų)


def fresh_domain(name):
    """Unikalus domenas šiam paleidimui (planas §P6): du functional.py paleidimai vienas kito domenų netrina."""
    d = domain.make_domain(name)
    DOMAINS.append(d)
    return d


def cleanup_domains():
    left = [str(d) for d in DOMAINS if not domain.remove_domain(d)]
    if left:
        print(f"[functional] domenai su gyvais savininkais nepašalinti: {left}", flush=True)


def leftovers(domain):
    return sorted(p.name for p in pathlib.Path(domain).iterdir() if p.name != "options.json")


def result(tid, status, echo=True, **details):
    if echo:  # echo=False — tarpinis įvertinimas (pvz. INFO rankos), kurio eilutė neturi atrodyti kaip vartų FAIL
        print(f"[{status:7s}] {tid}: {details.get('why', '')}", flush=True)
    return {"id": tid, "status": status, "keepwarm": KW.on(), **details}


def f5a(out):
    gpu_mode("gpu")  # vramflow-verify writer / reader — GPU (cuda-linear)
    d = fresh_domain("f5a")
    a = Proc("writer", d, out / "f5a/writerA", "--duration", 25, "--gc-before-create")
    time.sleep(2)
    r = Proc("reader", d, out / "f5a/reader", "--duration", 18)
    time.sleep(3)
    sock = d / "4d584c56-0000-4000-8000-000000000000.mxl-flow" / "payload.sock"
    ino0 = sock.stat().st_ino if sock.exists() else None
    t0 = time.monotonic()
    b = Proc("writer", d, out / "f5a/writerB", "--duration", 10)
    bcode = b.wait(timeout=10)
    dt = time.monotonic() - t0
    ino1 = sock.stat().st_ino if sock.exists() else None
    r.wait(timeout=40)
    a.term()
    rs = r.summary()
    c = counters(rs)
    ok = bcode == 5 and dt < 5 and ino0 is not None and ino0 == ino1 and verify_errors(rs) == 0 and c.get("gap_grains") == 0
    return result("F5a", "PASS" if ok else "FAIL", why=f"antro writer'io exit={bcode} per {dt:.2f}s; socket inode "
                  f"{'nepakito' if ino0 == ino1 else 'PAKITO'}; reader verif={verify_errors(rs)} gap={c.get('gap_grains')} "
                  f"ok={c.get('grains_ok')}", second_writer_exit=bcode, seconds=dt, reader=c)


def f5b(out, iters=10):
    gpu_mode("gpu")  # vramflow-verify writer / reader — GPU (cuda-linear)
    bad = []
    for it in range(iters):
        d = fresh_domain(f"f5b")
        w1 = Proc("writer", d, out / f"f5b/{it}/w1", "--duration", 12)
        w2 = Proc("writer", d, out / f"f5b/{it}/w2", "--duration", 12)
        time.sleep(3)
        codes = [w1.p.poll(), w2.p.poll()]
        alive = [x is None for x in codes]
        r = Proc("reader", d, out / f"f5b/{it}/reader", "--duration", 3)
        r.wait(timeout=30)
        rs = r.summary()
        for w in (w1, w2):
            if w.p.poll() is None:
                w.term()
        exited = [c for c in codes if c is not None]
        ok = sum(alive) == 1 and exited == [5] and verify_errors(rs) == 0 and counters(rs).get("grains_ok", 0) >= 100
        if not ok:
            bad.append({"iter": it, "codes": codes, "reader_ok": counters(rs).get("grains_ok"), "verif": verify_errors(rs)})
    left = leftovers(d)
    return result("F5b", "PASS" if not bad else "FAIL",
                  why=f"{iters - len(bad)}/{iters} iteracijų: lygiai vienas writer'is gyvas, kitas exit 5, reader ≥100 grain'ų be klaidų",
                  failures=bad, domain_leftovers=left)


def f6(out):
    gpu_mode("gpu")  # vramflow-verify writer / reader — GPU (cuda-linear)
    res = []
    d = fresh_domain("f6")
    w1 = Proc("writer", d, out / "f6/w1", "--duration", 60, "--gc-before-create")
    time.sleep(2)
    r = Proc("reader", d, out / "f6/reader", "--duration", 40, "--reopen-timeout-ms", 20000, "--stall-timeout-ms", 1000)
    time.sleep(8)
    w1.kill9()
    t_kill = time.monotonic()
    # F6d: naujas reader'is prie mirusio flow — turi baigtis švariai
    r2 = Proc("reader", d, out / "f6/reader_new", "--duration", 10, "--wait-flow-ms", 3000)
    r2code = r2.wait(timeout=30)
    res.append(result("F6d", "PASS" if r2code in (2, 4) else "FAIL",
                      why=f"naujas reader'is prie mirusio flow: exit={r2code} (laukta 2 arba 4, be pakibimo)", exit=r2code))
    # F6b: restart be GC
    w2 = Proc("writer", d, out / "f6/w2_nogc", "--duration", 10)
    w2code = w2.wait(timeout=15)
    if w2code is None:
        w2.term()
    res.append(result("F6b", "INFO" if w2code is not None and w2code >= 0 else "FAIL",
                      why=f"restart be GC po kill -9: exit={w2code} (architektūroje laukta 5 CONFLICT)", exit=w2code))
    # F6c: restart su GC — senas reader'is turi atsidaryti iš naujo ir toliau tikrinti
    t_restart = time.monotonic()
    w3 = Proc("writer", d, out / "f6/w3_gc", "--duration", 40, "--gc-before-create")
    rcode = r.wait(timeout=60)
    w3.term()
    rs = r.summary()
    c = counters(rs)
    ok_a = rcode is not None and rcode >= 0 and c.get("cuda_errors") == 0 and verify_errors(rs) == 0 and c.get("stalls", 0) >= 1
    res.append(result("F6a", "PASS" if ok_a else "FAIL",
                      why=f"writer kill -9: reader exit={rcode} (ne signalas), cuda_errors={c.get('cuda_errors')}, "
                          f"verif={verify_errors(rs)}, stalls={c.get('stalls')}", reader=c, restart_after_s=t_restart - t_kill))
    ok_c = c.get("reopens", 0) >= 1 and verify_errors(rs) == 0 and c.get("grains_ok", 0) >= 500
    res.append(result("F6c", "PASS" if ok_c else "FAIL",
                      why=f"restart su GC: reopens={c.get('reopens')}, grains_ok={c.get('grains_ok')}, verif={verify_errors(rs)}"))
    return res


def f7(out, cycles=10):
    gpu_mode("gpu")  # vramflow-verify writer / reader — GPU (cuda-linear)
    d = fresh_domain("f7")
    w = Proc("writer", d, out / "f7/writer", "--duration", 18 + cycles * 3, "--gc-before-create")
    time.sleep(2)
    ra = Proc("reader", d, out / "f7/readerA", "--duration", 10 + cycles * 3)
    time.sleep(2)
    for k in range(cycles):
        rb = Proc("reader", d, out / f"f7/readerB{k}", "--duration", 30)
        time.sleep(2.5)
        rb.kill9()
    racode = ra.wait(timeout=90)
    wcode = w.term()
    ws, rs = w.summary(), ra.summary()
    proc = (out / "f7/writer/proc-writer.csv").read_text().strip().splitlines()[1:]
    fds = [int(l.split(",")[3]) for l in proc]
    c = counters(rs)
    # Nuotėkis = FD skaičius auga su kiekvienu nužudytu reader'iu. Pabaigoje writer'is atlaisvina flow, todėl
    # FD sumažėja — tai ne nuotėkis. Kriterijus: maksimumas per visą laiką == pirmas matavimas.
    ok = racode == 0 and verify_errors(rs) == 0 and c.get("gap_grains") == 0 and fds and max(fds) == fds[0]
    return result("F7", "PASS" if ok else "FAIL",
                  why=f"{cycles}× reader kill -9: reader A exit={racode} verif={verify_errors(rs)} gap={c.get('gap_grains')}; "
                      f"writer fd pradžia={fds[0] if fds else '?'} max={max(fds) if fds else '?'} pabaiga={fds[-1] if fds else '?'}, "
                      f"late_skips={counters(ws).get('writer_late_skips')}",
                  writer_exit=wcode, writer_fds={"first": fds[0], "max": max(fds), "last": fds[-1]} if fds else None)


def f8(out):
    gpu_mode("gpu")  # vramflow-verify writer / reader — GPU (cuda-linear)
    d = fresh_domain("f8")
    w = Proc("writer", d, out / "f8/writer", "--duration", 50, "--gc-before-create")
    time.sleep(2)
    r = Proc("reader", d, out / "f8/reader", "--duration", 40)
    time.sleep(3)
    hog = subprocess.Popen([BIN, "hog", "--leave-free-mb", "150"], stdout=subprocess.PIPE, text=True, env=ENV)
    hog_line = hog.stdout.readline().strip()
    time.sleep(2)
    big = Proc("writer", d, out / "f8/writer_big", "--duration", 10, "--flows", 40, "--flow-base", 100, "--gc-before-create")
    bcode = big.wait(timeout=60)
    if bcode is None:
        big.term()
    time.sleep(1)
    left = [x for x in leftovers(d) if not x.startswith("4d584c56-0000-4000-8000-000000000000")]
    rcode = r.wait(timeout=60)
    hog.send_signal(signal.SIGTERM)
    hog.wait(timeout=30)
    w.term()
    rs = r.summary()
    clean = bcode is not None and bcode >= 0
    ok = clean and verify_errors(rs) == 0 and rcode == 0
    return result("F8", "PASS" if ok else "FAIL",
                  why=f"hog {hog_line}; 40 papildomų flow'ų writer exit={bcode} (be crash={clean}); veikiantis flow: "
                      f"reader exit={rcode} verif={verify_errors(rs)}; likę katalogai po nesėkmės: {len(left)}",
                  big_writer_exit=bcode, leftovers=left[:10], reader=counters(rs))


def gfx_flow_id(n, k):
    return "4d584c47-0005-4000-8000-%012x" % (n * 8 + k)


def fgfx1(out):
    """gfxgen: N x K sintetinių v210a šaltinių (pace režimas), abu variantai: host/cpu ir cuda-linear/gpu.
    Tikrina: mxl-info mato N*K v210a flow'us, summary be klaidų, commit'ų >= 240 kiekvienam per 5 s (250 laukų)."""
    res = []
    N, K, DUR = 2, 2, 5
    for tag, impl, storage in (("host-cpu", "cpu", "host"), ("cuda-gpu", "gpu", "cuda-linear")):
        tid = f"fgfx1[{tag}]"
        d = fresh_domain(f"fgfx1-{tag}")
        o = pathlib.Path(out) / f"fgfx1-{tag}"
        o.mkdir(parents=True, exist_ok=True)
        err = open(o / "gfxgen.err", "w")
        gpu_mode(impl)
        p = subprocess.Popen([FUNC_BIN, "gfxgen", "--domain", str(d), "--flows", str(N), "--layers", str(K), "--layer-geom",
                              "logo,full", "--mode", "pace", "--impl", impl, "--storage", storage, "--duration", str(DUR),
                              "--out", str(o)], stdout=subprocess.PIPE, stderr=err, env=ENV, text=True)
        time.sleep(DUR / 2)
        seen, bad = 0, []
        for n in range(N):
            for k in range(K):
                r = subprocess.run([MXL_INFO, "-d", str(d), "-f", gfx_flow_id(n, k)], capture_output=True, text=True, env=ENV)
                fd = d / f"{gfx_flow_id(n, k)}.mxl-flow" / "flow_def.json"  # mxl-info nerodo media_type — žiūrim flow aprašą
                if (r.returncode == 0 and "Format: Video" in r.stdout and "50/1" in r.stdout and fd.exists()
                        and json.loads(fd.read_text()).get("media_type") == "video/v210a"):
                    seen += 1
                else:
                    bad.append(gfx_flow_id(n, k))
        try:
            rc = p.wait(timeout=DUR + 30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rc = None
        sf = o / "summary-gfxgen.json"
        s = json.loads(sf.read_text()) if sf.exists() else None
        commits = [f.get("commits", 0) for f in s["flows"]] if s else []
        errors = [f.get("errors", -1) for f in s["flows"]] if s else []
        ok = (rc == 0 and seen == N * K and s is not None and len(commits) == N * K and min(commits) >= 240
              and all(e == 0 for e in errors))
        res.append(result(tid, "PASS" if ok else "FAIL", why=f"exit={rc} v210a flow'ų={seen}/{N * K} commits={commits} klaidos={errors}"
                          + (f" nerasti={bad}" if bad else "")))
    return res

# --- composite (alfa kompozitorius, užduotis 8): fcomp1 / fcomp2 [PF-3] / fcomp3 / fcomp4 (verify) / fcomp5 / fcomp6 -------------------------------
# Variantai C (cpu+host), H (gpu+host), G (gpu+cuda-linear); aplinka FCOMP_VARIANTS=C,G apriboja.
COMP_VARIANTS = [("C", "cpu", "host"), ("H", "gpu", "host"), ("G", "gpu", "cuda-linear")]


# Įėjimų laukimo riba: gen / gfxgen (kiti procesai WSL2) kartais vėluoja > 15 ms (išmatuota: ~1 iš 3000 lauko-flow'ų), todėl
# funkciniuose testuose 30 ms (kaip logo numatytoji); našumo matavimas (užduotis 10) naudoja 15.
COMP_DEADLINE_MS = int(os.environ.get("FCOMP_DEADLINE_MS", "30"))
# V20 (K-T2): funkcijų numatytoji riba — auto 3·P/4 (15 ms prie 50 Hz). Funkciniai testai veikia WSL2, kur gen / GPU procesų
# sutrikimai 17–30 ms nėra reti (pilname paleidime prie 15 ms — 2 tokie indeksai per 65 testus), todėl, kaip kampanijos (K-T2),
# jie nurodo 30 ms aiškiai: FProc prideda --deadline-ms FUNC_DEADLINE_MS gen / logo / enc procesams, kurių testas jos nenurodė.
# Numatytąją laiką tikrinantys testai (izoliacijos vartai, ffmt_logo) nurodo --deadline-ms 0 (auto) patys.
FUNC_DEADLINE_MS = int(os.environ.get("FFUNC_DEADLINE_MS", "30"))
DEADLINE_SUBS = ("gen", "logo", "enc")


def comp_variants():
    want = [x for x in os.environ.get("FCOMP_VARIANTS", "").upper().split(",") if x]
    return [v for v in COMP_VARIANTS if not want or v[0] in want]


def role_flow_id(role, n):
    return "4d584c47-%04d-4000-8000-%012x" % (role, n)


# CPU rinkiniai pagal subkomandą (taskset -c): {"gen": "0-3", "enc": "4-11", …}; tuščia — be susiejimo. Nustato testų
# moduliai (izoliacijos vartai: gamintojai ir tikrinama funkcija — atskiruose branduoliuose, peržiūra 2026-11-13 p. 7b).
CPUSETS = {}


class FProc:
    """vramflow-func subkomanda kaip procesas (stderr į failą, summary-<role>.json out kataloge)."""

    def __init__(self, out, tag, sub, domain, *args):
        self.out = pathlib.Path(out) / tag
        self.out.mkdir(parents=True, exist_ok=True)
        self.sub = sub
        self.err_path = self.out / f"{sub}.err"
        args = [str(a) for a in args]
        if sub in DEADLINE_SUBS and "--deadline-ms" not in args:
            args += ["--deadline-ms", str(FUNC_DEADLINE_MS)]
        cpus = CPUSETS.get(sub)
        pre = ["taskset", "-c", cpus] if cpus else []  # taskset exec'ina programą: tas pats pid (signalai, wait)
        self.p = subprocess.Popen([*pre, FUNC_BIN, sub, "--domain", str(domain), "--out", str(self.out), *args],
                                  stdout=subprocess.DEVNULL, stderr=open(self.err_path, "w"), env=ENV)

    def wait(self, timeout=None):
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def term(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
        try:
            return self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
            return self.p.wait()

    def kill9(self):
        self.p.send_signal(signal.SIGKILL)
        self.p.wait()

    def summary(self):
        f = self.out / f"summary-{self.sub}.json"
        return json.loads(f.read_text()) if f.exists() else None


def wait_flows(domain, ids, timeout=30):
    """Laukia, kol visų flow'ų kataloguose atsiranda flow_def.json. True — visi atsirado."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if all((pathlib.Path(domain) / f"{i}.mxl-flow" / "flow_def.json").exists() for i in ids):
            return True
        time.sleep(0.1)
    return False


def flow_media_type(domain, fid):
    f = pathlib.Path(domain) / f"{fid}.mxl-flow" / "flow_def.json"
    return json.loads(f.read_text()).get("media_type") if f.exists() else None


def flow_head(domain, fid):
    """Flow head indeksas per mxl-info (None, jei neperskaitoma)."""
    r = subprocess.run([MXL_INFO, "-d", str(domain), "-f", fid], capture_output=True, text=True, env=ENV)
    for line in r.stdout.splitlines():
        if "Head index" in line:
            try:
                return int(line.split(":")[1].strip())
            except ValueError:
                return None
    return None


def flow_ctr(s, n):
    return s["flows"][n] if s and len(s["flows"]) > n else {}


def vcommon(impl, storage):
    """Varianto parinktys; kartu nustato GPU šildymą šiam variantui (gpu_mode)."""
    gpu_mode(impl)
    return ["--impl", impl, "--storage", storage]


def fcomp1(out):
    """gen + gfxgen(pace) + composite 2 išėjimai x 2 sluoksniai 10 s: išėjimo flow'ai yra, late_fields == 0."""
    res = []
    N, K, DUR = 2, 2, 10
    for tag, impl, storage in comp_variants():
        tid = f"fcomp1[{tag}]"
        d = fresh_domain(f"fcomp1-{tag}")
        v = vcommon(impl, storage)
        g = FProc(out, f"fcomp1-{tag}", "gen", d, "--flows", N, "--duration", 60, *v)
        x = FProc(out, f"fcomp1-{tag}", "gfxgen", d, "--flows", N, "--layers", K, "--layer-geom", "logo,full", "--mode", "pace",
                  "--duration", 60, *v)
        ids = [role_flow_id(1, n) for n in range(N)] + [role_flow_id(5, n * 8 + k) for n in range(N) for k in range(K)]
        c = None
        seen = types = False
        rc = None
        try:
            if not wait_flows(d, ids):
                res.append(result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                continue
            c = FProc(out, f"fcomp1-{tag}", "composite", d, "--flows", N, "--layers", K, "--layer-xy", "1488,64", "--layer-xy", "0,0",
                      "--opacity", "1.0", "--opacity", "0.5", "--base", "gen", "--out-format", "v210", "--deadline-ms", COMP_DEADLINE_MS,
                      "--duration", DUR, "--warmup", 2, *v)
            out_ids = [role_flow_id(6, n) for n in range(N)]
            seen = wait_flows(d, out_ids, 20)
            types = seen and all(flow_media_type(d, i) == "video/v210" for i in out_ids)
            rc = c.wait(timeout=DUR + 40)
        finally:
            for pr in (c, x, g):
                if pr is not None:
                    pr.term()
        s = c.summary() if c else None
        fl = [flow_ctr(s, n) for n in range(N)]
        late = [f.get("late_fields", 0) for f in fl]
        okc = [f.get("fields_ok", 0) for f in fl]
        ok = (rc == 0 and seen and types and s is not None and all(l == 0 for l in late) and all(o >= 0.9 * DUR * 50 for o in okc))
        res.append(result(tid, "PASS" if ok else "FAIL",
                          why=f"exit={rc} išėjimo flow'ai={seen} v210={types} late_fields={late} fields_ok={okc} "
                              f"proc_ms={(s or {}).get('hist', {}).get('proc_ms')}", composite=s))
    return res


def fcomp2(out):
    """[PF-3] gfxgen užmušamas 2 s ir paleidžiamas iš naujo: composite neišeina, reopened >= 1, išėjimai vėl commit'inami."""
    res = []
    DUR = 18
    for tag, impl, storage in comp_variants():
        tid = f"fcomp2[{tag}]"
        d = fresh_domain(f"fcomp2-{tag}")
        v = vcommon(impl, storage)
        gen_args = ["--flows", 1, "--duration", 80, *v]
        gx_args = ["--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace", "--duration", 80, *v]
        g = FProc(out, f"fcomp2-{tag}", "gen", d, *gen_args)
        x = FProc(out, f"fcomp2-{tag}/a", "gfxgen", d, *gx_args)
        x2 = c = None
        rc = None
        alive_during = False
        head_before = h1 = h2 = None
        try:
            if not wait_flows(d, [role_flow_id(1, 0), role_flow_id(5, 0)]):
                res.append(result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                continue
            c = FProc(out, f"fcomp2-{tag}", "composite", d, "--flows", 1, "--layers", 1, "--layer-xy", "1488,64", "--base", "gen",
                      "--out-format", "v210", "--deadline-ms", COMP_DEADLINE_MS, "--duration", DUR, "--warmup", 1, *v)
            wait_flows(d, [role_flow_id(6, 0)], 20)
            time.sleep(3)
            head_before = flow_head(d, role_flow_id(6, 0))
            x.kill9()
            time.sleep(2)
            alive_during = c.p.poll() is None
            x2 = FProc(out, f"fcomp2-{tag}/b", "gfxgen", d, *gx_args)
            time.sleep(5)
            h1 = flow_head(d, role_flow_id(6, 0))
            time.sleep(2)
            h2 = flow_head(d, role_flow_id(6, 0))
            rc = c.wait(timeout=DUR + 40)
        finally:
            for pr in (c, x2, x, g):
                if pr is not None:
                    pr.term()
        s = c.summary() if c else None
        f0 = flow_ctr(s, 0)
        advancing = h1 is not None and h2 is not None and h2 - h1 >= 80  # 2 s = 100 laukų
        ok = (alive_during and rc in (0, 4) and s is not None and f0.get("reopened", 0) >= 1 and advancing
              and f0.get("fields_ok", 0) > 0)
        res.append(result(tid, "PASS" if ok else "FAIL",
                          why=f"gyvas per outage={alive_during} exit={rc} reopened={f0.get('reopened')} flow_invalid={f0.get('flow_invalid')} "
                              f"missing={f0.get('missing_fields')} head po paleidimo {h1}->{h2} (prieš outage {head_before}) "
                              f"fields_ok={f0.get('fields_ok')}", composite=f0))
    return res


def fcomp3(out):
    """--base none --out-format v210a be gen; neteisingos konfigūracijos atmetamos aiškia klaida (R12, R1)."""
    res = []
    DUR = 8
    for tag, impl, storage in comp_variants():
        tid = f"fcomp3[{tag}]"
        d = fresh_domain(f"fcomp3-{tag}")
        v = vcommon(impl, storage)
        x = FProc(out, f"fcomp3-{tag}", "gfxgen", d, "--flows", 1, "--layers", 2, "--layer-geom", "logo,full", "--mode", "pace",
                  "--duration", 60, *v)
        c = None
        r12rc = r1rc = rc = mt = None
        try:
            if not wait_flows(d, [role_flow_id(5, 0), role_flow_id(5, 1)]):
                res.append(result(tid, "FAIL", why="gfxgen flow'ai neatsirado"))
                continue
            # R12: be bazės tik v210a
            r12 = FProc(out, f"fcomp3-{tag}/r12", "composite", d, "--flows", 1, "--layers", 1, "--layer-xy", "0,0", "--base", "none",
                        "--out-format", "v210", "--duration", 3, *v)
            r12rc = r12.wait(timeout=30)
            # R1: nelyginis y interlaced rastre atmetamas pradžioje
            r1 = FProc(out, f"fcomp3-{tag}/r1", "composite", d, "--flows", 1, "--layers", 1, "--layer-xy", "0,33", "--base", "none",
                       "--out-format", "v210a", "--duration", 3, *v)
            r1rc = r1.wait(timeout=30)
            c = FProc(out, f"fcomp3-{tag}", "composite", d, "--flows", 1, "--layers", 2, "--layer-xy", "1488,64", "--layer-xy", "0,0",
                      "--opacity", "0.8", "--opacity", "1", "--base", "none", "--out-format", "v210a", "--deadline-ms", COMP_DEADLINE_MS, "--duration", DUR,
                      "--warmup", 1, *v)
            seen = wait_flows(d, [role_flow_id(6, 0)], 20)
            mt = flow_media_type(d, role_flow_id(6, 0)) if seen else None
            rc = c.wait(timeout=DUR + 40)
        finally:
            for pr in (c, x):
                if pr is not None:
                    pr.term()
        s = c.summary() if c else None
        f0 = flow_ctr(s, 0)
        ok = (r12rc == 1 and r1rc == 1 and rc == 0 and mt == "video/v210a" and f0.get("late_fields", 0) == 0
              and f0.get("fields_ok", 0) >= 0.9 * DUR * 50)
        res.append(result(tid, "PASS" if ok else "FAIL",
                          why=f"R12 exit={r12rc} R1 exit={r1rc}; be gen: exit={rc} media_type={mt} late_fields={f0.get('late_fields')} "
                              f"fields_ok={f0.get('fields_ok')}", composite=f0))
    return res


def fcomp4(out):
    """verify --mode composite (CPU etalonas): pilna grandinė gen + gfxgen(manual / pace) + composite + verify --every 5, 10 s,
    C / H / G. Tikrasis verify: verification_errors == 0 ir patikrintų > 0; tyčinis neatitikimas (kita --opacity) —
    verification_errors > 0 (įrodo, kad patikra veikia). Abu verify procesai skaito tą patį composite išėjimą."""
    res = []
    N, K, DUR = 1, 2, 6
    MIN_SAMPLES = 20  # 6 s, --every 5: pasiekta >= ~35 (manual, apkrauta) / ~50 (pace)
    xy = ["--layer-xy", "1488,64", "--layer-xy", "0,0"]
    for tag, impl, storage in comp_variants():
        for gm, ofmt in (("manual", "v210a"), ("manual", "v210"), ("pace", "v210a"), ("pace", "v210")):
            tid = f"fcomp4[{tag}/{gm}/{ofmt}]"
            sub = f"fcomp4-{tag}-{gm}-{ofmt}"
            d = fresh_domain(sub)
            v = vcommon(impl, storage)
            g = FProc(out, sub, "gen", d, "--flows", N, "--duration", 80, *v)
            x = FProc(out, sub, "gfxgen", d, "--flows", N, "--layers", K, "--layer-geom", "logo,full", "--mode", gm,
                      "--duration", 80, *v)
            ids = [role_flow_id(1, n) for n in range(N)] + [role_flow_id(5, n * 8 + k) for n in range(N) for k in range(K)]
            c = va = vb = None
            try:
                if not wait_flows(d, ids):
                    res.append(result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                    continue
                c = FProc(out, sub, "composite", d, "--flows", N, "--layers", K, *xy, "--opacity", "0.7", "--opacity", "0.5",
                          "--base", "gen", "--out-format", ofmt, "--deadline-ms", COMP_DEADLINE_MS, "--duration", DUR + 14,
                          "--warmup", 2, *v)
                if not wait_flows(d, [role_flow_id(6, n) for n in range(N)], 20):
                    res.append(result(tid, "FAIL", why="composite išėjimo flow'ai neatsirado"))
                    continue
                vargs = ["--mode", "composite", "--flows", N, "--layers", K, "--layer-geom", "logo,full", "--gfx-mode", gm,
                         "--base", "gen", "--out-format", ofmt, "--every", 5, "--duration", DUR, "--warmup", 1, *xy, *v]
                va = FProc(out, f"{sub}/ok", "verify", d, *vargs, "--opacity", "0.7", "--opacity", "0.5")
                vb = FProc(out, f"{sub}/bad", "verify", d, *vargs, "--opacity", "0.7", "--opacity", "0.3")
                ra, rb = va.wait(timeout=DUR + 60), vb.wait(timeout=DUR + 60)
            finally:
                for pr in (va, vb, c, x, g):
                    if pr is not None:
                        pr.term()

            def vsum(pr):
                f = pr.out / "summary-verify-composite.json"
                return json.loads(f.read_text()) if f.exists() else None

            sa, sb = vsum(va), vsum(vb)
            pa, pb = (sa or {}).get("proc", {}), (sb or {}).get("proc", {})
            ok = (sa is not None and sb is not None and pa.get("verification_errors", 1) == 0 and pa.get("samples_total", 0) >= MIN_SAMPLES
                  and pb.get("verification_errors", 0) > 0 and pb.get("samples_total", 0) > 0)
            res.append(result(tid, "PASS" if ok else "FAIL",
                              why=f"exit ok={ra} bad={rb} ok.verification_errors={pa.get('verification_errors')} ok.samples={pa.get('samples_total')} "
                                  f"ok.capacity_errors={pa.get('capacity_errors')} check_dropped={((sa or {}).get('flows') or [{}])[0].get('check_dropped')} "
                                  f"bad.verification_errors={pb.get('verification_errors')} "
                                  f"expect_ms={((sa or {}).get('hist') or {}).get('expect_ms')}", ok=sa, bad=sb))
    return res


def fcomp5(out):
    """Grandinė: composite(--base none, v210a) išėjimas = sluoksnis antram composite su gen baze (--layer-flow)."""
    res = []
    DUR = 10
    for tag, impl, storage in comp_variants():
        tid = f"fcomp5[{tag}]"
        d = fresh_domain(f"fcomp5-{tag}")
        v = vcommon(impl, storage)
        # gen n=1 antram composite (flow-base 1); gfxgen n=0 pirmam
        g = FProc(out, f"fcomp5-{tag}", "gen", d, "--flows", 1, "--flow-base", 1, "--duration", 80, *v)
        x = FProc(out, f"fcomp5-{tag}", "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace",
                  "--duration", 80, *v)
        c1 = c2 = None
        rc1 = rc2 = None
        try:
            if not wait_flows(d, [role_flow_id(1, 1), role_flow_id(5, 0)]):
                res.append(result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                continue
            c1 = FProc(out, f"fcomp5-{tag}/c1", "composite", d, "--flows", 1, "--layers", 1, "--layer-xy", "1488,64", "--base", "none",
                       "--out-format", "v210a", "--deadline-ms", COMP_DEADLINE_MS, "--duration", DUR + 12, "--warmup", 1, *v)
            if not wait_flows(d, [role_flow_id(6, 0)], 30):
                res.append(result(tid, "FAIL", why="pirmo composite išėjimas neatsirado"))
                continue
            c2 = FProc(out, f"fcomp5-{tag}/c2", "composite", d, "--flows", 1, "--flow-base", 1, "--layers", 1, "--layer-xy", "0,0",
                       "--layer-flow", "0:" + role_flow_id(6, 0), "--base", "gen", "--out-format", "v210", "--deadline-ms", COMP_DEADLINE_MS,
                       "--duration", DUR, "--warmup", 1, *v)
            rc2 = c2.wait(timeout=DUR + 40)
            rc1 = c1.wait(timeout=40)
        finally:
            for pr in (c2, c1, x, g):
                if pr is not None:
                    pr.term()
        s1, s2 = c1.summary() if c1 else None, c2.summary() if c2 else None
        f1, f2 = flow_ctr(s1, 0), flow_ctr(s2, 0)
        ok = (rc1 == 0 and rc2 == 0 and f1.get("late_fields", 0) == 0 and f2.get("late_fields", 0) == 0
              and f1.get("fields_ok", 0) >= 0.9 * DUR * 50 and f2.get("fields_ok", 0) >= 0.9 * DUR * 50)
        res.append(result(tid, "PASS" if ok else "FAIL",
                          why=f"c1 exit={rc1} late={f1.get('late_fields')} ok={f1.get('fields_ok')}; c2 exit={rc2} "
                              f"late={f2.get('late_fields')} ok={f2.get('fields_ok')}", c1=f1, c2=f2))
    return res


def fcli1(out):
    """CR2 #1: --layer-flow K:UUID argumentų tikrinimas (exit 1 prieš bet kokį MXL darbą)."""
    u = "5fbec3b8-0000-4000-8000-000000000001"
    bad = {"0x": f"0x:{u}", "neigiamas": f"-1:{u}", "per didelis": f"1:{u}", "be K": f":{u}", "be UUID": "0:",
           "be dvitaškio": "0", "dublikatas": None}
    d = fresh_domain("fcli1")
    res = []
    for name, val in bad.items():
        lf = ["--layer-flow", f"0:{u}", "--layer-flow", f"0:{u}"] if val is None else ["--layer-flow", val]
        r = subprocess.run([FUNC_BIN, "composite", "--domain", str(d), "--out", str(out / f"fcli1-{name}"), "--flows", "1", "--layers", "1",
                            "--layer-xy", "0,0", *lf], capture_output=True, text=True, timeout=30)
        res.append(result(f"fcli1[{name}]", "PASS" if r.returncode == 1 else "FAIL", why=f"exit {r.returncode}"))
    ok = subprocess.run([FUNC_BIN, "composite", "--domain", str(d), "--out", str(out / "fcli1-ok"), "--flows", "1", "--layers", "1",
                         "--layer-xy", "0,0", "--layer-flow", f"0:{u}", "--duration", "1", "--input-wait-ms", "2000"], capture_output=True, text=True, timeout=60)
    # galiojantis argumentas praeina parsinimą (toliau gali baigtis kitu kodu — įėjimų nėra), bet ne exit 1 dėl --layer-flow
    res.append(result("fcli1[galiojantis]", "PASS" if "tikimasi K:UUID" not in ok.stderr + ok.stdout and "daugiau nei kartą" not in ok.stderr else "FAIL",
                      why=f"exit {ok.returncode}"))
    return res


KW_SUMMARY_RE = r"\[keepwarm\] baigta: (\d+) užpildymų po (\d+) MB kas ([\d.]+) ms, ([\d.]+) s"


def fkeepwarm(out):
    """keepwarm (docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md §6): --duration 2 — exit 0, viena santraukos eilutė su
    tuo, ką padarė (užpildymų ≈ trukmė / intervalas); --duration 0 — veikia, kol sustabdomas, ir kol veikia nespausdina nieko;
    SIGTERM — exit 0 ir santrauka. Be GPU (nėra nvidia-smi) — SKIP."""
    import re
    if not shutil.which("nvidia-smi") and not pathlib.Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        return [result("fkeepwarm", "SKIP", why="nėra GPU")]
    res = []
    r = subprocess.run([FUNC_BIN, "keepwarm", "--duration", "2", "--size-mb", "16", "--interval-ms", "5"], capture_output=True,
                       text=True, timeout=60, env=ENV)
    m = re.search(KW_SUMMARY_RE, r.stderr)
    fills = int(m.group(1)) if m else 0
    ok = r.returncode == 0 and m is not None and m.group(2) == "16" and 200 <= fills <= 410 and r.stdout == ""
    res.append(result("fkeepwarm[duration 2]", "PASS" if ok else "FAIL",
                      why=f"exit {r.returncode} užpildymų {fills} stderr={r.stderr.strip()[-200:]!r}"))
    errp = out / "fkeepwarm.err"
    p = subprocess.Popen([FUNC_BIN, "keepwarm", "--duration", "0"], stdout=subprocess.DEVNULL, stderr=open(errp, "w"), env=ENV)
    time.sleep(3.0)
    alive = p.poll() is None
    quiet = errp.read_text(errors="replace") == ""
    p.send_signal(signal.SIGTERM)
    try:
        rc = p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = p.wait()
    txt = errp.read_text(errors="replace")
    m = re.search(KW_SUMMARY_RE, txt)
    ok = alive and quiet and rc == 0 and m is not None and int(m.group(1)) > 0 and m.group(2) == "64" and m.group(3) == "5.0"
    res.append(result("fkeepwarm[duration 0 + SIGTERM]", "PASS" if ok else "FAIL",
                      why=f"gyvas po 3 s={alive} tylus={quiet} exit {rc} stderr={txt.strip()[-200:]!r}"))
    # harness'o taisyklė: GPU variantas (vcommon gpu …) — keepwarm veikia (jei neišjungtas), CPU variantas — ne; rezultatas
    # įrašo būseną (laukas keepwarm)
    def running():
        r = subprocess.run(["pgrep", "-f", f"{FUNC_BIN} keepwarm"], capture_output=True, text=True)
        return r.stdout.split()
    vcommon("gpu", "cuda-linear")
    g_on, g_procs, g_rec = KW.on(), running(), result("fkeepwarm[harness G]", "INFO")
    vcommon("gpu", "host")
    h_on = KW.on()
    vcommon("cpu", "host")
    c_on, c_procs, c_rec = KW.on(), running(), result("fkeepwarm[harness C]", "INFO")
    want = keepwarm.ENABLED
    ok = (g_on == want and h_on == want and (len(g_procs) == 1) == want and g_rec.get("keepwarm") is want and not c_on
          and not c_procs and c_rec.get("keepwarm") is False)
    res.append(result("fkeepwarm[harness]", "PASS" if ok else "FAIL",
                      why=f"įjungta={want}: G on={g_on} procesų={len(g_procs)} įraše={g_rec.get('keepwarm')}; H on={h_on}; "
                          f"C on={c_on} procesų={len(c_procs)} įraše={c_rec.get('keepwarm')}"))
    res.append(keepwarm_orphan(out))
    return res


def keepwarm_orphan(out):
    """Harness'as nužudomas SIGKILL (timeout, CI) — keepwarm (--duration 0) neturi likti veikti: po 3 s jo procesas baigtas
    (keepwarm --parent-pid + PR_SET_PDEATHSIG; SIGKILL atveju harness'o handler'iai nesuveikia)."""
    perf = pathlib.Path(__file__).resolve().parents[1] / "perf"
    code = (f"import sys, time; sys.path.insert(0, {str(perf)!r}); import keepwarm; k = keepwarm.KeepWarm({FUNC_BIN!r}); "
            f"k.start(); print(k.p.pid if k.p else -1, flush=True); time.sleep(120)")
    h = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=open(out / "kw-orphan.err", "w"), env=ENV,
                         text=True)
    try:
        line = h.stdout.readline().strip()
        kpid = int(line) if line.lstrip("-").isdigit() else -1
    finally:
        h.send_signal(signal.SIGKILL)
        h.wait()
    if kpid <= 0:
        return result("fkeepwarm[harness SIGKILL]", "FAIL", why=f"keepwarm nepasileido (pid {line!r})")
    time.sleep(3.0)

    def gone(pid):
        try:
            st = pathlib.Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return True
        return "\nState:\tZ" in st  # zombis — baigėsi, laukia, kol surinks init
    ok = gone(kpid)
    if not ok:  # nepalikti šiukšlių kitiems testams
        try:
            os.kill(kpid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return result("fkeepwarm[harness SIGKILL]", "PASS" if ok else "FAIL",
                  why=f"harness'as nužudytas SIGKILL; keepwarm pid {kpid} po 3 s {'baigtas' if ok else 'VIS DAR VEIKIA'}")


def fcomp6(out):
    """verify --mode composite su N = 8 išėjimų (vienas gfxgen procesas, logo sluoksnis), C / H / G, pace / manual. Regresija:
    capacity G N >= 2 — gfxgen cudaMemcpy H2D (pageable) grįždavo prieš DMA pabaigą, biblioteka konvertuodavo dalinai įkeltą
    kadrą → n >= 1 lyginiai laukai logo srityje neteisingi. Kiekvienas flow: patikrinta >= MIN ir mismatch_fields == 0."""
    res = []
    N, K, DUR = 8, 1, 6  # be pataisos išmatuota: N = 4 klaida pagaunama ~1 iš 4 paleidimų, N = 8 — dažniau (ctest [upload] — deterministiškai)
    MIN_SAMPLES = 10  # vienam flow: 6 s, --every 5 → ~60
    xy = ["--layer-xy", "1488,64"]
    for tag, impl, storage in comp_variants():
        for gm in ("pace", "manual"):
            tid = f"fcomp6[{tag}/{gm}]"
            sub = f"fcomp6-{tag}-{gm}"
            d = fresh_domain(sub)
            v = vcommon(impl, storage)
            g = FProc(out, sub, "gen", d, "--flows", N, "--duration", 80, *v)
            x = FProc(out, sub, "gfxgen", d, "--flows", N, "--layers", K, "--layer-geom", "logo", "--mode", gm, "--duration", 80, *v)
            ids = [role_flow_id(1, n) for n in range(N)] + [role_flow_id(5, n * 8) for n in range(N)]
            c = va = None
            try:
                if not wait_flows(d, ids):
                    res.append(result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                    continue
                c = FProc(out, sub, "composite", d, "--flows", N, "--layers", K, *xy, "--opacity", "1.0", "--base", "gen",
                          "--out-format", "v210", "--deadline-ms", COMP_DEADLINE_MS, "--duration", DUR + 14, "--warmup", 2, *v)
                if not wait_flows(d, [role_flow_id(6, n) for n in range(N)], 20):
                    res.append(result(tid, "FAIL", why="composite išėjimo flow'ai neatsirado"))
                    continue
                va = FProc(out, sub, "verify", d, "--mode", "composite", "--flows", N, "--layers", K, "--layer-geom", "logo",
                           "--gfx-mode", gm, "--base", "gen", "--out-format", "v210", "--every", 5, "--duration", DUR, "--warmup", 1,
                           *xy, "--opacity", "1.0", *v)
                ra = va.wait(timeout=DUR + 60)
            finally:
                for pr in (va, c, x, g):
                    if pr is not None:
                        pr.term()
            f = va.out / "summary-verify-composite.json" if va is not None else None
            sa = json.loads(f.read_text()) if f is not None and f.exists() else None
            flows = (sa or {}).get("flows") or []
            per = [(fl.get("samples", 0), fl.get("mismatch_fields", 0)) for fl in flows]
            ok = (sa is not None and len(per) == N and all(sm >= MIN_SAMPLES and mm == 0 for sm, mm in per)
                  and (sa.get("proc") or {}).get("verification_errors", 1) == 0)
            res.append(result(tid, "PASS" if ok else "FAIL", why=f"exit {ra} (samples, mismatch_fields) per flow={per}", summary=sa))
    return res


# --- P2: formatui neutralūs flow'ai (planas §P2): kiekviena funkcija 1080i50, 1080p25, 1080p50 ---------------------------
# Aplinka FFMT_FORMATS=1080i50,1080p25 apriboja formatus; FCOMP_VARIANTS — variantus (kaip composite testams).
FMT_DEFAULT = ["1080i50", "1080p25", "1080p50"]
FMT_GRAIN_HZ = {"1080i50": 50, "1080p25": 25, "1080p50": 50, "720p50": 50, "1080i59.94": 59.94, "1080p59.94": 59.94,
                "1080p29.97": 29.97, "720p59.94": 59.94}
FMT_FPS = {"1080i50": 25, "1080p25": 25, "1080p50": 50, "720p50": 50, "1080i59.94": 29.97, "1080p59.94": 59.94,
           "1080p29.97": 29.97, "720p59.94": 59.94}
FMT_DUR = int(os.environ.get("FFMT_DURATION", "20"))


def fmt_list():
    want = [x for x in os.environ.get("FFMT_FORMATS", "").split(",") if x]
    return want or FMT_DEFAULT


def vimpl(impl, storage):
    """verify varianto argumentai: host flow'ams — be CUDA (kaip capacity.py)."""
    return ["--impl", "cpu" if (impl == "cpu" or storage == "host") else "gpu", "--storage", storage]


def vsummary(pr, role):
    f = pr.out / f"summary-{role}.json" if pr is not None else None
    return json.loads(f.read_text()) if f is not None and f.exists() else None


LOGO_LOSS_KEYS = ("late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips")


def logo_loss_indices(pr):
    """logo stderr eilutės „praradimas: flow n, indeksas i (raktas)“ → {n: [i, …]} (raktai — LOGO_LOSS_KEYS)."""
    import re
    rx = re.compile(r"\[logo\] praradimas: flow (\d+), indeksas (\d+) \((\w+)\)")
    res = {}
    try:
        text = pr.err_path.read_text(errors="replace") if pr is not None else ""
    except OSError:
        text = ""
    for m in rx.finditer(text):
        if m.group(3) in LOGO_LOSS_KEYS:
            res.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    return res


def gen_upstream_indices(pr):
    """gen stderr: vėlūs commit'ai ir praleisti indeksai → aibė indeksų, kurių žemupys su ta pačia riba nesulaukė."""
    import re
    try:
        text = pr.err_path.read_text(errors="replace") if pr is not None else ""
    except OSError:
        text = ""
    up = {int(m.group(1)) for m in re.finditer(r"\[gen\] vėlus commit: indeksas (\d+)", text)}
    for m in re.finditer(r"\[gen\] praleista: indeksai (\d+)…(\d+)", text):
        up.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return up


def ffmt_logo(out, fmts=None, variants=None, tag_prefix="ffmt_logo"):
    """gen | logo | verify --mode logo kiekvienam formatui ir variantui C/H/G, FFMT_DURATION s: logo be praradimų
    (late/lapped/too_late == 0, fields_ok >= 90 % laukų), verify 0 teisingumo klaidų ir > 0 imčių."""
    res = []
    N, D = 2, FMT_DUR
    for fmt in fmts or fmt_list():
        hz = FMT_GRAIN_HZ[fmt]
        for tag, impl, storage in variants or comp_variants():
            tid = f"{tag_prefix}[{fmt}/{tag}]"
            sub = f"ffmt-logo-{fmt}-{tag}"
            d = fresh_domain(sub)
            v = vcommon(impl, storage) + ["--format", fmt]
            g = FProc(out, sub, "gen", d, "--flows", N, "--deadline-ms", 0, "--duration", D + 40, *v)  # numatytoji riba (V20)
            lg = vf = None
            try:
                if not wait_flows(d, [role_flow_id(1, n) for n in range(N)]):
                    res.append(result(tid, "FAIL", why="gen flow'ai neatsirado"))
                    continue
                lg = FProc(out, sub, "logo", d, "--flows", N, "--deadline-ms", 0, "--duration", D + 8, "--warmup", 2, *v)
                if not wait_flows(d, [role_flow_id(2, n) for n in range(N)], 20):
                    res.append(result(tid, "FAIL", why="logo išėjimo flow'ai neatsirado"))
                    continue
                vf = FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 5, "--duration", D, "--warmup", 2,
                           *vimpl(impl, storage), "--format", fmt)
                rv = vf.wait(timeout=D + 60)
                rl = lg.wait(timeout=D + 60)
            finally:
                for pr in (vf, lg, g):
                    if pr is not None:
                        pr.term()
            sl, sv = vsummary(lg, "logo"), vsummary(vf, "verify-logo")
            fl = (sl or {}).get("flows") or []
            lost_f = [sum(f.get(k, 0) for k in LOGO_LOSS_KEYS) for f in fl]
            lost = sum(lost_f)
            # Priskyrimas (peržiūra 2026-11-13 p. 7c — per flow'ą ir indeksą, ne skaičiais): kiekvienas logo prarastas indeksas
            # (stderr eilutė „praradimas: flow n, indeksas i“) turi būti tas, kurį pats gen commit'ino vėliau nei riba („vėlus
            # commit: indeksas i“) ar praleido („praleista: indeksai a…b“). Visos logo praradimų eilutės turi sutapti su jo
            # skaitikliais (kitaip — nepriskirta). Be aukštupio sutrikimų — 0 praradimų.
            up_idx = gen_upstream_indices(g)
            mine = logo_loss_indices(lg)
            unattr = {n: sorted(set(ix) - up_idx)[:5] for n, ix in mine.items() if set(ix) - up_idx}
            lines_match = all(len(mine.get(n, [])) == lost_f[n] for n in range(len(lost_f)))
            budget = len(up_idx)
            okf = [f.get("fields_ok", 0) for f in fl]
            pv = (sv or {}).get("proc") or {}
            # V20: numatytoji riba D = 3·P/4 (P — grain'o periodas): 25p — 30 ms (nepakitusi), 50 Hz — 15 ms
            dl, want_dl = ((sl or {}).get("proc") or {}).get("deadline_us"), 0.75 * 1e6 / hz
            attributed = lines_match and not unattr
            exit_ok = rl == 0 or (rl == 4 and lost > 0 and attributed)  # griežtas exit 4 — tik kai KIEKVIENAS praradimas — aukštupio
            ok = (sl is not None and sv is not None and exit_ok and attributed and len(okf) == N and min(okf) >= 0.9 * D * hz - budget
                  and pv.get("verification_errors", 1) == 0 and pv.get("samples_total", 0) > 0 and sl.get("format") == fmt
                  and dl is not None and abs(dl - want_dl) <= 1)
            res.append(result(tid, "PASS" if ok else "FAIL",
                              why=f"deadline_us={dl} (laukta {want_dl:.0f}) logo exit={rl} praradimai={lost_f} (aukštupio indeksų "
                                  f"{budget}; nepriskirti {unattr or '-'}; eilutės == skaitikliai {lines_match}) fields_ok={okf} "
                                  f"(>= {0.9 * D * hz:.0f}) verify exit={rv} "
                                  f"verification_errors={pv.get('verification_errors')} samples={pv.get('samples_total')} "
                                  f"capacity_errors={pv.get('capacity_errors')} proc_ms={((sl or {}).get('hist') or {}).get('proc_ms')}",
                              logo=sl, verify=sv))
    return res


def ffmt_logo720(out):
    """720p50 su numatytąja logotipo vieta (be --logo-x / --logo-y): vieta skaičiuojama nuo rastro (viršutinis dešinys
    kampas, ta pati 48 px paraštė kaip 1080) — gen | logo | verify, C ir G."""
    return ffmt_logo(out, ["720p50"], [v for v in comp_variants() if v[0] in ("C", "G")], "ffmt_logo720")


def ffmt_comp(out):
    """gen | gfxgen(pace) | composite | verify --mode composite kiekvienam formatui ir variantui: 0 teisingumo klaidų,
    late_fields == 0, o composite laukia savo grain'o ritmu: early_wakes == 0 (jei laukiama 50/1 laikrodžiu, 25p tick'as
    pabunda prieš grain'o laiką ir laukia įėjimo beveik visą grain'ą — išmatuota wait_ms p50 39 ms)."""
    res = []
    N, K, D = 1, 2, max(6, FMT_DUR // 2)
    xy = ["--layer-xy", "1488,64", "--layer-xy", "0,0"]
    for fmt in fmt_list():
        for tag, impl, storage in comp_variants():
            tid = f"ffmt_comp[{fmt}/{tag}]"
            sub = f"ffmt-comp-{fmt}-{tag}"
            d = fresh_domain(sub)
            v = vcommon(impl, storage) + ["--format", fmt]
            g = FProc(out, sub, "gen", d, "--flows", N, "--duration", D + 60, *v)
            x = FProc(out, sub, "gfxgen", d, "--flows", N, "--layers", K, "--layer-geom", "logo,full", "--mode", "pace",
                      "--duration", D + 60, *v)
            ids = [role_flow_id(1, n) for n in range(N)] + [role_flow_id(5, n * 8 + k) for n in range(N) for k in range(K)]
            c = va = None
            rc = rv = None
            try:
                if not wait_flows(d, ids):
                    res.append(result(tid, "FAIL", why="gen / gfxgen flow'ai neatsirado"))
                    continue
                c = FProc(out, sub, "composite", d, "--flows", N, "--layers", K, *xy, "--opacity", "0.7", "--opacity", "0.5",
                          "--base", "gen", "--out-format", "v210", "--deadline-ms", COMP_DEADLINE_MS, "--duration", D + 14,
                          "--warmup", 2, *v)
                if not wait_flows(d, [role_flow_id(6, n) for n in range(N)], 20):
                    res.append(result(tid, "FAIL", why="composite išėjimo flow'ai neatsirado"))
                    continue
                va = FProc(out, sub, "verify", d, "--mode", "composite", "--flows", N, "--layers", K, "--layer-geom", "logo,full",
                           "--gfx-mode", "pace", "--base", "gen", "--out-format", "v210", "--every", 5, "--duration", D, "--warmup", 1,
                           *xy, "--opacity", "0.7", "--opacity", "0.5", *v)
                rv = va.wait(timeout=D + 60)
                rc = c.wait(timeout=D + 60)
            finally:
                for pr in (va, c, x, g):
                    if pr is not None:
                        pr.term()
            sc, sv = vsummary(c, "composite"), vsummary(va, "verify-composite")
            late = sum(f.get("late_fields", 0) for f in (sc or {}).get("flows") or [])
            pv = (sv or {}).get("proc") or {}
            wait_p50 = (((sc or {}).get("hist") or {}).get("wait_ms") or {}).get("p50_ms")
            early = ((sc or {}).get("proc") or {}).get("early_wakes", 0)
            ok = (sc is not None and sv is not None and rc == 0 and late == 0 and pv.get("verification_errors", 1) == 0
                  and pv.get("samples_total", 0) > 0 and early == 0)
            res.append(result(tid, "PASS" if ok else "FAIL",
                              why=f"composite exit={rc} late_fields={late} early_wakes={early} wait_ms p50={wait_p50} verify exit={rv} "
                                  f"verification_errors={pv.get('verification_errors')} samples={pv.get('samples_total')}",
                              composite=sc, verify=sv))
    return res


def ts_totals(path):
    try:
        return json.loads(pathlib.Path(path).read_text()).get("totals", {})
    except (OSError, ValueError):
        return {}


def ffmt_enc(out):
    """gen(+audio) | enc | ts_sink kiekvienam formatui: C (cpu + x264) ir G (gpu + nvenc). enc be praradimų, kadrų >= 90 %,
    TS be sync / CC klaidų, video PES > 0."""
    res = []
    N, D = 1, max(8, FMT_DUR // 2)
    for fmt in fmt_list():
        for tag, impl, storage, enc in (("C", "cpu", "host", "x264"), ("G", "gpu", "cuda-linear", "nvenc")):
            if tag not in [t for t, _, _ in comp_variants()]:
                continue
            tid = f"ffmt_enc[{fmt}/{tag}]"
            sub = f"ffmt-enc-{fmt}-{tag}"
            d = fresh_domain(sub)
            port = 23000 + 10 * len(res)
            v = vcommon(impl, storage) + ["--format", fmt]
            g = FProc(out, sub, "gen", d, "--flows", N, "--audio", 1, "--duration", D + 30, *v)
            sink_json = pathlib.Path(out) / sub / "sink.json"
            sink = e = None
            re_ = None
            try:
                if not wait_flows(d, [role_flow_id(1, n) for n in range(N)]):
                    res.append(result(tid, "FAIL", why="gen flow'ai neatsirado"))
                    continue
                sink = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "perf/ts_sink.py"),
                                         "--port-base", str(port), "--flows", str(N), "--seconds", str(D + 2), "--out", str(sink_json)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ea = ["--encoder", enc, "--audio", 1, "--port-base", port] + (["--x264-preset", "ultrafast"] if enc == "x264" else [])
                e = FProc(out, sub, "enc", d, "--flows", N, "--duration", D, "--warmup", 2, *ea, *v)
                re_ = e.wait(timeout=D + 60)
                sink.wait(timeout=D + 30)
            finally:
                if sink is not None and sink.poll() is None:
                    sink.kill()
                for pr in (e, g):
                    if pr is not None:
                        pr.term()
            se = vsummary(e, "enc")
            fl = ((se or {}).get("flows") or [{}])[0]
            frames = fl.get("frames_encoded", 0)
            tt = ts_totals(sink_json)
            want = 0.9 * (D - 2) * FMT_FPS[fmt]
            ok = (se is not None and re_ == 0 and frames >= want and tt.get("sync_errors", 1) == 0 and tt.get("cc_errors", 1) == 0
                  and tt.get("video_pes", 0) > 0 and se.get("format") == fmt)
            res.append(result(tid, "PASS" if ok else "FAIL",
                              why=f"enc exit={re_} frames_encoded={frames} (>= {want:.0f}) ts={tt} fatal={(se or {}).get('fatal')}", enc=se))
    return res


# Chromos riba gen | enc | dec grandinei (Cb/Cr PSNR prieš gen etaloną, 10 bitų): teisinga chromos padėtis — virš, maišyti
# laukai (kita padėtis) — žemiau: 1080i50 NVENC 18 Mbit/s — field 44,9 dB min, auto 36,8 dB (docs/rezultatai/2026-11-11-p2-formatai.md).
# x264 ultrafast (C variantas, realiu laiku) — triukšmingesnis (min 39,1 dB 1080p50), todėl jam riba 37 dB.
DEC_MIN_CHROMA_DB = {"nvenc": float(os.environ.get("FFMT_DEC_MIN_CHROMA_DB", "41")), "x264": 37.0}


def dec_siting(fmt, enc):
    """Mūsų NVENC srautai interlaced formatui žymimi progresyviais (lauko kodavimo nėra), bet nešioja lauko chromą."""
    return ["--chroma-siting", "field"] if fmt.startswith("1080i") and enc == "nvenc" else []


def ffmt_dec(out, fmts=None, cases=None, prefix="ffmt_dec", siting=None, expect_low=False):
    """gen | enc | dec | verify --mode dec kiekvienam formatui: C (x264 + programinis dekoderis) ir G (nvenc + NVDEC).
    dec be praradimų po paleidimo, verify: 0 teisingumo klaidų (lauko tvarka interlaced formatams, chroma prieš gen etaloną
    >= DEC_MIN_CHROMA_DB), imčių > 0. expect_low — neigiamas atvejis: chroma_low > 0 (patikra skiria padėtis)."""
    res = []
    N, D = 1, max(10, FMT_DUR // 2)
    for fmt in fmts or fmt_list():
        for tag, impl, storage, enc in cases or (("C", "cpu", "host", "x264"), ("G", "gpu", "cuda-linear", "nvenc")):
            if tag not in [t for t, _, _ in comp_variants()]:
                continue
            tid = f"{prefix}[{fmt}/{tag}]"
            sub = f"{prefix.replace('_', '-')}-{fmt}-{tag}"
            sit = dec_siting(fmt, enc) if siting is None else siting
            d = fresh_domain(sub)
            port = 24000 + 10 * len(res)
            v = vcommon(impl, storage) + ["--format", fmt]
            g = FProc(out, sub, "gen", d, "--flows", N, "--duration", D + 60, *v)
            dc = e = vf = None
            rd = rv = None
            try:
                if not wait_flows(d, [role_flow_id(1, n) for n in range(N)]):
                    res.append(result(tid, "FAIL", why="gen flow'ai neatsirado"))
                    continue
                dc = FProc(out, sub, "dec", d, "--flows", N, "--port-base", port, "--duration", D + 14, "--warmup", 6, *sit, *v)
                time.sleep(1)
                # aukšta sparta (H.264 4.0 riba 20 Mbit/s): kodavimo triukšmas mažas, chromos padėties klaida (laukų maišymas) aiškiai matoma
                ea = (["--encoder", enc, "--port-base", port, "--vbitrate", 18_000_000, "--muxrate", 22_000_000]
                      + (["--x264-preset", "ultrafast"] if enc == "x264" else []))
                e = FProc(out, sub, "enc", d, "--flows", N, "--duration", D + 16, *ea, *v)
                if not wait_flows(d, [role_flow_id(4, n) for n in range(N)], 20):
                    res.append(result(tid, "FAIL", why="dec išėjimo flow'ai neatsirado"))
                    continue
                vf = FProc(out, sub, "verify", d, "--mode", "dec", "--flows", N, "--every", 5, "--duration", D, "--warmup", 6,
                           "--min-chroma-db", DEC_MIN_CHROMA_DB[enc], *vimpl(impl, storage), "--format", fmt)
                rv = vf.wait(timeout=D + 60)
                rd = dc.wait(timeout=D + 60)
            finally:
                for pr in (vf, e, dc, g):
                    if pr is not None:
                        pr.term()
            sd, sv = vsummary(dc, "dec"), vsummary(vf, "verify-dec")
            fl = ((sd or {}).get("flows") or [{}])[0]
            pv = (sv or {}).get("proc") or {}
            vfl = ((sv or {}).get("flows") or [{}])[0]
            low = vfl.get("chroma_low", 0)
            cs = vfl.get("chroma_samples", 0)
            cmin = vfl.get("chroma_psnr_min_x100", 0) / 100
            cavg = vfl.get("chroma_psnr_sum_x100", 0) / 100 / cs if cs else 0
            if expect_low:
                ok = sd is not None and sv is not None and cs > 0 and low > 0
            else:
                ok = (sd is not None and sv is not None and fl.get("fields_ok", 0) > 0 and pv.get("verification_errors", 1) == 0
                      and pv.get("samples_total", 0) > 0 and fl.get("decode_errors", 0) == 0 and sd.get("format") == fmt and cs > 0)
            res.append(result(tid, "PASS" if ok else "FAIL",
                              why=f"chroma_siting={sit[1] if sit else 'auto'} chroma PSNR min={cmin:.2f} vid={cavg:.2f} dB "
                                  f"(riba {DEC_MIN_CHROMA_DB[enc]}) chroma_low={low}/{cs} "
                                  f"dec exit={rd} fields_ok={fl.get('fields_ok')} underrun={fl.get('underrun_fields')} "
                                  f"decode_errors={fl.get('decode_errors')} flag_mismatch={fl.get('flag_mismatch')} verify exit={rv} "
                                  f"verification_errors={pv.get('verification_errors')} samples={pv.get('samples_total')} "
                                  f"parity_mismatch={vfl.get('parity_mismatch')} seq_jump={vfl.get('seq_jump')}", dec=sd, verify=sv))
    return res


def ffmt_dec_chroma(out):
    """Regresija (peržiūros pastaba 2): gen | enc nvenc | dec 1080i50 — NVENC kadrus žymi progresyviais, nors turinys —
    supinti laukai su lauko chroma. Su --chroma-siting field chroma >= riba; su auto (kadro žymė → kadro chroma) — chroma_low
    > 0, t. y. patikra aptinka laukų maišymą."""
    g = [("G", "gpu", "cuda-linear", "nvenc")]
    return (ffmt_dec(out, ["1080i50"], g, "ffmt_dec_chroma_field", ["--chroma-siting", "field"])
            + ffmt_dec(out, ["1080i50"], g, "ffmt_dec_chroma_auto", [], expect_low=True))


def ffmt_neg(out):
    """Neigiamas: logo su 720p50 įėjimu ir logotipu už ribų (X 1488 + 384 > 1280) → exit != 0 su aiškiu tekstu; nežinomas
    formatas (1080i25) → exit != 0 su tekstu."""
    gpu_mode("cpu")
    res = []
    d = fresh_domain("ffmt-neg")
    p = FProc(out, "ffmt-neg/logo", "logo", d, "--flows", 1, "--format", "720p50", "--logo-x", 1488, "--duration", 2,
              "--impl", "cpu", "--storage", "host")
    rc = p.wait(timeout=30)
    s = p.summary()
    fatal = (s or {}).get("fatal", "")
    ok = rc not in (0, None) and "netelpa" in fatal and "720p50" in fatal
    res.append(result("ffmt_neg[logo-720p]", "PASS" if ok else "FAIL", why=f"exit={rc} fatal={fatal!r}"))
    q = FProc(out, "ffmt-neg/fmt", "gen", d, "--flows", 1, "--format", "1080i25", "--duration", 1)
    rq = q.wait(timeout=30)
    txt = q.err_path.read_text(errors="replace")
    ok = rq not in (0, None) and "1080i25" in txt
    res.append(result("ffmt_neg[1080i25]", "PASS" if ok else "FAIL", why=f"exit={rq} stderr={txt.strip()[:160]!r}"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    KW.log_dir = out
    tests = {"F5a": lambda: [f5a(out)], "F5b": lambda: [f5b(out)], "F6": lambda: f6(out), "F7": lambda: [f7(out)],
             "F8": lambda: [f8(out)], "fgfx1": lambda: fgfx1(out), "fcomp1": lambda: fcomp1(out),
             "fcomp2": lambda: fcomp2(out), "fcomp3": lambda: fcomp3(out), "fcomp4": lambda: fcomp4(out), "fcomp5": lambda: fcomp5(out),
             "fcomp6": lambda: fcomp6(out), "fcli1": lambda: fcli1(out), "fkeepwarm": lambda: fkeepwarm(out), "ffmt_neg": lambda: ffmt_neg(out),
             "ffmt_logo": lambda: ffmt_logo(out), "ffmt_logo720": lambda: ffmt_logo720(out), "ffmt_comp": lambda: ffmt_comp(out), "ffmt_enc": lambda: ffmt_enc(out),
             "ffmt_dec": lambda: ffmt_dec(out), "ffmt_dec_chroma": lambda: ffmt_dec_chroma(out)}
    # Paleidimo / sustabdymo testai (planas §P6) — atskirame modulyje; jis naudoja šį modulį (ne antrą jo kopiją).
    sys.modules.setdefault("functional", sys.modules[__name__])
    import startstop
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in startstop.TESTS.items()})
    import avsync  # A/V sinchronizacija po prarastų kadrų (favsync_drop, favsync_restart)
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in avsync.TESTS.items()})
    import golden  # „auksiniai“ regresijos vartai (fgold_logo, fgold_composite, fgold_dec)
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in golden.TESTS.items()})
    import isolation  # gedimų izoliacijos zondai (fiso_kill, fiso_stall, fiso_composite, fiso_enc)
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in isolation.TESTS.items()})
    import targeted  # tiksliniai migracijos peržiūros testai (fsplit, flate_stall, fcrashloop)
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in targeted.TESTS.items()})
    import service  # tarnybos gyvavimo ciklas (fsvc_run, fsvc_loss, fsvc_second_signal, fsvc_fail_unit)
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in service.TESTS.items()})
    import review6  # šeštos peržiūros testai (composite sluoksniai, INVALID turinys, aukštupio spragos, enc garsas, re-attach)
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in review6.TESTS.items()})
    import chain  # briaunos laiko sutartis (V24): fchain5, fchain5_kill, fforeign_expect
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in chain.TESTS.items()})
    import interop  # sąveika su MXL įrankiais (P5b): interop_info, interop_sink_host, interop_sink_cuda, interop_foreign_writer
    tests.update({k: (lambda fn=fn: fn(out)) for k, fn in interop.TESTS.items()})
    results = []
    try:
        for name, fn in tests.items():
            if a.only and name.lower() not in a.only.lower().split(","):
                continue
            try:
                results += fn()
            except Exception as e:  # testo infrastruktūros klaida — FAIL, ne tylėjimas
                results.append(result(name, "FAIL", why=f"išimtis: {e!r}"))
    finally:
        KW.stop()  # ir po KeyboardInterrupt / SIGTERM (KeepWarm handler'iai) — keepwarm nelieka veikti
    cleanup_domains()
    (out / "functional.json").write_text(json.dumps(results, indent=1))
    failed = [r["id"] for r in results if r["status"] == "FAIL"]
    print(f"FUNCTIONAL: {'PASS' if not failed else 'FAIL ' + ','.join(failed)} ({len(results)} testai)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
