#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Ar KITO proceso įvykis toje pačioje GPU (proceso startas / pabaiga, CUDA konteksto sukūrimas / sunaikinimas, didelis
cudaMalloc / cudaFree, cuda-linear flow'ų sukūrimas / naikinimas, SIGSTOP / SIGCONT) vėlina realaus laiko gamintoją?
(fiso_stall[G] vieno „pliūpsnio“ priežasties tyrimas; rezultatai — docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md.)

Auka — vienas `gen` (7 flow'ai, 1080i50, 50 Hz; G = gpu + cuda-linear arba C = cpu + host). Kiekvienam aukos commit'ui
pasyvus stebėtojas (atskiras procesas, be CUDA) kas ~1 ms skaito paskutinio aukos flow'o `data` failą (MXL flow būsena:
head indeksas ir paskutinio rašymo TAI laikas) — taip gaunamas KIEKVIENO indekso vėlavimas ir praleisti indeksai, ne tik
gen'o `vėlus commit` eilutės (> 30 ms). Stebėtojo paties „užmigimai“ (> 5 ms tarp apklausų) registruojami — tai OS
planuoklio sutrikimo požymis, nepriklausomas nuo GPU.

Įvykių scenarijai (raundai; tarp įvykių ≥ 5 s, kiekvieno įvykio TAI laikas užrašomas):
  lifecycle — antras gen (1 flow'as, G) startas → SIGTERM; startas → kill -9; verify (7 aukos flow'ai) startas → pabaiga;
              logo N=1 ir logo N=7 (skaito auką) startas → pabaiga (--duration, kaip fiso_stall'e);
  ctxcpu    — gpu_ctx_probe ctx (tik konteksto sukūrimas / sunaikinimas); antras gen C (cpu + host) startas → SIGTERM;
              gpu_ctx_probe host (tik CPU procesas: 512 MB host atminties, pabaiga); antras gen G startas → SIGSTOP 5 s →
              SIGCONT → SIGTERM;
  alloc     — gpu_ctx_probe serve (gyvas procesas): konteksto sukūrimas, tada cudaMalloc + cudaMemset ir cudaFree
              64 MB / 512 MB / 2 GB, pabaiga;
  baseline  — jokių įvykių (aukos savas fonas tam pačiam laikui).
Įvykio langas: [pradžia − 0,2 s, pabaiga + 1,0 s]; „pabaiga“ — kai fazė baigėsi (procesas surinktas, flow'ai atsirado,
zondo fazė grįžo). Natūraliai (--duration) pasibaigusio proceso pradžia — jo santraukos failo įrašymo laikas (po jo seka
flow'ų ir CUDA išvalymas). Kartu kas 100 ms imamas nvidia-smi (pstate, taktai) — ar taktų kritimas sutampa su vėlavimu.

Naudojimas (WSL2; FUNC_BIN, PROBE_BIN, LD_LIBRARY_PATH — kaip functional.py):
  python3 tests/perf/gpu_event_hiccup.py run <out> --victim G --rounds lifecycle,ctxcpu,alloc --reps 10 [--baseline 3]
  python3 tests/perf/gpu_event_hiccup.py analyze <out> [<out> …]       # lentelė + <out>/analysis.json
Papildomai (siaurinimui ir mitigacijoms): --keep-ctx (visą laiką gyvas gpu_ctx_probe serve su kontekstu),
--prealloc-mb N (gyvas zondas prieš auką užsiima N MB), --keep-busy MB,ms (nuolatinė lengva GPU apkrova, kad GPU
neišeitų į P8), --keep-warm MB,ms (tas pats per `vramflow-func keepwarm`), --narrow "komanda;…" (zondo komandos narrow raundui), --logo-n, --logo-dur, --only logo8 ir t. t.
Pastaba: `--tag` reikšmę, prasidedančią „-“, rašyti `--tag=-n2`.
"""
import argparse
import json
import mmap
import os
import pathlib
import signal
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import domain  # noqa: E402  unikalūs domenai /dev/shm/vf-<žyma>-<hex>, trynimas tik be gyvų savininkų
import gpuinfo  # noqa: E402  nvidia-smi paieška (PATH, tada WSL2 kelias)

FUNC_BIN = os.environ.get("FUNC_BIN", "/root/build/vramflow/func/vramflow-func")
PROBE_BIN = os.environ.get("PROBE_BIN", "/root/build/vramflow/perf-probe/gpu_ctx_probe")
_LIB = "/root/src/mxl-pr653/install/Linux-Clang-Release/lib:/root/build/vramflow/func"
ENV = dict(os.environ, LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH") or _LIB)

GRAIN_NS = 20_000_000  # 1080i50: grain'as = laukas, 50 Hz
N_VICTIM = 7
DEADLINE_MS = 30.0  # gen numatytasis --deadline-ms (func_common.hpp)
PRE_S, POST_S = 0.2, 1.0  # įvykio langas
GAP_S = 5.0  # tarpas tarp įvykių
# MXL flow `data` failo poslinkiai (patikrinta empiriškai su MXL 1.2.0 / PR #653: head == einamasis indeksas, šalia —
# paskutinio rašymo TAI ns). Paleidžiant tikrinama, ar head ≈ dabartinis indeksas; kitaip — klaida, ne tylūs skaičiai.
OFF_HEAD, OFF_WRITE = 200, 208
VAR = {"G": ("gpu", "cuda-linear"), "C": ("cpu", "host")}


def tai():
    return time.clock_gettime_ns(time.CLOCK_TAI)


def flow_id(role, n):
    return "4d584c47-%04d-4000-8000-%012x" % (role, n)


def wait_files(paths, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if all(pathlib.Path(p).exists() for p in paths):
            return tai()
        time.sleep(0.01)
    return None


# --- pasyvus stebėtojas (atskiras procesas) -------------------------------------------------------------------------------

def poll_main(out, data_file):
    """Kas ~1 ms: head ir paskutinio rašymo laikas; įrašas kiekvienam naujam head. SIGTERM → JSON į out."""
    fd = os.open(data_file, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    recs, starve = [], []
    last = -1
    prev = tai()
    while not stop:
        t = tai()
        if t - prev > 5_000_000:
            starve.append([prev, t])
        prev = t
        h = struct.unpack_from("<Q", mm, OFF_HEAD)[0]
        if h != last:
            recs.append([h, struct.unpack_from("<Q", mm, OFF_WRITE)[0], t])
            last = h
        time.sleep(0.001)
    pathlib.Path(out).write_text(json.dumps({"recs": recs, "starve": starve}))


# --- procesai ----------------------------------------------------------------------------------------------------------------

class P:
    """Procesas su TAI pradžios / pabaigos laikais; zondo stdout eilutės (`fazė t0 t1 rc`) renkamos fone."""

    def __init__(self, run, tag, argv, stdin=False):
        self.tag = tag
        self.err_path = run.dir / f"{tag}.err"
        self.t_start = tai()
        self.p = subprocess.Popen([str(a) for a in argv], stdout=subprocess.PIPE, stderr=open(self.err_path, "w"),
                                  stdin=subprocess.PIPE if stdin else subprocess.DEVNULL, env=ENV, text=True)
        self.t_exit = None
        self.lines = []
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._reap, daemon=True).start()

    def _read(self):
        for ln in self.p.stdout:
            f = ln.split()
            if len(f) >= 3 and f[1].isdigit():
                with self.cv:
                    self.lines.append({"phase": f[0], "t0": int(f[1]), "t1": int(f[2]), "rc": int(f[3]) if len(f) > 3 else 0})
                    self.cv.notify_all()

    def _reap(self):
        self.p.wait()
        self.t_exit = tai()

    def wait_phase(self, phase, n=1, timeout=60):
        """Laukia n-osios fazės eilutės; grąžina ją (arba None)."""
        end = time.monotonic() + timeout
        with self.cv:
            while sum(x["phase"] == phase for x in self.lines) < n:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.cv.wait(left)
            return [x for x in self.lines if x["phase"] == phase][n - 1]

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait(self, timeout=120):
        try:
            self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        while self.t_exit is None:
            time.sleep(0.001)
        return self.p.returncode


class Run:
    """Vienas aukos paleidimas su įvykių scenarijumi."""

    def __init__(self, out, victim, name, args):
        self.victim, self.name, self.args = victim, name, args
        self.dir = pathlib.Path(out) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.d = domain.make_domain("ghv")
        self.events = []
        self.procs = []
        self.rt_to_tai = tai() - time.time_ns()  # santraukų mtime (REALTIME) → TAI

    def ev(self, cls, t0, t1=None, **info):
        self.events.append({"cls": cls, "t0": t0, "t1": t1 if t1 is not None else t0, **info})

    def func(self, tag, sub, *args):
        pr = P(self, tag, [FUNC_BIN, sub, "--domain", self.d, "--out", self.dir / tag, *args])
        self.procs.append(pr)
        return pr

    def probe(self, tag, *args, stdin=False):
        pr = P(self, tag, [PROBE_BIN, *args], stdin=stdin)
        self.procs.append(pr)
        return pr

    def gap(self, s=GAP_S):
        time.sleep(s)

    def summary_tai(self, pr, sub):
        f = self.dir / pr.tag / f"summary-{sub}.json"
        return f.stat().st_mtime_ns + self.rt_to_tai if f.exists() else None

    # --- įvykiai ---
    def gen2(self, var, cls):
        impl, storage = VAR[var]
        pr = self.func(f"gen2{var}-{len(self.events)}", "gen", "--flows", 1, "--flow-base", N_VICTIM, "--duration", 600,
                       "--impl", impl, "--storage", storage)
        ready = wait_files([self.d / f"{flow_id(1, N_VICTIM)}.mxl-flow" / "flow_def.json"])
        self.ev(f"{cls}_start", pr.t_start, ready, ready_ms=None if ready is None else round((ready - pr.t_start) / 1e6, 1))
        return pr

    def end_signal(self, pr, cls, sig):
        t = tai()
        pr.p.send_signal(sig)
        rc = pr.wait()
        self.ev(cls, t, pr.t_exit, rc=rc, exit_ms=round((pr.t_exit - t) / 1e6, 1))

    def natural(self, pr, sub, cls):
        """Procesas baigiasi pats (--duration): langas nuo santraukos įrašymo iki surinkimo."""
        rc = pr.wait()
        ts = self.summary_tai(pr, sub)
        t0 = min(ts, pr.t_exit) if ts else pr.t_exit
        self.ev(cls, t0, pr.t_exit, rc=rc, teardown_ms=round((pr.t_exit - t0) / 1e6, 1))

    def logo(self, n, dur):
        impl, storage = VAR[self.victim]
        impl = "gpu"  # logo — visada CUDA procesas (C aukai: H variantas, gpu + host)
        pr = self.func(f"logo{n}-{len(self.events)}", "logo", "--flows", n, "--duration", dur, "--warmup", 1,
                       "--impl", impl, "--storage", storage)
        ready = wait_files([self.d / f"{flow_id(2, k)}.mxl-flow" / "flow_def.json" for k in range(n)])
        self.ev(f"logo{n}_start", pr.t_start, ready)
        self.natural(pr, "logo", f"logo{n}_exit")

    def verify(self, dur):
        impl, storage = VAR[self.victim]
        vi = "cpu" if storage == "host" else "gpu"
        pr = self.func(f"verify-{len(self.events)}", "verify", "--mode", "gen", "--flows", N_VICTIM, "--duration", dur,
                       "--warmup", 1, "--every", 5, "--impl", vi, "--storage", storage)
        self.ev("verify_start", pr.t_start, pr.t_start + 500_000_000)
        self.natural(pr, "verify-gen", "verify_exit")

    def probe_phases(self, pr, prefix, skip=()):
        for x in pr.lines:
            if x["phase"] not in skip:  # pabaiga — iki proceso surinkimo (CUDA / proceso išvalymas po paskutinės eilutės)
                t1 = pr.t_exit if x["phase"] == "exit" and pr.t_exit else x["t1"]
                self.ev(f"{prefix}_{x['phase']}", x["t0"], t1, rc=x["rc"])


# --- raundai -----------------------------------------------------------------------------------------------------------------

def round_lifecycle(r):
    a = r.args
    only = set(a.only.split(",")) if a.only else None
    want = lambda k: only is None or k in only  # noqa: E731
    if want("gen2"):
        g = r.gen2("G", "gen2G")
        r.gap()
        r.end_signal(g, "gen2G_term", signal.SIGTERM)
        r.gap()
        g = r.gen2("G", "gen2G")
        r.gap()
        r.end_signal(g, "gen2G_kill9", signal.SIGKILL)
        r.gap()
    if want("verify"):
        r.verify(4)
        r.gap()
    for n in [int(x) for x in a.logo_n.split(",")]:
        if want(f"logo{n}"):
            g = None
            if n > N_VICTIM:  # kaip fiso_stall: paskutinis logo įėjimas — antro gen'o flow'as (--flow-base N_VICTIM)
                g = r.gen2("G", "gen2G")
                r.gap()
            r.logo(n, a.logo_dur)
            r.gap()
            if g is not None:
                r.end_signal(g, "gen2G_term", signal.SIGTERM)
                r.gap()


def round_ctxcpu(r):
    c = r.probe("ctx", "ctx", GAP_S)
    c.wait()
    r.probe_phases(c, "ctxprobe")
    r.gap()
    g = r.gen2("C", "gen2C")
    r.gap()
    r.end_signal(g, "gen2C_term", signal.SIGTERM)
    r.gap()
    h = r.probe("host", "host", 512)
    h.wait()
    r.probe_phases(h, "hostprobe")
    r.gap()
    g = r.gen2("G", "gen2G")
    r.gap()
    t = tai()
    g.p.send_signal(signal.SIGSTOP)
    r.ev("gen2G_sigstop", t)
    r.gap()
    t = tai()
    g.p.send_signal(signal.SIGCONT)
    r.ev("gen2G_sigcont", t)
    r.gap()
    r.end_signal(g, "gen2G_term", signal.SIGTERM)
    r.gap()


def round_alloc(r):
    s = r.probe("serve", "serve", stdin=True)
    s.wait_phase("exec")
    r.gap()
    s.send("ctx")
    s.wait_phase("ctx_create")
    r.gap()
    for mb in (64, 512, 2048):
        k = sum(x["phase"] == "malloc" for x in s.lines) + 1
        s.send(f"malloc {mb}")
        m = s.wait_phase("memset", k)
        mm = s.wait_phase("malloc", k)
        r.ev(f"malloc{mb}", mm["t0"], m["t1"], rc=mm["rc"], malloc_ms=round((mm["t1"] - mm["t0"]) / 1e6, 1),
             memset_ms=round((m["t1"] - m["t0"]) / 1e6, 1))
        r.gap()
        s.send("free")
        f = s.wait_phase("free", k)
        r.ev(f"free{mb}", f["t0"], f["t1"], rc=f["rc"], free_ms=round((f["t1"] - f["t0"]) / 1e6, 1))
        r.gap()
    s.send("quit")
    s.wait()
    for x in s.lines:
        if x["phase"] in ("exec", "ctx_create"):
            r.ev(f"serve_{x['phase']}", x["t0"], x["t1"])
    r.ev("serve_exit", s.lines[-1]["t0"], s.t_exit)
    r.gap()


def round_narrow(r):
    """Siaurinimas: gyvas zondas su kontekstu vykdo --narrow komandas (kiekviena — atskiras įvykis, tarpas GAP_S):
    tik rezervavimas, tik memset, rezervavimas + memset, vieno kernel'io trukmė (spin)."""
    s = r.probe("serve", "serve", "ctx", stdin=True)
    c = s.wait_phase("ctx_create")
    r.ev("serve_ctx_create", c["t0"], c["t1"])
    r.gap()
    for cmd in [x.strip() for x in r.args.narrow.split(";") if x.strip()]:
        n0 = len(s.lines)
        s.send(cmd)
        want = n0 + (2 if cmd.startswith("malloc ") else 1)
        end = time.monotonic() + 60
        with s.cv:
            while len(s.lines) < want and time.monotonic() < end:
                s.cv.wait(1)
        xs = s.lines[n0:want]
        r.ev("n_" + cmd.replace(" ", ""), xs[0]["t0"], xs[-1]["t1"], rc=max(x["rc"] for x in xs),
             phase_ms=[round((x["t1"] - x["t0"]) / 1e6, 1) for x in xs])
        r.gap()
    s.send("quit")
    s.wait()
    r.ev("serve_exit", s.lines[-1]["t0"], s.t_exit)
    r.gap()


def round_baseline(r):
    time.sleep(r.args.baseline_s)


ROUNDS = {"lifecycle": round_lifecycle, "ctxcpu": round_ctxcpu, "alloc": round_alloc, "narrow": round_narrow,
          "baseline": round_baseline}


def smi_sampler(path, stop):
    """nvidia-smi kas 100 ms; kiekviena eilutė su atvykimo TAI laiku."""
    q = "pstate,clocks.gr,clocks.mem,utilization.gpu,power.draw,clocks_event_reasons.active"
    p = subprocess.Popen([gpuinfo.nvidia_smi(), f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms", "100"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    rows = []

    def rd():
        for ln in p.stdout:
            rows.append([tai(), *[x.strip() for x in ln.split(",")]])

    th = threading.Thread(target=rd, daemon=True)
    th.start()
    stop.wait()
    p.terminate()
    p.wait()
    th.join(timeout=2)
    path.write_text(json.dumps(rows))


def run_one(out, victim, rnd, k, args):
    r = Run(out, victim, f"{victim}-{rnd}{args.tag}-{k:02d}", args)
    impl, storage = VAR[victim]
    stop = threading.Event()
    smi = None
    if not args.no_smi:
        smi = threading.Thread(target=smi_sampler, args=(r.dir / "smi.json", stop), daemon=True)
        smi.start()
    keep = None
    if args.keep_ctx or args.prealloc_mb:  # mitigacija: gyvas kontekstas (ir užimta atmintis) dar prieš auką
        keep = r.probe("keepctx", "serve", "ctx", stdin=True)
        keep.wait_phase("ctx_create")
        if args.prealloc_mb:
            keep.send(f"malloc {args.prealloc_mb}")
            keep.wait_phase("memset")
    if args.keep_busy:  # mitigacija: nuolatinė lengva apkrova laiko GPU virš P8 (be P8 → P3 perėjimų įvykio metu)
        mb, ms = args.keep_busy.split(",")
        busy = r.probe("keepbusy", "busy", mb, ms)
        busy.wait_phase("busy")
        time.sleep(3)  # leisti taktams pakilti prieš aukos startą
    if args.keep_warm:  # ta pati mitigacija gamybiniu įrankiu: vramflow-func keepwarm
        mb, ms = args.keep_warm.split(",")
        r.procs.append(P(r, "keepwarm", [FUNC_BIN, "keepwarm", "--size-mb", mb, "--interval-ms", ms, "--duration", 0]))
        time.sleep(3)
    vic = r.func("victim", "gen", "--flows", N_VICTIM, "--duration", 900, "--warmup", 2, "--impl", impl, "--storage", storage)
    data = r.d / f"{flow_id(1, N_VICTIM - 1)}.mxl-flow" / "data"
    poll = None
    try:
        if wait_files([data, r.d / f"{flow_id(1, N_VICTIM - 1)}.mxl-flow" / "flow_def.json"]) is None:
            raise RuntimeError("aukos flow'ai neatsirado")
        time.sleep(0.5)
        h = struct.unpack_from("<Q", open(data, "rb").read(OFF_HEAD + 8), OFF_HEAD)[0]
        if abs(h - tai() // GRAIN_NS) > 5:
            raise RuntimeError(f"data failo head ({h}) nesutampa su dabartiniu indeksu — pasikeitė MXL išdėstymas?")
        poll = subprocess.Popen([sys.executable, __file__, "poll", r.dir / "poll.json", data])
        r.t_victim_ready = tai()
        time.sleep(args.settle)
        r.t_events_from = tai()
        ROUNDS[rnd](r)
        time.sleep(2)
        r.t_events_to = tai()
    finally:
        for pr in r.procs:
            if pr is not vic and pr.p.poll() is None:
                try:
                    pr.p.send_signal(signal.SIGCONT)
                    if pr.p.stdin:
                        pr.send("quit")
                except (ProcessLookupError, BrokenPipeError, OSError):
                    pass
                pr.p.send_signal(signal.SIGTERM)
                pr.wait(30)
        if poll is not None:
            poll.send_signal(signal.SIGTERM)
            poll.wait(30)
        vic.p.send_signal(signal.SIGTERM)
        vrc = vic.wait(60)
        stop.set()
        if smi is not None:
            smi.join(timeout=10)
        domain.remove_domain(r.d)
    summ = r.dir / "victim" / "summary-gen.json"
    late = []
    for ln in pathlib.Path(vic.err_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("[gen] vėlus commit"):
            f = ln.replace(",", "").split()
            late.append({"index": int(f[4]), "late_ms": float(f[5]), "tai": int(f[-1])})
    res = {"victim": victim, "round": rnd, "rep": k, "tag": args.tag, "victim_rc": vrc,
           "keep_ctx": bool(args.keep_ctx), "prealloc_mb": args.prealloc_mb, "keep_busy": args.keep_busy, "keep_warm": args.keep_warm,
           "window": [r.t_events_from, r.t_events_to], "events": r.events, "late_lines": late,
           "summary": json.loads(summ.read_text()) if summ.exists() else None}
    (r.dir / "run.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [e for e in r.events if e.get("rc") not in (None, 0, -signal.SIGTERM, -signal.SIGKILL)]
    print(f"[ghv] {r.name}: įvykių {len(r.events)}, vėlių commit'ų {len(late)}, max "
          f"{(((res['summary'] or {}).get('hist') or {}).get('commit_lateness') or {}).get('max_ms')} ms"
          + (f", nenuliniai rc: {[(e['cls'], e['rc']) for e in bad]}" if bad else ""), flush=True)


# --- analizė -----------------------------------------------------------------------------------------------------------------

def load_run(d):
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    pj = d / "poll.json"
    poll = json.loads(pj.read_text()) if pj.exists() else {"recs": [], "starve": []}
    sj = d / "smi.json"
    smi = json.loads(sj.read_text()) if sj.exists() else []
    return run, poll, smi


def per_index(poll, t_from, t_to, late_lines=()):
    """Indeksas → vėlavimas ms (paskutinio rašymo laikas − T(i)); praleisti indeksai (head šuolis) — atskirai.
    Po vėlaus commit'o gen'as kitą indeksą įrašo iškart — stebėtojas gali nespėti pamatyti vėlaujančio head (1 ms
    apklausa), todėl gen'o paties `vėlus commit` eilutės (visi > 30 ms) papildo / perrašo stebėtojo duomenis."""
    lat, missing = {}, set()
    prev = None
    for h, w, _t in poll["recs"]:
        if prev is not None and h > prev + 1:
            missing.update(range(prev + 1, h))
        lat[h] = (w - h * GRAIN_NS) / 1e6
        prev = h
    for ln in late_lines:
        lat[ln["index"]] = ln["late_ms"]
        missing.discard(ln["index"])
    lo, hi = t_from // GRAIN_NS, t_to // GRAIN_NS
    return {i: v for i, v in lat.items() if lo <= i <= hi}, sorted(i for i in missing if lo <= i <= hi)


def window_stats(lat, missing, w0, w1, starve):
    lo, hi = w0 // GRAIN_NS, w1 // GRAIN_NS + 1
    vals = [v for i, v in lat.items() if lo <= i <= hi]
    miss = [i for i in missing if lo <= i <= hi]
    st = max([(b - a) / 1e6 for a, b in starve if b >= w0 and a <= w1] or [0.0])
    return {"max_ms": round(max(vals), 2) if vals else None, "skips": len(miss), "n": len(vals), "poll_starve_ms": round(st, 1)}


def smi_window(smi, w0, w1):
    rows = [x for x in smi if w0 <= x[0] <= w1]
    pre = [x for x in smi if w0 - 2_000_000_000 <= x[0] < w0]
    clk = lambda rr: [int(float(x[2])) for x in rr if len(x) > 2 and x[2].replace(".", "").isdigit()]  # noqa: E731
    return {"pstates": sorted({x[1] for x in rows if len(x) > 1}), "gr_min": min(clk(rows) or [None], key=lambda v: v or 0),
            "gr_pre": min(clk(pre) or [None], key=lambda v: v or 0)}


def analyze(outs):
    rows, hiccups, base = [], [], {}
    lw_err = []
    for out in outs:
        for d in sorted(pathlib.Path(out).iterdir()):
            if not (d / "run.json").exists():
                continue
            run, poll, smi = load_run(d)
            t_from, t_to = run["window"]
            lat, missing = per_index(poll, t_from, t_to, run["late_lines"])
            byidx = per_index(poll, 0, 1 << 63)[0]
            for ln in run["late_lines"]:  # stebėtojo vėlavimas prieš gen'o paties užrašytą (> 30 ms) — metodo patikra
                if ln["index"] in byidx:
                    lw_err.append(abs(byidx[ln["index"]] - ln["late_ms"]))
            vkey = run["victim"] + (run["tag"] or "")
            b = base.setdefault(vkey, {"indices": 0, "bad": 0, "bad_in_win": 0, "late_lines": 0, "late_skips": 0})
            b["indices"] += len(lat) + len(missing)
            b["late_lines"] += len(run["late_lines"])
            b["late_skips"] += ((run["summary"] or {}).get("proc") or {}).get("late_skips", 0)
            wins = []
            for e in run["events"]:
                w0, w1 = e["t0"] - int(PRE_S * 1e9), e["t1"] + int(POST_S * 1e9)
                ws = window_stats(lat, missing, w0, w1, poll["starve"])
                ws.update(smi_window(smi, w0, w1))
                wins.append((w0, w1))
                rows.append({"victim": vkey, "round": run["round"], "run": d.name, "cls": e["cls"],
                             "dur_ms": round((e["t1"] - e["t0"]) / 1e6, 1), **ws,
                             **{k: v for k, v in e.items() if k not in ("cls", "t0", "t1")}})
            # visi „blogi“ indeksai (> DEADLINE_MS arba praleisti) — sugrupuoti į pliūpsnius su artimiausiu įvykiu
            bad = sorted([i for i, v in lat.items() if v > DEADLINE_MS] + missing)
            b["bad"] += len(bad)
            clusters = []
            for i in bad:
                if clusters and i - clusters[-1][-1] <= 5:
                    clusters[-1].append(i)
                else:
                    clusters.append([i])
            for c in clusters:
                t = c[0] * GRAIN_NS
                near = sorted(run["events"], key=lambda e: min(abs(t - e["t0"]), abs(t - e["t1"])))[:2]
                inwin = any(w0 <= t <= w1 for w0, w1 in wins)
                b["bad_in_win"] += len(c) if inwin else 0
                before = [x for x in smi if x[0] <= t]
                hiccups.append({"victim": vkey, "run": d.name, "first_index": c[0], "n_bad": len(c),
                                "smi_before": before[-1][1:3] if before else None,
                                "max_ms": round(max([lat.get(i, 0) for i in c]), 1), "skips": sum(i in missing for i in c),
                                "in_window": inwin,
                                "nearest": [{"cls": e["cls"], "dt_from_t0_ms": round((t - e["t0"]) / 1e6, 1),
                                             "dt_from_t1_ms": round((t - e["t1"]) / 1e6, 1)} for e in near]})
    table = {}
    for r in rows:
        k = (r["victim"], r["cls"])
        t = table.setdefault(k, {"events": 0, "hiccups": 0, "worst_ms": 0.0, "skips": 0, "maxes": [], "starve_max": 0.0,
                                 "clk_drop": 0})
        t["events"] += 1
        m = r["max_ms"] or 0.0
        t["hiccups"] += int(m > DEADLINE_MS or r["skips"] > 0)
        t["worst_ms"] = max(t["worst_ms"], m)
        t["skips"] += r["skips"]
        t["maxes"].append(m)
        t["starve_max"] = max(t["starve_max"], r["poll_starve_ms"])
        if r["gr_min"] and r["gr_min"] < 500:  # žemas taktas (P8 ~210 MHz) bent vienoje lango imtyje
            t["clk_drop"] += 1
    out = {"table": [{"victim": k[0], "cls": k[1], **{kk: vv for kk, vv in v.items() if kk != "maxes"},
                      "median_max_ms": round(sorted(v["maxes"])[len(v["maxes"]) // 2], 2)} for k, v in sorted(table.items())],
           "hiccups": hiccups, "background": base, "method_check": {"n": len(lw_err), "max_abs_diff_ms":
                                                                     round(max(lw_err), 2) if lw_err else None},
           "rows": rows}
    pathlib.Path(outs[0], "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("| auka | įvykis | įvykių | pliūpsnių > 30 ms | blogiausias ms | mediana ms | praleista | stebėtojo užmigimas ms | langų su žemu taktu (<500 MHz) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in out["table"]:
        print(f"| {t['victim']} | {t['cls']} | {t['events']} | {t['hiccups']} | {t['worst_ms']:.1f} | {t['median_max_ms']:.1f} | "
              f"{t['skips']} | {t['starve_max']:.1f} | {t['clk_drop']} |")
    print("\nfonas:", json.dumps(base, ensure_ascii=False))
    print("metodo patikra (stebėtojas prieš gen vėlus commit):", out["method_check"])
    for h in hiccups:
        print("pliūpsnis:", json.dumps(h, ensure_ascii=False))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "poll":
        return poll_main(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("out")
    r.add_argument("--victim", default="G", choices=list(VAR))
    r.add_argument("--rounds", default="lifecycle,ctxcpu,alloc")
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--baseline", type=int, default=3, help="kiek baseline paleidimų (be įvykių)")
    r.add_argument("--baseline-s", type=float, default=60.0)
    r.add_argument("--settle", type=float, default=6.0, help="s po aukos flow'ų atsiradimo iki pirmo įvykio")
    r.add_argument("--logo-n", default="1,7")
    r.add_argument("--logo-dur", type=float, default=4.0)
    r.add_argument("--only", default="", help="lifecycle: tik šie (gen2,verify,logo1,logo7 …)")
    r.add_argument("--narrow", default="mallocraw 512;memset 512;memset 512;free;malloc 512;free;"
                                       "spin 5;spin 20;spin 50;spin 100;spin 200",
                   help="narrow raundo zondo komandos, atskirtos ;")
    r.add_argument("--keep-ctx", action="store_true", help="mitigacija: gyvas CUDA kontekstas visą paleidimą")
    r.add_argument("--prealloc-mb", type=int, default=0)
    r.add_argument("--keep-busy", default="", help="mitigacija: gpu_ctx_probe busy <MB>,<pauzė ms> visą paleidimą")
    r.add_argument("--keep-warm", default="", help="mitigacija: vramflow-func keepwarm <MB>,<periodas ms> visą paleidimą")
    r.add_argument("--no-smi", action="store_true")
    r.add_argument("--tag", default="")
    a = sp.add_parser("analyze")
    a.add_argument("outs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "analyze":
        return analyze(args.outs)
    rounds = [x for x in args.rounds.split(",") if x]
    for k in range(args.reps):
        for rnd in rounds:
            run_one(args.out, args.victim, rnd, k, args)
    for k in range(args.baseline):
        run_one(args.out, args.victim, "baseline", k, args)


if __name__ == "__main__":
    main()
