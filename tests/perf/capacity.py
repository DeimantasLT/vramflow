#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""GPU media funkcijų talpos harness (architektūra GPU funkcijų §8.1).

Kiekvienai N pakopai paleidžia gamintojus → funkciją → vartotojus atskiruose procesuose, matuoja `seconds`
po `warmup` ir surenka santraukas. PASS(N) = 0 praradimų ir 0 teisingumo klaidų, realaus laiko santykis ≥ 0,999,
visi procesai exit 0. N* = didžiausias N, kai PASS ir visi mažesni PASS.

  capacity.py <out> --func logo --n 1,2,4,8,16,32,48,64
  capacity.py <out> --func enc-nvenc-ll --n 1,2,4,6,8,10,12,13
  capacity.py <out> --func enc-x264-veryfast --n 1,2,3,4,5,6,8
  capacity.py <out> --func dec --n 1,2,4,8,12,16,24,32 --clip clip.ts --ref ref_crc.json
  capacity.py <out> --func chain --n 1,2,4,6,8,10,12 --clip clip.ts
  capacity.py <out> --func composite --layers 2 --n 1,2,4,8,16,32 [--out-format v210|v210a]
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

import domain  # tests/perf/domain.py

REPO = pathlib.Path(__file__).resolve().parents[2]
BIN = os.environ.get("FUNC_BIN", str(pathlib.Path.home() / "build/vramflow/func/vramflow-func"))
LIB = os.environ.get("MXL_LIB", str(pathlib.Path.home() / "src/mxl-pr653/install/Linux-Clang-Release/lib"))
ENV = dict(os.environ, LD_LIBRARY_PATH=LIB)
TICK = os.sysconf("SC_CLK_TCK")


def cpu_s(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / TICK  # utime + stime (13, 14 laukai)
    except (OSError, IndexError, ValueError):
        return None


def vram_mb():
    try:
        out = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


GPU_FIELDS = ["util_gpu", "util_mem", "util_enc", "util_dec", "power_w", "sm_mhz", "mem_used_mb"]


def power_scheme():
    """Windows energijos profilis (WSL2): tylusis profilis riboja GPU/CPU galią ir keičia talpos rezultatus."""
    try:
        out = subprocess.run(["powercfg.exe", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout
        return out.strip().split("(")[-1].rstrip(")") if "(" in out else out.strip()
    except Exception:
        return None


def start_gpu_probe(path):
    """GPU apkrova matavimo lange, 1 Hz: vienas ilgai veikiantis nvidia-smi (pats zondas beveik nekrauna CPU).
    util_gpu — SM užimtumo laiko dalis (NVML), util_enc/util_dec — NVENC/NVDEC variklių užimtumas."""
    try:
        return subprocess.Popen(["/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else "nvidia-smi",
                                 "--query-gpu=utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,"
                                 "power.draw,clocks.sm,memory.used", "--format=csv,noheader,nounits", "-lms", "1000"],
                                stdout=open(path, "w"), stderr=subprocess.DEVNULL)
    except Exception:
        return None


def stop_gpu_probe(p):
    if p:
        p.terminate()
        try:
            p.wait(5)
        except Exception:
            p.kill()


def gpu_stats(path):
    rows = []
    try:
        for line in open(path):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == len(GPU_FIELDS):
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass
    except OSError:
        return None
    if not rows:
        return None
    out = {"samples": len(rows)}
    for i, k in enumerate(GPU_FIELDS):
        col = sorted(r[i] for r in rows)
        out[k] = {"mean": round(sum(col) / len(col), 2), "p95": col[min(len(col) - 1, int(0.95 * len(col)))], "max": col[-1]}
    return out


# --- composite (N išėjimų × K grafikos sluoksnių) -------------------------------------------------------------------------
# Sluoksnių geometrija pagal specifikaciją §4.7: K=1 → logo; K=2 → logo,full; K=4 → logo,full,full,full (K=3 — logo,full,full).
# (geometrija, X,Y kadro koordinatės, nepermatomumas). Pozicijos fiksuotos, kad kampanijos būtų palyginamos.
# composite --duration = total + COMP_TAIL_S (nuo composite starto). Matavimo langas baigiasi ≤ verify starto + total + 5 s (5 s
# mėginių ciklas), todėl realus rezervas = COMP_TAIL_S - tarpas - 5. Reikalaujame ≥ 6 s: leistinas tarpas = COMP_TAIL_S - 11 (= 9 s).
COMP_TAIL_S = 20
COMP_MAX_GAP_S = COMP_TAIL_S - 11
COMP_LAYERS = [("logo", "1488,64", 1.0), ("full", "0,0", 0.8), ("full", "0,0", 0.8), ("full", "0,0", 0.8)]
# Raster aprašai (plotis, aukštis, interlaced): lauko grain'as = h/2 eilučių
GEOM_WH = {"logo": (384, 216), "full": (1920, 1080)}
RING_GRAINS = 10  # MXL ring (laukai) vienam flow (išmatuota: logo G ΔVRAM ≈ 27,6 MB/flow = 10 × 2 764 800 B)
# G ΔVRAM modelis = VRAM_OFFSET_MB + N·(VRAM_K_BASE·ring(bazė+logo+išėjimas) + VRAM_K_FULL·ring(full sluoksniai)) · VRAM_MARGIN.
# Koeficientai — tiesinė regresija pagal išmatuotą G ΔVRAM (compare-composite-g, K ∈ {1,2,4} × N ≤ 16 ir K=1,2 N=32):
# poslinkis (CUDA kontekstai, 4 procesai) ≈ 507 MiB visiems K; kiekvienas `full` sluoksnis kainuoja 56,2 MiB/N (= 1,42 × jo ring'as
# 39,6 MiB), likusi dalis (bazė + logo + išėjimas) 58,9 MiB/N (= 1,085 × ring'as). Paklaida išmatuotuose taškuose ≤ 1 %.
VRAM_OFFSET_MB = 507
VRAM_K_BASE = 1.085
VRAM_K_FULL = 1.42
VRAM_MARGIN = 1.05  # 5 % atsarga ekstrapoliacijai (K=4 N=32 nematuota: pakilo iki 7223 MiB ir pakibo)
VRAM_CAP_MB = 7168  # R19: ~7 GiB iš 8 GiB kortos (WDDM ir kiti procesai laiko ~0,3–0,5 GB)


def comp_layers(k):
    return COMP_LAYERS[:k]


def v210_field_bytes(w, h):
    return -(-w // 48) * 128 * (h // 2)


def key_field_bytes(w, h):
    return -(-w // 3) * 4 * (h // 2)


def comp_budget_mb(a, n):
    """R19: ring'ų atmintis (MiB) pakopai N: bazė N + sluoksniai N·K + išėjimai N (visų RING_GRAINS grain'ų).
    Grąžina (sk. aprašas, bendra MiB, GPU procesų skaičius)."""
    parts = [("bazė (gen) v210 1920x1080i", v210_field_bytes(1920, 1080), n)]
    for k, (g, _, _) in enumerate(comp_layers(a.layers)):
        w, h = GEOM_WH[g]
        parts.append((f"sluoksnis {k} ({g} {w}x{h} v210a)", v210_field_bytes(w, h) + key_field_bytes(w, h), n))
    ob = v210_field_bytes(1920, 1080) + (key_field_bytes(1920, 1080) if a.out_format == "v210a" else 0)
    parts.append((f"išėjimas ({a.out_format})", ob, n))
    ring = sum(b * c for _, b, c in parts) * RING_GRAINS
    return parts, ring / 2**20


def comp_vram_mb(a, n):
    """G: numatomas ΔVRAM (MiB) pagal regresijos modelį (žr. VRAM_* konstantas)."""
    parts, _ = comp_budget_mb(a, n)
    base = sum(b * c for nm, b, c in parts if "(full " not in nm) * RING_GRAINS / 2**20
    full = sum(b * c for nm, b, c in parts if "(full " in nm) * RING_GRAINS / 2**20
    return (VRAM_OFFSET_MB + VRAM_K_BASE * base + VRAM_K_FULL * full) * VRAM_MARGIN


def comp_guard(a, n):
    """Grąžina None (galima vykdyti) arba skip priežasties tekstą su skaičiavimu (R19). G: VRAM; C/H: /dev/shm."""
    parts, ring_mb = comp_budget_mb(a, n)
    desc = " + ".join(f"{b}B×{c}" for _, b, c in parts)
    if a.impl == "gpu" and a.storage == "cuda-linear":
        tot = comp_vram_mb(a, n)
        if tot > VRAM_CAP_MB:
            return (f"VRAM riba (apskaičiuota): ({desc}) × {RING_GRAINS} grain'ų = ring {ring_mb:.0f} MiB; regresijos modelis "
                    f"({VRAM_OFFSET_MB} + {VRAM_K_BASE}·bazė + {VRAM_K_FULL}·full) × {VRAM_MARGIN} = {tot:.0f} MiB > {VRAM_CAP_MB} MiB")
        return None
    try:
        st = os.statvfs("/dev/shm")
        shm_mb = st.f_blocks * st.f_frsize / 2**20
    except OSError:
        return None
    # C/H: ring'ai /dev/shm (RAM); gfxgen/gen tame pačiame domene — iš viso ~ring_mb
    if ring_mb > 0.75 * shm_mb:
        return (f"/dev/shm riba (apskaičiuota): ({desc}) × {RING_GRAINS} = {ring_mb:.0f} MiB > 75 % × {shm_mb:.0f} MiB")
    return None


def wait_flow_defs(domain, ids, timeout):
    """Laukia, kol visų flow'ų kataloguose atsiranda flow_def.json (gen / gfxgen / composite startas)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if all((pathlib.Path(domain) / f"{i}.mxl-flow" / "flow_def.json").exists() for i in ids):
            return True
        time.sleep(0.2)
    return False


def role_flow_id(role, n):
    return "4d584c47-%04d-4000-8000-%012x" % (role, n)


class Procs:
    def __init__(self, d):
        self.d = d
        self.items = []  # (vardas, Popen, kategorija)

    def start(self, name, args, cat="func"):
        err = open(self.d / f"{name}.err", "w")
        p = subprocess.Popen(args, stdout=subprocess.DEVNULL if cat != "sink" else open(self.d / f"{name}.out", "w"),
                             stderr=err, env=ENV)
        self.items.append((name, p, cat))
        return p

    def cpu(self):
        return {name: cpu_s(p.pid) for name, p, _ in self.items}

    def stop_all(self, timeout=40):
        # Pirma laukiam, kol savo trukme baigs funkcijos ir vartotojai; siuntėjus stabdom TIK PO TO —
        # kitaip dekoderis paskutines sekundes liktų be įėjimo ir tai būtų užskaityta kaip underrun.
        codes = {}
        for name, p, cat in self.items:
            if cat in ("sender", "src"):
                continue
            try:
                codes[name] = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                codes[name] = "killed"
        for name, p, cat in self.items:
            if cat not in ("sender", "src"):
                continue
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
            try:
                codes[name] = p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                codes[name] = "killed"
        return codes


def load(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def flow_sum(summary, keys):
    tot = 0
    for f in (summary or {}).get("flows", []):
        tot += sum(f.get(k, 0) for k in keys)
    return tot


LOSS_KEYS = ["late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips", "late_frames",
             "frames_dropped_queue", "late_skips_frames", "audio_underrun_frames", "encode_errors", "mux_errors",
             "underrun_fields", "pkts_corrupt", "decode_errors", "overrun_frames", "overflow_drops", "pitch_mismatch",
             "frame_alloc_errors", "audio_write_errors"]


def func_tag(a):
    """Failų/katalogų žyma: composite — su sluoksnių skaičiumi (kampanijos K ∈ {1,2,4} nesimaišo)."""
    return f"{a.func}-k{a.layers}" if a.func == "composite" else a.func


def run_level(a, n):
    variant = "C" if a.impl == "cpu" else ("H" if a.storage == "host" else "G")
    tag = func_tag(a)
    d = pathlib.Path(a.out) / f"{tag}-{variant}-n{n}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    # Unikalus domenas kiekvienam lygiui (planas §P6): du lygiagretūs capacity.py vienas kito domeno netrina.
    dom = str(domain.make_domain(f"mxlf-{tag}-{n}"))
    P = Procs(d)
    S, W = a.seconds, a.warmup
    total = S + W
    common = ["--domain", dom, "--flows", str(n), "--complexity", str(a.complexity), "--warmup", str(W),
              "--impl", a.impl, "--storage", a.storage, "--threads", str(a.threads)]
    vram0 = vram_mb()
    f = a.func
    port = 20000
    if f == "logo":
        # Šaltinis: --source same — gen tuo pačiu variantu (sistemos N*); --source fair — C variantui šaltinis
        # GPU→host (pigus), kad matuotume FUNKCIJOS talpą, o ne sintetinio CPU generatoriaus (≈ 0,5 branduolio/flow)
        gcommon = list(common)
        if a.source == "fair" and a.impl == "cpu":
            gcommon[gcommon.index("--impl") + 1] = "gpu"
            gcommon[gcommon.index("--storage") + 1] = "host"
        P.start("gen", [BIN, "gen", *gcommon, "--duration", str(total + 12), "--out", str(d)], "producer")
        time.sleep(3)
        P.start("logo", [BIN, "logo", *common, "--duration", str(total + 6), "--out", str(d)])
        time.sleep(3)
        # host flow'us tikrina CPU (be CUDA konteksto), kad matavimo įrankis neiškreiptų C/H varianto
        vcommon = list(common)
        if a.storage == "host" or a.impl == "cpu":
            vcommon[vcommon.index("--impl") + 1] = "cpu"
        P.start("verify", [BIN, "verify", *vcommon, "--mode", "logo", "--every", str(a.every), "--duration", str(total), "--out", str(d)], "consumer")
    elif f == "composite":
        K = a.layers
        lay = comp_layers(K)
        geom = ",".join(g for g, _, _ in lay)
        # Šaltiniai (gen + gfxgen): --source fair — C variantui GPU→host (pigu), kad matuotume FUNKCIJOS talpą; kitaip tas pats variantas
        gcommon = list(common)
        if a.source == "fair" and a.impl == "cpu":
            gcommon[gcommon.index("--impl") + 1] = "gpu"
            gcommon[gcommon.index("--storage") + 1] = "host"
        # Gamintojai veikia ilgai ir stabdomi (SIGTERM) PO composite ir verify pabaigos ("src" kategorija)
        long = total + 120
        P.start("gen", [BIN, "gen", *gcommon, "--duration", str(long), "--out", str(d)], "src")
        P.start("gfxgen", [BIN, "gfxgen", *gcommon, "--layers", str(K), "--layer-geom", geom, "--mode", "pace",
                           "--duration", str(long), "--out", str(d)], "src")
        src_ids = [role_flow_id(1, i) for i in range(n)] + [role_flow_id(5, i * 8 + k) for i in range(n) for k in range(K)]
        src_ok = wait_flow_defs(dom, src_ids, 60 + 2 * n * K)
        xy = [x for _, p, _ in lay for x in ("--layer-xy", p)]
        op = [x for _, _, o in lay for x in ("--opacity", str(o))]
        # composite baigiasi savo trukme (total + COMP_TAIL_S). SIGTERM netinka: nutraukti laukai iškreiptų RT. Tarpas nuo
        # composite starto iki verify starto turi būti ≤ COMP_MAX_GAP_S = COMP_TAIL_S − 11 (= 9 s): langas gali baigtis iki 5 s po
        # verify pabaigos (5 s mėginių ciklas), todėl realus rezervas ≥ 6 s. Kitaip pakopa NEDERINA (žr. invalid_reason), ne PASS/FAIL
        cstart = time.monotonic()
        P.start("composite", [BIN, "composite", *common, "--layers", str(K), *xy, *op, "--base", "gen",
                              "--out-format", a.out_format, "--deadline-ms", str(a.deadline_ms),
                              "--duration", str(total + COMP_TAIL_S), "--out", str(d)])
        out_ok = wait_flow_defs(dom, [role_flow_id(6, i) for i in range(n)], 60 + n)
        time.sleep(2)
        # verify: CPU etalonas (be CUDA konteksto host / C variantuose), kaip logo
        vcommon = list(common)
        if a.storage == "host" or a.impl == "cpu":
            vcommon[vcommon.index("--impl") + 1] = "cpu"
        P.start("verify", [BIN, "verify", *vcommon, "--mode", "composite", "--layers", str(K), "--layer-geom", geom,
                           *xy, *op, "--base", "gen", "--out-format", a.out_format, "--gfx-mode", "pace",
                           "--every", str(a.every), "--duration", str(total), "--out", str(d)], "consumer")
        comp_gap_s = time.monotonic() - cstart  # composite starto → verify starto tarpas (informacinis)
    elif f.startswith("enc-"):
        _, enc, prof = f.split("-", 2)
        P.start("gen", [BIN, "gen", *common, "--audio", "1", "--duration", str(total + 14), "--out", str(d)], "producer")
        time.sleep(3)
        P.start("sink", [sys.executable, str(REPO / "tests/perf/ts_sink.py"), "--port-base", str(port), "--flows", str(n),
                         "--seconds", str(total + 4), "--warmup", str(W + 4), "--out", str(d / "sink.json"),
                         *(["--dump", f"0:{d}/stream0.ts"] if a.dump else [])], "sink")
        ea = ["--encoder", enc, "--audio", "1", "--port-base", str(port)]
        if enc == "nvenc":
            ea += ["--preset", "p4", "--tune", prof]
        else:
            ea += ["--x264-preset", prof, "--x264-threads", str(a.x264_threads)]
        P.start("enc", [BIN, "enc", *common, *ea, "--duration", str(total + 2), "--out", str(d)])
    elif f in ("dec", "chain"):
        if not a.clip:
            raise SystemExit("--clip privalomas dec/chain")
        dport = 21000
        # chain: dec startuoja ~5 s anksčiau už logo (total+6) — turi gyventi ilgiau, kitaip logo pabaigoje gauna FLOW_INVALID
        P.start("dec", [BIN, "dec", *common, "--port-base", str(dport), "--duration", str(total + (16 if f == "chain" else 10)),
                        "--out", str(d)])
        time.sleep(2)
        for k in range(n):
            # be -stream_loop: kilpos sandūra praranda IDR (išmatuota); klipas ilgesnis už pakopą (61-make-test-clip.sh)
            P.start(f"send{k}", ["ffmpeg", "-nostdin", "-loglevel", "error", "-re", "-i", a.clip,
                                 "-c", "copy", "-f", "mpegts", f"udp://127.0.0.1:{dport + k}?pkt_size=1316"], "sender")
        time.sleep(3)
        if f == "dec":
            vcommon = list(common)
            if a.storage == "host" or a.impl == "cpu":
                vcommon[vcommon.index("--impl") + 1] = "cpu"
            P.start("verify", [BIN, "verify", *vcommon, "--mode", "dec", "--every", str(a.every), "--duration", str(total),
                               *(["--ref", a.ref] if a.ref else []), "--out", str(d)], "consumer")
        else:
            P.start("gen_audio", [BIN, "gen", *common, "--video", "0", "--audio", "1", "--duration", str(total + 10), "--out", str(d / "audio")], "producer")
            P.start("logo", [BIN, "logo", *common, "--in-role", "4", "--duration", str(total + 6), "--out", str(d)])
            time.sleep(2)
            P.start("sink", [sys.executable, str(REPO / "tests/perf/ts_sink.py"), "--port-base", str(port), "--flows", str(n),
                             "--seconds", str(total + 2), "--warmup", str(W + 2), "--out", str(d / "sink.json")], "sink")
            # pilnas flow: G — NVENC; C — x264 (CPU), tas pats bitrate ir GOP
            enc_args = (["--encoder", "x264", "--x264-preset", a.chain_x264, "--x264-threads", str(a.x264_threads)]
                        if a.impl == "cpu" else ["--encoder", "nvenc", "--preset", "p4", "--tune", "ll"])
            P.start("enc", [BIN, "enc", *common, "--in-role", "2", *enc_args,
                            "--audio", "1", "--port-base", str(port), "--duration", str(total), "--out", str(d)])
    # matavimo langas: CPU tarp warmup pabaigos ir lango pabaigos
    time.sleep(W)
    c0, t0 = P.cpu(), time.monotonic()
    gpu_csv = d / "gpu.csv"
    gpu_probe = start_gpu_probe(gpu_csv) if a.gpu_probe else None
    vram_peak = vram_mb()
    end = t0 + S
    while time.monotonic() < end:
        time.sleep(5)
        v = vram_mb()
        if v and (vram_peak is None or v > vram_peak):
            vram_peak = v
    c1, t1 = P.cpu(), time.monotonic()
    stop_gpu_probe(gpu_probe)
    gpu = gpu_stats(gpu_csv)
    codes = P.stop_all()
    if not domain.remove_domain(dom):  # trinama tik be gyvų savininkų (nuomos <domenas>.vramflow/)
        print(f"[capacity] domenas {dom} turi gyvų savininkų {domain.live_owners(dom)} — nepašalintas", flush=True)
    wall = t1 - t0
    cpu = {k: (c1[k] - c0[k]) / wall for k in c0 if c0.get(k) is not None and c1.get(k) is not None}

    sums = {p.stem.replace("summary-", ""): load(p) for p in d.rglob("summary-*.json")}
    sink = load(d / "sink.json")
    func_name = {"logo": "logo", "dec": "dec", "chain": "enc", "composite": "composite"}.get(f, "enc")
    fs = sums.get(func_name) or {}
    losses = {}
    for name, s in sums.items():
        for k in LOSS_KEYS:
            v = flow_sum(s, [k])
            if v:
                losses[f"{name}.{k}"] = v
    correctness = {}
    for name, s in sums.items():
        if name.startswith("verify"):
            correctness[name] = (s or {}).get("proc", {})
    verr = sum(c.get("verification_errors", 0) for c in correctness.values())
    vcap = sum(c.get("capacity_errors", 0) for c in correctness.values())
    # WSL2 laikrodžio šuoliai: per šuolį praleisti laukai (clock_skips) — aplinkos įvykis, ne pajėgumas. Jų sukelti
    # INVALID laukai downstream toleruojami TIK iki upstream clock_skips sumos (vienam flow'ui), likutis — FAIL.
    clock_fields = 0
    for name, s in sums.items():
        if not s:
            continue
        pr = s.get("proc", {})
        clock_fields += pr.get("clock_skips", 0) + pr.get("warmup_skips", 0)
        # dec paleidimas: kol pildomas atkūrimo buferis, viršutinis laukas skaičiuojamas waiting_buffer, apatinis —
        # warmup_underrun_fields (abu — paleidimo pereinamasis procesas, ne pajėgumas)
        per_flow = [fl.get("clock_skips", 0) + fl.get("warmup_skips", 0) + fl.get("warmup_underrun_fields", 0) +
                    fl.get("waiting_buffer", 0) +
                    2 * (fl.get("clock_skips_frames", 0) + fl.get("warmup_skips_frames", 0)) for fl in s.get("flows", [])]
        clock_fields += max(per_flow) if per_flow else 0
    allowance = clock_fields * n
    for key in [k for k in losses if k.endswith(".in_invalid")]:
        cut = min(losses[key], allowance)
        losses[key] -= cut
        if not losses[key]:
            del losses[key]
    # verify'aus paties atsilikimas (too_late / verify_skipped) — matavimo įrankio, ne funkcijos pajėgumas
    verify_lag = sum(flow_sum(s, ["too_late", "verify_skipped"]) for nm, s in sums.items() if nm.startswith("verify"))
    for key in [k for k in losses if k.startswith("verify") and k.endswith(".too_late")]:
        del losses[key]
    vcap_adj = max(0, vcap - allowance - verify_lag)
    if sink:
        for k in ("cc_errors", "sync_errors"):
            if sink["totals"][k]:
                losses[f"sink.{k}"] = sink["totals"][k]
    # realaus laiko santykis: funkcijos išvesti laukai/kadrai per visą jos veikimą vs laukta
    if func_name in ("logo", "dec", "composite"):
        produced = flow_sum(fs, ["fields_ok"])
        span = (fs.get("last_index", 0) - fs.get("first_index", 0) + 1) * n if fs else 0
    else:
        produced = flow_sum(fs, ["frames_encoded"])
        span = ((fs.get("last_index", 0) - fs.get("first_index", 0) + 1) // 2) * n if fs else 0
    rt = produced / span if span else 0
    hist = fs.get("hist", {})
    res = {
        "func": f, "layers": a.layers if f == "composite" else None, "out_format": a.out_format if f == "composite" else None,
        "deadline_ms": a.deadline_ms if f == "composite" else None, "every": a.every,
        "variant": variant, "impl": a.impl, "storage": a.storage, "threads": a.threads, "source": a.source,
        "n": n, "seconds": S, "exit_codes": codes, "fatal": fs.get("fatal"),
        "realtime_ratio": rt, "produced": produced, "expected": span,
        "losses": losses, "verification_errors": verr, "verify_capacity_errors": vcap, "verify_capacity_errors_adj": vcap_adj,
        "clock_skip_fields": clock_fields, "verify_lag": verify_lag, "verify": correctness,
        "cpu_cores": cpu,
        # composite: šaltinių (gen, gfxgen) ir patikros CPU — informacinis, į funkcijos CPU neįskaičiuojamas
        "cpu_cores_gen_per_stream": (cpu.get("gen", 0) / n) if n and f == "composite" else None,
        "cpu_cores_gfxgen_per_stream": (cpu.get("gfxgen", 0) / n) if n and f == "composite" else None,
        "cpu_cores_verify_per_stream": (cpu.get("verify", 0) / n) if n and f == "composite" else None,
        # chain: visa grandinė (dec + logo + enc), kitur — pati funkcija
        "cpu_cores_func_per_stream": (sum(cpu.get(k, 0) for k in (("dec", "logo", "enc") if f == "chain" else (func_name,))) / n)
        if n else 0,
        "vram_delta_mb": (vram_peak - vram0) if vram_peak and vram0 else None,
        "gpu": gpu, "power_scheme": power_scheme(),
        "hist": hist, "sink": sink["flows"] if sink else None,
        "e2e_latency_p99_ms": max((x["e2e_latency_p99_ms"] or 0) for x in sink["flows"]) if sink and sink["flows"] else None,
    }
    # Realaus laiko santykis skaičiuojamas be PAČIOS funkcijos laikrodžio šuolių / paleidimo praleistų laukų (kitų
    # procesų paleidimas, pvz. dec grandinėje, funkcijos span neliečia). enc span — kadrais, todėl laukai / 2.
    own = 0
    if fs:
        pr = fs.get("proc", {})
        pf = [fl.get("clock_skips", 0) + fl.get("warmup_skips", 0) + fl.get("warmup_underrun_fields", 0) +
              fl.get("waiting_buffer", 0) + 2 * (fl.get("clock_skips_frames", 0) + fl.get("warmup_skips_frames", 0))
              for fl in fs.get("flows", [])]
        own = pr.get("clock_skips", 0) + pr.get("warmup_skips", 0) + (max(pf) if pf else 0)
        if func_name == "enc":
            own //= 2
    res["own_skip_units"] = own
    if span:
        rt = produced / max(1, span - own * n)
        res["realtime_ratio"] = rt
    exit_ok = lambda k, c: c == 0 or (c == 4 and not losses and vcap_adj == 0)  # 4 = tik laikrodžio praradimai
    # "src" (composite: gen/gfxgen) stabdomi SIGTERM pabaigoje — jų išėjimo kodas informacinis (res["src_exit"])
    cats = {name: cat for name, _, cat in P.items}
    procs_ok = all(exit_ok(k, c) for k, c in codes.items() if not k.startswith("send") and cats.get(k) != "src")
    res["src_exit"] = {k: c for k, c in codes.items() if cats.get(k) == "src"}
    if f == "composite":
        res["src_flows_ready"], res["out_flows_ready"] = src_ok, out_ok
        res["composite_to_verify_gap_s"] = round(comp_gap_s, 1)
        if comp_gap_s > COMP_MAX_GAP_S:
            # composite baigtųsi anksčiau už matavimo langą → rezultatas neinterpretuotinas (ne talpos FAIL)
            res["invalid_reason"] = (f"composite starto → verify starto tarpas {comp_gap_s:.1f} s > {COMP_MAX_GAP_S} s: "
                                     "composite pasibaigė anksčiau už matavimo langą")
        # R21: patikros aprėptis — patikrinta mėginių / laukiama (lauko indeksų intervalas × N / --every) ir nuleistos patikros
        vs = next((sm for nm, sm in sums.items() if nm.startswith("verify") and sm), None)
        if vs:
            exp = (vs.get("last_index", 0) - vs.get("first_index", 0) + 1) * n // a.every
            got = vs.get("proc", {}).get("samples_total", 0)
            res["verify_coverage"] = {"samples": got, "expected": exp, "coverage": round(got / exp, 3) if exp else None,
                                      "check_dropped": sum(fl.get("check_dropped", 0) for fl in vs.get("flows", []))}
        procs_ok = procs_ok and src_ok and out_ok
    res["pass"] = procs_ok and not losses and verr == 0 and vcap_adj == 0 and rt >= 0.999
    # PASS_tol: teisingumas be išimčių (0 klaidų, procesai be fatalių klaidų), praradimai ≤ --tol (numatyta 0,2 %) lauktų —
    # atskiria platformos drebėjimą (WSL2/WDDM pavieniai praleidimai) nuo tikros talpos ribos. Skelbiami abu.
    lost = sum(losses.values()) + vcap_adj
    res["loss_fraction"] = lost / max(1, span)
    no_fatal = all(c in (0, 4) for k, c in codes.items() if not k.startswith("send") and cats.get(k) != "src")
    res["pass_tol"] = no_fatal and verr == 0 and res["loss_fraction"] <= a.tol and rt >= 1 - a.tol
    if res.get("invalid_reason"):
        res["pass"] = res["pass_tol"] = False
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--func", required=True)
    ap.add_argument("--n", required=True)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--complexity", type=int, default=2)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--x264-threads", type=int, default=2)
    ap.add_argument("--clip", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--impl", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--storage", default="cuda-linear", choices=["cuda-linear", "host"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.002, help="PASS_tol: leistina praradimų dalis (platformos drebėjimui)")
    ap.add_argument("--chain-x264", default="superfast", help="grandinės C variante naudojamas x264 preset'as")
    ap.add_argument("--source", default="same", choices=["same", "fair"],
                    help="logo šaltinis: same — tas pats variantas (sistema); fair — C variantui GPU→host šaltinis (funkcija)")
    ap.add_argument("--gpu-probe", type=int, default=1, help="1 Hz nvidia-smi GPU/NVENC/NVDEC apkrovos zondas (0 — išjungti)")
    ap.add_argument("--stop-after-fails", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2, choices=[1, 2, 3, 4], help="composite: grafikos sluoksnių skaičius K")
    ap.add_argument("--out-format", default="v210", choices=["v210", "v210a"], help="composite: išėjimo formatas")
    ap.add_argument("--deadline-ms", type=int, default=30, help="composite: įėjimų laukimo riba (R18: kaip logo numatytoji)")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    results, fails = [], 0
    for n in [int(x) for x in a.n.split(",")]:
        variant = "C" if a.impl == "cpu" else ("H" if a.storage == "host" else "G")
        skip = comp_guard(a, n) if a.func == "composite" else None
        if skip:
            r = {"func": a.func, "layers": a.layers, "out_format": a.out_format, "variant": variant, "impl": a.impl,
                 "storage": a.storage, "source": a.source, "n": n, "skipped": True, "skip_reason": skip, "pass": False,
                 "pass_tol": False, "seconds": a.seconds, "power_scheme": power_scheme()}
            results.append(r)
            print(f"[{func_tag(a)}/{variant} N={n:2d}] PRALEISTA: {skip}", flush=True)
            (pathlib.Path(a.out) / f"capacity-{func_tag(a)}-{variant}.json").write_text(json.dumps(results, indent=1))
            continue
        r = run_level(a, n)
        results.append(r)
        if r.get("invalid_reason"):
            print(f"[{func_tag(a)}/{variant} N={n:2d}] NEDERINA: {r['invalid_reason']}", flush=True)
            (pathlib.Path(a.out) / f"capacity-{func_tag(a)}-{variant}.json").write_text(json.dumps(results, indent=1))
            continue
        print(f"[{func_tag(a)}/{variant} N={n:2d}] griežtai={'PASS' if r['pass'] else 'FAIL'} tol={'PASS' if r['pass_tol'] else 'FAIL'} "
              f"praradimai {r['loss_fraction'] * 100:.3f}% RT={r['realtime_ratio']:.4f} "
              f"{r['losses'] or ''} verif={r['verification_errors']} CPU/srautui={r['cpu_cores_func_per_stream']:.3f} "
              f"ΔVRAM={r['vram_delta_mb']} MB "
              + (f"GPU={g['util_gpu']['mean']:.0f}% NVENC={g['util_enc']['mean']:.0f}% NVDEC={g['util_dec']['mean']:.0f}% "
                 f"P={g['power_w']['mean']:.0f}W " if (g := r.get("gpu")) else "")
              + (f"gfxgen/gen CPU={r['cpu_cores_gfxgen_per_stream']:.3f}/{r['cpu_cores_gen_per_stream']:.3f} "
                 f"proc_ms p50/p99={(r['hist'].get('proc_ms') or {}).get('p50_ms')}/{(r['hist'].get('proc_ms') or {}).get('p99_ms')} "
                 f"src_exit={r['src_exit']} " if a.func == "composite" else "")
              + f"e2e p99={r['e2e_latency_p99_ms']} exit={r['exit_codes']} {r['fatal'] or ''}",
              flush=True)
        (pathlib.Path(a.out) / f"capacity-{func_tag(a)}-{variant}.json").write_text(json.dumps(results, indent=1))
        fails = 0 if r["pass_tol"] else fails + 1  # sustojam, kai viršijama talpa (tol kriterijus)
        if fails >= a.stop_after_fails:
            break

    def nstar(key):
        best = 0
        for r in results:
            if not r[key]:
                break
            best = r["n"]
        return best

    print(f"N* ({a.func}/{variant}) griežtai = {nstar('pass')}, tol(≤{a.tol * 100:.1f} %) = {nstar('pass_tol')}")


if __name__ == "__main__":
    main()
