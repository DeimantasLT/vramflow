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
            if cat == "sender":
                continue
            try:
                codes[name] = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                codes[name] = "killed"
        for name, p, cat in self.items:
            if cat != "sender":
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


def run_level(a, n):
    variant = "C" if a.impl == "cpu" else ("H" if a.storage == "host" else "G")
    d = pathlib.Path(a.out) / f"{a.func}-{variant}-n{n}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    dom = f"/dev/shm/mxlf-{a.func}-{n}"
    shutil.rmtree(dom, ignore_errors=True)
    os.makedirs(dom)
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
    shutil.rmtree(dom, ignore_errors=True)
    wall = t1 - t0
    cpu = {k: (c1[k] - c0[k]) / wall for k in c0 if c0.get(k) is not None and c1.get(k) is not None}

    sums = {p.stem.replace("summary-", ""): load(p) for p in d.rglob("summary-*.json")}
    sink = load(d / "sink.json")
    func_name = {"logo": "logo", "dec": "dec", "chain": "enc"}.get(f, "enc")
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
    if func_name in ("logo", "dec"):
        produced = flow_sum(fs, ["fields_ok"])
        span = (fs.get("last_index", 0) - fs.get("first_index", 0) + 1) * n if fs else 0
    else:
        produced = flow_sum(fs, ["frames_encoded"])
        span = ((fs.get("last_index", 0) - fs.get("first_index", 0) + 1) // 2) * n if fs else 0
    rt = produced / span if span else 0
    func_exit = codes.get({"logo": "logo", "dec": "dec"}.get(f, "enc"))
    hist = fs.get("hist", {})
    res = {
        "func": f, "variant": variant, "impl": a.impl, "storage": a.storage, "threads": a.threads, "source": a.source,
        "n": n, "seconds": S, "exit_codes": codes, "fatal": fs.get("fatal"),
        "realtime_ratio": rt, "produced": produced, "expected": span,
        "losses": losses, "verification_errors": verr, "verify_capacity_errors": vcap, "verify_capacity_errors_adj": vcap_adj,
        "clock_skip_fields": clock_fields, "verify_lag": verify_lag, "verify": correctness,
        "cpu_cores": cpu,
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
    procs_ok = all(exit_ok(k, c) for k, c in codes.items() if not k.startswith("send"))
    res["pass"] = procs_ok and not losses and verr == 0 and vcap_adj == 0 and rt >= 0.999
    # PASS_tol: teisingumas be išimčių (0 klaidų, procesai be fatalių klaidų), praradimai ≤ 0,1 % lauktų —
    # atskiria platformos drebėjimą (WSL2/WDDM pavieniai praleidimai) nuo tikros talpos ribos. Skelbiami abu.
    lost = sum(losses.values()) + vcap_adj
    res["loss_fraction"] = lost / max(1, span)
    no_fatal = all(c in (0, 4) for k, c in codes.items() if not k.startswith("send"))
    res["pass_tol"] = no_fatal and verr == 0 and res["loss_fraction"] <= a.tol and rt >= 1 - a.tol
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
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    results, fails = [], 0
    for n in [int(x) for x in a.n.split(",")]:
        r = run_level(a, n)
        results.append(r)
        variant = "C" if a.impl == "cpu" else ("H" if a.storage == "host" else "G")
        print(f"[{a.func}/{variant} N={n:2d}] griežtai={'PASS' if r['pass'] else 'FAIL'} tol={'PASS' if r['pass_tol'] else 'FAIL'} "
              f"praradimai {r['loss_fraction'] * 100:.3f}% RT={r['realtime_ratio']:.4f} "
              f"{r['losses'] or ''} verif={r['verification_errors']} CPU/srautui={r['cpu_cores_func_per_stream']:.3f} "
              f"ΔVRAM={r['vram_delta_mb']} MB "
              + (f"GPU={g['util_gpu']['mean']:.0f}% NVENC={g['util_enc']['mean']:.0f}% NVDEC={g['util_dec']['mean']:.0f}% "
                 f"P={g['power_w']['mean']:.0f}W " if (g := r.get("gpu")) else "")
              + f"e2e p99={r['e2e_latency_p99_ms']} exit={r['exit_codes']} {r['fatal'] or ''}",
              flush=True)
        (pathlib.Path(a.out) / f"capacity-{a.func}-{variant}.json").write_text(json.dumps(results, indent=1))
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
