#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""`ingest` talpa ir sąnaudos (planas §P3 „Matavimai“): N sintetinių TS šaltinių per UDP → vienas `ingest` procesas → `verify --mode
ingest` (visi N). Kiekvienai pakopai: ar praėjo (kiekvienam flow'ui ≥ 99,8 % grain'ų, 0 pavėluotų, |A/V| ≤ 1 laukas / kadras),
NVDEC % ir SM % (nvidia-smi 1 Hz, vidurkis), VRAM (vidurkis − prieš paleidimą), ingest proceso CPU branduoliai, pateikimo
vėlavimas (out_lateness p50 / p99 — commit'as po T(i)), A/V vidurkis / maksimumas.

Naudojimas: ingest_capacity.py OUT --format 1080i50 --decoder nvdec --levels 1,2,4,8 [--seconds 30] [--lead-ms 400]
            [--clips ~/media-test/av]
Išvestis: OUT/ingest_capacity.json ir eilutės stdout. Tik sintetinė medžiaga, tik loopback.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capacity  # noqa: E402  cpu_s, vram_mb, start_gpu_probe, gpu_stats
import domain  # noqa: E402
import keepwarm  # noqa: E402

FUNC_BIN = os.environ.get("FUNC_BIN", str(pathlib.Path.home() / "build/vramflow/func/vramflow-func"))
LIB = os.environ.get("MXL_LIB", str(pathlib.Path.home() / "src/mxl-pr653/install/Linux-Clang-Release/lib"))
ENV = dict(os.environ, LD_LIBRARY_PATH=LIB)


def run_level(a, n, kw):
    tag = f"ingcap-{a.format}-{a.decoder}-{a.lead_ms}-{n}"
    d = str(domain.make_domain(tag))
    out = pathlib.Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    cpu = a.decoder == "sw"
    impl = ["--impl", "cpu", "--storage", "host"] if cpu else ["--impl", "gpu", "--storage", "cuda-linear"]
    kw.want(not cpu)
    time.sleep(1.0)
    vram0 = capacity.vram_mb()
    ports = [a.port_base + k for k in range(n)]
    # --format mix: pirma pusė 1080i50, antra 1080p25 (US 12: 4 × 1080i50 + 4 × 1080p25 viename procese)
    fmts = ["1080i50" if k < (n + 1) // 2 else "1080p25" for k in range(n)] if a.format == "mix" else [a.format] * n
    args = []
    for p in ports:
        args += ["--input", f"udp://127.0.0.1:{p}"]
    warm, T = 8, a.seconds
    vimpl = ["--impl", "cpu", "--storage", "host"] if cpu else ["--impl", "gpu", "--storage", "cuda-linear"]
    ing = subprocess.Popen([FUNC_BIN, "ingest", "--domain", d, *args, "--decoder", "sw" if cpu else a.decoder, *impl,
                            "--decode-lead-ms", str(a.lead_ms), "--duration", str(T + warm + 10), "--warmup", str(warm), "--out", str(out)],
                           stdout=subprocess.DEVNULL, stderr=open(out / "ingest.err", "w"), env=ENV)
    snd = [subprocess.Popen([sys.executable, str(HERE / "ts_burst.py"), str(pathlib.Path(a.clips) / f"av_{f}.ts"), "--port", str(p),
                             "--duration", str(T + warm + 12)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
           for p, f in zip(ports, fmts)]
    groups = []  # (pirmas flow'as, kiekis, formatas) — verify tikrina vieną formatą procese
    for k, f in enumerate(fmts):
        if groups and groups[-1][2] == f:
            groups[-1][1] += 1
        else:
            groups.append([k, 1, f])
    vfs = []
    for g0, gn, f in groups:
        vo = out / f"v{g0}"
        vo.mkdir(exist_ok=True)
        vfs.append(subprocess.Popen([FUNC_BIN, "verify", "--domain", d, "--mode", "ingest", "--flows", str(gn), "--flow-base", str(g0),
                                     "--duration", str(T + warm), "--warmup", str(warm), "--lag-fields", "3", "--audio", "1",
                                     "--input-wait-ms", "30000", *vimpl, "--out", str(vo)],
                                    stdout=subprocess.DEVNULL, stderr=open(vo / "verify.err", "w"), env=ENV))
    time.sleep(warm + 2)
    probe = capacity.start_gpu_probe(out / "gpu.csv")
    c0, t0 = capacity.cpu_s(ing.pid), time.monotonic()
    time.sleep(max(1, T - 4))
    c1, t1 = capacity.cpu_s(ing.pid), time.monotonic()
    capacity.stop_gpu_probe(probe)
    rv = [vf.wait(timeout=T + 120) for vf in vfs]
    ri = ing.wait(timeout=T + 120)
    for s_ in snd:
        s_.terminate()
        s_.wait()
    domain.remove_domain(d)
    si = json.loads((out / "summary-ingest.json").read_text()) if (out / "summary-ingest.json").exists() else {}
    vflows, vpairs = [{}] * n, [[]] * n
    for g0, gn, f in groups:
        pth = out / f"v{g0}" / "summary-verify-ingest.json"
        sv = json.loads(pth.read_text()) if pth.exists() else {}
        for j in range(gn):
            if j < len(sv.get("flows") or []):
                vflows[g0 + j] = sv["flows"][j]
            if j < len(sv.get("ingest") or []):
                vpairs[g0 + j] = sv["ingest"][j].get("pairs") or []
    g = capacity.gpu_stats(out / "gpu.csv") or {}
    ok_flows, offs = 0, []
    for k in range(n):
        limit = 20.0 if fmts[k] in ("1080i50", "1080p25") else 10.0
        fv = vflows[k]
        fi = (si.get("flows") or [{}] * n)[k] if k < len(si.get("flows") or []) else {}
        tot = sum(fv.get(x, 0) for x in ("present", "missing", "invalid", "too_late"))
        o = [q[1] / 1000.0 for q in vpairs[k]]
        offs += o
        good = tot > 0 and fv.get("present", 0) / tot >= 0.998 and fi.get("late_frames", 1) == 0 and o and max(abs(x) for x in o) <= limit
        ok_flows += 1 if good else 0
    h = (si.get("hist") or {}).get("out_lateness") or {}
    vram = (g.get("mem_used_mb") or {}).get("mean")
    res = {"n": n, "formats": sorted(set(fmts)), "pass": ok_flows == n and ri in (0, 4), "ok_flows": ok_flows, "ingest_exit": ri, "verify_exit": rv,
           "nvdec_pct": (g.get("util_dec") or {}).get("mean"), "sm_pct": (g.get("util_gpu") or {}).get("mean"),
           "vram_mb": None if vram is None or vram0 is None else round(vram - vram0, 1),
           "cpu_cores": round((c1 - c0) / (t1 - t0), 3) if c0 is not None and c1 is not None else None,
           "late_p50_ms": h.get("p50_ms"), "late_p99_ms": h.get("p99_ms"),
           "av_mean_ms": round(sum(offs) / len(offs), 3) if offs else None, "av_max_ms": round(max(abs(x) for x in offs), 3) if offs else None,
           "late_frames": sum(f.get("late_frames", 0) for f in si.get("flows") or []),
           "skip_to_key": sum(f.get("skip_to_key", 0) for f in si.get("flows") or [])}
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--format", default="1080i50", help="1080i50 | 1080p25 | mix (pusė 1080i50, pusė 1080p25)")
    ap.add_argument("--decoder", default="nvdec", choices=["nvdec", "cuvid", "sw"])
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--lead-ms", type=int, default=400)
    ap.add_argument("--port-base", type=int, default=27000)
    ap.add_argument("--clips", default=str(pathlib.Path.home() / "media-test/av"))
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    kw = keepwarm.KeepWarm(FUNC_BIN, ENV)
    kw.log_dir = pathlib.Path(a.out)
    results = []
    try:
        for n in [int(x) for x in a.levels.split(",")]:
            r = run_level(a, n, kw)
            results.append(r)
            if not r["pass"]:
                break
    finally:
        kw.stop()
    (pathlib.Path(a.out) / f"ingest_capacity_{a.format}_{a.decoder}_{a.lead_ms}.json").write_text(
        json.dumps({"format": a.format, "decoder": a.decoder, "lead_ms": a.lead_ms, "seconds": a.seconds, "levels": results}, indent=1))
    passed = [r["n"] for r in results if r["pass"]]
    print(f"N* = {max(passed) if passed else 0}")


if __name__ == "__main__":
    main()
