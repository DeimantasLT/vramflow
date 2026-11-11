#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Skalavimas ir soak (architektūra §6.4, §7): 1 writer procesas su N flow'ų + 1 reader procesas su N flow'ų.

Naudojimas:
  scale.py <out> --storage cuda-linear --n 1,4,8,16,32 --seconds 120 --warmup 10
  scale.py <out> --storage cuda-linear --n 16 --seconds 7200 --warmup 300 --label soak
Rezultatas: <out>/<label>-<storage>.json (vienas įrašas kiekvienam N) ir neapdoroti CSV kataloguose.
"""
import argparse
import csv
import json
import os
import pathlib
import shutil
import signal
import statistics
import subprocess
import time

import domain  # tests/perf/domain.py

REPO = pathlib.Path(__file__).resolve().parents[2]
BIN = os.environ.get("VERIFY_BIN", str(pathlib.Path.home() / "build/vramflow/release/vramflow-verify"))
LIB = os.environ.get("MXL_LIB", str(pathlib.Path.home() / "src/mxl-pr653/install/Linux-Clang-Release/lib"))
ENV = dict(os.environ, LD_LIBRARY_PATH=LIB)
TH = json.loads((REPO / "config/thresholds.json").read_text())


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def cpu_cores(proc_rows, warm_s):
    """Vidutiniai branduoliai po warmup iš getrusage (user+sys ms) ir monotoninio laiko."""
    rows = [r for r in proc_rows if r["cpu_user_ms"]]
    if len(rows) < 3:
        return None
    t0 = int(rows[0]["ts_mono_ns"])
    rows = [r for r in rows if (int(r["ts_mono_ns"]) - t0) / 1e9 >= warm_s] or rows
    a, b = rows[0], rows[-1]
    cpu_ms = (int(b["cpu_user_ms"]) + int(b["cpu_sys_ms"])) - (int(a["cpu_user_ms"]) + int(a["cpu_sys_ms"]))
    wall_ms = (int(b["ts_mono_ns"]) - int(a["ts_mono_ns"])) / 1e6
    return cpu_ms / wall_ms if wall_ms > 0 else None


def clock_steps(proc_rows, limit_ns=50_000_000):
    vals = [int(r["tai_minus_mono_ns"]) for r in proc_rows]
    return [(i, (b - a) / 1e6) for i, (a, b) in enumerate(zip(vals, vals[1:])) if abs(b - a) > limit_ns]


def drift(proc_rows, field, warm_s, window_s=600):
    rows = [r for r in proc_rows if r[field] not in ("", None)]
    t0 = int(rows[0]["ts_mono_ns"])
    t = [(int(r["ts_mono_ns"]) - t0) / 1e9 for r in rows]
    v = [float(r[field]) for r in rows]
    post = [(ti, vi) for ti, vi in zip(t, v) if ti >= warm_s]
    if len(post) < 4:
        return None
    end = post[-1][0]
    first = [vi for ti, vi in post if ti <= post[0][0] + window_s]
    last = [vi for ti, vi in post if ti >= end - window_s]
    return {"first_median": statistics.median(first), "last_median": statistics.median(last),
            "delta": statistics.median(last) - statistics.median(first)}


def run_level(out, storage, n, seconds, warm, history_ms, rate_num=50):
    d = out / f"{storage}-n{n}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    dom = domain.make_domain(f"mxlv-scale-{storage}-{n}")  # unikalus (planas §P6)
    if history_ms:
        (dom / "options.json").write_text(json.dumps({"urn:x-mxl:option:history_duration/v1.0": history_ms * 1_000_000}))
    common = ["--domain", str(dom), "--storage", storage, "--flows", str(n), "--warmup", str(warm),
              "--rate-num", str(rate_num), "--rate-den", "1"]
    with open(d / "writer.err", "w") as we, open(d / "reader.err", "w") as re_, open(d / "nvidia-smi.csv", "w") as sm:
        smi = subprocess.Popen(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
                                "--format=csv", "-l", "5"], stdout=sm, stderr=subprocess.DEVNULL)
        w = subprocess.Popen([BIN, "writer", *common, "--duration", str(seconds + 8), "--gc-before-create", "--out", str(d)],
                             stderr=we, env=ENV)
        time.sleep(3)
        r = subprocess.Popen([BIN, "reader", *common, "--duration", str(seconds), "--out", str(d),
                              "--host-crosscheck-every", "500"], stderr=re_, env=ENV)
        rcode = r.wait()
        w.send_signal(signal.SIGTERM)
        wcode = w.wait()
        smi.terminate()
        smi.wait()
    domain.remove_domain(dom)
    ws = json.loads((d / "summary-writer.json").read_text())
    rs = json.loads((d / "summary-reader.json").read_text())
    return d, ws, rs, rcode, wcode


def invalid_committed(ws, rs):
    """INVALID grain'ai, kuriuos writer'is TIKRAI commit'ino — MXL teisingumo klaida, ne pajėgumas."""
    bad = 0
    for wf, rf in zip(ws["flows"], rs["flows"]):
        ranges = wf.get("committed", [])
        for idx, bits, _ in rf["errors"]:
            if bits & (1 << 9) and any(a <= idx <= z for a, z in ranges):
                bad += 1
    return bad


def vram_delta_mb(d):
    """Įrenginio VRAM prieaugis nuo testo pradžios (nvidia-smi): max(memory.used) − pirmas matavimas."""
    try:
        rows = [l.split(",") for l in (d / "nvidia-smi.csv").read_text().splitlines()[1:] if l.strip()]
        used = [int(r[1].strip().split()[0]) for r in rows]
        return max(used) - used[0] if used else None
    except (OSError, ValueError, IndexError):
        return None


def summarize(d, ws, rs, n, seconds, warm, rate=50.0):
    rf, wf = rs["flows"], ws["flows"]
    tot = lambda key, fl: sum(f["counters"][key] for f in fl)
    ml, mlw = rs.get("merged_latency", {}), ws.get("merged_latency", {})
    c2v, e2e, jit_m = ml.get("commit_to_verified", {}), ml.get("origin_to_verified", {}), mlw.get("period_jitter", {})
    lags = [f.get("lag_grains", {}) for f in rf]
    preader, pwriter = read_csv(d / "proc-reader.csv"), read_csv(d / "proc-writer.csv")
    verified = tot("grains_ok", rf)
    expected = seconds * rate * n
    res = {
        "n": n, "seconds": seconds,
        "verification_errors": rs["totals"]["verification_errors"], "capacity_errors": rs["totals"]["capacity_errors"],
        "grains_verified": verified, "grains_expected": expected, "verified_fraction": verified / expected,
        "gap_grains": tot("gap_grains", rf), "grains_invalid": tot("grains_invalid", rf), "stalls": tot("stalls", rf),
        "reopens": tot("reopens", rf), "lapped_discarded": tot("lapped_discarded", rf), "cuda_errors": tot("cuda_errors", rf) + tot("cuda_errors", wf),
        "host_crosschecks": tot("host_crosschecks", rf), "host_crosscheck_fail": tot("host_crosscheck_fail", rf),
        "writer_late_skips": tot("writer_late_skips", wf), "writer_clock_backward": tot("clock_backward", wf),
        "invalid_committed": invalid_committed(ws, rs),
        "coverage_min": min(f.get("coverage", 0) for f in rf),
        # Sujungtos histogramos (ne medianų mediana). c2v imama tik kai head == index; jos imties dalis rodo šališkumą.
        "c2v_p50_us": c2v.get("p50_us"), "c2v_p99_us": c2v.get("p99_us"), "c2v_p999_us": c2v.get("p999_us"),
        "c2v_max_us": c2v.get("max_us"), "c2v_sample_fraction": (c2v.get("count", 0) / verified) if verified else 0,
        # e2e (writer'io fill pradžia → patikrinta) imama KIEKVIENAM grain'ui — nešališka metrika
        "e2e_p50_us": e2e.get("p50_us"), "e2e_p99_us": e2e.get("p99_us"), "e2e_p999_us": e2e.get("p999_us"),
        "lag_grains_p99_max": max((x.get("p99", 0) for x in lags), default=0),
        "writer_jitter_p99_us": jit_m.get("p99_us"),
        "writer_gpu_op_p50_us": mlw.get("gpu_op", {}).get("p50_us"),
        "reader_gpu_op_p50_us": ml.get("gpu_op", {}).get("p50_us"),
        "cpu_cores_writer": cpu_cores(pwriter, warm), "cpu_cores_reader": cpu_cores(preader, warm),
        "rss_mb_writer": int(pwriter[-1]["rss_kb"]) / 1024, "rss_mb_reader": int(preader[-1]["rss_kb"]) / 1024,
        # įrenginio VRAM (įskaitant ekraną ir kitus procesus); vram_delta_mb — prieaugis nuo testo pradžios
        "vram_used_mb": max(int(r["vram_total_mb"]) - int(r["vram_free_mb"]) for r in preader if r["vram_total_mb"] != "0"),
        "vram_delta_mb": vram_delta_mb(d),
        "clock_steps": clock_steps(pwriter),
        "reader_exit": rs["exit_code"], "writer_exit": ws["exit_code"],
    }
    res["cpu_cores_per_flow"] = ((res["cpu_cores_writer"] or 0) + (res["cpu_cores_reader"] or 0)) / n
    checks = {
        "verification_errors": res["verification_errors"] == TH["verification_errors"],
        "host_crosscheck_fail": res["host_crosscheck_fail"] == 0,
        "cuda_errors": res["cuda_errors"] == TH["cuda_errors"],
        "gap_grains": res["gap_grains"] <= TH["gap_grains"],
        "stalls": res["stalls"] <= TH["stalls"],
        "reopens": res["reopens"] <= TH["reopens"],
        "verified_fraction": res["verified_fraction"] >= TH["verified_fraction_min"],
        "c2v_p50": res["c2v_p50_us"] is not None and res["c2v_p50_us"] <= TH["lat_commit_to_verified_p50_us"],
        "c2v_p99": res["c2v_p99_us"] is not None and res["c2v_p99_us"] <= TH["lat_commit_to_verified_p99_us"],
        "c2v_p999": res["c2v_p999_us"] is not None and res["c2v_p999_us"] <= TH["lat_commit_to_verified_p999_us"],
        "writer_jitter_p99": res["writer_jitter_p99_us"] is not None and res["writer_jitter_p99_us"] <= TH["writer_period_jitter_p99_us"],
    }
    res["checks"] = checks
    checks["invalid_committed"] = res["invalid_committed"] == 0
    res["correctness_pass"] = (checks["verification_errors"] and checks["host_crosscheck_fail"] and checks["cuda_errors"]
                               and checks["invalid_committed"])
    checks["lapped"] = res["lapped_discarded"] == 0
    res["capacity_pass"] = all(checks[k] for k in ("gap_grains", "stalls", "reopens", "verified_fraction", "lapped"))
    res["latency_pass"] = all(checks[k] for k in ("c2v_p50", "c2v_p99", "c2v_p999", "writer_jitter_p99"))
    return res


def soak_checks(d, res, warm):
    s = TH["soak"]
    pw, pr = read_csv(d / "proc-writer.csv"), read_csv(d / "proc-reader.csv")
    out = {}
    for role, rows in (("writer", pw), ("reader", pr)):
        rss = drift(rows, "rss_kb", warm)
        fds = drift(rows, "fd_count", warm)
        vram = drift(rows, "vram_free_mb", warm)
        out[role] = {"rss": rss, "fd": fds, "vram_free": vram}
        out[f"{role}_rss_ok"] = rss is not None and rss["delta"] <= s["rss_growth_max_frac"] * rss["first_median"]
        out[f"{role}_fd_ok"] = fds is not None and fds["delta"] <= s["fd_growth_max"]
        out[f"{role}_vram_ok"] = vram is not None and -vram["delta"] <= s["vram_growth_max_mb"]
    # latencijos dreifas: P99 per sekundę, pirma vs paskutinė valanda (ar pusė, jei trumpiau)
    rows = []
    for f in sorted(d.glob("flow-*-reader.csv")):
        rows += [r for r in read_csv(f) if r["lat_c2v_p99_us"]]
    if rows:
        t0 = min(int(r["ts_mono_ns"]) for r in rows)
        span = (max(int(r["ts_mono_ns"]) for r in rows) - t0) / 1e9
        win = min(3600, (span - warm) / 2)
        first = [float(r["lat_c2v_p99_us"]) for r in rows if warm <= (int(r["ts_mono_ns"]) - t0) / 1e9 < warm + win]
        last = [float(r["lat_c2v_p99_us"]) for r in rows if (int(r["ts_mono_ns"]) - t0) / 1e9 >= span - win]
        if first and last:
            out["p99_first_median_us"] = statistics.median(first)
            out["p99_last_median_us"] = statistics.median(last)
            out["p99_drift_ok"] = statistics.median(last) <= s["p99_drift_max_ratio"] * statistics.median(first)
    out["clock_steps_ok"] = len(res["clock_steps"]) <= s["max_clock_steps"]
    out["pass"] = all(v for k, v in out.items() if k.endswith("_ok"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--storage", default="cuda-linear")
    ap.add_argument("--n", default="1,4,8,16,32")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--history-ms", type=int, default=0)
    ap.add_argument("--label", default="scale")
    ap.add_argument("--rate", type=int, default=50, help="kadrų per sekundę (sveikas, --rate-den 1)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in [int(x) for x in a.n.split(",")]:
        d, ws, rs, rc, wc = run_level(out, a.storage, n, a.seconds, a.warmup, a.history_ms, a.rate)
        res = summarize(d, ws, rs, n, a.seconds, a.warmup, a.rate)
        if a.label == "soak":
            res["soak"] = soak_checks(d, res, a.warmup)
        results.append(res)
        print(f"[{a.storage} N={n:2d}] teisingumas={'PASS' if res['correctness_pass'] else 'FAIL'} "
              f"pajėgumas={'PASS' if res['capacity_pass'] else 'FAIL'} latencija={'PASS' if res['latency_pass'] else 'FAIL'} "
              f"verif={res['verification_errors']} patikrinta={res['grains_verified']}/{res['grains_expected']} "
              f"gap={res['gap_grains']} lapped={res['lapped_discarded']} "
              f"c2v p50/p99/p99.9={res['c2v_p50_us']}/{res['c2v_p99_us']}/{res['c2v_p999_us']} us (imtis {res['c2v_sample_fraction']:.0%}) "
              f"e2e p50/p99={res['e2e_p50_us']}/{res['e2e_p99_us']} us "
              f"CPU/flow={res['cpu_cores_per_flow']:.3f} ΔVRAM={res['vram_delta_mb']} MB clock_steps={len(res['clock_steps'])}"
              + (f" soak={'PASS' if res['soak']['pass'] else 'FAIL'}" if a.label == "soak" else ""), flush=True)
        (out / f"{a.label}-{a.storage}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
