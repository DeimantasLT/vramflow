# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Nestabilių laiko testų kartotuvas (planas docs/planai/2026-11-17-production-ready.md §3.1, G3).

Kartoja vieną testą N kartų ir kiekvienam paleidimui įrašo: būseną, išmatuotas reikšmes iš santraukos ir aplinkos
pėdsaką — GPU taktus, galios būseną, galią ir apkrovą 10 Hz (`nvidia-smi -lms 100`), ar veikė keepwarm. Rezultatas —
`<out>/flake.json` ir skirstinio santrauka stdout.

Naudojimas:
  flake.py fiso_stall --runs 20 [--out DIR] [--field gap_late_commit_max_us] [--prefix chrt -f 50]
  flake.py "ctest:grįžta per < 1 ms" --runs 20 [--regex 'max (-?\\d+) µs']

Testo vardas — funkcinis (`functional.py --only <vardas>`) arba `ctest:<regex>` (`ctest -R`). `--field` — skaitinio lauko
vardo galas santraukos JSON medyje (funkciniams; kartojama), `--regex` — reikšmė iš `ctest -V` išvesties (pirma grupė;
Catch2 `INFO` / `UNSCOPED_INFO` matomi tik kritus testui — praėjusio paleidimo reikšmėms kviesk testo binarą su `-s`
tiesiogiai, kaip `tests/perf/mxl_hop_probe.cpp` matavime). `--prefix` — komanda prieš testą
(pvz. `chrt -f 50`), kad būtų galima atskirti planuoklės hipotezę.
"""
import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpuinfo import nvidia_smi  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GPU_Q = "timestamp,pstate,clocks.gr,clocks.mem,power.draw,utilization.gpu"


class GpuTrace:
    """nvidia-smi 10 Hz fone; `stop()` grąžina santrauką (pstate rinkinys, taktų min / maks., imčių skaičius)."""

    def __init__(self):
        self.rows, self.p = [], None
        try:
            self.p = subprocess.Popen([nvidia_smi(), f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits", "-lms", "100"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 6:
                self.rows.append(parts)

    def stop(self):
        if not self.p:
            return {"samples": 0}
        self.p.terminate()
        self.p.wait(5)
        self.t.join(2)
        num = lambda i: [float(r[i]) for r in self.rows if re.fullmatch(r"[\d.]+", r[i])]  # noqa: E731
        gr, mem, pw = num(2), num(3), num(4)
        return {"samples": len(self.rows), "pstates": sorted({r[1] for r in self.rows}),
                "clk_gr_min": min(gr, default=None), "clk_gr_max": max(gr, default=None),
                "clk_mem_min": min(mem, default=None), "clk_mem_max": max(mem, default=None),
                "power_max": max(pw, default=None)}


def leaves(obj, path=""):
    """Visi skaitiniai JSON medžio lapai kaip (kelias, reikšmė)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


def run_functional(name, out, prefix):
    cmd = prefix + [sys.executable, str(REPO / "tests/functional/functional.py"), str(out), "--only", name]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    try:
        res = json.loads((out / "functional.json").read_text())
    except (OSError, ValueError):
        res = []
    return p.returncode, res, p.stdout[-4000:]


def run_ctest(regex, prefix):
    build = os.environ.get("VERIFY_BUILD", "/root/build/vramflow/func")
    p = subprocess.run(prefix + ["ctest", "-R", regex, "-V"], cwd=build, capture_output=True, text=True)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--field", action="append", default=[])
    ap.add_argument("--regex", default="")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    out = Path(a.out or f"/root/build/vramflow/flake/{re.sub(r'[^A-Za-z0-9_]+', '_', a.test)[:40]}")
    out.mkdir(parents=True, exist_ok=True)
    prefix = shlex.split(a.prefix)
    runs = []
    for i in range(a.runs):
        t0 = time.time()
        tr = GpuTrace()
        rec = {"run": i}
        if a.test.startswith("ctest:"):
            rc, text = run_ctest(a.test[6:], prefix)
            rec["status"] = {a.test: "PASS" if rc == 0 else "FAIL"}
            if a.regex:
                rec["values"] = {a.regex: [float(x) for x in re.findall(a.regex, text)]}
        else:
            rc, res, tail = run_functional(a.test, out / f"r{i:02d}", prefix)
            rec["status"] = {r["id"]: r["status"] for r in res}
            rec["keepwarm"] = sorted({bool(r.get("keepwarm")) for r in res})
            rec["values"] = {f: [v for r in res if r["status"] != "INFO" for k, v in leaves(r) if k.endswith(f)] for f in a.field}
            if rc != 0 and not res:
                rec["error"] = tail
        rec["exit"], rec["seconds"], rec["gpu"] = rc, round(time.time() - t0, 1), tr.stop()
        runs.append(rec)
        bad = [k for k, s in rec["status"].items() if s == "FAIL"]
        print(f"#{i:02d} exit={rc} {rec['seconds']} s FAIL={bad or '-'} gpu={rec['gpu'].get('pstates')} "
              f"clk={rec['gpu'].get('clk_gr_min')}..{rec['gpu'].get('clk_gr_max')}"
              + "".join(f" {k}={v}" for k, v in rec.get("values", {}).items()), flush=True)
    (out / "flake.json").write_text(json.dumps({"test": a.test, "prefix": a.prefix, "runs": runs}, indent=1))
    ids = sorted({k for r in runs for k in r["status"]})
    print(f"== {a.test}: {a.runs} paleidimų, išvestis {out}/flake.json")
    for k in ids:
        n_fail = sum(r["status"].get(k) == "FAIL" for r in runs)
        if n_fail or len(ids) <= 12:
            print(f"   {k}: FAIL {n_fail} / {sum(k in r['status'] for r in runs)}")
    for f in (a.field or ([a.regex] if a.regex else [])):
        vals = [v for r in runs for v in r.get("values", {}).get(f, [])]
        if vals:
            q = statistics.quantiles(vals, n=20) if len(vals) >= 2 else vals
            print(f"   {f}: n={len(vals)} min={min(vals)} p50={statistics.median(vals)} p95={q[-1]} max={max(vals)}")
    return 1 if any(s == "FAIL" for r in runs for s in r["status"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
