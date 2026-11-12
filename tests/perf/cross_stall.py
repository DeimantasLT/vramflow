#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Ar sustabdytas (SIGSTOP) CUDA procesas vėlina KITĄ procesą toje pačioje GPU? (fiso_stall[G] priežasties tyrimas,
docs/rezultatai/2026-11-12-izoliacija-po.md.)

Paleidžiami tik du gamintojai, kaip fiso_stall: gen A (7 flow'ai, „sveikas“) ir gen B (1 flow'as, kitas --flow-base);
B sustabdomas ties 10 s 5 s laikui ir tęsiamas. Matuojama A: `late_commits` (commit'ai vėlesni nei --deadline-ms 30),
didžiausias commit'o vėlavimas ir vėlaus tick'o laikas nuo SIGSTOP / SIGCONT. Konfigūracijos (A / B variantas):
  G/G — abu GPU + cuda-linear (kaip fiso_stall[G]);
  C/G — A CPU + host, B GPU + cuda-linear (A GPU nenaudoja);
  H/G — A GPU + host saugykla, B GPU + cuda-linear;
  G/C — A GPU, sustabdomas B CPU (sustabdomas ne CUDA procesas);
  G/G-none — be gedimo (kontrolė).
Naudojimas: VERIFY_BIN=… FUNC_BIN=… python3 tests/perf/cross_stall.py <out> [kartai=3]
"""
import json
import pathlib
import re
import signal
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "functional"))
import functional as F  # noqa: E402

VAR = {"G": ("gpu", "cuda-linear"), "H": ("gpu", "host"), "C": ("cpu", "host")}
CONFIGS = [("G", "G", True), ("C", "G", True), ("H", "G", True), ("G", "C", True), ("G", "G", False)]
LATE_RE = re.compile(r"\[gen\] vėlus commit: indeksas (\d+), ([\d.]+) ms po T\(i\), TAI (\d+)")


def run(out, a, b, fault, k):
    sub = f"xstall-{a}{b}-{'stall' if fault else 'none'}-{k}"
    d = F.fresh_domain(sub)
    ga = F.FProc(out, f"{sub}/A", "gen", d, "--flows", 7, "--duration", 20, "--warmup", 2, *F.vcommon(*VAR[a]))
    gb = F.FProc(out, f"{sub}/B", "gen", d, "--flows", 1, "--flow-base", 7, "--duration", 40, "--warmup", 2, *F.vcommon(*VAR[b]))
    t = {}
    try:
        time.sleep(10)
        if fault:
            t["stop"] = time.clock_gettime_ns(time.CLOCK_TAI)
            gb.p.send_signal(signal.SIGSTOP)
            time.sleep(5)
            t["cont"] = time.clock_gettime_ns(time.CLOCK_TAI)
            gb.p.send_signal(signal.SIGCONT)
        ga.wait(timeout=60)
    finally:
        for pr in (gb, ga):
            if pr.p.poll() is None:
                pr.p.send_signal(signal.SIGCONT)
                pr.term()
    s = ga.summary() or {}
    late = [{"late_ms": float(m.group(2)), **{f"vs_{x}_ms": round((int(m.group(3)) - v) / 1e6, 1) for x, v in t.items()}}
            for m in LATE_RE.finditer(ga.err_path.read_text(encoding="utf-8", errors="replace"))]
    h = (s.get("hist") or {}).get("commit_lateness") or {}
    return {"cfg": f"{a}/{b}{'' if fault else '-none'}", "late_commits": (s.get("proc") or {}).get("late_commits", 0),
            "late_skips": (s.get("proc") or {}).get("late_skips", 0), "p99_ms": h.get("p99_ms"), "max_ms": h.get("max_ms"),
            "late_ticks": late[:5]}


def main():
    out = pathlib.Path(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b, fault in CONFIGS:
        for k in range(reps):
            r = run(out, a, b, fault, k)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    (out / "cross_stall.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
