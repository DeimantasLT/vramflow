#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""GPU prieš CPU palyginimo ataskaita iš capacity-*.json (architektūra 2026-11-11-gpu-vs-cpu-ir-upstream.md §1.6).

Naudojimas: compare_report.py <out.md> <katalogas> [<katalogas> ...]
Surenka visus capacity-<func>-<V>[.all].json, grupuoja pagal (funkcija, variantas, šaltinis), skaičiuoja N* (griežtą ir
tolerancijos), naudos santykį N*_G / N*_C ir CPU branduolius srautui. Kiekvienas skaičius atsekamas iki pakopos katalogo.
"""
import glob
import json
import os
import sys
from collections import defaultdict

VARIANT_NAME = {"C": "C — CPU + host flow", "H": "H — GPU + host flow (MXL v1.1)", "G": "G — GPU + cuda-linear (PR #653)"}


def gcol(r):
    """GPU apkrova matavimo lange (vidurkis, 1 Hz nvidia-smi); be zondo — „–“."""
    g = r.get("gpu")
    if not g:
        return "– | – | – | –"
    return " | ".join(f"{g[k]['mean']:.0f}" for k in ("util_gpu", "util_enc", "util_dec", "power_w"))


def by_n(rows, key):
    """Pakopa N laikoma PASS, jei praėjo bent vienas jos pakartojimas; grąžina {N: (praėjo, iš viso)}."""
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["n"]][0] += bool(r.get(key))
        agg[r["n"]][1] += 1
    return dict(sorted(agg.items()))


def nstar(rows, key):
    best = 0
    for n, (ok, _) in by_n(rows, key).items():
        if not ok:
            break
        best = n
    return best


def main():
    out = sys.argv[1]
    rows = []
    for d in sys.argv[2:]:
        for p in glob.glob(os.path.join(d, "capacity-*.json")):
            for r in json.load(open(p)):
                r["_src"] = os.path.relpath(p)
                rows.append(r)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["func"], r.get("variant", "G"), r.get("source", "same"))].append(r)
    md = ["# GPU prieš CPU — talpos palyginimas", "",
          "PASS griežtai = 0 praradimų ir 0 teisingumo klaidų; PASS tol = 0 teisingumo klaidų ir praradimai ≤ 0,2 % "
          "(atskiria pavienius platformos sutrikimus nuo talpos ribos). CPU — funkcijos proceso branduoliai srautui "
          "(be šaltinio ir patikros). Šaltinis „fair“ — C variantui naudojamas pigus GPU→host generatorius, kad būtų matuojama "
          "funkcija, o ne sintetinis CPU generatorius.", ""]
    summary = defaultdict(dict)
    for (func, v, src), rs in sorted(groups.items()):
        # pasikartojančios pakopos: imamas paskutinis matavimas, bet visi parodomi lentelėje
        md += [f"## {func} — {VARIANT_NAME.get(v, v)} (šaltinis: {src})", "",
               "| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | CPU/srautui | ΔVRAM MB | GPU % | NVENC % | NVDEC % | W | šaltinis |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: r["n"]):
            md.append(f"| {r['n']} | {'PASS' if r['pass'] else 'FAIL'} | {'PASS' if r.get('pass_tol') else 'FAIL'} | "
                      f"{r.get('loss_fraction', 0) * 100:.3f} | {r['realtime_ratio']:.4f} | {r['verification_errors']} | "
                      f"{r['cpu_cores_func_per_stream']:.3f} | {r.get('vram_delta_mb')} | {gcol(r)} | `{r['_src']}` |")
        ns, nt = nstar(rs, "pass"), nstar(rs, "pass_tol")
        reps = {n: m for n, (_, m) in by_n(rs, "pass_tol").items() if m > 1}
        if reps:
            md += ["", "Pakartotos pakopos (PASS, jei praėjo ≥1 pakartojimas; visi rodomi lentelėje): " +
                   ", ".join(f"N={n}: {by_n(rs, 'pass_tol')[n][0]}/{m} tol" for n, m in reps.items())]
        top = max(r["n"] for r in rs)
        ok = [r for r in rs if r.get("pass_tol")]
        cpu1 = min((r["cpu_cores_func_per_stream"] for r in ok), default=None)
        passed = ",".join(str(n) for n, (ok, _) in by_n(rs, "pass_tol").items() if ok)
        summary[func][(v, src)] = (ns, nt, top, cpu1, passed)
        md += ["", f"N* griežtai = **{ns}**, N* tol = **{nt}**" + (" (ribos nepasiekta — didžiausia pakopa)" if nt == top else ""), ""]
    md += ["## Suvestinė", "",
           "N* tol su „+“ — ribos nepasiekta (didžiausia išbandyta pakopa praėjo). Nauda = N*_G / N*_C (tol); "
           "CPU/srautui — mažiausias PASS_tol pakopose.", "",
           "| Funkcija | Variantas | Šaltinis | N* griežtai | N* tol | PASS_tol pakopos | CPU/srautui | Nauda G/C |",
           "|---|---|---|---|---|---|---|---|"]
    for func, d in summary.items():
        for (v, src), (ns, nt, top, cpu1, passed) in sorted(d.items()):
            c = d.get(("C", "fair")) or d.get(("C", src))
            gain = ""
            if v == "G" and c and c[1]:
                gain = f"{nt / c[1]:.1f}×" + ("+" if nt == top else "")
            md.append(f"| {func} | {v} | {src} | {ns} | {nt}{'+' if nt == top else ''} | {passed} | "
                      f"{'' if cpu1 is None else f'{cpu1:.3f}'} | {gain} |")
    open(out, "w").write("\n".join(md) + "\n")
    print(f"parašyta {out}: {len(rows)} pakopų, {len(groups)} grupių")


if __name__ == "__main__":
    main()
