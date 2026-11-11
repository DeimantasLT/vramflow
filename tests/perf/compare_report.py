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


def hq(r, name, q):
    """Histogramos (hist.<name>.<q>_ms) kvantilis; be duomenų — „–“."""
    v = ((r.get("hist") or {}).get(name) or {}).get(f"{q}_ms")
    return "–" if v is None else f"{v:.2f}"


def composite_section(rows):
    """composite N × K × {C,H,G}: lentelės pagal (K, variantas), N* matrica, praleistos pakopos (VRAM) — be nuorodų į kodą,
    kiekviena eilutė atsekama iki capacity-composite-k<K>-<V>.all.json."""
    md = ["# composite — talpa N × K (C / H / G)", "",
          "Funkcijos CPU — tik `composite` procesas (šaltiniai gen, gfxgen ir patikra `verify` atskirai, informacijai). "
          "Latencija: `proc` — nuo visų įėjimų paruošimo iki išėjimo commit'o; `vėlavimas` — commit'as po lauko laiko "
          "(`out_lateness`), p50/p99 ms visame composite veikimo laike. GPU % / W — nvidia-smi 1 Hz matavimo lange.", ""]
    ks = sorted({r["layers"] for r in rows})
    vs = [v for v in "CHG" if any(r["variant"] == v for r in rows)]
    nstar = {}
    for k in ks:
        for v in vs:
            rs = [r for r in rows if r["layers"] == k and r["variant"] == v]
            if not rs:
                continue
            md += [f"## K = {k} — {VARIANT_NAME.get(v, v)} (šaltinis: {rs[0].get('source')}, išėjimas {rs[0].get('out_format')}, "
                   f"deadline {rs[0].get('deadline_ms')} ms)", "",
                   "| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | "
                   "gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in sorted(rs, key=lambda r: r["n"]):
                if r.get("skipped"):
                    md.append(f"| {r['n']} | praleista | praleista | | | | | | | | | | | | | {r['skip_reason']} |")
                    continue
                g = r.get("gpu")
                samples = sum(c.get("samples_total", 0) for c in (r.get("verify") or {}).values())
                md.append(f"| {r['n']} | {'PASS' if r['pass'] else 'FAIL'} | {'PASS' if r.get('pass_tol') else 'FAIL'} | "
                          f"{r.get('loss_fraction', 0) * 100:.3f} | {r['realtime_ratio']:.4f} | {r['verification_errors']} | {samples} | "
                          f"{r['cpu_cores_func_per_stream']:.3f} | {r['cpu_cores_gfxgen_per_stream']:.3f} | {r['cpu_cores_gen_per_stream']:.3f} | "
                          f"{hq(r, 'proc_ms', 'p50')} / {hq(r, 'proc_ms', 'p99')} | "
                          f"{hq(r, 'out_lateness', 'p50')} / {hq(r, 'out_lateness', 'p99')} | {r.get('vram_delta_mb')} | "
                          f"{'–' if not g else f'{g['util_gpu']['mean']:.0f}'} | {'–' if not g else f'{g['power_w']['mean']:.0f}'} | "
                          f"`{r['_src']}` |")
            ran = [r for r in rs if not r.get("skipped")]
            nstar[(k, v)] = (nstar_of(ran, "pass"), nstar_of(ran, "pass_tol"), max((r["n"] for r in ran), default=0),
                             sorted({r["n"] for r in rs if r.get("skipped")}))
            ns, nt, top, sk = nstar[(k, v)]
            md += ["", f"N* griežtai = **{ns}**, N* tol = **{nt}**" + (" (ribos nepasiekta)" if nt == top and nt else ""), ""]
    md += ["## Suvestinė — didžiausias N (N*), kai visi mažesni PASS", "",
           "Reikšmė „griežtai / tol“. „+“ — didžiausia išbandyta pakopa praėjo (riba nepasiekta); kitu atveju pirmas FAIL "
           "ties kita pakopa. Pakopos, kurių nevykdė (VRAM riba arba 2 FAIL iš eilės), nėra N* dalis.", "",
           "| K | " + " | ".join(f"{v}" for v in vs) + " |", "|---|" + "---|" * len(vs)]
    for k in ks:
        cells = []
        for v in vs:
            ns, nt, top, sk = nstar.get((k, v), (0, 0, 0, []))
            cells.append(f"{ns} / {nt}{'+' if nt == top and nt else ''}")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    return md + [""]


def nstar_of(rows, key):
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
    comp = []
    for r in rows:
        if r["func"] == "composite":
            comp.append(r)  # composite — atskira N × K ataskaita (composite_section)
            continue
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
    if comp:
        if not groups:
            md = []  # tik composite — be tuščios bendros antraštės ir suvestinės
        md += composite_section(comp)
    open(out, "w").write("\n".join(md) + "\n")
    print(f"parašyta {out}: {len(rows)} pakopų, {len(groups)} grupių")


if __name__ == "__main__":
    main()
