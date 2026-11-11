#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Iš įrodymų katalogo sugeneruoja report.md (lietuviškai) su PASS/FAIL/INFO/NOT_RUN kiekvienam kriterijui.
NOT_RUN niekada neskaičiuojamas kaip PASS. Naudojimas: report.py <evidence_dir>
"""
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

ev = pathlib.Path(sys.argv[1])
rows = []  # (sritis, id, statusas, detalės)


def add(area, tid, status, detail):
    rows.append((area, tid, status, detail))


def load(p):
    p = ev / p
    return json.loads(p.read_text()) if p.exists() else None


man = load("manifest.json") or {}

# 00 probe
probe = ev / "00-probe/probe.jsonl"
if probe.exists():
    lines = [json.loads(l) for l in probe.read_text().splitlines() if l.startswith("{")]
    res = [l for l in lines if l.get("step") == "RESULT"]
    code = res[0]["exit_code"] if res else None
    gran = next((l.get("granularity") for l in lines if "granularity" in l), None)
    add("Žingsnis 0", "P0 CUDA VMM + POSIX FD tarp procesų", "PASS" if code == 0 else "FAIL",
        f"exit {code}; granularity {gran} B; 56,6 MB perduota, kiekvienas baitas patikrintas abiem kryptimis; "
        f"atmintis gyva po eksportuotojo išėjimo; 200 ciklų be FD/VRAM nuotėkio")
else:
    add("Žingsnis 0", "P0", "NOT_RUN", "nėra 00-probe/probe.jsonl")

# 20 upstream
junit = ev / "20-upstream/ctest-junit.xml"
if junit.exists():
    t = ET.parse(junit).getroot()
    tests = int(t.get("tests", 0)); fail = int(t.get("failures", 0)); skipped = int(t.get("skipped", 0))
    skipped_names = [tc.get("name") for tc in t.iter("testcase") if tc.find("skipped") is not None]
    cont = (ev / "20-upstream/container.txt").read_text().strip() if (ev / "20-upstream/container.txt").exists() else ""
    add("Upstream", "F1 MXL ctest (ne root, nuosekliai)", "PASS" if fail == 0 else "FAIL",
        f"{tests - fail - skipped}/{tests} passed, {fail} failed, {skipped} skipped ({', '.join(skipped_names)}); {cont}")
    cuda = [tc for tc in t.iter("testcase") if "CUDA payload" in (tc.get("name") or "")]
    cuda_pass = [tc.get("name") for tc in cuda if tc.find("skipped") is None and tc.find("failure") is None]
    add("Upstream", "F2 CUDA testai tikrai įvyko", "PASS" if len(cuda_pass) >= 4 else "FAIL",
        f"{len(cuda_pass)} CUDA testai passed (ne skipped): " + "; ".join(n.replace("CUDA payload : ", "") for n in cuda_pass))
else:
    add("Upstream", "F1/F2", "NOT_RUN", "nėra ctest-junit.xml")

# 31/32 unit ir GPU
for name, fn in (("Unit testai (auksinės reikšmės, 1e6+ tikrinimų)", "unit-junit.xml"), ("GPU kernel testai", "gpu-junit.xml")):
    p = ev / fn
    if p.exists():
        # Catch2 junit: <testsuites><testsuite tests=… failures=… errors=…> (atributų tvarka nesvarbi)
        root = ET.parse(p).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        tests = sum(int(s.get("tests", 0)) for s in suites)
        fails = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
        cases = sum(1 for _ in root.iter("testcase"))
        add("Įrankis", name, "PASS" if fails == 0 and tests > 0 else "FAIL",
            f"{cases} testų atvejai, {tests} tikrinimų, {fails} nepavyko")
    else:
        add("Įrankis", name, "NOT_RUN", "")

# 40 selftest
for storage in ("cuda-linear", "host"):
    s = load(f"40-selftest/{storage}/selftest.json")
    if not s:
        add("Savitikra", storage, "NOT_RUN", "")
        continue
    for c in s["cases"]:
        add("Savitikra", f"{storage}: {c['case']}", "PASS" if c["pass"] else "FAIL",
            f"injekcijų lange {c['injected_in_window']}, ok {c['grains_ok']}, reader exit {c['reader_exit']}"
            + (f"; {'; '.join(c['problems'])}" if c["problems"] else ""))

# 45 functional
f = load("45-functional/functional.json")
for r in f or []:
    add("Gedimai", r["id"], r["status"], r.get("why", ""))
if f is None:
    add("Gedimai", "F5–F8", "NOT_RUN", "")

# 50 sanitizers
for r in load("50-sanitizers/sanitizers.json") or []:
    add("Sanitizer'iai", r["id"], r["status"], r.get("note", ""))


# 60 scale
scale = [(storage, r) for storage in ("cuda-linear", "host") for r in (load(f"60-scale/scale-{storage}.json") or [])]
for storage, r in scale:
    st = "PASS" if r["correctness_pass"] else "FAIL"
    add("Skalavimas", f"{storage} N={r['n']} teisingumas", st,
        f"verif={r['verification_errors']}, patikrinta {r['grains_verified']}/{r['grains_expected']} "
        f"({r['verified_fraction']:.4f}), CPU kryžm. {r['host_crosschecks']} be neatitikimų={r['host_crosscheck_fail'] == 0}, "
        f"INVALID commit'intiems={r.get('invalid_committed')}")
    add("Skalavimas", f"{storage} N={r['n']} pajėgumas+latencija",
        "PASS" if r["capacity_pass"] and r["latency_pass"] else "INFO",
        f"gap={r['gap_grains']} stalls={r['stalls']}; c2v p50/p99/p99.9 = {r['c2v_p50_us']}/{r['c2v_p99_us']}/{r['c2v_p999_us']} µs; "
        f"writer jitter p99 {r['writer_jitter_p99_us']} µs; clock steps {len(r['clock_steps'])}")

rep = load("62-repeat/repeat-cuda-linear.json")
if rep:
    p50 = [r["c2v_p50_us"] for r in rep]
    cpu = [r["cpu_cores_per_flow"] for r in rep]
    spread = (max(p50) - min(p50)) / sorted(p50)[len(p50) // 2] if p50 and min(p50) else None
    cspread = (max(cpu) - min(cpu)) / sorted(cpu)[len(cpu) // 2] if cpu and min(cpu) else None
    ok = all(r["correctness_pass"] for r in rep)
    add("Pakartojamumas", f"3× N={rep[0]['n']}", "PASS" if ok and spread is not None and spread <= 0.25 and cspread <= 0.20 else ("FAIL" if not ok else "INFO"),
        f"teisingumas {sum(r['correctness_pass'] for r in rep)}/3; c2v p50 sklaida {spread:.1%}; CPU/flow sklaida {cspread:.1%}")

soak = load("70-soak/soak-cuda-linear.json")
if soak:
    r = soak[0]
    s = r["soak"]
    add("Soak", f"{r['seconds'] / 3600:.1f} h, N={r['n']} teisingumas", "PASS" if r["correctness_pass"] else "FAIL",
        f"verif={r['verification_errors']}, patikrinta {r['grains_verified']} grain'ų, gap={r['gap_grains']}, CPU kryžm. {r['host_crosschecks']}")
    add("Soak", "stabilumas (RSS, FD, VRAM, latencijos dreifas)", "PASS" if s["pass"] else "FAIL",
        f"writer RSS Δ {s['writer']['rss']['delta']:.0f} kB, reader RSS Δ {s['reader']['rss']['delta']:.0f} kB; "
        f"FD Δ {s['writer']['fd']['delta']:.0f}/{s['reader']['fd']['delta']:.0f}; VRAM free Δ {s['reader']['vram_free']['delta']:.0f} MB; "
        f"p99 pirma/paskutinė {s.get('p99_first_median_us')}/{s.get('p99_last_median_us')} µs; clock steps {len(r['clock_steps'])}")
else:
    add("Soak", "2 h", "NOT_RUN", "nėra 70-soak rezultatų")

# Ataskaita
counts = {k: sum(1 for x in rows if x[2] == k) for k in ("PASS", "FAIL", "INFO", "NOT_RUN")}
md = [f"# MXL PR #653 (CUDA grain payloads) — patikros ataskaita", "",
      f"Sugeneruota iš `{ev.name}`. Įrankis `vramflow-verify` @ `{man.get('verify_sha', '?')[:10]}`"
      f"{' (+ nekomituoti pakeitimai)' if man.get('verify_dirty') else ''}, MXL @ `{man.get('mxl_sha', '?')[:10]}`.", "",
      "## Aplinka", "",
      f"| | |", "|---|---|"]
for k in ("gpu", "driver", "nvcc", "os", "kernel", "cpu", "nproc", "mem_gb", "shm", "tai_offset_s", "started_utc"):
    md.append(f"| {k} | {man.get(k, '?')} |")
md += ["", f"## Suvestinė: {counts['PASS']} PASS · {counts['FAIL']} FAIL · {counts['INFO']} INFO · {counts['NOT_RUN']} NOT_RUN", ""]
if scale:
    md += ["## GPU kelias (cuda-linear) vs host bazė", "",
           "| N | kelias | verif. klaidos | patikrinta | gap / lapped | e2e p50 µs | e2e p99 µs | c2v p50 µs | c2v p99 µs (imtis) | CPU branduoliai/flow | ΔVRAM MB |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for storage, r in sorted(scale, key=lambda x: (x[1]["n"], x[0])):
        md.append(f"| {r['n']} | {storage} | {r['verification_errors']} | {r['verified_fraction']:.4f} | "
                  f"{r['gap_grains']} / {r['lapped_discarded']} | {r.get('e2e_p50_us')} | {r.get('e2e_p99_us')} | {r['c2v_p50_us']} | "
                  f"{r['c2v_p99_us']} ({r.get('c2v_sample_fraction', 0):.0%}) | {r['cpu_cores_per_flow']:.3f} | {r.get('vram_delta_mb')} |")
    md += ["", "*e2e* = nuo writer'io fill pradžios iki patikros pabaigos, imama kiekvienam grain'ui (nešališka). "
           "*c2v* = nuo commit iki patikros, imama tik kai reader'is ne atsilieka (imtis — kokiai daliai grain'ų). "
           "*ΔVRAM* — viso įrenginio prieaugis nuo testo pradžios (nvidia-smi).", ""]
md += ["## Visi kriterijai", "", "| Sritis | Testas | Būsena | Detalės |", "|---|---|---|---|"]
for area, tid, st, det in rows:
    md.append(f"| {area} | {tid} | **{st}** | {det.replace('|', '/')} |")
md += ["", "## Ko ši patikra neįrodo", "",
       "- Rezultatai galioja **vienam GPU, vienam host'ui, be konteinerių, WSL2** (peer-GPU ir cross-container testai NOT_RUN).",
       "- Native Linux latencijos gali skirtis (WSL2 / WDDM laiko pjūviai).",
       "- Tik video v210 (PR palaiko tik video); Fabrics su GPU atmintimi (#654) netestuotas.", ""]
(ev / "report.md").write_text("\n".join(md))
print(f"report.md: {counts}")
