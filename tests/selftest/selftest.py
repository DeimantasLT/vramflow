#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Verifikatoriaus savitikra (architektūra §5.1): detektorius PRIVALO aptikti kiekvieną injekciją.

Kiekvienam atvejui paleidžiama writer+reader pora (scripts/wsl/lib/pair.sh). Writer'is užrašo kiekvienos
injekcijos indeksą, reader'is — kiekvieno aptikto pažeidimo indeksą ir kategorijų bitus. Tikrinama, kad
aptiktų indeksų aibė TIKSLIAI sutampa su injekcijų aibe reader'io lange (išskyrus indeksus, kurių reader'is
negalėjo patikrinti: gap / lapped), ir kad exit kodas toks, kokio laukiama.

Naudojimas: selftest.py <out_dir> [--storage cuda-linear|host] [--seconds 20] [--every 50]
Išėjimas: 0 tik jei VISI atvejai PASS. Rezultatai: <out_dir>/selftest.json
"""
import argparse
import json
import pathlib
import subprocess
import sys

# Bitai — src/mxlio/common_io.hpp ErrBit
ERR = dict(index=1 << 0, slot=1 << 1, size=1 << 2, hdr=1 << 3, content_index=1 << 4, mismatch=1 << 5,
           checksum=1 << 6, dup=1 << 7, crosscheck=1 << 8, invalid_flag=1 << 9, gap=1 << 10, lapped=1 << 11)
VERIFY_BITS = sum(ERR[k] for k in ("index", "slot", "size", "hdr", "content_index", "mismatch", "checksum", "dup",
                                   "crosscheck"))
UNCHECKED_BITS = ERR["gap"] | ERR["lapped"]

# (vardas, writer argumentai, kokie bitai privalo būti kiekvienoje injekcijoje, laukiamas reader exit, min dalis)
CASES = [
    ("control", "", 0, 0, 1.0),
    ("corrupt", "--corrupt-every {N}", ERR["mismatch"] | ERR["checksum"], 3, 1.0),
    ("corrupt_header", "--corrupt-header-every {N}", ERR["hdr"], 3, 1.0),
    ("skip", "--skip-every {N}", ERR["invalid_flag"], 4, 1.0),
    ("wrong_index", "--wrong-index-every {N}", ERR["content_index"] | ERR["mismatch"], 3, 1.0),
    ("stale", "--stale-every {N}", ERR["content_index"], 3, 1.0),
    # antra pusė rašoma po commit → reader'is, pabudęs per ≪ 5 ms, privalo matyti neatitinkančius žodžius
    ("commit_before_sync", "--commit-before-sync-every {N} --late-write-ms 5", ERR["mismatch"], 3, 0.9),
    ("dup", "--dup-every {N}", ERR["dup"], 3, 1.0),
]


def run_case(repo, out, name, wargs, storage, seconds, crosscheck):
    d = out / name
    subprocess.run(["rm", "-rf", str(d)], check=True)
    r = f"--storage {storage} --host-crosscheck-every {crosscheck}"
    w = f"--storage {storage} --gc-before-create {wargs}"
    subprocess.run(["bash", str(repo / "scripts/wsl/lib/pair.sh"), str(d), str(seconds), w, r], check=False,
                   stdout=subprocess.DEVNULL)
    ws = json.loads((d / "summary-writer.json").read_text())
    rs = json.loads((d / "summary-reader.json").read_text())
    return ws, rs


def evaluate(name, required_bits, want_exit, min_frac, ws, rs, n_words):
    wf, rf = ws["flows"][0], rs["flows"][0]
    lo, hi = rf["first_index"], rf["last_index"]
    errs = {}
    for idx, bits, first_bad in rf["errors"]:
        e = errs.setdefault(idx, [0, first_bad])
        e[0] |= bits
    unchecked = {i for i, (b, _) in errs.items() if b & UNCHECKED_BITS and not b & ERR["invalid_flag"]}
    injected = sorted(i for i, _ in wf["injected"] if lo <= i <= hi and i not in unchecked)
    if name == "dup":
        # dublikatą galima aptikti tik turint ankstesnį patikrintą grain'ą — lango pradžia jo neturi
        injected = [i for i in injected if i != lo]
    detected_any = {i for i, (b, _) in errs.items() if b & (VERIFY_BITS | ERR["invalid_flag"])}
    problems = []

    # Aprėptis (visiems atvejams): reader'is turi būti įvertinęs beveik visus lango indeksus,
    # kitaip „0 klaidų" gali reikšti „nieko netikrinta".
    cov = rf.get("coverage", 0)
    span = max(1, hi - lo + 1)
    # skip atveju praleisti (INVALID) indeksai sąmoningai nepatikrinami — jie atimami iš slenksčio
    cov_min = 0.99 - (len(injected) / span if name == "skip" else 0)
    if cov < cov_min:
        problems.append(f"aprėptis {cov:.4f} < {cov_min:.4f} (lapped={rf['counters']['lapped_discarded']}, gap={rf['counters']['gap_grains']})")
    # INVALID vėliavėlė grain'ui, kurį writer'is TIKRAI commit'ino = MXL teisingumo klaida (ne pajėgumas)
    committed = ws["flows"][0].get("committed", [])
    invalid_committed = [i for i, (b, _) in errs.items() if b & ERR["invalid_flag"]
                         and any(a <= i <= z for a, z in committed)]
    if invalid_committed:
        problems.append(f"INVALID grain'ai, kurie buvo commit'inti: {invalid_committed[:10]}")

    if name == "control":
        if rs["totals"]["verification_errors"] or any(b & VERIFY_BITS for b, _ in errs.values()):
            problems.append(f"kontrolėje aptikta klaidų: {rs['totals']}")
        if rf["counters"]["lapped_discarded"]:
            problems.append(f"kontrolėje lapped_discarded={rf['counters']['lapped_discarded']}")
        if rf["counters"]["grains_ok"] < (hi - lo + 1) * 0.99:
            problems.append(f"kontrolėje patikrinta tik {rf['counters']['grains_ok']} iš {hi - lo + 1}")
    else:
        if not injected:
            problems.append("lange nėra nė vienos injekcijos — atvejis netestuoja nieko")
        if required_bits:
            hit = [i for i in injected if i in errs and (errs[i][0] & required_bits) == required_bits]
        else:
            hit = [i for i in injected if i in detected_any]
        frac = len(hit) / len(injected) if injected else 0
        if frac < min_frac:
            miss = [i for i in injected if i not in hit][:10]
            problems.append(f"aptikta {len(hit)}/{len(injected)} ({frac:.0%}) < {min_frac:.0%}; nepagauti pvz. {miss}")
        # Netikri teigiami: pažeidimai indeksuose, kur injekcijos nebuvo (stale/wrong_index/dup gali paveikti
        # kitą grain'ą tik informaciniais skaitikliais, ne VERIFY bitais)
        # INVALID be injekcijos = pajėgumo įvykis (writer'is pavėlavo), ne verifikacijos klaida — jį skaičiuojam atskirai.
        inj_all = {i for i, _ in wf["injected"]}
        false_pos = sorted(i for i, (b, _) in errs.items() if b & VERIFY_BITS and i not in inj_all)
        if false_pos:
            problems.append(f"netikri teigiami indeksuose {false_pos[:10]} (iš viso {len(false_pos)})")
        if name == "corrupt":
            # writer'is skaičiuoja uint64 (su perpildymu), todėl orakulas privalo daryti tą patį
            wrong_pos = [i for i in hit if errs[i][1] != 16 + ((i * 2654435761) & (2**64 - 1)) % (n_words - 16)]
            if wrong_pos:
                problems.append(f"first_bad_word nesutampa su injekcijos pozicija: {wrong_pos[:5]}")
    if rf["counters"].get("host_crosscheck_fail", 0):
        problems.append(f"CPU kryžminė patikra nesutapo su GPU {rf['counters']['host_crosscheck_fail']} kartų")
    if rs["exit_code"] != want_exit:
        problems.append(f"reader exit {rs['exit_code']} != laukiamas {want_exit}")
    if ws["exit_code"] != 0:
        problems.append(f"writer exit {ws['exit_code']}")
    return {
        "case": name, "pass": not problems, "problems": problems, "window": [lo, hi],
        "injected_in_window": len(injected), "reader_exit": rs["exit_code"], "writer_exit": ws["exit_code"],
        "grains_ok": rf["counters"]["grains_ok"], "host_crosschecks": rf["counters"].get("host_crosschecks", 0),
        "coverage": cov, "totals": rs["totals"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--storage", default="cuda-linear")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--crosscheck", type=int, default=10)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(a.out) / a.storage
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, wargs, bits, want_exit, frac in CASES:
        if a.only and name not in a.only.split(","):
            continue
        ws, rs = run_case(repo, out, name, wargs.format(N=a.every), a.storage, a.seconds, a.crosscheck)
        n_words = rs["flows"][0]["grain_size"] // 4
        res = evaluate(name, bits, want_exit, frac, ws, rs, n_words)
        results.append(res)
        flag = "PASS" if res["pass"] else "FAIL"
        print(f"[{flag}] {a.storage:11s} {name:20s} injekcijų={res['injected_in_window']:3d} "
              f"ok={res['grains_ok']:5d} exit={res['reader_exit']} {'; '.join(res['problems'])}", flush=True)
    ok = all(r["pass"] for r in results)
    (out / "selftest.json").write_text(json.dumps({"storage": a.storage, "pass": ok, "cases": results}, indent=1))
    print(f"SELFTEST {a.storage}: {'PASS' if ok else 'FAIL'} ({sum(r['pass'] for r in results)}/{len(results)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
