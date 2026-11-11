#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""„Auksiniai“ regresijos vartai tick variklio perkėlimui (planas docs/planai/2026-11-12-tick-variklis.md §9, V0a).
Kviečiami iš functional.py (--only fgold_logo,fgold_composite,fgold_dec).

fgold_logo      — gen | logo | verify --mode logo --every 1, N = 2, 10 s, variantai C ir G. Tvirtinama: kiekvienam flow'ui
                  samples == present == last_index − first_index + 1 (Σ = span · N), mismatch_fields == 0, nėra
                  missing / invalid / too_late / verify_skipped / read_errors; logo santraukoje Σ fields_ok == savo
                  (last_index − first_index + 1) · N ir visi praradimų skaitikliai 0; santraukų raktų aibės == įrašytoms.
fgold_composite — gen + gfxgen (pace, 2 sluoksniai) | composite | verify --mode composite --every 1, tos pačios lygybės.
fgold_dec       — klipas (ffmpeg -re → UDP) | dec | verify --mode dec --ref, N = 1: samples > 0, CRC neatitikimų 0,
                  parity_mismatch == 0, dec be praradimų po paleidimo.

Raktų sąrašai: tests/functional/data/summary_keys_<rolė>.json — įrašyti iš dabartinių binarų (`golden.py record <out>`).
`required` — raktai, kurie yra visuose įrašymo paleidimuose; `optional` — kurie pasitaiko ne visuose (skaitikliai, kuriuos funkcija
sukuria tik juos padidinusi, pvz. warmup_skips). Tvirtinama: required ⊆ faktiniai ⊆ required ∪ optional. Naujas raktas (po
refaktoringo) gali būti leistas: FGOLD_ALLOW_NEW_KEYS=1 — bet sąrašą reikia atnaujinti tame pačiame PR.

Testai tikrina tik iš išorės (procesai, exit kodai, santraukų skaitikliai) — nepriklauso nuo to, kuris ciklas veikia viduje.
Aplinka: FGOLD_VARIANTS=C,G (numatyta), FGOLD_CLIP / FGOLD_REF (numatyta ~/media-test/clip.ts, ref_crc.json),
FGOLD_DURATION=10 (s).
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import functional as F  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data"
VARIANTS = {"C": ("cpu", "host"), "H": ("gpu", "host"), "G": ("gpu", "cuda-linear")}
DUR = int(os.environ.get("FGOLD_DURATION", "10"))
CLIP = os.environ.get("FGOLD_CLIP", str(pathlib.Path.home() / "media-test" / "clip.ts"))
REF = os.environ.get("FGOLD_REF", str(pathlib.Path.home() / "media-test" / "ref_crc.json"))
DEC_PORT = int(os.environ.get("FGOLD_DEC_PORT", "21500"))
COMPOSITE_EVERY = int(os.environ.get("FGOLD_COMPOSITE_EVERY", "1"))  # planas: 1; diagnostikai (verify apkrova)
COMPOSITE_MIN_CHECKED = float(os.environ.get("FGOLD_COMPOSITE_MIN_CHECKED", "0.5"))
SECTIONS = ("top", "proc", "hist", "flows")
# Praradimų skaitikliai: funkcijos santraukoje jų turi nebūti (arba 0); verify — tikrinamas teisingumas ir aprėptis.
LOSS = ("late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips", "missing_fields", "clock_skips",
        "warmup_skips")


def variants():
    want = [x for x in os.environ.get("FGOLD_VARIANTS", "C,G").split(",") if x]
    return [(t, *VARIANTS[t]) for t in want]


def flows_of(s):
    return (s or {}).get("flows") or []


def fsum(s, key):
    return sum(f.get(key, 0) for f in flows_of(s))


# --- santraukų raktų aibės ---------------------------------------------------------------------------------------------

def key_sets(s):
    """Santraukos struktūros pirštų atspaudas: top = aukščiausio lygio raktai, proc / hist = jų vidiniai raktai, flows = visų
    flow'ų raktų sąjunga."""
    s = s or {}
    return {"top": set(s.keys()), "proc": set((s.get("proc") or {}).keys()), "hist": set((s.get("hist") or {}).keys()),
            "flows": {k for f in flows_of(s) for k in f.keys()}}


def data_file(role):
    return DATA / f"summary_keys_{role}.json"


def keys_check(role, s):
    """(ok, aprašas): faktiniai raktai prieš įrašytą sąrašą. Trūkstamas raktas — visada klaida; naujas — klaida, nebent
    FGOLD_ALLOW_NEW_KEYS=1."""
    f = data_file(role)
    if not f.exists():
        return False, f"nėra {f.name}"
    rec = json.loads(f.read_text())
    act = key_sets(s)
    gone, new = {}, {}
    for sec in SECTIONS:
        req, opt = set(rec["required"][sec]), set(rec["optional"][sec])
        if req - act[sec]:
            gone[sec] = sorted(req - act[sec])
        if act[sec] - req - opt:
            new[sec] = sorted(act[sec] - req - opt)
    allow_new = os.environ.get("FGOLD_ALLOW_NEW_KEYS") == "1"
    ok = not gone and (not new or allow_new)
    return ok, f"{role}: dingę={gone or '-'} nauji={new or '-'}"


SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "func"
ROLE_SRC = {"gen": "gen.cpp", "logo": "logo.cpp", "composite": "composite.cpp", "gfxgen": "gfxgen.cpp", "dec": "dec.cpp"}


def lazy_keys(role):
    """Skaitikliai, kuriuos funkcija sukuria TIK įvykio metu (`inc("x")` be reikšmės, `inc(sąlyga ? "a" : "b", …)`, `proc["x"]++`),
    todėl sveikame paleidime jų gali nebūti (pvz. dec `startup_discard`). Naudojama tik įrašant sąrašą (ne testuose): tokie raktai
    patenka į `optional`, net jei visuose įrašymo paleidimuose pasitaikė. `verify` santraukos šių raktų neturi — jos `summarize()`
    visus skaitiklius sukuria `operator[]`, todėl jų sąrašas griežtas."""
    f = SRC / ROLE_SRC.get(role, "")
    if role not in ROLE_SRC or not f.exists():
        return set()
    names = set()
    for line in f.read_text(encoding="utf-8").splitlines():
        names.update(re.findall(r'\.inc\("(\w+)"\)', line))
        names.update(re.findall(r'proc\["(\w+)"\]\+\+', line))
        if ".inc(" in line and "?" in line:
            names.update(re.findall(r'"(\w+)"', line))
    return names


def record_keys(role, summaries):
    """Įrašo summary_keys_<rolė>.json iš kelių paleidimų santraukų: required = sankirta (be „įvykio“ skaitiklių, žr. lazy_keys),
    optional = sąjunga − required."""
    sets = [key_sets(s) for s in summaries if s]
    lazy = lazy_keys(role)
    req = {sec: set.intersection(*[x[sec] for x in sets]) - (lazy if sec in ("proc", "flows") else set()) for sec in SECTIONS}
    uni = {sec: set.union(*[x[sec] for x in sets]) for sec in SECTIONS}
    rec = {"role": role, "runs": len(sets),
           "required": {sec: sorted(req[sec]) for sec in SECTIONS},
           "optional": {sec: sorted(uni[sec] - req[sec]) for sec in SECTIONS}}
    DATA.mkdir(exist_ok=True)
    data_file(role).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")


# --- scenarijai (grąžina santraukas ir exit kodus; tvirtinimai — atskirai) -------------------------------------------------

def run_logo(out, tag, impl, storage):
    N = 2
    sub = f"fgold-logo-{tag}"
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    g = F.FProc(out, sub, "gen", d, "--flows", N, "--duration", DUR + 40, *v)
    lg = vf = None
    rl = rv = None
    err = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)]):
            return {"err": "gen flow'ai neatsirado"}
        lg = F.FProc(out, sub, "logo", d, "--flows", N, "--duration", DUR + 8, "--warmup", 2, *v)
        if not F.wait_flows(d, [F.role_flow_id(2, n) for n in range(N)], 20):
            return {"err": "logo išėjimo flow'ai neatsirado"}
        vf = F.FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 1, "--duration", DUR, "--warmup", 2,
                     *F.vimpl(impl, storage))
        rv = vf.wait(timeout=DUR + 60)
        rl = lg.wait(timeout=DUR + 60)
    finally:
        for pr in (vf, lg, g):
            if pr is not None:
                pr.term()
    return {"N": N, "rc": {"logo": rl, "verify": rv}, "gen": F.vsummary(g, "gen"), "logo": F.vsummary(lg, "logo"),
            "verify-logo": F.vsummary(vf, "verify-logo"), "err": err}


def run_composite(out, tag, impl, storage):
    N, K = 2, 2
    sub = f"fgold-composite-{tag}"
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    xy = ["--layer-xy", "1488,64", "--layer-xy", "0,0"]
    g = F.FProc(out, sub, "gen", d, "--flows", N, "--duration", DUR + 60, *v)
    x = F.FProc(out, sub, "gfxgen", d, "--flows", N, "--layers", K, "--layer-geom", "logo,full", "--mode", "pace",
                "--duration", DUR + 60, *v)
    c = va = None
    rc = rv = None
    try:
        ids = [F.role_flow_id(1, n) for n in range(N)] + [F.role_flow_id(5, n * 8 + k) for n in range(N) for k in range(K)]
        if not F.wait_flows(d, ids):
            return {"err": "gen / gfxgen flow'ai neatsirado"}
        c = F.FProc(out, sub, "composite", d, "--flows", N, "--layers", K, *xy, "--opacity", "0.7", "--opacity", "0.5",
                    "--base", "gen", "--out-format", "v210", "--deadline-ms", F.COMP_DEADLINE_MS, "--duration", DUR + 14,
                    "--warmup", 2, *v)
        if not F.wait_flows(d, [F.role_flow_id(6, n) for n in range(N)], 20):
            return {"err": "composite išėjimo flow'ai neatsirado"}
        va = F.FProc(out, sub, "verify", d, "--mode", "composite", "--flows", N, "--layers", K, "--layer-geom", "logo,full",
                     "--gfx-mode", "pace", "--base", "gen", "--out-format", "v210", "--every", COMPOSITE_EVERY, "--duration", DUR,
                     "--warmup", 2, *xy, "--opacity", "0.7", "--opacity", "0.5", *F.vimpl(impl, storage))
        rv = va.wait(timeout=DUR + 60)
        rc = c.wait(timeout=DUR + 60)
    finally:
        for pr in (va, c, x, g):
            if pr is not None:
                pr.term()
    return {"N": N, "rc": {"composite": rc, "verify": rv}, "gen": F.vsummary(g, "gen"), "gfxgen": F.vsummary(x, "gfxgen"),
            "composite": F.vsummary(c, "composite"), "verify-composite": F.vsummary(va, "verify-composite")}


def run_dec(out, tag, impl, storage):
    N = 1
    sub = f"fgold-dec-{tag}"
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    dc = vf = None
    senders = []
    rd = rv = None
    try:
        dc = F.FProc(out, sub, "dec", d, "--flows", N, "--port-base", DEC_PORT, "--duration", DUR + 14, "--warmup", 6, *v)
        time.sleep(2)
        for k in range(N):  # kaip capacity.py: be -stream_loop (kilpos sandūra praranda IDR); klipas ilgesnis už langą
            senders.append(subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-re", "-i", CLIP, "-c", "copy", "-f",
                                             "mpegts", f"udp://127.0.0.1:{DEC_PORT + k}?pkt_size=1316"],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if not F.wait_flows(d, [F.role_flow_id(4, n) for n in range(N)], 30):
            return {"err": "dec išėjimo flow'ai neatsirado"}
        time.sleep(3)
        vf = F.FProc(out, sub, "verify", d, "--mode", "dec", "--flows", N, "--every", 1, "--duration", DUR, "--warmup", 6,
                     "--ref", REF, *F.vimpl(impl, storage))
        rv = vf.wait(timeout=DUR + 60)
        rd = dc.wait(timeout=DUR + 60)
    finally:
        for s in senders:
            s.kill()
            s.wait()
        for pr in (vf, dc):
            if pr is not None:
                pr.term()
    return {"N": N, "rc": {"dec": rd, "verify": rv}, "dec": F.vsummary(dc, "dec"), "verify-dec": F.vsummary(vf, "verify-dec")}


# --- tvirtinimai ----------------------------------------------------------------------------------------------------------

def verify_lines(sv, N, min_checked=1.0):
    """Bendri verify tvirtinimai: Σ present == span · N, samples + check_dropped == present, mismatch / praradimų 0.
    min_checked — mažiausia patikrintų dalis (samples / present): logo — 1,0 (samples == present, kaip planas); composite —
    verify CPU etalonas (dešimtys ms laukui) ne visada spėja prie --every 1, tada užpildyta eilė skaičiuojama `check_dropped`
    (informacinis, žr. verify.cpp): tikrinta mažiau, bet klaidos nėra."""
    fl = flows_of(sv)
    span = (sv or {}).get("last_index", 0) - (sv or {}).get("first_index", 0) + 1 if sv else 0
    pv = (sv or {}).get("proc") or {}
    zero = ("mismatch_fields", "missing", "invalid", "too_late", "verify_skipped", "read_errors", "marker_bad", "expect_failed",
            "cuda_errors", "outside_diff_fields", "chroma_low")
    samples, present, dropped = fsum(sv, "samples"), fsum(sv, "present"), fsum(sv, "check_dropped")
    bad = {k: fsum(sv, k) for k in zero if fsum(sv, k)}
    ok = (sv is not None and len(fl) == N and span > 0 and present == span * N and samples + dropped == present
          and samples >= min_checked * present and not bad
          and pv.get("verification_errors", 1) == 0 and pv.get("capacity_errors", 1) == 0)
    return ok, (f"verify: samples={samples} check_dropped={dropped} present={present} span·N={span * N} ({span}·{N}) "
                f"nenuliniai={bad or '-'} "
                f"verification_errors={pv.get('verification_errors')} capacity_errors={pv.get('capacity_errors')}")


def fn_lines(sf, N):
    """Funkcijos (logo / composite) santrauka: Σ fields_ok == (last − first + 1) · N, praradimų skaitikliai 0."""
    span = (sf or {}).get("last_index", 0) - (sf or {}).get("first_index", 0) + 1 if sf else 0
    okf = fsum(sf, "fields_ok")
    bad = {k: fsum(sf, k) for k in LOSS if fsum(sf, k)}
    ok = sf is not None and len(flows_of(sf)) == N and span > 0 and okf == span * N and not bad
    return ok, f"funkcija: Σ fields_ok={okf} span·N={span * N} ({span}·{N}) praradimai={bad or '-'}"


def finish(tid, r, fn_role, ver_role, extra_roles=(), min_checked=1.0):
    if r.get("err"):
        return F.result(tid, "FAIL", why=r["err"])
    N = r["N"]
    okv, tv = verify_lines(r[ver_role], N, min_checked)
    okf, tf = fn_lines(r[fn_role], N)
    kchecks = [keys_check(role, r[role]) for role in (fn_role, ver_role, *extra_roles)]
    okk = all(k[0] for k in kchecks)
    okrc = all(x == 0 for x in r["rc"].values())
    ok = okv and okf and okk and okrc
    return F.result(tid, "PASS" if ok else "FAIL",
                    why=f"exit={r['rc']} {tf}; {tv}; raktai: " + "; ".join(k[1] for k in kchecks),
                    **{fn_role: r[fn_role], ver_role: r[ver_role]})


def fgold_logo(out):
    return [finish(f"fgold_logo[{tag}]", run_logo(out, tag, impl, storage), "logo", "verify-logo", ("gen",))
            for tag, impl, storage in variants()]


def fgold_composite(out):
    return [finish(f"fgold_composite[{tag}]", run_composite(out, tag, impl, storage), "composite", "verify-composite",
                   ("gen", "gfxgen"), min_checked=COMPOSITE_MIN_CHECKED) for tag, impl, storage in variants()]


def fgold_dec(out):
    res = []
    for tag, impl, storage in variants():
        tid = f"fgold_dec[{tag}]"
        if not (pathlib.Path(CLIP).exists() and pathlib.Path(REF).exists()):
            res.append(F.result(tid, "FAIL", why=f"nėra klipo {CLIP} ar etalono {REF} (scripts/wsl/61-make-test-clip.sh)"))
            continue
        r = run_dec(out, tag, impl, storage)
        if r.get("err"):
            res.append(F.result(tid, "FAIL", why=r["err"]))
            continue
        sd, sv = r["dec"], r["verify-dec"]
        pv = (sv or {}).get("proc") or {}
        samples = fsum(sv, "samples")
        bad = {k: fsum(sv, k) for k in ("dec_mismatch", "dec_mismatch_near_loop", "parity_mismatch", "ref_marker_unknown",
                                        "marker_bad", "mismatch_fields", "cuda_errors", "chroma_low", "missing", "invalid",
                                        "too_late", "read_errors") if fsum(sv, k)}
        loss = {k: fsum(sd, k) for k in ("late_fields", "lapped", "too_late", "in_invalid", "read_errors", "decode_errors",
                                         "pkts_corrupt", "send_eagain", "clock_overflow_drops") if fsum(sd, k)}
        kchecks = [keys_check(role, r[role]) for role in ("dec", "verify-dec")]
        ok = (sd is not None and sv is not None and samples > 0 and fsum(sv, "dec_bitexact") > 0 and not bad and not loss
              and pv.get("verification_errors", 1) == 0 and all(x == 0 for x in r["rc"].values()) and all(k[0] for k in kchecks))
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"exit={r['rc']} samples={samples} dec_bitexact={fsum(sv, 'dec_bitexact')} nenuliniai verify={bad or '-'} dec praradimai={loss or '-'} "
                                f"fields_ok={fsum(sd, 'fields_ok')} verification_errors={pv.get('verification_errors')} raktai: "
                                + "; ".join(k[1] for k in kchecks), dec=sd, verify=sv))
    return res


TESTS = {"fgold_logo": fgold_logo, "fgold_composite": fgold_composite, "fgold_dec": fgold_dec}


def rebuild(out):
    """Perskaičiuoja summary_keys_*.json iš jau įrašytų paleidimų (rec*/…/summary-<rolė>.json) be naujų paleidimų."""
    out = pathlib.Path(out)
    for role in ("gen", "logo", "verify-logo", "gfxgen", "composite", "verify-composite", "dec", "verify-dec"):
        files = sorted(out.glob(f"rec*/**/summary-{role}.json"))
        record_keys(role, [json.loads(f.read_text()) for f in files])
        print(f"[rebuild] {data_file(role).name}: {len(files)} santraukų", flush=True)


def record(out, runs):
    """Įrašo summary_keys_<rolė>.json: kiekvienas scenarijus (C ir G) `runs` kartų, raktai — sankirta / sąjunga."""
    out = pathlib.Path(out)
    acc = {}
    for it in range(runs):
        for tag, impl, storage in variants():
            for fn, roles in ((run_logo, ("gen", "logo", "verify-logo")),
                              (run_composite, ("gen", "gfxgen", "composite", "verify-composite")),
                              (run_dec, ("dec", "verify-dec"))):
                r = fn(out / f"rec{it}", f"{tag}", impl, storage)
                if r.get("err"):
                    print(f"[record] {fn.__name__}[{tag}] #{it}: {r['err']}", flush=True)
                    continue
                for role in roles:
                    acc.setdefault(role, []).append(r[role])
                print(f"[record] {fn.__name__}[{tag}] #{it} ok", flush=True)
    for role, ss in acc.items():
        record_keys(role, ss)
        print(f"[record] {data_file(role).name}: {len(ss)} santraukų", flush=True)
    F.cleanup_domains()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "rebuild":
        rebuild(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "record":
        record(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        sys.exit("naudojimas: golden.py record <out_dir> [paleidimų_skaičius] | rebuild <out_dir>  (testai — per functional.py --only fgold_*)")
