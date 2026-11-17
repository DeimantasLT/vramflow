#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Klaidų kelių funkciniai testai (planas docs/planai/2026-11-17-production-ready.md §2.5; kriterijai K4, S9).

Kiekvienas testas sukelia vieną gedimą ir tikrina keturis dalykus:
  1. sugedęs procesas baigiasi exit 2, o santraukos „fatal“ tekste parašyta, kas nutiko;
  2. jis baigiasi per LIMIT_S (5 s);
  3. po savęs domene nieko nepalieka (nei flow'ų, nei laikinų katalogų) — turinys toks pat kaip prieš jį;
  4. tame pačiame domene šalia veikianti pora gen → verify --mode gen nepraranda nė vieno grain'o.

fault_vram_exhausted[gen|logo|dec] — VRAM užima vramflow-verify hog (palieka HOG_FREE_MB); funkcijai su VRAM_FLOWS srautų neužtenka
fault_alloc_fail[gen|logo|dec]     — paslėpta parinktis --test-create-fail K: K-ojo išėjimo saugyklos skirti „nepavyksta“
fault_nvenc_sessions              — veikia enc --flows NVENC_SESSIONS (tvarkyklės riba); antras enc --admission off sesijos negauna
fault_cuda_midrun[gen|logo]        — paslėpta parinktis --test-cuda-fail-after S: CUDA klaida tick'e po S sekundžių veikimo
fault_shm_full                     — domenas 16 MiB tmpfs (tik root; kitaip SKIP): gen su host saugykla netelpa
fault_domain_readonly              — domeno katalogas 0500 (tik ne root; kitaip SKIP): gen negali sukurti flow

fault_vram_exhausted tikro išsekimo WSL2 nesukelia: Windows (WDDM) kitam procesui duoda daugiau VRAM, nei CUDA rodo laisvos
(išmatuota 2026-11-17: hog paliko 334 MiB, gen gavo ~578 MiB ir 14 s dirbo be praradimų; perteklius — į bendrą sistemos
atmintį). Kai sugesti turėjęs procesas sukuria visus savo flow'us, testas tai užrašo kaip SKIP su skaičiais — tikras
variantas lieka natyviam Linux. Tą patį funkcijos kelią (dalis išėjimų jau sukurta, kitam saugyklos nėra) visur dengia
fault_alloc_fail: gedimas dirbtinis, bet valymas, exit kodas ir pranešimas — tikri.

Laikas (2 punktas): paleidimo gedimams — nuo proceso paleidimo iki pabaigos (apima ir CUDA inicializaciją, taigi griežčiau
nei „nuo gedimo“); veikimo metu injektuotam gedimui — nuo injekcijos eilutės TAI laiko iki pabaigos.
Medžiaga — sintetinė (scripts/linux/62-make-av-clip.sh), siuntimas — tik loopback.
"""
import json
import os
import pathlib
import re
import signal
import subprocess
import time

import functional as F
import isolation as ISO
import overload as O

LIMIT_S = 5.0      # kriterijai K4, S9
HOG_FREE_MB = 300  # tiek VRAM hog palieka: CUDA kontekstui užtenka, VRAM_FLOWS srautų — ne
VRAM_FLOWS = 24    # 24 × 1080i50 žiedai po 26,4 MiB (10 laukų × 2 764 800 B, MXL numatytoji 200 ms istorija) ≈ 633 MiB
RING_MB = 27       # vieno 1080i50 flow žiedas (26,4 MiB), suapvalinta aukštyn
NVENC_SESSIONS = int(os.environ.get("FAULT_NVENC_SESSIONS", "12"))  # tvarkyklės vienalaikių sesijų riba šiai GPU (P8)
INJECT_AFTER_S = 5
BASE_PORT = 26500
G = ("--impl", "gpu", "--storage", "cuda-linear")
ROLE = {"gen": 1, "logo": 2, "dec": 4}  # išėjimo flow rolė (src/func/flows.hpp: kRoleGen, kRoleLogo, kRoleDec)
OUT_NAME = {"gen": "video", "logo": "out", "dec": "out"}  # kaip funkcija pranešime vadina savo išėjimą
FAULT_RE = re.compile(r"--test-cuda-fail-after: dirbtinė CUDA klaida, TAI (\d+)")


class Neighbour:
    """Sveika pora tame pačiame domene: gen --flows 1 → verify --mode gen (G variantas). Paleidžiama prieš gedimą;
    health(since_ns, gate_late) grąžina problemų sąrašą (tuščias — pora gedimo nepajuto).

    Praradimai (praleisti indeksai, patikros klaidos, exit kodai) vertinami visada ir per visą poros gyvavimą.
    Vėlūs commit'ai (grain'as parašytas, bet po ribos) skaičiuojami atskirai — prieš since_ns (late_before) ir po jo
    (late_after) — ir į problemas patenka tik kai gate_late ir tik „po“:
      - gedimas veikimo metu (fault_cuda_midrun): since_ns — injekcijos momentas, gate_late=True — vėlavimas po gedimo = FAIL;
      - gedimas paleidžiant (visi kiti): since_ns — proceso paleidimas, gate_late=False — vėlavimai tik užrašomi.
    Kodėl paleidimo gedimams nevertinama: WSL2 bet kurio GPU proceso startas šalia kartais sustabdo kaimyno tick'ą ~20 ms,
    ir tai nepriklauso nuo to, ar startuojantis procesas vėliau suges. Išmatuota 2026-11-17 šiame teste: kaimynas pavėlavo
    23,1 ms startuojant jo paties verify (7 s prieš injekciją) ir 22,1 ms praėjus 178 ms nuo sveikai startuojančio gen
    (5 s prieš injekciją); atskirame matavime 20 gen startų šalia vieno gen — 0 vėlavimų. Priežastis — 3 fazės darbas (G3)."""

    def __init__(self, out, tag, d, seconds):
        F.gpu_mode("gpu")
        self.seconds = seconds
        self.gen = F.FProc(out, f"{tag}/nb", "gen", d, "--flows", 1, "--duration", seconds + 30, *G)
        self.ver = None
        self.ok = F.wait_flows(d, [F.role_flow_id(1, 0)], 30)
        if self.ok:
            self.ver = F.FProc(out, f"{tag}/nb", "verify", d, "--mode", "gen", "--flows", 1, "--every", 5, "--duration", seconds,
                               "--warmup", 2, *G)

    late_before = late_after = 0

    def health(self, since_ns=0, gate_late=False):
        if not self.ok:
            self.stop()
            return ["kaimyno gen flow neatsirado"]
        rv = self.ver.wait(timeout=self.seconds + 60)
        rg = self.gen.term()
        self.stop()
        sg, sv = F.vsummary(self.gen, "gen"), F.vsummary(self.ver, "verify-gen")
        pv, pg = (sv or {}).get("proc") or {}, (sg or {}).get("proc") or {}
        problems = []
        if rv != 0:
            problems.append(f"kaimyno verify exit {rv}")
        if rg != 0 or sg is None or sg.get("any_loss") is not False:
            problems.append(f"kaimyno gen exit {rg}, any_loss {(sg or {}).get('any_loss')}")
        late = [(int(m.group(3)), float(m.group(2))) for m in ISO.LATE_RE.finditer(O.err_text(self.gen))]  # (TAI ns, ms po T(i))
        hit = [(t, ms) for t, ms in late if t >= since_ns]
        self.late_before, self.late_after = len(late) - len(hit), len(hit)
        if hit and gate_late:
            problems.append(f"kaimyno gen vėlavo po gedimo: {len(hit)} k., pirmas po {(hit[0][0] - since_ns) / 1e6:.0f} ms "
                            f"({hit[0][1]:.1f} ms po T(i))")
        if pg.get("late_commits", 0) != len(late):  # kiekvienas vėlus commit'as turi eilutę su TAI — kitaip lango patikra nepilna
            problems.append(f"kaimyno gen: late_commits {pg.get('late_commits', 0)}, o eilučių su laiku {len(late)}")
        if pg.get("late_skips", 0):
            problems.append(f"kaimyno gen praleido indeksų: late_skips {pg.get('late_skips', 0)}")
        if pv.get("verification_errors", 1) != 0 or pv.get("capacity_errors", 1) != 0 or pv.get("samples_total", 0) <= 0:
            problems.append(f"kaimyno verify: klaidų {pv.get('verification_errors')}, praradimų {pv.get('capacity_errors')}, "
                            f"imčių {pv.get('samples_total')}")
        return problems

    def stop(self):
        for pr in (self.ver, self.gen):
            if pr is not None:
                pr.term()


def run_faulty(out, tag, sub, d, *args, timeout=45):
    """Paleidžia procesą, kuris turi sugesti. Grąžina (procesas, exit kodas arba None, sekundės nuo paleidimo iki pabaigos,
    pabaigos TAI laikas sekundėmis, paleidimo TAI laikas ns — nuo jo kaimyno vėlavimai priskiriami šiam procesui)."""
    t_spawn = time.clock_gettime_ns(time.CLOCK_TAI)
    t0 = time.monotonic()
    p = F.FProc(out, tag, sub, d, *args)
    rc = p.wait(timeout=timeout)
    dt, t_end = time.monotonic() - t0, time.clock_gettime(time.CLOCK_TAI)
    if rc is None:
        p.term()
    return p, rc, dt, t_end, t_spawn


def verdict(tid, p, rc, dt, needles, before, after, nb_problems, extra=(), **details):
    """Bendri tvirtinimai. needles — grupių sąrašas: iš kiekvienos grupės bent vienas tekstas turi būti santraukos „fatal“
    lauke (be didžiųjų / mažųjų skirtumo). dt — sekundės nuo gedimo iki pabaigos (None — nepavyko išmatuoti)."""
    s = p.summary() or {}
    fatal = s.get("fatal") or ""
    problems = list(extra)
    if rc != 2:
        problems.append(f"exit {rc} (laukta 2)")
    if s.get("exit_code") != 2:
        problems.append(f"santraukos exit_code {s.get('exit_code')} (laukta 2)")
    for group in needles:
        if not any(n.lower() in fatal.lower() for n in group):
            problems.append(f"santraukos fatal neturi nė vieno iš {list(group)}: „{fatal[:200]}“; stderr: „{O.err_text(p).strip()[-200:]}“")
    if dt is None or dt > LIMIT_S:
        problems.append(f"nuo gedimo iki pabaigos {'neišmatuota' if dt is None else f'{dt:.2f} s'} (riba {LIMIT_S:.0f} s)")
    if after != before:
        problems.append(f"domeno turinys pasikeitė: liko {sorted(set(after) - set(before))[:6]}, dingo {sorted(set(before) - set(after))[:6]}")
    problems += nb_problems
    why = "; ".join(problems) if problems else f"exit 2 per {dt:.2f} s, fatal: {fatal[:160]}"
    if any(details.get("nb_late") or ()):  # (prieš, po) since_ns — žr. Neighbour
        why += f" [kaimyno vėlūs commit'ai: {details['nb_late'][0]} prieš, {details['nb_late'][1]} po — procesų startas, žr. Neighbour]"
    return F.result(tid, "FAIL" if problems else "PASS", why=why, seconds=dt, fatal=fatal, **details)


def _vram(out, sub):
    """Grąžina (rezultatas, ar platforma skyrė VRAM virš laisvos ribos)."""
    tid, tag = f"fault_vram_exhausted[{sub}]", f"fault-vram-{sub}"
    d = F.fresh_domain(f"flt-vram-{sub}")
    nb = Neighbour(out, tag, d, 24)
    hog = p = None
    rc = dt = None
    before = after = []
    hog_line = ""
    created_all = False
    t_spawn = 0
    try:
        if not nb.ok:
            return F.result(tid, "FAIL", why="; ".join(nb.health())), False
        time.sleep(3)  # pora įsibėgėja, kol VRAM dar laisva
        hog = subprocess.Popen([F.BIN, "hog", "--leave-free-mb", str(HOG_FREE_MB)], stdout=subprocess.PIPE, text=True, env=F.ENV)
        hog_line = hog.stdout.readline().strip()
        before = F.leftovers(d)
        # dec įėjimo srauto nereikia: gedimas turi įvykti kuriant išėjimus, dar prieš laukiant paketų
        args = ["--flows", VRAM_FLOWS, "--flow-base", 100, "--duration", 20, "--admission", "off", *G]
        if sub == "dec":
            args += ["--port-base", BASE_PORT]
        mine = [pathlib.Path(d) / f"{F.role_flow_id(ROLE[sub], 100 + k)}.mxl-flow" / "flow_def.json" for k in range(VRAM_FLOWS)]
        t_spawn = time.clock_gettime_ns(time.CLOCK_TAI)
        t0 = time.monotonic()
        p = F.FProc(out, f"{tag}/bad", sub, d, *args)
        while p.p.poll() is None and time.monotonic() - t0 < 30:
            if all(f.exists() for f in mine):
                created_all = True  # visi žiedai gauti, nors laisvos VRAM tiek nebuvo — gedimo čia nebus
                break
            time.sleep(0.05)
        dt = time.monotonic() - t0
        rc = p.term() if created_all or p.p.poll() is None else p.p.poll()
        after = F.leftovers(d)
    finally:
        if hog is not None:
            hog.send_signal(signal.SIGTERM)
            hog.wait(timeout=30)
        if p is None or created_all:
            nb.stop()
    try:
        free_mb = json.loads(hog_line).get("free_mb", 1 << 30)
    except ValueError:
        free_mb = 1 << 30
    if free_mb > HOG_FREE_MB + 64:
        nb.stop()
        return F.result(tid, "FAIL", why=f"hog VRAM neužėmė: „{hog_line}“", hog=hog_line), False
    if created_all:
        return F.result(tid, "SKIP", hog=hog_line, seconds=dt,
                        why=f"platforma skyrė VRAM virš laisvos ribos: po hog laisva {free_mb} MiB, o {VRAM_FLOWS} flow'ų "
                            f"(≈ {VRAM_FLOWS * RING_MB} MiB) sukurta per {dt:.1f} s — išsekimo čia sukelti negalima (WSL2 / WDDM); "
                            f"tikras variantas — natyviame Linux, tą patį kelią dengia fault_alloc_fail"), True
    return verdict(tid, p, rc, dt, [("vram", "cudamalloc", "out of memory")], before, after, nb.health(t_spawn), hog=hog_line,
                   nb_late=(nb.late_before, nb.late_after)), False


def fault_vram_exhausted(out):
    results = []
    for k, sub in enumerate(("gen", "logo", "dec")):
        r, overcommit = _vram(out, sub)
        results.append(r)
        if overcommit:  # platformos savybė, ne funkcijos — kitoms funkcijoms rezultatas būtų tas pats
            results += [F.result(f"fault_vram_exhausted[{s}]", "SKIP", why=f"kaip fault_vram_exhausted[{sub}]: platforma skiria VRAM virš laisvos ribos")
                        for s in ("gen", "logo", "dec")[k + 1:]]
            break
    return results


def _alloc(out, sub):
    """Trečio iš keturių išėjimų saugyklos skirti „nepavyksta“ (--test-create-fail 2): du jau sukurti išėjimai turi būti
    atlaisvinti, procesas — baigtis exit 2 su pranešimu, kuriame yra išėjimo numeris ir išmatuota VRAM."""
    tid, tag = f"fault_alloc_fail[{sub}]", f"fault-alloc-{sub}"
    d = F.fresh_domain(f"flt-alloc-{sub}")
    nb = Neighbour(out, tag, d, 16)
    p = None
    rc = dt = None
    t_spawn = 0
    before = after = []
    try:
        if not nb.ok:
            return F.result(tid, "FAIL", why="; ".join(nb.health()))
        time.sleep(2)
        before = F.leftovers(d)
        args = ["--flows", 4, "--flow-base", 100, "--duration", 10, "--admission", "off", *G, "--test-create-fail", 2]
        if sub == "dec":
            args += ["--port-base", BASE_PORT + 300]
        p, rc, dt, _, t_spawn = run_faulty(out, f"{tag}/bad", sub, d, *args)
        after = F.leftovers(d)
    finally:
        if p is None:
            nb.stop()
    return verdict(tid, p, rc, dt, [(f"{OUT_NAME[sub]} 2:",), ("--test-create-fail",), ("vram laisva",)], before, after, nb.health(t_spawn),
                   nb_late=(nb.late_before, nb.late_after))


def fault_alloc_fail(out):
    return [_alloc(out, sub) for sub in ("gen", "logo", "dec")]


def fault_nvenc_sessions(out):
    """Pirmas enc užima visas NVENC sesijas ir turi dirbti be praradimų; antras negauna sesijos → exit 2 su aiškiu tekstu."""
    tid, tag = "fault_nvenc_sessions", "fault-nvenc"
    d = F.fresh_domain("flt-nvenc")
    F.gpu_mode("gpu")
    n, dur = NVENC_SESSIONS, 26
    g = F.FProc(out, tag, "gen", d, "--flows", n, "--duration", dur + 40, *G)
    first = p = None
    rc = dt = r1 = None
    before = after = []
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, k) for k in range(n)], 40):
            return [F.result(tid, "FAIL", why="gen flow'ai neatsirado")]
        first = F.FProc(out, f"{tag}/first", "enc", d, "--flows", n, "--duration", dur, "--warmup", 2, "--encoder", "nvenc", "--audio", 0,
                        "--port-base", BASE_PORT + 100, "--admission", "off", *G)
        time.sleep(10)  # visos sesijos atidarytos ir koduoja
        if first.p.poll() is not None:
            return [F.result(tid, "FAIL", why=f"pirmas enc baigėsi per 10 s (exit {first.p.poll()}): {O.err_text(first).strip()[-300:]}")]
        before = F.leftovers(d)
        p, rc, dt, _, _ = run_faulty(out, f"{tag}/second", "enc", d, "--flows", 1, "--duration", 10, "--encoder", "nvenc", "--audio", 0,
                                  "--port-base", BASE_PORT + 200, "--admission", "off", *G)
        after = F.leftovers(d)
        r1 = first.wait(timeout=dur + 60)
    finally:
        for pr in (p, first, g):
            if pr is not None:
                pr.term()
    s1 = first.summary() or {}
    fl = s1.get("flows") or []
    want = 0.9 * (dur - 2) * 25
    extra = []
    if r1 != 0 or s1.get("any_loss") is not False:
        extra.append(f"pirmas enc: exit {r1}, any_loss {s1.get('any_loss')}")
    if len(fl) != n or min((f.get("frames_encoded", 0) for f in fl), default=0) < want or any(f.get("encode_errors", 1) for f in fl):
        extra.append(f"pirmas enc: flow'ų {len(fl)} (laukta {n}), mažiausiai kadrų {min((f.get('frames_encoded', 0) for f in fl), default=0)} "
                     f"(laukta ≥ {want:.0f}), kodavimo klaidų {sum(f.get('encode_errors', 0) for f in fl)}")
    # priežastis turi būti ir srauto „error“ lauke (ilgas tekstas su libav žurnalo eilute — santrauka turi likti tinkamas JSON)
    err2 = (((p.summary() or {}).get("streams") or [{}])[0].get("error") or "") if p is not None else ""
    if "nvenc" not in err2.lower():
        extra.append(f"antro enc santraukos streams[0].error be priežasties: „{err2[:120]}“")
    return [verdict(tid, p, rc, dt, [("nvenc",), ("sesij",)], before, after, [], extra, first=s1.get("flows_n"))]


def _midrun(out, sub):
    tid, tag = f"fault_cuda_midrun[{sub}]", f"fault-midrun-{sub}"
    d = F.fresh_domain(f"flt-mid-{sub}")
    nb = Neighbour(out, tag, d, 22)
    p = None
    rc = dt_run = t_end = None
    t_spawn = 0
    before = after = []
    try:
        if not nb.ok:
            return F.result(tid, "FAIL", why="; ".join(nb.health()))
        time.sleep(2)
        before = F.leftovers(d)
        # gen — savo flow'ai (bazė 100); logo skaito kaimyno gen flow 0 ir rašo savo išėjimą
        args = ["--flows", 2, "--flow-base", 100] if sub == "gen" else ["--flows", 1, "--warmup", 2]
        p, rc, dt_run, t_end, t_spawn = run_faulty(out, f"{tag}/bad", sub, d, *args, "--duration", 30, "--admission", "off", *G,
                                          "--test-cuda-fail-after", INJECT_AFTER_S)
        after = F.leftovers(d)
    finally:
        if p is None:
            nb.stop()
    m = FAULT_RE.search(O.err_text(p))
    dt = t_end - int(m.group(1)) / 1e9 if m else None
    since = int(m.group(1)) - 100_000_000 if m else t_spawn  # kaimyno vėlavimai vertinami nuo 0,1 s prieš injekciją
    extra = [] if m else ["stderr nėra injekcijos eilutės (--test-cuda-fail-after)"]
    if dt_run < INJECT_AFTER_S:
        extra.append(f"procesas baigėsi per {dt_run:.1f} s — anksčiau, nei turėjo įvykti injekcija ({INJECT_AFTER_S} s)")
    s = p.summary() or {}
    key = "fields_written" if sub == "gen" else "fields_ok"
    done = [f.get(key, 0) for f in s.get("flows") or []]
    if not done or min(done) <= 0:
        extra.append(f"iki gedimo neapdorota nė vieno lauko ({key} = {done}) — gedimas ne veikimo metu")
    return verdict(tid, p, rc, dt, [("cuda:",)], before, after, nb.health(since, gate_late=True), extra, run_seconds=dt_run,
                   nb_late=(nb.late_before, nb.late_after))


def fault_cuda_midrun(out):
    return [_midrun(out, sub) for sub in ("gen", "logo")]


def fault_shm_full(out):
    """Domenas — 16 MiB tmpfs: gen su host saugykla (4 × 27,6 MiB žiedai) netelpa. Kaimynas (cuda-linear — maži failai) telpa."""
    tid, tag = "fault_shm_full", "fault-shm"
    if os.geteuid() != 0:
        return [F.result(tid, "SKIP", why="reikia root (mount -t tmpfs)")]
    d = F.fresh_domain("flt-shm")
    m = subprocess.run(["mount", "-t", "tmpfs", "-o", "size=16m,mode=0755", "tmpfs", str(d)], capture_output=True, text=True)
    if m.returncode != 0:
        return [F.result(tid, "SKIP", why=f"tmpfs prijungti nepavyko: {m.stderr.strip()[:200]}")]
    nb = p = None
    rc = dt = None
    before = after = []
    try:
        nb = Neighbour(out, tag, d, 18)
        if not nb.ok:
            return [F.result(tid, "FAIL", why="; ".join(nb.health()))]
        time.sleep(3)
        before = F.leftovers(d)
        p, rc, dt, _, t_spawn = run_faulty(out, f"{tag}/bad", "gen", d, "--flows", 4, "--flow-base", 100, "--duration", 10, "--admission", "off",
                                           "--impl", "cpu", "--storage", "host")
        after = F.leftovers(d)
        res = verdict(tid, p, rc, dt, [("no space left", "nėra vietos", "enospc")], before, after, nb.health(t_spawn),
                      nb_late=(nb.late_before, nb.late_after))
    finally:
        if nb is not None:
            nb.stop()
        if subprocess.run(["umount", str(d)], capture_output=True).returncode != 0:
            subprocess.run(["umount", "-l", str(d)], capture_output=True)
    return [res]


def fault_domain_readonly(out):
    """Domeno katalogas be rašymo teisės (0500): gen negali sukurti flow. Kaimynas paleistas anksčiau ir dirba toliau."""
    tid, tag = "fault_domain_readonly", "fault-ro"
    if os.geteuid() == 0:
        return [F.result(tid, "SKIP", why="root apeina katalogo teises (paleiskite scripts/linux/45-nonroot.sh)")]
    d = F.fresh_domain("flt-ro")
    nb = Neighbour(out, tag, d, 18)
    p = None
    rc = dt = None
    t_spawn = 0
    before = after = []
    try:
        if not nb.ok:
            return [F.result(tid, "FAIL", why="; ".join(nb.health()))]
        time.sleep(3)
        before = F.leftovers(d)
        os.chmod(d, 0o500)
        try:
            p, rc, dt, _, t_spawn = run_faulty(out, f"{tag}/bad", "gen", d, "--flows", 1, "--flow-base", 100, "--duration", 10, "--admission", "off", *G)
            after = F.leftovers(d)
        finally:
            os.chmod(d, 0o700)  # kaimynas baigdamas šalina savo flow — tam reikia rašymo teisės
    finally:
        if p is None:
            nb.stop()
    return [verdict(tid, p, rc, dt, [("permission denied", "eacces", "nerašomas")], before, after, nb.health(t_spawn), nb_late=(nb.late_before, nb.late_after))]


TESTS = {"fault_vram_exhausted": fault_vram_exhausted, "fault_alloc_fail": fault_alloc_fail, "fault_nvenc_sessions": fault_nvenc_sessions,
         "fault_cuda_midrun": fault_cuda_midrun, "fault_shm_full": fault_shm_full, "fault_domain_readonly": fault_domain_readonly}
