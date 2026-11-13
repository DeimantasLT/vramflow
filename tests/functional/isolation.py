#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Gedimų izoliacijos vartai (planas docs/planai/2026-11-12-tick-variklis.md §9, V0b, V10–V12). Kviečiami iš functional.py
(--only fiso_teeth,fiso_kill,fiso_stall,fiso_composite,fiso_enc,fiso_enc_audio).

Klausimas: ką sveikiems flow'ams daro vieno aukštupio rašytojo mirtis (kill -9) ar užstrigimas (SIGSTOP / SIGCONT)? N = 8 flow'ų;
paveikto (victim) flow'o šaltinis — atskiras procesas, sveikų — kiti procesai (po vieną kiekvienai ištisinei flow'ų atkarpai).
Kiekvienas vartų atvejis kiekvienam variantui vykdomas su paveiktu flow'u PIRMU, VIDURINIU ir PASKUTINIU (POSITIONS): variklio
tvarka (vienetų eilė batch'e, ašies parinkimas pagal numerį) negali „pasislėpti“ už sugedusio vietos.

fiso_kill      — gen(-ai) sveikiems + gen paveiktam → logo N → verify N; ties FAULT_AT s paveikto gamintojas nužudomas (kill -9).
fiso_stall     — tas pats, bet paveikto gamintojas sustabdomas (SIGSTOP) 5 s ir tęsiamas (SIGCONT); po to paveiktas flow'as
                 POST_S s be praradimų (verify, paleistas iš anksto — jo CUDA kontekstas nekuriamas gedimo lange).
fiso_composite — gen (N bazių) + gfxgen sveikiems + gfxgen paveiktam (vieno išėjimo sluoksnio šaltinis) → composite N → verify N.
fiso_enc       — gen(-ai) sveikiems (video + garsas) + gen paveiktam → enc N (+ ts_sink); paveikto gamintojas nužudomas.
fiso_enc_audio — kaip fiso_enc, bet paveikto srauto video ir garsas — skirtingi procesai; nužudomas TIK garso gamintojas.
                 Sveikiems — kaip fiso_enc; paveiktam — video nenukenčia, prarastas garsas suskaičiuotas.
fiso_teeth     — vartų „dantų“ meta-testas: logo C, logo G ir enc C su --test-fault-proc-delay-ms (funkcija vėluoja, kol trūksta
                 įėjimo) privalo būti FAIL (30 ms ir 10 ms; 10 ms — ir dėl lango p99), be injekcijos — PASS.
Variantai: C (cpu + host, x264) ir G (gpu + cuda-linear, nvenc); FISO_VARIANTS=C,G, FISO_N=8.

Savikontrolė (vienas paleidimas, ne atskiras kontrolinis): funkcija įrašo kiekvieno flow'o didžiausią commit'o vėlavimą
kiekvienai indekso laiko sekundei (santraukos `lateness_s`), gen — savo commit'o vėlavimą (`commit_lateness_s`). Langai (TAI):
  prieš gedimą — [įšilimo pabaiga, gedimas − 1 s);   gedimo — [gedimas − 0,5 s, T(funkcijos paskutinis indeksas)].

Ką vartai APTINKA (FAIL):
  * sveiko flow'o funkcijos praradimą (FN_LOSS / ENC_LOSS, verify missing + invalid) virš sveiko aukštupio praradimų gedimo
    lange (gen eilutės „vėlus commit“ / „praleista“ su TAI) — ir vieną; neatitikimą (mismatch) — visada;
  * sveiko flow'o vėlavimą bet kurią gedimo lango sekundę s > kontrolės atskaita + Wr + ro + Δup(s) + 1 ms. Kontrolės atskaita
    — sveikų flow'ų maks. kontroliniame lange (≥ MIN_PRE_S s kiekvienam flow'ui), bet ne daugiau kaip CTRL_CAP × jo mediana:
    sutrikdyta kontrolė ribos neplečia. Wr + ro — vienkartinis laikymas iki a(i), kai įėjimas miršta ar užstringa (planas §6
    D2); Δup(s) — sveiko aukštupio (gen) commit'o vėlavimo padidėjimas TĄ PAČIĄ sekundę (jo maks. sekundę s − maks.
    kontroliniame lange, ≥ 0), ne viso lango. Funkcijos apdorojimo laikas (proc_ms) ribos NEDIDINA: funkcija, kuri dėl
    sugedusio įėjimo apdoroja lėčiau (fiso_teeth: +10 ms), — FAIL;
  * lango p99: sveiko flow'o per-sekundinių p99 (santraukos p99_s) mediana gedimo lange > ta pati mediana kontroliniame lange +
    Wr + 1 ms (kaip buvęs p99 + Wr + 1 ms tarp paleidimų; mediana — ilgalaikis pablogėjimas, kurio maksimumas nepagauna);
  * paveikto flow'o būseną, neatitinkančią gedimo (missing_fields == 0, fiso_stall: stalls < 1, po SIGCONT praradimai).
Ko vartai NEAPTINKA:
  * vėlavimo, mažesnio už Wr + ro + 1 ms (~6,5 ms prie 50 Hz) virš prieš gedimą buvusio maksimumo, ir vėlavimo, kurį
    sveikas aukštupis tuo pačiu metu pats padidino (Δup jį „paaiškina“);
  * sutrikimų, kurie vienodai paveikia ir langą prieš gedimą (jie pakelia atskaitą), ir trukmesnių nei paleidimas;
  * gfxgen (composite sluoksnių) aukštupio vėlavimo: jis Δup neįskaičiuojamas (gfxgen eilučių su TAI nerašo; jo praradimai
    biudžete — visam gyvenimui);
  * enc: ts_sink tikrina tik CC / sync klaidas ir PES skaičių, ne turinį (turinį tikrina golden / avsync testai).
FAIL ir tada, jei neveikia pati infrastruktūra (flow'ai neatsirado, nėra santraukos, tuščias langas prieš gedimą).
Struktūruoti skaičiai — rezultato `after` lauke (functional.json). „Prieš“ skaičiai — docs/rezultatai/2026-11-12-izoliacija-pries.md.
Testai tikrina tik iš išorės: procesai, signalai, exit kodai, santraukų skaitikliai, verify išvestis.
"""
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

import functional as F  # noqa: E402

VARIANTS = {"C": ("cpu", "host", "x264"), "G": ("gpu", "cuda-linear", "nvenc")}
N = int(os.environ.get("FISO_N", "8"))
FAULT_AT = float(os.environ.get("FISO_FAULT_AT", "12"))  # s po funkcijos starto
STALL_S = 5.0
RUN_S = 26  # funkcijos trukmė
LOGO_RUN_S = 29  # logo: po SIGCONT (ties 17 s) lieka ≥ 10 s patikrai „0 praradimų po atsigavimo“
POST_S = 10  # logo stall: verify paveiktam flow'ui po SIGCONT
WR_MS = 5.0  # Wr = P/4 prie 50 Hz (planas §6 D2)
RO_MS = 0.5  # --read-offset-us numatytoji (ro)
# Funkcijos --warmup. Langas prieš gedimą (kontrolinis) prasideda ne anksčiau kaip po įšilimo IR po visų procesų paleidimo
# (FAULT_TAI["ready"]: paskutinis verify paleistas + SETTLE_S — jo CUDA konteksto kūrimas G variante yra GPU įvykis, kuris
# vėlina sveiką aukštupį; fiso_stall[G]: 37,7 ms vėlus tick'as ~0,5 s po logo starto). Baigiasi PRE_GAP_S iki gedimo.
WARM_S = 5.0
SETTLE_S = 2.0
PRE_GAP_S = 1.0  # langas prieš gedimą baigiasi 1 s iki gedimo
LEAD_S = 0.5  # gedimo langas prasideda 0,5 s iki gedimo
MIN_PRE_S = 4  # tiek sekundžių prieš gedimą privalo būti kiekvienam sveikam flow'ui (kitaip atskaitos nėra — FAIL)
CTRL_CAP = 2.0  # kontrolės maksimumas ribojamas iki CTRL_CAP × jos mediana (sutrikdyta kontrolė ribos neplečia)
VERIFY_S = 22
ENC_PORT = int(os.environ.get("FISO_ENC_PORT", "25000"))
# Funkcijos santraukos srautų (flow'ų) skaitikliai, kuriuos registruojame (praradimai ir būsena).
FN_KEYS = ("fields_ok", "late_fields", "late_skips", "warmup_skips", "clock_skips", "lapped", "too_late", "in_invalid", "read_errors",
           "missing_fields", "detached", "flow_invalid", "reopened", "reattached", "stalls", "lateness_max_us", "gap_invalid",
           "gap_frozen", "gap_black", "gap_startup")
ENC_KEYS = ("frames_encoded", "late_frames", "late_skips_frames", "warmup_skips_frames", "clock_skips_frames", "missing_frames",
            "lapped", "too_late", "in_invalid", "frames_dropped_queue", "encode_errors", "mux_errors", "reattached",
            "audio_underrun_frames", "audio_silence_samples", "packets_audio", "lateness_max_us")
VER_KEYS = ("samples", "present", "missing", "invalid", "mismatch_fields", "too_late", "read_errors", "reattached", "in_reattached",
            "marker_bad", "verify_skipped", "check_dropped")
# Paveikto flow'o spragų turinio patikra (verify --flows 1 --every 1; V17 K3): kaip klasifikuoti išėjimo grain'ai be įėjimo.
GAP_KEYS = ("samples", "present", "missing", "invalid", "mismatch_fields", "outside_diff_fields", "marker_bad", "verify_skipped",
            "sample_lapped", "gap_repeat", "gap_black", "gap_other", "gap_repeat_first", "gap_repeat_last", "gap_black_first",
            "gap_black_last", "last_good_index")
HOLD_FIELDS = 50  # --hold-ms 1000 (numatyta) prie 50 laukų/s


def variants():
    want = [x for x in os.environ.get("FISO_VARIANTS", "C,G").split(",") if x]
    return [(t, *VARIANTS[t]) for t in want]


def hist_of(s, name):
    h = ((s or {}).get("hist") or {}).get(name) or {}
    return {k: h.get(k) for k in ("count", "p50_ms", "p99_ms", "max_ms")}


def flow_view(s, keys, idx):
    fl = (s or {}).get("flows") or []
    return {k: sum(fl[n].get(k, 0) for n in idx if n < len(fl)) for k in keys}


def healthy_of(victim):
    return [n for n in range(N) if n != victim]


def split_summary(s, keys, victim):
    """Sveiki — sumos, paveiktas (victim) — atskirai."""
    return {"healthy": flow_view(s, keys, healthy_of(victim)), "affected": flow_view(s, keys, [victim])}


POSITIONS = tuple(x for x in os.environ.get("FISO_POSITIONS", "first,middle,last").split(",") if x)


def victim_at(pos):
    return {"first": 0, "middle": N // 2, "last": N - 1}[pos]


def ranges(victim):
    """Sveikų flow'ų ištisinės atkarpos (flow_base, kiekis) — po vieną gamintojo procesą kiekvienai."""
    out = []
    if victim > 0:
        out.append((0, victim))
    if victim < N - 1:
        out.append((victim + 1, N - 1 - victim))
    return out


def sgn(pr):
    return pr.p.poll()


def kill_at(t0, at):
    d = t0 + at - time.monotonic()
    if d > 0:
        time.sleep(d)


def tai_ns():
    return time.clock_gettime_ns(time.CLOCK_TAI)


FAULT_TAI = {}  # paskutinio paleidimo momentai TAI ns: {"start"|"ready"|"kill"|"stop"|"cont": ns}


def mark_start():
    """Funkcijos paleidimo momentas (TAI): langas prieš gedimą prasideda ne anksčiau kaip start + WARM_S."""
    FAULT_TAI.clear()
    FAULT_TAI["start"] = tai_ns()


def mark_ready():
    """Visi paleidimo procesai (funkcija, verify) paleisti: + SETTLE_S jų paleidimo GPU įvykiams nusistovėti."""
    FAULT_TAI["ready"] = tai_ns() + int(SETTLE_S * 1e9)


def apply_fault(fault, victim, t0):
    """kill: kill -9 ties FAULT_AT; stall: SIGSTOP ties FAULT_AT, SIGCONT po STALL_S. Grąžina faktinį gedimo laiką (s nuo t0);
    momentai TAI laiku — FAULT_TAI (aukštupio vėlaus tick'o laikui palyginti)."""
    for k in ("kill", "stop", "cont"):
        FAULT_TAI.pop(k, None)
    if fault == "none":
        return None
    kill_at(t0, FAULT_AT)
    t_fault = time.monotonic() - t0
    if fault == "kill":
        FAULT_TAI["kill"] = tai_ns()
        victim.kill9()
    elif fault == "stall":
        FAULT_TAI["stop"] = tai_ns()
        victim.p.send_signal(signal.SIGSTOP)
        time.sleep(STALL_S)
        FAULT_TAI["cont"] = tai_ns()
        victim.p.send_signal(signal.SIGCONT)
    return round(t_fault, 2)


LATE_RE = re.compile(r"\[gen\] vėlus commit: indeksas (\d+), ([\d.]+) ms po T\(i\), TAI (\d+)")
SKIP_RE = re.compile(r"\[gen\] praleista: indeksai (\d+)…(\d+) \((\d+)\), TAI (\d+)")
FIELD_NS = 20_000_000  # 1080i50: T(i) = i · 20 ms


def series(pairs):
    """[[sekundė, µs], …] → {sekundė: ms}."""
    return {int(k): v / 1000.0 for k, v in (pairs or [])}


def upstream(prs):
    """Sveikų aukštupių (gen procesų sąrašas) sutrikimai: praleisti / vėlūs indeksai (eilutės su TAI), commit'o vėlavimas
    kiekvienai indekso laiko sekundei (maks. per procesus), vėlių tick'ų laikas (ms) nuo gedimo momentų."""
    if not isinstance(prs, (list, tuple)):
        prs = [prs]
    late, skipped, sec = [], [], {}
    late_skips = late_commits = 0
    for pr in prs:
        if pr is None:
            continue
        s = pr.summary()
        proc = (s or {}).get("proc") or {}
        late_skips += proc.get("late_skips", 0)
        late_commits += proc.get("late_commits", 0)
        for k, v in series((s or {}).get("commit_lateness_s")).items():
            sec[k] = max(sec.get(k, 0.0), v)
        try:
            txt = pr.err_path.read_text(encoding="utf-8", errors="replace")
            for m in LATE_RE.finditer(txt):
                tai = int(m.group(3))
                late.append({"index": int(m.group(1)), "late_ms": float(m.group(2)),
                             **{f"vs_{k}_ms": round((tai - v) / 1e6, 1) for k, v in FAULT_TAI.items()}})
            skipped += [(int(m.group(1)), int(m.group(2))) for m in SKIP_RE.finditer(txt)]
        except OSError:
            pass
    return {"late_skips": late_skips, "late_commits": late_commits, "late_ticks": late[:10],
            "late_index": [x["index"] for x in late], "skipped": skipped, "lat_s": sec}


def window_budget(up, fault_tai, last_index):
    """Aukštupio praradimai gedimo lange [fault_tai − 0,5 s, T(last_index)]: vėlūs commit'ai ir praleisti indeksai (T(i))."""
    if fault_tai is None or last_index is None:
        return None
    lo, hi = fault_tai - int(LEAD_S * 1e9), last_index * FIELD_NS
    n = sum(1 for i in up.get("late_index") or [] if lo <= i * FIELD_NS <= hi)
    for a, b in up.get("skipped") or []:
        n += sum(1 for i in range(a, b + 1) if lo <= i * FIELD_NS <= hi)
    return n


def split_windows(sec, fault_tai, end_tai, from_tai=0):
    """{sekundė: reikšmė} → (prieš gedimą, gedimo lange) reikšmių sąrašai. Sekundė s apima [s, s + 1) s TAI (indekso laikas).
    Prieš gedimą — tik sekundės, prasidėjusios ne anksčiau kaip from_tai (įšilimas ir visų procesų paleidimas)."""
    pre, flt = [], []
    for k, v in sec.items():
        lo, hi = k * 1_000_000_000, (k + 1) * 1_000_000_000
        if hi <= fault_tai - int(PRE_GAP_S * 1e9):
            if lo >= from_tai:
                pre.append(v)
        elif hi > fault_tai - int(LEAD_S * 1e9) and lo <= end_tai:
            flt.append(v)
    return pre, flt


def ctl_from(r):
    """Kontrolinio lango pradžia (TAI): po funkcijos įšilimo ir po visų procesų paleidimo (+ SETTLE_S)."""
    return max((r.get("start_tai") or 0) + int(WARM_S * 1e9), r.get("ready_tai") or 0)


def window_losses(r, n):
    """Sveiko flow'o n funkcijos praradimai GEDIMO LANGE (loss_s sekundės [gedimas − LEAD_S, T(paskutinis)]) — ta pati
    atkarpa kaip aukštupio biudžetas (window_budget); paleidimo sutrikimai į ją nepatenka."""
    _, flt = split_windows((r.get("loss_s") or {}).get(n) or {}, r["fault_tai"], r["end_tai"])
    return int(sum(flt))


def stop_all(procs):
    for pr in procs:
        if pr is not None and pr.p.poll() is None:
            try:
                pr.p.send_signal(signal.SIGCONT)  # sustabdytas procesas SIGTERM'o nematytų
            except ProcessLookupError:
                pass
            pr.term()


def common_record(fault, t_fault, victim, pos):
    return {"fault": fault, "fault_at_s": t_fault, "stall_s": STALL_S if fault == "stall" else None,
            "fault_tai": FAULT_TAI.get("kill", FAULT_TAI.get("stop")), "start_tai": FAULT_TAI.get("start"),
            "ready_tai": FAULT_TAI.get("ready"), "victim": victim, "position": pos}


def with_budget(rec, s_fn, up_whole_extra=0):
    """Gedimo lango biudžetas (window_budget) iš aukštupio eilučių; up_whole_extra — aukštupis be eilučių (gfxgen: visam
    gyvenimui). Ir savikontrolės duomenys: sveikų flow'ų lateness_s, gedimo lango pabaiga."""
    w = window_budget(rec.get("upstream") or {}, rec.get("fault_tai"), (s_fn or {}).get("last_index"))
    rec["budget"] = None if w is None else w + up_whole_extra
    fl = (s_fn or {}).get("flows") or []
    rec["lat_s"] = {n: series(fl[n].get("lateness_s")) if n < len(fl) else {} for n in healthy_of(rec["victim"])}
    rec["loss_s"] = {n: {int(k): v for k, v in (fl[n].get("loss_s") or [])} if n < len(fl) else {}
                     for n in healthy_of(rec["victim"])}
    rec["p99_s"] = {n: series(fl[n].get("p99_s")) if n < len(fl) else {} for n in healthy_of(rec["victim"])}
    rec["end_tai"] = ((s_fn or {}).get("last_index") or 0) * FIELD_NS
    return rec


def gens(out, sub, d, v, victim, *extra):
    """Sveikų flow'ų gamintojai — po vieną kiekvienai ištisinei atkarpai."""
    return [F.FProc(out, f"{sub}/genA{k}", "gen", d, "--flows", cnt, "--flow-base", base, "--duration", RUN_S + 60, *extra, *v)
            for k, (base, cnt) in enumerate(ranges(victim))]


# --- logo ----------------------------------------------------------------------------------------------------------------

def run_logo(out, tag, impl, storage, fault, pos="last", extra=(), suffix="", policy=None):
    """policy (V17): None — numatyta (--on-loss invalid); "freeze" / "black" — logo su --on-loss, paveiktam flow'ui atskiras
    verify --every 1 (vg) klasifikuoja kiekvieną spragos grain'ą (gap_repeat / gap_black / gap_other)."""
    sub = f"fiso-logo-{tag}-{fault}-{pos}{suffix}"
    victim = victim_at(pos)
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    ga = gens(out, sub, d, v, victim)
    gb = F.FProc(out, f"{sub}/genB", "gen", d, "--flows", 1, "--flow-base", victim, "--duration", RUN_S + 60, *v)
    if policy:
        extra = (*extra, "--on-loss", policy)
    lg = vf = vp = vg = None
    rl = rv = rp = rg = None
    t_fault = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)]):
            return {"err": "gen flow'ai neatsirado"}
        t0 = time.monotonic()
        mark_start()
        lg = F.FProc(out, sub, "logo", d, "--flows", N, "--duration", LOGO_RUN_S, "--warmup", WARM_S, *extra, *v)
        if not F.wait_flows(d, [F.role_flow_id(2, n) for n in range(N)], 30):
            return {"err": "logo išėjimo flow'ai neatsirado"}
        vf = F.FProc(out, sub, "verify", d, "--mode", "logo", "--flows", N, "--every", 5, "--duration", VERIFY_S, "--warmup", 1,
                     *F.vimpl(impl, storage))
        if policy:
            vg = F.FProc(out, f"{sub}/gap", "verify", d, "--mode", "logo", "--flows", 1, "--flow-base", victim, "--every", 1,
                         "--duration", VERIFY_S, "--warmup", 1, *F.vimpl(impl, storage))
        if fault == "stall":
            # po SIGCONT: paveiktas flow'as POST_S s — be praradimų (1 s atsigavimui). Paleidžiamas dabar (ne ties SIGCONT): jo
            # CUDA kontekstas (G) kuriamas prieš langą, ne gedimo lange (kito proceso GPU įvykis — ne tikrinamos funkcijos).
            w = round(FAULT_AT + STALL_S + 1 - (time.monotonic() - t0), 2)
            vp = F.FProc(out, f"{sub}/post", "verify", d, "--mode", "logo", "--flows", 1, "--flow-base", victim, "--every", 1,
                         "--duration", w + POST_S, "--warmup", w, *F.vimpl(impl, storage))
        mark_ready()
        t_fault = apply_fault(fault, gb, t0)
        if vp is not None:
            rp = vp.wait(timeout=POST_S + 60)
        rv = vf.wait(timeout=VERIFY_S + 60)
        if vg is not None:
            rg = vg.wait(timeout=VERIFY_S + 60)
        rl = lg.wait(timeout=LOGO_RUN_S + 60)
    finally:
        stop_all([vg, vp, vf, lg, gb, *ga])
    sl, sv = F.vsummary(lg, "logo"), F.vsummary(vf, "verify-logo")
    if sl is None or sv is None:
        return {"err": f"nėra santraukų (logo={sl is not None}, verify={sv is not None})"}
    hv = healthy_of(victim)
    rec = {**common_record(fault, t_fault, victim, pos), "exit": {"logo": rl, "verify": rv}, "fn": split_summary(sl, FN_KEYS, victim),
           "out_lateness": hist_of(sl, "out_lateness"), "proc": hist_of(sl, "proc_ms"),
           "verify": split_summary(sv, VER_KEYS, victim), "verify_proc": (sv.get("proc") or {}), "fatal": sl.get("fatal"),
           "upstream": upstream(ga), "fn_flows": [flow_view(sl, FN_KEYS, [n]) for n in hv],
           "verify_flows": [flow_view(sv, VER_KEYS, [n]) for n in hv]}
    if vp is not None:
        sp = F.vsummary(vp, "verify-logo")
        rec["post"] = {"exit": rp, **(flow_view(sp, VER_KEYS, [0]) if sp else {"err": "nėra post santraukos"})}
    rec["policy"] = policy or "invalid"
    rec["verify_last_index"] = sv.get("last_index")
    if vg is not None:
        sg = F.vsummary(vg, "verify-logo")
        rec["gapv"] = {"exit": rg, "last_index": (sg or {}).get("last_index"),
                       **({k: ((sg.get("flows") or [{}])[0]).get(k) for k in GAP_KEYS} if sg else {"err": "nėra gap santraukos"})}
    return with_budget(rec, sl)


# --- composite -----------------------------------------------------------------------------------------------------------

def run_composite(out, tag, impl, storage, fault, pos="last"):
    sub = f"fiso-comp-{tag}-{fault}-{pos}"
    victim = victim_at(pos)
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    xy = ["--layer-xy", "1488,64"]
    g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", N, "--duration", RUN_S + 60, *v)
    xa = [F.FProc(out, f"{sub}/gfxA{k}", "gfxgen", d, "--flows", cnt, "--flow-base", base, "--layers", 1, "--layer-geom", "logo",
                  "--mode", "pace", "--duration", RUN_S + 60, *v) for k, (base, cnt) in enumerate(ranges(victim))]
    xb = F.FProc(out, f"{sub}/gfxB", "gfxgen", d, "--flows", 1, "--flow-base", victim, "--layers", 1, "--layer-geom", "logo",
                 "--mode", "pace", "--duration", RUN_S + 60, *v)
    c = va = None
    rc = rv = None
    t_fault = None
    try:
        ids = [F.role_flow_id(1, n) for n in range(N)] + [F.role_flow_id(5, n * 8) for n in range(N)]
        if not F.wait_flows(d, ids):
            return {"err": "gen / gfxgen flow'ai neatsirado"}
        t0 = time.monotonic()
        mark_start()
        c = F.FProc(out, sub, "composite", d, "--flows", N, "--layers", 1, *xy, "--opacity", "1.0", "--base", "gen",
                    "--out-format", "v210", "--deadline-ms", F.COMP_DEADLINE_MS, "--duration", RUN_S, "--warmup", WARM_S, *v)
        if not F.wait_flows(d, [F.role_flow_id(6, n) for n in range(N)], 30):
            return {"err": "composite išėjimo flow'ai neatsirado"}
        va = F.FProc(out, sub, "verify", d, "--mode", "composite", "--flows", N, "--layers", 1, "--layer-geom", "logo",
                     "--gfx-mode", "pace", "--base", "gen", "--out-format", "v210", "--every", 5, "--duration", VERIFY_S,
                     "--warmup", 1, *xy, "--opacity", "1.0", *F.vimpl(impl, storage))
        mark_ready()
        t_fault = apply_fault(fault, xb, t0)
        rv = va.wait(timeout=VERIFY_S + 60)
        rc = c.wait(timeout=RUN_S + 60)
    finally:
        stop_all([va, c, xb, *xa, g])
    sc, sv = F.vsummary(c, "composite"), F.vsummary(va, "verify-composite")
    if sc is None or sv is None:
        return {"err": f"nėra santraukų (composite={sc is not None}, verify={sv is not None})"}
    # sveiki aukštupiai: gen (bazės; vėlavimas commit_lateness_s ir eilutės) ir gfxgen A (sluoksniai; tik praradimai visam
    # gyvenimui — gfxgen eilučių su TAI nerašo)
    up = upstream(g)
    ux = {"late_skips": 0, "late_commits": 0}
    for x in xa:
        sx = (x.summary() or {}).get("proc") or {}
        ux["late_skips"] += sx.get("late_skips", 0)
        ux["late_commits"] += sx.get("late_commits", 0)
    up["gfxgen"] = ux
    hv = healthy_of(victim)
    rec = {**common_record(fault, t_fault, victim, pos), "exit": {"composite": rc, "verify": rv},
           "fn": split_summary(sc, FN_KEYS, victim), "out_lateness": hist_of(sc, "out_lateness"), "proc": hist_of(sc, "proc_ms"),
           "verify": split_summary(sv, VER_KEYS, victim), "verify_proc": (sv.get("proc") or {}), "fatal": sc.get("fatal"),
           "upstream": up, "fn_flows": [flow_view(sc, FN_KEYS, [n]) for n in hv],
           "verify_flows": [flow_view(sv, VER_KEYS, [n]) for n in hv]}
    rec["policy"] = "invalid"  # V17: numatyta --on-loss (gap_policy_bad tikrina paveiktą flow'ą kaip logo)
    rec["verify_last_index"] = sv.get("last_index")
    return with_budget(rec, sc, ux["late_skips"] + ux["late_commits"])


# --- enc -----------------------------------------------------------------------------------------------------------------

def run_enc(out, tag, impl, storage, enc, fault, pos="last", audio_only=False, extra=(), suffix=""):
    sub = f"fiso-enc{'a' if audio_only else ''}-{tag}-{fault}-{pos}{suffix}"
    victim = victim_at(pos)
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    sink_json = pathlib.Path(out) / sub / "sink.json"
    ga = gens(out, sub, d, v, victim, "--audio", 1)
    gbv = None
    if audio_only:  # paveiktas srautas: video ir garsas — skirtingi procesai, gedimas — tik garso gamintojui
        gbv = F.FProc(out, f"{sub}/genBv", "gen", d, "--flows", 1, "--flow-base", victim, "--duration", RUN_S + 60, *v)
        gb = F.FProc(out, f"{sub}/genBa", "gen", d, "--flows", 1, "--flow-base", victim, "--video", 0, "--audio", 1,
                     "--duration", RUN_S + 60, *v)
    else:
        gb = F.FProc(out, f"{sub}/genB", "gen", d, "--flows", 1, "--flow-base", victim, "--audio", 1, "--duration", RUN_S + 60, *v)
    e = sink = None
    re_ = None
    t_fault = None
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, n) for n in range(N)] + [F.role_flow_id(3, n) for n in range(N)]):
            return {"err": "gen flow'ai neatsirado"}
        sink = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "perf/ts_sink.py"),
                                 "--port-base", str(ENC_PORT), "--flows", str(N), "--seconds", str(RUN_S + 4), "--out",
                                 str(sink_json)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ea = ["--encoder", enc, "--port-base", ENC_PORT] + (["--x264-preset", "ultrafast"] if enc == "x264" else [])
        t0 = time.monotonic()
        mark_start()
        e = F.FProc(out, sub, "enc", d, "--flows", N, "--audio", 1, "--duration", RUN_S, "--warmup", WARM_S, *ea, *extra, *v)
        mark_ready()
        t_fault = apply_fault(fault, gb, t0)
        re_ = e.wait(timeout=RUN_S + 60)
        sink.wait(timeout=60)
    finally:
        if sink is not None and sink.poll() is None:
            sink.kill()
        stop_all([e, gb, gbv, *ga])
    se = F.vsummary(e, "enc")
    if se is None:
        return {"err": "nėra enc santraukos"}
    try:
        sk = json.loads(sink_json.read_text())
    except (OSError, ValueError):
        sk = {}
    sflows = sk.get("flows") or []
    hv = healthy_of(victim)
    sink = {"healthy_video_pes_min": min((sflows[n].get("video_pes", 0) for n in hv if n < len(sflows)), default=None),
            "healthy_cc_errors": sum(sflows[n].get("cc_errors", 0) for n in hv if n < len(sflows)),
            "healthy_sync_errors": sum(sflows[n].get("sync_errors", 0) for n in hv if n < len(sflows)),
            "affected_video_pes": sflows[victim].get("video_pes") if len(sflows) > victim else None}
    rec = {**common_record(fault, t_fault, victim, pos), "exit": {"enc": re_}, "fn": split_summary(se, ENC_KEYS, victim),
           "out_lateness": hist_of(se, "frame_lateness"), "proc": hist_of(se, "weave_ms"), "fatal": se.get("fatal"),
           "upstream": upstream(ga), "fn_flows": [flow_view(se, ENC_KEYS, [n]) for n in hv], "sink": sink}
    return with_budget(rec, se)


def run_enc_audio(out, tag, impl, storage, enc, fault, pos="last"):
    return run_enc(out, tag, impl, storage, enc, fault, pos, audio_only=True)


# --- vartai --------------------------------------------------------------------------------------------------------------

def describe(r):
    if r.get("err"):
        return r["err"]
    fh, fa = r["fn"]["healthy"], r["fn"]["affected"]
    pick = lambda d: {k: v for k, v in d.items() if v and k not in ("fields_ok", "frames_encoded")}  # noqa: E731
    lat = r["out_lateness"]
    s = (f"paveiktas #{r.get('victim')} ({r.get('position')}) exit={r['exit']} sveiki: "
         f"ok={fh.get('fields_ok', fh.get('frames_encoded'))} nenuliniai={pick(fh) or '-'} | paveiktas: nenuliniai={pick(fa) or '-'} | "
         f"lateness p50={lat['p50_ms']} p99={lat['p99_ms']} max={lat['max_ms']} ms")
    if "verify" in r:
        vh, va = r["verify"]["healthy"], r["verify"]["affected"]
        s += (f" | verify sveiki: samples={vh.get('samples')} missing={vh.get('missing')} invalid={vh.get('invalid')} "
              f"mismatch={vh.get('mismatch_fields')}; paveiktas: samples={va.get('samples')} present={va.get('present')} "
              f"missing={va.get('missing')} invalid={va.get('invalid')} mismatch={va.get('mismatch_fields')} "
              f"reattached={va.get('reattached')} too_late={va.get('too_late')}")
    return s


FN_LOSS = ("late_fields", "late_skips", "lapped", "too_late", "in_invalid", "read_errors", "missing_fields", "flow_invalid")


def upstream_budget(r):
    """Kiek indeksų kiekvienam sveikam flow'ui prarado / pavėlavo (> --deadline-ms) pats sveikas aukštupis GEDIMO LANGE
    (with_budget / window_budget); paleidimo ir pabaigos sutrikimai neįskaičiuojami."""
    b = r.get("budget")
    return 0 if b is None else b


def median(v):
    s = sorted(v)
    return (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2 if s else None


def fault_seconds(sec, fault_tai, end_tai):
    """{sekundė: reikšmė} → [(sekundė, reikšmė)] gedimo lange (kaip split_windows)."""
    lo_f = fault_tai - int(LEAD_S * 1e9)
    return sorted((k, v) for k, v in sec.items()
                  if (k + 1) * 1_000_000_000 > lo_f and k * 1_000_000_000 <= end_tai
                  and (k + 1) * 1_000_000_000 > fault_tai - int(PRE_GAP_S * 1e9))


def lateness_bound(r):
    """Savikontrolės riba be Δup: (bazė ms, atskaita ms, Δup maks. ms) arba (None, priežastis, None).
    bazė = atskaita + Wr + ro + 1 ms; atskaita = min(sveikų flow'ų maks. kontroliniame lange, CTRL_CAP × jo mediana).
    Sekundės s riba — bazė + up_delta(r, s). Kiekvienam sveikam flow'ui kontroliniame lange ≥ MIN_PRE_S s."""
    ft, et = r.get("fault_tai"), r.get("end_tai")
    if ft is None:
        return None, "nėra gedimo laiko", None
    pre = []
    for n, sec in (r.get("lat_s") or {}).items():
        p, _ = split_windows(sec, ft, et, ctl_from(r))
        if len(p) < MIN_PRE_S:
            return None, f"sveiko #{n} kontrolinis langas {len(p)} s < {MIN_PRE_S} s", None
        pre += p
    if not pre:
        return None, "kontrolinio lango nėra", None
    ref = min(max(pre), CTRL_CAP * median(pre))
    dmax = max((up_delta(r, s) for n_sec in (r.get("lat_s") or {}).values() for s, _ in fault_seconds(n_sec, ft, et)), default=0.0)
    return ref + WR_MS + RO_MS + 1.0, ref, dmax


def up_delta(r, s):
    """Δup(s): sveiko aukštupio commit'o vėlavimo padidėjimas sekundę s (maks. tą sekundę − maks. kontroliniame lange, ≥ 0)."""
    up = (r.get("upstream") or {}).get("lat_s") or {}
    up_pre, _ = split_windows(up, r["fault_tai"], r["end_tai"], ctl_from(r))
    if not up_pre or s not in up:
        return 0.0
    return max(0.0, up[s] - max(up_pre))


def max_lateness_bad(r, what):
    """Kiekvieno sveiko flow'o vėlavimas kiekvieną gedimo lango sekundę s ≤ bazė + Δup(s) (lateness_bound)."""
    base, ref, _ = lateness_bound(r)
    if base is None:
        return [f"{what}: {ref}"]
    bad = []
    for n, sec in (r.get("lat_s") or {}).items():
        flt = fault_seconds(sec, r["fault_tai"], r["end_tai"])
        if not flt:
            bad.append(f"sveiko #{n} gedimo lange nėra {what} duomenų")
            continue
        worst = max(flt, key=lambda kv: kv[1] - up_delta(r, kv[0]))
        s, v = worst
        lim = base + up_delta(r, s)
        if v > lim:
            bad.append(f"sveiko #{n} maks. vėlavimas gedimo lange {v:.2f} ms (sek. {s}) > {lim:.2f} ms "
                       f"(atskaita {ref:.2f}, Δup {up_delta(r, s):.2f})")
    return bad


def p99_bad(r, what):
    """Lango p99: kiekvieno sveiko flow'o per-sekundinių p99 mediana gedimo lange ≤ ta pati mediana kontroliniame lange +
    Wr + 1 ms (buvęs „p99 po gedimo ≤ p99 be gedimo + Wr + 1 ms“, dabar — to paties paleidimo langai)."""
    bad = []
    ft, et = r.get("fault_tai"), r.get("end_tai")
    for n, sec in (r.get("p99_s") or {}).items():
        pre, _ = split_windows(sec, ft, et, ctl_from(r))
        flt = [v for _, v in fault_seconds(sec, ft, et)]
        if len(pre) < MIN_PRE_S or not flt:
            bad.append(f"sveiko #{n} {what} p99: kontrolė {len(pre)} s, gedimo langas {len(flt)} s")
            continue
        a, b = median(flt), median(pre)
        if a > b + WR_MS + 1.0:
            bad.append(f"sveiko #{n} {what} lango p99 (per-sekundinių mediana) {b:.2f} → {a:.2f} ms > +{WR_MS + 1.0} ms")
    return bad


def logo_gate(fault, r):
    """V10 / V11 vartai: kiekvienam sveikam flow'ui funkcijos praradimai (FN_LOSS) ir verify missing + invalid ≤ aukštupio
    biudžetas (upstream_budget; be aukštupio sutrikimų — 0), mismatch == 0 visada; kiekvieno sveiko maks. vėlavimas gedimo lange
    ≤ max_lateness_bad riba; paveiktam — missing_fields > 0, fiso_stall: stalls ≥ 1 ir po SIGCONT POST_S s be praradimų.
    Grąžina neatitikimų sąrašą (tuščias — PASS)."""
    bad = []
    fa = r["fn"]["affected"]
    budget = upstream_budget(r)
    hv = healthy_of(r["victim"])
    for n, f in zip(hv, r.get("fn_flows") or []):
        lost = window_losses(r, n)  # gedimo lange — ta pati atkarpa kaip aukštupio biudžetas (ne viso gyvenimo skaitikliai)
        if lost > budget:
            bad.append(f"sveiko #{n} funkcijos praradimai gedimo lange {lost} > aukštupio {budget} "
                       f"(viso gyvenimo { {k: f[k] for k in FN_LOSS if f.get(k)} })")
    for n, v in zip(hv, r.get("verify_flows") or []):
        lost = v.get("missing", 0) + v.get("invalid", 0)
        if lost > budget:
            bad.append(f"verify sveiko #{n} missing + invalid {lost} > aukštupio {budget}")
        if v.get("mismatch_fields", 0):
            bad.append(f"verify sveiko #{n} mismatch={v['mismatch_fields']}")
    bad += max_lateness_bad(r, "out_lateness")
    bad += p99_bad(r, "out_lateness")
    if not fa.get("missing_fields", 0):
        bad.append("paveikto missing_fields == 0")
    if fault == "kill":
        bad += gap_policy_bad(r)
    if fault == "stall":
        if fa.get("stalls", 0) < 1:
            bad.append(f"paveikto stalls={fa.get('stalls', 0)} < 1")
        post = r.get("post") or {}
        if post.get("err") or post.get("exit") != 0 or post.get("samples", 0) == 0 or any(
                post.get(k, 0) for k in ("missing", "invalid", "mismatch_fields")):
            bad.append(f"po SIGCONT {POST_S} s: {post}")
    return bad


def gap_policy_bad(r):
    """V17 (K3) paveiktam flow'ui po kill -9: rašytojas commit'ina KIEKVIENĄ spragos indeksą pagal --on-loss.
    invalid — verify invalid == spragos indeksų skaičius jo lange ± 2 (nuo pirmo indekso po mirties iki verify paskutinio),
              missing == 0; logo gap_invalid > 0.
    freeze  — (verify --every 1) invalid == 0, missing == 0; spragos grain'ai == paskutinis geras to paties pariteto nuo g + 1
              iki g + 50 (+1, jei kadras baigiamas: --hold-ms 1000), tada juodi nuo lyginio indekso iki galo; gap_other == 0.
    black   — invalid == 0, missing == 0; juodi nuo g + 1 iki galo; gap_repeat == 0, gap_other == 0."""
    bad = []
    pol = r.get("policy", "invalid")
    fa = r["fn"]["affected"]
    va = r["verify"]["affected"]
    if va.get("missing", 0):
        bad.append(f"paveikto verify missing={va['missing']} (K3: kiekvienas indeksas commit'inamas)")
    if pol == "invalid":
        ft, last = r.get("fault_tai"), r.get("verify_last_index")
        if ft is None or last is None:
            return bad + ["nėra gedimo laiko / verify paskutinio indekso"]
        want = last - ft // FIELD_NS  # indeksai po paskutinio gamintojo grain'o (T(i) ≤ mirtis) iki verify pabaigos
        if abs(va.get("invalid", 0) - want) > 2:
            bad.append(f"paveikto verify invalid={va.get('invalid', 0)} ≠ spragos indeksų {want} ± 2")
        if not fa.get("gap_invalid", 0) or fa.get("gap_frozen", 0) or fa.get("gap_black", 0):
            bad.append(f"paveikto logo gap_invalid={fa.get('gap_invalid', 0)} gap_frozen={fa.get('gap_frozen', 0)} "
                       f"gap_black={fa.get('gap_black', 0)} (laukta tik invalid)")
        return bad
    if va.get("invalid", 0):
        bad.append(f"paveikto verify invalid={va['invalid']} (su {pol} turi būti 0)")
    g = r.get("gapv") or {}
    if g.get("err") or g.get("exit") not in (0, 4):
        return bad + [f"spragų verify: {g}"]
    for k in ("missing", "invalid", "mismatch_fields", "outside_diff_fields", "gap_other", "verify_skipped", "sample_lapped"):
        if g.get(k):
            bad.append(f"spragų verify {k}={g[k]}")
    lg_ = g.get("last_good_index")
    bf, bl, nb = g.get("gap_black_first"), g.get("gap_black_last"), g.get("gap_black") or 0
    if lg_ is None or bf is None or bl is None:
        return bad + [f"spragų verify be geros / juodos atkarpos: {g}"]
    if nb != bl - bf + 1 or bl != g.get("last_index"):
        bad.append(f"juodi ne ištisai iki galo: {nb} grain'ų [{bf}, {bl}], verify paskutinis {g.get('last_index')}")
    if pol == "freeze":
        rf, rl_, nr = g.get("gap_repeat_first"), g.get("gap_repeat_last"), g.get("gap_repeat") or 0
        if rf is None or rl_ is None:
            return bad + [f"freeze be kartojimų: {g}"]
        if rf != lg_ + 1 or nr != rl_ - rf + 1:
            bad.append(f"kartojimai ne nuo g + 1 ištisai: g={lg_} [{rf}, {rl_}] n={nr}")
        if rl_ - lg_ not in (HOLD_FIELDS, HOLD_FIELDS + 1):
            bad.append(f"kartota {rl_ - lg_} laukų po paskutinio gero (laukta {HOLD_FIELDS} arba {HOLD_FIELDS + 1} — --hold-ms 1000)")
        if bf != rl_ + 1 or bf % 2:
            bad.append(f"juoda ne iškart po kartojimų kadro riba: kartota iki {rl_}, juoda nuo {bf}")
        if not fa.get("gap_frozen", 0) or not fa.get("gap_black", 0) or fa.get("gap_invalid", 0):
            bad.append(f"logo gap_frozen={fa.get('gap_frozen', 0)} gap_black={fa.get('gap_black', 0)} gap_invalid={fa.get('gap_invalid', 0)}")
    else:
        if g.get("gap_repeat"):
            bad.append(f"black: gap_repeat={g['gap_repeat']}")
        if bf != lg_ + 1:
            bad.append(f"black: juoda ne iškart (g={lg_}, juoda nuo {bf})")
        if not fa.get("gap_black", 0) or fa.get("gap_frozen", 0) or fa.get("gap_invalid", 0):
            bad.append(f"logo gap_black={fa.get('gap_black', 0)} gap_frozen={fa.get('gap_frozen', 0)} gap_invalid={fa.get('gap_invalid', 0)}")
    return bad


ENC_LOSS = ("late_frames", "missing_frames", "lapped", "too_late", "in_invalid", "frames_dropped_queue", "late_skips_frames",
            "audio_underrun_frames")
SAMPLES_PER_FRAME = 1920  # 1080i50: 48 kHz / 25
AAC_FRAME = 1024
FRAME_NS = 2 * FIELD_NS


def aac_packets(frames):
    """Laukiamas AAC paketų skaičius srautui, kurio visi kadrai su garsu: frames · 1920 / 1024 (± 2 — paskutinis neužpildytas
    ir koderio išleidimas)."""
    return frames * SAMPLES_PER_FRAME / AAC_FRAME


def enc_gate(fault, r):
    """V12 vartai: kiekvienam sveikam srautui enc praradimai (ENC_LOSS, ir garsas) ≤ aukštupio biudžetas, frames_encoded ≥
    25 · RUN_S − 1 − biudžetas, tyla ≤ 1920 · biudžetas mėginių, packets_audio = frames_encoded · 1920 / 1024 ± 2; TS be
    CC / sync klaidų; maks. vėlavimas
    gedimo lange kiekvienam sveikam (max_lateness_bad); paveiktam — missing_frames > 0."""
    bad = []
    budget = upstream_budget(r)
    for n, f in zip(healthy_of(r["victim"]), r.get("fn_flows") or []):
        lost = window_losses(r, n)  # gedimo lange (kaip aukštupio biudžetas)
        if lost > budget:
            bad.append(f"sveiko #{n} enc praradimai gedimo lange {lost} > aukštupio {budget} "
                       f"(viso gyvenimo { {k: f[k] for k in ENC_LOSS if f.get(k)} })")
        if f.get("frames_encoded", 0) < 25 * RUN_S - 1 - budget:
            bad.append(f"sveiko #{n} frames_encoded={f.get('frames_encoded', 0)} < {25 * RUN_S - 1 - budget}")
        if f.get("audio_silence_samples", 0) > 1920 * budget:  # tyla tik už aukštupio prarastus kadrus (1080i50: 1920 / kadrui)
            bad.append(f"sveiko #{n} audio_silence_samples={f.get('audio_silence_samples', 0)} > {1920 * budget}")
        want_pa = aac_packets(f.get("frames_encoded", 0))
        if abs(f.get("packets_audio", 0) - want_pa) > 2:  # anksčiau — tik „> 0“
            bad.append(f"sveiko #{n} packets_audio={f.get('packets_audio', 0)} ≠ {want_pa:.0f} ± 2 (frames_encoded · 1920 / 1024)")
    sk = r.get("sink") or {}
    if sk.get("healthy_cc_errors", 0) or sk.get("healthy_sync_errors", 0):
        bad.append(f"TS klaidos sveikuose: {sk}")
    bad += max_lateness_bad(r, "frame_lateness")
    bad += p99_bad(r, "frame_lateness")
    if not r["fn"]["affected"].get("missing_frames", 0):
        bad.append("paveikto missing_frames == 0")
    return bad


def enc_audio_gate(fault, r):
    """fiso_enc_audio: sveikiems — enc_gate taisyklės (be paveikto missing_frames); paveiktam — video nenukentėjo
    (frames_encoded ≥ 25 · RUN_S − 1 − biudžetas, video praradimų nėra), audio_underrun_frames = (T(paskutinis) − gedimas) /
    40 ms ± 3."""
    bad = [b for b in enc_gate(fault, r) if not b.startswith("paveikto missing_frames")]
    fa, budget = r["fn"]["affected"], upstream_budget(r)
    # Garsas prarastas nuo gedimo iki pabaigos: laukiama (T(paskutinis) − gedimas) / 40 ms kadrų ± 3 (aptikimas ≤ 1 kadras,
    # apvalinimas, gedimo momentas tarp kadrų) — anksčiau tik „> 0“.
    want_u = round(((r.get("end_tai") or 0) - (r.get("fault_tai") or 0)) / FRAME_NS)
    if abs(fa.get("audio_underrun_frames", 0) - want_u) > 3:
        bad.append(f"paveikto audio_underrun_frames={fa.get('audio_underrun_frames', 0)} ≠ {want_u} ± 3 (nuo gedimo iki pabaigos)")
    if fa.get("frames_encoded", 0) < 25 * RUN_S - 1 - budget:
        bad.append(f"paveikto video frames_encoded={fa.get('frames_encoded', 0)} < {25 * RUN_S - 1 - budget}")
    vlost = {k: fa[k] for k in ENC_LOSS if k != "audio_underrun_frames" and fa.get(k)}
    if sum(vlost.values()) > budget:
        bad.append(f"paveikto video praradimai {vlost} > {budget}")
    return bad


def probe(out, name, runner, faults):
    """Kiekvienam variantui ir kiekvienai paveikto flow'o vietai (pirmas, vidurinis, paskutinis) — vienas paleidimas su gedimu
    (savikontrolė: langas prieš gedimą tame pačiame paleidime)."""
    res = []
    for (tag, impl, storage, enc), pos in [(v, p) for v in variants() for p in POSITIONS]:
        tid = f"{name}[{tag}:{pos}]"
        args = (out, tag, impl, storage) + ((enc,) if runner in (run_enc, run_enc_audio) else ())
        gate = {run_enc: enc_gate, run_enc_audio: enc_audio_gate}.get(runner, logo_gate)
        for fault in faults:
            r = runner(*args, fault, pos)
            if r.get("err"):
                res.append(F.result(tid, "FAIL", why=f"su gedimu ({fault}): {describe(r)}"))
                continue
            lim, pmax, dup = lateness_bound(r)
            why = (f"{fault} ties {r['fault_at_s']} s: {describe(r)} | vėlavimo riba "
                   f"{None if lim is None else round(lim, 2)} ms + Δup(s) (atskaita {pmax}, Δup maks. {dup})")
            u = r["upstream"] or {}
            brief = {k: v for k, v in u.items() if k not in ("late_index", "skipped", "lat_s")}
            res.append(F.result(f"{tid}:upstream", "INFO", why=f"sveikas aukštupis (gen): {brief}; biudžetas gedimo lange "
                                                              f"{r.get('budget')} (visam gyvenimui "
                                                              f"{u.get('late_skips', 0) + u.get('late_commits', 0)})"))
            bad = gate(fault, r)
            if r.get("post"):
                why += f" | po SIGCONT: {r['post']}"
            res.append(F.result(tid, "FAIL" if bad else "PASS", why=(f"{'; '.join(bad)} || " if bad else "") + why, after=r))
    return res


POLICY_POS = os.environ.get("FISO_POLICY_POS", "middle")  # freeze / black — viena vieta kiekvienam variantui


def fiso_kill(out):
    """kill -9 (numatyta --on-loss invalid: visos vietos) ir V17 politikos freeze / black (POLICY_POS vietoje)."""
    res = probe(out, "fiso_kill", run_logo, ["kill"])
    for pol in ("freeze", "black"):
        for tag, impl, storage, _ in variants():
            tid = f"fiso_kill[{tag}:{POLICY_POS}:{pol}]"
            r = run_logo(out, tag, impl, storage, "kill", POLICY_POS, suffix=f"-{pol}", policy=pol)
            if r.get("err"):
                res.append(F.result(tid, "FAIL", why=describe(r)))
                continue
            bad = logo_gate("kill", r)
            res.append(F.result(tid, "FAIL" if bad else "PASS", why=(f"{'; '.join(bad)} || " if bad else "") +
                                f"{describe(r)} | spragų verify: {r.get('gapv')}", after=r))
    return res


def fiso_stall(out):
    return probe(out, "fiso_stall", run_logo, ["stall"])


def fiso_composite(out):
    return probe(out, "fiso_composite", run_composite, ["kill"])


def fiso_enc(out):
    return probe(out, "fiso_enc", run_enc, ["kill"])


def fiso_enc_audio(out):
    return probe(out, "fiso_enc_audio", run_enc_audio, ["kill"])


TEETH = (("logo", "C", "middle"), ("logo", "G", "first"), ("enc", "C", "last"))


def fiso_teeth(out):
    """Vartų „dantys“ (meta-testas; kill): --test-fault-proc-delay-ms D — funkcija, kol vieno įėjimo nėra, A fazėje apdorojime
    užmiega D ms (sveiki vėluoja dėl sugedusio įėjimo). logo C (paveiktas vidurinis), logo G (pirmas), enc C x264 (paskutinis):
    D = 30 — vartai FAIL; D = 10 (< P: be praradimų, tik vėlavimas) — FAIL dėl vėlavimo ribos ir dėl lango p99; D = 0 — PASS."""
    res = []
    for func, tag, pos in TEETH:
        impl, storage, enc = VARIANTS[tag]
        tid = f"fiso_teeth[{func}-{tag}]"
        if tag not in [t for t, *_ in variants()]:
            continue
        runs = {}
        for d in (30, 10, 0):
            extra = ["--test-fault-proc-delay-ms", d] if d else []
            if func == "logo":
                runs[d] = run_logo(out, tag, impl, storage, "kill", pos, extra=extra, suffix=f"-teeth{d}")
            else:
                runs[d] = run_enc(out, tag, impl, storage, enc, "kill", pos, extra=extra, suffix=f"-teeth{d}")
        if any(x.get("err") for x in runs.values()):
            res.append(F.result(tid, "FAIL", why=" | ".join(describe(x) for x in runs.values())))
            continue
        gate = logo_gate if func == "logo" else enc_gate
        bad = {d: gate("kill", r) for d, r in runs.items()}
        lat10 = [b for b in bad[10] if "maks. vėlavimas" in b]
        p99_10 = [b for b in bad[10] if "lango p99" in b]
        ok = bool(bad[30]) and bool(lat10) and bool(p99_10) and not bad[0]
        show = lambda b: ("FAIL " + "; ".join(b)[:500]) if b else "PASS"  # noqa: E731
        res.append(F.result(tid, "PASS" if ok else "FAIL",
                            why=f"30 ms: {show(bad[30])} | 10 ms: {show(bad[10])} (vėlavimo ribos radinių {len(lat10)}, "
                                f"p99 radinių {len(p99_10)}) | 0 ms: {show(bad[0])}"))
    return res


TESTS = {"fiso_teeth": fiso_teeth, "fiso_kill": fiso_kill, "fiso_stall": fiso_stall, "fiso_composite": fiso_composite,
         "fiso_enc": fiso_enc, "fiso_enc_audio": fiso_enc_audio}
