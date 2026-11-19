#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""isolation.py vartų logika su sintetiniais duomenimis (planas production-ready §3.1; be GPU ir be procesų).

fiso_stall paveikto flow'o spragų commit'ai: vertinami tik GEDIMO LANGO [gedimas − LEAD_S, T(paskutinis)] commit'ai
(gap_late_recent [T(i) ns, µs]). Paleidimo GPU įvykio spraga (visi flow'ai, ~11 s prieš gedimą; fiso_stall2 r10) į vartus
nepatenka; gedimo lango vėlavimas — FAIL kaip anksčiau. Ribos nekeičiamos.
  python3 tests/functional/test_isolation_gates.py   → exit 0 PASS, 1 FAIL
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import isolation as I  # noqa: E402

S = 1_000_000_000
FAULT = 1795003221 * S  # SIGSTOP
END = FAULT + 17 * S
CUT = I.DEADLINE_US  # expect_us 0


def rec(recent, n=None):
    """Minimalus logo įrašas: tik spragų commit'ų vartams reikalingi laukai."""
    us = [u for _, u in recent or []]
    fa = {"gap_late_commits": len(us) if n is None else n, "gap_late_commit_min_us": min(us) if us else None,
          "gap_late_commit_max_us": max(us) if us else None}
    r = {"fault_tai": FAULT, "end_tai": END, "fn": {"affected": fa}, "expect_us": 0}
    if recent is not None:
        r["gap_recent"] = recent
    return r


def main():
    bad = []
    ok3 = [(FAULT + 20_000_000 * k, CUT + 700 + 10 * k) for k in range(3)]
    startup = [(FAULT - 11 * S, 30_300)]  # paleidimo GPU įvykis — už gedimo lango
    cases = [
        ("3 gedimo lango spragos ties riba", ok3, True),
        ("paleidimo vėli spraga + 3 gedimo lango (r10)", startup + ok3, True),
        ("gedimo lange vėli spraga", ok3 + [(FAULT + 80_000_000, 30_300)], False),
        ("gedimo lange tik 2", ok3[:2], False),
        ("tik paleidimo spraga", startup, False),
        ("gedimo lange per anksti", [(t, CUT - I.DEADLINE_TOL_US - 1) for t, _ in ok3], False),
    ]
    for name, recent, want in cases:
        lo, hi, n = I.gap_commits_in_window(rec(recent))
        got = n >= 3 and lo is not None and lo >= CUT - I.DEADLINE_TOL_US and hi <= CUT + I.DEADLINE_WAKE_US
        print(f"{'ok ' if got == want else 'BAD'} {name}: lo={lo} hi={hi} n={n} → {'PASS' if got else 'FAIL'} (laukta {want})")
        if got != want:
            bad.append(name)
    # be gap_recent (sena santrauka) — visas gyvenimas, kaip anksčiau
    lo, hi, n = I.gap_commits_in_window(rec(None) | {"fn": {"affected": {"gap_late_commits": 4, "gap_late_commit_min_us": 1,
                                                                          "gap_late_commit_max_us": 2}}})
    if (lo, hi, n) != (1, 2, 4):
        bad.append(f"be gap_recent: {(lo, hi, n)}")
    bad += platform_cases()
    bad += common_rise_cases()
    bad += late_delivery_cases()
    print("PASS" if not bad else f"FAIL: {bad}")
    return 1 if bad else 0


def platform_cases():
    """Sprendimas (b), 2026-11-17: vėlavimo / kadrų pažeidimas → INFO „platformos įvykis“ tik kai KIEKVIENĄ jo laiko langą dengia
    keepwarm GPU įvykis (TAI pradžia + trukmė) IR sveikų flow'ų pažeidimai apima visus sveikus flow'us; kitaip — FAIL."""
    import tempfile
    bad = []
    s = FAULT // S + 1  # gedimo lango sekundė
    healthy = [0, 1, 2]
    ev_hit = [(s * S + 990_000_000, s * S + 1_006_000_000)]  # 16 ms, persidengia su s sek. vėlavimu (commit'as kitą sekundę)
    ev_miss = [(s * S - 2 * S, s * S - 2 * S + 16_000_000)]

    all_lat = [I.Timed(f"sveiko #{n} vėlavimas", [(s * S, (s + 1) * S + 33_200_000)], flow=n) for n in healthy]
    cases = [
        ("visi sveiki, liudytas", all_lat, ev_hit, 0, 3),
        ("visi sveiki, įvykis kitur", all_lat, ev_miss, 3, 0),
        ("ne visi sveiki (#2 nėra)", all_lat[:2], ev_hit, 2, 0),
        ("be liudininkų (C — keepwarm neveikia)", all_lat, [], 3, 0),
        ("liudytas + p99 (be laiko)", all_lat + ["sveiko #0 lango p99"], ev_hit, 4, 0),
        ("tuščias langų sąrašas (verify be funkcijos praradimų)", all_lat + [I.Timed("verify #0", [], flow=0)], ev_hit, 4, 0),
        ("paveikto spragos commit'as, liudytas", [I.Timed("spragų commit'ai", [(s * S + 995_000_000, s * S + 1_001_000_000)])],
         ev_hit, 0, 1),
    ]
    for name, viol, ev, want_fail, want_info in cases:
        fail, info = I.platform_split(viol, healthy, ev)
        ok = (len(fail), len(info)) == (want_fail, want_info)
        print(f"{'ok ' if ok else 'BAD'} (b) {name}: FAIL {len(fail)}, INFO {len(info)} (laukta {want_fail}, {want_info})")
        if not ok:
            bad.append(f"(b) {name}")
    # max_lateness_bad pažeidimas — Timed su langu [s, s + 1 s + v]
    pre = {s - k: 10.0 + 0.1 * k for k in range(4, 9)}  # kontrolinis langas: 5 s iki gedimo − PRE_GAP_S
    r = {"lat_s": {n: pre | {s: 33.2} for n in healthy}, "fault_tai": FAULT, "end_tai": END,
         "start_tai": FAULT - 12 * S, "ready_tai": FAULT - 12 * S}
    got = I.max_lateness_bad(r, "out_lateness")
    if not got or not all(isinstance(b, I.Timed) and b.wins == [(s * S, (s + 1) * S + 33_200_000)] for b in got):
        bad.append(f"max_lateness_bad → Timed: {[(str(b), getattr(b, 'wins', None)) for b in got]}")
    # fsplit (targeted.py): nepaaiškinti praradimai → langai tik toms sekundėms, kurių aukštupis nepaaiškina
    import targeted as T
    i_s = s * S // I.FIELD_NS + 5  # aukštupio vėlus indeksas sekundėje s
    w = (s * S, (s + 1) * S + I.LOSS_SLACK_NS)
    for name, f, up, want in [("2 prarasti, aukštupis 1", {"loss_s": [[s, 2]]}, {"late_index": [i_s]}, [w]),
                              ("2 prarasti, aukštupis 2", {"loss_s": [[s, 2]]}, {"late_index": [i_s, i_s + 1]}, []),
                              ("kita sekundė", {"loss_s": [[s, 1], [s + 2, 1]]}, {"late_index": [i_s]},
                               [((s + 2) * S, (s + 3) * S + I.LOSS_SLACK_NS)])]:
        got = T.unexplained_wins(f, up)
        if got != want:
            bad.append(f"unexplained_wins {name}: {got} ≠ {want}")
    # verdict su vienu sveiku flow'u (fsplit): liudytas → INFO
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "keepwarm-1.err").write_text(f"[keepwarm] GPU įvykis: TAI {s * S + 200_000_000} trukmė 12.5 ms\n")
        st, _ = I.verdict([I.Timed("sveiko #3 nepaaiškinti", [w], flow=3)], [3], d)
        st2, _ = I.verdict([I.Timed("sveiko #3 nepaaiškinti", [w], flow=3), "held_ms tuščia"], [3], d)
    if (st, st2) != ("INFO", "FAIL"):
        bad.append(f"verdict fsplit: {st}, {st2}")
    # (b) išplėsta 2026-11-18: host'o pauzė (hostwatch-*.err; ir C variantui, be keepwarm) — toks pat liudininkas
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "hostwatch-0.err").write_text(f"[hostwatch] host įvykis: TAI {s * S + 200_000_000} trukmė 91.100 ms\n"
                                                      "[hostwatch] baigta: 10 pabudimų\n")
        st3, pre3 = I.verdict([I.Timed("sveiko #3 nepaaiškinti", [w], flow=3)], [3], d)
        ev = I.platform_events(d)
    if st3 != "INFO" or "host" not in pre3 or ev != [(s * S + 200_000_000, s * S + 200_000_000 + 91_100_000)]:
        bad.append(f"verdict host: {st3}, {pre3[:80]}, {ev}")
    # keepwarm-*.err → [(pradžia, pabaiga)] ns
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "keepwarm-1.err").write_text("[keepwarm] GPU įvykis: TAI 1795009526012345678 trukmė 16.084 ms\nkita\n")
        ev = I.gpu_events(d)
    if ev != [(1795009526012345678, 1795009526012345678 + 16_084_000)]:
        bad.append(f"gpu_events: {ev}")
    return bad


def common_rise_cases():
    """(b) išplėsta 2026-11-19 (autorius: „Išplėsti (b)“; fiso_stall5 r09, r14). Ribos nekeičiamos. „Bendras pakilimas“
    sekundę s — VISŲ sveikų flow'ų vėlavimas > atskaita + Wr + ro. (1) Pažeidimo langas apima visą ištisinę bendro pakilimo
    sekundžių seką (liudininkas bet kurioje jų sekundėje); (2) bendro pakilimo pažeidimas — INFO, net jei ribą peržengė ne
    visi sveiki. Be bendro pakilimo — kaip anksčiau (visi sveiki turi pažeisti, langas — tik sekundė s)."""
    bad = []
    s = FAULT // S + 1
    healthy = [0, 1, 2]
    pre = {s - k: 10.0 + 0.1 * k for k in range(4, 9)}  # atskaita 10.8 → riba 17.3, bendro pakilimo slenkstis 16.3
    base = {"fault_tai": FAULT, "end_tai": END, "start_tai": FAULT - 12 * S, "ready_tai": FAULT - 12 * S}

    def lat(per_flow):
        return base | {"lat_s": {n: pre | per_flow[n] for n in healthy}}

    # r09: visi sveiki s 23.8 ms (pažeidimas), s+1 17.0 ms (pakilę, po riba); host liudininkas tik s+1 sekundėje
    r09 = lat({n: {s: 23.8, s + 1: 17.0, s + 2: 10.9} for n in healthy})
    got = I.max_lateness_bad(r09, "out_lateness")
    want_w = [(s * S, (s + 2) * S + 23_800_000)]
    if len(got) != 3 or not all(isinstance(b, I.Timed) and b.wins == want_w and getattr(b, "common", False) for b in got):
        bad.append(f"r09 langas: {[(str(b), getattr(b, 'wins', None), getattr(b, 'common', None)) for b in got]}")
    ev_next = [((s + 1) * S + 129_000_000, (s + 1) * S + 138_000_000)]
    ev_far = [((s + 3) * S + 129_000_000, (s + 3) * S + 138_000_000)]
    # r14: visi sveiki pakilę s (17.0), ribą peržengė tik #2 (33.8); GPU liudininkas s sekundėje
    r14 = lat({0: {s: 17.0}, 1: {s: 17.0}, 2: {s: 33.8}})
    one = I.max_lateness_bad(r14, "out_lateness")
    # #2 vienas pakilęs (kiti 11.0 — ne bendras pakilimas): kaip anksčiau
    solo = I.max_lateness_bad(lat({0: {s: 11.0}, 1: {s: 11.0}, 2: {s: 33.8}}), "out_lateness")
    ev_s = [(s * S + 180_000_000, s * S + 196_000_000)]
    # vienas flow'as be duomenų sekundę s — bendro pakilimo nėra
    gap = I.max_lateness_bad(lat({0: {s + 1: 10.9}, 1: {s: 17.0}, 2: {s: 33.8}}), "out_lateness")
    cases = [
        ("r09: liudininkas kitoje pakilimo sekundėje", got, ev_next, 0, 3),
        ("r09: liudininkas už pakilimo sekos", got, ev_far, 3, 0),
        ("r14: bendras pakilimas, pažeidė 1 iš 3, liudytas", one, ev_s, 0, 1),
        ("r14: bendras pakilimas, neliudytas", one, ev_far, 1, 0),
        ("vienas pakilęs (ne bendras), liudytas", solo, ev_s, 1, 0),
        ("vienas flow'as be duomenų, liudytas", gap, ev_s, 1, 0),
        ("bendras pakilimas + ne laiko pažeidimas", one + ["sveiko #0 lango p99"], ev_s, 2, 0),
    ]
    for name, viol, ev, want_fail, want_info in cases:
        fail, info = I.platform_split(viol, healthy, ev)
        ok = (len(fail), len(info)) == (want_fail, want_info)
        print(f"{'ok ' if ok else 'BAD'} (b+) {name}: FAIL {len(fail)}, INFO {len(info)} (laukta {want_fail}, {want_info})")
        if not ok:
            bad.append(f"(b+) {name}")
    if [b.wins for b in solo] != [[(s * S, (s + 1) * S + 33_800_000)]] or any(getattr(b, "common", False) for b in solo):
        bad.append(f"ne bendras: langas / common {[(b.wins, getattr(b, 'common', None)) for b in solo]}")
    return bad


def late_delivery_cases():
    """fsplit r01 (2026-11-18): per 91 ms sustojimą logo laukas jau Ready, commit'as — po verify skaitymo (lag 4 laukai = 80 ms):
    verify „missing“, funkcija — fields_ok. Toks indeksas apskaitomas tik pagal funkcijos eilutę „vėlus commit“ (Ready po D);
    sveikam flow'ui tai laiko pažeidimas (Timed — sprendimas (b)), ne paprasta apskaitos eilutė."""
    import tempfile
    import targeted as T
    bad = []
    i0 = 89751406305
    ferr = "".join(f"[logo] praradimas: flow 1, indeksas {i0 + k} (late_skips)\n" for k in (0, 1, 2, 5)) + \
        f"[logo] vėlus commit: flow 1, indeksas {i0 + 3}, 91059 µs\n[logo] vėlus commit: flow 1, indeksas {i0 + 9}, 31000 µs\n"
    verr = f"[verify] trūkumas: flow 1, indeksas {i0} (missing)\n" + \
        "".join(f"[verify] trūkumas: flow 1, indeksas {i0 + k} (invalid_skipped)\n" for k in (1, 2, 5)) + \
        f"[verify] trūkumas: flow 1, indeksas {i0 + 3} (missing)\n[verify] trūkumas: flow 1, indeksas {i0 + 4} (missing)\n"
    got = T.late_hits(verr, ferr)
    if got != {1: {i0 + 3: 91059}}:  # i0 — praradimas; i0+4 — be vėlaus commit'o eilutės; i0+9 — verify matė
        bad.append(f"late_hits: {got}")
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "logo.err").write_text(ferr)
        pathlib.Path(d, "verify.err").write_text(verr.replace(f"indeksas {i0 + 4} (missing)", f"indeksas {i0 + 4} (invalid_skipped)"))
        fl = {"late_skips": 4, "loss_s": [[i0 * I.FIELD_NS // S, 4]], "gap_upstream": 1}
        vv = {"samples": 700, "missing": 2, "invalid": 4}  # 6 = 4 praradimai + 1 aukštupio spraga + 1 vėlus commit
        mk = lambda: {"fn": {"flows": [{}, fl], "hist": {"held_ms": {"count": 3}}}, "verify": {"flows": [{"samples": 700}, vv]},
                      "special": {0}, "upstream": {"late_index": [i0 + k for k in (0, 1, 2, 5)]}, "dir": pathlib.Path(d)}
        orig_n, T.N = T.N, 2
        try:
            got, _ = T.check_split(mk(), "logo")
            vv["missing"] = 3  # dar vienas nepaaiškintas — paprasta apskaitos eilutė (FAIL)
            got2, _ = T.check_split(mk(), "logo")
        finally:
            T.N = orig_n
    t = i0 * I.FIELD_NS + I.FIELD_NS * 3
    if [(str(b)[:20], getattr(b, "wins", None), getattr(b, "flow", None)) for b in got] != \
            [("sveiko #1 vėlus comm", [(t, t + 91_059_000)], 1)]:
        bad.append(f"check_split vėlus commit: {[(str(b), getattr(b, 'wins', None)) for b in got]}")
    if not any(not isinstance(b, I.Timed) and "verify missing + invalid 7" in b for b in got2):
        bad.append(f"check_split +1 nepaaiškintas: {[str(b) for b in got2]}")
    return bad


if __name__ == "__main__":
    sys.exit(main())
