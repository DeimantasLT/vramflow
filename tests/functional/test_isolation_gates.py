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
    # keepwarm-*.err → [(pradžia, pabaiga)] ns
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "keepwarm-1.err").write_text("[keepwarm] GPU įvykis: TAI 1795009526012345678 trukmė 16.084 ms\nkita\n")
        ev = I.gpu_events(d)
    if ev != [(1795009526012345678, 1795009526012345678 + 16_084_000)]:
        bad.append(f"gpu_events: {ev}")
    return bad


if __name__ == "__main__":
    sys.exit(main())
