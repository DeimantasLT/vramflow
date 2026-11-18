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
    print("PASS" if not bad else f"FAIL: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
