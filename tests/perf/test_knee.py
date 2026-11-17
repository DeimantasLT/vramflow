#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""knee.py testai (planas P8): gryna logika, GPU nereikia. Paleidimas: python3 tests/perf/test_knee.py"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import knee  # noqa: E402


def levels(result):
    return [n for n, _ in result.tried]


class FindKnee(unittest.TestCase):
    def test_exact_threshold(self):
        # knee: tikslus slenkstis — pakopos 1, 2, 4, 8, 16, 12, 14, 13
        r = knee.find_knee(lambda n: n <= 13, n_max=64)
        self.assertEqual(r.knee, 13)
        self.assertEqual(levels(r), [1, 2, 4, 8, 16, 12, 14, 13])
        self.assertEqual(r.stopped_by, "fail")

    def test_unstable_level_counts_as_fail(self):
        # knee: nestabili pakopa — N = 10 praeina 2 iš 3 → FAIL, kelias 9
        calls = {}

        def run(n):
            calls[n] = calls.get(n, 0) + 1
            if n == 10:
                return calls[n] != 3
            return n < 10

        r = knee.find_knee(run, n_max=64)
        self.assertEqual(r.knee, 9)
        self.assertEqual(dict(r.tried)[10], [True, True, False])
        self.assertEqual(calls[10], 3)

    def test_limit_not_reached(self):
        # knee: riba nepasiekta — visada PASS, n_max = 32
        r = knee.find_knee(lambda n: True, n_max=32)
        self.assertEqual(r.knee, 32)
        self.assertEqual(r.stopped_by, "n_max")
        self.assertEqual(levels(r), [1, 2, 4, 8, 16, 32])

    def test_n_max_not_power_of_two(self):
        r = knee.find_knee(lambda n: True, n_max=20)
        self.assertEqual(levels(r), [1, 2, 4, 8, 16, 20])
        self.assertEqual(r.knee, 20)

    def test_first_level_fails(self):
        # knee: N = 1 krenta → 0, ne išimtis
        r = knee.find_knee(lambda n: False, n_max=64)
        self.assertEqual(r.knee, 0)
        self.assertEqual(levels(r), [1])
        self.assertEqual(r.stopped_by, "fail")

    def test_failure_stops_repeats_early(self):
        r = knee.find_knee(lambda n: n <= 1, n_max=4)
        self.assertEqual(dict(r.tried)[2], [False])
        self.assertEqual(dict(r.tried)[1], [True, True, True])

    def test_vram_guard_caps_search(self):
        # VRAM sargas: N ≥ 6 nebeleidžiamas, nors pakopa praeitų
        r = knee.find_knee(lambda n: True, n_max=64, guard=lambda n: n < 6)
        self.assertEqual(r.knee, 5)
        self.assertEqual(r.stopped_by, "vram_guard")
        self.assertNotIn(8, levels(r))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            knee.find_knee(lambda n: True, n_max=0)
        with self.assertRaises(ValueError):
            knee.find_knee(lambda n: True, n_max=4, repeats=0)


class VramPerFlow(unittest.TestCase):
    def test_linear_points(self):
        slope, intercept, r2 = knee.vram_per_flow([(1, 310), (2, 420), (4, 640), (8, 1080)])
        self.assertAlmostEqual(slope, 110.0, delta=0.1)
        self.assertAlmostEqual(intercept, 200.0, delta=0.1)
        self.assertGreaterEqual(r2, 0.999)

    def test_noisy_points_lower_r2(self):
        _, _, r2 = knee.vram_per_flow([(1, 300), (2, 450), (4, 600), (8, 1100)])
        self.assertLess(r2, 0.999)
        self.assertGreater(r2, 0.9)

    def test_needs_two_distinct_n(self):
        with self.assertRaises(ValueError):
            knee.vram_per_flow([(4, 640)])
        with self.assertRaises(ValueError):
            knee.vram_per_flow([(4, 640), (4, 650)])


if __name__ == "__main__":
    unittest.main()
