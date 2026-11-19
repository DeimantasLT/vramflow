#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""regress.py testai (planas production-ready §3.3): gryna logika, GPU nereikia."""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import regress as R  # noqa: E402

BASE = {"logo-G": {"knee": 64, "p99_ms": 10.0, "vram_per_flow_mb": 56.0},
        "chain-G": {"knee": 11, "p99_ms": 19.0, "vram_per_flow_mb": 284.27}}


def cur(**over):
    c = json.loads(json.dumps(BASE))
    for tag, fields in over.items():
        c[tag.replace("_", "-")].update(fields)
    return c


class Compare(unittest.TestCase):
    def test_identical_pass(self):
        self.assertEqual(R.compare(BASE, cur()), [])

    def test_knee_minus_one_fails_with_function_name(self):
        bad = R.compare(BASE, cur(chain_G={"knee": 10}))
        self.assertEqual(len(bad), 1)
        self.assertIn("chain-G", bad[0])
        self.assertIn("kelias", bad[0])

    def test_knee_higher_pass(self):
        self.assertEqual(R.compare(BASE, cur(chain_G={"knee": 12})), [])

    def test_p99_plus_11_percent_fails(self):
        bad = R.compare(BASE, cur(logo_G={"p99_ms": 11.1}))
        self.assertEqual(len(bad), 1)
        self.assertIn("logo-G", bad[0])
        self.assertIn("p99", bad[0])

    def test_p99_plus_9_percent_pass(self):
        self.assertEqual(R.compare(BASE, cur(logo_G={"p99_ms": 10.9})), [])

    def test_p99_exactly_10_percent_pass(self):
        self.assertEqual(R.compare(BASE, cur(logo_G={"p99_ms": 11.0})), [])

    def test_vram_plus_6_percent_fails(self):
        bad = R.compare(BASE, cur(logo_G={"vram_per_flow_mb": 59.4}))
        self.assertEqual(len(bad), 1)
        self.assertIn("VRAM", bad[0])

    def test_vram_plus_4_percent_pass(self):
        self.assertEqual(R.compare(BASE, cur(logo_G={"vram_per_flow_mb": 58.2})), [])

    def test_missing_function_fails_unmeasured(self):
        c = cur()
        del c["chain-G"]
        bad = R.compare(BASE, c)
        self.assertEqual(len(bad), 1)
        self.assertIn("chain-G", bad[0])
        self.assertIn("nematuota", bad[0])

    def test_missing_metric_fails_unmeasured(self):
        bad = R.compare(BASE, cur(logo_G={"p99_ms": None}))
        self.assertEqual(len(bad), 1)
        self.assertIn("nematuota", bad[0])

    def test_extra_function_in_campaign_ignored(self):
        c = cur()
        c["dec-G"] = {"knee": 1, "p99_ms": 99.0, "vram_per_flow_mb": 999.0}
        self.assertEqual(R.compare(BASE, c), [])


def level(n, p99, ok=True, key="out_lateness", rep=0):
    return {"n": n, "pass": ok, "repeat": rep, "hist": {key: {"p50_ms": p99 / 2, "p99_ms": p99}, "proc_ms": {"p99_ms": 1.0}}}


class Campaign(unittest.TestCase):
    def write(self, d, tag, knee, mb, levels):
        func = tag.rsplit("-", 1)[0]
        pathlib.Path(d, f"knee-{tag}.json").write_text(json.dumps(
            {"func": func, "variant": "G", "knee": knee, "stopped_by": "fail", "vram": {"per_flow_mb": mb}}))
        pathlib.Path(d, f"capacity-{tag}.json").write_text(json.dumps(levels))

    def test_load_median_p99_of_passing_repeats_at_knee(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "logo-G", 64, 56.0, [level(32, 5.0), level(64, 10.0, rep=0), level(64, 12.0, rep=1),
                                               level(64, 11.0, rep=2), level(64, 50.0, ok=False, rep=3)])
            self.write(d, "chain-G", 11, 284.27, [level(11, 19.0, key="frame_lateness")])
            got = R.load_campaign(d)
        self.assertEqual(got, {"logo-G": {"knee": 64, "p99_ms": 11.0, "vram_per_flow_mb": 56.0},
                               "chain-G": {"knee": 11, "p99_ms": 19.0, "vram_per_flow_mb": 284.27}})

    def test_load_uses_knee_criterion_pass_tol(self):
        # kelio paieška vertina pass_tol (capacity.py --tol), ne griežtą pass — p99 imamas iš tų pačių pakartojimų
        lv = [level(22, 1.5, ok=False, rep=0) | {"pass_tol": True}, level(22, 1.6, ok=False, rep=1) | {"pass_tol": True},
              level(22, 9.0, ok=True, rep=2) | {"pass_tol": False}]
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "dec-G", 22, 182.18, lv)
            got = R.load_campaign(d)
        self.assertEqual(got["dec-G"]["p99_ms"], 1.55)

    def test_load_without_capacity_levels_p99_none(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "knee-dec-G.json").write_text(json.dumps({"knee": 22, "vram": None}))
            got = R.load_campaign(d)
        self.assertEqual(got, {"dec-G": {"knee": 22, "p99_ms": None, "vram_per_flow_mb": None}})

    def test_main_update_then_compare(self):
        with tempfile.TemporaryDirectory() as d:
            camp = pathlib.Path(d, "camp")
            camp.mkdir()
            self.write(camp, "logo-G", 64, 56.0, [level(64, 10.0)])
            base = pathlib.Path(d, "base.json")
            self.assertEqual(R.main([str(camp), "--baseline", str(base), "--update"]), 0)
            self.assertEqual(json.loads(base.read_text())["functions"], {"logo-G": {"knee": 64, "p99_ms": 10.0,
                                                                                   "vram_per_flow_mb": 56.0}})
            self.assertEqual(R.main([str(camp), "--baseline", str(base)]), 0)
            self.write(camp, "logo-G", 63, 56.0, [level(63, 10.0)])
            self.assertEqual(R.main([str(camp), "--baseline", str(base)]), 1)

    def test_main_missing_baseline_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(R.main([d, "--baseline", str(pathlib.Path(d, "nėra.json"))]), 1)

    def test_main_empty_campaign_fails(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d, "base.json")
            base.write_text(json.dumps({"schema": 1, "functions": BASE}))
            self.assertEqual(R.main([d, "--baseline", str(base)]), 1)


if __name__ == "__main__":
    unittest.main()
