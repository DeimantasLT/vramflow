#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""capacity_file.py testai (planas P8 → P4): gryna logika, GPU nereikia."""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import capacity_file as cf  # noqa: E402

GPU = {"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "uuid": "GPU-be7b6e9b", "driver": "596.49"}


def summ(func, knee, stopped_by, mb, r2=1.0, variant="G"):
    return {"func": func, "variant": variant, "knee": knee, "stopped_by": stopped_by, "repeats": 3,
            "vram": {"per_flow_mb": mb, "offset_mb": 300.0, "r2": r2}}


KNEES = {
    "logo": summ("logo", 64, "n_max", 56.0),
    "enc-nvenc-ll": summ("enc-nvenc-ll", 12, "nvenc_sessions", 126.21),
    "dec": summ("dec", 22, "fail", 182.18, 0.997),
    "chain": summ("chain", 11, "fail", 284.27),
}


class Build(unittest.TestCase):
    def test_engines_from_knees(self):
        d = cf.build(KNEES, GPU, "1080i50", "2026-11-17T02:17:24Z")
        self.assertEqual(d["engines"]["nvdec"], {"mpix_s": 1140.48, "bound": "throughput", "from": "dec"})
        enc = d["engines"]["nvenc"]
        self.assertEqual(enc["mpix_s"], 622.08)
        self.assertEqual(enc["bound"], "lower_bound")  # riba — sesijos, ne pralaidumas
        self.assertEqual(enc["sessions"], 12)

    def test_functions_vram_bytes(self):
        d = cf.build(KNEES, GPU, "1080i50", "2026-11-17T02:17:24Z")
        self.assertEqual(d["functions"]["logo"]["vram_per_flow_bytes"], 56 * 2**20)
        self.assertEqual(d["functions"]["logo"]["knee_bound"], "lower_bound")  # n_max — riba nepasiekta
        self.assertEqual(d["functions"]["dec"]["knee_bound"], "throughput")
        self.assertEqual(d["functions"]["dec"]["vram_r2"], 0.997)

    def test_nvenc_throughput_bound_without_session_limit(self):
        k = dict(KNEES, **{"enc-nvenc-ll": summ("enc-nvenc-ll", 9, "fail", 126.0)})
        d = cf.build(k, GPU, "1080i50", "t")
        self.assertEqual(d["engines"]["nvenc"]["bound"], "throughput")
        self.assertNotIn("sessions", d["engines"]["nvenc"])

    def test_only_g_variant_used(self):
        k = {"dec": summ("dec", 8, "fail", 10.0, variant="C")}
        with self.assertRaisesRegex(ValueError, "nvdec"):
            cf.build(k, GPU, "1080i50", "t")

    def test_unknown_format(self):
        with self.assertRaisesRegex(ValueError, "720p"):
            cf.build(KNEES, GPU, "720p", "t")


class Validate(unittest.TestCase):
    def test_roundtrip(self):
        d = cf.build(KNEES, GPU, "1080i50", "t")
        with tempfile.TemporaryDirectory() as td:
            p = cf.write(d, pathlib.Path(td))
            self.assertEqual(p.name, "capacity-GPU-be7b6e9b.json")
            self.assertEqual(cf.load(p), d)

    def test_missing_nvdec_names_field(self):
        d = cf.build(KNEES, GPU, "1080i50", "t")
        del d["engines"]["nvdec"]["mpix_s"]
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "c.json"
            p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, r"engines\.nvdec\.mpix_s"):
                cf.load(p)

    def test_wrong_schema_version(self):
        d = cf.build(KNEES, GPU, "1080i50", "t")
        d["schema"] = 99
        with self.assertRaisesRegex(ValueError, "schema"):
            cf.validate(d)


class FromDir(unittest.TestCase):
    def test_reads_knee_files(self):
        with tempfile.TemporaryDirectory() as td:
            for tag, s in KNEES.items():
                (pathlib.Path(td) / f"knee-{tag}-G.json").write_text(json.dumps(s))
            (pathlib.Path(td) / "knee-dec-C.json").write_text(json.dumps(summ("dec", 8, "fail", 1.0, variant="C")))
            k = cf.read_knees(pathlib.Path(td))
            self.assertEqual(sorted(k), ["chain", "dec", "enc-nvenc-ll", "logo"])
            self.assertEqual(k["dec"]["knee"], 22)


if __name__ == "__main__":
    unittest.main()
