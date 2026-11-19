#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""vramflow_run.config testai (OK-2 k. 3, T3): griežta schema, klaidos su lauko keliu. GPU nereikia.
Paleidimas: python3 tools/vramflow-run/test_config.py (ctest run_config)."""
import json
import os
import pathlib
import struct
import sys
import tempfile
import unittest
import zlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from vramflow_run import config  # noqa: E402
from vramflow_run.config import ConfigError  # noqa: E402
from vramflow_run.redact import redact_url  # noqa: E402


def minimal(**over):
    c = {"version": 1, "inputs": [{"url": "udp://127.0.0.1:9000"}], "process": {"logo": "builtin"},
         "outputs": [{"url": "udp://239.1.1.1:5000?ttl=4"}]}
    c.update(over)
    return c


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, obj, name="cfg.json"):
        p = pathlib.Path(self.tmp.name) / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return p

    def err(self, obj):
        with self.assertRaises(ConfigError) as cm:
            config.load(self.write(obj))
        return cm.exception


class Minimal(Base):
    def test_minimal_plan(self):
        p = config.load(self.write(minimal()))
        self.assertEqual(len(p.inputs), 1)
        self.assertEqual(p.inputs[0].url, "udp://127.0.0.1:9000")
        self.assertEqual(p.format, "1080i50")
        self.assertEqual(p.storage, "cuda-linear")
        self.assertEqual((p.restart.max, p.restart.window_s), (5, 60))
        self.assertEqual(p.input_wait_s, 20)
        self.assertEqual(p.process.kind, "logo")
        self.assertEqual(p.process.logo.file, "builtin")
        self.assertEqual((p.outputs[0].vbitrate, p.outputs[0].abitrate), (7_500_000, 192_000))

    def test_full_schema_example_from_brief(self):
        os.environ["SRT_IN_TEST"] = "0123456789abc"
        c = {"version": 1, "format": "1080i50", "storage": "cuda-linear",
             "inputs": [{"url": "srt://0.0.0.0:9000?mode=listener", "srt_passphrase_from": "env:SRT_IN_TEST"}],
             "process": {"logo": {"file": "builtin", "x": 1488, "y": 60}},  # brief'o x 1600 — tik ≤ 320 px logotipui
             "outputs": [{"url": "udp://239.1.1.1:5000?ttl=4", "vbitrate": 7500000, "abitrate": 192000}],
             "restart": {"max": 3, "window_s": 30}, "input_wait_s": 12}
        p = config.load(self.write(c))
        self.assertEqual(p.inputs[0].srt_passphrase_from, "env:SRT_IN_TEST")
        self.assertEqual((p.process.logo.x, p.process.logo.y), (1488, 60))
        self.assertEqual((p.restart.max, p.restart.window_s, p.input_wait_s), (3, 30, 12))

    def test_composite(self):
        c = minimal(process={"composite": {"layers": [{"source": "synthetic", "geom": "logo", "x": 1488, "y": 64, "opacity": 0.5}]}})
        p = config.load(self.write(c))
        self.assertEqual(p.process.kind, "composite")
        L = p.process.composite.layers[0]
        self.assertEqual((L.source, L.geom, L.x, L.y, L.opacity), ("synthetic", "logo", 1488, 64, 0.5))


class Errors(Base):
    def test_unknown_field(self):
        c = minimal(inputs=[{"urll": "udp://127.0.0.1:9000"}])
        e = self.err(c)
        self.assertEqual((e.path, e.why), ("inputs[0].urll", "nežinomas laukas"))
        self.assertEqual(str(e), "inputs[0].urll: nežinomas laukas")

    def test_unknown_top_level_and_nested(self):
        self.assertEqual(self.err(minimal(extra=1)).path, "extra")
        self.assertEqual(self.err(minimal(restart={"max": 5, "windows": 60})).path, "restart.windows")
        self.assertEqual(self.err(minimal(process={"logo": {"file": "builtin", "z": 1}})).path, "process.logo.z")

    def test_missing_outputs(self):
        c = minimal()
        del c["outputs"]
        e = self.err(c)
        self.assertEqual(e.path, "outputs")
        self.assertIn("privalomas", e.why)

    def test_count_mismatch(self):
        c = minimal(inputs=[{"url": "udp://127.0.0.1:9000"}, {"url": "udp://127.0.0.1:9001"}])
        self.assertEqual(str(self.err(c)), "outputs: 1, inputs: 2 — turi sutapti")

    def test_logo_and_composite(self):
        c = minimal(process={"logo": "builtin", "composite": {"layers": [{"source": "synthetic"}]}})
        e = self.err(c)
        self.assertEqual(e.path, "process")
        self.assertIn("lygiai vienas", e.why)

    def test_neither_logo_nor_composite(self):
        self.assertEqual(self.err(minimal(process={})).path, "process")

    def test_bad_json_line_col(self):
        e = self.err('{\n  "version": 1,\n  "inputs": [\n}')
        self.assertIn("eilutė 4", str(e))
        self.assertIn("stulpelis", str(e))

    def test_duplicate_key(self):
        e = self.err('{"version": 1, "version": 1}')
        self.assertEqual(e.path, "version")
        self.assertIn("pasikartoja", e.why)

    def test_passphrase_in_url(self):
        c = minimal(inputs=[{"url": "srt://0.0.0.0:9000?mode=listener&passphrase=Secret12345"}])
        e = self.err(c)
        self.assertEqual(e.path, "inputs[0].url")
        self.assertIn("naudokite srt_passphrase_from", e.why)
        self.assertNotIn("Secret12345", str(e))
        c = minimal(outputs=[{"url": "srt://127.0.0.1:9100?passphrase=Secret12345&mode=caller"}])
        e = self.err(c)
        self.assertEqual(e.path, "outputs[0].url")
        self.assertNotIn("Secret12345", str(e))

    def test_duplicate_endpoints(self):
        c = minimal(inputs=[{"url": "udp://127.0.0.1:9000"}, {"url": "udp://127.0.0.1:9000"}],
                    outputs=[{"url": "udp://127.0.0.1:5000"}, {"url": "udp://127.0.0.1:5001"}])
        e = self.err(c)
        self.assertIn("inputs[0].url", str(e))
        self.assertIn("inputs[1].url", str(e))
        c = minimal(inputs=[{"url": "udp://127.0.0.1:9000"}, {"url": "udp://127.0.0.1:9001"}],
                    outputs=[{"url": "udp://239.1.1.1:5000?ttl=4"}, {"url": "udp://239.1.1.1:5000?ttl=1"}])
        e = self.err(c)
        self.assertIn("outputs[0].url", str(e))
        self.assertIn("outputs[1].url", str(e))

    def test_too_many_inputs(self):
        ins = [{"url": f"udp://127.0.0.1:{9000 + k}"} for k in range(17)]
        outs = [{"url": f"udp://127.0.0.1:{5000 + k}"} for k in range(17)]
        e = self.err(minimal(inputs=ins, outputs=outs))
        self.assertEqual(e.path, "inputs")
        self.assertIn("≤ 16", e.why)

    def test_empty_inputs(self):
        self.assertEqual(self.err(minimal(inputs=[], outputs=[])).path, "inputs")

    def test_types(self):
        self.assertEqual(self.err(minimal(version=2)).path, "version")
        self.assertEqual(self.err(minimal(version=True)).path, "version")
        self.assertEqual(self.err(minimal(input_wait_s="20")).path, "input_wait_s")
        self.assertEqual(self.err(minimal(input_wait_s=0)).path, "input_wait_s")
        self.assertEqual(self.err(minimal(restart={"max": -1})).path, "restart.max")
        self.assertEqual(self.err(minimal(restart={"max": 1.5})).path, "restart.max")
        self.assertEqual(self.err(minimal(restart={"window_s": 0})).path, "restart.window_s")
        self.assertEqual(self.err(minimal(outputs=[{"url": "udp://127.0.0.1:5000", "vbitrate": "7M"}])).path, "outputs[0].vbitrate")
        self.assertEqual(self.err(minimal(outputs=[{"url": "udp://127.0.0.1:5000", "abitrate": 1}])).path, "outputs[0].abitrate")
        self.assertEqual(self.err(minimal(inputs="udp://127.0.0.1:9000")).path, "inputs")
        self.assertEqual(self.err(minimal(inputs=["udp://127.0.0.1:9000"])).path, "inputs[0]")
        self.assertEqual(self.err([1]).path, "")

    def test_format_and_storage(self):
        self.assertEqual(config.load(self.write(minimal(format="1080p25"))).format, "1080p25")
        self.assertEqual(config.load(self.write(minimal(storage="host"))).storage, "host")
        e = self.err(minimal(format="1080i25"))
        self.assertEqual(e.path, "format")
        self.assertIn("1080i50", e.why)  # palaikomų sąrašas
        self.assertEqual(self.err(minimal(format="720i50")).path, "format")
        self.assertEqual(self.err(minimal(storage="cuda")).path, "storage")

    def test_url_schemes(self):
        self.assertEqual(self.err(minimal(inputs=[{"url": "file:///x.ts"}])).path, "inputs[0].url")
        self.assertEqual(self.err(minimal(inputs=[{"url": "/x.ts"}])).path, "inputs[0].url")
        self.assertEqual(self.err(minimal(inputs=[{"url": "udp://127.0.0.1"}])).path, "inputs[0].url")
        self.assertEqual(self.err(minimal(inputs=[{"url": "udp://127.0.0.1:70000"}])).path, "inputs[0].url")
        self.assertEqual(self.err(minimal(outputs=[{"url": "rtp://127.0.0.1:5000"}])).path, "outputs[0].url")
        self.assertEqual(self.err(minimal(outputs=[{"url": "srt://127.0.0.1:5000?mode=rendezvous"}])).path, "outputs[0].url")
        p = config.load(self.write(minimal(inputs=[{"url": "rtp://127.0.0.1:9000"}])))
        self.assertEqual(p.inputs[0].url, "rtp://127.0.0.1:9000")
        p = config.load(self.write(minimal(inputs=[{"url": "https://h.example/live.m3u8"}])))
        self.assertEqual(p.inputs[0].url, "https://h.example/live.m3u8")

    def test_passphrase_from(self):
        e = self.err(minimal(inputs=[{"url": "udp://127.0.0.1:9000", "srt_passphrase_from": "env:X"}]))
        self.assertEqual(e.path, "inputs[0].srt_passphrase_from")
        self.assertIn("srt", e.why)
        e = self.err(minimal(inputs=[{"url": "srt://0.0.0.0:9000?mode=listener", "srt_passphrase_from": "Secret12345"}]))
        self.assertEqual(e.path, "inputs[0].srt_passphrase_from")
        self.assertNotIn("Secret12345", str(e))
        # vienas ingest / enc procesas — vienas slaptažodis visiems srt adresams
        c = minimal(inputs=[{"url": "srt://0.0.0.0:9000?mode=listener", "srt_passphrase_from": "env:A"},
                            {"url": "srt://0.0.0.0:9001?mode=listener"}],
                    outputs=[{"url": "udp://127.0.0.1:5000"}, {"url": "udp://127.0.0.1:5001"}])
        self.assertEqual(self.err(c).path, "inputs[1].srt_passphrase_from")

    def test_outputs_same_bitrates(self):
        c = minimal(inputs=[{"url": "udp://127.0.0.1:9000"}, {"url": "udp://127.0.0.1:9001"}],
                    outputs=[{"url": "udp://127.0.0.1:5000"}, {"url": "udp://127.0.0.1:5001", "vbitrate": 5_000_000}])
        self.assertEqual(self.err(c).path, "outputs[1].vbitrate")

    def test_logo_geometry(self):
        self.assertEqual(self.err(minimal(process={"logo": {"file": "builtin", "x": 1601}})).path, "process.logo.x")
        self.assertEqual(self.err(minimal(process={"logo": {"file": "builtin", "x": 1700}})).path, "process.logo.x")
        self.assertEqual(self.err(minimal(process={"logo": {"file": "builtin", "y": 900}})).path, "process.logo.y")
        self.assertEqual(self.err(minimal(process={"logo": "other"})).path, "process.logo")
        self.assertEqual(self.err(minimal(process={"logo": {"x": 0}})).path, "process.logo.file")
        self.assertEqual(self.err(minimal(process={"logo": {"file": "/nera/logo.png"}})).path, "process.logo.file")

    def test_logo_png(self):
        png = pathlib.Path(self.tmp.name) / "l.png"
        png.write_bytes(tiny_png(200, 100))
        p = config.load(self.write(minimal(process={"logo": {"file": str(png), "x": 1720, "y": 980}})))
        self.assertEqual(p.process.logo.file, str(png))
        e = self.err(minimal(process={"logo": {"file": str(png), "x": 1722, "y": 0}}))
        self.assertEqual(e.path, "process.logo.x")
        bad = pathlib.Path(self.tmp.name) / "bad.png"
        bad.write_bytes(b"not a png")
        self.assertEqual(self.err(minimal(process={"logo": {"file": str(bad)}})).path, "process.logo.file")

    def test_composite_errors(self):
        def comp(*layers):
            return minimal(process={"composite": {"layers": list(layers)}})
        self.assertEqual(self.err(comp()).path, "process.composite.layers")
        self.assertEqual(self.err(comp(*[{"source": "synthetic"}] * 9)).path, "process.composite.layers")
        self.assertEqual(self.err(comp({"source": "ograf"})).path, "process.composite.layers[0].source")
        self.assertEqual(self.err(comp({})).path, "process.composite.layers[0].source")
        self.assertEqual(self.err(comp({"source": "synthetic", "opacity": 1.5})).path, "process.composite.layers[0].opacity")
        self.assertEqual(self.err(comp({"source": "synthetic", "geom": "big"})).path, "process.composite.layers[0].geom")
        self.assertEqual(self.err(comp({"source": "synthetic", "geom": "full", "x": 2})).path, "process.composite.layers[0].x")
        self.assertEqual(self.err(comp({"source": "synthetic", "x": 3})).path, "process.composite.layers[0].x")
        self.assertEqual(self.err(comp({"source": "synthetic", "y": 3})).path, "process.composite.layers[0].y")  # interlaced
        self.assertEqual(self.err(comp({"source": "synthetic", "q": 3})).path, "process.composite.layers[0].q")

    def test_not_found(self):
        with self.assertRaises(ConfigError) as cm:
            config.load(pathlib.Path(self.tmp.name) / "nera.json")
        self.assertIn("nera.json", str(cm.exception))


class Secrets(Base):
    def test_check_secrets(self):
        c = minimal(inputs=[{"url": "srt://0.0.0.0:9000?mode=listener", "srt_passphrase_from": "env:VF_TEST_NO_SUCH_VAR"}])
        p = config.load(self.write(c))
        os.environ.pop("VF_TEST_NO_SUCH_VAR", None)
        with self.assertRaises(ConfigError) as cm:
            config.check_secrets(p)
        self.assertEqual(cm.exception.path, "inputs[0].srt_passphrase_from")
        os.environ["VF_TEST_NO_SUCH_VAR"] = "0123456789abc"
        config.check_secrets(p)  # be klaidos
        f = pathlib.Path(self.tmp.name) / "pass.txt"
        c = minimal(outputs=[{"url": "srt://127.0.0.1:9100?mode=caller", "srt_passphrase_from": f"file:{f}"}])
        p = config.load(self.write(c))
        with self.assertRaises(ConfigError) as cm:
            config.check_secrets(p)
        self.assertEqual(cm.exception.path, "outputs[0].srt_passphrase_from")
        f.write_text("0123456789abc\n")
        config.check_secrets(p)


class Examples(unittest.TestCase):
    def test_every_example_loads(self):
        files = sorted((REPO / "config" / "examples").glob("*.json"))
        self.assertGreaterEqual(len(files), 2)
        names = {f.name for f in files}
        self.assertTrue({"logo-srt.json", "logo-udp-4.json"} <= names, names)
        for f in files:
            with self.subTest(f=f.name):
                config.load(f)


class Redact(unittest.TestCase):
    # tie patys atvejai kaip tests/unit/test_url_policy.cpp (C++ redact_url) — elgsena turi sutapti
    def test_same_as_cpp(self):
        self.assertEqual(redact_url("srt://127.0.0.1:9000?mode=listener&passphrase=VerySecret123"),
                         "srt://127.0.0.1:9000?mode=***&passphrase=***")
        self.assertEqual(redact_url("http://user:pw@host:80/a/b.m3u8?token=xyz#frag"), "http://***@host:80/a/b.m3u8?token=***#***")
        self.assertEqual(redact_url("udp://127.0.0.1:20000"), "udp://127.0.0.1:20000")
        self.assertEqual(redact_url("https://h/p?justatoken"), "https://h/p?***")
        self.assertEqual(redact_url("/local/file.ts"), "/local/file.ts")
        self.assertEqual(redact_url("data:text/plain,secret"), "data:***")

    def test_edges(self):
        self.assertEqual(redact_url("udp://239.1.1.1:5000?ttl=4"), "udp://239.1.1.1:5000?ttl=***")
        self.assertEqual(redact_url("http://h/p#x"), "http://h/p#***")
        self.assertEqual(redact_url("http://h/p?a&b=1"), "http://h/p?***&b=***")
        self.assertEqual(redact_url("HTTP://u@h"), "HTTP://***@h")


def tiny_png(w, h):
    """Minimalus PNG (IHDR + IDAT + IEND) — tik matmenų skaitymui."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x00\x00\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    unittest.main()
