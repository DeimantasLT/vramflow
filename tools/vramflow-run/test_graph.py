#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""vramflow_run.graph testai (OK-2, T2): Plan → vaikų komandos (tvarka, rolės, slaptažodžiai tik env). GPU nereikia.
Paleidimas: python3 tools/vramflow-run/test_graph.py (ctest run_graph)."""
import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vramflow_run import config, graph  # noqa: E402

EXE = "/opt/vramflow/bin/vramflow-func"
DOM = "/dev/shm/vf-run-0123abcd"
SECRET = "Tikrai-Slaptas-123"


def plan(obj):
    with tempfile.TemporaryDirectory() as t:
        p = pathlib.Path(t) / "c.json"
        p.write_text(json.dumps(obj), encoding="utf-8")
        return config.load(p)


def cfg(n=1, process=None, **over):
    c = {"version": 1, "inputs": [{"url": f"udp://127.0.0.1:{9000 + k}"} for k in range(n)],
         "process": process or {"logo": "builtin"}, "outputs": [{"url": f"udp://239.1.1.1:{5000 + k}?ttl=4"} for k in range(n)]}
    c.update(over)
    return c


def opt(argv, name):
    """Visos parinkties reikšmės (kartojamai parinkčiai — kelios)."""
    return [argv[i + 1] for i, a in enumerate(argv) if a == name]


def flow_id(role, n):
    return "4d584c47-%04d-4000-8000-%012x" % (role, n)


class Logo(unittest.TestCase):
    def test_order_and_roles(self):
        procs = graph.commands(plan(cfg()), DOM, EXE)
        self.assertEqual([p.name for p in procs], ["ingest", "logo", "enc"])
        ing, logo, enc = procs
        for p, sub in zip(procs, ("ingest", "logo", "enc")):
            self.assertEqual(p.argv[:2], [EXE, sub])
            self.assertEqual(opt(p.argv, "--domain"), [DOM])
            self.assertEqual(opt(p.argv, "--flows"), ["1"])
            self.assertEqual(opt(p.argv, "--storage"), ["cuda-linear"])
            self.assertEqual(opt(p.argv, "--format"), ["1080i50"])
        self.assertEqual(opt(ing.argv, "--flow-base"), ["0"])
        self.assertEqual(opt(ing.argv, "--input"), ["udp://127.0.0.1:9000"])
        self.assertEqual(opt(ing.argv, "--input-wait-s"), ["20"])
        self.assertEqual(opt(logo.argv, "--in-role"), ["7"])
        self.assertEqual(opt(logo.argv, "--logo"), ["builtin"])
        self.assertEqual(opt(enc.argv, "--in-role"), ["2"])
        self.assertEqual(opt(enc.argv, "--audio-role"), ["8"])
        self.assertEqual(opt(enc.argv, "--audio"), ["1"])
        self.assertEqual(opt(enc.argv, "--output"), ["udp://239.1.1.1:5000?ttl=4"])
        self.assertEqual(opt(enc.argv, "--vbitrate"), ["7500000"])
        self.assertEqual(opt(enc.argv, "--abitrate"), ["192000"])
        self.assertEqual(opt(enc.argv, "--muxrate"), ["8500000"])  # kaip enc numatytoji 7,5 M + 192 k
        # pasiruošimo žymė: ingest — video ir garso flow'ai, logo — savo išėjimas, enc — be flow'ų
        self.assertEqual(set(ing.flows), {flow_id(7, 0), flow_id(8, 0)})
        self.assertEqual(set(logo.flows), {flow_id(2, 0)})
        self.assertEqual(enc.flows, ())
        # OK-2.6: šaltinio spraga — esama rašytojų politika „freeze“ (paskutinis kadras ≤ --hold-ms, tada juoda), ne INVALID
        for p in (ing, logo):
            self.assertEqual(opt(p.argv, "--on-loss"), ["freeze"], p.name)
        self.assertEqual(opt(enc.argv, "--on-loss"), [])
        # paleidimo vartai: logo — kai yra ingest video (logo briaunos žymas skaito paleidimo metu, ne prisijungęs);
        # enc pats laukia savo įėjimų ir žymas skaito po prisijungimo — paleidžiamas iškart (CUDA / NVENC paruošimas
        # sutampa su ingest zondavimu)
        self.assertEqual(ing.wait_for, ())
        self.assertEqual(logo.wait_for, (flow_id(7, 0),))
        self.assertEqual(enc.wait_for, ())

    def test_logo_file_and_position(self):
        procs = graph.commands(plan(cfg(process={"logo": {"file": "builtin", "x": 1488, "y": 60}})), DOM, EXE)
        logo = procs[1]
        self.assertEqual(opt(logo.argv, "--logo-x"), ["1488"])
        self.assertEqual(opt(logo.argv, "--logo-y"), ["60"])

    def test_four_inputs(self):
        procs = graph.commands(plan(cfg(4)), DOM, EXE)
        self.assertEqual([p.name for p in procs], ["ingest", "logo", "enc"])
        self.assertEqual(len(opt(procs[0].argv, "--input")), 4)
        for p in procs:
            self.assertEqual(opt(p.argv, "--flows"), ["4"], p.name)
        self.assertEqual(opt(procs[2].argv, "--output"), [f"udp://239.1.1.1:{5000 + k}?ttl=4" for k in range(4)])
        self.assertEqual(len(procs[1].flows), 4)

    def test_secret_only_in_env(self):
        c = cfg(1)
        c["inputs"] = [{"url": "srt://0.0.0.0:9000?mode=listener", "srt_passphrase_from": "env:SRT_IN"}]
        c["outputs"] = [{"url": "srt://127.0.0.1:9100?mode=caller", "srt_passphrase_from": "env:SRT_OUT"}]
        procs = graph.commands(plan(c), DOM, EXE, base_env={"SRT_IN": SECRET, "SRT_OUT": SECRET + "x", "PATH": "/usr/bin"})
        for p in procs:
            self.assertFalse(any(SECRET in a for a in p.argv), p.argv)
            self.assertEqual(p.env.get("SRT_IN"), SECRET)
        self.assertEqual(opt(procs[0].argv, "--srt-passphrase-from"), ["env:SRT_IN"])
        self.assertEqual(opt(procs[2].argv, "--srt-passphrase-from"), ["env:SRT_OUT"])
        self.assertEqual(opt(procs[1].argv, "--srt-passphrase-from"), [])

    def test_no_passphrase_option_without_srt(self):
        procs = graph.commands(plan(cfg()), DOM, EXE)
        self.assertTrue(all("--srt-passphrase-from" not in p.argv for p in procs))

    def test_out_dir(self):
        procs = graph.commands(plan(cfg()), DOM, EXE, out_dir="/var/log/vr")
        for p in procs:
            self.assertEqual(opt(p.argv, "--out"), [f"/var/log/vr/{p.name}"])
            self.assertEqual(opt(p.argv, "--stats-file"), [f"/var/log/vr/{p.name}/stats.jsonl"])

    def test_bitrates_and_restart_settings(self):
        c = cfg(1, input_wait_s=7.5)
        c["outputs"][0].update(vbitrate=15_000_000, abitrate=256_000)
        procs = graph.commands(plan(c), DOM, EXE)
        self.assertEqual(opt(procs[0].argv, "--input-wait-s"), ["7.5"])
        self.assertEqual(opt(procs[2].argv, "--vbitrate"), ["15000000"])
        self.assertEqual(opt(procs[2].argv, "--abitrate"), ["256000"])
        self.assertEqual(opt(procs[2].argv, "--muxrate"), ["16800000"])  # ceil((15 M + 256 k) · 1,1, 100 k)


class Composite(unittest.TestCase):
    def test_composite_chain(self):
        layers = [{"source": "synthetic", "geom": "logo", "x": 1488, "y": 64},
                  {"source": "synthetic", "geom": "full", "opacity": 0.25}]
        procs = graph.commands(plan(cfg(2, process={"composite": {"layers": layers}})), DOM, EXE)
        self.assertEqual([p.name for p in procs], ["ingest", "gfx", "composite", "enc"])
        _, gfx, comp, enc = procs
        self.assertEqual(gfx.argv[1], "gfxgen")
        self.assertEqual(opt(gfx.argv, "--layers"), ["2"])
        self.assertEqual(opt(gfx.argv, "--layer-geom"), ["logo,full"])
        self.assertEqual(opt(gfx.argv, "--mode"), ["pace"])
        self.assertEqual(comp.argv[1], "composite")
        self.assertEqual(opt(comp.argv, "--in-role"), ["7"])
        self.assertEqual(opt(comp.argv, "--base"), ["gen"])
        self.assertEqual(opt(comp.argv, "--layers"), ["2"])
        self.assertEqual(opt(comp.argv, "--layer-xy"), ["1488,64", "0,0"])
        self.assertEqual(opt(comp.argv, "--opacity"), ["1", "0.25"])
        self.assertEqual(opt(comp.argv, "--out-format"), ["v210"])
        self.assertEqual(opt(enc.argv, "--in-role"), ["6"])
        self.assertEqual(opt(enc.argv, "--audio-role"), ["8"])
        self.assertEqual(set(gfx.flows), {flow_id(5, n * 8 + k) for n in range(2) for k in range(2)})
        self.assertEqual(set(comp.flows), {flow_id(6, 0), flow_id(6, 1)})
        for p in procs:
            self.assertEqual(opt(p.argv, "--flows"), ["2"], p.name)
        # composite sluoksnių žymas skaito paleidimo metu (sluoksnių nelaukia) — paleidžiamas, kai yra ir bazė, ir sluoksniai
        self.assertEqual(set(comp.wait_for), {flow_id(7, 0), flow_id(7, 1)} | set(gfx.flows))
        self.assertEqual(gfx.wait_for, ())
        self.assertEqual(opt(comp.argv, "--on-loss"), ["freeze"])
        self.assertEqual(opt(procs[0].argv, "--on-loss"), ["freeze"])
        self.assertEqual(enc.wait_for, ())


class Redacted(unittest.TestCase):
    def test_display_argv_redacts_urls(self):
        c = cfg(1)
        c["inputs"] = [{"url": "srt://0.0.0.0:9000?mode=listener&streamid=s3cr3tid"}]
        procs = graph.commands(plan(c), DOM, EXE)
        shown = graph.display(procs[0])
        self.assertIn("srt://0.0.0.0:9000?mode=***&streamid=***", shown)
        self.assertNotIn("s3cr3tid", shown)


if __name__ == "__main__":
    unittest.main()
