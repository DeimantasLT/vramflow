#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Sąveika su MXL įrankiais (production-grade §P5 „P5b“, K4; rezultatai docs/rezultatai/2026-11-13-saveika-su-mxl-irankiais.md).

Kviečiami iš functional.py --only <vardas>. Kiekvienas testas SKIP (ne PASS), kai trūksta įrankio arba Xvfb / xwd.

interop_info            — mxl-info -f kiekvienam VRAMFlow flow'ui: host ir cuda-linear, kiekvienas palaikomas formatas
                          (1080i50, 1080p25, 1080p50, 720p50, 1080i59.94), garsas, v210a (gfxgen): exit 0, „Format: Video“ /
                          „Audio“, „Active: true“, head > 0; aprašas su VRAMFlow žymomis MXL priimtas (flow sukurtas).
interop_sink_host       — mxl-gst-sink (Xvfb) skaito gen --storage host 1080p50 ir logo išėjimą: vaizdas juda; gen nužudomas
                          3 s ir paleidžiamas iš naujo, logo --on-loss invalid | freeze | black: sink neužstringa (procesas gyvas,
                          po gedimo vaizdas vėl juda), gedimo lange — invalid: sustingęs paskutinis kadras (sink INVALID praleidžia
                          ir kartoja paskutinį gerą), freeze (--hold-ms 1000): ~1 s sustingęs, tada juoda, black: juoda; --start immediate (logo be gen 4 s) — po gen
                          starto vaizdas juda.
interop_sink_cuda       — mxl-gst-cuda-sink — tas pats cuda-linear v210 flow'ams (G).
interop_foreign_writer  — mxl-gst-testsrc (svetimas rašytojas, be VRAMFlow žymų) → logo --input-expect-ms → mxl-gst-sink: logo
                          0 praradimų, sink vaizdas juda.

Vaizdo būsena matuojama Xvfb ekrano kopijomis (xwd, 5 Hz): `move` — kadras pasikeitė nuo ankstesnės kopijos, `still` — ne,
`black` — nepasikeitė ir vidutinis šviesis < 24. Sink'o vėlinimas — jo paties eilutės „latency initialized / increase detected“.
"""
import hashlib
import json
import os
import pathlib
import shutil
import signal
import struct
import subprocess
import time

import functional as F  # noqa: E402

TESTS = {}
MXL_BIN = pathlib.Path(os.environ.get("MXL_BIN", "/root/src/mxl-pr653/install/Linux-Clang-Release/bin"))
FORMATS = ["1080i50", "1080p25", "1080p50", "720p50", "1080i59.94"]
DL = ["--deadline-ms", 0]


def tool(name):
    p = MXL_BIN / name
    return p if p.exists() else None


def have_x():
    return shutil.which("Xvfb") and shutil.which("xwd")


class Xvfb:
    """Atskiras Xvfb ekranas (be langų tvarkyklės; juodas fonas)."""

    def __init__(self, num):
        self.num = num
        self.p = subprocess.Popen(["Xvfb", f":{num}", "-screen", "0", "1920x1080x24", "-br", "-nolisten", "tcp"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.env = dict(F.ENV, DISPLAY=f":{num}")
        end = time.monotonic() + 10
        while time.monotonic() < end and not pathlib.Path(f"/tmp/.X11-unix/X{num}").exists():
            time.sleep(0.05)

    def grab(self):
        """(vidutinis šviesis, piršto atspaudas) arba None."""
        r = subprocess.run(["xwd", "-root", "-silent", "-display", f":{self.num}"], capture_output=True, env=self.env)
        if r.returncode != 0 or len(r.stdout) < 100:
            return None
        h = struct.unpack(">25I", r.stdout[:100])
        hsize, w, hgt, bpl, ncol = h[0], h[4], h[5], h[12], h[19]
        img = r.stdout[hsize + ncol * 12:]
        import numpy as np
        a = np.frombuffer(img[: bpl * hgt], dtype=np.uint8).reshape(hgt, bpl)[::8, : w * 4 : 32]
        return float(a.mean()), hashlib.md5(a.tobytes()).hexdigest()

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()


def sample(x, seconds, hz=5):
    """[(t, būsena)] seconds sekundžių; būsena: move | still | black | none."""
    out, prev = [], None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        t0 = time.monotonic()
        g = x.grab()
        if g is None:
            st = "none"
        elif prev is not None and g[1] != prev:
            st = "move"  # pirmiau judesys: tamsus judantis vaizdas (testsrc „ball“) — ne juoda
        elif g[0] < 24:
            st = "black"
        else:
            st = "still"
        prev = g[1] if g else None
        out.append((round(t0, 2), st))
        time.sleep(max(0.0, 1 / hz - (time.monotonic() - t0)))
    return out


def counts(seq):
    c = {}
    for _, s in seq:
        c[s] = c.get(s, 0) + 1
    return c


def sink_latency(path):
    """Didžiausias sink'o užrašytas vėlinimas, ms (None — nėra eilutės)."""
    best = None
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if "latency" in line and " ns" in line:
            try:
                v = int(line.rsplit(":", 1)[1].split()[0]) / 1e6
            except (ValueError, IndexError):
                continue
            best = v if best is None else max(best, v)
    return best


def warn_count(path, needle):
    p = pathlib.Path(path)
    return p.read_text(errors="replace").count(needle) if p.exists() else 0


class Tool:
    def __init__(self, out, name, args, env):
        self.log = pathlib.Path(out) / f"{name}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.log, "w")
        self.p = subprocess.Popen(args, stdout=self.f, stderr=subprocess.STDOUT, env=env)

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGINT)
            try:
                self.p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.f.close()
        return self.p.returncode


# --- interop_info -----------------------------------------------------------------------------------------------------

def mxl_info(d, fid):
    r = subprocess.run([str(tool("mxl-info")), "-d", str(d), "-f", fid], capture_output=True, text=True, env=F.ENV, timeout=20)
    txt = r.stdout + r.stderr
    head = None
    for line in txt.splitlines():
        if "Head index" in line:
            try:
                head = int(line.split(":")[1])
            except ValueError:
                pass
    fmt = next((ln.split(":", 1)[1].strip() for ln in txt.splitlines() if ln.strip().startswith("Format:")), None)
    act = next((ln.split(":", 1)[1].strip() for ln in txt.splitlines() if ln.strip().startswith("Active:")), None)
    return {"rc": r.returncode, "head": head, "format": fmt, "active": act, "text": txt[-400:]}


def interop_info(out):
    if not tool("mxl-info"):
        return [F.result("interop_info", "SKIP", why=f"nėra {MXL_BIN / 'mxl-info'}")]
    res = []
    for storage, impl in (("host", "cpu"), ("cuda-linear", "gpu")):
        for fmt in FORMATS:
            tid = f"interop_info[{storage}/{fmt}]"
            d = F.fresh_domain(f"iinfo-{storage}-{fmt}")
            g = F.FProc(out, f"iinfo/{storage}-{fmt}", "gen", d, "--flows", 1, "--audio", 1, "--format", fmt, "--impl", impl,
                        "--storage", storage, *DL, "--duration", 20)
            try:
                if not F.wait_flows(d, [F.role_flow_id(1, 0), F.role_flow_id(3, 0)]):
                    res.append(F.result(tid, "FAIL", why="gen flow'ai neatsirado"))
                    continue
                time.sleep(1)
                v, a = mxl_info(d, F.role_flow_id(1, 0)), mxl_info(d, F.role_flow_id(3, 0))
                ok = (v["rc"] == 0 and v["format"] == "Video" and v["active"] == "true" and (v["head"] or 0) > 0 and a["rc"] == 0
                      and a["format"] == "Audio")
                res.append(F.result(tid, "PASS" if ok else "FAIL", why=f"video {v['rc']}/{v['format']}/{v['active']}/head {v['head']}; "
                                    f"audio {a['rc']}/{a['format']}/{a['active']}", video=v, audio=a))
            finally:
                g.term()
        # v210a (gfxgen)
        tid = f"interop_info[{storage}/v210a]"
        d = F.fresh_domain(f"iinfo-{storage}-v210a")
        x = F.FProc(out, f"iinfo/{storage}-v210a", "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "logo", "--mode", "pace",
                    "--impl", impl, "--storage", storage, "--duration", 20)
        try:
            if not F.wait_flows(d, [F.gfx_flow_id(0, 0)]):
                res.append(F.result(tid, "FAIL", why="gfxgen flow'as neatsirado"))
                continue
            time.sleep(1)
            v = mxl_info(d, F.gfx_flow_id(0, 0))
            ok = v["rc"] == 0 and v["format"] == "Video" and v["active"] == "true" and (v["head"] or 0) > 0
            res.append(F.result(tid, "PASS" if ok else "FAIL", why=f"{v['rc']}/{v['format']}/{v['active']}/head {v['head']}", video=v))
        finally:
            x.term()
    return res


# --- mxl-gst-sink / mxl-gst-cuda-sink -----------------------------------------------------------------------------------

def sink_args(cuda, d, fid):
    if cuda:
        return [str(tool("mxl-gst-cuda-sink")), "-d", str(d), "-v", fid]
    return [str(tool("mxl-gst-sink")), "-d", str(d), "-v", fid]


def sink_scenarios(out, cuda, xnum):
    """Visi sink'o scenarijai vienam saugyklos tipui; [(id, status, why, details)]."""
    name = "interop_sink_cuda" if cuda else "interop_sink_host"
    need = "mxl-gst-cuda-sink" if cuda else "mxl-gst-sink"
    if not tool(need):
        return [F.result(name, "SKIP", why=f"nėra {MXL_BIN / need}")]
    if not have_x():
        return [F.result(name, "SKIP", why="nėra Xvfb / xwd (apt install xvfb x11-apps — tik testų įrankiai)")]
    impl, storage = ("gpu", "cuda-linear") if cuda else ("cpu", "host")
    v = F.vcommon(impl, storage) + ["--format", "1080p50"]
    res = []
    x = Xvfb(xnum)
    try:
        # 1) gen tiesiai ir logo išėjimas, be gedimo
        for src in ("gen", "logo"):
            tid = f"{name}[{src}]"
            d = F.fresh_domain(f"isink-{int(cuda)}-{src}")
            sub = f"isink{int(cuda)}/{src}"
            g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 1, *DL, "--duration", 60, *v)
            lg = None
            s = None
            try:
                F.wait_flows(d, [F.role_flow_id(1, 0)])
                fid = F.role_flow_id(1, 0)
                if src == "logo":
                    lg = F.FProc(out, f"{sub}/logo", "logo", d, "--flows", 1, *DL, "--duration", 40, "--warmup", 1, *v)
                    F.wait_flows(d, [F.role_flow_id(2, 0)])
                    fid = F.role_flow_id(2, 0)
                time.sleep(1)
                s = Tool(pathlib.Path(out) / sub, "sink", sink_args(cuda, d, fid), x.env)
                time.sleep(3)
                seq = sample(x, 6)
                c = counts(seq)
                alive = s.alive()
                lat = sink_latency(s.log)
                ok = alive and c.get("move", 0) >= 0.8 * len(seq)
                res.append(F.result(tid, "PASS" if ok else "FAIL", why=f"sink gyvas={alive} kopijos={c} sink vėlinimas ≤ {lat} ms",
                                    states=c, latency_ms=lat))
            finally:
                if s is not None:
                    s.stop()
                for p in (lg, g):
                    if p is not None:
                        p.term()
        # 2) gamintojo gedimas po logo, kiekviena politika; 3) --start immediate
        for case in ("invalid", "freeze", "black", "immediate"):
            tid = f"{name}[{'outage/' + case if case != 'immediate' else 'start-immediate'}]"
            d = F.fresh_domain(f"isink-{int(cuda)}-{case}")
            sub = f"isink{int(cuda)}/{case}"
            pol = "invalid" if case == "immediate" else case
            g = g2 = lg = s = None
            try:
                if case != "immediate":
                    g = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 1, *DL, "--duration", 60, *v)
                    F.wait_flows(d, [F.role_flow_id(1, 0)])
                lg = F.FProc(out, f"{sub}/logo", "logo", d, "--flows", 1, "--on-loss", pol, "--hold-ms", 1000, *DL, "--duration", 40,
                             "--warmup", 1, *v)
                F.wait_flows(d, [F.role_flow_id(2, 0)])
                time.sleep(1)
                s = Tool(pathlib.Path(out) / sub, "sink", sink_args(cuda, d, F.role_flow_id(2, 0)), x.env)
                time.sleep(3)
                before = sample(x, 3)
                if case == "immediate":
                    g2 = F.FProc(out, f"{sub}/gen", "gen", d, "--flows", 1, *DL, "--duration", 60, *v)
                    during = []
                else:
                    g.kill9()
                    during = sample(x, 3)
                    g2 = F.FProc(out, f"{sub}/gen2", "gen", d, "--flows", 1, *DL, "--duration", 60, *v)
                time.sleep(2)
                after = sample(x, 4)
                alive = s.alive()
                cb, cd, ca = counts(before), counts(during), counts(after)
                lat = sink_latency(s.log)
                early = warn_count(s.log, "TOO EARLY")
                resumed = alive and ca.get("move", 0) >= 0.6 * len(after)
                if case == "immediate":
                    ok = resumed
                    expect = "po gen starto juda"
                elif case == "black":
                    ok = resumed and cd.get("black", 0) >= 0.6 * len(during)
                    expect = "gedimo lange juoda"
                elif case == "freeze":  # --hold-ms 1000: ~1 s paskutinis geras kadras, tada juoda (K3)
                    # „move“ ≤ 2: du perėjimai lange (paskutinis kadras prieš nužudymą ir sustingęs → juoda)
                    ok = resumed and cd.get("move", 0) <= 2 and cd.get("still", 0) >= 3 and cd.get("black", 0) >= 0.4 * len(during)
                    expect = "gedimo lange ~1 s sustingęs, tada juoda"
                else:
                    ok = resumed and cd.get("move", 0) <= 1 and cd.get("black", 0) == 0
                    expect = "gedimo lange sustingęs paskutinis kadras"
                res.append(F.result(tid, "PASS" if ok else "FAIL",
                                    why=f"{expect}: prieš={cb} gedimas={cd} po={ca} sink gyvas={alive} TOO EARLY eilučių={early} "
                                        f"sink vėlinimas ≤ {lat} ms", before=cb, during=cd, after=ca, latency_ms=lat,
                                    too_early_lines=early))
            finally:
                if s is not None:
                    s.stop()
                for p in (lg, g2, g):
                    if p is not None and p.p.poll() is None:
                        p.term()
    finally:
        x.close()
    return res


def interop_sink_host(out):
    return sink_scenarios(out, False, 91)


def interop_sink_cuda(out):
    return sink_scenarios(out, True, 92)


# --- svetimas rašytojas ---------------------------------------------------------------------------------------------------

def interop_foreign_writer(out):
    tid = "interop_foreign_writer"
    if not tool("mxl-gst-testsrc") or not tool("mxl-gst-sink"):
        return [F.result(tid, "SKIP", why="nėra mxl-gst-testsrc / mxl-gst-sink")]
    if not have_x():
        return [F.result(tid, "SKIP", why="nėra Xvfb / xwd")]
    import chain
    d = F.fresh_domain("iforeign")
    base = pathlib.Path(out) / "iforeign"
    base.mkdir(parents=True, exist_ok=True)
    fid = F.role_flow_id(1, 7)
    cfg = base / "foreign_video.json"
    cfg.write_text(json.dumps(chain.foreign_video_def(fid)))
    x = Xvfb(93)
    ts = Tool(base, "testsrc", [str(tool("mxl-gst-testsrc")), "-d", str(d), "-v", str(cfg), "-p", "ball"], F.ENV)
    lg = s = None
    try:
        if not F.wait_flows(d, [fid], timeout=20):
            return [F.result(tid, "FAIL", why="mxl-gst-testsrc flow'as neatsirado")]
        lg = F.FProc(out, "iforeign/logo", "logo", d, "--flows", 1, "--in-flow-base", 7, "--flow-base", 7, "--impl", "cpu", "--storage",
                     "host", "--format", "1080p50", "--input-expect-ms", 12, *DL, "--duration", 16, "--warmup", 2)
        F.wait_flows(d, [F.role_flow_id(2, 7)])
        time.sleep(2)
        s = Tool(base, "sink", sink_args(False, d, F.role_flow_id(2, 7)), x.env)
        time.sleep(3)
        seq = sample(x, 5)
        rc = lg.wait(timeout=60)
        sm = F.vsummary(lg, "logo") or {}
        f = F.flow_ctr(sm, 0)
        lost = {k: f[k] for k in chain.CHAIN_LOSS if f.get(k)}
        c = counts(seq)
        ok = rc == 0 and not lost and f.get("fields_ok", 0) > 0 and c.get("move", 0) >= 0.6 * len(seq)
        return [F.result(tid, "PASS" if ok else "FAIL",
                         why=f"logo exit={rc} fields_ok={f.get('fields_ok')} praradimai={lost or 0} expect_us={(sm.get('proc') or {}).get('expect_us')} "
                             f"sink kopijos={c} sink vėlinimas ≤ {sink_latency(s.log)} ms", states=c)]
    finally:
        if s is not None:
            s.stop()
        if lg is not None and lg.p.poll() is None:
            lg.term()
        ts.stop()
        x.close()


TESTS["interop_info"] = interop_info
TESTS["interop_sink_host"] = interop_sink_host
TESTS["interop_sink_cuda"] = interop_sink_cuda
TESTS["interop_foreign_writer"] = interop_foreign_writer
