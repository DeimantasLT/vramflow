#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Briaunos laiko sutartis (planas docs/planai/2026-11-12-tick-variklis.md §6 D7, V24; kviečiami iš functional.py --only).

Visi procesai — su NUMATYTOSIOMIS laiko parinktimis (--deadline-ms 0 aiškiai, kad harness'as neprisegtų 30 ms).

fchain5          — 5 pakopų grandinė gen(+garsas) → logo → logo → composite (+1 gfxgen sluoksnis) → enc, 30 s, C ir G:
                   kiekvienoje pakopoje 0 vėlavimų / praradimų; kiekvienos pakopos skelbiama riba (commit_bound_us) ≥ jos
                   išmatuoto commit'o vėlavimo p99.9 (gen commit_lateness, logo / composite out_lateness, enc frame_lateness);
                   kiekviena pakopa skaito aukštupio žymą (inputs_untagged == 0); galinis vėlinimas — enc frame_lateness.
fchain5_kill     — ta pati grandinė; viršutinis gamintojas (gen) nužudomas (kill -9) 10-ą s ir po 2 s paleidžiamas iš naujo:
                   praradimą skaičiuoja tik pirma pakopa po jo (logo1 missing_fields); žemiau (logo2, composite, enc) — 0 savų
                   praradimų, aukštupio spragos — gap_upstream (logo2, composite) / enc gap kadrai be late / missing.
fforeign_expect  — svetimas rašytojas (mxl-gst-testsrc, flow be VRAMFlow žymų) → logo --input-expect-ms 12: fazė E = 12 ms
                   (expect_us == 12000, inputs_untagged == 1), skelbiama riba = E + 8,5 ms. SKIP, jei įrankio nėra.
"""
import json
import os
import pathlib
import shutil
import subprocess
import time

import functional as F  # noqa: E402

TESTS = {}
DUR = 30
WARM = 5
CHAIN_LOSS = ("late_fields", "late_skips", "lapped", "too_late", "in_invalid", "read_errors", "missing_fields", "flow_invalid",
              "output_errors", "commit_errors")
ENC_LOSS = ("late_frames", "missing_frames", "lapped", "too_late", "in_invalid", "frames_dropped_queue", "late_skips_frames",
            "audio_underrun_frames", "encode_errors", "mux_errors", "frame_alloc_errors")
DL = ["--deadline-ms", 0]  # numatytoji riba 3·P/4 nuo T + E (V20 + V24)
GST_BIN = pathlib.Path(os.environ.get("MXL_BIN", "/root/src/mxl-pr653/install/Linux-Clang-Release/bin"))


def hist(s, name):
    return ((s or {}).get("hist") or {}).get(name) or {}


def losses(f, keys):
    return {k: f[k] for k in keys if f.get(k)}


def chain5(out, sub, impl, storage, kill=False):
    """Grandinė viename domene. Flow ID: gen 1:0 (+ garsas 3:0), logo1 2:0, logo2 2:1, composite 6:0, gfx 5:0."""
    d = F.fresh_domain(sub)
    v = F.vcommon(impl, storage)
    enc = "x264" if impl == "cpu" else "nvenc"
    total = DUR + WARM + 8
    procs = {}

    def gen(tag):  # video grandinės viršus (be garso: garsas — atskiras šaltinis, nes VRAMFlow video funkcijos garso neperneša)
        return F.FProc(out, f"{sub}/{tag}", "gen", d, "--flows", 1, "--audio", 0, *DL, "--duration", total + 30, *v)

    try:
        procs["gen"] = gen("gen")
        # garso šaltinis (enc rinkinio narys): atskiras gen --video 0, kad nužudytas video viršus nebūtų ir garso gamintojas —
        # kitaip enc būtų PIRMA pakopa po garso gamintojo ir jo garso trūkumas teisėtai būtų jo praradimas
        procs["gena"] = F.FProc(out, f"{sub}/gena", "gen", d, "--flows", 1, "--video", 0, "--audio", 1, *DL, "--duration", total + 30, *v)
        if not F.wait_flows(d, [F.role_flow_id(1, 0), F.role_flow_id(3, 0)]):
            return {"err": "gen flow'ai neatsirado"}
        procs["gfxgen"] = F.FProc(out, f"{sub}/gfxgen", "gfxgen", d, "--flows", 1, "--layers", 1, "--layer-geom", "logo",
                                  "--mode", "pace", "--duration", total + 30, *v)
        procs["logo1"] = F.FProc(out, f"{sub}/logo1", "logo", d, "--flows", 1, *DL, "--duration", total + 20, "--warmup", WARM, *v)
        if not F.wait_flows(d, [F.role_flow_id(2, 0), F.gfx_flow_id(0, 0)]):
            return {"err": "logo1 / gfxgen flow'ai neatsirado"}
        procs["logo2"] = F.FProc(out, f"{sub}/logo2", "logo", d, "--flows", 1, "--in-role", 2, "--in-flow-base", 0, "--flow-base", 1,
                                 *DL, "--duration", total + 14, "--warmup", WARM, *v)
        if not F.wait_flows(d, [F.role_flow_id(2, 1)]):
            return {"err": "logo2 flow'as neatsirado"}
        procs["composite"] = F.FProc(out, f"{sub}/composite", "composite", d, "--flows", 1, "--layers", 1, "--layer-xy", "1488,64",
                                     "--base", "gen", "--in-role", 2, "--in-flow-base", 1, "--out-format", "v210", *DL,
                                     "--duration", total + 8, "--warmup", WARM, *v)
        if not F.wait_flows(d, [F.role_flow_id(6, 0)]):
            return {"err": "composite flow'as neatsirado"}
        time.sleep(1)
        ea = ["--encoder", enc, "--audio", 1, "--port-base", 24000 + (100 if kill else 0) + (10 if impl == "cpu" else 0)]
        if enc == "x264":
            ea += ["--x264-preset", "ultrafast"]
        procs["enc"] = F.FProc(out, f"{sub}/enc", "enc", d, "--flows", 1, "--in-role", 6, *DL, "--duration", DUR + WARM,
                               "--warmup", WARM, *ea, *v)
        if kill:
            time.sleep(WARM + 5)
            procs["gen"].kill9()
            time.sleep(2)
            procs["gen2"] = gen("gen2")
        rcs = {}
        for k in ("enc", "composite", "logo2", "logo1"):
            rcs[k] = procs[k].wait(timeout=total + 90)
        for k in ("gen", "gen2", "gena", "gfxgen"):  # šaltiniai — iki sustabdymo; santrauka parašoma po SIGTERM
            if k in procs:
                procs[k].term()
        return {"rc": rcs, "s": {k: F.vsummary(p, p.sub) for k, p in procs.items()}}
    finally:
        for p in procs.values():
            if p.p.poll() is None:
                p.term()


def stage_rows(r):
    """Pakopų lentelė: (vardas, skelbiama riba ms, p50 ms, p99.9 ms, savi praradimai, santrauka)."""
    s = r["s"]
    rows = []
    for name, role, h, keys in (("gen", "gen", "commit_lateness", ("late_skips",)), ("logo1", "logo", "out_lateness", CHAIN_LOSS),
                                ("logo2", "logo", "out_lateness", CHAIN_LOSS), ("composite", "composite", "out_lateness", CHAIN_LOSS),
                                ("enc", "enc", "frame_lateness", ENC_LOSS)):
        sm = s.get(name) or {}
        f = F.flow_ctr(sm, 0)
        lost = losses(f, keys)
        if name == "gen" and (sm.get("proc") or {}).get("late_skips"):
            lost["late_skips"] = sm["proc"]["late_skips"]
        hh = hist(sm, h)
        rows.append({"stage": name, "bound_ms": (sm.get("proc") or {}).get("commit_bound_us", 0) / 1000, "p50_ms": hh.get("p50_ms"),
                     "p999_ms": hh.get("p999_ms"), "max_ms": hh.get("max_ms"), "lost": lost, "flow": f, "proc": sm.get("proc") or {},
                     "exit": (r["rc"] or {}).get(name), "any_loss": sm.get("any_loss")})
    return rows


def fchain5(out):
    res = []
    for tag, impl, storage in [x for x in F.comp_variants() if x[0] in ("C", "G")]:
        tid = f"fchain5[{tag}]"
        r = chain5(out, f"fchain5-{tag}", impl, storage)
        if "err" in r:
            res.append(F.result(tid, "FAIL", why=r["err"]))
            continue
        rows = stage_rows(r)
        bad = []
        for x in rows:
            if x["lost"]:
                bad.append(f"{x['stage']} praradimai {x['lost']}")
            if x["stage"] != "gen" and x["exit"] != 0:
                bad.append(f"{x['stage']} exit {x['exit']}")
            if x["p999_ms"] is None or x["bound_ms"] <= 0 or x["p999_ms"] > x["bound_ms"]:
                bad.append(f"{x['stage']} p99.9 {x['p999_ms']} ms > riba {x['bound_ms']} ms")
            if x["stage"] not in ("gen",) and x["proc"].get("inputs_untagged", 0) not in (0,) and x["stage"] != "composite":
                bad.append(f"{x['stage']} įėjimai be žymos {x['proc'].get('inputs_untagged')}")
        # composite: bazė su žyma; sluoksnis (gfxgen) — taip pat su žyma (gfxgen skelbia ribą)
        comp = next(x for x in rows if x["stage"] == "composite")
        if comp["proc"].get("inputs_untagged", 0):
            bad.append(f"composite įėjimai be žymos {comp['proc'].get('inputs_untagged')}")
        e2e = next(x for x in rows if x["stage"] == "enc")
        table = " | ".join(f"{x['stage']}: riba {x['bound_ms']:.1f}, p50 {x['p50_ms']}, p99.9 {x['p999_ms']}, praradimų "
                           f"{sum(x['lost'].values()) if x['lost'] else 0}" for x in rows)
        res.append(F.result(tid, "FAIL" if bad else "PASS",
                            why=("; ".join(bad) or "ok") + f" | {table} | galinis vėlinimas (enc frame_lateness) p50 {e2e['p50_ms']} ms, "
                                f"p99.9 {e2e['p999_ms']} ms", rows=[{k: x[k] for k in ('stage', 'bound_ms', 'p50_ms', 'p999_ms', 'max_ms',
                                                                                       'lost', 'exit')} for x in rows]))
    return res


def fchain5_kill(out):
    res = []
    for tag, impl, storage in [x for x in F.comp_variants() if x[0] in ("C", "G")]:
        tid = f"fchain5_kill[{tag}]"
        r = chain5(out, f"fchain5k-{tag}", impl, storage, kill=True)
        if "err" in r:
            res.append(F.result(tid, "FAIL", why=r["err"]))
            continue
        rows = {x["stage"]: x for x in stage_rows(r)}
        bad = []
        l1 = rows["logo1"]["flow"]
        if l1.get("missing_fields", 0) < 50:
            bad.append(f"logo1 missing_fields {l1.get('missing_fields', 0)} < 50 (gedimas nepastebėtas?)")
        for st in ("logo2", "composite"):
            if rows[st]["lost"]:
                bad.append(f"{st} savi praradimai {rows[st]['lost']}")
            if rows[st]["flow"].get("gap_upstream", 0) < 50:
                bad.append(f"{st} gap_upstream {rows[st]['flow'].get('gap_upstream', 0)} < 50")
            if rows[st]["exit"] != 0:
                bad.append(f"{st} exit {rows[st]['exit']}")
        el = {k: v for k, v in rows["enc"]["lost"].items()}
        if el:
            bad.append(f"enc savi praradimai {el}")
        if rows["enc"]["exit"] != 0:
            bad.append(f"enc exit {rows['enc']['exit']}")
        # D5: video spragos kadrų garsas (atskiras, gyvas garso šaltinis) koduojamas — audio_only_frames ≈ spragos kadrai
        ef = rows["enc"]["flow"]
        gap_frames = rows["composite"]["flow"].get("gap_upstream", 0) // 2
        if ef.get("audio_only_frames", 0) < gap_frames - 2 or ef.get("audio_underrun_gap_frames", 0):
            bad.append(f"enc spragų kadrų garsas: audio_only_frames {ef.get('audio_only_frames', 0)} < {gap_frames} - 2 arba "
                       f"audio_underrun_gap_frames {ef.get('audio_underrun_gap_frames', 0)}")
        info = " | ".join(f"{k}: {({kk: v['flow'].get(kk) for kk in ('missing_fields', 'gap_upstream', 'gap_invalid', 'fields_ok', 'frames_encoded', 'gap_frames', 'audio_only_frames') if v['flow'].get(kk)})}"
                          for k, v in rows.items() if k != "gen")
        res.append(F.result(tid, "FAIL" if bad else "PASS", why=("; ".join(bad) or "ok") + " | " + info,
                            rows={k: {"flow": v["flow"], "exit": v["exit"]} for k, v in rows.items()}))
    return res


def foreign_video_def(fid):
    """Svetimo rašytojo flow aprašas: 1080p50 v210 be VRAMFlow žymų (tik grouphint; mxl-gst-testsrc interlaced nepalaiko)."""
    return {"id": fid, "description": "foreign testsrc", "label": "foreign testsrc",
            "tags": {"urn:x-nmos:tag:grouphint/v1.0": ["foreign-testsrc:Video"]}, "format": "urn:x-nmos:format:video",
            "parents": [], "media_type": "video/v210", "grain_rate": {"numerator": 50, "denominator": 1}, "frame_width": 1920,
            "frame_height": 1080, "interlace_mode": "progressive", "colorspace": "BT709",
            "components": [{"name": "Y", "width": 1920, "height": 1080, "bit_depth": 10},
                           {"name": "Cb", "width": 960, "height": 1080, "bit_depth": 10},
                           {"name": "Cr", "width": 960, "height": 1080, "bit_depth": 10}]}


def fforeign_expect(out):
    tid = "fforeign_expect"
    src = GST_BIN / "mxl-gst-testsrc"
    if not src.exists():
        return [F.result(tid, "SKIP", why=f"nėra {src}")]
    d = F.fresh_domain("fforeign")
    base = pathlib.Path(out) / "fforeign"
    base.mkdir(parents=True, exist_ok=True)
    fid = F.role_flow_id(1, 7)
    cfg = base / "foreign_video.json"
    cfg.write_text(json.dumps(foreign_video_def(fid)))
    err = open(base / "testsrc.err", "w")
    ts = subprocess.Popen([str(src), "-d", str(d), "-v", str(cfg), "-p", "ball"], stdout=err, stderr=err, env=F.ENV)
    lg = None
    try:
        if not F.wait_flows(d, [fid], timeout=20):
            return [F.result(tid, "FAIL", why="mxl-gst-testsrc flow'as neatsirado")]
        lg = F.FProc(out, "fforeign/logo", "logo", d, "--flows", 1, "--in-flow-base", 7, "--flow-base", 7, "--impl", "cpu",
                     "--storage", "host", "--format", "1080p50", "--input-expect-ms", 12, *DL, "--duration", 20, "--warmup", 3)
        rc = lg.wait(timeout=90)
        s = F.vsummary(lg, "logo") or {}
        p = s.get("proc") or {}
        f = F.flow_ctr(s, 0)
        bad = []
        if p.get("expect_us") != 12000 or p.get("inputs_untagged") != 1 or p.get("inputs_tagged") != 0:
            bad.append(f"fazė: expect_us={p.get('expect_us')} untagged={p.get('inputs_untagged')} (laukta 12000 / 1)")
        if p.get("commit_bound_us") != 20500:
            bad.append(f"commit_bound_us={p.get('commit_bound_us')} (laukta 12 + 0,5 + 5 + 3 = 20,5 ms)")
        if not f.get("fields_ok"):
            bad.append("fields_ok == 0")
        h = hist(s, "out_lateness")
        return [F.result(tid, "FAIL" if bad else "PASS",
                         why=("; ".join(bad) or "ok") + f" | logo exit={rc} fields_ok={f.get('fields_ok')} praradimai="
                             f"{losses(f, CHAIN_LOSS) or 0} out_lateness p50 {h.get('p50_ms')} p99.9 {h.get('p999_ms')} ms",
                         flow=f, proc=p)]
    finally:
        if lg is not None and lg.p.poll() is None:
            lg.term()
        ts.terminate()
        try:
            ts.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ts.kill()


TESTS["fchain5"] = fchain5
TESTS["fchain5_kill"] = fchain5_kill
TESTS["fforeign_expect"] = fforeign_expect
