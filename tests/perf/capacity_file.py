#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Talpos failas priėmimo kontrolei (planas P8 → P4): išmatuota GPU talpa iš `capacity.py --n auto` kelių.

  capacity_file.py <kampanijos katalogas> [--format 1080i50] [--out <katalogas>]

Skaito `knee-<funkcija>-G.json`, užklausia `nvidia-smi` GPU vardo, UUID ir tvarkyklės, rašo
`capacity-<gpu-uuid>.json`. Variklių talpa — Mpix/s (1080i50 = 51,84 Mpix/s srautui); `bound`:
`throughput` — kelią nulėmė pralaidumas, `lower_bound` — tikra talpa didesnė (paieškos arba tvarkyklės riba).
"""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys

SCHEMA = 1
# Mpix/s vienam srautui: plotis × aukštis × kadrai/s (pusės kadrai sudaro vieną kadrą)
FORMATS = {"1080i50": 1920 * 1080 * 25 / 1e6, "1080p25": 1920 * 1080 * 25 / 1e6, "1080p50": 1920 * 1080 * 50 / 1e6}
# Variklis → funkcija, kurios kelias jį matuoja
ENGINE_FUNCS = {"nvdec": "dec", "nvenc": "enc-nvenc-ll"}
REQUIRED = ["schema", "gpu.name", "gpu.uuid", "gpu.driver", "format", "measured_utc",
            "engines.nvdec.mpix_s", "engines.nvdec.bound", "engines.nvenc.mpix_s", "engines.nvenc.bound", "functions"]


def _bound(stopped_by: str) -> str:
    return "throughput" if stopped_by == "fail" else "lower_bound"


def build(knees: dict, gpu: dict, fmt: str, measured_utc: str) -> dict:
    """knees: {funkcijos žymė: knee-*.json santrauka} (tik G variantas) → talpos failo turinys."""
    if fmt not in FORMATS:
        raise ValueError(f"nežinomas formatas {fmt!r}; žinomi: {', '.join(FORMATS)}")
    per = FORMATS[fmt]
    g = {k: v for k, v in knees.items() if v.get("variant") == "G"}
    engines = {}
    for eng, func in ENGINE_FUNCS.items():
        s = g.get(func)
        if s is None:
            raise ValueError(f"{eng}: trūksta funkcijos {func!r} G varianto kelio")
        e = {"mpix_s": round(s["knee"] * per, 2), "bound": _bound(s["stopped_by"]), "from": func}
        if s["stopped_by"] == "nvenc_sessions":
            e["sessions"] = s["knee"]  # tvarkyklės leidžiamų vienalaikių sesijų skaičius
        engines[eng] = e
    functions = {}
    for tag, s in sorted(g.items()):
        v = s.get("vram") or {}
        functions[tag] = {"knee": s["knee"], "knee_bound": _bound(s["stopped_by"]), "stopped_by": s["stopped_by"],
                          "repeats": s.get("repeats"),
                          "vram_per_flow_bytes": round(v["per_flow_mb"] * 2**20) if v else None,
                          "vram_offset_bytes": round(v["offset_mb"] * 2**20) if v else None,
                          "vram_r2": v.get("r2")}
    d = {"schema": SCHEMA, "gpu": dict(gpu), "format": fmt, "mpix_s_per_flow": round(per, 2),
         "measured_utc": measured_utc, "engines": engines, "functions": functions}
    validate(d)
    return d


def validate(d: dict) -> None:
    for path in REQUIRED:
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise ValueError(f"talpos faile trūksta lauko {path}")
            cur = cur[part]
    if d["schema"] != SCHEMA:
        raise ValueError(f"talpos failo schema {d['schema']}, palaikoma {SCHEMA}")


def write(d: dict, out_dir: pathlib.Path) -> pathlib.Path:
    p = out_dir / f"capacity-{d['gpu']['uuid']}.json"
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def load(p: pathlib.Path) -> dict:
    d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    validate(d)
    return d


def read_knees(campaign: pathlib.Path) -> dict:
    """knee-<žymė>-G.json → {žymė: santrauka}; C / H variantai praleidžiami."""
    out = {}
    for p in sorted(campaign.glob("knee-*-G.json")):
        out[p.name[len("knee-"):-len("-G.json")]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def gpu_identity() -> dict:
    import gpuinfo  # tests/perf/gpuinfo.py — nvidia-smi kelias (WSL / Linux)
    q = subprocess.run([gpuinfo.nvidia_smi(), "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True).stdout.splitlines()[0]
    name, uuid, driver = (x.strip() for x in q.split(","))
    return {"name": name, "uuid": uuid, "driver": driver}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--format", default="1080i50", choices=sorted(FORMATS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    camp = pathlib.Path(a.campaign)
    knees = read_knees(camp)
    # Matavimo laikas — naujausio kelio failo keitimo laikas (UTC)
    newest = max((camp / f"knee-{t}-G.json").stat().st_mtime for t in knees) if knees else 0
    stamp = datetime.datetime.fromtimestamp(newest, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        d = build(knees, gpu_identity(), a.format, stamp)
        p = write(d, pathlib.Path(a.out) if a.out else camp)
    except ValueError as e:  # trūksta kelio varikliui, nežinomas formatas, netinkama schema — viena eilutė, ne traceback
        print(f"capacity_file: {e}", file=sys.stderr)
        return 2
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
