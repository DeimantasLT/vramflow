#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Trumpa summary-writer.json / summary-reader.json santrauka terminalui. Naudojimas: show_summary.py <out_dir>"""
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
for role in ("writer", "reader"):
    p = out / f"summary-{role}.json"
    if not p.exists():
        print(role, "— nėra santraukos")
        continue
    s = json.loads(p.read_text())
    print(f"{role}: exit={s['exit_code']} totals={s['totals']}")
    for f in s["flows"]:
        nz = {k: v for k, v in f["counters"].items() if v}
        print(f"  flow {f['flow']} {f['storage']} slots={f['slot_count']} grain={f['grain_size']} "
              f"idx={f['first_index']}..{f['last_index']} errors={len(f['errors'])} injected={len(f['injected'])}")
        print(f"    {nz}")
        lat = {k: (v["count"], v["p50_us"], v["p99_us"], v["p999_us"], v["max_us"])
               for k, v in f["latency"].items() if isinstance(v, dict) and v["count"]}
        for k, v in lat.items():
            print(f"    {k:20s} n={v[0]} p50={v[1]}us p99={v[2]}us p99.9={v[3]}us max={v[4]}us")
        if f.get("fatal"):
            print("    FATAL:", f["fatal"])
