#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Našumo regresijos vartai (planas production-ready §3.3, G1): talpos kampanija prieš bazę.

  regress.py <kampanijos katalogas> --baseline <failas> [--update]   → exit 0 PASS, 1 FAIL

Kampanija — `capacity.py --n auto` (scripts/linux/70-capacity.sh PROFILE=knee) katalogas: `knee-<žymė>.json` (kelias, VRAM
srautui) ir `capacity-<žymė>.json` (pakopos su santraukų histogramomis). Žymė — funkcija ir variantas, pvz. `logo-G`,
`composite-k2-G`. Kiekvienai bazės funkcijai:
  * kelias < bazės — FAIL;
  * p99 > P99_MAX × bazės — FAIL (išėjimo vėlavimo histogramos `out_lateness` / `frame_lateness` p99, prie kelio N
    praėjusių pakartojimų mediana; „praėjo“ — tas pats kriterijus kaip kelio paieškoje, `pass_tol`);
  * VRAM srautui > VRAM_MAX × bazės — FAIL;
  * funkcijos ar rodiklio kampanijoje nėra — FAIL „nematuota“.
Kampanijos funkcijos, kurių bazėje nėra, nevertinamos. `--update` perrašo bazę iš kampanijos (exit 0).
"""
import argparse
import datetime
import json
import pathlib
import sys

SCHEMA = 1
P99_MAX = 1.10
VRAM_MAX = 1.05
LATENCY_KEYS = ("out_lateness", "frame_lateness")  # išėjimo vėlavimas: logo / composite / dec — out, enc / chain — frame


def _median(v):
    s = sorted(v)
    return (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2 if s else None


def _p99_at(levels, n):
    """PASS pakartojimų prie N išėjimo vėlavimo p99 mediana (ms) arba None. PASS — kelio paieškos kriterijus: pass_tol
    (capacity.py --tol), senesnėse kampanijose be jo — pass."""
    vals = []
    for lv in levels:
        if lv.get("n") != n or not lv.get("pass_tol", lv.get("pass")):
            continue
        hist = lv.get("hist") or {}
        key = next((k for k in LATENCY_KEYS if k in hist), None)
        if key is not None and hist[key].get("p99_ms") is not None:
            vals.append(float(hist[key]["p99_ms"]))
    return _median(vals)


def load_campaign(d) -> dict:
    """Kampanijos katalogas → {žymė: {"knee": int, "p99_ms": float | None, "vram_per_flow_mb": float | None}}."""
    out = {}
    for kf in sorted(pathlib.Path(d).glob("knee-*.json")):
        tag = kf.name[len("knee-"):-len(".json")]
        k = json.loads(kf.read_text())
        cf = kf.with_name(f"capacity-{tag}.json")
        levels = json.loads(cf.read_text()) if cf.exists() else []
        vram = (k.get("vram") or {}).get("per_flow_mb")
        out[tag] = {"knee": int(k["knee"]), "p99_ms": _p99_at(levels, k["knee"]) if isinstance(levels, list) else None,
                    "vram_per_flow_mb": None if vram is None else float(vram)}
    return out


def compare(base: dict, cur: dict) -> list:
    """Bazė ir kampanija ({žymė: rodikliai}) → pažeidimų tekstai (tuščias — PASS)."""
    bad = []
    for tag, b in sorted(base.items()):
        c = cur.get(tag)
        if c is None:
            bad.append(f"{tag}: nematuota (kampanijoje nėra knee-{tag}.json)")
            continue
        if c.get("knee") is None:
            bad.append(f"{tag}: kelias nematuotas")
        elif c["knee"] < b["knee"]:
            bad.append(f"{tag}: kelias {c['knee']} < bazės {b['knee']}")
        for key, name, lim, unit in (("p99_ms", "p99", P99_MAX, "ms"), ("vram_per_flow_mb", "VRAM srautui", VRAM_MAX, "MB")):
            if b.get(key) is None:
                continue
            if c.get(key) is None:
                bad.append(f"{tag}: {name} nematuota")
            elif c[key] > lim * b[key]:
                bad.append(f"{tag}: {name} {c[key]:.2f} {unit} > {lim:.2f} × bazės {b[key]:.2f} {unit}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--update", action="store_true", help="perrašyti bazę iš kampanijos")
    a = ap.parse_args(argv)
    cur = load_campaign(a.campaign)
    base_p = pathlib.Path(a.baseline)
    if a.update:
        if not cur:
            print(f"FAIL: kampanijoje {a.campaign} nėra knee-*.json")
            return 1
        base_p.parent.mkdir(parents=True, exist_ok=True)
        base_p.write_text(json.dumps({"schema": SCHEMA, "source": str(a.campaign),
                                      "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      "functions": cur}, indent=1, ensure_ascii=False) + "\n")
        print(f"bazė atnaujinta: {base_p} ({len(cur)} funkcijos)")
        return 0
    if not base_p.exists():
        print(f"FAIL: bazės {base_p} nėra")
        return 1
    base = json.loads(base_p.read_text()).get("functions") or {}
    if not base:
        print(f"FAIL: bazė {base_p} tuščia")
        return 1
    bad = compare(base, cur)
    for tag in sorted(base):
        c = cur.get(tag) or {}
        print(f"  {tag}: kelias {c.get('knee')} (bazė {base[tag]['knee']}), p99 {c.get('p99_ms')} ms "
              f"(bazė {base[tag].get('p99_ms')}), VRAM {c.get('vram_per_flow_mb')} MB (bazė {base[tag].get('vram_per_flow_mb')})")
    print("PASS" if not bad else "FAIL: " + "; ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
