#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Laikrodžio kokybės zondas (tik skaito; architektūra 2026-11-11-laikas-ir-sinchronizacija.md T0).

Lygina CLOCK_TAI / REALTIME / MONOTONIC su neslew'inamu CLOCK_MONOTONIC_RAW ir Hyper-V PTP laikrodžiu (/dev/ptp_hyperv,
jei yra — tai host laikas). Rodo: dažnio nuokrypį (ppm), žingsnius (> 5 ms per 0,5 s), TAI−UTC poslinkį.
Naudojimas: 80-clock-probe.py [trukmė_s=60] [--json failas]
"""
import json
import os
import sys
import time

dur = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 60.0
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
RAW, MONO, REAL, TAI = time.CLOCK_MONOTONIC_RAW, time.CLOCK_MONOTONIC, time.CLOCK_REALTIME, time.CLOCK_TAI
g = time.clock_gettime_ns
phc = None
if os.path.exists("/dev/ptp_hyperv"):
    fd = os.open("/dev/ptp_hyperv", os.O_RDONLY)
    phc = ((~fd) << 3) | 3  # FD_TO_CLOCKID


def sample():
    a = g(RAW)
    p = g(phc) if phc is not None else 0
    b = g(RAW)
    return (a + b) // 2, p, g(MONO), g(REAL), g(TAI)


r0, p0, m0, re0, t0 = sample()
prev = (r0, re0, t0)
steps = []
tai_off = set()
while True:
    time.sleep(0.5)
    r, p, m, re, t = sample()
    el = (r - r0) / 1e9
    d_raw = r - prev[0]
    d_real = re - prev[1]
    if abs(d_real - d_raw) > 5_000_000:  # realtime per 0,5 s pakito > 5 ms daugiau nei RAW
        steps.append({"t_s": round(el, 2), "step_ms": round((d_real - d_raw) / 1e6, 1)})
    tai_off.add(round((t - re) / 1e9))
    prev = (r, re, t)
    if el >= dur:
        break
res = {
    "duration_s": round(el, 1),
    "mono_vs_raw_ppm": round(((m - m0) - (r - r0)) / (r - r0) * 1e6, 1),
    "realtime_vs_raw_ppm": round(((re - re0) - (r - r0)) / (r - r0) * 1e6, 1),
    "phc_vs_raw_ppm": round(((p - p0) - (r - r0)) / (r - r0) * 1e6, 1) if phc is not None else None,
    "realtime_minus_phc_ms": round((re - p) / 1e6, 3) if phc is not None else None,
    "tai_minus_utc_s": sorted(tai_off),
    "steps": steps,
}
print(json.dumps(res, indent=1, ensure_ascii=False))
if out:
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
