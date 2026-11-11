#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Gyvai įrašytų NV12 kadrų palyginimas su etalonu pagal barkodą (klipo kadro indeksas iš viršutinio lauko žymės).
Naudojimas: compare_by_marker.py ref.nv12 captured.nv12 [k0=1000]
"""
import sys

import numpy as np

W, H = 1920, 1080
FRAME = W * H * 3 // 2


def marker(y):
    top = y[0::2]  # viršutinio lauko eilutės
    blk = top[4:12, :].astype(np.int32)
    bits = 0
    for k in range(32):
        if blk[:, 32 * k + 12:32 * k + 20].mean() > 125:
            bits |= 1 << k
    return bits


ref = np.memmap(sys.argv[1], dtype=np.uint8, mode="r")
cap = np.memmap(sys.argv[2], dtype=np.uint8, mode="r")
k0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
nref = len(ref) // FRAME
ok = bad = unknown = 0
bad_list = []
for k in range(len(cap) // FRAME):
    f = np.asarray(cap[k * FRAME:(k + 1) * FRAME])
    mk = marker(f[:W * H].reshape(H, W))
    idx = (mk & 0xFFFFFF) - k0
    if idx < 0 or idx >= 2 * nref or idx % 2:
        unknown += 1
        continue
    r = np.asarray(ref[(idx // 2) * FRAME:(idx // 2 + 1) * FRAME])
    if np.array_equal(f, r):
        ok += 1
    else:
        bad += 1
        y = f[:W * H].reshape(H, W) != r[:W * H].reshape(H, W)
        rows = np.where(y.any(axis=1))[0]
        bad_list.append((k, idx // 2, int(y.sum()), int(rows.min()) if rows.size else -1))
print(f"kadrų {len(cap) // FRAME}: sutampa {ok}, skiriasi {bad}, nežinomas barkodas {unknown}")
for b in bad_list[:20]:
    print(f"  įrašo kadras {b[0]} = klipo kadras {b[1]}: Y skiriasi {b[2]} px, nuo eilutės {b[3]}")
