#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Palygina du raw NV12 1920x1080 failus kadras po kadro, atskirai kiekvienam laukui (luma ir chroma).
Naudojimas: compare_nv12.py etalonas.nv12 kitas.nv12
"""
import sys

import numpy as np

W, H = 1920, 1080
FRAME = W * H * 3 // 2
a = np.memmap(sys.argv[1], dtype=np.uint8, mode="r")
b = np.memmap(sys.argv[2], dtype=np.uint8, mode="r")
n = min(len(a), len(b)) // FRAME
bad = []
for k in range(n):
    fa = np.asarray(a[k * FRAME:(k + 1) * FRAME])
    fb = np.asarray(b[k * FRAME:(k + 1) * FRAME])
    if np.array_equal(fa, fb):
        continue
    ya, yb = fa[:W * H].reshape(H, W), fb[:W * H].reshape(H, W)
    ca, cb = fa[W * H:].reshape(H // 2, W), fb[W * H:].reshape(H // 2, W)
    res = []
    for p, name in ((0, "top"), (1, "bottom")):
        dy = int(np.count_nonzero(ya[p::2] != yb[p::2]))
        dc = int(np.count_nonzero(ca[p::2] != cb[p::2]))
        if dy or dc:
            res.append(f"{name}: Y {dy} px, C {dc}")
    bad.append(k)
    print(f"kadras {k}: " + "; ".join(res))
print(f"kadrų: {n} (a {len(a) // FRAME}, b {len(b) // FRAME}); skirtingų: {len(bad)} {bad[:40]}")
