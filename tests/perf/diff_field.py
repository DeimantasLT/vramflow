#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Diagnostika: palygina neatitinkantį dekoduotą v210 lauką su programinio dekoderio NV12 kadru (luma).

Naudojimas: diff_field.py ref.nv12 mismatch_clip<idx>_i<i>.v210 ...
Lauko klipo indeksas idx → kadras idx//2, paritetas idx%2 (lyginis = viršutinis). Spausdina, kiek luma pikselių skiriasi,
maksimalų skirtumą ir skirtumų stačiakampį (makroblokais 16×16 kadro koordinatėmis).
"""
import re
import sys

import numpy as np

W, H = 1920, 1080
FRAME = W * H * 3 // 2


def field_luma_from_v210(path):
    w = np.fromfile(path, dtype="<u4").reshape(540, 1280)
    s0, s1, s2 = w & 0x3FF, (w >> 10) & 0x3FF, (w >> 20) & 0x3FF
    # žodžių seka: w0 Cb0 Y0 Cr0 | w1 Y1 Cb1 Y2 | w2 Cr1 Y3 Cb2 | w3 Y4 Cr2 Y5
    g = np.stack([s0, s1, s2], axis=-1).reshape(540, 320, 12)
    y = g[:, :, [1, 3, 5, 7, 9, 11]].reshape(540, 1920)
    return y


def main():
    ref = np.memmap(sys.argv[1], dtype=np.uint8, mode="r")
    for p in sys.argv[2:]:
        m = re.search(r"clip(\d+)_i(\d+)", p)
        idx = int(m.group(1))
        frame, parity = idx // 2, idx % 2
        y8 = np.asarray(ref[frame * FRAME: frame * FRAME + W * H]).reshape(H, W)
        ref_y = y8[parity::2].astype(np.int32) << 2
        got = field_luma_from_v210(p).astype(np.int32)
        d = np.abs(got - ref_y)
        bad = np.argwhere(d > 0)
        if bad.size == 0:
            print(f"{p}: luma sutampa (skirtumas chroma)")
            continue
        r0, c0 = bad.min(axis=0)
        r1, c1 = bad.max(axis=0)
        print(f"{p}: kadras {frame} p={parity} luma skiriasi {len(bad)} px, max {d.max() >> 2} (8 bit), "
              f"sritis kadro eil. {2 * r0 + parity}..{2 * r1 + parity}, x {c0}..{c1}")


if __name__ == "__main__":
    main()
