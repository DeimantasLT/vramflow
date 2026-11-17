# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Talpos „kelio“ paieška ir VRAM vienam flow (planas P8). Gryna logika: GPU, procesų ir laiko čia nėra.

Kelias = didžiausias N, kuriam pakopa praeina `repeats` pakartojimų iš eilės. N dvigubinamas nuo 1 (paskutinė
pakopa — `n_max`) iki pirmo FAIL arba VRAM sargo; tada dalijama pusiau tarp paskutinio PASS ir pirmo FAIL.
Prielaida — monotoniškumas: jei N praeina, praeina ir kiekvienas mažesnis N.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class KneeResult:
    knee: int
    tried: list = field(default_factory=list)  # [(n, [bool, …])] bandymo tvarka
    stopped_by: str = "n_max"  # "fail" | "n_max" | "vram_guard" — kas nustatė pirmą viršutinę ribą


def find_knee(run_level: Callable[[int], bool], n_max: int, repeats: int = 3,
              guard: Optional[Callable[[int], bool]] = None) -> KneeResult:
    """run_level(n) -> PASS?; guard(n) -> ar N dar leidžiama paleisti (pvz. VRAM sargas), None = visada."""
    if n_max < 1:
        raise ValueError(f"n_max turi būti ≥ 1, gauta {n_max}")
    if repeats < 1:
        raise ValueError(f"repeats turi būti ≥ 1, gauta {repeats}")
    res = KneeResult(knee=0)

    def passes(n: int) -> bool:
        runs = []
        res.tried.append((n, runs))
        for _ in range(repeats):
            ok = bool(run_level(n))
            runs.append(ok)
            if not ok:  # vienas FAIL — pakopa krenta, likę pakartojimai nebereikalingi
                return False
        return True

    lo, hi = 0, None  # lo — paskutinis PASS, hi — mažiausias žinomas netinkamas N
    n = 1
    while hi is None:
        if guard is not None and not guard(n):
            hi, res.stopped_by = n, "vram_guard"
        elif passes(n):
            lo = n
            if n == n_max:
                res.knee = n_max
                return res
            n = min(2 * n, n_max)
        else:
            hi, res.stopped_by = n, "fail"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (guard is None or guard(mid)) and passes(mid):
            lo = mid
        else:
            hi = mid
    res.knee = lo
    return res


def vram_per_flow(levels: list) -> tuple:
    """[(N, mem_used MB)] PASS pakopų → (nuolydis MB/flow, laisvasis narys MB, R²), mažiausi kvadratai."""
    xs = [float(n) for n, _ in levels]
    ys = [float(m) for _, m in levels]
    if len(set(xs)) < 2:
        raise ValueError("reikia bent dviejų skirtingų N pakopų")
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return slope, intercept, r2
