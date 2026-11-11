// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bandomojo lauko turinio APIBRĖŽIMAS (architektūra GPU funkcijų §6.5, §6.1, §5.1). __host__ __device__:
// tą pačią funkciją naudoja GPU generatorius ir CPU etalonas (tai yra turinio specifikacija, ne tikrinama logika).
// Kadro koordinatės: x 0..1919, fy = 2j + p (j — lauko eilutė, p = index & 1). Visos reikšmės legalios: Y 64..940, C 64..960.
#include <cstdint>

#include "common/hd.hpp"
#include "media/v210.hpp"

namespace mxlv::media {

MXLV_HD std::uint32_t fmix32(std::uint32_t x) {
    x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    return x;
}

MXLV_HD int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct GenParams {
    std::uint64_t run_seed;
    std::uint32_t flow;
    std::uint32_t complexity;  // 0 statinis … 3 sunkus koderiui
};

// Barkodo bitas k (0..31) laukui `index` flow'ui `flow`: b0..23 = index mod 2^24, b24 = paritetas, b25..31 = flow mod 128.
MXLV_HD std::uint32_t marker_word(std::uint64_t index, std::uint32_t flow) {
    return static_cast<std::uint32_t>(index & 0xFFFFFFu) | (static_cast<std::uint32_t>(index & 1u) << 24) |
           ((flow & 127u) << 25);
}

// Luma pikseliui (x, fy) lauke `index`.
MXLV_HD std::uint16_t gen_y(const GenParams& g, std::uint64_t index, int x, int fy) {
    const int j = fy >> 1;
    // barkodas: lauko eilutės 0..15, blokai po 32 px; tik 32 blokai (x < 1024), už jų — bitas 0 (Y=64).
    // Pastaba: postūmis ≥ 32 yra UB (x86 maskuoja iki 5 bitų, GPU — 0), todėl blokas tikrinamas aiškiai.
    if (j < 16) {
        const int blk = x >> 5;
        const std::uint32_t bit = blk < 32 ? (marker_word(index, g.flow) >> blk) & 1u : 0u;
        return bit ? 940 : 64;
    }
    // sync blyksnis: index mod 50 == 0, x 1600..1855, lauko eil. 32..95
    if (index % 50 == 0 && x >= 1600 && x < 1856 && j >= 32 && j < 96) return 940;
    int y = 64 + (x + fy) * 876 / (1920 + 1080);  // įstrižas gradientas
    const int bar = static_cast<int>((index * 8) % 1920);
    if (((x - bar + 1920) % 1920) < 16) y = 900;
    if (g.complexity > 0) {
        const int speed = 1 << g.complexity;  // 2/4/8 px per lauką
        const int bx = (x + static_cast<int>(index % 100000) * speed) >> 3;
        const int by = fy >> 3;
        const std::uint32_t n = fmix32(static_cast<std::uint32_t>(g.run_seed) ^ (g.flow * 0x9E3779B1u) ^
                                       (static_cast<std::uint32_t>(bx) * 0x85EBCA6Bu) ^ (static_cast<std::uint32_t>(by) * 0xC2B2AE35u));
        const int amp = 32 << g.complexity;  // ±64/128/256
        y += static_cast<int>(n % static_cast<std::uint32_t>(2 * amp + 1)) - amp;
    }
    return static_cast<std::uint16_t>(clampi(y, 64, 940));
}

// Chroma porai, kurios lyginis pikselis x (x lyginis).
MXLV_HD void gen_c(const GenParams& g, std::uint64_t index, int x, int fy, std::uint16_t& cb, std::uint16_t& cr) {
    const int j = fy >> 1;
    if (j < 16 || (index % 50 == 0 && x >= 1600 && x < 1856 && j >= 32 && j < 96)) {
        cb = 512;
        cr = 512;
        return;
    }
    int b = 412 + x * 200 / 1920;
    int r = 412 + fy * 200 / 1080;
    if (g.complexity > 1) {
        const std::uint32_t n = fmix32(static_cast<std::uint32_t>(index) * 0x27D4EB2Fu ^ static_cast<std::uint32_t>(x >> 4) ^
                                       (static_cast<std::uint32_t>(fy >> 4) << 12) ^ g.flow);
        b += static_cast<int>(n & 63) - 32;
        r += static_cast<int>((n >> 8) & 63) - 32;
    }
    cb = static_cast<std::uint16_t>(clampi(b, 64, 960));
    cr = static_cast<std::uint16_t>(clampi(r, 64, 960));
}

}  // namespace mxlv::media
