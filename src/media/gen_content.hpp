// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bandomojo lauko turinio APIBRĖŽIMAS (architektūra GPU funkcijų §6.5, §6.1, §5.1). __host__ __device__:
// tą pačią funkciją naudoja GPU generatorius ir CPU etalonas (tai yra turinio specifikacija, ne tikrinama logika).
// Kadro koordinatės: x 0..width−1, fy — kadro eilutė (laukui 2j + p, j — lauko eilutė, p = index & 1). Visos reikšmės
// legalios: Y 64..940, C 64..960. Rastras ir grain'o dažnis — GenParams (numatytieji = 1080i50, turinys tada bitas į bitą
// toks pat kaip prieš P2: gradientai, juosta ir blyksnis skaičiuojami nuo pločio / aukščio, blyksnis — kartą per sekundę).
#include <cstdint>

#include "common/hd.hpp"
#include "media/v210.hpp"

namespace mxlv::media {

MXLV_HD std::uint32_t fmix32(std::uint32_t x) {
    x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    return x;
}

MXLV_HD int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Sveikoji dalyba iš konstantos daugyba (be dalybos instrukcijos kiekvienam pikseliui): floor(n / d) == (n · m) >> 40,
// m = ceil(2^40 / d). Tikslu, kai n · d < 2^40 (čia n ≤ 1,1·10^7, d ≤ 12 512) — tikrinama test_media_format.
constexpr std::uint64_t div_magic(std::uint32_t d) { return ((std::uint64_t{1} << 40) + d - 1) / d; }
MXLV_HD std::uint32_t div_m(std::uint32_t n, std::uint64_t m) { return static_cast<std::uint32_t>((n * m) >> 40); }

struct GenParams {
    std::uint64_t run_seed;
    std::uint32_t flow;
    std::uint32_t complexity;  // 0 statinis … 3 sunkus koderiui
    int width = 1920, height = 1080;              // kadro rastras (keisti tik per set_raster)
    std::uint32_t rate_num = 50, rate_den = 1;    // grain'o dažnis (sync blyksnio periodui)
    std::uint64_t m_grad = div_magic(1920 + 1080), m_x = div_magic(1920), m_y = div_magic(1080);
    // grain'o konstantos (prepare_index): blyksnis ir juostos x — skaičiuojami kartą grain'ui, ne pikseliui
    std::uint64_t prep_index = ~std::uint64_t{0};
    bool prep_flash = false;
    int prep_bar = 0;
};
// Rastras ir jo dalikliai (vieną kartą grain'ui, ne pikseliui).
MXLV_HD void set_raster(GenParams& g, int w, int h) {
    g.width = w;
    g.height = h;
    g.m_grad = ((std::uint64_t{1} << 40) + static_cast<std::uint32_t>(w + h) - 1) / static_cast<std::uint32_t>(w + h);
    g.m_x = ((std::uint64_t{1} << 40) + static_cast<std::uint32_t>(w) - 1) / static_cast<std::uint32_t>(w);
    g.m_y = ((std::uint64_t{1} << 40) + static_cast<std::uint32_t>(h) - 1) / static_cast<std::uint32_t>(h);
}

// Sync blyksnio grain'as: grain'as [index·den/num, (index+1)·den/num) s apima sveiką sekundę k (ten prasideda 1 s garso
// pyptelėjimas, s % 48000 == 0). 50/1: index % 50 == 0; 25/1: index % 25 == 0; 59,94: kas 59–60 grain'ų.
MXLV_HD bool flash_grain(const GenParams& g, std::uint64_t index) {
    const std::uint64_t num = g.rate_num, den = g.rate_den;
    const std::uint64_t k = (index * den + num - 1) / num;  // ceil
    return k * num < (index + 1) * den;
}
MXLV_HD int bar_x(const GenParams& g, std::uint64_t index) { return static_cast<int>((index * 8) % static_cast<std::uint64_t>(g.width)); }
// Grain'o konstantos į g (kviesti po set_raster, prieš pikselių ciklą). Be jo gen_y / gen_c skaičiuoja tą patį kiekvienam pikseliui.
MXLV_HD void prepare_index(GenParams& g, std::uint64_t index) {
    g.prep_index = index;
    g.prep_flash = flash_grain(g, index);
    g.prep_bar = bar_x(g, index);
}
// Blyksnio stačiakampis: x [w·1600/1920, w·1856/1920), lauko eilutės j 32..95.
MXLV_HD bool flash_at(const GenParams& g, std::uint64_t index, int x, int j) {
    if (j < 32 || j >= 96 || x < g.width * 1600 / 1920 || x >= g.width * 1856 / 1920) return false;
    return g.prep_index == index ? g.prep_flash : flash_grain(g, index);
}

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
    // sync blyksnis: kartą per sekundę (flash_grain), x [w·1600/1920, w·1856/1920), lauko eil. 32..95
    if (flash_at(g, index, x, j)) return 940;
    int y = 64 + static_cast<int>(div_m(static_cast<std::uint32_t>((x + fy) * 876), g.m_grad));  // įstrižas gradientas per visą rastrą
    const int bar = g.prep_index == index ? g.prep_bar : bar_x(g, index);
    int dx = x - bar;  // x, bar ∈ [0, w): (x − bar) mod w be dalybos
    if (dx < 0) dx += g.width;
    if (dx < 16) y = 900;
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
    if (j < 16 || flash_at(g, index, x, j)) {
        cb = 512;
        cr = 512;
        return;
    }
    int b = 412 + static_cast<int>(div_m(static_cast<std::uint32_t>(x * 200), g.m_x));
    int r = 412 + static_cast<int>(div_m(static_cast<std::uint32_t>(fy * 200), g.m_y));
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
