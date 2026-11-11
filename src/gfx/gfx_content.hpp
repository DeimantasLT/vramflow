// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Deterministinis sintetinis RGBA turinys grafikos šaltiniams (gfxgen) ir jų patikrai (verify perskaičiuoja lauktą turinį iš
// to paties (seed, n, k, frame)). Header-only, pigus (be daugybos per pikselį, išskyrus premultiply).
//  - a8: kiekvienoje eilutėje slenkanti rampa (x + poslinkis) & 255, todėl kai w >= 256 VISOS reikšmės 0…255 pasikartoja
//    kiekvienoje eilutėje (alfa kvantavimo kraštai tikrinami visur); poslinkis priklauso nuo (seed, n, k, frame, y);
//  - k lyginis -> straight alfa; k nelyginis -> premultiplied (RGB <= a);
//  - RGB: gradientai su skirtingais žingsniais, fazė priklauso nuo (n, k, frame), todėl sumaišytas srautas / kadras
//    iškart matomas.
#include <cstdint>
#include <cstring>

namespace vramflow::gfx {

namespace detail {
constexpr std::uint64_t mix64(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
}  // namespace detail

// rgba: w * h * 4 B, stride = w * 4 (be tarpų). Baitų tvarka R, G, B, A.
inline void gfx_content(std::uint64_t seed, std::uint32_t n, std::uint32_t k, std::uint64_t frame, int w, int h, std::uint8_t* rgba) {
    const std::uint64_t base = detail::mix64(seed ^ detail::mix64((static_cast<std::uint64_t>(n) << 8 | k) + 0x5EEDull));
    const std::uint64_t fm = detail::mix64(base ^ (frame * 0xD1B54A32D192ED03ull));
    const bool premul = (k & 1u) != 0;
    const std::uint32_t pr = static_cast<std::uint32_t>(fm), pg = static_cast<std::uint32_t>(fm >> 16), pb = static_cast<std::uint32_t>(fm >> 32);
    const std::uint32_t sr = 1 + (k % 3) + 2 * (n % 2), sg = 2 + (n % 3), sb = 3 + (k % 2) * 2;  // gradientų žingsniai
    for (int y = 0; y < h; ++y) {
        const std::uint32_t yy = static_cast<std::uint32_t>(y);
        const std::uint32_t off = static_cast<std::uint32_t>(fm >> 8) + yy * 3u + static_cast<std::uint32_t>(base >> 40);
        std::uint8_t* row = rgba + static_cast<std::size_t>(y) * static_cast<std::size_t>(w) * 4;
        for (int x = 0; x < w; ++x) {
            const std::uint32_t xx = static_cast<std::uint32_t>(x);
            const std::uint32_t a = (xx + off) & 255u;
            std::uint32_t r = (xx * sr + yy + pr) & 255u, g = (xx * sg + yy * 2u + pg) & 255u, b = (xx * sb + 255u - yy + pb) & 255u;
            if (premul) {
                r = r * a / 255u;
                g = g * a / 255u;
                b = b * a / 255u;
            }
            const std::uint32_t px = r | (g << 8) | (b << 16) | (a << 24);  // LE: R, G, B, A
            std::memcpy(row + static_cast<std::size_t>(x) * 4, &px, 4);
        }
    }
}

}  // namespace vramflow::gfx
