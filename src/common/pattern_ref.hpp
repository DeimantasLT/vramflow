// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// NEPRIKLAUSOMA (skaliarinė, CPU) rašto implementacija CPU kryžminei patikrai. Sąmoningai parašyta kitaip
// nei pattern.hpp (kitokia aritmetikos tvarka, be bendrų pagalbinių funkcijų), kad klaida viename generatoriuje
// nepasikartotų kitame. Ekvivalentiškumą su pattern.hpp ir auksines reikšmes tikrina tests/unit.
#include <cstdint>

namespace mxlv::ref {

inline std::uint32_t word(std::uint64_t seed, std::uint64_t index, std::uint32_t i) {
    const std::uint64_t golden64 = 11400714819323198485ull;  // 0x9E3779B97F4A7C15
    const std::uint64_t k = seed ^ static_cast<std::uint64_t>(static_cast<unsigned __int128>(index) * golden64);
    const std::uint32_t lo = static_cast<std::uint32_t>(k & 0xFFFFFFFFull);
    const std::uint32_t hi = static_cast<std::uint32_t>(k >> 32);
    std::uint32_t x = (lo ^ hi) + static_cast<std::uint32_t>((static_cast<std::uint64_t>(i) * 2654435761ull) & 0xFFFFFFFFull);
    x = x ^ (x >> 16);
    x = static_cast<std::uint32_t>((static_cast<std::uint64_t>(x) * 2246822507ull) & 0xFFFFFFFFull);  // 0x85EBCA6B
    x = x ^ (x >> 13);
    x = static_cast<std::uint32_t>((static_cast<std::uint64_t>(x) * 3266489909ull) & 0xFFFFFFFFull);  // 0xC2B2AE35
    x = x ^ (x >> 16);
    return x % (1u << 30);
}

inline std::uint64_t fold64_of(const std::uint32_t* w, std::uint32_t n) {
    unsigned __int128 s = 0;
    for (std::uint32_t i = 16; i < n; ++i) s += static_cast<unsigned __int128>(w[i]) * (2u * static_cast<std::uint64_t>(i) + 1u);
    return static_cast<std::uint64_t>(s);  // mod 2^64
}

}  // namespace mxlv::ref
