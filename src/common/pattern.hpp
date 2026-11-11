// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Deterministinis grain'o turinys (architektūra §4.2). Tas pats kodas veikia GPU ir CPU.
#include <cstdint>

#include "hd.hpp"

namespace mxlv {

constexpr std::uint32_t kHeaderWords = 16;  // pirmi 64 B — GrainHeader

MXLV_HD std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Žodis i (i ≥ 16) grain'e `index` flow'ui su `seed`. Bitai 30–31 = 0, todėl žodis legalus v210.
MXLV_HD std::uint32_t word(std::uint64_t seed, std::uint64_t index, std::uint32_t i) {
    std::uint64_t k = seed ^ (index * 0x9E3779B97F4A7C15ull);
    std::uint32_t x = static_cast<std::uint32_t>(k ^ (k >> 32)) + i * 0x9E3779B1u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x & 0x3FFFFFFFu;
}

// Vieno žodžio indėlis į fold64 (priklauso nuo pozicijos; suma mod 2^64 nepriklauso nuo tvarkos).
MXLV_HD std::uint64_t fold_term(std::uint32_t w, std::uint32_t i) {
    return static_cast<std::uint64_t>(w) * (2ull * i + 1ull);
}

// Etaloninė (skaliarinė) fold64 — CPU kryžminei patikrai ir testams.
inline std::uint64_t fold64_reference(std::uint64_t seed, std::uint64_t index, std::uint32_t n) {
    std::uint64_t s = 0;
    for (std::uint32_t i = kHeaderWords; i < n; ++i) s += fold_term(word(seed, index, i), i);
    return s;
}

// fold64 iš faktinių žodžių (CPU kryžminei patikrai).
inline std::uint64_t fold64_of(const std::uint32_t* words, std::uint32_t n) {
    std::uint64_t s = 0;
    for (std::uint32_t i = kHeaderWords; i < n; ++i) s += fold_term(words[i], i);
    return s;
}

}  // namespace mxlv
