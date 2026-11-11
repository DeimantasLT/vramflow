// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// gfx_content: deterministinis, visi a8 kiekvienoje eilutėje, k lyginis straight / nelyginis premultiplied, kinta su kadru / n / k.
#include <catch2/catch_test_macros.hpp>

#include <bitset>
#include <vector>

#include "gfx/gfx_content.hpp"

using vramflow::gfx::gfx_content;

namespace {
std::vector<std::uint8_t> make(std::uint64_t seed, std::uint32_t n, std::uint32_t k, std::uint64_t fr, int w = 384, int h = 40) {
    std::vector<std::uint8_t> v(static_cast<std::size_t>(w) * h * 4);
    gfx_content(seed, n, k, fr, w, h, v.data());
    return v;
}
}  // namespace

TEST_CASE("gfx_content: kiekvienoje eilutėje yra visos a8 reikšmės 0…255", "[gfx][content]") {
    for (std::uint32_t k : {0u, 1u})
        for (int w : {256, 384, 1920}) {
            const auto v = make(7, 3, k, 12345, w, 6);
            for (int y = 0; y < 6; ++y) {
                std::bitset<256> seen;
                for (int x = 0; x < w; ++x) seen.set(v[(static_cast<std::size_t>(y) * w + x) * 4 + 3]);
                CHECK(seen.all());
            }
        }
}

TEST_CASE("gfx_content: nelyginis k premultiplied (RGB <= a), lyginis straight (kai kur RGB > a)", "[gfx][content]") {
    const auto odd = make(1, 0, 1, 5), even = make(1, 0, 2, 5);
    bool straight_over = false;
    for (std::size_t i = 0; i < odd.size(); i += 4) {
        const std::uint8_t a = odd[i + 3];
        CHECK(odd[i] <= a);
        CHECK(odd[i + 1] <= a);
        CHECK(odd[i + 2] <= a);
        const std::uint8_t ae = even[i + 3];
        straight_over |= even[i] > ae || even[i + 1] > ae || even[i + 2] > ae;
    }
    CHECK(straight_over);
}

TEST_CASE("gfx_content: deterministinis ir kinta su seed / n / k / kadru", "[gfx][content]") {
    const auto a = make(1, 2, 3, 4);
    CHECK(a == make(1, 2, 3, 4));
    CHECK(a != make(2, 2, 3, 4));
    CHECK(a != make(1, 3, 3, 4));
    CHECK(a != make(1, 2, 5, 4));  // tas pats paritetas, kitas k
    CHECK(a != make(1, 2, 3, 5));
    CHECK(make(1, 2, 3, 4, 384, 40) == make(1, 2, 3, 4, 384, 40));
}
