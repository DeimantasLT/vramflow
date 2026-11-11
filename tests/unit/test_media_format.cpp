// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Formatui neutralūs media etalonai (planas §P2): senas 1080i kelias nepakito (CRC32C prieš pakeitimą), kadro chromos
// formulė rankinėmis reikšmėmis, matmenys parametrais (720p / 1080 / 2160, lygiavimo grupės = 0), progresyvus weave.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

#include "common/crc32c.hpp"
#include "media/gen_content.hpp"
#include "media/media_ref.hpp"
#include "media/v210.hpp"

using namespace mxlv::media;

namespace {

std::uint32_t crc_of(const std::vector<std::uint32_t>& v) {
    return mxlv::crc32c(reinterpret_cast<const std::uint8_t*>(v.data()), v.size() * 4);
}
std::uint32_t crc_of(const std::vector<std::uint8_t>& v) { return mxlv::crc32c(v.data(), v.size()); }

// Deterministinis NV12 kadras (pitch_y = pitch_uv = pitch).
struct Nv12 {
    int w, h, pitch;
    std::vector<std::uint8_t> y, uv;
    Nv12(int w_, int h_, int pitch_, std::uint32_t seed) : w(w_), h(h_), pitch(pitch_), y(pitch_ * h_), uv(pitch_ * (h_ / 2)) {
        std::mt19937 rng(seed);
        for (auto& b : y) b = static_cast<std::uint8_t>(16 + rng() % 220);
        for (auto& b : uv) b = static_cast<std::uint8_t>(16 + rng() % 225);
    }
};

Group group_at(const std::vector<std::uint32_t>& f, int words, int j, int g) {
    Group gr;
    unpack_group(f.data() + static_cast<std::size_t>(j) * words + 4 * g, gr);
    return gr;
}

}  // namespace

// Auksinės CRC32C reikšmės užfiksuotos SENU (1080i50, prieš P2) etalonu; nauji keliai su kDim1080 privalo jas atkartoti.
TEST_CASE("media: senas kelias nepakito", "[media][format][golden]") {
    const GenParams g{0x4C52544D584C4756ull, 3, 2};
    std::vector<std::uint32_t> top(kFieldWords), bot(kFieldWords), f(kFieldWords);
    ref::gen_field(g, 1000, top.data());
    ref::gen_field(g, 1001, bot.data());
    CHECK(crc_of(top) == 0x581da7a8u);
    CHECK(crc_of(bot) == 0x9addafb4u);
    std::vector<std::uint8_t> y(1920 * 1080), uv(1920 * 540);
    ref::weave_nv12(top.data(), bot.data(), y.data(), uv.data(), 1920, 1920);
    CHECK(crc_of(y) == 0x559611e0u);
    CHECK(crc_of(uv) == 0x82336c6au);
    const Nv12 src(1920, 1080, 2048, 77);
    ref::split_nv12(src.y.data(), src.uv.data(), src.pitch, src.pitch, 0, f.data());
    CHECK(crc_of(f) == 0x6e68bc95u);
    ref::split_nv12(src.y.data(), src.uv.data(), src.pitch, src.pitch, 1, f.data());
    CHECK(crc_of(f) == 0xb29193c0u);
    const ref::Logo logo = ref::builtin_logo();
    ref::logo_field(top.data(), f.data(), logo, 1488, 64, 0);
    CHECK(crc_of(f) == 0x078137ddu);
    ref::logo_field(bot.data(), f.data(), logo, 1488, 64, 1);
    CHECK(crc_of(f) == 0x5f601ab3u);
}
