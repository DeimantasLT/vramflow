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

TEST_CASE("media: nauji keliai su kDim1080 == senas (auksinės CRC)", "[media][format][golden]") {
    const GenParams g{0x4C52544D584C4756ull, 3, 2};
    std::vector<std::uint32_t> top(kFieldWords), bot(kFieldWords), f(kFieldWords);
    ref::gen_field(g, 1000, kDim1080, true, top.data());
    ref::gen_field(g, 1001, kDim1080, true, bot.data());
    CHECK(crc_of(top) == 0x581da7a8u);
    CHECK(crc_of(bot) == 0x9addafb4u);
    std::vector<std::uint8_t> y(1920 * 1080), uv(1920 * 540);
    ref::weave_nv12(top.data(), bot.data(), y.data(), uv.data(), 1920, 1920, kDim1080, true);
    CHECK(crc_of(y) == 0x559611e0u);
    CHECK(crc_of(uv) == 0x82336c6au);
    const Nv12 src(1920, 1080, 2048, 77);
    ref::split_nv12(src.y.data(), src.uv.data(), src.pitch, src.pitch, kDim1080, 0, ChromaMode::FieldSited, f.data());
    CHECK(crc_of(f) == 0x6e68bc95u);
    ref::split_nv12(src.y.data(), src.uv.data(), src.pitch, src.pitch, kDim1080, 1, ChromaMode::FieldSited, f.data());
    CHECK(crc_of(f) == 0xb29193c0u);
    const ref::Logo logo = ref::builtin_logo();
    ref::logo_field(top.data(), f.data(), logo, 1488, 64, kDim1080, 0);
    CHECK(crc_of(f) == 0x078137ddu);
    ref::logo_field(bot.data(), f.data(), logo, 1488, 64, kDim1080, 1);
    CHECK(crc_of(f) == 0x5f601ab3u);
}

TEST_CASE("media ref: split kadro chroma — rankinės reikšmės", "[media][format][split]") {
    // 4 eilučių kadras: chromos eilės {100, 200} → luma eilės 0..3: 400, 500, 700, 800 (10 bitų, be apvalinimo)
    const Dim d{1920, 4};
    const int pitch = 2048;
    std::vector<std::uint8_t> y(pitch * 4, 50), uv(pitch * 2);
    for (int i = 0; i < pitch; ++i) {
        uv[i] = 100;
        uv[pitch + i] = 200;
    }
    const int words = v210_words_per_line(d.w);
    std::vector<std::uint32_t> f(static_cast<std::size_t>(words) * 4);
    ref::split_nv12_progressive(y.data(), uv.data(), pitch, pitch, d, f.data());
    const int want[4] = {400, 500, 700, 800};
    int bad = 0;
    for (int r = 0; r < 4; ++r)
        for (int g = 0; g < 320; ++g) {
            const Group gr = group_at(f, words, r, g);
            for (int c = 0; c < 3; ++c) bad += gr.cb[c] != want[r] || gr.cr[c] != want[r];
            for (int q = 0; q < 6; ++q) bad += gr.y[q] != 200;
        }
    CHECK(bad == 0);
    // interlaced flow, progresyviai pažymėtas kadras: laukas p ima kadro eilutes 2j + p su KADRO chroma
    std::vector<std::uint32_t> fl(static_cast<std::size_t>(words) * 2);
    for (int p = 0; p < 2; ++p) {
        ref::split_nv12(y.data(), uv.data(), pitch, pitch, d, p, ChromaMode::FrameSited, fl.data());
        bad = 0;
        for (int j = 0; j < 2; ++j)
            for (int g = 0; g < 320; ++g) bad += group_at(fl, words, j, g).cb[1] != want[2 * j + p];
        INFO("paritetas " << p);
        CHECK(bad == 0);
    }
}

TEST_CASE("media ref: progresyvus weave — chroma (c[2k] + c[2k+1] + 4) >> 3, luma kaip lauko", "[media][format][weave]") {
    const Dim d{1920, 4};
    const int words = v210_words_per_line(d.w);
    std::vector<std::uint32_t> fr(static_cast<std::size_t>(words) * 4);
    const int c10[4] = {400, 600, 1023, 1020};
    for (int r = 0; r < 4; ++r)
        for (int g = 0; g < 320; ++g) {
            Group gr;
            for (auto& v : gr.y) v = static_cast<std::uint16_t>(r == 0 ? 1023 : 940);
            for (auto& v : gr.cb) v = static_cast<std::uint16_t>(c10[r]);
            for (auto& v : gr.cr) v = static_cast<std::uint16_t>(c10[3 - r]);
            pack_group(gr, fr.data() + static_cast<std::size_t>(r) * words + 4 * g);
        }
    std::vector<std::uint8_t> y(1920 * 4), uv(1920 * 2);
    ref::weave_nv12(fr.data(), nullptr, y.data(), uv.data(), 1920, 1920, d, false);
    CHECK(int{uv[0]} == 125);          // (400 + 600 + 4) >> 3
    CHECK(int{uv[1]} == 255);          // (1020 + 1023 + 4) >> 3 = 255
    CHECK(int{uv[1920]} == 255);       // (1023 + 1020 + 4) >> 3
    CHECK(int{uv[1920 + 1]} == 125);
    CHECK(int{y[0]} == 255);           // (1023 + 2) >> 2 apkerpama
    CHECK(int{y[1920]} == 235);
}

TEST_CASE("media ref: 720p / 2160p geometrija — lygiavimo grupės nulinės, barkodas dekoduojasi", "[media][format][gen]") {
    for (const Dim d : {Dim{1280, 720}, Dim{1920, 1080}, Dim{3840, 2160}}) {
        const int words = v210_words_per_line(d.w);
        std::vector<std::uint32_t> f(static_cast<std::size_t>(words) * d.h, 0xFFFFFFFFu);
        for (std::uint64_t index : {0ull, 1ull, 77ull, 0x123456ull}) {
            ref::gen_field({9, 5, 3}, index, d, false, f.data());
            INFO("w " << d.w << " index " << index);
            CHECK(ref::decode_marker(f.data(), words) == marker_word(index, 5));
            int bad = 0;
            for (int r = 0; r < d.h; ++r)
                for (int g = 0; g < words / 4; ++g) {
                    const Group gr = group_at(f, words, r, g);
                    for (int q = 0; q < 6; ++q) {
                        const bool in = 6 * g + q < d.w;
                        bad += in ? (gr.y[q] < 64 || gr.y[q] > 940) : gr.y[q] != 0;
                    }
                    for (int c = 0; c < 3; ++c) {
                        const bool in = 6 * g + 2 * c < d.w;
                        bad += in ? (gr.cb[c] < 64 || gr.cb[c] > 960 || gr.cr[c] < 64 || gr.cr[c] > 960) : (gr.cb[c] | gr.cr[c]) != 0;
                    }
                    for (int k = 0; k < 4; ++k) bad += (f[static_cast<std::size_t>(r) * words + 4 * g + k] & 0xC0000000u) != 0;
                }
            CHECK(bad == 0);
        }
    }
    // progresyvaus grain'o eilutė r = kadro eilutė r (ne 2j + p)
    const Dim d{1920, 1080};
    std::vector<std::uint32_t> fr(static_cast<std::size_t>(1280) * 1080), top(kFieldWords), bot(kFieldWords);
    ref::gen_field({1, 2, 0}, 6, d, false, fr.data());
    ref::gen_field({1, 2, 0}, 6, d, true, top.data());  // laukas 0 to paties indekso: eilutės 0, 2, 4…
    int bad = 0;
    for (int j = 40; j < 540; ++j)
        for (int w = 0; w < 1280; ++w) bad += fr[static_cast<std::size_t>(2 * j) * 1280 + w] != top[static_cast<std::size_t>(j) * 1280 + w];
    CHECK(bad == 0);
}

TEST_CASE("media ref: logo progresyviame kadre ir 720p", "[media][format][logo]") {
    const ref::Logo logo = ref::builtin_logo();
    for (const Dim d : {Dim{1280, 720}, Dim{1920, 1080}}) {
        const int words = v210_words_per_line(d.w);
        std::vector<std::uint32_t> in(static_cast<std::size_t>(words) * d.h), out(in.size());
        ref::gen_field({3, 1, 2}, 10, d, false, in.data());
        const int X = d.w - logo.w - 16, Y = 40;
        ref::logo_field(in.data(), out.data(), logo, X, Y, d, -1);
        // už logotipo stačiakampio — bitas į bitą; viduje (nepermatoma vieta) — pakeista
        int outside = 0, changed = 0;
        for (int r = 0; r < d.h; ++r)
            for (int w = 0; w < words; ++w) {
                const std::size_t k = static_cast<std::size_t>(r) * words + w;
                const int x0 = (w / 4) * 6;
                const bool inside = r >= Y && r < Y + logo.h && x0 + 6 > X && x0 < X + logo.w;
                if (!inside) outside += in[k] != out[k];
                else changed += in[k] != out[k];
            }
        INFO("w " << d.w);
        CHECK(outside == 0);
        CHECK(changed > 1000);
    }
}
