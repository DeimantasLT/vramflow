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

// --- Turinys bet kokiam rastrui ir dažniui (peržiūros pastaba 4) ------------------------------------------------------
TEST_CASE("gen: sync blyksnis kartą per sekundę bet kuriuo grain'o dažniu (sutampa su 1 s garso pyptelėjimu)", "[media][format][gen]") {
    struct R {
        std::uint32_t num, den;
    };
    for (const R r : {R{50, 1}, R{25, 1}, R{60000, 1001}, R{30000, 1001}}) {
        GenParams g{1, 0, 0};
        g.rate_num = r.num;
        g.rate_den = r.den;
        std::uint64_t flashes = 0, bad = 0, prev = 0;
        const std::uint64_t n = 600ull * r.num / r.den;  // 600 s
        for (std::uint64_t i = 0; i < n; ++i) {
            if (!flash_grain(g, i)) continue;
            // grain'as [i·den/num, (i+1)·den/num) s turi sveiką sekundę (garso pyptelėjimo pradžią)
            const std::uint64_t k = (i * r.den + r.num - 1) / r.num;
            bad += !(k * r.num < (i + 1) * r.den);
            if (flashes && (i - prev) * r.den * 10 < 9ull * r.num) ++bad;  // ne dažniau nei ~kas sekundę
            prev = i;
            ++flashes;
        }
        INFO(r.num << "/" << r.den);
        CHECK(flashes == 600);
        CHECK(bad == 0);
    }
    GenParams g50{1, 0, 0};
    for (std::uint64_t i = 0; i < 100'000; ++i) CHECK((flash_grain(g50, i) == (i % 50 == 0)) == true);
}

TEST_CASE("gen: blyksnis ir gradientas pagal rastrą (720p blyksnis matomas, UHD neprisotintas)", "[media][format][gen]") {
    // 720p50: blyksnio grain'e luma == 940 blyksnio stačiakampyje, kuris telpa į plotį
    const Dim d720{1280, 720};
    const int w720 = v210_words_per_line(d720.w);
    std::vector<std::uint32_t> f(static_cast<std::size_t>(w720) * 720);
    GenParams g{5, 1, 0};
    ref::gen_field(g, 100, d720, false, f.data());  // 100 % 50 == 0 — blyksnis
    int lit = 0;
    for (int r = 64; r < 192; ++r)
        for (int gi = 0; gi < 213; ++gi) {
            const Group gr = group_at(f, w720, r, gi);
            for (int q = 0; q < 6; ++q) lit += gr.y[q] == 940;
        }
    CHECK(lit > 128 * 150);
    // UHD, complexity 0, ne blyksnio grain'as: už barkodo ir judančios juostos luma niekada neprisotinta (940)
    const Dim d4k{3840, 2160};
    const int w4k = v210_words_per_line(d4k.w);
    std::vector<std::uint32_t> u(static_cast<std::size_t>(w4k) * 2160);
    ref::gen_field(g, 7, d4k, false, u.data());
    int sat = 0;
    for (int r = 64; r < 2160; r += 7)
        for (int gi = 0; gi < 640; ++gi) {
            const Group gr = group_at(u, w4k, r, gi);
            for (int q = 0; q < 6; ++q) sat += gr.y[q] == 940;
        }
    CHECK(sat == 0);
}

TEST_CASE("gen: dalyba daugyba (div_magic) tiksli visiems naudojamiems skaitikliams ir dalikliams", "[media][format][gen]") {
    std::uint64_t bad = 0;
    for (const std::uint32_t d : {1080u, 1920u, 720u, 1280u, 2160u, 3840u, 2000u, 3000u, 6000u, 8192u + 4320u})
        for (std::uint32_t n = 0; n <= (8192u + 4320u) * 876u; n += (n < 4'000'000u ? 1u : 7u))
            bad += div_m(n, div_magic(d)) != n / d;
    CHECK(bad == 0);
}
