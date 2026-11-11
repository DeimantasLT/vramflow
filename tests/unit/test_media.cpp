// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Media CPU testai be GPU (architektūra GPU funkcijų §6.6): v210, spalvos, blend, weave/split vektoriai, barkodas.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "media/gen_content.hpp"
#include "media/media_ref.hpp"
#include "media/v210.hpp"

using namespace mxlv::media;

namespace {

// Lauko pikselio Y / poros chroma per v210.hpp (testams; etalonas media_ref.cpp turi savo nepriklausomą prieigą).
Group group_at(const std::vector<std::uint32_t>& f, int j, int g) {
    Group gr;
    unpack_group(f.data() + static_cast<std::size_t>(j) * kWordsPerLine + 4 * g, gr);
    return gr;
}
int field_y(const std::vector<std::uint32_t>& f, int j, int x) { return group_at(f, j, x / 6).y[x % 6]; }
int field_cb(const std::vector<std::uint32_t>& f, int j, int i) { return group_at(f, j, i / 3).cb[i % 3]; }
int field_cr(const std::vector<std::uint32_t>& f, int j, int i) { return group_at(f, j, i / 3).cr[i % 3]; }

// Laukas, kurio eilutė j užpildoma fn(j, g) grupe.
template <class Fn>
std::vector<std::uint32_t> make_field(Fn fn) {
    std::vector<std::uint32_t> f(kFieldWords);
    for (int j = 0; j < kFieldHeight; ++j)
        for (int g = 0; g < kGroupsPerLine; ++g) {
            const Group gr = fn(j, g);
            pack_group(gr, f.data() + static_cast<std::size_t>(j) * kWordsPerLine + 4 * g);
        }
    return f;
}

Group uniform_group(int y, int cb, int cr) {
    Group g;
    for (auto& v : g.y) v = static_cast<std::uint16_t>(y);
    for (auto& v : g.cb) v = static_cast<std::uint16_t>(cb);
    for (auto& v : g.cr) v = static_cast<std::uint16_t>(cr);
    return g;
}

// 2×2 RGBA ir 2×1 RGB PNG (sugeneruoti python zlib; vienkartinis testinis turinys)
const unsigned char kPngRgba[] = {0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x08,0x06,0x00,0x00,0x00,0x72,0xB6,0x0D,0x24,0x00,0x00,0x00,0x16,0x49,0x44,0x41,0x54,0x78,0x9C,0x63,0xF8,0xCF,0xC0,0xF0,0x1F,0x08,0x1B,0x18,0x40,0x34,0x97,0x88,0x9C,0x06,0x00,0x36,0xA9,0x04,0xE1,0x66,0x07,0x05,0x49,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,0x82};
const unsigned char kPngRgb[] = {0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x01,0x08,0x02,0x00,0x00,0x00,0x7B,0x40,0xE8,0xDD,0x00,0x00,0x00,0x0F,0x49,0x44,0x41,0x54,0x78,0x9C,0x63,0x60,0x64,0x62,0xFE,0xF5,0xFB,0x0F,0x00,0x06,0x03,0x02,0xF8,0x4E,0x96,0xAD,0xB6,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,0x82};

std::filesystem::path write_tmp(const char* name, const unsigned char* d, std::size_t n) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(n));
    return p;
}

}  // namespace

TEST_CASE("v210: §6.1 vektorius", "[media][v210]") {
    Group g{};
    g.cb[0] = 1; g.y[0] = 2; g.cr[0] = 3; g.y[1] = 4; g.cb[1] = 5; g.y[2] = 6;
    g.cr[1] = 7; g.y[3] = 8; g.cb[2] = 9; g.y[4] = 10; g.cr[2] = 11; g.y[5] = 12;
    std::uint32_t w[4];
    pack_group(g, w);
    CHECK(w[0] == 0x00300801u);
    CHECK(w[1] == 0x00601404u);
    CHECK(w[2] == 0x00902007u);
    CHECK(w[3] == 0x00C02C0Au);
    Group u{};
    unpack_group(w, u);
    for (int i = 0; i < 6; ++i) CHECK(u.y[i] == g.y[i]);
    for (int i = 0; i < 3; ++i) {
        CHECK(u.cb[i] == g.cb[i]);
        CHECK(u.cr[i] == g.cr[i]);
    }
}

TEST_CASE("v210: round-trip 1e5 atsitiktinių grupių, bitai 30–31 išsaugomi", "[media][v210]") {
    std::mt19937_64 rng(0xC0FFEEu);
    int bad_words = 0, bad_values = 0, bad_hi = 0;
    for (int it = 0; it < 100000; ++it) {
        // (a) bet kokie 32 bitų žodžiai → unpack → pack su hi = šaltinis → identiška
        std::uint32_t w[4], o[4];
        for (auto& x : w) x = static_cast<std::uint32_t>(rng());
        Group g;
        unpack_group(w, g);
        pack_group(g, o, w);
        for (int i = 0; i < 4; ++i) bad_words += o[i] != w[i];
        // (b) atsitiktinės 10 bitų reikšmės → pack (be hi) → unpack → tos pačios, bitai 30–31 = 0
        Group a;
        for (auto& v : a.y) v = static_cast<std::uint16_t>(rng() & 0x3FF);
        for (auto& v : a.cb) v = static_cast<std::uint16_t>(rng() & 0x3FF);
        for (auto& v : a.cr) v = static_cast<std::uint16_t>(rng() & 0x3FF);
        pack_group(a, o);
        for (int i = 0; i < 4; ++i) bad_hi += (o[i] & 0xC0000000u) != 0;
        Group b;
        unpack_group(o, b);
        for (int i = 0; i < 6; ++i) bad_values += a.y[i] != b.y[i];
        for (int i = 0; i < 3; ++i) bad_values += (a.cb[i] != b.cb[i]) + (a.cr[i] != b.cr[i]);
    }
    CHECK(bad_words == 0);
    CHECK(bad_values == 0);
    CHECK(bad_hi == 0);
}

TEST_CASE("colour: §6.3 auksinės reikšmės ir RGB kubas legalus", "[media][colour]") {
    struct G { int r, g, b, y, cb, cr; };
    const G gold[] = {{255, 255, 255, 940, 512, 512}, {0, 0, 0, 64, 512, 512}, {255, 0, 0, 250, 409, 960},
                      {0, 255, 0, 691, 167, 105},     {0, 0, 255, 127, 960, 471}};
    for (const auto& c : gold) {
        int y, cb, cr;
        ref::rgb_to_ycbcr10(c.r, c.g, c.b, y, cb, cr);
        INFO("RGB " << c.r << "," << c.g << "," << c.b);
        CHECK(y == c.y);
        CHECK(cb == c.cb);
        CHECK(cr == c.cr);
    }
    int illegal = 0;
    for (int r = 0; r <= 255; r += 17)
        for (int g = 0; g <= 255; g += 17)
            for (int b = 0; b <= 255; b += 17) {
                int y, cb, cr;
                ref::rgb_to_ycbcr10(r, g, b, y, cb, cr);
                illegal += y < 64 || y > 940 || cb < 64 || cb > 960 || cr < 64 || cr > 960;
            }
    CHECK(illegal == 0);
}

TEST_CASE("blend: §6.3 vektoriai ir a=0 / a=255 tapatybės", "[media][blend]") {
    CHECK(ref::blend(64, 940, 128) == 504);
    CHECK(ref::blend(1023, 4, 1) == 1019);
    for (int a = 0; a <= 255; ++a) CHECK(ref::blend(512, 512, a) == 512);
    for (int s : {4, 64, 512, 940, 1019})
        for (int l : {4, 64, 512, 940, 1019, 1023}) {
            CHECK(ref::blend(s, l, 0) == s);
            CHECK(ref::blend(s, l, 255) == l);
        }
}

TEST_CASE("weave (ref): §6.2 chroma ir luma vektoriai", "[media][weave]") {
    // abu laukai: lyginės lauko eilutės C=400, nelyginės C=600; Y grupėje 1023, 64, 940, ...
    auto fill = [](int j, int) {
        Group g = uniform_group(940, j % 2 ? 600 : 400, j % 2 ? 600 : 400);
        g.y[0] = 1023; g.y[1] = 64; g.y[2] = 940;
        return g;
    };
    const auto top = make_field(fill), bot = make_field(fill);
    std::vector<std::uint8_t> y(1920 * 1080), uv(1920 * 540);
    ref::weave_nv12(top.data(), bot.data(), y.data(), uv.data(), 1920, 1920);
    int bad = 0;
    for (int u = 0; u < 540; ++u)
        for (int i = 0; i < 1920; ++i) bad += uv[u * 1920 + i] != (u % 2 == 0 ? 113 : 138);
    CHECK(bad == 0);
    bad = 0;
    for (int r = 0; r < 1080; ++r)
        for (int x = 0; x < 1920; ++x) {
            const int e = x % 6 == 0 ? 255 : (x % 6 == 1 ? 16 : 235);
            bad += y[r * 1920 + x] != e;
        }
    CHECK(bad == 0);
}

TEST_CASE("split (ref): §6.4 vektoriai, Y10 = Y8<<2, pastovi chroma 128 → 512", "[media][split]") {
    const int pitch = 2048;
    std::vector<std::uint8_t> y(pitch * 1080), uv(pitch * 540);
    for (int r = 0; r < 1080; ++r)
        for (int x = 0; x < 1920; ++x) y[r * pitch + x] = static_cast<std::uint8_t>((x + r) & 0xFF);
    // kiekvieno lauko c[0]=100, c[1]=140, c[2..]=180
    for (int u = 0; u < 540; ++u) {
        const int k = u / 2;
        const int v = k == 0 ? 100 : (k == 1 ? 140 : 180);
        for (int i = 0; i < 1920; ++i) uv[u * pitch + i] = static_cast<std::uint8_t>(v);
    }
    std::vector<std::uint32_t> f(kFieldWords);
    for (int p = 0; p < 2; ++p) {
        ref::split_nv12(y.data(), uv.data(), pitch, pitch, p, f.data());
        // j = 2 (k=1, lyginis), j = 3 (k=1, nelyginis)
        const int e2 = p == 0 ? 540 : 500, e3 = p == 0 ? 620 : 580;
        int bad = 0;
        for (int i = 0; i < 960; ++i) {
            bad += field_cb(f, 2, i) != e2 || field_cr(f, 2, i) != e2;
            bad += field_cb(f, 3, i) != e3 || field_cr(f, 3, i) != e3;
            bad += field_cb(f, 0, i) != 400;  // kraštas: c[-1] → c[0]: 8·100 → 400
        }
        CHECK(bad == 0);
        bad = 0;
        for (int j = 0; j < 540; ++j)
            for (int x = 0; x < 1920; ++x) bad += field_y(f, j, x) != (((x + 2 * j + p) & 0xFF) << 2);
        CHECK(bad == 0);
    }
    std::fill(uv.begin(), uv.end(), std::uint8_t{128});
    for (int p = 0; p < 2; ++p) {
        ref::split_nv12(y.data(), uv.data(), pitch, pitch, p, f.data());
        int bad = 0;
        for (int j = 0; j < 540; ++j)
            for (int i = 0; i < 960; ++i) bad += field_cb(f, j, i) != 512 || field_cr(f, j, i) != 512;
        CHECK(bad == 0);
    }
}

TEST_CASE("gen_field: visos reikšmės legalios, bitai 30–31 = 0", "[media][gen]") {
    for (std::uint32_t cx = 0; cx <= 3; ++cx)
        for (std::uint64_t index : {0ull, 1ull, 50ull, 12345ull, 0x1000000ull + 7}) {
            const GenParams g{0xABCDEF0123ull, 3, cx};
            std::vector<std::uint32_t> f(kFieldWords);
            ref::gen_field(g, index, f.data());
            int bad = 0;
            for (std::uint32_t i = 0; i < kFieldWords; ++i) bad += (f[i] & 0xC0000000u) != 0;
            for (int j = 0; j < 540; ++j)
                for (int gi = 0; gi < 320; ++gi) {
                    const Group gr = group_at(f, j, gi);
                    for (int q = 0; q < 6; ++q) bad += gr.y[q] < 64 || gr.y[q] > 940;
                    for (int c = 0; c < 3; ++c) bad += gr.cb[c] < 64 || gr.cb[c] > 960 || gr.cr[c] < 64 || gr.cr[c] > 960;
                }
            INFO("complexity " << cx << " index " << index);
            CHECK(bad == 0);
        }
}

TEST_CASE("marker: gen_field barkodas dekoduojasi (v210 ir NV12 po weave)", "[media][marker]") {
    std::vector<std::uint32_t> top(kFieldWords), bot(kFieldWords);
    std::vector<std::uint8_t> y(1920 * 1080), uv(1920 * 540);
    for (std::uint32_t flow : {0u, 5u, 127u, 200u})
        for (std::uint64_t F : {0ull, 1ull, 6172ull, 0x7FFFFFull, 0x123456789ull}) {
            const GenParams g{42, flow, static_cast<std::uint32_t>(F % 4)};
            ref::gen_field(g, 2 * F, top.data());
            ref::gen_field(g, 2 * F + 1, bot.data());
            INFO("flow " << flow << " F " << F);
            CHECK(ref::decode_marker(top.data()) == marker_word(2 * F, flow));
            CHECK(ref::decode_marker(bot.data()) == marker_word(2 * F + 1, flow));
            CHECK(((ref::decode_marker(top.data()) >> 24) & 1u) == 0u);
            CHECK(((ref::decode_marker(bot.data()) >> 24) & 1u) == 1u);
            ref::weave_nv12(top.data(), bot.data(), y.data(), uv.data(), 1920, 1920);
            CHECK(ref::decode_marker_nv12(y.data(), 1920, 0) == marker_word(2 * F, flow));
            CHECK(ref::decode_marker_nv12(y.data(), 1920, 1) == marker_word(2 * F + 1, flow));
        }
}

TEST_CASE("logo_from_rgba / builtin_logo / logo_field (ref)", "[media][logo]") {
    std::vector<std::uint8_t> rgba(3 * 1 * 4, 255);
    ref::Logo l;
    CHECK_FALSE(ref::logo_from_rgba(rgba.data(), 3, 1, l));  // nelyginis plotis

    // pora: raudonas a=255 + mėlynas a=0 → chroma = raudonos; abu a=0 → 512; Ac = (a0+a1+1)>>1
    const std::uint8_t px[] = {255, 0, 0,  255,  // raudonas a=255
                               0,   0, 255, 0,   // mėlynas a=0
                               0,   255, 0, 0,   // žalias a=0
                               255, 0, 0,  0,    // raudonas a=0
                               10,  10, 10, 1,   //
                               200, 30, 60, 2};
    REQUIRE(ref::logo_from_rgba(px, 6, 1, l));
    CHECK(l.ac8[0] == 128);
    CHECK(l.cb10[0] == 409);
    CHECK(l.cr10[0] == 960);
    CHECK(l.ac8[1] == 0);
    CHECK(l.cb10[1] == 512);
    CHECK(l.cr10[1] == 512);
    CHECK(l.ac8[2] == 2);  // (1+2+1)>>1
    CHECK(l.y10[0] == 250);
    CHECK(l.a8[3] == 0);

    const ref::Logo b1 = ref::builtin_logo(), b2 = ref::builtin_logo();
    REQUIRE(b1.w == 384);
    REQUIRE(b1.h == 216);
    CHECK(b1.y10 == b2.y10);
    CHECK(b1.a8 == b2.a8);
    CHECK(b1.cb10 == b2.cb10);
    int border = 0;
    bool seen[256] = {};
    for (int y = 0; y < 216; ++y)
        for (int x = 0; x < 384; ++x) {
            const int a = b1.a8[y * 384 + x];
            seen[a] = true;
            if (x < 8 || x >= 376 || y < 8 || y >= 208) border += a != 0;
        }
    CHECK(border == 0);
    int missing = 0;
    for (bool s : seen) missing += !s;
    CHECK(missing == 0);

    // a=0 visur → išėjimas == įėjimas (atsitiktiniai žodžiai, su bitais 30–31)
    std::mt19937 rng(7);
    std::vector<std::uint32_t> in(kFieldWords), out(kFieldWords);
    for (auto& w : in) w = rng();
    ref::Logo zero = b1;
    std::fill(zero.a8.begin(), zero.a8.end(), std::uint8_t{0});
    std::fill(zero.ac8.begin(), zero.ac8.end(), std::uint8_t{0});
    ref::logo_field(in.data(), out.data(), zero, 100, 51, 1);
    CHECK(in == out);
}

TEST_CASE("load_png_rgba: RGBA, RGB24 ir klaida", "[media][png]") {
    std::vector<std::uint8_t> rgba;
    int w = 0, h = 0;
    const auto p1 = write_tmp("mxlv_test_rgba.png", kPngRgba, sizeof kPngRgba);
    REQUIRE(ref::load_png_rgba(p1.string(), rgba, w, h) == "");
    CHECK(w == 2);
    CHECK(h == 2);
    CHECK(rgba == std::vector<std::uint8_t>{255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 10, 20, 30, 40});
    const auto p2 = write_tmp("mxlv_test_rgb.png", kPngRgb, sizeof kPngRgb);
    REQUIRE(ref::load_png_rgba(p2.string(), rgba, w, h) == "");
    CHECK(w == 2);
    CHECK(h == 1);
    CHECK(rgba == std::vector<std::uint8_t>{1, 2, 3, 255, 250, 251, 252, 255});
    std::filesystem::remove(p1);
    std::filesystem::remove(p2);
    CHECK_FALSE(ref::load_png_rgba("/nonexistent/mxlv_nera.png", rgba, w, h).empty());
}
