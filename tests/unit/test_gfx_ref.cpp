// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Nepriklausomo etalono (gfx::ref) testai — alfa kompozitoriaus 3 užduotis. Laukiamos reikšmės skaičiuotos ranka
// (komentaruose); mėginių dekoderiai čia rašyti atskirai nuo bibliotekos (aiškios v210 / key bitų pozicijos).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gfx/gfx_ref.hpp"
#include "media/media_ref.hpp"

using namespace vramflow::gfx;
using Buf = std::vector<std::uint8_t>;

namespace {

std::uint32_t word(const Buf& b, std::size_t off) {
    std::uint32_t v;
    std::memcpy(&v, b.data() + off, 4);
    return v;
}

// v210 grupė (6 px, 16 B): w0 = Cb0|Y0|Cr0, w1 = Y1|Cb1|Y2, w2 = Cr1|Y3|Cb2, w3 = Y4|Cr2|Y5 (10 bitų laukai, postūmiai 0/10/20).
int fill_y(const Buf& g, int w, int row, int x) {
    static const int wi[6] = {0, 1, 1, 2, 3, 3}, sh[6] = {10, 0, 20, 10, 0, 20};
    const std::size_t o = static_cast<std::size_t>(row) * fill_line_bytes(w) + static_cast<std::size_t>(x / 6) * 16 + wi[x % 6] * 4;
    return (word(g, o) >> sh[x % 6]) & 0x3FF;
}
int fill_cb(const Buf& g, int w, int row, int i) {  // i — chroma indeksas (pikselių pora)
    static const int wi[3] = {0, 1, 2}, sh[3] = {0, 10, 20};
    const std::size_t o = static_cast<std::size_t>(row) * fill_line_bytes(w) + static_cast<std::size_t>(i / 3) * 16 + wi[i % 3] * 4;
    return (word(g, o) >> sh[i % 3]) & 0x3FF;
}
int fill_cr(const Buf& g, int w, int row, int i) {
    static const int wi[3] = {0, 2, 3}, sh[3] = {20, 0, 10};
    const std::size_t o = static_cast<std::size_t>(row) * fill_line_bytes(w) + static_cast<std::size_t>(i / 3) * 16 + wi[i % 3] * 4;
    return (word(g, o) >> sh[i % 3]) & 0x3FF;
}
int key_at(const Buf& g, const Raster& r, int row, int x) {
    const std::size_t o = fill_bytes(r) + static_cast<std::size_t>(row) * key_line_bytes(r.w) + static_cast<std::size_t>(x / 3) * 4;
    return (word(g, o) >> (10 * (x % 3))) & 0x3FF;
}

// Deterministinis RGBA kadras. alpha_mode: 0 = atsitiktinė 1..255, 1 = tik {0,255}, 2 = visada 255.
Buf make_rgba(int w, int h, std::uint32_t seed, int alpha_mode) {
    Buf b(static_cast<std::size_t>(w) * h * 4);
    std::uint32_t s = seed * 2654435761u + 1;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return static_cast<int>(s >> 24); };
    for (std::size_t i = 0; i < b.size(); i += 4) {
        b[i] = static_cast<std::uint8_t>(rnd());
        b[i + 1] = static_cast<std::uint8_t>(rnd());
        b[i + 2] = static_cast<std::uint8_t>(rnd());
        const int a = rnd();
        b[i + 3] = alpha_mode == 2 ? 255 : alpha_mode == 1 ? (a & 1 ? 255 : 0) : static_cast<std::uint8_t>(1 + a % 255);
    }
    return b;
}

Buf make_grain(int w, int h, const Raster& r, std::uint32_t seed, int alpha_mode, int parity = 0) {
    const Buf rgba = make_rgba(w, h, seed, alpha_mode);
    Buf g(grain_bytes(r), 0xEE);
    REQUIRE(ref::rgba_to_v210a(rgba.data(), w, h, static_cast<std::size_t>(w) * 4, AlphaIn::Straight, r, parity, g.data()));
    return g;
}

Buf run_composite(const Buf* base, const Raster& base_r, const std::vector<Layer>& L, const Raster& out_r, int parity = 0) {
    Buf out(grain_bytes(out_r), 0xA7);  // iš anksto purvinas
    REQUIRE(ref::composite(base ? base->data() : nullptr, base_r, L.data(), static_cast<int>(L.size()), out_r, parity, out.data()));
    return out;
}

Buf fill_only(const Buf& g, const Raster& r) { return Buf(g.begin(), g.begin() + static_cast<std::ptrdiff_t>(fill_bytes(r))); }

}  // namespace

TEST_CASE("gfx ref: RGBA -> key", "[gfx]") {
    // key = 64 + (a8·876 + 127)/255. a8=0: 64; a8=255: (223380+127)/255 = 876 -> 940; a8=128: (112128+127)/255 = 440 -> 504;
    // a8=1: (876+127)/255 = 3 -> 67.
    const Raster r{6, 2, false, true};
    Buf rgba(6 * 2 * 4, 0);
    const int a[6] = {0, 255, 128, 1, 254, 3};
    for (int x = 0; x < 6; ++x) rgba[x * 4 + 3] = static_cast<std::uint8_t>(a[x]);
    Buf g(grain_bytes(r), 0xFF);
    REQUIRE(ref::rgba_to_v210a(rgba.data(), 6, 2, 24, AlphaIn::Straight, r, 0, g.data()));
    CHECK(key_at(g, r, 0, 0) == 64);
    CHECK(key_at(g, r, 0, 1) == 940);
    CHECK(key_at(g, r, 0, 2) == 504);
    CHECK(key_at(g, r, 0, 3) == 67);
    CHECK(key_at(g, r, 0, 4) == 64 + (254 * 876 + 127) / 255);
    CHECK(key_at(g, r, 1, 0) == 64);  // antra eilutė: a8 = 0
}

TEST_CASE("gfx ref: premultiplied -> straight", "[gfx]") {
    // (200,100,50,a=128): c' = (c·255 + 64)/128: R (51000+64)/128 = 398 -> apkirpta 255; G (25500+64)/128 = 199; B (12750+64)/128 = 100.
    // Abu pikseliai vienodi, a0=a1=128: cb = (2·128·cb0 + 128)/256 = cb0 (tiksliai), todėl chroma = rgb_to_ycbcr10(255,199,100).
    const Raster r{2, 2, false, true};
    Buf rgba = {200, 100, 50, 128, 200, 100, 50, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    Buf g(grain_bytes(r), 0);
    REQUIRE(ref::rgba_to_v210a(rgba.data(), 2, 2, 8, AlphaIn::Premultiplied, r, 0, g.data()));
    int y, cb, cr;
    mxlv::media::ref::rgb_to_ycbcr10(255, 199, 100, y, cb, cr);
    CHECK(fill_y(g, 2, 0, 0) == y);
    CHECK(fill_y(g, 2, 0, 1) == y);
    CHECK(fill_cb(g, 2, 0, 0) == cb);
    CHECK(fill_cr(g, 2, 0, 0) == cr);
    CHECK(key_at(g, r, 0, 0) == 504);
    // a8 = 0 (premultiplied): c' = 0 -> juoda Y 64; chroma a0+a1 == 0 -> 512; key 64.
    CHECK(fill_y(g, 2, 1, 0) == 64);
    CHECK(fill_cb(g, 2, 1, 0) == 512);
    CHECK(fill_cr(g, 2, 1, 0) == 512);
    CHECK(key_at(g, r, 1, 0) == 64);
}

TEST_CASE("gfx ref: chroma pora svertinė alfa", "[gfx]") {
    // Pikseliai (255,0,0,a=3) ir (0,0,255,a=1): cb = (3·cb0 + 1·cb1 + 2)/4 su cbX iš rgb_to_ycbcr10.
    const Raster r{2, 1, false, true};
    Buf rgba = {255, 0, 0, 3, 0, 0, 255, 1};
    Buf g(grain_bytes(r), 0);
    REQUIRE(ref::rgba_to_v210a(rgba.data(), 2, 1, 8, AlphaIn::Straight, r, 0, g.data()));
    int y0, cb0, cr0, y1, cb1, cr1;
    mxlv::media::ref::rgb_to_ycbcr10(255, 0, 0, y0, cb0, cr0);
    mxlv::media::ref::rgb_to_ycbcr10(0, 0, 255, y1, cb1, cr1);
    CHECK(fill_cb(g, 2, 0, 0) == (3 * cb0 + cb1 + 2) / 4);
    CHECK(fill_cr(g, 2, 0, 0) == (3 * cr0 + cr1 + 2) / 4);
    CHECK(fill_y(g, 2, 0, 0) == y0);
    CHECK(fill_y(g, 2, 0, 1) == y1);
}

TEST_CASE("gfx ref: paritetas ima atitinkamas kadro eilutes", "[gfx]") {
    // Interlaced 4×8: pikselis eilutėje i = pilka (20·i + 10), a = 255. Paritetas 0 -> eilutės 0,2,4,6; 1 -> 1,3,5,7.
    const Raster r{4, 8, true, true};
    Buf rgba(4 * 8 * 4);
    for (int i = 0; i < 8; ++i)
        for (int x = 0; x < 4; ++x) {
            std::uint8_t* p = &rgba[(i * 4 + x) * 4];
            p[0] = p[1] = p[2] = static_cast<std::uint8_t>(20 * i + 10);
            p[3] = 255;
        }
    for (int parity = 0; parity < 2; ++parity) {
        Buf g(grain_bytes(r), 0);
        REQUIRE(ref::rgba_to_v210a(rgba.data(), 4, 8, 16, AlphaIn::Straight, r, parity, g.data()));
        for (int j = 0; j < 4; ++j) {
            int y, cb, cr;
            const int c = 20 * (2 * j + parity) + 10;
            mxlv::media::ref::rgb_to_ycbcr10(c, c, c, y, cb, cr);
            CHECK(fill_y(g, 4, j, 0) == y);
            CHECK(fill_y(g, 4, j, 3) == y);
            CHECK(fill_cb(g, 4, j, 1) == cb);
        }
    }
}

TEST_CASE("gfx ref: key ir fill padding = 0 purvname buferyje", "[gfx]") {
    for (int w : {382, 1280, 6, 2}) {
        const Raster r{w, 4, false, true};
        const Buf rgba = make_rgba(w, 4, 7, 0);
        Buf g(grain_bytes(r), 0xFF);
        REQUIRE(ref::rgba_to_v210a(rgba.data(), w, 4, static_cast<std::size_t>(w) * 4, AlphaIn::Straight, r, 0, g.data()));
        for (int row = 0; row < 4; ++row) {
            // key: žodis wi turi 3 mėginius; mėginiai >= w ir bitai 30–31 = 0, tikri mėginiai >= 64
            for (int wi = 0; wi < static_cast<int>(key_line_bytes(w) / 4); ++wi) {
                const std::uint32_t wd = word(g, fill_bytes(r) + static_cast<std::size_t>(row) * key_line_bytes(w) + static_cast<std::size_t>(wi) * 4);
                CHECK((wd >> 30) == 0);
                for (int s = 0; s < 3; ++s) {
                    if (wi * 3 + s >= w) CHECK(((wd >> (10 * s)) & 0x3FF) == 0);
                    else CHECK(((wd >> (10 * s)) & 0x3FF) >= 64);
                }
            }
            // fill: baitai už realių grupių (iki eilutės ilgio) = 0; paskutinės grupės nenaudoti mėginiai = 0; bitai 30–31 = 0
            const std::size_t used_groups = static_cast<std::size_t>((w + 5) / 6);
            const std::size_t base = static_cast<std::size_t>(row) * fill_line_bytes(w);
            for (std::size_t b = used_groups * 16; b < fill_line_bytes(w); ++b) CHECK(g[base + b] == 0);
            for (int x = w; x < static_cast<int>(used_groups) * 6; ++x) CHECK(fill_y(g, w, row, x) == 0);
            for (int i = w / 2; i < static_cast<int>(used_groups) * 3; ++i) {
                CHECK(fill_cb(g, w, row, i) == 0);
                CHECK(fill_cr(g, w, row, i) == 0);
            }
            for (std::size_t wi = 0; wi < fill_line_bytes(w) / 4; ++wi) CHECK((word(g, base + wi * 4) >> 30) == 0);
        }
    }
}

TEST_CASE("gfx ref: composite opacity 0 = bazė", "[gfx]") {
    const Raster br{96, 8, false, true};
    const Raster lr{48, 4, false, true};
    const Buf base = make_grain(96, 8, br, 1, 0);
    const Buf lay = make_grain(48, 4, lr, 2, 0);
    const std::vector<Layer> L = {{lay.data(), lr, 10, 2, 0.0f}};
    // v210a -> v210a: ab = key-64, ae = 0 -> n = ab·A, ao = (ab·A + 438)/A = ab; fill = (b·ab·A + n/2)/n = b; key = ab + 64.
    CHECK(run_composite(&base, br, L, br) == base);
    // v210a -> v210: ae = 0 -> (b·A + 438)/A = b; tik fill plokštuma
    Raster vr = br;
    vr.alpha = false;
    CHECK(run_composite(&base, br, L, vr) == fill_only(base, br));
}

TEST_CASE("gfx ref: sluoksnis a = A -> išėjimas = sluoksnio fill", "[gfx]") {
    const Raster outr{96, 8, false, false};
    const Raster brv{96, 8, false, true};
    const Raster lr{48, 4, false, true};
    const Buf base = make_grain(96, 8, brv, 3, 0);
    const Buf lay = make_grain(48, 4, lr, 4, 2);  // visi a8 = 255 -> a = 876 -> ae = 876 (opacity 1)
    const std::vector<Layer> L = {{lay.data(), lr, 20, 2, 1.0f}};
    // v210 išėjimas: (b·0 + f·876 + 438)/876 = f.
    const Buf o = run_composite(&base, brv, L, outr);
    for (int r = 0; r < 8; ++r)
        for (int x = 0; x < 96; ++x) {
            const bool in = x >= 20 && x < 68 && r >= 2 && r < 6;
            CHECK(fill_y(o, 96, r, x) == (in ? fill_y(lay, 48, r - 2, x - 20) : fill_y(base, 96, r, x)));
            if (x % 2 == 0) {
                CHECK(fill_cb(o, 96, r, x / 2) == (in ? fill_cb(lay, 48, r - 2, (x - 20) / 2) : fill_cb(base, 96, r, x / 2)));
                CHECK(fill_cr(o, 96, r, x / 2) == (in ? fill_cr(lay, 48, r - 2, (x - 20) / 2) : fill_cr(base, 96, r, x / 2)));
            }
        }
}

TEST_CASE("gfx ref: v210a + base none, vienas sluoksnis = sluoksnis", "[gfx]") {
    // Alfa 1..255 visur (ae >= 3 > 0). ab = 0: n = ae·A, ao = (ae·A + 438)/876 = ae -> key = ae + 64 = sluoksnio key (opacity 1: o8 = 255,
    // ae = (a·255 + 127)/255 = a). fill = (f·n + n/2)/n = f (n/2 < n), chroma tas pat (ac >= 3).
    const Raster lr{96, 8, false, true};
    const Buf lay = make_grain(96, 8, lr, 5, 0);
    CHECK(run_composite(nullptr, lr, {{lay.data(), lr, 0, 0, 1.0f}}, lr) == lay);
}

TEST_CASE("gfx ref: permatomas taškas su base none -> Y 64, C 512, key 64", "[gfx]") {
    const Raster lr{2, 1, false, true};
    Buf rgba = {255, 255, 255, 0, 255, 255, 255, 0};
    Buf lay(grain_bytes(lr));
    REQUIRE(ref::rgba_to_v210a(rgba.data(), 2, 1, 8, AlphaIn::Straight, lr, 0, lay.data()));
    const Buf o = run_composite(nullptr, lr, {{lay.data(), lr, 0, 0, 1.0f}}, lr);
    CHECK(fill_y(o, 2, 0, 0) == 64);
    CHECK(fill_y(o, 2, 0, 1) == 64);
    CHECK(fill_cb(o, 2, 0, 0) == 512);
    CHECK(fill_cr(o, 2, 0, 0) == 512);
    CHECK(key_at(o, lr, 0, 0) == 64);
}

TEST_CASE("gfx ref: du sluoksniai ranka (R8 kaupiama alfa)", "[gfx]") {
    // Plokšti 2×1 sluoksniai: L1 balta (Y 940, C 512), L2 juoda (Y 64, C 512), abu a8 = 255 su opacity 0.5 -> o8 = lround(127.5) = 128,
    // ae = (876·128 + 127)/255 = 440. Bazė none (ab = 0, Y 64).
    // Po L1: n = 440·876 = 385 440; ao = (385 440 + 438)/876 = 440; Y = (940·385 440 + 192 720)/385 440 = 940.
    // Po L2 (ab = 440): n = 440·876 + 440·436 = 577 280; ao = (577 280 + 438)/876 = 659 (876·659 = 577 284 <= 577 718 < 578 160);
    //   Y = (64·385 440 + 940·440·436 + 288 640)/577 280 = 205 286 400/577 280 = 355 (577 280·355 = 204 934 400; liekana < n).
    // key = 659 + 64 = 723. Chroma: ac = (440+440+1)>>1 = 440, abc = 440 (po L1) -> viskas 512: (512·n + n/2)/n = 512.
    const Raster lr{2, 1, false, true};
    Buf white = {255, 255, 255, 255, 255, 255, 255, 255}, black = {0, 0, 0, 255, 0, 0, 0, 255};
    Buf g1(grain_bytes(lr)), g2(grain_bytes(lr));
    REQUIRE(ref::rgba_to_v210a(white.data(), 2, 1, 8, AlphaIn::Straight, lr, 0, g1.data()));
    REQUIRE(ref::rgba_to_v210a(black.data(), 2, 1, 8, AlphaIn::Straight, lr, 0, g2.data()));
    const Buf o = run_composite(nullptr, lr, {{g1.data(), lr, 0, 0, 0.5f}, {g2.data(), lr, 0, 0, 0.5f}}, lr);
    CHECK(fill_y(o, 2, 0, 0) == 355);
    CHECK(fill_y(o, 2, 0, 1) == 355);
    CHECK(key_at(o, lr, 0, 0) == 723);
    CHECK(fill_cb(o, 2, 0, 0) == 512);
    CHECK(fill_cr(o, 2, 0, 0) == 512);
}

TEST_CASE("gfx ref: grandinė su dvejetaine alfa lygi tiesioginiam", "[gfx]") {
    for (bool interlaced : {false, true}) {
        const Raster br{96, 16, interlaced, false};
        const Raster lr{48, 8, interlaced, true};
        const Raster mr{96, 16, interlaced, true};
        const Buf basev = fill_only(make_grain(96, 16, mr, 6, 2), br);
        const Buf lay = make_grain(48, 8, lr, 7, 1);  // a8 ∈ {0, 255}
        const Layer A1{lay.data(), lr, 12, 4, 1.0f};
        // 1) composite(none, [A]) -> v210a (viso išėjimo dydžio)
        const Buf mid = run_composite(nullptr, mr, {A1}, mr);
        // 2) composite(bazė, [mid]) prieš composite(bazė, [A]). Dvejetainė alfa: ae ∈ {0, A}; kai chroma pora (A, 0): ac = 438 abiem būdais.
        const Buf via_chain = run_composite(&basev, br, {{mid.data(), mr, 0, 0, 1.0f}}, br);
        const Buf direct = run_composite(&basev, br, {A1}, br);
        CHECK(via_chain == direct);
    }
}

TEST_CASE("gfx ref: grandinė su bendra alfa - apvalinimo skirtumo matavimas", "[gfx][measure]") {
    // Bendra alfa: composite(none,[A]) -> composite(bazė,[tas]) nebūtinai lygu composite(bazė,[A]) (dvigubas apvalinimas).
    // Skirtumas tik matuojamas ir spausdinamas (nereikalaujamas lygus); skaičius užrašomas užduoties ataskaitoje.
    // Scenarijai: bazė v210 (nepermatoma) ir v210a (bendra alfa), išėjimas v210a (matuojame ir fill, ir key), opacity 1 ir 0.7.
    const Raster lr{48, 8, false, true};
    const Raster mr{96, 16, false, true};
    const Buf lay = make_grain(48, 8, lr, 9, 0);
    const Buf base_a = make_grain(96, 16, mr, 8, 0);
    const Buf lay2 = make_grain(48, 8, lr, 10, 0);
    for (int two_layers = 0; two_layers < 2; ++two_layers)
        for (int base_kind = 0; base_kind < 2; ++base_kind)
            for (float op : {1.0f, 0.7f}) {
            const Raster br = base_kind == 0 ? Raster{96, 16, false, false} : mr;
            const Buf basev = base_kind == 0 ? fill_only(base_a, mr) : base_a;
            const Layer A1{lay.data(), lr, 12, 4, op};
            // two_layers: antras (persidengiantis) sluoksnis -> grandinė jau nebūtinai tapati tiesioginei
            const Layer A2{lay2.data(), lr, 30, 6, op};
            const std::vector<Layer> stack = two_layers ? std::vector<Layer>{A1, A2} : std::vector<Layer>{A1};
            const Buf mid = run_composite(nullptr, mr, stack, mr);
            const Buf chain = run_composite(&basev, br, {{mid.data(), mr, 0, 0, 1.0f}}, mr);
            const Buf direct = run_composite(&basev, br, stack, mr);
            int ndiff = 0, maxd = 0, n = 0, kdiff = 0, kmax = 0;
            auto acc = [&](int d) { ndiff += d != 0; maxd = std::max(maxd, d); ++n; };
            for (int r = 0; r < 16; ++r)
                for (int x = 0; x < 96; ++x) {
                    acc(std::abs(fill_y(chain, 96, r, x) - fill_y(direct, 96, r, x)));
                    if (x % 2 == 0) {
                        acc(std::abs(fill_cb(chain, 96, r, x / 2) - fill_cb(direct, 96, r, x / 2)));
                        acc(std::abs(fill_cr(chain, 96, r, x / 2) - fill_cr(direct, 96, r, x / 2)));
                    }
                    const int kd = std::abs(key_at(chain, mr, r, x) - key_at(direct, mr, r, x));
                    kdiff += kd != 0; kmax = std::max(kmax, kd);
                }
            std::printf("[MATAVIMAS] sluoksnių=%d bazė=%s opacity=%.1f: fill %d/%d mėginių skiriasi (max %d); key %d/%d (max %d)\n",
                        two_layers + 1, base_kind == 0 ? "v210" : "v210a", static_cast<double>(op), ndiff, n, maxd, kdiff, 96 * 16, kmax);
        }
    SUCCEED();
}

TEST_CASE("gfx ref: nukirpimas ties riba, jokio rašymo už buferio [PF-1]", "[gfx]") {
    for (bool alpha_out : {false, true}) {
        Raster out_r = kHd1080i;
        out_r.alpha = alpha_out;
        const Raster lr{384, 216, true, true};
        const Buf lay = make_grain(384, 216, lr, 11, 2);  // nepermatomas
        const int LX = 1800, LY = 1000;  // dešinėje liko 1920-1800 = 120 px; apačioje 540 - 500 = 40 lauko eilučių
        const std::size_t guard = 4096, n = grain_bytes(out_r);
        Buf mem(guard + n + guard, 0xA5);
        std::memset(mem.data() + guard, 0x5A, n);  // purvinas išėjimas
        Layer L{lay.data(), lr, LX, LY, 1.0f};
        REQUIRE(ref::composite(nullptr, out_r, &L, 1, out_r, 0, mem.data() + guard));
        for (std::size_t i = 0; i < guard; ++i) {
            REQUIRE(mem[i] == 0xA5);
            REQUIRE(mem[guard + n + i] == 0xA5);
        }
        const Buf out(mem.begin() + static_cast<std::ptrdiff_t>(guard), mem.begin() + static_cast<std::ptrdiff_t>(guard + n));
        for (int r = 0; r < 540; r += 7)
            for (int x = 0; x < 1920; x += 5) {
                const int lrow = r - LY / 2, lx = x - LX;
                const bool in = lrow >= 0 && lrow < 108 && lx >= 0 && lx < 384;
                REQUIRE(fill_y(out, 1920, r, x) == (in ? fill_y(lay, 384, lrow, lx) : 64));
                if (alpha_out) REQUIRE(key_at(out, out_r, r, x) == (in ? 940 : 64));
            }
        // kraštiniai taškai: paskutinis stulpelis, pirma ir paskutinė sluoksnio eilutės
        CHECK(fill_y(out, 1920, 539, 1919) == fill_y(lay, 384, 539 - 500, 119));
        CHECK(fill_y(out, 1920, 500, 1800) == fill_y(lay, 384, 0, 0));
        CHECK(fill_y(out, 1920, 499, 1919) == 64);  // virš sluoksnio
    }
}

TEST_CASE("gfx ref: interlaced y/2 eilučių atitiktis", "[gfx]") {
    // Išėjimas 8×8 interlaced (4 lauko eilutės), sluoksnis 4×4 interlaced (2 eilutės) ties y = 2 -> išėjimo lauko eilutės 1 ir 2.
    const Raster out_r{8, 8, true, false}, lr{4, 4, true, true};
    const Buf lay = make_grain(4, 4, lr, 12, 2);
    for (int parity : {0, 1}) {
        const Buf o = run_composite(nullptr, out_r, {{lay.data(), lr, 2, 2, 1.0f}}, out_r, parity);
        for (int r = 0; r < 4; ++r)
            for (int x = 0; x < 8; ++x) {
                const bool in = r >= 1 && r < 3 && x >= 2 && x < 6;
                CHECK(fill_y(o, 8, r, x) == (in ? fill_y(lay, 4, r - 1, x - 2) : 64));
            }
    }
}

TEST_CASE("gfx ref: k = 0", "[gfx]") {
    const Raster br{48, 4, false, true};
    const Buf base = make_grain(48, 4, br, 13, 0);
    CHECK(run_composite(&base, br, {}, br) == base);
    const Buf none = run_composite(nullptr, br, {}, br);  // 64/512, key 64, padding 0
    CHECK(fill_y(none, 48, 3, 47) == 64);
    CHECK(fill_cb(none, 48, 3, 23) == 512);
    CHECK(key_at(none, br, 3, 47) == 64);
}

TEST_CASE("gfx ref: netinkama geometrija atmetama, nieko nerašoma", "[gfx]") {
    const Raster pr{8, 8, false, true}, ir{8, 8, true, true}, odd{7, 8, false, true}, lr{4, 4, false, true};
    const Buf lay = make_grain(4, 4, lr, 14, 2);
    Buf out(grain_bytes(pr), 0xA5), outi(grain_bytes(ir), 0xA5);
    const Buf untouched(out.size(), 0xA5), untouchedi(outi.size(), 0xA5);
    // nelyginis x
    Layer L{lay.data(), lr, 1, 0, 1.0f};
    CHECK_FALSE(ref::composite(nullptr, pr, &L, 1, pr, 0, out.data()));
    // neigiami
    L.x = -2;
    CHECK_FALSE(ref::composite(nullptr, pr, &L, 1, pr, 0, out.data()));
    L.x = 0;
    L.y = -1;
    CHECK_FALSE(ref::composite(nullptr, pr, &L, 1, pr, 0, out.data()));
    // nelyginis y interlaced
    const Raster lir{4, 4, true, true};
    const Buf layi = make_grain(4, 4, lir, 15, 2);
    Layer Li{layi.data(), lir, 0, 1, 1.0f};
    CHECK_FALSE(ref::composite(nullptr, ir, &Li, 1, ir, 0, outi.data()));
    // paritetas 2
    Li.y = 0;
    CHECK_FALSE(ref::composite(nullptr, ir, &Li, 1, ir, 2, outi.data()));
    // nelyginis išėjimo plotis; nelyginis sluoksnio plotis
    Buf outo(grain_bytes(odd), 0xA5);
    CHECK_FALSE(ref::composite(nullptr, odd, nullptr, 0, odd, 0, outo.data()));
    Layer Lo{lay.data(), Raster{3, 4, false, true}, 0, 0, 1.0f};
    CHECK_FALSE(ref::composite(nullptr, pr, &Lo, 1, pr, 0, out.data()));
    // sluoksnio/išėjimo interlace neatitikimas, nullptr, NaN opacity
    Layer Lm{lay.data(), lr, 0, 0, 1.0f};
    CHECK_FALSE(ref::composite(nullptr, ir, &Lm, 1, ir, 0, outi.data()));
    CHECK_FALSE(ref::composite(nullptr, pr, nullptr, 1, pr, 0, out.data()));
    CHECK_FALSE(ref::composite(nullptr, pr, &Lm, 1, pr, 0, nullptr));
    Lm.opacity = std::nanf("");
    CHECK_FALSE(ref::composite(nullptr, pr, &Lm, 1, pr, 0, out.data()));
    CHECK(out == untouched);
    CHECK(outi == untouchedi);
    CHECK(outo == Buf(outo.size(), 0xA5));
    // rgba_to_v210a: nelyginis plotis, paritetas, ne-alfa rastras, per mažas stride
    Buf rgba(7 * 8 * 4, 255), g(grain_bytes(odd), 0xA5);
    CHECK_FALSE(ref::rgba_to_v210a(rgba.data(), 7, 8, 28, AlphaIn::Straight, odd, 0, g.data()));
    Buf rgba8(8 * 8 * 4, 255), g8(grain_bytes(ir), 0xA5);
    CHECK_FALSE(ref::rgba_to_v210a(rgba8.data(), 8, 8, 32, AlphaIn::Straight, ir, 2, g8.data()));
    CHECK_FALSE(ref::rgba_to_v210a(rgba8.data(), 8, 8, 32, AlphaIn::Straight, Raster{8, 8, false, false}, 0, g8.data()));
    CHECK_FALSE(ref::rgba_to_v210a(rgba8.data(), 8, 8, 16, AlphaIn::Straight, pr, 0, out.data()));
    CHECK(g == Buf(g.size(), 0xA5));
    CHECK(g8 == Buf(g8.size(), 0xA5));
    CHECK(out == untouched);
}

TEST_CASE("gfx ref: nelyginis y progresyviame rastre leidžiamas (R1)", "[gfx]") {
    const Raster pr{8, 8, false, false}, lr{4, 4, false, true};
    const Buf lay = make_grain(4, 4, lr, 16, 2);
    const Buf o = run_composite(nullptr, pr, {{lay.data(), lr, 2, 3, 1.0f}}, pr);
    for (int r = 0; r < 8; ++r)
        for (int x = 0; x < 8; ++x) {
            const bool in = r >= 3 && r < 7 && x >= 2 && x < 6;
            CHECK(fill_y(o, 8, r, x) == (in ? fill_y(lay, 4, r - 3, x - 2) : 64));
        }
}
