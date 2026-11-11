// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// CPU varianto (gfx::cpu) testai: baitas į baitą lygu nepriklausomam etalonui (gfx::ref). Alfa kompozitoriaus 4 užduotis.
// Išėjimo buferiai tyčia purvini (skirtingi užpildai CPU ir etalonui), kad būtų tikrinamas ir padding = 0.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "func/thread_pool.hpp"
#include "gfx/cpu/gfx_cpu.hpp"
#include "gfx/gfx_ref.hpp"

using namespace vramflow::gfx;
using Buf = std::vector<std::uint8_t>;

namespace {

struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    std::uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    int byte() { return static_cast<int>(next() & 255u); }
};

// RGBA kadras, alfa visos reikšmės 0…255 (dalis pikselių — deterministinis ciklas, dalis atsitiktinė), stride gali būti didesnis.
Buf make_rgba(int w, int h, std::size_t stride, std::uint32_t seed) {
    Buf b(stride * h, 0x77);
    Rng g(seed);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            std::uint8_t* p = &b[y * stride + 4 * static_cast<std::size_t>(x)];
            p[0] = static_cast<std::uint8_t>(g.byte());
            p[1] = static_cast<std::uint8_t>(g.byte());
            p[2] = static_cast<std::uint8_t>(g.byte());
            p[3] = (x + y) % 3 == 0 ? static_cast<std::uint8_t>((x * 7 + y) & 255) : static_cast<std::uint8_t>(g.byte());
        }
    return b;
}

// Švarus grain'as per etaloną; r.alpha == false -> tik fill dalis (v210).
Buf make_grain(const Raster& r, std::uint32_t seed) {
    Raster ra = r;
    ra.alpha = true;
    const Buf rgba = make_rgba(r.w, r.h, static_cast<std::size_t>(r.w) * 4, seed);
    Buf g(grain_bytes(ra));
    REQUIRE(ref::rgba_to_v210a(rgba.data(), r.w, r.h, static_cast<std::size_t>(r.w) * 4, AlphaIn::Straight, ra, 0, g.data()));
    g.resize(grain_bytes(r));
    return g;
}
// Atsitiktiniai baitai: fill/key reikšmės 0…1023, key už 64…940 ribų, padding su šiukšlėmis (etalonas turi ignoruoti).
Buf random_grain(const Raster& r, std::uint32_t seed) {
    Buf g(grain_bytes(r));
    Rng rg(seed);
    for (auto& v : g) v = static_cast<std::uint8_t>(rg.byte());
    return g;
}

std::size_t diff_bytes(const Buf& a, const Buf& b) {
    REQUIRE(a.size() == b.size());
    std::size_t d = 0, first = ~std::size_t{0};
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            if (d++ == 0) first = i;
        }
    if (d) std::printf("  skirtingi baitai: %zu, pirmas indeksas %zu\n", d, first);
    return d;
}

// Vienas composite palyginimas su etalonu. Grąžina skirtingų baitų skaičių.
std::size_t run_both(const Buf* base, const Raster& base_r, const std::vector<Layer>& layers, const Raster& out_r, int parity,
                     mxlv::func::ThreadPool* pool = nullptr) {
    Buf a(grain_bytes(out_r), 0xA5), b(grain_bytes(out_r), 0x5A);
    const std::uint8_t* bp = base ? base->data() : nullptr;
    const Layer* lp = layers.empty() ? nullptr : layers.data();
    REQUIRE(ref::composite(bp, base_r, lp, static_cast<int>(layers.size()), out_r, parity, a.data()));
    REQUIRE(cpu::composite(bp, base_r, lp, static_cast<int>(layers.size()), out_r, parity, b.data(), pool));
    return diff_bytes(a, b);
}

}  // namespace

TEST_CASE("gfx cpu: rgba_to_v210a == ref (visi plotai, paritetai, alfa režimai)", "[gfx]") {
    struct Case { int w, h; bool il; int parity; AlphaIn mode; };
    std::vector<Case> cases;
    for (int w : {1920, 1280, 382})
        for (AlphaIn m : {AlphaIn::Straight, AlphaIn::Premultiplied}) {
            cases.push_back({w, 108, true, 0, m});
            cases.push_back({w, 108, true, 1, m});
            cases.push_back({w, 54, false, 0, m});
        }
    std::uint32_t seed = 1;
    for (const Case& c : cases) {
        const Raster r{c.w, c.h, c.il, true};
        const std::size_t stride = static_cast<std::size_t>(c.w) * 4 + (seed % 2 ? 0 : 64);  // dalis su stride > w*4
        const Buf rgba = make_rgba(c.w, c.h, stride, seed++);
        Buf a(grain_bytes(r), 0xA5), b(grain_bytes(r), 0x5A);
        REQUIRE(ref::rgba_to_v210a(rgba.data(), c.w, c.h, stride, c.mode, r, c.parity, a.data()));
        REQUIRE(cpu::rgba_to_v210a(rgba.data(), c.w, c.h, stride, c.mode, r, c.parity, b.data()));
        INFO("w=" << c.w << " il=" << c.il << " parity=" << c.parity << " premult=" << (c.mode == AlphaIn::Premultiplied));
        CHECK(diff_bytes(a, b) == 0);
    }
}

TEST_CASE("gfx cpu: rgba_to_v210a ribiniai atvejai (visos alfa, c > a premultiplied)", "[gfx]") {
    const int w = 256, h = 256;
    Buf rgba(static_cast<std::size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            std::uint8_t* p = &rgba[(static_cast<std::size_t>(y) * w + x) * 4];
            p[0] = static_cast<std::uint8_t>(x);
            p[1] = static_cast<std::uint8_t>(255 - x);
            p[2] = static_cast<std::uint8_t>((x * 5 + y * 3) & 255);
            p[3] = static_cast<std::uint8_t>(y);
        }
    for (AlphaIn m : {AlphaIn::Straight, AlphaIn::Premultiplied}) {
        const Raster r{w, h, false, true};
        Buf a(grain_bytes(r), 0xA5), b(grain_bytes(r), 0x5A);
        REQUIRE(ref::rgba_to_v210a(rgba.data(), w, h, w * 4, m, r, 0, a.data()));
        REQUIRE(cpu::rgba_to_v210a(rgba.data(), w, h, w * 4, m, r, 0, b.data()));
        CHECK(diff_bytes(a, b) == 0);
    }
}

TEST_CASE("gfx cpu: composite == ref (1080i, K, opacity, pozicijos, išėjimas, bazė)", "[gfx]") {
    const Raster out_v{1920, 1080, true, false}, out_a{1920, 1080, true, true};
    const Buf base_v = make_grain(out_v, 11);
    const Buf base_a = make_grain(out_a, 12);
    const Raster lr[4] = {{384, 216, true, true}, {256, 144, true, true}, {200, 100, true, false}, {384, 216, true, true}};
    Buf lg[4];
    for (int i = 0; i < 4; ++i) lg[i] = make_grain(lr[i], 20 + i);
    // R1: interlaced -> (100,34) vietoj (100,33).
    const int pos[4][2] = {{0, 0}, {1488, 64}, {1800, 900}, {100, 34}};
    int combos = 0;
    for (int k : {1, 2, 4})
        for (float op : {0.0f, 0.5f, 1.0f})
            for (int p = 0; p < 4; ++p)
                for (bool oa : {false, true})
                    for (int bs = 0; bs < 3; ++bs) {  // 0 = v210, 1 = v210a, 2 = none (tik v210a išėjimui, R12)
                        if (bs == 2 && !oa) continue;
                        std::vector<Layer> L;
                        for (int j = 0; j < k; ++j) {
                            const int pi = (p + j) % 4;
                            L.push_back({lg[j].data(), lr[j], pos[pi][0], pos[pi][1], op});
                        }
                        const Buf* base = bs == 0 ? &base_v : bs == 1 ? &base_a : nullptr;
                        const Raster br = bs == 0 ? out_v : out_a;
                        INFO("k=" << k << " op=" << op << " pos=" << p << " out_alpha=" << oa << " base=" << bs);
                        REQUIRE(run_both(base, br, L, oa ? out_a : out_v, combos & 1) == 0);
                        ++combos;
                    }
}

TEST_CASE("gfx cpu: composite == ref, progresyvus rastras, (100,33) leidžiamas", "[gfx]") {
    const Raster out_v{640, 360, false, false}, out_a{640, 360, false, true};
    const Buf base_v = make_grain(out_v, 31), base_a = make_grain(out_a, 32);
    const Raster lr[2] = {{128, 96, false, true}, {100, 50, false, false}};
    Buf lg[2] = {make_grain(lr[0], 33), make_grain(lr[1], 34)};
    const int pos[4][2] = {{0, 0}, {500, 100}, {600, 300}, {100, 33}};
    for (int k : {1, 2})
        for (float op : {0.0f, 0.5f, 1.0f})
            for (int p = 0; p < 4; ++p)
                for (bool oa : {false, true})
                    for (int bs = 0; bs < 3; ++bs) {
                        if (bs == 2 && !oa) continue;  // R12
                        std::vector<Layer> L;
                        for (int j = 0; j < k; ++j) L.push_back({lg[j].data(), lr[j], pos[(p + j) % 4][0], pos[(p + j) % 4][1], op});
                        const Buf* base = bs == 0 ? &base_v : bs == 1 ? &base_a : nullptr;
                        const Raster br = bs == 0 ? out_v : out_a;
                        INFO("k=" << k << " op=" << op << " pos=" << p << " out_alpha=" << oa << " base=" << bs);
                        REQUIRE(run_both(base, br, L, oa ? out_a : out_v, 0) == 0);
                    }
}

TEST_CASE("gfx cpu: composite == ref, plotai nesidalijantys iš 3/6/48, nukirpimas", "[gfx]") {
    for (int w : {382, 1280, 94}) {
        const Raster out_v{w, 64, true, false}, out_a{w, 64, true, true};
        const Buf base_v = make_grain(out_v, 41), base_a = make_grain(out_a, 42);
        const Raster lr[3] = {{130, 40, true, true}, {46, 20, true, false}, {66, 64, true, true}};
        Buf lg[3];
        for (int i = 0; i < 3; ++i) lg[i] = make_grain(lr[i], 43 + i);
        for (int x : {0, 2, 6, 48, 100, w - 30, w - 2, w})
            for (bool oa : {false, true})
                for (int bs = 0; bs < 3; ++bs) {
                    if (bs == 2 && !oa) continue;  // R12
                    std::vector<Layer> L;
                    for (int j = 0; j < 3; ++j) L.push_back({lg[j].data(), lr[j], (x + 14 * j) & ~1, 2 * j * 4, 0.5f + 0.25f * j});
                    const Buf* base = bs == 0 ? &base_v : bs == 1 ? &base_a : nullptr;
                    INFO("w=" << w << " x=" << x << " out_alpha=" << oa << " base=" << bs);
                    REQUIRE(run_both(base, bs == 0 ? out_v : out_a, L, oa ? out_a : out_v, 1) == 0);
                }
    }
}

TEST_CASE("gfx cpu: composite == ref su atsitiktiniais baitais (kraštutinės 10 bitų reikšmės, key už ribų)", "[gfx]") {
    const Raster out_a{640, 80, true, true}, out_v{640, 80, true, false};
    const Buf base = random_grain(out_a, 51);
    const Raster lr[4] = {{200, 40, true, true}, {64, 30, true, true}, {222, 20, true, false}, {640, 40, true, true}};
    Buf lg[4];
    for (int i = 0; i < 4; ++i) lg[i] = random_grain(lr[i], 52 + i);
    for (float op : {0.0f, 0.3f, 0.7f, 0.9f, 1.0f})
        for (bool oa : {false, true})
            for (int bs = 0; bs < 2; ++bs) {
                std::vector<Layer> L;
                for (int j = 0; j < 4; ++j) L.push_back({lg[j].data(), lr[j], 20 * j, 4 * j, op});
                INFO("op=" << op << " out_alpha=" << oa << " base_alpha=" << bs);
                REQUIRE(run_both(&base, bs ? out_a : out_v, L, oa ? out_a : out_v, 0) == 0);
            }
}

TEST_CASE("gfx cpu: k = 0, bazė ir none", "[gfx]") {
    const Raster out_a{382, 32, true, true}, out_v{382, 32, true, false};
    const Buf base_a = random_grain(out_a, 61);
    const std::vector<Layer> none;
    CHECK(run_both(&base_a, out_a, none, out_a, 0) == 0);
    CHECK(run_both(&base_a, out_a, none, out_v, 1) == 0);
    CHECK(run_both(nullptr, out_a, none, out_a, 0) == 0);
    // R12: be bazės — tik v210a išėjimas; v210 išėjimas atmetamas (ir ref, ir cpu), nieko nerašoma.
    Buf r(grain_bytes(out_v), 0xA5), c(grain_bytes(out_v), 0xA5);
    CHECK_FALSE(ref::composite(nullptr, out_v, nullptr, 0, out_v, 0, r.data()));
    CHECK_FALSE(cpu::composite(nullptr, out_v, nullptr, 0, out_v, 0, c.data()));
    CHECK(r == Buf(grain_bytes(out_v), 0xA5));
    CHECK(c == Buf(grain_bytes(out_v), 0xA5));
}

TEST_CASE("gfx cpu: composite su ThreadPool == etalonas", "[gfx]") {
    const Raster out_a{1920, 1080, true, true};
    const Buf base = make_grain(out_a, 71);
    const Raster lr{384, 216, true, true};
    const Buf lg = make_grain(lr, 72);
    std::vector<Layer> L{{lg.data(), lr, 100, 34, 0.7f}, {lg.data(), lr, 1800, 900, 1.0f}};
    mxlv::func::ThreadPool pool(4);
    CHECK(run_both(&base, out_a, L, out_a, 0, &pool) == 0);
}

TEST_CASE("gfx cpu: composite_batch (N išėjimų) == ref", "[gfx]") {
    const Raster out_v{1920, 1080, true, false}, out_a{1920, 1080, true, true};
    const Buf base_v = make_grain(out_v, 81), base_a = make_grain(out_a, 82);
    const Raster lr[2] = {{384, 216, true, true}, {200, 100, true, false}};
    Buf lg[2] = {make_grain(lr[0], 83), make_grain(lr[1], 84)};
    const Layer ls[2][2] = {{{lg[0].data(), lr[0], 1488, 64, 1.0f}, {lg[1].data(), lr[1], 100, 34, 0.5f}},
                            {{lg[1].data(), lr[1], 0, 0, 0.7f}, {lg[0].data(), lr[0], 1800, 900, 0.9f}}};
    constexpr int N = 5;
    Buf outs[N], refs[N];
    CompositeJob jobs[N];
    const Buf* bases[N] = {&base_v, &base_a, nullptr, &base_a, &base_v};
    const Raster brs[N] = {out_v, out_a, out_a, out_a, out_v};
    const Raster ors[N] = {out_a, out_a, out_a, out_v, out_v};  // be bazės (job 2) — tik v210a (R12)
    const int ks[N] = {2, 1, 2, 2, 0};
    for (int i = 0; i < N; ++i) {
        outs[i].assign(grain_bytes(ors[i]), 0xC3);
        refs[i].assign(grain_bytes(ors[i]), 0x3C);
        jobs[i] = {bases[i] ? bases[i]->data() : nullptr, brs[i], ls[i % 2], ks[i], ors[i], outs[i].data()};
        REQUIRE(ref::composite(jobs[i].base, brs[i], ls[i % 2], ks[i], ors[i], 1, refs[i].data()));
    }
    for (int threads : {0, 1, 3, 8}) {
        for (auto& o : outs) std::fill(o.begin(), o.end(), 0xC3);
        mxlv::func::ThreadPool pool(threads > 0 ? threads : 1);
        REQUIRE(cpu::composite_batch(jobs, N, 1, threads ? &pool : nullptr));
        for (int i = 0; i < N; ++i) {
            INFO("threads=" << threads << " job=" << i);
            CHECK(diff_bytes(refs[i], outs[i]) == 0);
        }
    }
}

TEST_CASE("gfx cpu: netinkama geometrija -> false, nieko nerašoma", "[gfx]") {
    const Raster out_a{64, 32, true, true}, out_p{64, 32, false, true};
    const Raster lr{16, 8, true, true}, lrp{16, 8, false, true};
    const Buf lg = make_grain(lr, 91), lgp = make_grain(lrp, 92), base = make_grain(out_a, 93);
    const Raster wrongbase{32, 32, true, true};
    auto bad = [&](const Layer& l, const Raster& o, const Raster* br = nullptr) {
        Buf ob(grain_bytes(o), 0xA5);
        const bool ok = cpu::composite(base.data(), br ? *br : o, &l, 1, o, 0, ob.data());
        return !ok && ob == Buf(grain_bytes(o), 0xA5);
    };
    CHECK(bad({lg.data(), lr, 3, 0, 1.0f}, out_a));                                     // nelyginis x
    CHECK(bad({lg.data(), lr, -2, 0, 1.0f}, out_a));                                    // neigiamas x
    CHECK(bad({lg.data(), lr, 0, -2, 1.0f}, out_a));                                    // neigiamas y
    CHECK(bad({lg.data(), lr, 0, 3, 1.0f}, out_a));                                     // nelyginis y interlaced (R1)
    CHECK(bad({nullptr, lr, 0, 0, 1.0f}, out_a));                                       // nullptr grain
    CHECK(bad({lg.data(), lr, 0, 0, std::numeric_limits<float>::quiet_NaN()}, out_a));  // NaN
    CHECK(bad({lg.data(), {15, 8, true, true}, 0, 0, 1.0f}, out_a));                    // nelyginis plotis (R7)
    CHECK(bad({lgp.data(), lrp, 0, 0, 1.0f}, out_a));                                   // interlaced neatitinka
    CHECK(bad({lg.data(), lr, 0, 0, 1.0f}, out_a, &wrongbase));                         // bazės rastras kitoks
    Buf out(grain_bytes(out_a), 0xA5);
    CHECK(!cpu::composite(nullptr, out_a, nullptr, 0, out_a, 2, out.data()));           // paritetas
    CHECK(!cpu::composite(nullptr, out_a, nullptr, 0, out_a, 0, nullptr));              // nullptr išėjimas
    CHECK(out == Buf(grain_bytes(out_a), 0xA5));
    // Progresyvus: nelyginis y leidžiamas (R1).
    Buf op(grain_bytes(out_p), 0xA5);
    const Layer okp{lgp.data(), lrp, 4, 3, 1.0f};
    CHECK(cpu::composite(nullptr, out_p, &okp, 1, out_p, 0, op.data()));
    // rgba_to_v210a
    const Buf rgba = make_rgba(64, 32, 256, 7);
    Buf g(grain_bytes(out_a), 0xA5);
    CHECK(!cpu::rgba_to_v210a(nullptr, 64, 32, 256, AlphaIn::Straight, out_a, 0, g.data()));
    CHECK(!cpu::rgba_to_v210a(rgba.data(), 64, 32, 255, AlphaIn::Straight, out_a, 0, g.data()));  // stride per mažas
    CHECK(!cpu::rgba_to_v210a(rgba.data(), 64, 32, 256, AlphaIn::Straight, out_a, 2, g.data()));  // paritetas
    CHECK(!cpu::rgba_to_v210a(rgba.data(), 62, 32, 256, AlphaIn::Straight, out_a, 0, g.data()));  // rastras != w
    CHECK(!cpu::rgba_to_v210a(rgba.data(), 64, 32, 256, AlphaIn::Straight, Raster{64, 32, true, false}, 0, g.data()));  // be alfa
    CHECK(g == Buf(grain_bytes(out_a), 0xA5));
    // composite_batch (R11): bent vienas netinkamas darbas -> atmetamas VISAS batch'as, abu išėjimai nepaliesti.
    Buf o1(grain_bytes(out_a), 0xA5), o2(grain_bytes(out_a), 0xA5), r2(grain_bytes(out_a), 0x11);
    const Layer badl{lg.data(), lr, 3, 0, 1.0f}, goodl{lg.data(), lr, 4, 4, 1.0f};
    CompositeJob jobs[2] = {{nullptr, out_a, &badl, 1, out_a, o1.data()}, {nullptr, out_a, &goodl, 1, out_a, o2.data()}};
    CHECK(!cpu::composite_batch(jobs, 2, 0, nullptr));
    CHECK(o1 == Buf(grain_bytes(out_a), 0xA5));
    CHECK(o2 == Buf(grain_bytes(out_a), 0xA5));  // tinkamas darbas taip pat nerašomas
    jobs[0].layers = &goodl;
    REQUIRE(cpu::composite_batch(jobs, 2, 0, nullptr));  // pataisius abu — vykdoma
    REQUIRE(ref::composite(nullptr, out_a, &goodl, 1, out_a, 0, r2.data()));
    CHECK(o2 == r2);
    CHECK(!cpu::composite_batch(jobs, 2, 7, nullptr));  // paritetas
    // R12: base == nullptr su v210 išėjimu (ir su geru kaimynu) -> viskas atmetama.
    Buf ov(grain_bytes(Raster{64, 32, true, false}), 0xA5), o3(grain_bytes(out_a), 0xA5);
    CompositeJob nb[2] = {{nullptr, out_a, &goodl, 1, out_a, o3.data()}, {nullptr, Raster{64, 32, true, false}, &goodl, 1, Raster{64, 32, true, false}, ov.data()}};
    CHECK(!cpu::composite_batch(nb, 2, 0, nullptr));
    CHECK(o3 == Buf(grain_bytes(out_a), 0xA5));
    CHECK(ov == Buf(ov.size(), 0xA5));
}

TEST_CASE("gfx cpu: laikas 1080i laukui (informacinis)", "[.bench]") {
    using clk = std::chrono::steady_clock;
    const Raster out_a{1920, 1080, true, true};
    const Buf base = make_grain(out_a, 101);
    const Raster lr{384, 216, true, true};
    const Buf lg = make_grain(lr, 102);
    Buf out(grain_bytes(out_a));
    for (int k : {1, 2, 4}) {
        std::vector<Layer> L;
        for (int j = 0; j < k; ++j) L.push_back({lg.data(), lr, 100 + 200 * j, 34, 0.7f});
        CompositeJob job{base.data(), out_a, L.data(), k, out_a, out.data()};
        cpu::composite_batch(&job, 1, 0, nullptr);  // šildymas
        const auto t0 = clk::now();
        constexpr int it = 20;
        for (int i = 0; i < it; ++i) cpu::composite_batch(&job, 1, 0, nullptr);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / it;
        std::printf("[gfx cpu timing] 1080i laukas, K=%d, 1 gija: %.3f ms\n", k, ms);
    }
}
