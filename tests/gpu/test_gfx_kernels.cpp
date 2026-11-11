// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// CUDA variantų (gfx::cuda, H ir G) testai: baitas į baitą lygu nepriklausomam etalonui (gfx::ref). Alfa kompozitoriaus
// 5 užduotis. Device išėjimo buferiai tyčia purvini (cudaMemset 0xA5), kad būtų tikrinamas ir padding = 0.
// Be GPU testai praleidžiami (SKIP).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gfx/gfx_kernels.hpp"
#include "gfx/gfx_ref.hpp"

using namespace vramflow::gfx;
using Buf = std::vector<std::uint8_t>;

namespace {

// Catch2 REQUIRE nėra saugus iš kelių gijų, o CUDA klaidos čia — išimtys (kaip test_media_kernels.cpp).
#define CUDA_OK(x)                                                                                        \
    do {                                                                                                  \
        const cudaError_t _e = (x);                                                                       \
        if (_e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(_e)); \
    } while (0)

#define REQUIRE_GPU()                                                                 \
    do {                                                                              \
        int _n = 0;                                                                   \
        if (cudaGetDeviceCount(&_n) != cudaSuccess || _n == 0) SKIP("nėra CUDA GPU"); \
    } while (0)

struct DevBuf {
    std::uint8_t* p = nullptr;
    std::size_t n = 0;
    explicit DevBuf(std::size_t n_) : n(n_) { CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), n ? n : 1)); }
    explicit DevBuf(const Buf& b) : DevBuf(b.size()) { CUDA_OK(cudaMemcpy(p, b.data(), n, cudaMemcpyHostToDevice)); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
    ~DevBuf() { cudaFree(p); }
    void fill(int byte) { CUDA_OK(cudaMemset(p, byte, n)); }
    Buf down() const {
        Buf b(n);
        CUDA_OK(cudaMemcpy(b.data(), p, n, cudaMemcpyDeviceToHost));
        return b;
    }
};

struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    std::uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    int byte() { return static_cast<int>(next() & 255u); }
};

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

// Grain'as host + device kopija.
struct G {
    Buf h;
    DevBuf d;
    explicit G(Buf b) : h(std::move(b)), d(h) {}
};
// Sluoksnis: host grain (etalonui) + device (kernel'iui).
struct L {
    const G* g;
    Raster r;
    int x, y;
    float op;
};

// Vienas composite palyginimas su etalonu (device purvinas 0xA5). Grąžina skirtingų baitų skaičių.
std::size_t run_both(const G* base, const Raster& base_r, const std::vector<L>& layers, const Raster& out_r, int parity,
                     cudaStream_t s = nullptr) {
    std::vector<Layer> hl, dl;
    for (const L& l : layers) {
        hl.push_back({l.g->h.data(), l.r, l.x, l.y, l.op});
        dl.push_back({l.g->d.p, l.r, l.x, l.y, l.op});
    }
    Buf ref_out(grain_bytes(out_r), 0x5A);
    REQUIRE(ref::composite(base ? base->h.data() : nullptr, base_r, hl.empty() ? nullptr : hl.data(),
                           static_cast<int>(hl.size()), out_r, parity, ref_out.data()));
    DevBuf out(grain_bytes(out_r));
    out.fill(0xA5);
    CompositeJob job{base ? base->d.p : nullptr, base_r, dl.empty() ? nullptr : dl.data(), static_cast<int>(dl.size()), out_r, out.p};
    CUDA_OK(cuda::composite_batch(&job, 1, parity, s));
    CUDA_OK(cudaDeviceSynchronize());
    return diff_bytes(ref_out, out.down());
}

}  // namespace

TEST_CASE("gfx cuda: rgba_to_v210a == ref (visi plotai, paritetai, alfa režimai)", "[gfx][gpu]") {
    REQUIRE_GPU();
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
        Buf a(grain_bytes(r), 0x5A);
        REQUIRE(ref::rgba_to_v210a(rgba.data(), c.w, c.h, stride, c.mode, r, c.parity, a.data()));
        DevBuf drgba(rgba), dout(grain_bytes(r));
        dout.fill(0xA5);
        CUDA_OK(cuda::rgba_to_v210a(drgba.p, c.w, c.h, stride, c.mode, r, c.parity, dout.p, nullptr));
        CUDA_OK(cudaDeviceSynchronize());
        INFO("w=" << c.w << " il=" << c.il << " parity=" << c.parity << " premult=" << (c.mode == AlphaIn::Premultiplied));
        CHECK(diff_bytes(a, dout.down()) == 0);
    }
}

TEST_CASE("gfx cuda: rgba_to_v210a ribiniai atvejai (visos alfa, c > a premultiplied)", "[gfx][gpu]") {
    REQUIRE_GPU();
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
    DevBuf drgba(rgba);
    for (AlphaIn m : {AlphaIn::Straight, AlphaIn::Premultiplied}) {
        const Raster r{w, h, false, true};
        Buf a(grain_bytes(r), 0x5A);
        REQUIRE(ref::rgba_to_v210a(rgba.data(), w, h, w * 4, m, r, 0, a.data()));
        DevBuf dout(grain_bytes(r));
        dout.fill(0xA5);
        CUDA_OK(cuda::rgba_to_v210a(drgba.p, w, h, w * 4, m, r, 0, dout.p, nullptr));
        CUDA_OK(cudaDeviceSynchronize());
        CHECK(diff_bytes(a, dout.down()) == 0);
    }
}

// Visos 2^24 RGB reikšmės (a = 255, straight): double aritmetika RGB->YCbCr privalo sutapti be FMA. Kiekviena spalva
// užima abu poros pikselius, todėl chroma = jos pačios Cb/Cr (be vidurkinimo, kuris slėptų ±1 klaidas).
TEST_CASE("gfx cuda: rgba_to_v210a visos 2^24 RGB reikšmės == ref", "[gfx][gpu]") {
    REQUIRE_GPU();
    const int w = 8192, h = 4096;
    Buf rgba(static_cast<std::size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::uint32_t c = static_cast<std::uint32_t>(y) * 4096u + static_cast<std::uint32_t>(x / 2);
            std::uint8_t* p = &rgba[(static_cast<std::size_t>(y) * w + x) * 4];
            p[0] = static_cast<std::uint8_t>(c >> 16);
            p[1] = static_cast<std::uint8_t>(c >> 8);
            p[2] = static_cast<std::uint8_t>(c);
            p[3] = 255;
        }
    const Raster r{w, h, false, true};
    Buf a(grain_bytes(r), 0x5A);
    REQUIRE(ref::rgba_to_v210a(rgba.data(), w, h, static_cast<std::size_t>(w) * 4, AlphaIn::Straight, r, 0, a.data()));
    DevBuf drgba(rgba), dout(grain_bytes(r));
    dout.fill(0xA5);
    CUDA_OK(cuda::rgba_to_v210a(drgba.p, w, h, static_cast<std::size_t>(w) * 4, AlphaIn::Straight, r, 0, dout.p, nullptr));
    CUDA_OK(cudaDeviceSynchronize());
    CHECK(diff_bytes(a, dout.down()) == 0);
}

TEST_CASE("gfx cuda: composite == ref (1080i, K, opacity, pozicijos, išėjimas, bazė)", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_v{1920, 1080, true, false}, out_a{1920, 1080, true, true};
    const G base_v(make_grain(out_v, 11)), base_a(make_grain(out_a, 12));
    const Raster lr[4] = {{384, 216, true, true}, {256, 144, true, true}, {200, 100, true, false}, {384, 216, true, true}};
    std::vector<G> lg;
    for (int i = 0; i < 4; ++i) lg.emplace_back(make_grain(lr[i], 20 + i));
    // R1: interlaced -> (100,34) vietoj (100,33).
    const int pos[4][2] = {{0, 0}, {1488, 64}, {1800, 900}, {100, 34}};
    int combos = 0;
    for (int k : {1, 2, 4})
        for (float op : {0.0f, 0.5f, 1.0f})
            for (int p = 0; p < 4; ++p)
                for (bool oa : {false, true})
                    for (int bs = 0; bs < 3; ++bs) {  // 0 = v210, 1 = v210a, 2 = none
                        std::vector<L> Ls;
                        for (int j = 0; j < k; ++j) {
                            const int pi = (p + j) % 4;
                            Ls.push_back({&lg[j], lr[j], pos[pi][0], pos[pi][1], op});
                        }
                        const G* base = bs == 0 ? &base_v : bs == 1 ? &base_a : nullptr;
                        const Raster br = bs == 0 ? out_v : out_a;
                        INFO("k=" << k << " op=" << op << " pos=" << p << " out_alpha=" << oa << " base=" << bs);
                        REQUIRE(run_both(base, br, Ls, oa ? out_a : out_v, combos & 1) == 0);
                        ++combos;
                    }
}

TEST_CASE("gfx cuda: composite == ref, progresyvus rastras, (100,33) leidžiamas", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_v{640, 360, false, false}, out_a{640, 360, false, true};
    const G base_v(make_grain(out_v, 31)), base_a(make_grain(out_a, 32));
    const Raster lr[2] = {{128, 96, false, true}, {100, 50, false, false}};
    std::vector<G> lg;
    lg.emplace_back(make_grain(lr[0], 33));
    lg.emplace_back(make_grain(lr[1], 34));
    const int pos[4][2] = {{0, 0}, {500, 100}, {600, 300}, {100, 33}};
    for (int k : {1, 2})
        for (float op : {0.0f, 0.5f, 1.0f})
            for (int p = 0; p < 4; ++p)
                for (bool oa : {false, true})
                    for (int bs = 0; bs < 3; ++bs) {
                        std::vector<L> Ls;
                        for (int j = 0; j < k; ++j) Ls.push_back({&lg[j], lr[j], pos[(p + j) % 4][0], pos[(p + j) % 4][1], op});
                        const G* base = bs == 0 ? &base_v : bs == 1 ? &base_a : nullptr;
                        const Raster br = bs == 0 ? out_v : out_a;
                        INFO("k=" << k << " op=" << op << " pos=" << p << " out_alpha=" << oa << " base=" << bs);
                        REQUIRE(run_both(base, br, Ls, oa ? out_a : out_v, 0) == 0);
                    }
}

TEST_CASE("gfx cuda: composite == ref, plotai nesidalijantys iš 3/6/48, nukirpimas", "[gfx][gpu]") {
    REQUIRE_GPU();
    for (int w : {382, 1280, 94}) {
        const Raster out_v{w, 64, true, false}, out_a{w, 64, true, true};
        const G base_v(make_grain(out_v, 41)), base_a(make_grain(out_a, 42));
        const Raster lr[3] = {{130, 40, true, true}, {46, 20, true, false}, {66, 64, true, true}};
        std::vector<G> lg;
        for (int i = 0; i < 3; ++i) lg.emplace_back(make_grain(lr[i], 43 + i));
        for (int x : {0, 2, 6, 48, 100, w - 30, w - 2, w})
            for (bool oa : {false, true})
                for (int bs = 0; bs < 3; ++bs) {
                    std::vector<L> Ls;
                    for (int j = 0; j < 3; ++j) Ls.push_back({&lg[j], lr[j], (x + 14 * j) & ~1, 2 * j * 4, 0.5f + 0.25f * j});
                    const G* base = bs == 0 ? &base_v : bs == 1 ? &base_a : nullptr;
                    INFO("w=" << w << " x=" << x << " out_alpha=" << oa << " base=" << bs);
                    REQUIRE(run_both(base, bs == 0 ? out_v : out_a, Ls, oa ? out_a : out_v, 1) == 0);
                }
    }
}

TEST_CASE("gfx cuda: composite == ref su atsitiktiniais baitais (kraštutinės 10 bitų reikšmės, key už ribų)", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_a{640, 80, true, true}, out_v{640, 80, true, false};
    const G base(random_grain(out_a, 51));
    const Raster lr[4] = {{200, 40, true, true}, {64, 30, true, true}, {222, 20, true, false}, {640, 40, true, true}};
    std::vector<G> lg;
    for (int i = 0; i < 4; ++i) lg.emplace_back(random_grain(lr[i], 52 + i));
    for (float op : {0.0f, 0.3f, 0.7f, 0.9f, 1.0f})
        for (bool oa : {false, true})
            for (int bs = 0; bs < 2; ++bs) {
                std::vector<L> Ls;
                for (int j = 0; j < 4; ++j) Ls.push_back({&lg[j], lr[j], 20 * j, 4 * j, op});
                INFO("op=" << op << " out_alpha=" << oa << " base_alpha=" << bs);
                REQUIRE(run_both(&base, bs ? out_a : out_v, Ls, oa ? out_a : out_v, 0) == 0);
            }
}

TEST_CASE("gfx cuda: k = 0, bazė ir none", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_a{382, 32, true, true}, out_v{382, 32, true, false};
    const G base_a(random_grain(out_a, 61));
    const std::vector<L> none;
    CHECK(run_both(&base_a, out_a, none, out_a, 0) == 0);
    CHECK(run_both(&base_a, out_a, none, out_v, 1) == 0);
    CHECK(run_both(nullptr, out_a, none, out_a, 0) == 0);
    CHECK(run_both(nullptr, out_v, none, out_v, 0) == 0);
}

TEST_CASE("gfx cuda: composite_batch n=64, skirtingos pozicijos / bazės / išėjimai == ref", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_v{640, 360, true, false}, out_a{640, 360, true, true};
    const G base_v(make_grain(out_v, 81)), base_a(make_grain(out_a, 82));
    const Raster lr[3] = {{128, 72, true, true}, {100, 50, true, false}, {64, 40, true, true}};
    std::vector<G> lg;
    for (int i = 0; i < 3; ++i) lg.emplace_back(make_grain(lr[i], 83 + i));
    constexpr int N = 64;
    std::vector<std::vector<Layer>> hl(N), dl(N);
    std::vector<Buf> refs(N);
    std::vector<DevBuf> outs;
    std::vector<CompositeJob> jobs(N);
    std::vector<Raster> ors(N);
    for (int i = 0; i < N; ++i) {
        const int k = 1 + i % 3;
        const int bs = i % 3;  // 0 = v210, 1 = v210a, 2 = none
        ors[i] = (i / 3) % 2 ? out_a : out_v;
        const Raster br = bs == 0 ? out_v : out_a;
        for (int j = 0; j < k; ++j) {
            const int lx = ((i * 38 + j * 90) % 700) & ~1;  // iš dalies už dešiniojo krašto -> nukirpimas
            const int ly = ((i * 14 + j * 30) % 380) & ~1;
            const float op = 0.25f + 0.25f * ((i + j) % 4);
            hl[i].push_back({lg[j].h.data(), lr[j], lx, ly, op});
            dl[i].push_back({lg[j].d.p, lr[j], lx, ly, op});
        }
        refs[i].assign(grain_bytes(ors[i]), 0x5A);
        REQUIRE(ref::composite(bs == 0 ? base_v.h.data() : bs == 1 ? base_a.h.data() : nullptr, br, hl[i].data(), k, ors[i], i & 1,
                               refs[i].data()));
        outs.emplace_back(grain_bytes(ors[i]));
        outs.back().fill(0xA5);
        jobs[i] = {bs == 0 ? base_v.d.p : bs == 1 ? base_a.d.p : nullptr, br, dl[i].data(), k, ors[i], outs.back().p};
    }
    CUDA_OK(cuda::composite_batch(jobs.data(), N, 1, nullptr));
    CUDA_OK(cudaDeviceSynchronize());
    for (int i = 0; i < N; ++i) {
        INFO("job=" << i);
        CHECK(diff_bytes(refs[i], outs[i].down()) == 0);
    }
}

TEST_CASE("gfx cuda: composite_batch, sluoksnių suma > kMaxBatchLayers dalijama į kelis paleidimus", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_a{256, 64, true, true};
    const Raster lr{64, 16, true, true};
    const G base(make_grain(out_a, 91)), lg(make_grain(lr, 92));
    constexpr int N = 64, K = 12;  // 768 > 640
    static_assert(N * K > vramflow::gfx::cuda::kMaxBatchLayers);
    std::vector<std::vector<Layer>> hl(N), dl(N);
    std::vector<Buf> refs(N);
    std::vector<DevBuf> outs;
    std::vector<CompositeJob> jobs(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            const int lx = ((i * 10 + j * 16) % 220) & ~1, ly = ((i * 2 + j * 4) % 60) & ~1;
            hl[i].push_back({lg.h.data(), lr, lx, ly, 0.5f});
            dl[i].push_back({lg.d.p, lr, lx, ly, 0.5f});
        }
        refs[i].assign(grain_bytes(out_a), 0);
        REQUIRE(ref::composite(base.h.data(), out_a, hl[i].data(), K, out_a, 0, refs[i].data()));
        outs.emplace_back(grain_bytes(out_a));
        outs.back().fill(0xA5);
        jobs[i] = {base.d.p, out_a, dl[i].data(), K, out_a, outs.back().p};
    }
    CUDA_OK(cuda::composite_batch(jobs.data(), N, 0, nullptr));
    CUDA_OK(cudaDeviceSynchronize());
    for (int i = 0; i < N; ++i) {
        INFO("job=" << i);
        CHECK(diff_bytes(refs[i], outs[i].down()) == 0);
    }
}

TEST_CASE("gfx cuda: composite_batch, bazė == išėjimas (vietoje)", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_a{382, 64, true, true};
    const Raster lr{130, 40, true, true};
    const Buf bh = make_grain(out_a, 95);
    const G lg(make_grain(lr, 96));
    const Layer hl{lg.h.data(), lr, 100, 4, 0.6f}, dl{lg.d.p, lr, 100, 4, 0.6f};
    Buf expect(grain_bytes(out_a));
    REQUIRE(ref::composite(bh.data(), out_a, &hl, 1, out_a, 0, expect.data()));
    DevBuf io(bh);
    CompositeJob job{io.p, out_a, &dl, 1, out_a, io.p};
    CUDA_OK(cuda::composite_batch(&job, 1, 0, nullptr));
    CUDA_OK(cudaDeviceSynchronize());
    CHECK(diff_bytes(expect, io.down()) == 0);
}

TEST_CASE("gfx cuda: netinkama geometrija -> cudaErrorInvalidValue, nieko nerašoma", "[gfx][gpu]") {
    REQUIRE_GPU();
    const Raster out_a{64, 32, true, true}, out_p{64, 32, false, true};
    const Raster lr{16, 8, true, true}, lrp{16, 8, false, true};
    const G lg(make_grain(lr, 91)), lgp(make_grain(lrp, 92)), base(make_grain(out_a, 93));
    const Raster wrongbase{32, 32, true, true};
    auto bad = [&](const Layer& l, const Raster& o, const Raster* br = nullptr) {
        DevBuf ob(grain_bytes(o));
        ob.fill(0xA5);
        CompositeJob job{base.d.p, br ? *br : o, &l, 1, o, ob.p};
        const cudaError_t e = cuda::composite_batch(&job, 1, 0, nullptr);
        CUDA_OK(cudaDeviceSynchronize());
        return e == cudaErrorInvalidValue && ob.down() == Buf(grain_bytes(o), 0xA5);
    };
    CHECK(bad({lg.d.p, lr, 3, 0, 1.0f}, out_a));                                     // nelyginis x
    CHECK(bad({lg.d.p, lr, -2, 0, 1.0f}, out_a));                                    // neigiamas x
    CHECK(bad({lg.d.p, lr, 0, -2, 1.0f}, out_a));                                    // neigiamas y
    CHECK(bad({lg.d.p, lr, 0, 3, 1.0f}, out_a));                                     // nelyginis y interlaced (R1)
    CHECK(bad({nullptr, lr, 0, 0, 1.0f}, out_a));                                    // nullptr grain
    CHECK(bad({lg.d.p, lr, 0, 0, std::numeric_limits<float>::quiet_NaN()}, out_a));  // NaN
    CHECK(bad({lg.d.p, {15, 8, true, true}, 0, 0, 1.0f}, out_a));                    // nelyginis plotis (R7)
    CHECK(bad({lgp.d.p, lrp, 0, 0, 1.0f}, out_a));                                   // interlaced neatitinka
    CHECK(bad({lg.d.p, lr, 0, 0, 1.0f}, out_a, &wrongbase));                         // bazės rastras kitoks
    DevBuf out(grain_bytes(out_a));
    out.fill(0xA5);
    CHECK(cuda::composite_batch(nullptr, 0, 0, nullptr) == cudaSuccess);              // n == 0
    CompositeJob nb{nullptr, out_a, nullptr, 0, out_a, out.p};
    CHECK(cuda::composite_batch(&nb, 1, 2, nullptr) == cudaErrorInvalidValue);        // paritetas
    CompositeJob no{nullptr, out_a, nullptr, 0, out_a, nullptr};
    CHECK(cuda::composite_batch(&no, 1, 0, nullptr) == cudaErrorInvalidValue);        // nullptr išėjimas
    CHECK(cuda::composite_batch(&nb, 65, 0, nullptr) == cudaErrorInvalidValue);       // n > kMaxBatch
    CHECK(cuda::composite_batch(&nb, -1, 0, nullptr) == cudaErrorInvalidValue);
    CompositeJob nl{nullptr, out_a, nullptr, 1, out_a, out.p};
    CHECK(cuda::composite_batch(&nl, 1, 0, nullptr) == cudaErrorInvalidValue);        // k > 0, layers == nullptr
    std::vector<Layer> many(vramflow::gfx::cuda::kMaxBatchLayers + 1, Layer{lg.d.p, lr, 0, 0, 1.0f});
    CompositeJob nm{nullptr, out_a, many.data(), static_cast<int>(many.size()), out_a, out.p};
    CHECK(cuda::composite_batch(&nm, 1, 0, nullptr) == cudaErrorInvalidValue);        // k per didelis
    CHECK(out.down() == Buf(grain_bytes(out_a), 0xA5));
    // netinkamas darbas batch'e -> visas batch'as atmetamas, tinkamo darbo išėjimas irgi nerašomas
    DevBuf o2(grain_bytes(out_a));
    o2.fill(0xA5);
    const Layer badl{lg.d.p, lr, 3, 0, 1.0f}, goodl{lg.d.p, lr, 4, 4, 1.0f};
    CompositeJob jobs[2] = {{nullptr, out_a, &goodl, 1, out_a, o2.p}, {nullptr, out_a, &badl, 1, out_a, out.p}};
    CHECK(cuda::composite_batch(jobs, 2, 0, nullptr) == cudaErrorInvalidValue);
    CUDA_OK(cudaDeviceSynchronize());
    CHECK(o2.down() == Buf(grain_bytes(out_a), 0xA5));
    // Progresyvus: nelyginis y leidžiamas (R1).
    DevBuf op(grain_bytes(out_p));
    const Layer okp{lgp.d.p, lrp, 4, 3, 1.0f};
    CompositeJob jp{nullptr, out_p, &okp, 1, out_p, op.p};
    CHECK(cuda::composite_batch(&jp, 1, 0, nullptr) == cudaSuccess);
    // rgba_to_v210a
    const Buf rgba = make_rgba(64, 32, 256, 7);
    DevBuf drgba(rgba), g(grain_bytes(out_a));
    g.fill(0xA5);
    using cuda::rgba_to_v210a;
    CHECK(rgba_to_v210a(nullptr, 64, 32, 256, AlphaIn::Straight, out_a, 0, g.p, nullptr) == cudaErrorInvalidValue);
    CHECK(rgba_to_v210a(drgba.p, 64, 32, 255, AlphaIn::Straight, out_a, 0, g.p, nullptr) == cudaErrorInvalidValue);  // stride
    CHECK(rgba_to_v210a(drgba.p, 64, 32, 256, AlphaIn::Straight, out_a, 2, g.p, nullptr) == cudaErrorInvalidValue);  // paritetas
    CHECK(rgba_to_v210a(drgba.p, 62, 32, 256, AlphaIn::Straight, out_a, 0, g.p, nullptr) == cudaErrorInvalidValue);  // rastras != w
    CHECK(rgba_to_v210a(drgba.p, 64, 32, 256, AlphaIn::Straight, Raster{64, 32, true, false}, 0, g.p, nullptr) ==
          cudaErrorInvalidValue);  // be alfa
    CHECK(rgba_to_v210a(drgba.p, 64, 32, 256, AlphaIn::Straight, out_a, 0, nullptr, nullptr) == cudaErrorInvalidValue);
    CUDA_OK(cudaDeviceSynchronize());
    CHECK(g.down() == Buf(grain_bytes(out_a), 0xA5));
}

// Informacinis; paleidžiamas aiškiai: vramflow-gfx-gpu-tests "[timing]"
TEST_CASE("gfx cuda: laikas 1080i laukui (informacinis)", "[gfx][gpu][timing]") {
    REQUIRE_GPU();
    const Raster out_a{1920, 1080, true, true};
    const G base(make_grain(out_a, 101));
    const Raster lr{384, 216, true, true};
    const G lg(make_grain(lr, 102));
    cudaStream_t s;
    CUDA_OK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    std::vector<DevBuf> outs;
    for (int i = 0; i < 64; ++i) outs.emplace_back(grain_bytes(out_a));
    for (int n : {1, 8, 64})
        for (int k : {1, 2, 4}) {
            std::vector<Layer> Ly;
            for (int j = 0; j < k; ++j) Ly.push_back({lg.d.p, lr, 100 + 200 * j, 34, 0.7f});
            std::vector<CompositeJob> jobs;
            for (int i = 0; i < n; ++i) jobs.push_back({base.d.p, out_a, Ly.data(), k, out_a, outs[i].p});
            for (int w = 0; w < 3; ++w) CUDA_OK(cuda::composite_batch(jobs.data(), n, 0, s));
            CUDA_OK(cudaStreamSynchronize(s));
            constexpr int it = 50;
            CUDA_OK(cudaEventRecord(e0, s));
            for (int i = 0; i < it; ++i) CUDA_OK(cuda::composite_batch(jobs.data(), n, 0, s));
            CUDA_OK(cudaEventRecord(e1, s));
            CUDA_OK(cudaEventSynchronize(e1));
            float ms = 0;
            CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("[gfx cuda timing] 1080i laukas, n=%2d K=%d: %.3f ms / batch, %.3f ms / isejima\n", n, k, ms / it,
                        ms / it / n);
        }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(s);
}
