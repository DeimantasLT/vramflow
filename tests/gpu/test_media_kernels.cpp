// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// GPU media kernel'ių testai be MXL (architektūra GPU funkcijų §6.6): kiekvienas kernel'is prieš nepriklausomą
// skaliarinį CPU etaloną (media_ref.cpp) — 0 skirtingų baitų.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "media/media_kernels.hpp"
#include "media/media_ref.hpp"
#include "media/v210.hpp"

using namespace mxlv::media;

namespace {

// Catch2 REQUIRE nėra saugus iš kelių gijų — CUDA klaidos kaip išimtys (kaip test_kernels.cpp).
#define CUDA_OK(x)                                                                                        \
    do {                                                                                                  \
        const cudaError_t _e = (x);                                                                       \
        if (_e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(_e)); \
    } while (0)

template <class T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Dev(std::size_t n_) : n(n_) { CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { up(v); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
    Dev& operator=(Dev&&) = delete;
    ~Dev() { cudaFree(p); }
    void up(const std::vector<T>& v) { CUDA_OK(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const {
        std::vector<T> v(n);
        CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void fill(int byte) { CUDA_OK(cudaMemset(p, byte, n * sizeof(T))); }
};

struct Stream {
    cudaStream_t s{};
    Stream() { CUDA_OK(cudaStreamCreate(&s)); }
    ~Stream() { cudaStreamDestroy(s); }
    void sync() { CUDA_OK(cudaStreamSynchronize(s)); }
};

template <class T>
std::size_t diff_count(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return static_cast<std::size_t>(-1);
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}

// Laukas, kurio kiekviena grupė = fn(j, g).
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

std::vector<std::uint32_t> random_legal_field(std::mt19937& rng) {
    std::uniform_int_distribution<int> dy(64, 940), dc(64, 960);
    return make_field([&](int, int) {
        Group g;
        for (auto& v : g.y) v = static_cast<std::uint16_t>(dy(rng));
        for (auto& v : g.cb) v = static_cast<std::uint16_t>(dc(rng));
        for (auto& v : g.cr) v = static_cast<std::uint16_t>(dc(rng));
        return g;
    });
}

std::vector<std::uint32_t> random_words(std::mt19937& rng, std::size_t n = kFieldWords) {
    std::vector<std::uint32_t> v(n);
    for (auto& w : v) w = rng();
    return v;
}

std::vector<std::uint8_t> random_bytes(std::mt19937& rng, std::size_t n) {
    std::vector<std::uint8_t> v(n);
    for (auto& b : v) b = static_cast<std::uint8_t>(rng());
    return v;
}

Group group_at(const std::vector<std::uint32_t>& f, int j, int g) {
    Group gr;
    unpack_group(f.data() + static_cast<std::size_t>(j) * kWordsPerLine + 4 * g, gr);
    return gr;
}

// Logotipo plokštumos device atmintyje.
struct DevLogo {
    Dev<std::uint16_t> y10, cb10, cr10;
    Dev<std::uint8_t> a8, ac8;
    explicit DevLogo(const ref::Logo& l) : y10(l.y10), cb10(l.cb10), cr10(l.cr10), a8(l.a8), ac8(l.ac8) {}
    LogoDesc desc(const ref::Logo& l, int x, int y) const {
        return LogoDesc{y10.p, a8.p, cb10.p, cr10.p, ac8.p, l.w, l.h, x, y};
    }
};

ref::Logo logo_all_alpha(std::mt19937& rng) {
    // 384×216 atsitiktinis RGBA; pirmi 256 pikseliai turi alfa 0..255 (visos reikšmės), kiti — atsitiktinės
    constexpr int W = 384, H = 216;
    auto rgba = random_bytes(rng, static_cast<std::size_t>(W) * H * 4);
    for (int i = 0; i < 256; ++i) rgba[static_cast<std::size_t>(i) * 4 + 3] = static_cast<std::uint8_t>(i);
    for (int i = 256; i < 512; i += 2) rgba[static_cast<std::size_t>(i) * 4 + 3] = rgba[static_cast<std::size_t>(i + 1) * 4 + 3] = 0;
    ref::Logo l;
    if (!ref::logo_from_rgba(rgba.data(), W, H, l)) throw std::runtime_error("logo_from_rgba");
    return l;
}

constexpr std::size_t kY = 1080, kUV = 540;

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------

TEST_CASE("gen == ref (complexity 0..3, N=1 ir N=5)", "[gpu][media][gen]") {
    Stream st;
    const std::uint64_t seed = 0x5EEDF00Dull;
    for (int n : {1, 5}) {
        std::vector<Dev<std::uint32_t>> d;
        d.reserve(n);
        std::vector<GenJob> jobs;
        for (int i = 0; i < n; ++i) {
            d.emplace_back(kFieldWords);
            jobs.push_back({d.back().p, static_cast<std::uint32_t>(i * 37 + 1)});
        }
        for (std::uint32_t cx = 0; cx <= 3; ++cx)
            for (std::uint64_t index : {0ull, 1ull, 50ull, 777ull, 0x200000001ull}) {
                CUDA_OK(gen_fields(jobs.data(), n, seed, cx, index, st.s));
                st.sync();
                std::vector<std::uint32_t> want(kFieldWords);
                for (int i = 0; i < n; ++i) {
                    ref::gen_field(GenParams{seed, jobs[i].flow, cx}, index, want.data());
                    const auto got = d[i].down();
                    INFO("n " << n << " cx " << cx << " index " << index << " job " << i);
                    CHECK(diff_count(got, want) == 0);
                    CHECK(ref::decode_marker(got.data()) == marker_word(index, jobs[i].flow));
                }
            }
    }
}

TEST_CASE("weave == ref (N=1 ir N=5, pitch 1920 ir 2048)", "[gpu][media][weave]") {
    Stream st;
    std::mt19937 rng(11);
    for (int pitch : {1920, 2048})
        for (int n : {1, 5}) {
            std::vector<std::vector<std::uint32_t>> ht, hb;
            std::vector<Dev<std::uint32_t>> dt, db;
            std::vector<Dev<std::uint8_t>> dy, duv;
            dt.reserve(n); db.reserve(n); dy.reserve(n); duv.reserve(n);
            std::vector<WeaveJob> jobs;
            for (int i = 0; i < n; ++i) {
                // paskutinis darbas — visiškai atsitiktiniai žodžiai (Y10 iki 1023 → min(255, …) riba)
                ht.push_back(i == n - 1 && n > 1 ? random_words(rng) : random_legal_field(rng));
                hb.push_back(random_legal_field(rng));
                dt.emplace_back(ht.back());
                db.emplace_back(hb.back());
                dy.emplace_back(pitch * kY);
                duv.emplace_back(pitch * kUV);
                dy.back().fill(0xCD);  // užpildas tarp pločio ir pitch — neturi būti liečiamas
                duv.back().fill(0xCD);
                jobs.push_back({dt.back().p, db.back().p, dy.back().p, duv.back().p});
            }
            CUDA_OK(weave_nv12(jobs.data(), n, pitch, pitch, st.s));
            st.sync();
            for (int i = 0; i < n; ++i) {
                std::vector<std::uint8_t> wy(pitch * kY, 0xCD), wuv(pitch * kUV, 0xCD);
                ref::weave_nv12(ht[i].data(), hb[i].data(), wy.data(), wuv.data(), pitch, pitch);
                INFO("pitch " << pitch << " n " << n << " job " << i);
                CHECK(diff_count(dy[i].down(), wy) == 0);
                CHECK(diff_count(duv[i].down(), wuv) == 0);
            }
        }
}

TEST_CASE("weave lauko izoliacija: laukai nemaišomi", "[gpu][media][weave]") {
    Stream st;
    auto uni = [](int y, int c) {
        return make_field([=](int, int) {
            Group g;
            for (auto& v : g.y) v = static_cast<std::uint16_t>(y);
            for (auto& v : g.cb) v = static_cast<std::uint16_t>(c);
            for (auto& v : g.cr) v = static_cast<std::uint16_t>(c);
            return g;
        });
    };
    Dev<std::uint32_t> t(uni(400, 300)), b(uni(800, 700));
    Dev<std::uint8_t> y(1920 * kY), uv(1920 * kUV);
    const WeaveJob job{t.p, b.p, y.p, uv.p};
    CUDA_OK(weave_nv12(&job, 1, 1920, 1920, st.s));
    st.sync();
    const auto hy = y.down(), huv = uv.down();
    std::size_t bad = 0;
    for (std::size_t r = 0; r < kY; ++r)
        for (int x = 0; x < 1920; ++x) bad += hy[r * 1920 + x] != (r % 2 == 0 ? 100 : 200);
    CHECK(bad == 0);
    bad = 0;
    for (std::size_t r = 0; r < kUV; ++r)
        for (int x = 0; x < 1920; ++x) bad += huv[r * 1920 + x] != (r % 2 == 0 ? 75 : 175);
    CHECK(bad == 0);
}

TEST_CASE("split == ref (atsitiktinis NV12, p=0 ir p=1, pitch 1920 ir 2048)", "[gpu][media][split]") {
    Stream st;
    std::mt19937 rng(23);
    for (int pitch : {1920, 2048}) {
        const auto hy = random_bytes(rng, pitch * kY), huv = random_bytes(rng, pitch * kUV);
        Dev<std::uint8_t> y(hy), uv(huv);
        for (int p = 0; p < 2; ++p) {
            Dev<std::uint32_t> f(kFieldWords);
            const SplitJob job{y.p, uv.p, f.p};
            CUDA_OK(split_nv12(&job, 1, p, pitch, pitch, st.s));
            st.sync();
            std::vector<std::uint32_t> want(kFieldWords);
            ref::split_nv12(hy.data(), huv.data(), pitch, pitch, p, want.data());
            INFO("pitch " << pitch << " p " << p);
            CHECK(diff_count(f.down(), want) == 0);
        }
    }
}

TEST_CASE("split lauko izoliacija: UV 75/175 → laukų chroma 300/700", "[gpu][media][split]") {
    Stream st;
    std::mt19937 rng(5);
    std::vector<std::uint8_t> huv(1920 * kUV);
    for (std::size_t r = 0; r < kUV; ++r) std::memset(huv.data() + r * 1920, r % 2 == 0 ? 75 : 175, 1920);
    Dev<std::uint8_t> y(random_bytes(rng, 1920 * kY)), uv(huv);
    for (int p = 0; p < 2; ++p) {
        Dev<std::uint32_t> f(kFieldWords);
        const SplitJob job{y.p, uv.p, f.p};
        CUDA_OK(split_nv12(&job, 1, p, 1920, 1920, st.s));
        st.sync();
        const auto h = f.down();
        const int want = p == 0 ? 300 : 700;
        std::size_t bad = 0;
        for (int j = 0; j < kFieldHeight; ++j)
            for (int g = 0; g < kGroupsPerLine; ++g) {
                const Group gr = group_at(h, j, g);
                for (int c = 0; c < 3; ++c) bad += gr.cb[c] != want || gr.cr[c] != want;
            }
        INFO("p " << p);
        CHECK(bad == 0);
    }
}

TEST_CASE("round-trip: laukai → weave → split → identiški", "[gpu][media][roundtrip]") {
    Stream st;
    std::mt19937 rng(99);
    // Y kartotiniai 4 (Y8 = Y/4 tiksliai), chroma kartotinė 4 ir pastovi vertikaliai kiekviename lauke
    std::uniform_int_distribution<int> dy(16, 235), dc(16, 240);
    std::vector<std::vector<std::uint32_t>> fields;
    for (int p = 0; p < 2; ++p) {
        std::vector<int> col_cb(kWidth / 2), col_cr(kWidth / 2);
        for (auto& v : col_cb) v = 4 * dc(rng);
        for (auto& v : col_cr) v = 4 * dc(rng);
        fields.push_back(make_field([&](int, int g) {
            Group gr;
            for (auto& v : gr.y) v = static_cast<std::uint16_t>(4 * dy(rng));
            for (int c = 0; c < 3; ++c) {
                gr.cb[c] = static_cast<std::uint16_t>(col_cb[3 * g + c]);
                gr.cr[c] = static_cast<std::uint16_t>(col_cr[3 * g + c]);
            }
            return gr;
        }));
    }
    Dev<std::uint32_t> t(fields[0]), b(fields[1]), o(kFieldWords);
    Dev<std::uint8_t> y(2048 * kY), uv(2048 * kUV);
    const WeaveJob wj{t.p, b.p, y.p, uv.p};
    CUDA_OK(weave_nv12(&wj, 1, 2048, 2048, st.s));
    for (int p = 0; p < 2; ++p) {
        const SplitJob sj{y.p, uv.p, o.p};
        CUDA_OK(split_nv12(&sj, 1, p, 2048, 2048, st.s));
        st.sync();
        INFO("p " << p);
        CHECK(diff_count(o.down(), fields[p]) == 0);
    }
}

TEST_CASE("logo == ref (visos alfa, pozicijos, abu paritetai)", "[gpu][media][logo]") {
    Stream st;
    std::mt19937 rng(77);
    const ref::Logo L = logo_all_alpha(rng);
    const DevLogo dl(L);
    const auto bg = random_words(rng);  // atsitiktiniai žodžiai, įskaitant bitus 30–31
    Dev<std::uint32_t> src(bg), dst(kFieldWords);
    struct Pos { int x, y; };
    for (Pos pos : {Pos{0, 0}, Pos{1536, 864}, Pos{100, 301}, Pos{702, 77}, Pos{1800, 1000}})
        for (int p = 0; p < 2; ++p) {
            dst.fill(0);
            const LogoJob job{src.p, dst.p};
            CUDA_OK(logo_fields(&job, 1, dl.desc(L, pos.x, pos.y), p, st.s));
            st.sync();
            std::vector<std::uint32_t> want(kFieldWords);
            ref::logo_field(bg.data(), want.data(), L, pos.x, pos.y, p);
            const auto got = dst.down();
            INFO("X " << pos.x << " Y " << pos.y << " p " << p);
            CHECK(diff_count(got, want) == 0);
            CHECK(diff_count(got, bg) > 0);  // logotipas iš tikrųjų uždėtas
        }
}

TEST_CASE("logo vietoje (src == dst) == ref", "[gpu][media][logo]") {
    Stream st;
    std::mt19937 rng(78);
    const ref::Logo L = logo_all_alpha(rng);
    const DevLogo dl(L);
    const auto bg = random_words(rng);
    Dev<std::uint32_t> f(bg);
    const LogoJob job{f.p, f.p};
    CUDA_OK(logo_fields(&job, 1, dl.desc(L, 400, 201), 1, st.s));
    st.sync();
    std::vector<std::uint32_t> want(kFieldWords);
    ref::logo_field(bg.data(), want.data(), L, 400, 201, 1);
    CHECK(diff_count(f.down(), want) == 0);
}

TEST_CASE("logo už ribų: bitas į bitą; a=0 → išėjimas == įėjimas", "[gpu][media][logo]") {
    Stream st;
    std::mt19937 rng(31);
    const ref::Logo L = ref::builtin_logo();
    const DevLogo dl(L);
    const auto bg = random_words(rng);
    Dev<std::uint32_t> src(bg), dst(kFieldWords);
    const int X = 1000, Y = 133;
    for (int p = 0; p < 2; ++p) {
        dst.fill(0);
        const LogoJob job{src.p, dst.p};
        CUDA_OK(logo_fields(&job, 1, dl.desc(L, X, Y), p, st.s));
        st.sync();
        const auto got = dst.down();
        std::uint32_t x_or = 0;
        std::size_t outside = 0;
        for (int j = 0; j < kFieldHeight; ++j) {
            const int ly = 2 * j + p - Y;
            for (int g = 0; g < kGroupsPerLine; ++g) {
                const bool in = ly >= 0 && ly < L.h && 6 * g + 6 > X && 6 * g < X + L.w;
                if (in) continue;
                for (int w = 0; w < 4; ++w) {
                    const std::size_t i = static_cast<std::size_t>(j) * kWordsPerLine + 4 * g + w;
                    x_or |= got[i] ^ bg[i];
                    ++outside;
                }
            }
        }
        INFO("p " << p);
        CHECK(outside > 600000);
        CHECK(x_or == 0);
    }
    ref::Logo zero = L;
    std::fill(zero.a8.begin(), zero.a8.end(), std::uint8_t{0});
    std::fill(zero.ac8.begin(), zero.ac8.end(), std::uint8_t{0});
    const DevLogo dz(zero);
    for (int p = 0; p < 2; ++p) {
        dst.fill(0);
        const LogoJob job{src.p, dst.p};
        CUDA_OK(logo_fields(&job, 1, dz.desc(zero, 0, 0), p, st.s));
        st.sync();
        CHECK(diff_count(dst.down(), bg) == 0);
    }
}

TEST_CASE("logo paritetas: lyginės eilutės baltos → viršutinis laukas, nelyginės juodos → apatinis", "[gpu][media][logo]") {
    Stream st;
    constexpr int W = 384, H = 216, X = 200, Y = 100;
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(W) * H * 4);
    for (int r = 0; r < H; ++r)
        for (int x = 0; x < W; ++x) {
            std::uint8_t* q = rgba.data() + (static_cast<std::size_t>(r) * W + x) * 4;
            q[0] = q[1] = q[2] = r % 2 == 0 ? 255 : 0;
            q[3] = 255;
        }
    ref::Logo L;
    REQUIRE(ref::logo_from_rgba(rgba.data(), W, H, L));
    const DevLogo dl(L);
    std::mt19937 rng(3);
    Dev<std::uint32_t> src(random_legal_field(rng)), dst(kFieldWords);
    for (int p = 0; p < 2; ++p) {
        const LogoJob job{src.p, dst.p};
        CUDA_OK(logo_fields(&job, 1, dl.desc(L, X, Y), p, st.s));
        st.sync();
        const auto got = dst.down();
        const int want_y = p == 0 ? 940 : 64;
        std::size_t bad = 0, checked = 0;
        for (int j = 0; j < kFieldHeight; ++j) {
            const int ly = 2 * j + p - Y;
            if (ly < 0 || ly >= H) continue;
            for (int x = X; x < X + W; ++x) {
                const Group gr = group_at(got, j, x / 6);
                bad += gr.y[x % 6] != want_y;
                if (x % 2 == 0) bad += gr.cb[(x % 6) / 2] != 512 || gr.cr[(x % 6) / 2] != 512;
                ++checked;
            }
        }
        INFO("p " << p);
        CHECK(checked == static_cast<std::size_t>(W) * (H / 2));
        CHECK(bad == 0);
    }
}

TEST_CASE("batch ribos: N=1, 64, 65 == N paleidimų po 1", "[gpu][media][batch]") {
    Stream st;
    std::mt19937 rng(2024);
    const ref::Logo L = logo_all_alpha(rng);
    const DevLogo dl(L);
    const LogoDesc ld = dl.desc(L, 640, 333);
    // 3 skirtingi įėjimai cikliškai (laukai ir NV12 kadrai)
    std::vector<Dev<std::uint32_t>> in_f;
    std::vector<Dev<std::uint8_t>> in_y, in_uv;
    in_f.reserve(6); in_y.reserve(3); in_uv.reserve(3);
    for (int i = 0; i < 6; ++i) in_f.emplace_back(random_legal_field(rng));
    for (int i = 0; i < 3; ++i) {
        in_y.emplace_back(random_bytes(rng, 1920 * kY));
        in_uv.emplace_back(random_bytes(rng, 1920 * kUV));
    }
    Dev<std::uint32_t> one_f(kFieldWords);
    Dev<std::uint8_t> one_y(1920 * kY), one_uv(1920 * kUV);
    for (int n : {1, 64, 65}) {
        std::vector<Dev<std::uint32_t>> out_f;
        std::vector<Dev<std::uint8_t>> out_y, out_uv;
        out_f.reserve(n); out_y.reserve(n); out_uv.reserve(n);
        for (int i = 0; i < n; ++i) {
            out_f.emplace_back(kFieldWords);
            out_y.emplace_back(1920 * kY);
            out_uv.emplace_back(1920 * kUV);
        }
        std::vector<GenJob> gj;
        std::vector<LogoJob> lj;
        std::vector<WeaveJob> wj;
        std::vector<SplitJob> sj;
        for (int i = 0; i < n; ++i) {
            gj.push_back({out_f[i].p, static_cast<std::uint32_t>(i)});
            lj.push_back({in_f[i % 6].p, out_f[i].p});
            wj.push_back({in_f[i % 6].p, in_f[(i + 1) % 6].p, out_y[i].p, out_uv[i].p});
            sj.push_back({in_y[i % 3].p, in_uv[i % 3].p, out_f[i].p});
        }
        std::size_t bad = 0;
        // gen
        CUDA_OK(gen_fields(gj.data(), n, 9, 3, 1001, st.s));
        st.sync();
        for (int i = 0; i < n; ++i) {
            const GenJob j1{one_f.p, gj[i].flow};
            CUDA_OK(gen_fields(&j1, 1, 9, 3, 1001, st.s));
            st.sync();
            bad += diff_count(out_f[i].down(), one_f.down());
        }
        INFO("n " << n);
        CHECK(bad == 0);
        // logo
        bad = 0;
        CUDA_OK(logo_fields(lj.data(), n, ld, 1, st.s));
        st.sync();
        for (int i = 0; i < n; ++i) {
            const LogoJob j1{lj[i].src, one_f.p};
            CUDA_OK(logo_fields(&j1, 1, ld, 1, st.s));
            st.sync();
            bad += diff_count(out_f[i].down(), one_f.down());
        }
        CHECK(bad == 0);
        // weave
        bad = 0;
        CUDA_OK(weave_nv12(wj.data(), n, 1920, 1920, st.s));
        st.sync();
        for (int i = 0; i < n; ++i) {
            const WeaveJob j1{wj[i].top, wj[i].bottom, one_y.p, one_uv.p};
            CUDA_OK(weave_nv12(&j1, 1, 1920, 1920, st.s));
            st.sync();
            bad += diff_count(out_y[i].down(), one_y.down()) + diff_count(out_uv[i].down(), one_uv.down());
        }
        CHECK(bad == 0);
        // split
        bad = 0;
        CUDA_OK(split_nv12(sj.data(), n, 0, 1920, 1920, st.s));
        st.sync();
        for (int i = 0; i < n; ++i) {
            const SplitJob j1{sj[i].src_y, sj[i].src_uv, one_f.p};
            CUDA_OK(split_nv12(&j1, 1, 0, 1920, 1920, st.s));
            st.sync();
            bad += diff_count(out_f[i].down(), one_f.down());
        }
        CHECK(bad == 0);
        // ir paskutinis batch'o darbas == CPU etalonas (patikra, kad abu keliai ne vienodai klaidingi)
        std::vector<std::uint32_t> want(kFieldWords);
        ref::split_nv12(in_y[(n - 1) % 3].down().data(), in_uv[(n - 1) % 3].down().data(), 1920, 1920, 0, want.data());
        CHECK(diff_count(out_f[n - 1].down(), want) == 0);
    }
}

TEST_CASE("nelygiuoti adresai (ne 16 B) duoda tą patį rezultatą", "[gpu][media][align]") {
    Stream st;
    std::mt19937 rng(8);
    const auto bg = random_words(rng);
    Dev<std::uint32_t> src(kFieldWords + 1), dst(kFieldWords + 1);
    CUDA_OK(cudaMemcpy(src.p + 1, bg.data(), kFieldBytes, cudaMemcpyHostToDevice));
    const ref::Logo L = ref::builtin_logo();
    const DevLogo dl(L);
    const LogoJob job{src.p + 1, dst.p + 1};
    CUDA_OK(logo_fields(&job, 1, dl.desc(L, 50, 21), 0, st.s));
    const GenJob gj{src.p + 1, 4};
    st.sync();
    std::vector<std::uint32_t> got(kFieldWords), want(kFieldWords);
    CUDA_OK(cudaMemcpy(got.data(), dst.p + 1, kFieldBytes, cudaMemcpyDeviceToHost));
    ref::logo_field(bg.data(), want.data(), L, 50, 21, 0);
    CHECK(diff_count(got, want) == 0);
    CUDA_OK(gen_fields(&gj, 1, 1, 2, 5, st.s));
    st.sync();
    CUDA_OK(cudaMemcpy(got.data(), src.p + 1, kFieldBytes, cudaMemcpyDeviceToHost));
    ref::gen_field(GenParams{1, 4, 2}, 5, want.data());
    CHECK(diff_count(got, want) == 0);
}

TEST_CASE("netinkami argumentai → cudaErrorInvalidValue", "[gpu][media][args]") {
    Stream st;
    const ref::Logo L = ref::builtin_logo();
    const DevLogo dl(L);
    Dev<std::uint32_t> f(kFieldWords);
    const LogoJob lj{f.p, f.p};
    CHECK(logo_fields(&lj, 1, dl.desc(L, 1, 0), 0, st.s) == cudaErrorInvalidValue);  // X nelyginis
    CHECK(logo_fields(&lj, 1, dl.desc(L, 0, 0), 2, st.s) == cudaErrorInvalidValue);  // paritetas
    const LogoJob bad{nullptr, f.p};
    CHECK(logo_fields(&bad, 1, dl.desc(L, 0, 0), 0, st.s) == cudaErrorInvalidValue);
    const SplitJob sj{nullptr, nullptr, f.p};
    CHECK(split_nv12(&sj, 1, 0, 1920, 1920, st.s) == cudaErrorInvalidValue);
    const WeaveJob wj{f.p, f.p, nullptr, nullptr};
    CHECK(weave_nv12(&wj, 1, 1920, 1920, st.s) == cudaErrorInvalidValue);
    CHECK(weave_nv12(&wj, 1, 1000, 1920, st.s) == cudaErrorInvalidValue);
    CHECK(gen_fields(nullptr, 0, 0, 0, 0, st.s) == cudaSuccess);  // N=0 — nieko
    CHECK(gen_fields(nullptr, -1, 0, 0, 0, st.s) == cudaErrorInvalidValue);
    CHECK(cudaGetLastError() == cudaSuccess);
}

// --- P2: formatui neutralūs kernel'iai (CUDA == nepriklausomas etalonas, 0 skirtingų baitų) ----------------------------
namespace {
struct Case {
    Dim d;
    int field;  // −1 kadras, 0 / 1 laukas
};
// 1080 (laukai ir kadras), 720p ir 2160p (tik kadras): ir pilnos, ir dalinės v210 grupės (1280 = 213 · 6 + 2).
const Case kCases[] = {{{1920, 1080}, 0}, {{1920, 1080}, 1}, {{1920, 1080}, -1}, {{1280, 720}, -1}, {{3840, 2160}, -1}};
std::size_t gwords(const Case& c) { return geom_of(c.d, c.field).grain_words(); }
}  // namespace

TEST_CASE("media cuda: gen == ref (visi matmenys, laukas ir kadras)", "[gpu][media][format][gen]") {
    Stream st;
    for (const Case& c : kCases)
        for (std::uint64_t index : {0ull, 1ull, 50ull, 777ull}) {
            if (c.field >= 0 && static_cast<int>(index & 1) != c.field) continue;
            Dev<std::uint32_t> f(gwords(c));
            f.fill(0xAB);
            const GenJob job{f.p, 3};
            CUDA_OK(gen_fields(&job, 1, 0x1234, 3, index, c.d, c.field >= 0, st.s));
            st.sync();
            std::vector<std::uint32_t> want(gwords(c));
            ref::gen_field({0x1234, 3, 3}, index, c.d, c.field >= 0, want.data());
            INFO("w " << c.d.w << " field " << c.field << " index " << index);
            CHECK(diff_count(f.down(), want) == 0);
        }
}

TEST_CASE("media cuda: logo == ref (visi matmenys, kraštai)", "[gpu][media][format][logo]") {
    Stream st;
    std::mt19937 rng(91);
    const ref::Logo L = logo_all_alpha(rng);
    const DevLogo dl(L);
    for (const Case& c : kCases) {
        const auto bg = random_words(rng, gwords(c));
        Dev<std::uint32_t> src(bg), dst(gwords(c));
        for (const int x : {0, c.d.w - L.w, c.d.w - 100, 700})
            for (const int y : {0, c.d.h - L.h, c.d.h - 50}) {
                dst.fill(0);
                const LogoJob job{src.p, dst.p};
                CUDA_OK(logo_fields(&job, 1, dl.desc(L, x, y), c.d, c.field, st.s));
                st.sync();
                std::vector<std::uint32_t> want(gwords(c));
                ref::logo_field(bg.data(), want.data(), L, x, y, c.d, c.field);
                INFO("w " << c.d.w << " field " << c.field << " X " << x << " Y " << y);
                CHECK(diff_count(dst.down(), want) == 0);
            }
    }
}

TEST_CASE("media cuda: weave == ref (interlaced ir progresyvus, visi matmenys)", "[gpu][media][format][weave]") {
    Stream st;
    std::mt19937 rng(5);
    for (const Case& c : kCases) {
        if (c.field == 1) continue;
        const bool inter = c.field >= 0;
        for (const int pitch : {c.d.w, c.d.w + 128}) {
            const auto t = random_words(rng, gwords(c)), b = random_words(rng, gwords(c));
            Dev<std::uint32_t> dt(t), db(b);
            Dev<std::uint8_t> y(static_cast<std::size_t>(pitch) * c.d.h), uv(static_cast<std::size_t>(pitch) * c.d.h / 2);
            y.fill(7);
            uv.fill(7);
            const WeaveJob job{dt.p, inter ? db.p : dt.p, y.p, uv.p};
            CUDA_OK(weave_nv12(&job, 1, c.d, inter, pitch, pitch, st.s));
            st.sync();
            std::vector<std::uint8_t> wy(y.n, 7), wuv(uv.n, 7);
            ref::weave_nv12(t.data(), inter ? b.data() : nullptr, wy.data(), wuv.data(), pitch, pitch, c.d, inter);
            INFO("w " << c.d.w << " interlaced " << inter << " pitch " << pitch);
            CHECK(diff_count(y.down(), wy) == 0);
            CHECK(diff_count(uv.down(), wuv) == 0);
        }
    }
}

TEST_CASE("media cuda: split_nv12_progressive == ref (1920×1080, 1280×720, 3840×2160; pitch w ir w+128)", "[gpu][media][format][split]") {
    Stream st;
    std::mt19937 rng(29);
    for (const Dim d : {Dim{1920, 1080}, Dim{1280, 720}, Dim{3840, 2160}})
        for (const int pitch : {d.w, d.w + 128}) {
            const auto hy = random_bytes(rng, static_cast<std::size_t>(pitch) * d.h);
            const auto huv = random_bytes(rng, static_cast<std::size_t>(pitch) * d.h / 2);
            Dev<std::uint8_t> y(hy), uv(huv);
            const std::size_t n = geom_of(d, -1).grain_words();
            Dev<std::uint32_t> f(n);
            f.fill(0xCD);
            const SplitJob job{y.p, uv.p, f.p};
            CUDA_OK(split_nv12_progressive(&job, 1, d, pitch, pitch, st.s));
            st.sync();
            std::vector<std::uint32_t> want(n);
            ref::split_nv12_progressive(hy.data(), huv.data(), pitch, pitch, d, want.data());
            INFO("w " << d.w << " pitch " << pitch);
            CHECK(diff_count(f.down(), want) == 0);
        }
}

TEST_CASE("media cuda: split FrameSited laukui == ref (abu paritetai), FieldSited su kDim1080 == senas", "[gpu][media][format][split]") {
    Stream st;
    std::mt19937 rng(31);
    for (const int pitch : {1920, 2048}) {
        const auto hy = random_bytes(rng, static_cast<std::size_t>(pitch) * kY), huv = random_bytes(rng, static_cast<std::size_t>(pitch) * kUV);
        Dev<std::uint8_t> y(hy), uv(huv);
        for (int p = 0; p < 2; ++p)
            for (const ChromaMode m : {ChromaMode::FrameSited, ChromaMode::FieldSited}) {
                Dev<std::uint32_t> f(kFieldWords);
                const SplitJob job{y.p, uv.p, f.p};
                CUDA_OK(split_nv12(&job, 1, kDim1080, p, m, pitch, pitch, st.s));
                st.sync();
                std::vector<std::uint32_t> want(kFieldWords);
                ref::split_nv12(hy.data(), huv.data(), pitch, pitch, kDim1080, p, m, want.data());
                INFO("pitch " << pitch << " p " << p << " frame " << (m == ChromaMode::FrameSited));
                CHECK(diff_count(f.down(), want) == 0);
                if (m == ChromaMode::FieldSited) {
                    std::vector<std::uint32_t> old(kFieldWords);
                    ref::split_nv12(hy.data(), huv.data(), pitch, pitch, p, old.data());
                    CHECK(diff_count(want, old) == 0);
                }
            }
    }
}

TEST_CASE("media cuda: formatui neutralūs argumentai → cudaErrorInvalidValue", "[gpu][media][format][args]") {
    Stream st;
    Dev<std::uint32_t> f(kFieldWords);
    Dev<std::uint8_t> y(1920 * 1080), uv(1920 * 540);
    const SplitJob sj{y.p, uv.p, f.p};
    CHECK(split_nv12(&sj, 1, kDim1080, -1, ChromaMode::FieldSited, 1920, 1920, st.s) == cudaErrorInvalidValue);
    CHECK(split_nv12_progressive(&sj, 1, Dim{1921, 1080}, 1920, 1920, st.s) == cudaErrorInvalidValue);
    CHECK(split_nv12_progressive(&sj, 1, Dim{1920, 1080}, 1900, 1920, st.s) == cudaErrorInvalidValue);
    CHECK(split_nv12(&sj, 1, Dim{1920, 1082}, 0, ChromaMode::FieldSited, 1920, 1920, st.s) == cudaErrorInvalidValue);
    const GenJob gj{f.p, 0};
    CHECK(gen_fields(&gj, 1, 0, 0, 0, Dim{0, 1080}, false, st.s) == cudaErrorInvalidValue);
    const WeaveJob wj{f.p, f.p, y.p, uv.p};
    CHECK(weave_nv12(&wj, 1, Dim{1280, 722}, true, 1280, 1280, st.s) == cudaErrorInvalidValue);  // interlaced — h % 4 != 0
}

TEST_CASE("media cuda: gen su grain'o dažniu (blyksnis kartą per sekundę) == ref, 720p ir 2160p", "[gpu][media][format][gen]") {
    Stream st;
    struct R {
        std::uint32_t num, den;
    };
    for (const Dim d : {Dim{1280, 720}, Dim{3840, 2160}})
        for (const R r : {R{25, 1}, R{60000, 1001}})
            for (std::uint64_t index : {0ull, 25ull, 60ull, 120ull}) {
                const std::size_t n = geom_of(d, -1).grain_words();
                Dev<std::uint32_t> f(n);
                const GenJob job{f.p, 2};
                CUDA_OK(gen_fields(&job, 1, 77, 2, index, d, false, r.num, r.den, st.s));
                st.sync();
                GenParams g{77, 2, 2};
                g.rate_num = r.num;
                g.rate_den = r.den;
                std::vector<std::uint32_t> want(n);
                ref::gen_field(g, index, d, false, want.data());
                INFO("w " << d.w << " rate " << r.num << "/" << r.den << " index " << index);
                CHECK(diff_count(f.down(), want) == 0);
            }
}
