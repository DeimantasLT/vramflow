// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Variantas C: eilutinis sveikųjų skaičių RGBA -> v210a ir composite (žr. gfx_cpu.hpp). Matematika — pinned formulės
// (planas §„Matematika“ + R1/R7/R8/R10), rezultatas bitas į bitą lygus gfx::ref. Vidinės eilutės — uint16_t masyvai
// (atskiri Y / Cb / Cr / ab), kad ciklai būtų vektorizuojami (-O3 -march=x86-64-v3).
// Skaičių ribos (kodėl 32 bitų int pakanka): fill ≤ 1023, ae, ab ≤ 876 -> n = ae·A + ab·(A−ae) ≤ A² = 767 376,
// num = f·ae·A + b·ab·(A−ae) + n/2 ≤ 1023·n + n/2 < 2³¹. Dalyba num/n: double dalyba duoda tikslų floor, nes tikrasis
// dalmuo nuo sveikojo skiriasi ≥ 1/n ≈ 1,3e-6 ≫ 2⁻⁵³ santykinės paklaidos.
#include "gfx/cpu/gfx_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace vramflow::gfx::cpu {
namespace {

using u16 = std::uint16_t;
using u32 = std::uint32_t;
constexpr int A = 876;  // alfa skalė 0…A (key 64…940)
constexpr int kChunkLines = 16;

inline u32 ld32(const std::uint8_t* p) { u32 v; std::memcpy(&v, p, 4); return v; }
inline void st32(std::uint8_t* p, u32 v) { std::memcpy(p, &v, 4); }

// --- v210 / key pakavimas eilutėmis -------------------------------------------------------------------------------
// v210 grupė (6 px, 16 B): w0 = Cb0|Y0|Cr0, w1 = Y1|Cb1|Y2, w2 = Cr1|Y3|Cb2, w3 = Y4|Cr2|Y5 (laukai po 10 bitų, postūmiai 0/10/20).
void unpack_fill(const std::uint8_t* line, int groups, u16* __restrict y, u16* __restrict cb, u16* __restrict cr) {
    for (int g = 0; g < groups; ++g) {
        const u32 w0 = ld32(line + 16 * g), w1 = ld32(line + 16 * g + 4), w2 = ld32(line + 16 * g + 8), w3 = ld32(line + 16 * g + 12);
        u16* yy = y + 6 * g;
        u16* bb = cb + 3 * g;
        u16* rr = cr + 3 * g;
        bb[0] = w0 & 0x3FF;         yy[0] = (w0 >> 10) & 0x3FF; rr[0] = (w0 >> 20) & 0x3FF;
        yy[1] = w1 & 0x3FF;         bb[1] = (w1 >> 10) & 0x3FF; yy[2] = (w1 >> 20) & 0x3FF;
        rr[1] = w2 & 0x3FF;         yy[3] = (w2 >> 10) & 0x3FF; bb[2] = (w2 >> 20) & 0x3FF;
        yy[4] = w3 & 0x3FF;         rr[2] = (w3 >> 10) & 0x3FF; yy[5] = (w3 >> 20) & 0x3FF;
    }
}
// Masyvai turi būti bent groups·6 / groups·3 ilgio, nenaudoti mėginiai = 0.
void pack_fill(std::uint8_t* line, int groups, const u16* __restrict y, const u16* __restrict cb, const u16* __restrict cr) {
    for (int g = 0; g < groups; ++g) {
        const u16* yy = y + 6 * g;
        const u16* bb = cb + 3 * g;
        const u16* rr = cr + 3 * g;
        st32(line + 16 * g, (bb[0] & 0x3FFu) | ((yy[0] & 0x3FFu) << 10) | ((rr[0] & 0x3FFu) << 20));
        st32(line + 16 * g + 4, (yy[1] & 0x3FFu) | ((bb[1] & 0x3FFu) << 10) | ((yy[2] & 0x3FFu) << 20));
        st32(line + 16 * g + 8, (rr[1] & 0x3FFu) | ((yy[3] & 0x3FFu) << 10) | ((bb[2] & 0x3FFu) << 20));
        st32(line + 16 * g + 12, (yy[4] & 0x3FFu) | ((rr[2] & 0x3FFu) << 10) | ((yy[5] & 0x3FFu) << 20));
    }
}
// Key eilutė -> k[3·words]; bitai 30–31 ignoruojami.
void unpack_key(const std::uint8_t* line, int words, u16* __restrict k) {
    for (int i = 0; i < words; ++i) {
        const u32 w = ld32(line + 4 * i);
        k[3 * i] = w & 0x3FF;
        k[3 * i + 1] = (w >> 10) & 0x3FF;
        k[3 * i + 2] = (w >> 20) & 0x3FF;
    }
}
// key[x] = v[x] + add, x < w; nenaudoti mėginiai ir bitai 30–31 = 0.
void pack_key(std::uint8_t* line, int w, const u16* __restrict v, int add) {
    const int full = w / 3;
    for (int i = 0; i < full; ++i)
        st32(line + 4 * i, ((v[3 * i] + add) & 0x3FFu) | (((v[3 * i + 1] + add) & 0x3FFu) << 10) | (((v[3 * i + 2] + add) & 0x3FFu) << 20));
    const int rem = w - 3 * full;
    if (rem) {
        u32 wd = (v[3 * full] + add) & 0x3FFu;
        if (rem == 2) wd |= ((v[3 * full + 1] + add) & 0x3FFu) << 10;
        st32(line + 4 * full, wd);
    }
}

// --- maišymas ---------------------------------------------------------------------------------------------------
// v210a išėjimo formulė (planas): n = ae·A + ab·(A−ae); out = n == 0 ? def : (f·ae·A + b·ab·(A−ae) + n/2) / n, apkirpta 4…1019;
// ao = (n + A/2) / A. kStore: ab[i] = ao (luma); chroma ab = abc (tik skaitomas).
template <bool kStore>
void mix_a(const u16* __restrict f, const u16* __restrict ae, u16* __restrict b, u16* __restrict ab, int n, int def) {
    for (int i = 0; i < n; ++i) {
        const int e = ae[i], a = ab[i];
        const int nn = e * A + a * (A - e);
        const int num = f[i] * e * A + b[i] * a * (A - e) + nn / 2;
        const int q = static_cast<int>(static_cast<double>(num) / static_cast<double>(nn ? nn : 1));
        b[i] = static_cast<u16>(nn == 0 ? def : std::min(1019, std::max(4, q)));
        if (kStore) ab[i] = static_cast<u16>((nn + A / 2) / A);
    }
}
// v210 išėjimas: out = (b·(A−ae) + f·ae + A/2) / A, apkirpta 4…1019.
void mix_v(const u16* __restrict f, const u16* __restrict ae, u16* __restrict b, int n) {
    for (int i = 0; i < n; ++i) {
        const int e = ae[i];
        const int v = (b[i] * (A - e) + f[i] * e + A / 2) / A;
        b[i] = static_cast<u16>(std::min(1019, std::max(4, v)));
    }
}

// --- composite ----------------------------------------------------------------------------------------------------
struct Scratch {
    std::vector<u16> y, cb, cr, ab;        // būsena (išėjimo eilutė)
    std::vector<u16> ly, lcb, lcr, lk, ae;  // sluoksnio eilutė
    std::vector<u16> ac, abc;              // chroma alfa (sluoksnio, sukaupta)
    static void need(std::vector<u16>& v, std::size_t n) { if (v.size() < n) v.resize(n); }
};

void process_row(const CompositeJob& job, int r, Scratch& s) {
    const Raster& out_r = job.out_r;
    const int W = out_r.w;
    const int groups = (W + 5) / 6, kwords = (W + 2) / 3;
    const bool oa = out_r.alpha;
    Scratch::need(s.y, 6 * groups);
    Scratch::need(s.cb, 3 * groups);
    Scratch::need(s.cr, 3 * groups);
    Scratch::need(s.ab, 3 * kwords);
    Scratch::need(s.lk, 3 * kwords);

    // 1. Pradinė būsena. Nenaudoti mėginiai (grupės uodega) = 0.
    if (job.base) {
        const Raster& br = job.base_r;
        unpack_fill(job.base + static_cast<std::size_t>(r) * fill_line_bytes(W), groups, s.y.data(), s.cb.data(), s.cr.data());
        if (oa) {
            if (br.alpha) {
                unpack_key(job.base + fill_bytes(br) + static_cast<std::size_t>(r) * key_line_bytes(W), kwords, s.lk.data());
                for (int x = 0; x < W; ++x) s.ab[x] = static_cast<u16>(std::min(A, std::max(0, static_cast<int>(s.lk[x]) - 64)));
            } else {
                std::fill_n(s.ab.begin(), W, static_cast<u16>(A));  // v210 bazė nepermatoma
            }
        }
    } else {
        std::fill_n(s.y.begin(), W, u16{64});
        std::fill_n(s.cb.begin(), W / 2, u16{512});
        std::fill_n(s.cr.begin(), W / 2, u16{512});
        if (oa) std::fill_n(s.ab.begin(), W, u16{0});
    }
    std::fill(s.y.begin() + W, s.y.begin() + 6 * groups, u16{0});
    std::fill(s.cb.begin() + W / 2, s.cb.begin() + 3 * groups, u16{0});
    std::fill(s.cr.begin() + W / 2, s.cr.begin() + 3 * groups, u16{0});

    // 2. Sluoksniai iš apačios į viršų (R8).
    for (int l = 0; l < job.k; ++l) {
        const Layer& L = job.layers[l];
        const int ly = r - (out_r.interlaced ? L.y / 2 : L.y);  // sluoksnio (lauko) eilutė
        if (ly < 0 || ly >= grain_lines(L.r)) continue;
        const int cw = std::min(L.r.w, W - L.x);  // nukirpimas (x, w lyginiai -> cw lyginis)
        if (cw <= 0) continue;
        const int lgroups = (cw + 5) / 6, lwords = (cw + 2) / 3, cpairs = cw / 2;
        Scratch::need(s.ly, 6 * lgroups);
        Scratch::need(s.lcb, 3 * lgroups);
        Scratch::need(s.lcr, 3 * lgroups);
        Scratch::need(s.lk, 3 * lwords);
        Scratch::need(s.ae, 6 * lgroups);
        Scratch::need(s.ac, 3 * lgroups);
        Scratch::need(s.abc, 3 * lgroups);
        unpack_fill(L.grain + static_cast<std::size_t>(ly) * fill_line_bytes(L.r.w), lgroups, s.ly.data(), s.lcb.data(), s.lcr.data());

        const int o8 = opacity_to_o8(L.opacity);
        if (L.r.alpha) {
            unpack_key(L.grain + fill_bytes(L.r) + static_cast<std::size_t>(ly) * key_line_bytes(L.r.w), lwords, s.lk.data());
            for (int x = 0; x < cw; ++x) {
                const int a = std::min(A, std::max(0, static_cast<int>(s.lk[x]) - 64));
                s.ae[x] = static_cast<u16>((a * o8 + 127) / 255);
            }
        } else {
            std::fill_n(s.ae.begin(), cw, static_cast<u16>((A * o8 + 127) / 255));  // be key: a = A
        }

        // Chroma pirma, su ab PRIEŠ šio sluoksnio atnaujinimą.
        for (int i = 0; i < cpairs; ++i) s.ac[i] = static_cast<u16>((s.ae[2 * i] + s.ae[2 * i + 1] + 1) >> 1);
        u16* cbo = s.cb.data() + L.x / 2;
        u16* cro = s.cr.data() + L.x / 2;
        if (oa) {
            const u16* abx = s.ab.data() + L.x;
            for (int i = 0; i < cpairs; ++i) s.abc[i] = static_cast<u16>((abx[2 * i] + abx[2 * i + 1] + 1) >> 1);
            mix_a<false>(s.lcb.data(), s.ac.data(), cbo, s.abc.data(), cpairs, 512);
            mix_a<false>(s.lcr.data(), s.ac.data(), cro, s.abc.data(), cpairs, 512);
            mix_a<true>(s.ly.data(), s.ae.data(), s.y.data() + L.x, s.ab.data() + L.x, cw, 64);
        } else {
            mix_v(s.lcb.data(), s.ac.data(), cbo, cpairs);
            mix_v(s.lcr.data(), s.ac.data(), cro, cpairs);
            mix_v(s.ly.data(), s.ae.data(), s.y.data() + L.x, cw);
        }
    }

    // 3. Supakuoti; visas eilutės plotas perrašomas (fill uodega ir key padding = 0).
    std::uint8_t* fl = job.out + static_cast<std::size_t>(r) * fill_line_bytes(W);
    pack_fill(fl, groups, s.y.data(), s.cb.data(), s.cr.data());
    std::memset(fl + 16 * groups, 0, fill_line_bytes(W) - 16 * static_cast<std::size_t>(groups));
    if (oa) pack_key(job.out + fill_bytes(out_r) + static_cast<std::size_t>(r) * key_line_bytes(W), W, s.ab.data(), 64);
}

}  // namespace

bool composite_batch(const CompositeJob* jobs, int n, int parity, mxlv::func::ThreadPool* pool) {
    if (n < 0 || (n > 0 && !jobs)) return false;
    // R11 / R13: bent vienas netinkamas job'as -> nieko nerašoma, false (kaip etalonas ir CUDA).
    for (int i = 0; i < n; ++i)
        if (!composite_job_valid(jobs[i], parity)) return false;
    struct Task { int job, row0; };
    // Užduočių sąrašas — kvietėjo gijos scratch (be malloc karštame kelyje po apšilimo). Nuoroda, o ne pavadinimas,
    // lambda'oje: thread_local pavadinimas pool darbinėje gijoje rodytų į ŠIOS gijos kopiją.
    thread_local std::vector<Task> scratch;
    scratch.clear();
    std::vector<Task>& tasks = scratch;
    for (int i = 0; i < n; ++i) {
        const int lines = grain_lines(jobs[i].out_r);
        for (int r0 = 0; r0 < lines; r0 += kChunkLines) tasks.push_back({i, r0});
    }
    auto body = [&](int t) {
        thread_local Scratch s;
        const CompositeJob& j = jobs[tasks[t].job];
        const int end = std::min(grain_lines(j.out_r), tasks[t].row0 + kChunkLines);
        for (int r = tasks[t].row0; r < end; ++r) process_row(j, r, s);
    };
    const int nt = static_cast<int>(tasks.size());
    if (pool && nt > 1)
        pool->run(nt, body);
    else
        for (int t = 0; t < nt; ++t) body(t);
    return true;
}

bool composite(const std::uint8_t* base, const Raster& base_r, const Layer* layers, int k, const Raster& out_r, int parity,
               std::uint8_t* out, mxlv::func::ThreadPool* pool) {
    const CompositeJob job{base, base_r, layers, k, out_r, out};
    return composite_batch(&job, 1, parity, pool);
}

// --- RGBA -> v210a ------------------------------------------------------------------------------------------------
namespace {

struct RgbaScratch {
    std::vector<int> r, g, b, a;
    std::vector<u16> y, cb, cr, key, cbp, crp;
};

// RI — R baito vieta pikselyje (0 = RGBA, 2 = BGRA); B — 2 - RI. Šablonas: vidinis ciklas be šakos kanalų tvarkai.
template <int RI>
void convert_row(const std::uint8_t* __restrict src, int w, AlphaIn mode, RgbaScratch& s) {
    for (int x = 0; x < w; ++x) {
        const int a = src[4 * x + 3];
        s.a[x] = a;
        if (mode == AlphaIn::Premultiplied) {
            // c' = a == 0 ? 0 : min(255, (c·255 + a/2) / a)
            const int ad = a ? a : 1, half = a / 2;
            s.r[x] = a ? std::min(255, (src[4 * x + RI] * 255 + half) / ad) : 0;
            s.g[x] = a ? std::min(255, (src[4 * x + 1] * 255 + half) / ad) : 0;
            s.b[x] = a ? std::min(255, (src[4 * x + 2 - RI] * 255 + half) / ad) : 0;
        } else {
            s.r[x] = src[4 * x + RI];
            s.g[x] = src[4 * x + 1];
            s.b[x] = src[4 * x + 2 - RI];
        }
    }
    // BT.709 siauras diapazonas, 10 bitų — TA PATI double aritmetika (tokia pati operacijų tvarka, be FMA: žr. cmake
    // -ffp-contract=off) kaip media::ref::rgb_to_ycbcr10, todėl rezultatas bitas į bitą toks pat.
    const int* __restrict rr = s.r.data();
    const int* __restrict gg = s.g.data();
    const int* __restrict bb = s.b.data();
    u16* __restrict yo = s.y.data();
    u16* __restrict cbo = s.cbp.data();
    u16* __restrict cro = s.crp.data();
    for (int x = 0; x < w; ++x) {
        const double R = rr[x] / 255.0, G = gg[x] / 255.0, B = bb[x] / 255.0;
        const double Yp = 0.2126 * R + 0.7152 * G + 0.0722 * B;
        const int y = static_cast<int>(std::floor(64.0 + 876.0 * Yp + 0.5));
        const int cb = static_cast<int>(std::floor(512.0 + 896.0 * (B - Yp) / 1.8556 + 0.5));
        const int cr = static_cast<int>(std::floor(512.0 + 896.0 * (R - Yp) / 1.5748 + 0.5));
        yo[x] = static_cast<u16>(std::min(940, std::max(64, y)));
        cbo[x] = static_cast<u16>(std::min(960, std::max(64, cb)));
        cro[x] = static_cast<u16>(std::min(960, std::max(64, cr)));
    }
    for (int x = 0; x < w; ++x) s.key[x] = static_cast<u16>((s.a[x] * A + 127) / 255);  // + 64 pakuojant
    // Chroma pora: alfa svertinis vidurkis; a0 + a1 == 0 -> 512.
    for (int i = 0; i < w / 2; ++i) {
        const int a0 = s.a[2 * i], a1 = s.a[2 * i + 1], sum = a0 + a1;
        const int sd = sum ? sum : 1;
        const int nb = a0 * s.cbp[2 * i] + a1 * s.cbp[2 * i + 1] + sum / 2;
        const int nr = a0 * s.crp[2 * i] + a1 * s.crp[2 * i + 1] + sum / 2;
        s.cb[i] = static_cast<u16>(sum ? nb / sd : 512);
        s.cr[i] = static_cast<u16>(sum ? nr / sd : 512);
    }
}

}  // namespace

bool rgba_to_v210a(const std::uint8_t* rgba, int w, int h, std::size_t stride, AlphaIn mode, const Raster& r, int parity,
                   std::uint8_t* grain, PixelOrder order) {
    if (!rgba || !grain || !raster_ok(r) || !r.alpha) return false;
    if (r.w != w || r.h != h) return false;
    if (stride < static_cast<std::size_t>(w) * 4) return false;
    if (r.interlaced && parity != 0 && parity != 1) return false;

    const int groups = (w + 5) / 6, kwords = (w + 2) / 3;
    RgbaScratch s;
    s.r.resize(w); s.g.resize(w); s.b.resize(w); s.a.resize(w);
    s.y.assign(6 * groups, 0); s.cb.assign(3 * groups, 0); s.cr.assign(3 * groups, 0);
    s.key.assign(3 * kwords, 0); s.cbp.resize(w); s.crp.resize(w);
    std::uint8_t* const key_plane = grain + fill_bytes(r);
    const int lines = grain_lines(r);
    for (int j = 0; j < lines; ++j) {
        const int frame_row = r.interlaced ? 2 * j + parity : j;
        if (order == PixelOrder::BGRA) convert_row<2>(rgba + static_cast<std::size_t>(frame_row) * stride, w, mode, s);
        else convert_row<0>(rgba + static_cast<std::size_t>(frame_row) * stride, w, mode, s);
        // s.y[w..], s.cb[w/2..] lieka 0 (priskirta vieną kartą, convert_row rašo tik tikrus mėginius).
        std::uint8_t* fl = grain + static_cast<std::size_t>(j) * fill_line_bytes(w);
        pack_fill(fl, groups, s.y.data(), s.cb.data(), s.cr.data());
        std::memset(fl + 16 * groups, 0, fill_line_bytes(w) - 16 * static_cast<std::size_t>(groups));
        pack_key(key_plane + static_cast<std::size_t>(j) * key_line_bytes(w), w, s.key.data(), 64);
    }
    return true;
}

}  // namespace vramflow::gfx::cpu
