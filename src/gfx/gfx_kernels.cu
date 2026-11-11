// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// CUDA RGBA -> v210a ir batch kompozitorius (žr. gfx_kernels.hpp). Viena gija = viena v210 grupė (6 px) vienoje išėjimo
// eilutėje: grupės iš eilės -> sujungtos 16 B prieigos. Matematika — pinned sveikųjų skaičių formulės (R8), išskyrus
// RGB->YCbCr, kuri yra double aritmetika ir privalo sutapti su media::ref::rgb_to_ycbcr10 bitas į bitą: čia naudojami
// tik __dmul_rn / __dadd_rn / __ddiv_rn / floor (jų nuo -fmad nepriklausomai nesujungia į FMA).
// Aprašai (darbai + sluoksniai) perduodami __grid_constant__ kernel parametru: du dydžiai (maža ≤ 4 KiB tipiniam n ≤ 8,
// didelė iki kMaxBatch darbų / kMaxBatchLayers sluoksnių), kad mažo batch'o paleidimas nemokėtų už 23 KiB parametrų.
#include "gfx/gfx_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "media/v210.hpp"

namespace vramflow::gfx::cuda {
namespace {

using u32 = std::uint32_t;
using mxlv::media::Group;
using mxlv::media::pack_group;
using mxlv::media::unpack_group;

constexpr int A = 876;  // alfa skalė 0…A (= ref::kAlphaScale)
constexpr int kThreads = 128;

// --- Host pusė: geometrija ir aprašai ----------------------------------------------------------------------------------
bool aligned(const void* p, std::uintptr_t a) { return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0; }

// Device pusės darbo aprašas (kompaktiškas). flags: bit0 out alpha, bit1 yra bazė, bit2 bazė su key, bit3 base/out 16 B lygiuoti.
struct DJob {
    const std::uint8_t* base;
    std::uint8_t* out;
    int w, lines, flags, lbeg, lcnt;
};
// flags: bit0 sluoksnis su key, bit1 grain'as 16 B lygiuotas.
struct DLayer {
    const std::uint8_t* grain;
    int w, lines, x, yoff, o8, flags;
};
constexpr int kJobAlpha = 1, kJobBase = 2, kJobBaseAlpha = 4, kJobVec = 8;
constexpr int kLayerAlpha = 1, kLayerVec = 2;

template <int MJ, int ML>
struct DBatch {
    DJob job[MJ];
    DLayer layer[ML];
    int n;
};
constexpr int kSmallJobs = 8, kSmallLayers = 64;
using Small = DBatch<kSmallJobs, kSmallLayers>;
using Large = DBatch<kMaxBatch, kMaxBatchLayers>;
static_assert(sizeof(Small) <= 4096, "maža partija turi tilpti į klasikinę 4 KiB parametrų ribą");
static_assert(sizeof(Large) <= 32764, "didelė partija viršija CUDA 12.1+ kernel parametrų ribą");

// Pilna užduoties patikra: bendra gfx::composite_job_valid (R13; parity — taip pat) + tik įrenginiui būdingi reikalavimai
// (k riba, 4 B lygiavimas).
bool job_valid(const CompositeJob& j, int parity) {
    if (j.k > kMaxBatchLayers) return false;  // prieš bendrą validatorių (jis eina per j.k sluoksnių)
    if (!composite_job_valid(j, parity)) return false;
    if (!aligned(j.out, 4)) return false;
    if (j.base && !aligned(j.base, 4)) return false;
    for (int l = 0; l < j.k; ++l)
        if (!aligned(j.layers[l].grain, 4)) return false;
    return true;
}

// --- Device pusė -------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void load4(const std::uint8_t* p, bool vec, u32 w[4]) {
    if (vec) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
        w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
    } else {
        const u32* q = reinterpret_cast<const u32*>(p);
        for (int i = 0; i < 4; ++i) w[i] = __ldg(q + i);
    }
}
__device__ __forceinline__ void store4(std::uint8_t* p, bool vec, const u32 w[4]) {
    if (vec) {
        *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        u32* q = reinterpret_cast<u32*>(p);
        for (int i = 0; i < 4; ++i) q[i] = w[i];
    }
}
__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// media::ref::rgb_to_ycbcr10 (double) be FMA kontrakcijos; operacijų tvarka identiška šaltiniui.
__device__ __forceinline__ int round_half_up(double v) { return static_cast<int>(floor(__dadd_rn(v, 0.5))); }
__device__ __forceinline__ void rgb_to_ycbcr10(int r, int g, int b, int& y, int& cb, int& cr) {
    const double R = __ddiv_rn(static_cast<double>(r), 255.0);
    const double G = __ddiv_rn(static_cast<double>(g), 255.0);
    const double B = __ddiv_rn(static_cast<double>(b), 255.0);
    const double Yp = __dadd_rn(__dadd_rn(__dmul_rn(0.2126, R), __dmul_rn(0.7152, G)), __dmul_rn(0.0722, B));
    y = clampi(round_half_up(__dadd_rn(64.0, __dmul_rn(876.0, Yp))), 64, 940);
    cb = clampi(round_half_up(__dadd_rn(512.0, __ddiv_rn(__dmul_rn(896.0, __dadd_rn(B, -Yp)), 1.8556))), 64, 960);
    cr = clampi(round_half_up(__dadd_rn(512.0, __ddiv_rn(__dmul_rn(896.0, __dadd_rn(R, -Yp)), 1.5748))), 64, 960);
}

// v210a taškas: f virš (b, ab) su efektyvia alfa ae. Grąžina fill; ao — išėjimo alfa. Tarpiniai ≤ 1023·876·876 < 2^32.
__device__ __forceinline__ int mix_v210a(int f, int ae, int b, int ab, int def, int& ao) {
    const u32 n = static_cast<u32>(ae * A + ab * (A - ae));
    ao = static_cast<int>((n + A / 2) / A);
    if (n == 0) return def;
    const u32 num = static_cast<u32>(f) * ae * A + static_cast<u32>(b) * ab * (A - ae) + n / 2;
    return clampi(static_cast<int>(num / n), 4, 1019);
}
__device__ __forceinline__ int mix_v210(int f, int ae, int b) {
    return clampi(static_cast<int>((static_cast<u32>(b) * (A - ae) + static_cast<u32>(f) * ae + A / 2) / A), 4, 1019);
}

// ---- rgba -> v210a ----
struct RgbaArgs {
    const std::uint8_t* rgba;
    std::uint8_t* grain;
    std::size_t stride, fill_line, key_line, fill_total;
    int w, lines, gpr, kwords, parity;  // gpr — grupių eilutėje su padding (fill_line / 16)
    bool premult, interlaced, rgba_vec, grain_vec;
};

__global__ void k_rgba_to_v210a(const __grid_constant__ RgbaArgs a) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::size_t>(a.lines) * a.gpr) return;
    const int row = static_cast<int>(t / a.gpr), g = static_cast<int>(t % a.gpr);
    std::uint8_t* fill = a.grain + static_cast<std::size_t>(row) * a.fill_line + static_cast<std::size_t>(g) * 16;
    const int px0 = 6 * g;
    Group grp{};  // nuliai: padding ir nenaudoti mėginiai
    u32 keyw[2] = {0, 0};
    if (px0 < a.w) {
        const int frow = a.interlaced ? 2 * row + a.parity : row;
        const std::uint8_t* src = a.rgba + static_cast<std::size_t>(frow) * a.stride;
        int pcb[6], pcr[6], al[6];
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            al[k] = 0;
            pcb[k] = pcr[k] = 512;
            if (px0 + k < a.w) {
                const std::uint8_t* p = src + 4 * static_cast<std::size_t>(px0 + k);
                int c[3], aa;
                if (a.rgba_vec) {
                    const u32 v = __ldg(reinterpret_cast<const u32*>(p));
                    c[0] = v & 255u; c[1] = (v >> 8) & 255u; c[2] = (v >> 16) & 255u; aa = v >> 24;
                } else {
                    c[0] = p[0]; c[1] = p[1]; c[2] = p[2]; aa = p[3];
                }
                if (a.premult) {
#pragma unroll
                    for (int q = 0; q < 3; ++q) c[q] = aa == 0 ? 0 : min(255, (c[q] * 255 + aa / 2) / aa);
                }
                int y;
                rgb_to_ycbcr10(c[0], c[1], c[2], y, pcb[k], pcr[k]);
                grp.y[k] = static_cast<std::uint16_t>(y);
                al[k] = aa;
                keyw[k / 3] |= static_cast<u32>(64 + (aa * A + 127) / 255) << (10 * (k % 3));
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            if (px0 + 2 * j >= a.w) break;
            const int a0 = al[2 * j], a1 = al[2 * j + 1], s = a0 + a1;
            if (s == 0) {
                grp.cb[j] = grp.cr[j] = 512;
            } else {
                grp.cb[j] = static_cast<std::uint16_t>((a0 * pcb[2 * j] + a1 * pcb[2 * j + 1] + s / 2) / s);
                grp.cr[j] = static_cast<std::uint16_t>((a0 * pcr[2 * j] + a1 * pcr[2 * j + 1] + s / 2) / s);
            }
        }
    }
    u32 words[4];
    pack_group(grp, words);
    store4(fill, a.grain_vec, words);
    if (px0 < a.w) {
        u32* key = reinterpret_cast<u32*>(a.grain + a.fill_total + static_cast<std::size_t>(row) * a.key_line);
#pragma unroll
        for (int s = 0; s < 2; ++s)
            if (2 * g + s < a.kwords) key[2 * g + s] = keyw[s];
    }
}

// ---- composite ----
template <class B>
__global__ void k_composite(const __grid_constant__ B b) {
    const DJob& jb = b.job[blockIdx.y];
    const int w = jb.w;
    const int gpr = ((w + 47) / 48) * 8;
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::size_t>(jb.lines) * gpr) return;
    const int row = static_cast<int>(t / gpr), g = static_cast<int>(t % gpr);
    const std::size_t fill_line = static_cast<std::size_t>(gpr) * 16;
    const std::size_t fill_total = fill_line * jb.lines;
    const bool vec = (jb.flags & kJobVec) != 0;
    const bool out_alpha = (jb.flags & kJobAlpha) != 0;
    const int px0 = 6 * g;
    std::uint8_t* const ofill = jb.out + static_cast<std::size_t>(row) * fill_line + static_cast<std::size_t>(g) * 16;
    if (px0 >= w) {  // fill eilutės padding
        const u32 z[4] = {0, 0, 0, 0};
        store4(ofill, vec, z);
        return;
    }
    const int kwords = (w + 2) / 3;
    const std::size_t key_line = static_cast<std::size_t>(kwords) * 4;

    // 1. Pradinė būsena: fill Y/Cb/Cr ir sukaupta alfa ab ∈ 0…A.
    int y[6], cb[3], cr[3], ab[6];
    if (jb.flags & kJobBase) {
        u32 bw[4];
        load4(jb.base + static_cast<std::size_t>(row) * fill_line + static_cast<std::size_t>(g) * 16, vec, bw);
        Group gr;
        unpack_group(bw, gr);
#pragma unroll
        for (int k = 0; k < 6; ++k) y[k] = gr.y[k];
#pragma unroll
        for (int j = 0; j < 3; ++j) { cb[j] = gr.cb[j]; cr[j] = gr.cr[j]; }
        if (out_alpha && (jb.flags & kJobBaseAlpha)) {
            const u32* bk = reinterpret_cast<const u32*>(jb.base + fill_total + static_cast<std::size_t>(row) * key_line);
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                const int wi = 2 * g + k / 3;
                const u32 wd = wi < kwords ? __ldg(bk + wi) : 0u;
                ab[k] = clampi(static_cast<int>((wd >> (10 * (k % 3))) & 0x3FFu) - 64, 0, A);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 6; ++k) ab[k] = A;  // v210 bazė nepermatoma; v210 išėjimui ab nenaudojamas
        }
    } else {  // --base none
#pragma unroll
        for (int k = 0; k < 6; ++k) { y[k] = 64; ab[k] = 0; }
#pragma unroll
        for (int j = 0; j < 3; ++j) cb[j] = cr[j] = 512;
    }

    // 2. Sluoksniai iš apačios į viršų. Poros (2j, 2j+1) sluoksnio mėginiai: chroma pirma, su ab PRIEŠ šio sluoksnio
    // atnaujinimą (R8), tada abu Y.
    for (int l = 0; l < jb.lcnt; ++l) {
        const DLayer& L = b.layer[jb.lbeg + l];
        const int ly = row - L.yoff;
        if (ly < 0 || ly >= L.lines) continue;
        const std::size_t lfill = static_cast<std::size_t>((L.w + 47) / 48) * 128;
        const std::size_t lkey = static_cast<std::size_t>((L.w + 2) / 3) * 4;
        const bool lalpha = (L.flags & kLayerAlpha) != 0;
        const std::uint32_t* lk = reinterpret_cast<const u32*>(L.grain + lfill * L.lines + static_cast<std::size_t>(ly) * lkey);
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            const int ox = px0 + 2 * p;
            const int lx = ox - L.x;
            if (ox >= w || lx < 0 || lx >= L.w) continue;
            const int c = lx >> 1;  // sluoksnio chromos pozicija: grupė c / 3, narys c % 3
            u32 lw[4];
            load4(L.grain + static_cast<std::size_t>(ly) * lfill + static_cast<std::size_t>(c / 3) * 16, (L.flags & kLayerVec) != 0, lw);
            Group lg;
            unpack_group(lw, lg);
            const int j = c % 3;
            int a0 = A, a1 = A;
            if (lalpha) {
                const u32 w0 = __ldg(lk + lx / 3);
                const u32 w1 = (lx + 1) / 3 == lx / 3 ? w0 : __ldg(lk + (lx + 1) / 3);
                a0 = clampi(static_cast<int>((w0 >> (10 * (lx % 3))) & 0x3FFu) - 64, 0, A);
                a1 = clampi(static_cast<int>((w1 >> (10 * ((lx + 1) % 3))) & 0x3FFu) - 64, 0, A);
            }
            const int ae0 = (a0 * L.o8 + 127) / 255, ae1 = (a1 * L.o8 + 127) / 255;
            const int ac = (ae0 + ae1 + 1) >> 1;
            int dummy;
            if (out_alpha) {
                const int abc = (ab[2 * p] + ab[2 * p + 1] + 1) >> 1;
                cb[p] = mix_v210a(lg.cb[j], ac, cb[p], abc, 512, dummy);
                cr[p] = mix_v210a(lg.cr[j], ac, cr[p], abc, 512, dummy);
                y[2 * p] = mix_v210a(lg.y[2 * j], ae0, y[2 * p], ab[2 * p], 64, ab[2 * p]);
                y[2 * p + 1] = mix_v210a(lg.y[2 * j + 1], ae1, y[2 * p + 1], ab[2 * p + 1], 64, ab[2 * p + 1]);
            } else {
                cb[p] = mix_v210(lg.cb[j], ac, cb[p]);
                cr[p] = mix_v210(lg.cr[j], ac, cr[p]);
                y[2 * p] = mix_v210(lg.y[2 * j], ae0, y[2 * p]);
                y[2 * p + 1] = mix_v210(lg.y[2 * j + 1], ae1, y[2 * p + 1]);
            }
        }
    }

    // 3. Supakuoti; mėginiai už eilutės ribos = 0.
    Group og{};
#pragma unroll
    for (int k = 0; k < 6; ++k)
        if (px0 + k < w) og.y[k] = static_cast<std::uint16_t>(y[k]);
#pragma unroll
    for (int j = 0; j < 3; ++j)
        if (px0 + 2 * j < w) { og.cb[j] = static_cast<std::uint16_t>(cb[j]); og.cr[j] = static_cast<std::uint16_t>(cr[j]); }
    u32 ow[4];
    pack_group(og, ow);
    store4(ofill, vec, ow);
    if (out_alpha) {
        u32* ok = reinterpret_cast<u32*>(jb.out + fill_total + static_cast<std::size_t>(row) * key_line);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            if (2 * g + s >= kwords) continue;
            u32 wd = 0;  // bitai 30–31 ir nenaudoti mėginiai = 0
#pragma unroll
            for (int i = 0; i < 3; ++i)
                if (px0 + 3 * s + i < w) wd |= static_cast<u32>(ab[3 * s + i] + 64) << (10 * i);
            ok[2 * g + s] = wd;
        }
    }
}

unsigned blocks_for(std::size_t threads) { return static_cast<unsigned>((threads + kThreads - 1) / kThreads); }
std::size_t job_threads(const DJob& j) { return static_cast<std::size_t>(j.lines) * (((j.w + 47) / 48) * 8); }

DJob make_job(const CompositeJob& j, int lbeg) {
    DJob d{};
    d.base = j.base;
    d.out = j.out;
    d.w = j.out_r.w;
    d.lines = grain_lines(j.out_r);
    d.flags = (j.out_r.alpha ? kJobAlpha : 0) | (j.base ? kJobBase : 0) | (j.base && j.base_r.alpha ? kJobBaseAlpha : 0) |
              (aligned(j.out, 16) && (!j.base || aligned(j.base, 16)) ? kJobVec : 0);
    d.lbeg = lbeg;
    d.lcnt = j.k;
    return d;
}
DLayer make_layer(const Layer& L, const Raster& out_r) {
    DLayer d{};
    d.grain = L.grain;
    d.w = L.r.w;
    d.lines = grain_lines(L.r);
    d.x = L.x;
    d.yoff = out_r.interlaced ? L.y / 2 : L.y;  // R1: interlaced -> sluoksnio lauko eilutė j patenka į išėjimo lauko eilutę j + y/2
    d.o8 = opacity_to_o8(L.opacity);            // R10: visada host'e, vienas apibrėžimas visiems variantams
    d.flags = (L.r.alpha ? kLayerAlpha : 0) | (aligned(L.grain, 16) ? kLayerVec : 0);
    return d;
}

// Vienas paleidimas jobs[off, off+cnt) (sluoksnių suma telpa į B).
template <class B>
cudaError_t launch(const CompositeJob* jobs, int cnt, cudaStream_t s) {
    B b{};
    b.n = cnt;
    int nl = 0;
    std::size_t maxt = 0;
    for (int i = 0; i < cnt; ++i) {
        const CompositeJob& j = jobs[i];
        b.job[i] = make_job(j, nl);
        for (int l = 0; l < j.k; ++l) b.layer[nl++] = make_layer(j.layers[l], j.out_r);
        maxt = std::max(maxt, job_threads(b.job[i]));
    }
    k_composite<B><<<dim3(blocks_for(maxt), static_cast<unsigned>(cnt)), kThreads, 0, s>>>(b);
    return cudaPeekAtLastError();  // neištrina svetimos (kitos gijos / ankstesnės) klaidos būsenos
}

}  // namespace

cudaError_t rgba_to_v210a(const std::uint8_t* rgba_dev, int w, int h, std::size_t stride, AlphaIn mode, const Raster& r,
                          int parity, std::uint8_t* grain_dev, cudaStream_t s) {
    if (!rgba_dev || !grain_dev || !raster_ok(r) || !r.alpha) return cudaErrorInvalidValue;
    if (r.w != w || r.h != h) return cudaErrorInvalidValue;
    if (stride < static_cast<std::size_t>(w) * 4) return cudaErrorInvalidValue;
    if (r.interlaced && parity != 0 && parity != 1) return cudaErrorInvalidValue;
    if (!aligned(grain_dev, 4)) return cudaErrorInvalidValue;
    RgbaArgs a{};
    a.rgba = rgba_dev;
    a.grain = grain_dev;
    a.stride = stride;
    a.fill_line = fill_line_bytes(w);
    a.key_line = key_line_bytes(w);
    a.fill_total = fill_bytes(r);
    a.w = w;
    a.lines = grain_lines(r);
    a.gpr = static_cast<int>(a.fill_line / 16);
    a.kwords = (w + 2) / 3;
    a.parity = parity;
    a.premult = mode == AlphaIn::Premultiplied;
    a.interlaced = r.interlaced;
    a.rgba_vec = aligned(rgba_dev, 4) && stride % 4 == 0;
    a.grain_vec = aligned(grain_dev, 16);
    k_rgba_to_v210a<<<blocks_for(static_cast<std::size_t>(a.lines) * a.gpr), kThreads, 0, s>>>(a);
    return cudaPeekAtLastError();
}

cudaError_t composite_batch(const CompositeJob* jobs, int n, int parity, cudaStream_t s) {
    if (n < 0 || n > kMaxBatch || (n > 0 && !jobs)) return cudaErrorInvalidValue;
    for (int i = 0; i < n; ++i)
        if (!job_valid(jobs[i], parity)) return cudaErrorInvalidValue;  // R11: bent vienas netinkamas -> nieko nepaleidžiama
    // Dalijimas: viskas patikrinta -> paleidimai nebeklysta dėl geometrijos. Mažas batch'as -> maža parametrų struktūra.
    for (int off = 0; off < n;) {
        int cnt = 0, layers = 0;
        while (off + cnt < n && cnt < kMaxBatch && layers + jobs[off + cnt].k <= kMaxBatchLayers) layers += jobs[off + cnt++].k;
        const cudaError_t e = (cnt <= kSmallJobs && layers <= kSmallLayers) ? launch<Small>(jobs + off, cnt, s)
                                                                           : launch<Large>(jobs + off, cnt, s);
        if (e != cudaSuccess) return e;
        off += cnt;
    }
    return cudaSuccess;
}

}  // namespace vramflow::gfx::cuda
