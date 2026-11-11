// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// GPU media kernel'iai (architektūra GPU funkcijų §6). Visi — BATCH: vienas paleidimas apdoroja iki kMaxBatch srautų,
// darbų masyvas perduodamas kernel'io argumentu by-value (be H2D kopijos). Grid: x — gijos lauko/kadro viduje
// (viena gija = viena v210 grupė = 6 px vienoje eilutėje; eilutės grupės iš eilės → sujungti 16 B prieigos), y — srautas.
#include "media/media_kernels.hpp"

#include <cstdint>

#include "media/v210.hpp"

namespace mxlv::media {
namespace {

constexpr int kThreads = 256;
constexpr int kFieldGroups = kFieldHeight * kGroupsPerLine;       // 172 800 grupių lauke
constexpr int kChromaRows = kFieldHeight / 2;                     // 270 lauko chromos eilių NV12 kadre
constexpr int kQuadGroups = kChromaRows * kGroupsPerLine;         // 86 400: grupė × (2 lauko eilutės × 2 laukai)

template <class J>
struct Batch {
    J job[kMaxBatch];
    int n;
};
// Kernel'io argumentų riba didesnė, bet laikomės ≤ 4 KiB (senas limitas, mažiau konstantų atminties per paleidimą).
static_assert(sizeof(Batch<GenJob>) <= 4096);
static_assert(sizeof(Batch<LogoJob>) + sizeof(LogoDesc) <= 4096);
static_assert(sizeof(Batch<WeaveJob>) <= 4096);
static_assert(sizeof(Batch<SplitJob>) <= 4096);

unsigned blocks_for(int threads) { return static_cast<unsigned>((threads + kThreads - 1) / kThreads); }

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

// 4 žodžių (vienos grupės) skaitymas/rašymas: kai visi batch'o adresai 16 B lygiuoti — vienas 128 bitų veiksmas.
template <bool V>
__device__ __forceinline__ void load4(const std::uint32_t* p, std::uint32_t w[4]) {
    if constexpr (V) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
    } else {
        for (int i = 0; i < 4; ++i) w[i] = p[i];
    }
}
template <bool V>
__device__ __forceinline__ void store4(std::uint32_t* p, const std::uint32_t w[4]) {
    if constexpr (V) {
        *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        for (int i = 0; i < 4; ++i) p[i] = w[i];
    }
}

__device__ __forceinline__ std::uint32_t blend_u(std::uint32_t s, std::uint32_t l, std::uint32_t a) {
    return (s * (255u - a) + l * a + 127u) / 255u;  // max 1023·255 + 127 < 2^32
}
__device__ __forceinline__ std::uint8_t to8(std::uint32_t v10) {
    const std::uint32_t v = (v10 + 2u) >> 2;
    return static_cast<std::uint8_t>(v > 255u ? 255u : v);
}

// --- Generatorius ------------------------------------------------------------------------------------------------------
template <bool V>
__global__ void k_gen_field_batch(const __grid_constant__ Batch<GenJob> b, std::uint64_t run_seed, std::uint32_t complexity, std::uint64_t index) {
    const int s = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= b.n || t >= kFieldGroups) return;
    const int j = t / kGroupsPerLine, g = t % kGroupsPerLine;
    const GenParams gp{run_seed, b.job[s].flow, complexity};
    const int fy = 2 * j + static_cast<int>(index & 1u);
    const int x0 = 6 * g;
    Group grp;
    for (int q = 0; q < 6; ++q) grp.y[q] = gen_y(gp, index, x0 + q, fy);
    for (int c = 0; c < 3; ++c) gen_c(gp, index, x0 + 2 * c, fy, grp.cb[c], grp.cr[c]);
    std::uint32_t w[4];
    pack_group(grp, w);
    store4<V>(b.job[s].dst + static_cast<std::size_t>(j) * kWordsPerLine + 4 * g, w);
}

// --- Logotipas ---------------------------------------------------------------------------------------------------------
template <bool V>
__global__ void k_logo_batch(const __grid_constant__ Batch<LogoJob> b, const __grid_constant__ LogoDesc L, int parity) {
    const int s = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= b.n || t >= kFieldGroups) return;
    const int j = t / kGroupsPerLine, g = t % kGroupsPerLine;
    const std::size_t off = static_cast<std::size_t>(j) * kWordsPerLine + 4 * g;
    const std::uint32_t* src = b.job[s].src + off;
    std::uint32_t* dst = b.job[s].dst + off;
    const int x0 = 6 * g;
    const int ly = 2 * j + parity - L.y;  // logotipo eilutė (kadro eilutė fy = 2j + parity)
    const bool inside = ly >= 0 && ly < L.lh && x0 + 6 > L.x && x0 < L.x + L.lw;
    if (!inside && src == dst) return;  // vietoje (in-place): už ribų nieko nedarom
    std::uint32_t w[4];
    load4<V>(src, w);
    if (inside) {
        Group grp;
        unpack_group(w, grp);
        const std::size_t row = static_cast<std::size_t>(ly) * L.lw;
        const std::size_t crow = static_cast<std::size_t>(ly) * (L.lw / 2);
        for (int q = 0; q < 6; ++q) {
            const int lx = x0 + q - L.x;
            if (lx >= 0 && lx < L.lw)
                grp.y[q] = static_cast<std::uint16_t>(blend_u(grp.y[q], L.y10[row + lx], L.a8[row + lx]));
        }
        for (int c = 0; c < 3; ++c) {
            const int lx = x0 + 2 * c - L.x;  // L.x lyginis → visa pora viduje arba visa išorėje
            if (lx >= 0 && lx < L.lw) {
                const std::size_t ci = crow + (lx >> 1);
                const std::uint32_t a = L.ac8[ci];
                grp.cb[c] = static_cast<std::uint16_t>(blend_u(grp.cb[c], L.cb10[ci], a));
                grp.cr[c] = static_cast<std::uint16_t>(blend_u(grp.cr[c], L.cr10[ci], a));
            }
        }
        std::uint32_t o[4];
        pack_group(grp, o, w);  // bitai 30–31 iš šaltinio
        store4<V>(dst, o);
    } else {
        store4<V>(dst, w);  // bitas į bitą
    }
}

// --- Du laukai → NV12 --------------------------------------------------------------------------------------------------
template <bool V>
__global__ void k_weave_nv12_batch(const __grid_constant__ Batch<WeaveJob> b, int pitch_y, int pitch_uv) {
    const int s = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= b.n || t >= kQuadGroups) return;
    const int k = t / kGroupsPerLine, g = t % kGroupsPerLine;
    const WeaveJob& job = b.job[s];
    for (int p = 0; p < 2; ++p) {
        const std::uint32_t* f = p == 0 ? job.top : job.bottom;
        Group l[2];
        for (int i = 0; i < 2; ++i) {
            std::uint32_t w[4];
            load4<V>(f + static_cast<std::size_t>(2 * k + i) * kWordsPerLine + 4 * g, w);
            unpack_group(w, l[i]);
            // luma: lauko p eilutė 2k+i → kadro eilutė 2(2k+i)+p
            std::uint8_t* dy = job.dst_y + static_cast<std::size_t>(2 * (2 * k + i) + p) * pitch_y + 6 * g;
            for (int q = 0; q < 6; ++q) dy[q] = to8(l[i].y[q]);
        }
        // chroma lauko viduje: viršutinis (3,1) → UV eilė 2k, apatinis (1,3) → UV eilė 2k+1
        const std::uint32_t w0 = p == 0 ? 3u : 1u, w1 = p == 0 ? 1u : 3u;
        std::uint8_t* duv = job.dst_uv + static_cast<std::size_t>(2 * k + p) * pitch_uv + 6 * g;  // 2·(3g)
        for (int c = 0; c < 3; ++c) {
            duv[2 * c] = to8((w0 * l[0].cb[c] + w1 * l[1].cb[c] + 2u) >> 2);
            duv[2 * c + 1] = to8((w0 * l[0].cr[c] + w1 * l[1].cr[c] + 2u) >> 2);
        }
    }
}

// --- NV12 → laukas -----------------------------------------------------------------------------------------------------
template <bool V>
__global__ void k_split_nv12_batch(const __grid_constant__ Batch<SplitJob> b, int parity, int pitch_y, int pitch_uv) {
    const int s = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= b.n || t >= kFieldGroups) return;
    const int j = t / kGroupsPerLine, g = t % kGroupsPerLine;
    const SplitJob& job = b.job[s];
    Group grp;
    const std::uint8_t* sy = job.src_y + static_cast<std::size_t>(2 * j + parity) * pitch_y + 6 * g;
    for (int q = 0; q < 6; ++q) grp.y[q] = static_cast<std::uint16_t>(sy[q] << 2);
    // lauko chromos eilės c[k] = UV eilė 2k + parity; svoriai aštuntosiomis (§6.4)
    const int k = j >> 1;
    int k0, k1;
    std::uint32_t a, bw;
    if ((j & 1) == 0) {
        k0 = k > 0 ? k - 1 : 0;
        k1 = k;
        a = parity == 0 ? 1u : 3u;
    } else {
        k0 = k;
        k1 = k < kChromaRows - 1 ? k + 1 : kChromaRows - 1;
        a = parity == 0 ? 5u : 7u;
    }
    bw = 8u - a;
    const std::uint8_t* c0 = job.src_uv + static_cast<std::size_t>(2 * k0 + parity) * pitch_uv + 6 * g;
    const std::uint8_t* c1 = job.src_uv + static_cast<std::size_t>(2 * k1 + parity) * pitch_uv + 6 * g;
    for (int c = 0; c < 3; ++c) {
        grp.cb[c] = static_cast<std::uint16_t>((a * c0[2 * c] + bw * c1[2 * c] + 1u) >> 1);
        grp.cr[c] = static_cast<std::uint16_t>((a * c0[2 * c + 1] + bw * c1[2 * c + 1] + 1u) >> 1);
    }
    std::uint32_t w[4];
    pack_group(grp, w);
    store4<V>(job.dst + static_cast<std::size_t>(j) * kWordsPerLine + 4 * g, w);
}

// Bendras dalijimas į paleidimus po kMaxBatch. `valid(job)` tikrina rodykles, `launch(batch, vec)` paleidžia kernel'į.
template <class J, class Valid, class Aligned, class Launch>
cudaError_t run_batches(const J* jobs, int n, Valid valid, Aligned aligned, Launch launch) {
    if (n < 0 || (n > 0 && jobs == nullptr)) return cudaErrorInvalidValue;
    for (int i = 0; i < n; ++i)
        if (!valid(jobs[i])) return cudaErrorInvalidValue;
    for (int off = 0; off < n; off += kMaxBatch) {
        Batch<J> b{};
        b.n = n - off < kMaxBatch ? n - off : kMaxBatch;
        bool vec = true;
        for (int i = 0; i < b.n; ++i) {
            b.job[i] = jobs[off + i];
            vec = vec && aligned(b.job[i]);
        }
        launch(b, vec);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

}  // namespace

cudaError_t gen_fields(const GenJob* jobs, int n, std::uint64_t run_seed, std::uint32_t complexity, std::uint64_t index,
                       cudaStream_t s) {
    return run_batches(
        jobs, n, [](const GenJob& j) { return j.dst != nullptr; }, [](const GenJob& j) { return aligned16(j.dst); },
        [&](const Batch<GenJob>& b, bool vec) {
            const dim3 grid(blocks_for(kFieldGroups), static_cast<unsigned>(b.n));
            if (vec)
                k_gen_field_batch<true><<<grid, kThreads, 0, s>>>(b, run_seed, complexity, index);
            else
                k_gen_field_batch<false><<<grid, kThreads, 0, s>>>(b, run_seed, complexity, index);
        });
}

cudaError_t logo_fields(const LogoJob* jobs, int n, const LogoDesc& logo, int parity, cudaStream_t s) {
    if (parity != 0 && parity != 1) return cudaErrorInvalidValue;
    if (logo.lw <= 0 || logo.lh <= 0 || (logo.lw & 1) || (logo.x & 1) || !logo.y10 || !logo.a8 || !logo.cb10 ||
        !logo.cr10 || !logo.ac8)
        return cudaErrorInvalidValue;
    return run_batches(
        jobs, n, [](const LogoJob& j) { return j.src != nullptr && j.dst != nullptr; },
        [](const LogoJob& j) { return aligned16(j.src) && aligned16(j.dst); },
        [&](const Batch<LogoJob>& b, bool vec) {
            const dim3 grid(blocks_for(kFieldGroups), static_cast<unsigned>(b.n));
            if (vec)
                k_logo_batch<true><<<grid, kThreads, 0, s>>>(b, logo, parity);
            else
                k_logo_batch<false><<<grid, kThreads, 0, s>>>(b, logo, parity);
        });
}

cudaError_t weave_nv12(const WeaveJob* jobs, int n, int pitch_y, int pitch_uv, cudaStream_t s) {
    if (pitch_y < kWidth || pitch_uv < kWidth) return cudaErrorInvalidValue;
    return run_batches(
        jobs, n, [](const WeaveJob& j) { return j.top && j.bottom && j.dst_y && j.dst_uv; },
        [](const WeaveJob& j) { return aligned16(j.top) && aligned16(j.bottom); },
        [&](const Batch<WeaveJob>& b, bool vec) {
            const dim3 grid(blocks_for(kQuadGroups), static_cast<unsigned>(b.n));
            if (vec)
                k_weave_nv12_batch<true><<<grid, kThreads, 0, s>>>(b, pitch_y, pitch_uv);
            else
                k_weave_nv12_batch<false><<<grid, kThreads, 0, s>>>(b, pitch_y, pitch_uv);
        });
}

cudaError_t split_nv12(const SplitJob* jobs, int n, int parity, int pitch_y, int pitch_uv, cudaStream_t s) {
    if (parity != 0 && parity != 1) return cudaErrorInvalidValue;
    if (pitch_y < kWidth || pitch_uv < kWidth) return cudaErrorInvalidValue;
    return run_batches(
        jobs, n, [](const SplitJob& j) { return j.src_y && j.src_uv && j.dst; },
        [](const SplitJob& j) { return aligned16(j.dst); },
        [&](const Batch<SplitJob>& b, bool vec) {
            const dim3 grid(blocks_for(kFieldGroups), static_cast<unsigned>(b.n));
            if (vec)
                k_split_nv12_batch<true><<<grid, kThreads, 0, s>>>(b, parity, pitch_y, pitch_uv);
            else
                k_split_nv12_batch<false><<<grid, kThreads, 0, s>>>(b, parity, pitch_y, pitch_uv);
        });
}

}  // namespace mxlv::media
