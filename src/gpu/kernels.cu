// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "gpu/kernels.hpp"

#include "common/pattern.hpp"

namespace mxlv::gpu {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ std::uint64_t warp_sum(std::uint64_t v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;
}

// Kiekviena gija apdoroja kelis žodžius (grid-stride), po to redukcija warp'e ir vienas atomicAdd warp'ui.
__global__ void k_fill(std::uint32_t* slot, std::uint64_t seed, std::uint64_t index, std::uint32_t begin,
                       std::uint32_t end, unsigned long long* d_fold) {
    std::uint64_t acc = 0;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    for (std::uint32_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += stride) {
        const std::uint32_t w = word(seed, index, i);
        slot[i] = w;
        acc += fold_term(w, i);
    }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0 && acc) atomicAdd(d_fold, static_cast<unsigned long long>(acc));
}

__global__ void k_finalize_header(std::uint32_t* slot, GrainHeader h, const std::uint64_t* d_fold) {
    h.payload_fold64 = *d_fold;
    seal_header(h);
    const auto* src = reinterpret_cast<const std::uint32_t*>(&h);
    for (int i = 0; i < 16; ++i) slot[i] = src[i];
}

__global__ void k_corrupt(std::uint32_t* slot, std::uint32_t pos, std::uint32_t mask) { slot[pos] ^= mask; }

__global__ void k_verify(const std::uint32_t* slot, std::uint64_t seed, std::uint64_t index, std::uint32_t n,
                         VerifyResult* res) {
    std::uint64_t fold = 0;
    unsigned long long bad = 0;
    std::uint32_t first = 0xFFFFFFFFu;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    for (std::uint32_t i = kHeaderWords + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint32_t got = slot[i];
        fold += fold_term(got, i);
        if (got != word(seed, index, i)) {
            ++bad;
            if (i < first) first = i;
        }
    }
    fold = warp_sum(fold);
    bad = warp_sum(bad);
    for (int o = 16; o > 0; o >>= 1) first = min(first, __shfl_down_sync(0xFFFFFFFFu, first, o));
    if ((threadIdx.x & 31) == 0) {
        if (fold) atomicAdd(reinterpret_cast<unsigned long long*>(&res->actual_fold64), static_cast<unsigned long long>(fold));
        if (bad) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&res->mismatch_words), bad);
            atomicMin(&res->first_bad_word, first);
        }
    }
    if (blockIdx.x == 0 && threadIdx.x < 16) reinterpret_cast<std::uint32_t*>(res->header)[threadIdx.x] = slot[threadIdx.x];
}

__global__ void k_init_result(VerifyResult* res) {
    res->mismatch_words = 0;
    res->actual_fold64 = 0;
    res->first_bad_word = 0xFFFFFFFFu;
    res->pad = 0;
}

unsigned blocks_for(std::uint32_t words) {
    // Iki 4 žodžių vienai gijai; ne daugiau nei 1024 blokai (RTX 4070 Laptop: 36 SM).
    const unsigned need = (words + kThreads * 4 - 1) / (kThreads * 4);
    return need == 0 ? 1 : (need > 1024 ? 1024 : need);
}

}  // namespace

cudaError_t fill(std::uint32_t* slot, std::uint64_t seed, std::uint64_t index, std::uint32_t n, std::uint32_t begin,
                 std::uint32_t end, std::uint64_t* d_fold, cudaStream_t s) {
    if (begin < kHeaderWords) begin = kHeaderWords;
    if (end > n) end = n;
    if (begin >= end) return cudaSuccess;
    k_fill<<<blocks_for(end - begin), kThreads, 0, s>>>(slot, seed, index, begin, end,
                                                       reinterpret_cast<unsigned long long*>(d_fold));
    return cudaGetLastError();
}

cudaError_t finalize_header(std::uint32_t* slot, const GrainHeader& h, const std::uint64_t* d_fold, cudaStream_t s) {
    k_finalize_header<<<1, 1, 0, s>>>(slot, h, d_fold);
    return cudaGetLastError();
}

cudaError_t corrupt(std::uint32_t* slot, std::uint32_t pos, std::uint32_t mask, cudaStream_t s) {
    k_corrupt<<<1, 1, 0, s>>>(slot, pos, mask);
    return cudaGetLastError();
}

cudaError_t verify(const std::uint32_t* slot, std::uint64_t seed, std::uint64_t index, std::uint32_t n,
                   VerifyResult* d_res, cudaStream_t s) {
    k_init_result<<<1, 1, 0, s>>>(d_res);
    k_verify<<<blocks_for(n > kHeaderWords ? n - kHeaderWords : 1), kThreads, 0, s>>>(slot, seed, index, n, d_res);
    return cudaGetLastError();
}

}  // namespace mxlv::gpu
