// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// GPU operacijos grain'ui (architektūra §4.3). Visos asinchroniškos nurodytame stream'e.
#include <cuda_runtime_api.h>

#include <cstdint>

#include "common/header.hpp"

namespace mxlv::gpu {

struct VerifyResult {
    std::uint64_t mismatch_words;
    std::uint64_t actual_fold64;
    std::uint32_t first_bad_word;  // 0xFFFFFFFF = nėra
    std::uint32_t pad;
    std::uint8_t header[64];
};
static_assert(sizeof(VerifyResult) == 88);

// Užpildo žodžius [begin, end) ∩ [16, n) ir prideda jų fold64 prie *d_fold (d_fold nenunulinamas).
cudaError_t fill(std::uint32_t* slot, std::uint64_t seed, std::uint64_t index, std::uint32_t n, std::uint32_t begin,
                 std::uint32_t end, std::uint64_t* d_fold, cudaStream_t s);

// Įrašo h (su payload_fold64 = *d_fold ir CRC) į slot[0..16).
cudaError_t finalize_header(std::uint32_t* slot, const GrainHeader& h, const std::uint64_t* d_fold, cudaStream_t s);

// slot[pos] ^= mask (gedimų injekcijai).
cudaError_t corrupt(std::uint32_t* slot, std::uint32_t pos, std::uint32_t mask, cudaStream_t s);

// Palygina [16, n) su word(seed, index, i), skaičiuoja faktinę fold64, nukopijuoja antraštę.
// d_res nunulinamas viduje. Rezultatą reikia nukopijuoti D2H ir sinchronizuoti stream'ą.
cudaError_t verify(const std::uint32_t* slot, std::uint64_t seed, std::uint64_t index, std::uint32_t n,
                   VerifyResult* d_res, cudaStream_t s);

}  // namespace mxlv::gpu
