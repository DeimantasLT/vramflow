// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// GPU media kernel'iai (architektūra GPU funkcijų §6). Visi — BATCH: vienas paleidimas apdoroja iki kMaxBatch srautų
// (apeina ~1,5 ms WDDM paleidimo/sinchronizacijos kainą vienam srautui). Batch'as perduodamas by-value (be H2D kopijos).
// N > kMaxBatch → funkcija pati padalina į kelis paleidimus. Visos funkcijos asinchroninės stream'e.
#include <cuda_runtime_api.h>

#include <cstdint>

#include "media/gen_content.hpp"

namespace mxlv::media {

constexpr int kMaxBatch = 64;

// --- Generatorius: v210 laukas `index` kiekvienam srautui (turinys — gen_content.hpp) -------------------------------
struct GenJob {
    std::uint32_t* dst;  // kFieldWords žodžių
    std::uint32_t flow;
};
cudaError_t gen_fields(const GenJob* jobs, int n, std::uint64_t run_seed, std::uint32_t complexity, std::uint64_t index,
                       cudaStream_t s);

// --- Logotipas ant v210 lauko ----------------------------------------------------------------------------------------
// Logotipo plokštumos device atmintyje (kadro raiška lw × lh; chroma — lw/2 pločio). X lyginis, lw lyginis.
struct LogoDesc {
    const std::uint16_t* y10;   // lw*lh
    const std::uint8_t* a8;     // lw*lh
    const std::uint16_t* cb10;  // (lw/2)*lh
    const std::uint16_t* cr10;  // (lw/2)*lh
    const std::uint8_t* ac8;    // (lw/2)*lh
    int lw, lh, x, y;           // kadro koordinatės
};
struct LogoJob {
    const std::uint32_t* src;
    std::uint32_t* dst;
};
// parity = index & 1 (0 — viršutinis laukas: kadro eilutė fy = 2j + parity)
cudaError_t logo_fields(const LogoJob* jobs, int n, const LogoDesc& logo, int parity, cudaStream_t s);

// --- Du v210 laukai → NV12 kadras (lauką suvokiantis 4:2:2 → 4:2:0) ---------------------------------------------------
struct WeaveJob {
    const std::uint32_t* top;     // laukas 2F
    const std::uint32_t* bottom;  // laukas 2F+1
    std::uint8_t* dst_y;          // 1080 eilučių, pitch_y
    std::uint8_t* dst_uv;         // 540 eilučių, pitch_uv (Cb,Cr persipynę)
};
cudaError_t weave_nv12(const WeaveJob* jobs, int n, int pitch_y, int pitch_uv, cudaStream_t s);

// --- NV12 kadras → vienas v210 laukas (lauką suvokiantis 4:2:0 → 4:2:2) ----------------------------------------------
struct SplitJob {
    const std::uint8_t* src_y;
    const std::uint8_t* src_uv;
    std::uint32_t* dst;  // laukas
};
cudaError_t split_nv12(const SplitJob* jobs, int n, int parity, int pitch_y, int pitch_uv, cudaStream_t s);

// --- Formatui neutralūs variantai (planas §P2; semantika — media_ref.hpp). d — kadro matmenys; field: −1 = kadras
// (progresyvus flow), 0 / 1 = lauko paritetas. Seni (be d) = kDim1080 ir field = parity. Netinkami matmenys, pitch < d.w
// arba FieldSited su field −1 → cudaErrorInvalidValue.
cudaError_t gen_fields(const GenJob* jobs, int n, std::uint64_t run_seed, std::uint32_t complexity, std::uint64_t index, Dim d,
                       bool interlaced, cudaStream_t s);
// Tas pats su grain'o dažniu rate_num / rate_den (sync blyksnis kartą per sekundę; ankstesnis — 50/1).
cudaError_t gen_fields(const GenJob* jobs, int n, std::uint64_t run_seed, std::uint32_t complexity, std::uint64_t index, Dim d,
                       bool interlaced, std::uint32_t rate_num, std::uint32_t rate_den, cudaStream_t s);
cudaError_t logo_fields(const LogoJob* jobs, int n, const LogoDesc& logo, Dim d, int field, cudaStream_t s);
cudaError_t weave_nv12(const WeaveJob* jobs, int n, Dim d, bool interlaced, int pitch_y, int pitch_uv, cudaStream_t s);
cudaError_t split_nv12(const SplitJob* jobs, int n, Dim d, int field, ChromaMode m, int pitch_y, int pitch_uv, cudaStream_t s);
cudaError_t split_nv12_progressive(const SplitJob* jobs, int n, Dim d, int pitch_y, int pitch_uv, cudaStream_t s);

}  // namespace mxlv::media
