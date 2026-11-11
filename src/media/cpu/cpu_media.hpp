// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// CPU media branduoliai variantui C (architektūra docs/architektura/2026-11-11-gpu-vs-cpu-ir-upstream.md §1.3).
// Rašyti „kaip kompetentingas CPU funkcijos autorius": tik būtinos kopijos (memcpy už logotipo ribų), apdorojimas tik
// reikalingoje srityje, ciklai be šakų vidinėje dalyje (kompiliatorius vektorizuoja su -O3 -march=x86-64-v3).
// Bitų tikslumu lygūs GPU kernel'iams ir nepriklausomam media_ref etalonui (tests/unit/test_media_cpu.cpp).
#include <cstdint>

#include "media/gen_content.hpp"
#include "media/media_kernels.hpp"  // LogoDesc
#include "media/v210.hpp"

namespace mxlv::media::cpu {

// Logotipo plokštumos host atmintyje (ta pati struktūra kaip GPU, rodyklės — host).
using LogoPlanes = LogoDesc;

// in → out (gali būti in == out). parity = index & 1.
void logo_field(const std::uint32_t* in, std::uint32_t* out, const LogoPlanes& logo, int parity);

// Du v210 laukai → NV12 kadras (lauką suvokiantis 4:2:2 → 4:2:0).
void weave_nv12(const std::uint32_t* top, const std::uint32_t* bottom, std::uint8_t* y, std::uint8_t* uv, int pitch_y,
                int pitch_uv);

// NV12 kadras → v210 laukas `parity` (lauką suvokiantis 4:2:0 → 4:2:2).
void split_nv12(const std::uint8_t* y, const std::uint8_t* uv, int pitch_y, int pitch_uv, int parity, std::uint32_t* out);

// Tas pats iš planar YUV420P (programinio H.264 dekoderio išvestis): u/v — atskiros plokštumos, pitch_c jų žingsnis.
void split_yuv420p(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v, int pitch_y, int pitch_c, int parity,
                   std::uint32_t* out);

// Generatoriaus laukas (turinio apibrėžimas — gen_content.hpp).
void gen_field(const GenParams& g, std::uint64_t index, std::uint32_t* out);

}  // namespace mxlv::media::cpu
