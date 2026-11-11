// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variantas C: optimizuotas CPU RGBA -> v210a ir composite (be CUDA). Sutartis ta pati kaip gfx::ref (gfx_ref.hpp):
// funkcijos grąžina false ir NIEKO nerašo, jei užduotis netinkama (composite_job_valid, įskaitant: be bazės — tik v210a išėjimas); sėkmės atveju visas išėjimo grain'as perrašomas
// (padding = 0, net jei buferis purvinas); rezultatas bitas į bitą lygus etalonui.
// Įgyvendinimas eilutinis, sveikaisiais skaičiais; darbinės eilutės — thread_local buferiai (po pirmo iškvietimo be allocation).
#include <cstddef>
#include <cstdint>

#include "func/thread_pool.hpp"
#include "gfx/raster.hpp"

namespace vramflow::gfx::cpu {

bool rgba_to_v210a(const std::uint8_t* rgba, int w, int h, std::size_t stride, AlphaIn mode, const Raster& r,
                   int parity, std::uint8_t* grain);

// Vienas darbas; pool == nullptr — viena gija (kviečianti), kitaip eilutės dalijamos tarp pool gijų.
bool composite(const std::uint8_t* base, const Raster& base_r, const Layer* layers, int k, const Raster& out_r,
               int parity, std::uint8_t* out, mxlv::func::ThreadPool* pool = nullptr);

// N išėjimų to paties lauko (parity) vienu paleidimu: visų darbų eilučių blokai dalijami tarp pool gijų.
// Patikra bendra visiems variantams (gfx::composite_job_valid, R11/R13): bent vienas netinkamas darbas -> NIEKO nerašoma,
// false (kaip etalonas ir CUDA). Užduočių sąrašas — thread_local scratch kvietėjo gijoje.
bool composite_batch(const CompositeJob* jobs, int n, int parity, mxlv::func::ThreadPool* pool);

}  // namespace vramflow::gfx::cpu
