// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variantai H / G: CUDA RGBA -> v210a ir batch kompozitorius (alfa kompozitoriaus 5 užduotis). Sutartis ta pati kaip
// gfx::ref (gfx_ref.hpp), rezultatas bitas į bitą lygus etalonui; skirtumas — klaidos grąžinamos kaip cudaError_t:
// netinkama geometrija (nelyginis plotis / x, neigiami x / y, nelyginis y interlaced rastre, neatitinkantys rastrai,
// nullptr, paritetas ne 0/1 interlaced rastre, NaN opacity, k per didelis) -> cudaErrorInvalidValue, NIEKAS nepaleidžiama
// (patikra visa host pusėje, prieš bet kokį paleidimą). Sėkmės atveju visas išėjimo grain'as perrašomas (padding = 0,
// net jei device buferis purvinas).
//
// Rodyklės (grain, rgba, bazė, sluoksniai, išėjimas) — device atmintis, išlygiuota bent 4 B. Funkcijos asinchroninės
// pateiktame stream'e. Būsenos tarp kvietimų nėra: aprašai perduodami __grid_constant__ kernel parametru (kopijuojami
// paleidimo metu), todėl jokio cudaMalloc / cudaMemcpy kiekvienam kvietimui ir host masyvus (jobs, layers) galima
// iškart po grįžimo keisti. Gijų sauga: funkcijos pačios saugios kviesti iš kelių gijų; kiekvienas kvietėjas naudoja savo
// stream'ą (rezultatas stream'e uždedamas eilės tvarka). Išėjimas negali sutapti su sluoksnio grain'u (bazė = išėjimas
// leidžiama: kiekviena gija skaito ir rašo tik savo grupę).
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "gfx/raster.hpp"

namespace vramflow::gfx::cuda {

constexpr int kMaxBatch = 64;      // darbų (išėjimų) skaičius vienam paleidimui
constexpr int kMaxBatchLayers = 640;  // sluoksnių suma vienam paleidimui; didesnis batch'as dalijamas į kelis paleidimus
// Vienam darbui (išėjimui) leidžiamas sluoksnių skaičius ribojamas kMaxBatchLayers (didesnis k -> cudaErrorInvalidValue).

// Vienas laukas (paritetas parity) iš RGBA kadro (device, w×h, stride B) -> v210a grain'as (device). Semantika kaip
// ref::rgba_to_v210a (interlaced: kadro eilutės 2j+parity; progresyvus: visos eilutės, parity ignoruojamas).
cudaError_t rgba_to_v210a(const std::uint8_t* rgba_dev, int w, int h, std::size_t stride, AlphaIn mode, const Raster& r,
                          int parity, std::uint8_t* grain_dev, cudaStream_t s);

// Vienas paleidimas laukui visiems n išėjimų (kaip media::logo_fields). jobs — HOST masyvas su DEVICE rodyklėmis
// (CompositeJob::base, layers[i].grain ir out — device; pats `layers` masyvas — host). n <= kMaxBatch; n == 0 —
// nieko nedaro. Jei sluoksnių suma > kMaxBatchLayers, batch'as dalijamas į kelis paleidimus tame pačiame stream'e.
// parity validuojamas kaip etalone (interlaced išėjimui 0/1); pats kompozitoriaus kernel'is nuo jo nepriklauso.
cudaError_t composite_batch(const CompositeJob* jobs, int n, int parity, cudaStream_t s);

}  // namespace vramflow::gfx::cuda
