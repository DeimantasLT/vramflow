// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// NEPRIKLAUSOMAS bitų tikslumo etalonas alfa grafikai ir kompozitoriui. Rašomas tiesiai iš pinned formulių
// (planas §„Matematika“ + kontrolieriaus R1/R7/R8), mėginių lygmeniu: išpakuoti -> skaičiuoti -> supakuoti, be
// optimizacijų. CPU ir CUDA įgyvendinimai testuojami baitas į baitą prieš šį etaloną; kodo iš jų čia nekopijuojama.
//
// Semantika: sluoksnio opacity -> o8 per gfx::opacity_to_o8 (float, lround, apkirpta į 0…255; t. y. opacity apkerpama į [0,1]).
// Sluoksnio DENGIAMAS taškas su ae = 0 virš ab = 0 tampa Y 64 / C 512 (taisyklė n == 0), o NEdengiami taškai pasilieka bazės
// fill (net kai jų ab = 0). Tai tyčia (R8) ir CPU / CUDA turi atkartoti.
// Sutartis (abi funkcijos): grąžina false ir NIEKO nerašo, jei geometrija netinkama: nelyginis plotis (R7), nelyginis x,
// neigiami x / y, nelyginis y interlaced rastre (R1), neatitinkantys rastrai, nullptr, paritetas ne 0/1 interlaced
// rastre, NaN opacity. Sėkmės atveju išėjimo grain'as (grain_bytes(out_r) B) perrašomas VISAS: key užpildymas
// (nenaudoti mėginiai, bitai 30–31) ir fill užpildymas (nenaudoti v210 mėginiai ir eilutės uodega) = 0, net jei buferis
// buvo purvinas.
#include <cstddef>
#include <cstdint>

#include "gfx/raster.hpp"

namespace vramflow::gfx::ref {

constexpr int kAlphaScale = 876;  // A: alfa skalė 0…A (key 64…940)

// Vienas laukas (paritetas p) iš progresyvinio RGBA kadro (w×h, stride B) -> v210a grain'as (fill + key).
// r.alpha privalo būti true, r.w == w, r.h == h. Interlaced r: laukas = kadro eilutės 2j+parity; progresyvus r: visos
// eilutės, parity ignoruojamas. order — kanalų tvarka (BGRA: baitai B, G, R, A; rezultatas kaip RGBA po sukeitimo).
bool rgba_to_v210a(const std::uint8_t* rgba, int w, int h, std::size_t stride, AlphaIn mode, const Raster& r,
                   int parity, std::uint8_t* grain, PixelOrder order = PixelOrder::RGBA);

// base: v210 arba v210a grain'as arba nullptr (--base none); out — v210 arba v210a pagal out_r.alpha. Sluoksniai
// dedami iš apačios į viršų (R8). Sluoksnio rastras be key (v210) = visiškai nepermatomas. Nukirpimas ties išėjimo
// riba. Interlaced: sluoksnio lauko eilutė j patenka į išėjimo lauko eilutę j + y/2 (parity pats nekeičia
// grain eilučių atitikties, bet privalo būti 0/1).
bool composite(const std::uint8_t* base, const Raster& base_r, const Layer* layers, int k, const Raster& out_r,
               int parity, std::uint8_t* out);

}  // namespace vramflow::gfx::ref
