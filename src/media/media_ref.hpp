// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// NEPRIKLAUSOMI CPU etalonai GPU media kernel'iams (architektūra GPU funkcijų §6.6): rašomi iš dokumento formulių,
// ne kopijuojant kernel'io kodą. Testai reikalauja 0 skirtingų baitų tarp GPU ir šių funkcijų.
#include <cstdint>
#include <string>
#include <vector>

#include "media/gen_content.hpp"
#include "media/v210.hpp"

namespace mxlv::media::ref {

// Logotipas CPU pusėje (kadro raiška). Konversija iš RGBA (straight alpha) — §6.3.
struct Logo {
    int w = 0, h = 0;
    std::vector<std::uint16_t> y10, cb10, cr10;  // y10: w*h; cb/cr: (w/2)*h
    std::vector<std::uint8_t> a8, ac8;           // a8: w*h; ac8: (w/2)*h
};
// rgba: w*h*4 baitų. Grąžina false, jei w nelyginis.
bool logo_from_rgba(const std::uint8_t* rgba, int w, int h, Logo& out);
// Procedūrinis bandomasis logotipas (384×216, alfa gradientas, baltos/raudonos/žalios/mėlynos juostos).
Logo builtin_logo();
// PNG per libavformat/libavcodec → RGBA. Grąžina klaidos tekstą arba "".
std::string load_png_rgba(const std::string& path, std::vector<std::uint8_t>& rgba, int& w, int& h);

// BT.709 siauro diapazono 10 bit iš 8 bit R'G'B'.
void rgb_to_ycbcr10(int r, int g, int b, int& y, int& cb, int& cr);
// Blend: out = (s·(255−a) + l·a + 127) / 255
int blend(int s, int l, int a);

// Laukas su logotipu (kadro koordinatės X,Y; parity — lauko paritetas). in/out — kFieldWords žodžių.
void logo_field(const std::uint32_t* in, std::uint32_t* out, const Logo& logo, int X, int Y, int parity);

// Du v210 laukai → NV12 (pitch_y ≥ 1920, pitch_uv ≥ 1920).
void weave_nv12(const std::uint32_t* top, const std::uint32_t* bottom, std::uint8_t* y, std::uint8_t* uv, int pitch_y, int pitch_uv);
// NV12 → laukas `parity`.
void split_nv12(const std::uint8_t* y, const std::uint8_t* uv, int pitch_y, int pitch_uv, int parity, std::uint32_t* out);
// Generatoriaus laukas (naudoja gen_content.hpp apibrėžimą).
void gen_field(const GenParams& g, std::uint64_t index, std::uint32_t* out);

// Barkodo dekodavimas iš v210 lauko (lauko eilutės 0..15). Grąžina 32 bitų žodį.
std::uint32_t decode_marker(const std::uint32_t* field);
// Barkodo dekodavimas iš NV12 kadro lauko `parity` (po kodavimo/dekodavimo; slenkstis 8 bit).
std::uint32_t decode_marker_nv12(const std::uint8_t* y, int pitch_y, int parity);

}  // namespace mxlv::media::ref
