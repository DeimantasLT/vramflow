// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Fuzz taikinys: MXL video flow aprašo skaitymas (vramflow::format_from_flow_def, src/common/format.hpp).
// Aprašą rašo kitas procesas tame pačiame domene — jo baitai ne mūsų (planas 2026-11-17 §1.5, kriterijus S2).
// Invariantai [RF-4]: priimtas formatas turi matmenis 1…65536, o dydžių aritmetika (64 bitai) nepersipildo;
// atmetimas visada turi priežastį.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "common/format.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string json(reinterpret_cast<const char*>(data), size);
    vramflow::VideoFormat f{};
    std::string err;
    if (!vramflow::format_from_flow_def(json, f, err)) {
        if (err.empty()) std::abort();
        return 0;
    }
    const vramflow::gfx::Raster& r = f.raster;
    if (r.w < 1 || r.w > 65536 || r.h < 1 || r.h > 65536) std::abort();
    if (r.interlaced && (r.h % 2) != 0) std::abort();
    // Didžiausias galimas grain'as: v210a 65536 × 65536 (fill 1366 · 128 B ir key 21846 · 4 B eilutei)
    const std::size_t gb = vramflow::gfx::grain_bytes(r);
    if (gb == 0 || gb > (std::size_t{1366} * 128 + std::size_t{21846} * 4) * 65536) std::abort();
    if (f.frame_rate.den == 0) std::abort();
    return 0;
}
