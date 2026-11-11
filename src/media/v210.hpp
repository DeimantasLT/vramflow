// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// v210 pakavimas (architektūra GPU funkcijų §6.1): 32 bitų LE žodis = 3 × 10 bitų (0–9, 10–19, 20–29), bitai 30–31 = 0.
// 6 pikseliai = 4 žodžiai: w0 = Cb0|Y0|Cr0, w1 = Y1|Cb1|Y2, w2 = Cr1|Y3|Cb2, w3 = Y4|Cr2|Y5.
// Cbk/Crk — pikselių porai (2k, 2k+1), ko-situota su Y(2k). 1920 px → 320 grupių = 1280 žodžių = 5120 B eilutei.
#include <cstdint>

#include "common/hd.hpp"

namespace mxlv::media {

constexpr int kWidth = 1920;
constexpr int kFrameHeight = 1080;
constexpr int kFieldHeight = 540;
constexpr int kGroupsPerLine = 320;
constexpr int kWordsPerLine = 1280;                          // uint32 žodžių eilutėje
constexpr std::uint32_t kFieldWords = kWordsPerLine * kFieldHeight;  // 691 200
constexpr std::uint32_t kFieldBytes = kFieldWords * 4;               // 2 764 800

struct Group {
    std::uint16_t y[6];
    std::uint16_t cb[3];
    std::uint16_t cr[3];
};

MXLV_HD void unpack_group(const std::uint32_t* w, Group& g) {
    g.cb[0] = w[0] & 0x3FF; g.y[0] = (w[0] >> 10) & 0x3FF; g.cr[0] = (w[0] >> 20) & 0x3FF;
    g.y[1] = w[1] & 0x3FF; g.cb[1] = (w[1] >> 10) & 0x3FF; g.y[2] = (w[1] >> 20) & 0x3FF;
    g.cr[1] = w[2] & 0x3FF; g.y[3] = (w[2] >> 10) & 0x3FF; g.cb[2] = (w[2] >> 20) & 0x3FF;
    g.y[4] = w[3] & 0x3FF; g.cr[2] = (w[3] >> 10) & 0x3FF; g.y[5] = (w[3] >> 20) & 0x3FF;
}

// hi — bitai 30–31 kiekvienam žodžiui (išsaugomi iš šaltinio; naujam turiniui 0).
MXLV_HD void pack_group(const Group& g, std::uint32_t* w, const std::uint32_t hi[4] = nullptr) {
    w[0] = g.cb[0] | (std::uint32_t(g.y[0]) << 10) | (std::uint32_t(g.cr[0]) << 20);
    w[1] = g.y[1] | (std::uint32_t(g.cb[1]) << 10) | (std::uint32_t(g.y[2]) << 20);
    w[2] = g.cr[1] | (std::uint32_t(g.y[3]) << 10) | (std::uint32_t(g.cb[2]) << 20);
    w[3] = g.y[4] | (std::uint32_t(g.cr[2]) << 10) | (std::uint32_t(g.y[5]) << 20);
    if (hi)
        for (int i = 0; i < 4; ++i) w[i] |= hi[i] & 0xC0000000u;
}

}  // namespace mxlv::media
