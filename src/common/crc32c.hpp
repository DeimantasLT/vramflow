// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// CRC32C (Castagnoli, atspindėtas polinomas 0x82F63B78), programinis, host + device.
#include <cstddef>
#include <cstdint>

#include "hd.hpp"

namespace mxlv {

MXLV_HD std::uint32_t crc32c(const std::uint8_t* p, std::size_t n) {
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int b = 0; b < 8; ++b) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u)));
    }
    return ~c;
}

}  // namespace mxlv
