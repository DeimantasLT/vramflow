// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// 64 B grain'o antraštė pirmuose payload baituose (architektūra §4.2).
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "crc32c.hpp"
#include "hd.hpp"

namespace mxlv {

constexpr std::uint32_t kMagic = 0x4D584C56u;  // "MXLV"
constexpr std::uint16_t kHeaderVersion = 1;

struct GrainHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t header_size;
    std::uint64_t flow_seed;
    std::uint64_t grain_index;
    std::uint64_t origin_tai_ns;
    std::uint64_t payload_fold64;
    std::uint32_t payload_words;
    std::uint32_t writer_pid;
    std::uint64_t seq;
    std::uint32_t reserved;
    std::uint32_t header_crc32c;
};
static_assert(sizeof(GrainHeader) == 64);
static_assert(offsetof(GrainHeader, flow_seed) == 8);
static_assert(offsetof(GrainHeader, grain_index) == 16);
static_assert(offsetof(GrainHeader, origin_tai_ns) == 24);
static_assert(offsetof(GrainHeader, payload_fold64) == 32);
static_assert(offsetof(GrainHeader, payload_words) == 40);
static_assert(offsetof(GrainHeader, seq) == 48);
static_assert(offsetof(GrainHeader, header_crc32c) == 60);

MXLV_HD std::uint32_t header_crc(const GrainHeader& h) {
    return crc32c(reinterpret_cast<const std::uint8_t*>(&h), 60);
}

MXLV_HD void seal_header(GrainHeader& h) { h.header_crc32c = header_crc(h); }

enum class HdrError : std::uint8_t { Ok = 0, Magic, Version, Size, Crc, Words, Seed };

inline const char* to_string(HdrError e) {
    switch (e) {
        case HdrError::Ok: return "ok";
        case HdrError::Magic: return "magic";
        case HdrError::Version: return "version";
        case HdrError::Size: return "header_size";
        case HdrError::Crc: return "crc";
        case HdrError::Words: return "payload_words";
        case HdrError::Seed: return "flow_seed";
    }
    return "?";
}

// CRC tikrinamas pirmiau už laukus, kad bet koks antraštės bito pakeitimas būtų aptiktas kaip korupcija.
inline HdrError validate(const GrainHeader& h, std::uint64_t expected_seed, std::uint32_t expected_words) {
    if (h.header_crc32c != header_crc(h)) return HdrError::Crc;
    if (h.magic != kMagic) return HdrError::Magic;
    if (h.version != kHeaderVersion) return HdrError::Version;
    if (h.header_size != sizeof(GrainHeader)) return HdrError::Size;
    if (h.payload_words != expected_words) return HdrError::Words;
    if (h.flow_seed != expected_seed) return HdrError::Seed;
    return HdrError::Ok;
}

inline GrainHeader header_from_bytes(const void* p) {
    GrainHeader h;
    std::memcpy(&h, p, sizeof h);
    return h;
}

}  // namespace mxlv
