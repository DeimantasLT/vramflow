// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Diskretaus ring'o lango aritmetika (sutampa su PosixDiscreteFlowReader: skaitomi grainCount-1 naujausių).
#include <cstdint>

namespace mxlv {

// Ar indeksas dar skaitomas, kai head = paskutinis commit'intas indeksas.
constexpr bool readable(std::uint64_t head, std::uint64_t idx, std::uint32_t grain_count) {
    return idx <= head && head - idx <= static_cast<std::uint64_t>(grain_count) - 2;
}

// Po patikros: jei head nuėjo per toli, slot'as galėjo būti perrašomas tikrinimo metu → rezultatą atmesti.
constexpr bool lapped(std::uint64_t head_after, std::uint64_t idx, std::uint32_t grain_count) {
    return head_after > idx + static_cast<std::uint64_t>(grain_count) - 2;
}

constexpr std::uint32_t slot_of(std::uint64_t idx, std::uint32_t slot_count) {
    return static_cast<std::uint32_t>(idx % slot_count);
}

}  // namespace mxlv
