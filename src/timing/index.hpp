// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// MXL indekso ↔ TAI laiko konversijos be MXL priklausomybės (header-only). Formulės bitas į bitą sutampa su
// mxlTimestampToIndex / mxlIndexToTimestamp (MXL lib/internal/include/mxl-internal/IndexConversion.hpp);
// sutapimą tvirtina testas „index: sutampa su MXL“ (tests/unit/test_timing_index.cpp).
#include <cstdint>

namespace mxlv::timing {

__extension__ typedef __int128 i128;  // 128 bitų tarpiniai (be -Wpedantic įspėjimo)

struct Rational {
    std::int64_t num, den;
};

// Negaliojantis dažnis (num arba den == 0) — kaip MXL_UNDEFINED_INDEX.
inline constexpr std::uint64_t kUndefinedIndex = ~std::uint64_t{0};

// round(t · rate), pusė į viršų (t ≥ 0). == mxlTimestampToIndex
constexpr std::uint64_t index_of(std::int64_t tai_ns, Rational rate) {
    if (rate.num == 0 || rate.den == 0) return kUndefinedIndex;
    return static_cast<std::uint64_t>((i128{tai_ns} * rate.num + i128{500'000'000} * rate.den) /
                                      (i128{1'000'000'000} * rate.den));
}

// round(index / rate) ns. == mxlIndexToTimestamp
constexpr std::int64_t time_of(std::uint64_t index, Rational rate) {
    if (rate.num == 0 || rate.den == 0) return static_cast<std::int64_t>(kUndefinedIndex);
    return static_cast<std::int64_t>((i128{index} * rate.den * 1'000'000'000 + rate.num / 2) / rate.num);
}

}  // namespace mxlv::timing
