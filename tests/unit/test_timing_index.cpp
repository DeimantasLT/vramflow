// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// timing/index.hpp prieš tikrą MXL (mxlTimestampToIndex / mxlIndexToTimestamp — grynos funkcijos, GPU ir domeno nereikia).
#include <catch2/catch_test_macros.hpp>

#include <mxl/time.h>

#include <cstdint>
#include <random>

#include "timing/index.hpp"

using namespace mxlv::timing;

TEST_CASE("index: sutampa su MXL", "[timing][index]") {
    constexpr Rational kRates[] = {{25, 1}, {50, 1}, {60000, 1001}, {48000, 1}};
    std::mt19937_64 rng(20261111);
    // Visas int64 teigiamas diapazonas nebūtinas: MXL TAI ~1,7·10¹⁸; imama [0, 4·10¹⁸] — apima ir šiandieną, ir ribas.
    std::uniform_int_distribution<std::int64_t> dist(0, 4'000'000'000'000'000'000);
    for (const Rational r : kRates) {
        const mxlRational mr{r.num, r.den};
        std::uint64_t mismatches = 0;
        for (int i = 0; i < 100'000; ++i) {
            const std::int64_t t = i < 4 ? std::int64_t{i} * 10'000'000 : dist(rng);  // pirmi — mažos ir pusės kadro ribos
            const std::uint64_t idx = index_of(t, r);
            if (idx != mxlTimestampToIndex(&mr, static_cast<std::uint64_t>(t))) ++mismatches;
            if (static_cast<std::uint64_t>(time_of(idx, r)) != mxlIndexToTimestamp(&mr, idx)) ++mismatches;
        }
        INFO("rate " << r.num << "/" << r.den);
        CHECK(mismatches == 0);
    }
}

TEST_CASE("index: auksinės reikšmės ir apvalinimas", "[timing][index]") {
    CHECK(index_of(19'999'999, {25, 1}) == 0);
    CHECK(index_of(20'000'000, {25, 1}) == 1);  // pusė — į viršų
    CHECK(time_of(1, {25, 1}) == 40'000'000);
    CHECK(time_of(1, {60000, 1001}) == 16'683'333);  // 16 683 333,3 → 16 683 333
    CHECK(time_of(2, {60000, 1001}) == 33'366'667);  // 33 366 666,7 → 33 366 667
    CHECK(index_of(time_of(123'456'789, {60000, 1001}), {60000, 1001}) == 123'456'789);
    CHECK(index_of(1'000'000'000, {0, 1}) == kUndefinedIndex);
}
