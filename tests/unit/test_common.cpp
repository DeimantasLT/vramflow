// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Unit testai be GPU ir be MXL (architektūra §5.2). Auksinės reikšmės — iš nepriklausomo Python skaičiavimo.
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>

#include "common/crc32c.hpp"
#include "common/header.hpp"
#include "common/histogram.hpp"
#include "common/index_math.hpp"
#include "common/pattern.hpp"
#include "common/pattern_ref.hpp"

using namespace mxlv;

TEST_CASE("pattern_ref: nepriklausoma implementacija sutampa su pattern.hpp ir auksinėmis reikšmėmis", "[pattern]") {
    CHECK(ref::word(0, 0, 16) == 0x197b8feeu);
    CHECK(ref::word(1, 2, 16) == 0x2b573fc4u);
    CHECK(ref::word(0x0123456789ABCDEFull, 88123456789ull, 17) == 0x09dd12f4u);
    CHECK(ref::word(0x0123456789ABCDEFull, 88123456789ull, 1382399) == 0x240e5868u);
    std::mt19937_64 rng(7);
    int diff = 0;
    for (int k = 0; k < 1'000'000; ++k) {
        const auto s = rng(), ix = rng();
        const auto i = static_cast<std::uint32_t>(rng());
        diff += ref::word(s, ix, i) != word(s, ix, i);
    }
    CHECK(diff == 0);
    std::vector<std::uint32_t> w(4096);
    for (std::uint32_t i = 0; i < w.size(); ++i) w[i] = word(3, 4, i);
    CHECK(ref::fold64_of(w.data(), 4096) == fold64_of(w.data(), 4096));
    CHECK(ref::fold64_of(w.data(), 4096) == fold64_reference(3, 4, 4096));
}

TEST_CASE("pattern: auksinės reikšmės", "[pattern]") {
    CHECK(splitmix64(0) == 0xe220a8397b1dcdafull);
    CHECK(splitmix64(1) == 0x910a2dec89025cc1ull);
    CHECK(word(0, 0, 16) == 0x197b8feeu);
    CHECK(word(1, 2, 16) == 0x2b573fc4u);
    CHECK(word(0x0123456789ABCDEFull, 88123456789ull, 17) == 0x09dd12f4u);
    CHECK(word(0x0123456789ABCDEFull, 88123456789ull, 1382399) == 0x240e5868u);
}

TEST_CASE("pattern: v210 legalumas ir kolizijos", "[pattern]") {
    std::mt19937_64 rng(42);
    std::unordered_set<std::uint32_t> seen;
    int collisions = 0;
    for (int k = 0; k < 1'000'000; ++k) {
        const auto w = word(rng(), rng(), static_cast<std::uint32_t>(rng()));
        REQUIRE(w <= 0x3FFFFFFFu);
        collisions += !seen.insert(w).second;
    }
    // 1e6 imčių iš 2^30 erdvės: tikėtina ~465 kolizijos (gimtadienio paradoksas); < 1e-3 dalis
    CHECK(collisions < 1000);
}

TEST_CASE("fold64: auksinės reikšmės ir jautrumas", "[fold]") {
    CHECK(fold64_reference(1, 2, 64) == 0x000001acda15a704ull);
    CHECK(fold64_reference(0x0123456789ABCDEFull, 88123456789ull, 1382400) == 0x9f7fd5fa9f07281bull);

    std::vector<std::uint32_t> w(256);
    for (std::uint32_t i = 0; i < w.size(); ++i) w[i] = word(7, 9, i);
    const auto base = fold64_of(w.data(), 256);
    CHECK(base == fold64_reference(7, 9, 256));
    auto swapped = w;
    std::swap(swapped[20], swapped[21]);
    CHECK(fold64_of(swapped.data(), 256) != base);
    auto flipped = w;
    flipped[100] ^= 1u << 7;
    CHECK(fold64_of(flipped.data(), 256) != base);
}

TEST_CASE("crc32c", "[crc]") {
    const char* s = "123456789";
    CHECK(crc32c(reinterpret_cast<const std::uint8_t*>(s), 9) == 0xe3069283u);
    CHECK(crc32c(nullptr, 0) == 0x00000000u);
}

namespace {
GrainHeader make_header() {
    GrainHeader h{};
    h.magic = kMagic;
    h.version = kHeaderVersion;
    h.header_size = 64;
    h.flow_seed = 0xABCDEFull;
    h.grain_index = 123456;
    h.origin_tai_ns = 999;
    h.payload_fold64 = 0x1122334455667788ull;
    h.payload_words = 1382400;
    h.writer_pid = 4242;
    h.seq = 17;
    seal_header(h);
    return h;
}
}  // namespace

TEST_CASE("header: round-trip ir kiekvieno baito sugadinimas", "[header]") {
    const auto h = make_header();
    std::uint8_t raw[64];
    std::memcpy(raw, &h, 64);
    const auto back = header_from_bytes(raw);
    CHECK(validate(back, 0xABCDEFull, 1382400) == HdrError::Ok);
    CHECK(back.grain_index == 123456);

    // Bet kurio bito pakeitimas baituose 0..63 turi būti aptiktas
    for (int byte = 0; byte < 64; ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            std::uint8_t c[64];
            std::memcpy(c, raw, 64);
            c[byte] ^= static_cast<std::uint8_t>(1u << bit);
            INFO("byte " << byte << " bit " << bit);
            CHECK(validate(header_from_bytes(c), 0xABCDEFull, 1382400) != HdrError::Ok);
        }
    }
}

TEST_CASE("header: lauko klaidos su teisingu CRC", "[header]") {
    auto h = make_header();
    h.magic = 1;
    seal_header(h);
    CHECK(validate(h, 0xABCDEFull, 1382400) == HdrError::Magic);
    h = make_header();
    CHECK(validate(h, 0xABCDEFull, 1000) == HdrError::Words);
    CHECK(validate(h, 1, 1382400) == HdrError::Seed);
}

TEST_CASE("index_math: langas sutampa su MXL reader'iu", "[index]") {
    // grainCount = 10, head = 100 → skaitomi 92..100
    CHECK_FALSE(readable(100, 91, 10));
    CHECK(readable(100, 92, 10));
    CHECK(readable(100, 100, 10));
    CHECK_FALSE(readable(100, 101, 10));
    CHECK_FALSE(lapped(100, 92, 10));
    CHECK(lapped(101, 92, 10));
    CHECK(slot_of(123, 10) == 3);
}

TEST_CASE("histogram: kvantiliai ≤ 3,2 % paklaida", "[histogram]") {
    Histogram h;
    for (std::uint64_t v = 1; v <= 100000; ++v) h.add(v * 1000);  // 1 µs .. 100 ms
    auto rel = [](double got, double want) { return std::abs(got - want) / want; };
    CHECK(rel(static_cast<double>(h.quantile(0.50)), 50'000'000.0) < 0.032);
    CHECK(rel(static_cast<double>(h.quantile(0.99)), 99'000'000.0) < 0.032);
    CHECK(rel(static_cast<double>(h.quantile(0.999)), 99'900'000.0) < 0.032);
    CHECK(h.max() == 100'000'000ull);
    CHECK(h.count() == 100000);

    Histogram a, b;
    a.add(10);
    b.add(1'000'000);
    a.merge(b);
    CHECK(a.count() == 2);
    CHECK(a.max() == 1'000'000);
    CHECK(Histogram{}.quantile(0.5) == 0);
    for (int bkt = 0; bkt < Histogram::kBuckets; ++bkt) CHECK(Histogram::bucket_of(Histogram::value_of(bkt)) == bkt);
    CHECK(Histogram::bucket_of(~0ull) == Histogram::kBuckets - 1);
}
