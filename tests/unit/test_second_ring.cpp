// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// SecondRing (B2): sekundžių žiedas lateness_s / loss_s / commit_lateness_s — atmintis nebeauga, tick'e be alokacijų.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "common/second_ring.hpp"

namespace {
std::atomic<std::uint64_t> g_new{0};
thread_local bool g_on = false;
}  // namespace

// Skaičiuojantis alokatorius (tik kai g_on — šio testo gijoje).
void* operator new(std::size_t n) {
    if (g_on) g_new.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n != 0 ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

TEST_CASE("second_ring: 100 000 imituotų sekundžių — atmintis pastovi, be alokacijų", "[unit][second_ring]") {
    mxlv::SecondRing lat, loss;
    REQUIRE(lat.capacity() == 3600);
    g_new = 0;
    g_on = true;
    for (std::uint64_t s = 1'000'000; s < 1'100'000; ++s)
        for (int f = 0; f < 50; ++f) {  // 50 laukų per sekundę
            lat.max(s, static_cast<std::uint64_t>(f * 100 + s % 7));
            if (f == 3) loss.add(s);
        }
    g_on = false;
    CHECK(g_new.load() == 0);
    CHECK(lat.size() == 3600);
    CHECK(lat.capacity() == 3600);
    CHECK(loss.size() == 3600);
    std::vector<std::uint64_t> secs;
    lat.for_each([&](std::uint64_t s, std::uint64_t v) {
        secs.push_back(s);
        CHECK(v == 4900 + s % 7);
    });
    REQUIRE(secs.size() == 3600);
    CHECK(secs.front() == 1'100'000 - 3600);  // seniausia išlikusi
    CHECK(secs.back() == 1'099'999);
    loss.for_each([&](std::uint64_t, std::uint64_t v) { CHECK(v == 1); });
    CHECK(lat.dropped() == 0);
}

TEST_CASE("second_ring: šiek tiek senesnė sekundė randama, per sena — atmetama ir suskaičiuojama", "[unit][second_ring]") {
    mxlv::SecondRing r(4);
    r.max(10, 5);
    r.max(11, 7);
    r.max(10, 9);  // pernešimas: senesnio indekso pabaiga po naujesnio
    r.add(13);
    r.max(12, 1);  // tarp 11 ir 13 įrašo nėra — neįterpiama (tvarka išlieka)
    std::vector<std::pair<std::uint64_t, std::uint64_t>> v;
    r.for_each([&](std::uint64_t s, std::uint64_t x) { v.emplace_back(s, x); });
    REQUIRE(v.size() == 3);
    CHECK(v[0] == std::pair<std::uint64_t, std::uint64_t>{10, 9});
    CHECK(v[1] == std::pair<std::uint64_t, std::uint64_t>{11, 7});
    CHECK(v[2] == std::pair<std::uint64_t, std::uint64_t>{13, 1});
    CHECK(r.dropped() == 1);
    for (std::uint64_t s = 14; s < 20; ++s) r.add(s);
    CHECK(r.size() == 4);
    r.add(15);  // jau išstumta iš žiedo
    CHECK(r.dropped() == 2);
}
