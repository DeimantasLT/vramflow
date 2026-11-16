// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// R13-2: libav kvietimų sargas — sprendimas (call_stuck) ir sargo gija su netikru laikrodžiu (be libav / MXL).
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ingest/libav_watchdog.hpp"

using namespace mxlv::ingest;

namespace {
constexpr std::uint64_t kS = 1'000'000'000ull;
}

TEST_CASE("libav_watchdog: call_stuck — riba, pažanga, ne libav viduje", "[ingest][watchdog]") {
    const std::uint64_t m = 30 * kS, t0 = 1000 * kS;
    // ne libav viduje — niekada
    CHECK(call_stuck(nullptr, t0, t0 + 5 * kS, 0, m, t0 + 3600 * kS) == Stuck::No);
    // riba: pertraukimo riba + atsarga (kill_at) praeita — užstrigo, net jei kabliukas ką tik kviestas
    CHECK(call_stuck("x", t0, t0 + 35 * kS, t0 + 35 * kS, m, t0 + 35 * kS) == Stuck::No);
    CHECK(call_stuck("x", t0, t0 + 35 * kS, t0 + 35 * kS, m, t0 + 35 * kS + 1) == Stuck::Deadline);
    // pažanga: be ribos (kill_at 0) — tik kol kabliukas kviečiamas; kilpa be kabliuko — po atsargos
    CHECK(call_stuck("x", t0, 0, t0 + 100 * kS, m, t0 + 120 * kS) == Stuck::No);
    CHECK(call_stuck("x", t0, 0, t0 + 100 * kS, m, t0 + 130 * kS + 1) == Stuck::NoProgress);
    // ankstesnio kvietimo kabliukas (senesnis už since) — skaičiuojama nuo įėjimo
    CHECK(call_stuck("x", t0, 0, t0 - 100 * kS, m, t0 + 29 * kS) == Stuck::No);
    CHECK(call_stuck("x", t0, 0, t0 - 100 * kS, m, t0 + 31 * kS) == Stuck::NoProgress);
    CHECK(call_stuck("x", t0, 0, 0, m, t0 + 31 * kS) == Stuck::NoProgress);
    // laikrodis atgal (now < since) — ne užstrigimas
    CHECK(call_stuck("x", t0, 0, 0, m, t0 - kS) == Stuck::No);
}

TEST_CASE("libav_watchdog: gija praneša apie užstrigusį kvietimą su šaltinio vardu, išėjęs — ne", "[ingest][watchdog]") {
    std::atomic<std::uint64_t> clock{1000 * kS};
    std::mutex m;
    std::vector<std::string> hits;
    std::vector<Stuck> why;
    LibavWatchdog wd(30 * kS, 7, [&] { return clock.load(); },
                     [&](const std::string& name, const char* what, Stuck s, std::uint64_t) {
                         std::lock_guard g(m);
                         hits.push_back(name + " / " + what);
                         why.push_back(s);
                     },
                     10);
    CallWatch* a = wd.add("šaltinis 0 (http://h/a.m3u8)");
    CallWatch* b = wd.add("šaltinis 1 (udp://127.0.0.1:1)");
    CHECK(a->margin_ns == 30 * kS);
    // a: įėjo į avio_open2 su riba; b: įėjo be ribos, kabliukas kviečiamas
    a->since = clock.load();
    a->kill_at = clock.load() + 32 * kS;
    a->what = "avio_open2";
    b->since = clock.load();
    b->what = "av_read_frame";
    auto settle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(80)); };
    settle();
    {
        std::lock_guard g(m);
        CHECK(hits.empty());
    }
    // 31 s: a dar ribose (pažangos taisyklė irgi — 31 > 30, bet kabliukas kviestas prieš 1 s), b — kabliukas kviečiamas
    clock += 31 * kS;
    a->last_poll = clock.load() - kS;
    b->last_poll = clock.load();
    settle();
    {
        std::lock_guard g(m);
        CHECK(hits.empty());
    }
    // a išėjo — nebestebimas, net praėjus ribai
    a->what = nullptr;
    clock += 10 * kS;
    b->last_poll = clock.load();
    settle();
    {
        std::lock_guard g(m);
        CHECK(hits.empty());
    }
    // b sukasi be kabliuko ilgiau nei atsarga — pranešama su vardu ir kvietimu
    clock += 31 * kS;
    settle();
    std::lock_guard g(m);
    REQUIRE_FALSE(hits.empty());
    CHECK(hits.front() == "šaltinis 1 (udp://127.0.0.1:1) / av_read_frame");
    CHECK(why.front() == Stuck::NoProgress);
}
