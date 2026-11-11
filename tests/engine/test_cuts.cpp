// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V1: laikrodis, žingsnių detektorius ir ribų matematika (planas docs/planai/2026-11-12-tick-variklis.md §6 D2, §9 V1).
#include <catch2/catch_test_macros.hpp>

#include <mxl/time.h>

#include <cstdint>
#include <random>
#include <string>

#include "engine/clock.hpp"
#include "engine/cuts.hpp"
#include "timing/index.hpp"

using namespace mxlv::engine;
using mxlv::timing::Rational;
using mxlv::timing::time_of;

TEST_CASE("cuts: numatytosios pagal dažnį", "[engine][cuts]") {
    {
        const CutConfig c = default_cuts({50, 1});
        CHECK(period_ns({50, 1}) == 20'000'000);
        CHECK(c.read_offset_ns == 500'000);
        CHECK(c.expect_ns == 0);
        const Index i = 123'456'789;
        const std::int64_t t = time_of(i, {50, 1});
        const Cuts k = cuts_for(i, c);
        CHECK(k.open - t == 500'000);
        CHECK(k.a - t == 5'500'000);
        CHECK(k.has_b);
        CHECK(k.b - t == 15'000'000);
        CHECK_FALSE(k.has_c);
    }
    {
        const CutConfig c = default_cuts({60000, 1001});
        CHECK(period_ns({60000, 1001}) == 16'683'333);
        CHECK(c.ready_wait_ns == 4'170'833);
        CHECK(c.deadline_ns == 12'512'499);
        CHECK(c.guard_ns == 2'502'499);
    }
    {
        const CutConfig c = default_cuts({25, 1});
        CHECK(period_ns({25, 1}) == 40'000'000);
        CHECK(c.ready_wait_ns == 10'000'000);
        CHECK(c.deadline_ns == 30'000'000);
        CHECK(c.guard_ns == 6'000'000);
    }
}

TEST_CASE("cuts: --deadline-ms 30 prie 50 Hz", "[engine][cuts]") {
    CutConfig c = default_cuts({50, 1});
    c.deadline_ns = 30'000'000;
    std::string warn;
    CHECK_FALSE(clamp_deadline(c, warn));
    CHECK(warn.empty());
    const Index i = 1000;
    const std::int64_t t = time_of(i, {50, 1});
    const Cuts k = cuts_for(i, c);
    CHECK(k.a - t == 5'500'000);
    CHECK(k.b - t == 17'500'000);  // t_open(i+1) − G = 20,5 − 3
    REQUIRE(k.has_c);
    CHECK(k.c - t == 30'000'000);
}

TEST_CASE("cuts: apkarpymas", "[engine][cuts]") {
    CutConfig c = default_cuts({50, 1});
    c.deadline_ns = 60'000'000;
    std::string warn;
    CHECK(clamp_deadline(c, warn));
    CHECK(c.deadline_ns == 37'000'000);
    CHECK(warn.find("60000000") != std::string::npos);
    CHECK(warn.find("37000000") != std::string::npos);
    const Index i = 77;
    const Cuts k = cuts_for(i, c);
    CHECK(k.c - time_of(i, {50, 1}) == 37'000'000);
}

TEST_CASE("cuts: legacy", "[engine][cuts]") {
    const CutConfig c = legacy_cuts({50, 1}, 500'000, 30'000'000);
    CHECK(c.single_phase);
    const Index i = 4242;
    const std::int64_t t = time_of(i, {50, 1});
    const Cuts k = cuts_for(i, c);
    CHECK(k.open - t == 500'000);
    CHECK(k.a - t == 30'000'000);  // viena riba = terminas
    CHECK_FALSE(k.has_b);
    CHECK_FALSE(k.has_c);
    CHECK(k.b == k.a);
}

TEST_CASE("step_detector", "[engine][clock]") {
    VirtualClock clk(1'700'000'000'000'000'000ull);
    StepDetector sd;
    CHECK(sd.check(clk) == 0);  // pirmas — tik atskaita
    clk.advance(7'000'000);
    CHECK(sd.check(clk) == 0);  // abu laikrodžiai kartu — ne žingsnis
    clk.step(4'900'000);
    CHECK(sd.check(clk) == 0);
    clk.step(5'100'000);
    CHECK(sd.check(clk) == 5'100'000);
    CHECK(sd.steps == 1);
    CHECK(sd.back_ns == 0);
    clk.step(-1'290'000'000);
    CHECK(sd.check(clk) == -1'290'000'000);
    CHECK(sd.steps == 2);
    CHECK(sd.back_ns == 1'290'000'000);
}

TEST_CASE("virtual_clock: miegas, suplanuotas žingsnis ir pavėlavimas", "[engine][clock]") {
    const std::atomic<bool> stop{false};
    VirtualClock clk(10'000'000'000);
    clk.sleep_until(10'020'000'000, stop);
    CHECK(clk.now_tai() == 10'020'000'000);
    CHECK(clk.now_mono_raw() - clk.mono_start() == 20'000'000);
    // Žingsnis atgal miegant: miegama iki tikslo naujoje laiko juostoje (monotoninis +1,3 s daugiau)
    clk.schedule_step(clk.now_mono_raw() + 5'000'000, -1'300'000'000);
    clk.sleep_until(10'040'000'000, stop);
    CHECK(clk.now_tai() == 10'040'000'000);
    CHECK(clk.now_mono_raw() - clk.mono_start() == 1'340'000'000);
    clk.oversleep_once(45'000'000);
    clk.sleep_until(10'060'000'000, stop);
    CHECK(clk.now_tai() == 10'105'000'000);
    clk.sleep_until(10'000'000'000, stop);  // praeityje — negrįžta atgal
    CHECK(clk.now_tai() == 10'105'000'000);
}

TEST_CASE("due_index: sutampa su mxlTimestampToIndex", "[engine][cuts]") {
    constexpr Rational kRates[] = {{25, 1}, {50, 1}, {30000, 1001}, {60000, 1001}};
    std::mt19937_64 rng(20261112);
    std::uniform_int_distribution<std::uint64_t> dist(0, 4'000'000'000'000'000'000ull);
    for (const Rational r : kRates) {
        const CutConfig c = default_cuts(r);
        const mxlRational mr{r.num, r.den};
        std::uint64_t mismatches = 0;
        for (int k = 0; k < 10'000; ++k) {
            const std::uint64_t t = dist(rng);
            if (due_index(t, c) != mxlTimestampToIndex(&mr, t)) ++mismatches;
        }
        INFO("rate " << r.num << "/" << r.den);
        CHECK(mismatches == 0);
    }
}
