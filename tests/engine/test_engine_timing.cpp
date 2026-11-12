// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V5: pernešimas (C riba), variklio vėlavimas ir laikrodžio žingsniai (T11), lap patikra (T12) — planas §9 V5.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
using Kind = RecordingOutput::Kind;
std::int64_t rel(const RecordingOutput::Event* e, Index i) { return static_cast<std::int64_t>(e->tai) - T(i); }
}  // namespace

TEST_CASE("engine: pernešimas", "[engine][carry]") {
    constexpr std::uint32_t kN = 4, kK = 2;
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;  // --deadline-ms 30 prie 50 Hz: B = T + 17,5, C = T + 30
    Rig r(kN, c);
    r.ins[kK]->arrival = [](Index j) { return j >= kI ? std::int64_t{26'000'000} : std::int64_t{1'200'000}; };
    r.stop_after(kI + 4);
    r.build();
    r.run();

    const auto* ki = r.outs[kK]->find(kI);
    REQUIRE(ki != nullptr);
    CHECK(ki->kind == Kind::Commit);
    CHECK(rel(ki, kI) == 26'000'000);  // k@i — C fazėje, po t_open(i + 1)
    const auto c1 = r.fn.calls_for(kI + 1);
    REQUIRE(c1.size() == 2);
    CHECK(c1[0]->phase == 0);
    CHECK(static_cast<std::int64_t>(c1[0]->tai) - T(kI + 1) == 1'200'000);  // kiti — kaip be gedimo
    CHECK(c1[0]->units.size() == 3);
    CHECK(c1[1]->phase == 2);
    CHECK(static_cast<std::int64_t>(c1[1]->tai) - T(kI + 1) == 26'000'000);  // k@(i + 1) po to
    const auto* ki1 = r.outs[kK]->find(kI + 1);
    REQUIRE(ki1 != nullptr);
    CHECK(ki1->kind == Kind::Commit);
    // Rašytojo k seka be spragos
    const auto& ev = r.outs[kK]->events;
    for (std::size_t n = 0; n < ev.size(); ++n) {
        CHECK(ev[n].kind == Kind::Commit);
        CHECK(ev[n].index == 1000 + n);
    }
    // i … i + 4: end_tick(i + 4) — tik po prisijungusio k@(i + 4) (ties T(i + 4) + 26 ms): prisijungę — tick'o nariai
    CHECK(r.out(kK).late == 5);
    CHECK(r.in(kK).lost == 0);
    CHECK(r.in(kK).stalls == 0);
    for (std::uint32_t u = 0; u < kN; ++u)
        if (u != kK) {
            CHECK(r.out(u).late_skips == 0);
            CHECK(r.out(u).late == 0);
        }
    CHECK(r.hub.engine().carry_overrun == 0);
}

TEST_CASE("engine: variklis pavėlavo 45 ms", "[engine][skips]") {
    constexpr Index kI = 1010;
    Rig r(3);
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI - 1) r.clk.oversleep_once(45'000'000);  // pabudimas iki t_open(i) — 45 ms per vėlai
        if (i >= kI + 4) r.stop = true;
    };
    r.build();
    r.run();
    for (std::uint32_t u = 0; u < 3; ++u) {
        INFO("u = " << u);
        CHECK(r.out(u).late_skips == 2);  // i ir i + 1: index_of(T(i) + 45,5 ms) = i + 2
        CHECK(r.out(u).clock_skips == 0);
        CHECK(r.outs[u]->find(kI) == nullptr);  // jokio pasivijimo
        CHECK(r.outs[u]->find(kI + 1) == nullptr);
        const auto* e = r.outs[u]->find(kI + 2);
        REQUIRE(e != nullptr);
        CHECK(e->kind == Kind::Commit);
        CHECK(rel(e, kI + 2) == 5'500'000);  // atidaryta pavėlavus, iškart (A riba jau praėjo)
        const auto* e3 = r.outs[u]->find(kI + 3);
        REQUIRE(e3 != nullptr);
        CHECK(rel(e3, kI + 3) == 1'200'000);
    }
}

TEST_CASE("engine: šuolis pirmyn +1,3 s", "[engine][clock]") {
    constexpr Index kI = 1010;
    Rig r(2);
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI) r.clk.schedule_step(r.clk.now_mono_raw() + 10'000'000, 1'300'000'000);  // miegant iki t_open(i+1)
        if (i >= kI + 68) r.stop = true;
    };
    r.build();
    r.run();
    REQUIRE(r.fn.steps.size() == 1);
    CHECK(r.fn.steps[0] == 1'300'000'000);
    CHECK(r.hub.engine().clock_steps == 1);
    CHECK(r.hub.engine().clock_backward == 0);
    for (std::uint32_t u = 0; u < 2; ++u) {
        INFO("u = " << u);
        // Pabusta ties T(i + 65) + 11,2 ms: t_open(i + 65) praėjo, t_open(i + 66) — ne. Praleidžiami tik indeksai,
        // kurių kito tick'o atidarymas jau praėjo: i + 1 … i + 64 (planas sakė 65 — su apvalinimu index_of).
        CHECK(r.out(u).clock_skips == 64);
        CHECK(r.out(u).late_skips == 0);
        CHECK(r.outs[u]->find(kI + 64) == nullptr);
        const auto* e = r.outs[u]->find(kI + 65);
        REQUIRE(e != nullptr);
        CHECK(e->kind == Kind::Commit);
        CHECK(static_cast<std::int64_t>(e->tai) - T(kI + 65) == 11'200'000);  // atidarytas pavėlavus, iškart
    }
}

TEST_CASE("engine: šuolis atgal −1,3 s", "[engine][clock]") {
    constexpr Index kI = 1010;
    Rig r(2);
    std::map<Index, std::int64_t> open_mono;
    r.fn.on_begin_tick = [&](Index i, TickContext&) { open_mono[i] = r.clk.now_mono_raw(); };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI) r.clk.schedule_step(r.clk.now_mono_raw() + 10'000'000, -1'300'000'000);
        if (i >= kI + 3) r.stop = true;
    };
    r.build();
    r.run();
    REQUIRE(r.fn.steps.size() == 1);
    CHECK(r.fn.steps[0] == -1'300'000'000);
    CHECK(r.hub.engine().clock_backward == 1);
    REQUIRE(open_mono.count(kI) == 1);
    REQUIRE(open_mono.count(kI + 1) == 1);
    CHECK(open_mono[kI + 1] - open_mono[kI] == 1'320'000'000);  // virtualus miegas P + 1,3 s
    for (std::uint32_t u = 0; u < 2; ++u) {
        INFO("u = " << u);
        const auto& ev = r.outs[u]->events;
        for (std::size_t n = 0; n < ev.size(); ++n) CHECK(ev[n].index == 1000 + n);  // joks indeksas du kartus
        CHECK(r.out(u).late_skips + r.out(u).clock_skips == 0);
    }
}

TEST_CASE("engine: žingsnis laukiant uždaro atvirus tick'us", "[engine][clock]") {
    constexpr Index kI = 1010;
    Rig r(2);
    r.ins[1]->arrival = [](Index j) { return j == kI ? std::int64_t{8'000'000} : std::int64_t{1'200'000}; };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI - 1) {  // žingsnis +20 ms ties T(i) + 7 ms (vienetas 1 laukia B fazėje)
            const std::int64_t mono_at = r.clk.now_mono_raw() + (T(kI) + 7'000'000 - static_cast<std::int64_t>(r.clk.now_tai()));
            r.clk.schedule_step(mono_at, 20'000'000);
        }
        if (i >= kI + 2) r.stop = true;
    };
    r.build();
    r.run();
    REQUIRE(r.fn.steps.size() == 1);
    CHECK(r.fn.steps[0] == 20'000'000);
    CHECK(r.hub.engine().step_gaps == 1);
    bool clock_gap = false;
    for (const auto& e : r.fn.ends)
        if (e.unit == 1 && e.index == kI) clock_gap = e.outcome == Outcome::GapClock;
    CHECK(clock_gap);
    const auto* g = r.outs[1]->find(kI);
    REQUIRE(g != nullptr);
    CHECK(g->kind == Kind::Gap);
}

TEST_CASE("engine: skip_align 2", "[engine][skips]") {
    constexpr Index kI = 1011;  // nelyginis: pabusta ties T(i) + 45,5 ms → paskutinis atidarytinas 1013 → lygiuojama į 1014
    EngineConfig c = Rig::test_config();
    c.skip_align = 2;
    Rig r(1, c);
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI - 1) r.clk.oversleep_once(45'000'000);
        if (i >= kI + 6) r.stop = true;
    };
    r.build();
    r.run();
    CHECK(r.out(0).late_skips == 3);
    Index after = kNoIndex;
    for (const auto& e : r.outs[0]->events)
        if (e.index > kI - 1) {
            after = e.index;
            break;
        }
    CHECK(after == 1014);
    CHECK(after % 2 == 0);
}

TEST_CASE("engine: lap patikra atšaukia išėjimą", "[engine][lap]") {
    constexpr Index kI = 1010;
    Rig r(2);
    r.ins[1]->lapped_at = [](Index j) { return j == kI; };
    r.stop_after(kI + 1);
    r.build();
    r.run();
    CHECK(r.out(1).lapped == 1);
    CHECK(r.outs[1]->cancels == 1);
    // Atšauktas grain'as užpildomas pagal politiką (invalid) — skaitytojas nemato skylės (TOO_EARLY iki i + 1)
    const auto* g = r.outs[1]->find(kI);
    REQUIRE(g != nullptr);
    CHECK(g->kind == Kind::Gap);
    CHECK(g->action == GapAction::Invalid);
    CHECK(r.out(1).gap_invalid == 0);  // priežastis — lapped, ne politikos spraga
    CHECK(r.hub.unit(r.eng->unit_id(1)).lapped == 1);
    CHECK(r.outs[0]->find(kI) != nullptr);
    bool lapped = false;
    for (const auto& e : r.fn.ends)
        if (e.unit == 1 && e.index == kI) lapped = e.outcome == Outcome::Lapped;
    CHECK(lapped);
}
