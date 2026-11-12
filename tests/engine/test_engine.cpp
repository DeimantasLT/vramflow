// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V4: tick ciklas, fazės A / B, ašinis laukimas — virtualiu laiku, tikslūs laikai ns (planas §9 V4).
// 50 Hz: P = 20 ms, ro = 0,5 ms, Wr = 5 ms, D = 15 ms, G = 3 ms, M = 3. Sveikas įėjimas ateina T + 1,2 ms.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
using Kind = RecordingOutput::Kind;
}  // namespace

TEST_CASE("engine: sveikas tick'as", "[engine][loop]") {
    Rig r(4);
    r.build();
    r.stop_after(1004);
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Stopped);
    CHECK(res.first_index == 1000);
    CHECK(res.last_index == 1004);
    for (Index i = 1001; i <= 1004; ++i) {  // 1000 — pirmas tick'as: naujai prijungti įėjimai dar Late
        INFO("i = " << i);
        const auto calls = r.fn.calls_for(i);
        REQUIRE(calls.size() == 1);
        CHECK(calls[0]->phase == 0);
        CHECK(static_cast<std::int64_t>(calls[0]->tai) - T(i) == 1'200'000);
        CHECK(calls[0]->units.size() == 4);
        for (const auto& o : r.outs) {
            const auto* e = o->find(i);
            REQUIRE(e != nullptr);
            CHECK(e->kind == Kind::Commit);
            CHECK(static_cast<std::int64_t>(e->tai) - T(i) == 1'200'000);
        }
    }
    std::uint64_t waits = 0;
    for (const auto& in : r.ins) waits += in->waits;
    CHECK(waits == 5);  // vienas blokuojantis laukimas tick'ui
    for (std::uint32_t u = 0; u < 4; ++u) {
        CHECK(r.out(u).ok == 5);  // ir 1000: visi Late, kvorumo nėra → A aibė = visi (žr. engine.cpp)
        CHECK(r.out(u).late == 0);
        CHECK(r.in(u).lost == 0);
    }
}

TEST_CASE("engine: miręs įėjimas", "[engine][loop][isolation]") {
    constexpr std::uint32_t kN = 32, kK = 6;
    constexpr Index kI = 1010;
    Rig r(kN);
    ScriptedInput& k = *r.ins[kK];
    k.arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    k.alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kI); };
    k.attachable = k.alive;
    r.build();
    r.stop_after(kI + 49);
    r.run();

    const auto ci = r.fn.calls_for(kI);
    REQUIRE(ci.size() == 1);
    CHECK(static_cast<std::int64_t>(ci[0]->tai) - T(kI) == 5'500'000);
    CHECK(ci[0]->units.size() == 31);
    CHECK(ci[0]->held_ns == 4'300'000);
    CHECK(r.eng->in_state(kK, 0) == InState::Detached);
    CHECK(r.in(kK).detaches == 1);
    const auto* g = r.outs[kK]->find(kI);
    REQUIRE(g != nullptr);
    CHECK(g->kind == Kind::Gap);
    CHECK(g->action == GapAction::Invalid);
    CHECK(static_cast<std::int64_t>(g->tai) - T(kI) == 5'500'000);
    for (Index i = kI + 1; i <= kI + 49; ++i) {
        INFO("i = " << i);
        const auto c = r.fn.calls_for(i);
        REQUIRE(c.size() == 1);
        CHECK(static_cast<std::int64_t>(c[0]->tai) - T(i) == 1'200'000);
        CHECK(c[0]->held_ns == 0);
        CHECK(c[0]->units.size() == 31);
        const auto* e = r.outs[kK]->find(i);
        REQUIRE(e != nullptr);
        CHECK(e->kind == Kind::Gap);
        CHECK(static_cast<std::int64_t>(e->tai) - T(i) == 500'000);  // spraga tick'o atidarymo metu
    }
    for (std::uint32_t u = 0; u < kN; ++u) {
        if (u == kK) continue;
        INFO("u = " << u);
        const auto& ev = r.outs[u]->events;
        REQUIRE(ev.size() == kI + 49 - 1000 + 1);
        for (std::size_t n = 0; n < ev.size(); ++n) {
            CHECK(ev[n].kind == Kind::Commit);
            CHECK(ev[n].index == 1000 + n);  // iš eilės
        }
        CHECK(r.out(u).late_skips == 0);
    }
}

TEST_CASE("engine: gyvas užstrigęs", "[engine][loop][isolation]") {
    constexpr std::uint32_t kN = 8, kK = 3;
    constexpr Index kI = 1010;
    Rig r(kN);
    ScriptedInput& k = *r.ins[kK];
    k.arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    std::uint64_t waits_at_i2 = 0, waits_at_i4 = 0;
    struct Sleep {
        std::uint64_t at, target;
    };
    std::vector<Sleep> sleeps;
    sleeps.reserve(1024);
    r.clk.on_sleep = [&](std::uint64_t target) { sleeps.push_back({r.clk.now_tai(), target}); };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI + 2) waits_at_i2 = k.waits;
        if (i == kI + 4) waits_at_i4 = k.waits;
        if (i >= kI + 5) r.stop = true;
    };
    r.build();
    r.run();

    const auto ci = r.fn.calls_for(kI);
    REQUIRE(ci.size() == 1);
    CHECK(static_cast<std::int64_t>(ci[0]->tai) - T(kI) == 5'500'000);
    CHECK(ci[0]->units.size() == 7);
    for (Index i = kI; i <= kI + 2; ++i) {
        INFO("i = " << i);
        const auto* g = r.outs[kK]->find(i);
        REQUIRE(g != nullptr);
        CHECK(g->kind == Kind::Gap);
        CHECK(static_cast<std::int64_t>(g->tai) - T(i) == 15'000'000);  // B riba
        if (i > kI) {
            const auto c = r.fn.calls_for(i);
            REQUIRE(c.size() == 1);
            CHECK(static_cast<std::int64_t>(c[0]->tai) - T(i) == 1'200'000);
        }
    }
    CHECK(r.in(kK).stalls == 1);
    CHECK(r.in(kK).lost == 3);
    CHECK(r.eng->in_state(kK, 0) == InState::Stalled);
    const auto* g3 = r.outs[kK]->find(kI + 3);
    REQUIRE(g3 != nullptr);
    CHECK(static_cast<std::int64_t>(g3->tai) - T(kI + 3) == 500'000);  // Stalled: spraga atidarant
    CHECK(waits_at_i4 == waits_at_i2);  // Stalled įėjimo nebelaukiama
    // Po A paleidimo i + 3 gija miega iki t_open(i + 4)
    bool found = false;
    for (const Sleep& s : sleeps)
        if (static_cast<std::int64_t>(s.at) == T(kI + 3) + 1'200'000) {
            found = true;
            CHECK(static_cast<std::int64_t>(s.target) == T(kI + 4) + 500'000);
        }
    CHECK(found);
    for (std::uint32_t u = 0; u < kN; ++u)
        if (u != kK) CHECK(r.out(u).late_skips == 0);
}

TEST_CASE("engine: užstrigęs grįžta", "[engine][loop][isolation]") {
    constexpr std::uint32_t kN = 4, kK = 1;
    constexpr Index kI = 1010;
    Rig r(kN);
    ScriptedInput& k = *r.ins[kK];
    k.arrival = [](Index j) { return j >= kI && j < kI + 10 ? kNever : std::int64_t{1'200'000}; };
    bool stalled_seen = false, late_at_i11_open = false, late_after_17 = false, late_after_18 = true;
    r.fn.on_begin_tick = [&](Index i, TickContext&) {
        if (i == kI + 5) stalled_seen = r.eng->in_state(kK, 0) == InState::Stalled;
        if (i == kI + 11) late_at_i11_open = r.eng->in_late(kK, 0) && r.eng->in_state(kK, 0) == InState::Live;
    };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI + 17) late_after_17 = r.eng->in_late(kK, 0);
        if (i == kI + 18) late_after_18 = r.eng->in_late(kK, 0);
        if (i >= kI + 20) r.stop = true;
    };
    r.build();
    r.run();
    CHECK(stalled_seen);
    const auto* g10 = r.outs[kK]->find(kI + 10);  // head dar nepadidėjo atidarant — spraga
    REQUIRE(g10 != nullptr);
    CHECK(g10->kind == Kind::Gap);
    CHECK(late_at_i11_open);  // head padidėjo → Live + Late
    for (Index i = kI + 11; i <= kI + 20; ++i) {  // Late, bet paruoštas A paleidimo metu — A batch'e
        INFO("i = " << i);
        const auto* c = r.outs[kK]->find(i);
        REQUIRE(c != nullptr);
        CHECK(c->kind == Kind::Commit);
        CHECK(static_cast<std::int64_t>(c->tai) - T(i) == 1'200'000);
        const auto calls = r.fn.calls_for(i);
        REQUIRE(calls.size() == 1);
        CHECK(calls[0]->phase == 0);
        CHECK(calls[0]->units.size() == 4);
    }
    CHECK(late_after_17);         // 7 laiku iš eilės (i + 11 … i + 17) — dar Late
    CHECK_FALSE(late_after_18);   // 8-as nuėmė Late (late_clear_run = 8)
    CHECK(r.eng->in_state(kK, 0) == InState::Live);
    CHECK(r.in(kK).stalls == 1);
}

TEST_CASE("engine: vėluojantis gyvas 8 ms", "[engine][loop]") {
    constexpr std::uint32_t kN = 4, kK = 2;
    constexpr Index kI = 1010;
    Rig r(kN);
    r.ins[kK]->arrival = [](Index j) { return j == kI ? std::int64_t{8'000'000} : std::int64_t{1'200'000}; };
    r.stop_after(kI + 2);
    r.build();
    r.run();
    const auto c = r.fn.calls_for(kI);
    REQUIRE(c.size() == 2);
    CHECK(c[0]->phase == 0);
    CHECK(static_cast<std::int64_t>(c[0]->tai) - T(kI) == 5'500'000);
    CHECK(c[0]->units.size() == 3);
    CHECK(c[1]->phase == 1);
    CHECK(static_cast<std::int64_t>(c[1]->tai) - T(kI) == 8'000'000);
    REQUIRE(c[1]->units.size() == 1);
    CHECK(c[1]->units[0] == kK);
    CHECK(r.out(kK).late == 1);
    CHECK(r.in(kK).late == 1);
    CHECK(r.in(kK).lost == 0);
    const auto* e = r.outs[kK]->find(kI);
    REQUIRE(e != nullptr);
    CHECK(e->kind == Kind::Commit);
    CHECK(static_cast<std::int64_t>(e->tai) - T(kI) == 8'000'000);
}

TEST_CASE("engine: sustabdymas laukiant", "[engine][loop][lifetime]") {
    constexpr Index kI = 1010;
    Rig r(2);
    ScriptedInput& k = *r.ins[1];
    k.arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    std::uint64_t waits_at_stop = 0;
    k.on_wait = [&](Index j, std::uint64_t) {
        if (j == kI + 1 && !r.stop) {
            r.stop = true;
            waits_at_stop = k.waits;
        }
    };
    r.build();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Stopped);
    CHECK(res.exit_code == 0);
    CHECK(waits_at_stop > 0);
    CHECK(k.waits == waits_at_stop);  // išėjo po to paties laukimo
    CHECK(r.outs[1]->find(kI + 1) == nullptr);  // neišspręstas — ne praradimas, nieko neįrašyta
    CHECK(r.outs[0]->find(kI + 2) == nullptr);
}

TEST_CASE("engine: duration", "[engine][loop][lifetime]") {
    SECTION("10 s prie 50 Hz") {
        EngineConfig c = Rig::test_config();
        c.duration_s = 10;
        Rig r(1, c);
        r.build();
        const RunResult res = r.run();
        CHECK(res.reason == EndReason::Duration);
        const Index ticks = res.last_index - res.first_index + 1;
        CHECK(ticks >= 499);
        CHECK(ticks <= 501);
        CHECK(r.fn.calls.size() == ticks);
        }
    SECTION("0 — tik stop") {
        Rig r(1);
        r.build();
        r.stop_after(1019);
        const RunResult res = r.run();
        CHECK(res.reason == EndReason::Stopped);
        CHECK(res.last_index - res.first_index + 1 == 20);
        }
}

TEST_CASE("engine: šaltinis be turinio → spraga atidarymo metu", "[engine][loop]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(2000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    RecordingOutput out(clk);
    RecordingFunction fn(clk);
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "gen";
    s.out.push_back(&out);
    std::uint32_t id = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, id, why));
    fn.on_begin_tick = [&](Index i, TickContext& ctx) {
        if (i % 2 == 1) ctx.no_content(id);
        if (i >= 2005) stop = true;
    };
    eng.run(fn);
    for (Index i = 2000; i <= 2005; ++i) {
        INFO("i = " << i);
        const auto* e = out.find(i);
        REQUIRE(e != nullptr);
        CHECK(e->kind == (i % 2 == 1 ? Kind::Gap : Kind::Commit));
        CHECK(static_cast<std::int64_t>(e->tai) - T(i) == 500'000);
    }
    CHECK(hub.out(eng.out_id(id, 0)).gap_invalid == 3);
}
