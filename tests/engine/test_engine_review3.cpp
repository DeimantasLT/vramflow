// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Trečios (migracijos) peržiūros radiniai (kodas 4567b1e): išėjimo klaidos — savas rezultatas ir skaitiklis, ilgalaikė
// klaida — fatalu (exit 2); any_loss — tik tikri praradimai (ne laikrodis, ne šaltinio „nėra turinio“); C fazės
// Straggler'ių ryšys ribojamas kaip B fazėje.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
std::map<Outcome, int> outcomes(const RecordingFunction& fn, std::uint32_t unit) {
    std::map<Outcome, int> m;
    for (const auto& e : fn.ends)
        if (e.unit == unit) ++m[e.outcome];
    return m;
}
}  // namespace

TEST_CASE("review3: vienkartinė open klaida — OutputError, skaitiklis, praradimas, ne fatalu", "[engine][review3][output]") {
    constexpr Index kI = 1010;
    Rig r(2);
    r.outs[0]->fail_open = [](Index i) { return i == kI; };
    r.stop_after(kI + 5);
    r.build();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Stopped);
    CHECK(res.exit_code == 0);
    CHECK(res.any_loss);
    CHECK(r.out(0).commit_errors == 1);
    CHECK(r.out(1).commit_errors == 0);
    CHECK(r.outs[0]->find(kI) == nullptr);
    const auto m = outcomes(r.fn, 0);
    CHECK(m.count(Outcome::OutputError) == 1);
    CHECK(m.at(Outcome::OutputError) == 1);
    CHECK(m.count(Outcome::Skipped) == 0);  // Skipped — tik AlreadyCommitted (laikrodis atgal)
    CHECK(r.outs[0]->find(kI + 1) != nullptr);  // kitas indeksas vėl commit'inamas
}

TEST_CASE("review3: vienkartinė commit klaida — OutputError", "[engine][review3][output]") {
    constexpr Index kI = 1010;
    Rig r(1);
    r.outs[0]->fail_commit = [](Index i) { return i == kI; };
    r.stop_after(kI + 3);
    r.build();
    const RunResult res = r.run();
    CHECK(res.exit_code == 0);
    CHECK(res.any_loss);
    const auto m = outcomes(r.fn, 0);
    CHECK(m.count(Outcome::OutputError) == 1);
    CHECK(m.count(Outcome::Skipped) == 0);
}

TEST_CASE("review3: ilgalaikė išėjimo klaida — fatalu, exit 2, pranešimas su išėjimo vardu", "[engine][review3][output]") {
    constexpr Index kI = 1010;
    SECTION("open") {
        Rig r(2);
        r.outs[1]->fail_open = [](Index i) { return i >= kI; };
        r.stop_after(kI + 50);
        r.build();
        const RunResult res = r.run();
        CHECK(res.reason == EndReason::Fatal);
        CHECK(res.exit_code == 2);
        CHECK(res.fatal.find("u1/out0") != std::string::npos);
        CHECK(res.last_index < kI + 10);  // baigiasi iškart po ribos (3 iš eilės), ne po stop_after
        CHECK(r.out(1).commit_errors == 3);
    }
    SECTION("commit") {
        Rig r(1);
        r.outs[0]->fail_commit = [](Index i) { return i >= kI; };
        r.stop_after(kI + 50);
        r.build();
        const RunResult res = r.run();
        CHECK(res.reason == EndReason::Fatal);
        CHECK(res.exit_code == 2);
        CHECK(res.fatal.find("u0/out0") != std::string::npos);
    }
    SECTION("klaidos ne iš eilės — ne fatalu") {
        Rig r(1);
        r.outs[0]->fail_open = [](Index i) { return i >= kI && i % 2 == 0; };
        r.stop_after(kI + 20);
        r.build();
        const RunResult res = r.run();
        CHECK(res.reason == EndReason::Stopped);
        CHECK(r.out(0).commit_errors >= 10);
    }
}

TEST_CASE("review3: AlreadyCommitted — Skipped be klaidos skaitiklio ir be praradimo", "[engine][review3][output]") {
    constexpr Index kI = 1010;
    Rig r(1);
    r.outs[0]->already_at = [](Index i) { return i == kI; };
    r.stop_after(kI + 3);
    r.build();
    const RunResult res = r.run();
    CHECK(res.exit_code == 0);
    CHECK_FALSE(res.any_loss);
    CHECK(r.out(0).commit_errors == 0);
    CHECK(r.out(0).already_committed == 1);
    const auto m = outcomes(r.fn, 0);
    CHECK(m.count(Outcome::Skipped) == 1);
}

TEST_CASE("review3: any_loss — laikrodžio spragos ir šaltinio „nėra turinio“ nėra praradimas", "[engine][review3][loss]") {
    SECTION("šaltinis be turinio") {
        VirtualClock clk(static_cast<std::uint64_t>(T(2000)));
        StatsHub hub;
        std::atomic<bool> stop{false};
        RecordingOutput out(clk);
        RecordingFunction fn(clk);
        fn.on_begin_tick = [&](Index i, TickContext& ctx) {
            if (i % 3 == 0) ctx.no_content(0);
        };
        fn.on_end_tick = [&](Index i, TickContext&) {
            if (i >= 2020) stop = true;
        };
        TickEngine eng(Rig::test_config(), clk, hub, stop);
        UnitSpec s;
        s.name = "src";
        s.out.push_back(&out);
        std::uint32_t id = 0;
        std::string why;
        REQUIRE(eng.add_unit(s, id, why));
        const RunResult res = eng.run(fn);
        CHECK(hub.unit(eng.unit_id(0)).gaps > 0);
        CHECK_FALSE(res.any_loss);
    }
    SECTION("laikrodžio žingsnis uždaro atvirą tick'ą") {
        Rig r(1);
        r.ins[0]->arrival = [](Index j) { return j == 1010 ? std::int64_t{12'000'000} : std::int64_t{1'200'000}; };
        r.clk.schedule_step(r.clk.now_mono_raw() + (1010 - 1000) * 20'000'000 + 6'000'000, -20'000'000);
        r.stop_after(1030);
        r.build();
        const RunResult res = r.run();
        CHECK(r.hub.engine().step_gaps >= 1);
        CHECK(r.hub.unit(r.eng->unit_id(0)).late_skips == 0);
        CHECK_FALSE(res.any_loss);
    }
    SECTION("tikra spraga — praradimas") {
        Rig r(1);
        r.ins[0]->arrival = [](Index j) { return j == 1010 ? kNever : std::int64_t{1'200'000}; };
        r.stop_after(1020);
        r.build();
        const RunResult res = r.run();
        CHECK(res.any_loss);
    }
}
