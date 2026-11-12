// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Antros peržiūros (9cb336a) radiniai: Wanted vienas vėlavimas, paleidimo spragos ne praradimas, B laukimas pagal laukiamą
// paruošimo laiką, prisijungiantys po pernešimo, saugikliai (open_wait, dažnis 0), Blocked vieneto gyvų brolių apskaita,
// naujai prijungto bandomasis laikotarpis, mutantai (gyvumas be kvorumo, commit_errors — praradimas).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
using Kind = RecordingOutput::Kind;
std::int64_t rel(const RecordingOutput& o, Index i) {
    const auto* e = o.find(i);
    return e == nullptr ? -1 : static_cast<std::int64_t>(e->tai) - T(i);
}
}  // namespace

TEST_CASE("round2: vienas vėlus Wanted grain'as kainuoja tik tą indeksą", "[engine][round2][wanted]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput video(clk), audio(clk);
    RecordingFunction fn(clk);
    audio.arrival = [](Index j) { return j == 1031 ? std::int64_t{19'000'000} : std::int64_t{1'300'000}; };
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "enc";
    s.in.push_back(InputSpec{&video, {}, 2, Need::Required, InMode::Exact});
    s.in.push_back(InputSpec{&audio, {}, 1, Need::Wanted, InMode::Exact});
    s.stride = 2;
    s.residue = 1;
    std::uint32_t u = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, u, why));
    std::map<Index, bool> with_audio;
    std::map<Index, std::int64_t> at;
    fn.on_process = [&](const Batch& b, TickContext&) {
        for (const UnitTick& ut : b.units) {
            with_audio[ut.index] = ut.in[2].ptr != nullptr;
            at[ut.index] = static_cast<std::int64_t>(clk.now_tai()) - T(ut.index);
        }
    };
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1061) stop = true;
    };
    const RunResult res = eng.run(fn);
    check_run(fn, res, u, nullptr, 2, 1);
    int without = 0;
    for (const auto& [i, a] : with_audio)
        if (i >= 1011 && !a) ++without;
    CHECK(without == 1);
    CHECK_FALSE(with_audio[1031]);
    CHECK(at[1031] == 5'500'000);  // Wanted laukiama tik iki a(i)
    CHECK(with_audio[1033]);
    CHECK(at[1033] == 1'300'000);  // kitas — vėl su garsu, laukiama iki jo
    CHECK(hub.in(eng.in_id(u, 1)).stalls == 0);
    CHECK(hub.in(eng.in_id(u, 1)).lost == 1);
}

TEST_CASE("round2: paleidimo spragos — ne praradimas", "[engine][round2][startup]") {
    Rig r(1);
    const std::uint64_t t0 = r.clk.now_tai();
    r.ins[0]->attachable = [t0](std::uint64_t now) { return now >= t0 + 200'000'000ull; };
    r.stop_after(1040);
    r.build();
    const RunResult res = r.run();
    CHECK(r.out(0).gap_startup == 10);
    CHECK(r.hub.unit(r.eng->unit_id(0)).startup == 10);
    CHECK(r.hub.unit(r.eng->unit_id(0)).gaps == 0);
    CHECK(r.in(0).lost == 0);
    CHECK_FALSE(res.any_loss);
}

TEST_CASE("round2: B nelaukia įprastai lėto Straggler'io (laukiamas laikas > t_r + Wr)", "[engine][round2][b_rule]") {
    constexpr Index kI = 1020;
    Rig r(4);
    r.ins[2]->arrival = [](Index j) { return j == kI ? std::int64_t{8'000'000} : std::int64_t{1'500'000}; };  // X
    r.ins[3]->arrival = [](Index) { return std::int64_t{12'000'000}; };                                        // S — visada 12 ms
    r.stop_after(kI + 14);
    r.build();
    r.run();
    // i: X (Quorum) laikė A iki 5,5; X 8 ms → t_r = 8, laukiama S (12 ≤ 8 + 5) → B ties 12
    CHECK(rel(*r.outs[2], kI) == 12'000'000);
    for (Index i = kI + 1; i <= kI + 14; ++i) {
        INFO("i = " << i);
        CHECK(rel(*r.outs[0], i) == (i <= kI + 8 ? 1'200'000 : 1'500'000));  // nuo i + 9 X vėl Quorum (A laukia 1,5)
        CHECK(rel(*r.outs[2], i) == 1'500'000);  // S laukiamas 12 > 1,5 + 5 — nelaukiama
        CHECK(rel(*r.outs[3], i) == 12'000'000);  // S — C fazėje, kai paruoštas
    }
}

TEST_CASE("round2: X vėluoja kas 9-ą — niekada nelaukia S", "[engine][round2][b_rule]") {
    constexpr Index kI = 1020;
    Rig r(4);
    r.ins[2]->arrival = [](Index j) { return (j >= kI && j % 9 == 0) ? std::int64_t{8'000'000} : std::int64_t{1'500'000}; };
    r.ins[3]->arrival = [](Index) { return std::int64_t{12'000'000}; };
    r.stop_after(kI + 200);
    r.build();
    r.run();
    std::uint64_t hostage = 0, healthy_bad = 0;
    for (Index i = kI + 1; i <= kI + 200; ++i) {
        const std::int64_t x = rel(*r.outs[2], i);
        hostage += (i % 9 == 0) ? x != 12'000'000 && x != 8'000'000 : x != 1'500'000;
        healthy_bad += rel(*r.outs[0], i) > 5'500'000;
    }
    CHECK(hostage == 0);
    CHECK(healthy_bad == 0);
}

TEST_CASE("round2: D = 30 — antras prisijungiantis neprarandamas", "[engine][round2][carry]") {
    constexpr Index kI = 1020;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    Rig r(3, c);
    r.ins[1]->arrival = [](Index j) { return j >= kI ? std::int64_t{26'000'000} : std::int64_t{1'200'000}; };
    r.ins[2]->arrival = [](Index j) {
        return j == kI ? std::int64_t{22'000'000} : j > kI ? std::int64_t{2'500'000} : std::int64_t{1'200'000};
    };
    r.ins[2]->invalid = [](Index j) { return j == kI; };
    r.stop_after(kI + 4);
    r.build();
    r.run();
    const auto* e = r.outs[1]->find(kI + 1);
    REQUIRE(e != nullptr);
    CHECK(e->kind == Kind::Commit);
    CHECK(r.hub.engine().carry_overrun == 0);
    CHECK(r.out(1).gap_invalid == 0);
}

TEST_CASE("round2: legacy — prisijungę po pernešimo neprarandami ir šalia pradėto vieneto", "[engine][round2][legacy]") {
    for (Index kI = 1030; kI < 1040; ++kI) {
        INFO("kI = " << kI);
        EngineConfig c = EngineConfig::legacy(k50, 500'000, 30'000'000);
        c.warmup_s = 0;
        Rig r(3, c);
        r.ins[0]->arrival = [kI](Index j) { return (j >= 1012 && j <= kI) ? kNever : std::int64_t{1'200'000}; };
        r.ins[0]->alive = [kI](std::uint64_t now) {
            const auto n = static_cast<std::int64_t>(now);
            return n < T(1012) || n >= T(kI + 1);
        };
        r.ins[0]->attachable = r.ins[0]->alive;
        r.ins[1]->arrival = [kI](Index j) { return j == kI ? std::int64_t{25'000'000} : std::int64_t{1'200'000}; };
        r.stop_after(kI + 4);
        r.build();
        r.run();
        CHECK(r.out(1).gap_left + r.out(2).gap_left == 0);
        CHECK(r.hub.engine().carry_overrun == 0);
    }
}

TEST_CASE("round2: open_wait grįžta per anksti — tick'ai vyksta, pažeidimas suskaičiuotas", "[engine][round2][failsafe]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(2000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    RecordingOutput out(clk);
    RecordingFunction fn(clk);
    EngineConfig cfg = Rig::test_config();
    std::uint64_t calls = 0;
    cfg.open_wait = [&](Index, std::int64_t, const std::atomic<bool>&) {
        if (++calls >= 10'000) stop = true;  // sargas
        return true;                           // „gerai“, bet nelaukė
    };
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "pace";
    s.out.push_back(&out);
    std::uint32_t id = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, id, why));
    fn.on_begin_tick = [&](Index i, TickContext&) {
        if (i >= 2010) stop = true;
    };
    const RunResult res = eng.run(fn);
    check_run(fn, res, id, &out);
    CHECK(hub.engine().ticks == 11);
    CHECK(calls == 11);
    CHECK(hub.engine().open_wait_early == 11);
}

TEST_CASE("round2: dažnis 0 atmetamas", "[engine][round2][failsafe]") {
    VirtualClock clk(1'768'000'000'000'000'000ull);
    StatsHub hub;
    std::atomic<bool> stop{false};
    RecordingOutput out(clk);
    RecordingFunction fn(clk);
    EngineConfig cfg;
    cfg.cuts = default_cuts({0, 1});
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "x";
    s.out.push_back(&out);
    std::uint32_t id = 0;
    std::string why;
    CHECK_FALSE(eng.add_unit(s, id, why));
    CHECK_FALSE(why.empty());
    const RunResult res = eng.run(fn);
    CHECK(res.reason == EndReason::Fatal);
    CHECK(res.exit_code != 0);
}

TEST_CASE("round2: Blocked vieneto gyvas brolis suskaičiuojamas", "[engine][round2][counters]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput base(clk), layer(clk);
    RecordingOutput o0(clk);
    RecordingFunction fn(clk);
    layer.arrival = [](Index j) { return j >= 1020 ? kNever : std::int64_t{1'200'000}; };  // gyvas, nerašo → Stalled
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "comp";
    s.in.push_back(InputSpec{&base, {}, 1, Need::Required, InMode::Exact});
    s.in.push_back(InputSpec{&layer, {}, 1, Need::Required, InMode::Exact});
    s.out.push_back(&o0);
    std::uint32_t u0 = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, u0, why));
    InCounters at_1030{};
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == 1030) at_1030 = hub.in(eng.in_id(u0, 0));
        if (i >= 1080) stop = true;
    };
    const RunResult res = eng.run(fn);
    check_run(fn, res, u0, &o0);
    REQUIRE(eng.in_state(u0, 1) == InState::Stalled);
    const InCounters& b = hub.in(eng.in_id(u0, 0));
    InCounters d{};
    d.ok = b.ok - at_1030.ok;
    d.late = b.late - at_1030.late;
    d.lost = b.lost - at_1030.lost;
    CHECK(d.ok >= 49);  // 50 tick'ų; paskutinis patikrinamas jau po sustabdymo
    CHECK(d.lost == 0);
    CHECK(input_health(eng.in_state(u0, 0), d) == Health::Healthy);
}

TEST_CASE("round2: naujai prijungtas — Late nuimama po pirmo laiku atėjusio (šaltinis tame pačiame variklyje)",
          "[engine][round2][startup]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput healthy(clk), slow(clk);
    RecordingOutput o0(clk), o1(clk), o2(clk);
    RecordingFunction fn(clk);
    slow.arrival = [](Index) { return std::int64_t{12'000'000}; };
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    std::uint32_t u = 0;
    std::string why;
    UnitSpec src;
    src.name = "src";
    src.out.push_back(&o0);
    REQUIRE(eng.add_unit(src, u, why));
    UnitSpec a;
    a.name = "healthy";
    a.in.push_back(InputSpec{&healthy, {}, 1, Need::Required, InMode::Exact});
    a.out.push_back(&o1);
    REQUIRE(eng.add_unit(a, u, why));
    UnitSpec b;
    b.name = "slow";
    b.in.push_back(InputSpec{&slow, {}, 1, Need::Required, InMode::Exact});
    b.out.push_back(&o2);
    REQUIRE(eng.add_unit(b, u, why));
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1012) stop = true;
    };
    const RunResult res = eng.run(fn);
    check_run(fn, res, 0, &o0);
    check_run(fn, res, 1, &o1);
    check_run(fn, res, 2, &o2);
    // 1000: naujai prijungtas — B, S laukiamas laikas dar nežinomas → laukiama ≤ t_r + Wr = 1,2 + 5
    CHECK(rel(o1, 1000) == 6'200'000);
    for (Index i = 1001; i <= 1012; ++i) CHECK(rel(o1, i) == 1'200'000);
}

TEST_CASE("round2: gyvumas tikrinamas ir be kvorumo (šviežias Straggler'is)", "[engine][round2][mutant]") {
    constexpr Index kI = 1020;
    Rig r(2);
    for (auto& in : r.ins) in->arrival = [](Index j) { return j == kI - 1 ? std::int64_t{8'000'000} : std::int64_t{1'200'000}; };
    ScriptedInput& k = *r.ins[1];
    const auto base = k.arrival;
    k.arrival = [base](Index j) { return j >= kI ? kNever : base(j); };
    k.alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kI); };
    k.attachable = k.alive;
    InState at_end = InState::Waiting;
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == kI) at_end = r.eng->in_state(1, 0);
        if (i >= kI + 2) r.stop = true;
    };
    r.build();
    r.run();
    CHECK(at_end == InState::Detached);  // be kvorumo (abu Late) — vis tiek ties A
    CHECK(rel(*r.outs[1], kI) == 5'500'000);
    CHECK(r.in(1).detaches == 1);
}

TEST_CASE("round2: commit klaida — praradimas", "[engine][round2][mutant]") {
    constexpr Index kI = 1010;
    Rig r(1);
    r.outs[0]->fail_commit = [](Index i) { return i == kI; };
    r.stop_after(kI + 2);
    r.build();
    const RunResult res = r.run();
    CHECK(r.out(0).commit_errors == 1);
    CHECK(res.any_loss);
    CHECK(r.outs[0]->find(kI) == nullptr);
}

TEST_CASE("round2: A17 — laikrodžio įvykio langas 3 s (žingsnis 1-ą min., perpildymas 50-ą — ne laikrodis)",
          "[engine][round2][clock]") {
    // dec perpildymą priskiria laikrodžiui tik per clock_event_recent(3 s): žingsnis −40 ms ties 60 s,
    // klausiama ties 60,5 s (true — laikrodis), 64 s (false) ir 50 min (false — overrun_frames).
    Rig r(1);
    const std::uint64_t t0 = r.clk.now_tai();
    bool armed = false;
    std::map<std::uint64_t, bool> recent;  // sekundė nuo starto → clock_event_recent(3 s)
    r.fn.on_begin_tick = [&](Index, TickContext& ctx) {
        const std::uint64_t mono_s = static_cast<std::uint64_t>(r.clk.now_mono_raw() - r.clk.mono_start()) / 1'000'000'000ull;
        if (!armed && mono_s >= 60) {
            armed = true;
            r.clk.schedule_step(r.clk.now_mono_raw() + 1'000'000, -40'000'000);
        }
        const std::uint64_t ms = static_cast<std::uint64_t>(r.clk.now_mono_raw() - r.clk.mono_start()) / 1'000'000ull;
        for (const std::uint64_t at : {60'500ull, 64'000ull, 3'000'000ull})
            if (ms >= at && recent.count(at) == 0) recent[at] = ctx.clock_event_recent(3'000'000'000ull);
        if (ms >= 3'000'000) r.stop = true;
    };
    r.build();
    r.run();
    (void)t0;
    REQUIRE(r.hub.engine().clock_steps == 1);
    REQUIRE(recent.count(60'500) == 1);
    CHECK(recent[60'500]);
    CHECK_FALSE(recent[64'000]);
    CHECK_FALSE(recent[3'000'000]);
}
