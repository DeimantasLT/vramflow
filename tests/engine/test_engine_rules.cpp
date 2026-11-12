// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Variklio taisyklės po nepriklausomos peržiūros (2026-11-12): legacy be praradimų, Late histerezė, paleidimų tvarka
// su pernešimu, open_wait, skaitikliai, kiekvieno indekso apskaita, B taisyklė, daugiaįėjimiai vienetai, Wanted /
// Optional, grains > 1, stride, WaitAll, freeze per variklį, ašies lygybė, pabudimo vėlavimo riba.
// 50 Hz: P = 20 ms, ro = 0,5 ms, Wr = 5 ms, D = 15 ms, G = 3 ms; sveikas įėjimas T + 1,2 ms.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <vector>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
using Kind = RecordingOutput::Kind;
std::int64_t rel(const RecordingOutput::Event* e, Index i) { return static_cast<std::int64_t>(e->tai) - T(i); }
std::int64_t relc(const RecordingFunction::Call* c, Index i) { return static_cast<std::int64_t>(c->tai) - T(i); }
EngineConfig legacy_cfg() {
    EngineConfig c = EngineConfig::legacy(k50, 500'000, 30'000'000);
    c.warmup_s = 0;
    return c;
}
}  // namespace

// --- 1. Legacy be praradimų --------------------------------------------------------------------------------------

TEST_CASE("legacy: vienas 21 ms sutrikimas — kitas indeksas apdorojamas, ne išmetamas", "[engine][legacy]") {
    constexpr Index kI = 1010;
    Rig r(3, legacy_cfg());
    r.ins[1]->arrival = [](Index j) { return j == kI ? std::int64_t{21'000'000} : std::int64_t{1'200'000}; };
    std::map<Index, std::int64_t> end_at;
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        end_at[i] = static_cast<std::int64_t>(r.clk.now_tai()) - T(i);
        if (i >= kI + 3) r.stop = true;
    };
    r.build();
    r.run();
    for (std::uint32_t u = 0; u < 3; ++u) {
        INFO("u = " << u);
        const auto* a = r.outs[u]->find(kI);
        REQUIRE(a != nullptr);
        CHECK(a->kind == Kind::Commit);
        CHECK(rel(a, kI) == 21'000'000);  // viena riba: visi kartu, kai paruoštas paskutinis (apklausa 0,5 ms)
        const auto* b = r.outs[u]->find(kI + 1);
        REQUIRE(b != nullptr);
        CHECK(b->kind == Kind::Commit);
        CHECK(rel(b, kI + 1) == 1'500'000);  // prisijungė ties T(i+1) + 1,0, grain'ai 1,2 → apklausa 1,5
        CHECK(r.out(u).gap_left == 0);
        CHECK(r.out(u).late_skips == 0);
    }
    CHECK(r.hub.engine().carry_overrun == 0);
    // tick'as i + 1 baigiasi tik po prisijungusių narių
    CHECK(end_at[kI + 1] == 1'500'000);
}

TEST_CASE("legacy: pakibęs įėjimas — vienas terminas visiems, be praleidimų", "[engine][legacy]") {
    constexpr Index kI = 1010;
    Rig r(3, legacy_cfg());
    r.ins[1]->arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    r.stop_after(kI + 6);
    r.build();
    r.run();
    for (Index i = kI; i <= kI + 6; ++i) {
        INFO("i = " << i);
        const auto* e = r.outs[0]->find(i);
        REQUIRE(e != nullptr);
        CHECK(e->kind == Kind::Commit);
        CHECK(rel(e, i) == 30'000'000);  // šiandieninė legacy elgsena: visi laukia termino
    }
    CHECK(r.out(0).late_skips == 0);
    CHECK(r.out(1).gap_left == 7);  // leave: priežastis suskaičiuota
    CHECK(r.eng->in_state(1, 0) == InState::Live);  // M = 0 — be pertraukiklio
}

TEST_CASE("legacy: negyvas rašytojas atjungiamas ties riba (writer_alive)", "[engine][legacy]") {
    constexpr Index kI = 1010;
    Rig r(2, legacy_cfg());
    ScriptedInput& k = *r.ins[1];
    k.arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    k.alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kI); };
    k.attachable = k.alive;
    r.stop_after(kI + 10);
    r.build();
    r.run();
    CHECK(r.eng->in_state(1, 0) == InState::Detached);
    CHECK(k.alive_calls >= 1);
    CHECK(rel(r.outs[0]->find(kI), kI) == 30'000'000);  // pirmas — iki ribos
    CHECK(rel(r.outs[0]->find(kI + 1), kI + 1) == 10'000'000);  // prisijungė po i ribos (T(i+1) + 10), grain'as jau yra
    for (Index i = kI + 2; i <= kI + 10; ++i) CHECK(rel(r.outs[0]->find(i), i) == 1'500'000);  // toliau — be laukimo
}

TEST_CASE("engine: D = 30 ms, visi įėjimai 26 ms vienam indeksui — i + 1 neišmetamas", "[engine][carry]") {
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    Rig r(2, c);
    // Rašytojas commit'ina eilės tvarka (MXL: head monotoniškas): kI ties T + 26, todėl kI + 1 — ne anksčiau kaip iškart po jo
    // (T(kI + 1) + 6). Anksčiau tvarkaraštis „kI + 1 ties T(kI + 1) + 1,2“ (prieš kI) netikroviškas: pakaitalas grain'us ≤ head
    // laiko paruoštais, o C grupės 1 ms gabalų laukimas (review3) tai pamatytų — lūkesčiai nepakito.
    for (auto& in : r.ins)
        in->arrival = [](Index j) {
            return j == kI ? std::int64_t{26'000'000} : j == kI + 1 ? std::int64_t{6'000'000} : std::int64_t{1'200'000};
        };
    std::map<Index, std::int64_t> end_at;
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        end_at[i] = static_cast<std::int64_t>(r.clk.now_tai()) - T(i);
        if (i >= kI + 3) r.stop = true;
    };
    r.build();
    r.run();
    for (std::uint32_t u = 0; u < 2; ++u) {
        const auto* a = r.outs[u]->find(kI);
        REQUIRE(a != nullptr);
        CHECK(rel(a, kI) == 26'000'000);
        const auto* b = r.outs[u]->find(kI + 1);
        REQUIRE(b != nullptr);
        CHECK(b->kind == Kind::Commit);
        CHECK(rel(b, kI + 1) == 6'000'000);  // prisijungė ties T(i+1) + 6 (A riba praėjo) → B iškart
    }
    CHECK(end_at[kI + 1] == 6'000'000);
    CHECK(r.hub.engine().carry_overrun == 0);
}

// --- 2. Late histerezė ---------------------------------------------------------------------------------------------

TEST_CASE("engine: 1,5 ms įėjimas po vieno vėlavimo grįžta į kvorumą po 8 tick'ų", "[engine][late]") {
    constexpr std::uint32_t kN = 4, kK = 2;
    constexpr Index kI = 1010;
    Rig r(kN);
    r.ins[kK]->arrival = [](Index j) { return j == kI ? std::int64_t{8'000'000} : std::int64_t{1'500'000}; };
    r.stop_after(kI + 40);
    r.build();
    r.run();
    for (Index i = kI + 1; i <= kI + 8; ++i) {  // Late: A be jo (1,2), jis — B (1,5)
        INFO("i = " << i);
        const auto c = r.fn.calls_for(i);
        REQUIRE(c.size() == 2);
        CHECK(relc(c[0], i) == 1'200'000);
        CHECK(c[1]->phase == 1);
        CHECK(relc(c[1], i) == 1'500'000);
    }
    for (Index i = kI + 9; i <= kI + 40; ++i) {  // 8 laiku iš eilės → vėl Quorum: vienas kvietimas
        INFO("i = " << i);
        const auto c = r.fn.calls_for(i);
        REQUIRE(c.size() == 1);
        CHECK(relc(c[0], i) == 1'500'000);
        CHECK(c[0]->units.size() == kN);
    }
    CHECK(r.out(kK).late == 9);  // i … i + 8, toliau nebeauga
    CHECK(r.in(kK).late == 1);   // tik 8 ms grain'as atėjo po a(i)
    CHECK(r.in(kK).ok == 1000 - 1000 + (kI - 1000) + 40);
    CHECK_FALSE(r.eng->in_late(kK, 0));
}

TEST_CASE("engine: plazdantis įėjimas sveikiems kainuoja 0 po pirmo praleidimo", "[engine][late]") {
    constexpr Index kI = 1010;
    Rig r(3);
    r.ins[2]->arrival = [](Index j) { return (j >= kI && j % 2 == 0) ? std::int64_t{8'000'000} : std::int64_t{1'200'000}; };
    r.stop_after(kI + 99);
    r.build();
    r.run();
    const auto c0 = r.fn.calls_for(kI);
    REQUIRE(!c0.empty());
    CHECK(c0[0]->held_ns == 4'300'000);  // pirmas — vienkartinis Wr
    for (Index i = kI + 1; i <= kI + 99; ++i) {
        INFO("i = " << i);
        const auto c = r.fn.calls_for(i);
        REQUIRE(!c.empty());
        CHECK(c[0]->phase == 0);
        CHECK(relc(c[0], i) == 1'200'000);
        CHECK(c[0]->held_ns == 0);
        CHECK(rel(r.outs[0]->find(i), i) == 1'200'000);
    }
    CHECK(r.eng->in_late(2, 0));  // niekada 8 iš eilės
}

// --- 3. Paleidimų tvarka su pernešimu ----------------------------------------------------------------------------

TEST_CASE("engine: sveikas A(i+1) — prieš pernešto C(i) apdorojimą", "[engine][carry]") {
    constexpr std::uint32_t kN = 3, kK = 1;
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    Rig r(kN, c);
    // k@j ateina T(j) + 21,2 = T(j+1) + 1,2 — tą pačią akimirką kaip sveiki j + 1
    r.ins[kK]->arrival = [](Index j) { return j >= kI ? std::int64_t{21'200'000} : std::int64_t{1'200'000}; };
    r.fn.on_process = [&](const Batch&, TickContext&) { r.clk.advance(4'000'000); };  // GPU 4 ms
    r.stop_after(kI + 10);
    r.build();
    r.run();
    for (Index i = kI + 1; i <= kI + 9; ++i) {
        INFO("i = " << i << (i % 2 ? " (nelyginis)" : " (lyginis)"));
        const auto calls = r.fn.calls_for(i);
        REQUIRE(!calls.empty());
        CHECK(calls[0]->phase == 0);
        CHECK(relc(calls[0], i) == 1'200'000);
        CHECK(rel(r.outs[0]->find(i), i) == 5'200'000);  // 1,2 + 4 ms apdorojimas
        const auto* k = r.outs[kK]->find(i - 1);           // k@(i−1) — po sveiko A(i)
        REQUIRE(k != nullptr);
        CHECK(k->kind == Kind::Commit);
        CHECK(rel(k, i - 1) == 29'200'000);  // T(i) + 5,2 + 4
    }
}

TEST_CASE("engine: D = 22 ms, pakibęs pernešimas nėra ašis", "[engine][carry][pivot]") {
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 22'000'000;  // C(i) = T(i+1) + 2 < A(i+1)
    Rig r(3, c);
    r.ins[1]->arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    r.stop_after(kI + 3);
    r.build();
    r.run();
    for (Index i = kI + 1; i <= kI + 3; ++i) {
        INFO("i = " << i);
        const auto calls = r.fn.calls_for(i);
        REQUIRE(!calls.empty());
        CHECK(relc(calls[0], i) == 1'200'000);
    }
}

// --- 4. open_wait ---------------------------------------------------------------------------------------------------

TEST_CASE("engine: open_wait — vienintelis būdas laukti tick'o atidarymo", "[engine][open_wait]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(2000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    RecordingOutput out(clk);
    RecordingFunction fn(clk);
    EngineConfig cfg = Rig::test_config();
    std::vector<Index> hooked;
    hooked.reserve(64);
    cfg.open_wait = [&](Index i, std::int64_t off, const std::atomic<bool>&) {
        hooked.push_back(i);
        CHECK(off == 500'000);
        const auto t = static_cast<std::uint64_t>(T(i) + off);
        if (clk.now_tai() < t) clk.advance(t - clk.now_tai());
        return true;
    };
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "pace";
    s.out.push_back(&out);
    std::uint32_t id = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, id, why));
    fn.on_begin_tick = [&](Index i, TickContext&) {
        if (i >= 2020) stop = true;
    };
    const RunResult res = eng.run(fn);
    CHECK(hooked.size() == 21);
    for (std::size_t n = 0; n < hooked.size(); ++n) CHECK(hooked[n] == 2000 + n);
    CHECK(clk.sleeps == 0);
    check_run(fn, res, id, &out);
}

// --- 5. Skaitikliai ---------------------------------------------------------------------------------------------

TEST_CASE("engine: daugiaįėjimis vienetas — laiku atėjęs bazės įėjimas skaičiuojamas ok", "[engine][multi]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput base(clk), layer(clk), other(clk);
    RecordingOutput o0(clk), o1(clk);
    RecordingFunction fn(clk);
    layer.arrival = [](Index j) { return j >= 1010 ? std::int64_t{8'000'000} : std::int64_t{1'200'000}; };
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "comp";
    s.in.push_back(InputSpec{&base, {}, 1, Need::Required, InMode::Exact});
    s.in.push_back(InputSpec{&layer, {}, 1, Need::Required, InMode::Exact});
    s.out.push_back(&o0);
    std::uint32_t u0 = 0, u1 = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, u0, why));
    UnitSpec s2;
    s2.name = "logo";
    s2.in.push_back(InputSpec{&other, {}, 1, Need::Required, InMode::Exact});
    s2.out.push_back(&o1);
    REQUIRE(eng.add_unit(s2, u1, why));
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1052) stop = true;
    };
    const RunResult res = eng.run(fn);
    const InCounters& b = hub.in(eng.in_id(u0, 0));
    const InCounters& l = hub.in(eng.in_id(u0, 1));
    CHECK(b.ok == 53);
    CHECK(b.late == 0);
    CHECK(l.ok == 10);
    CHECK(l.late == 43);
    CHECK(input_health(InState::Live, b) == Health::Healthy);
    CHECK(rel(o1.find(1010), 1010) == 5'500'000);  // vienkartinis Wr: sluoksnis tada dar buvo Quorum
    for (Index i = 1011; i <= 1052; ++i) CHECK(rel(o1.find(i), i) == 1'200'000);
    for (Index i = 1010; i <= 1052; ++i) CHECK(rel(o0.find(i), i) == 8'000'000);
    check_run(fn, res, u0, &o0);
    check_run(fn, res, u1, &o1);
}

TEST_CASE("engine: K = 2…4 įėjimai vienam išėjimui", "[engine][multi]") {
    for (std::uint32_t kK = 2; kK <= 4; ++kK) {
        INFO("K = " << kK);
        VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
        StatsHub hub;
        std::atomic<bool> stop{false};
        std::vector<std::unique_ptr<ScriptedInput>> ins;
        for (std::uint32_t k = 0; k < kK; ++k) {
            ins.push_back(std::make_unique<ScriptedInput>(clk));
            const auto lat = std::int64_t{1'000'000} + 100'000 * k;  // 1,0 … 1,3 ms
            ins.back()->arrival = [lat](Index j) { return j == 1015 && lat == 1'100'000 ? kNever : lat; };
        }
        RecordingOutput o(clk);
        RecordingFunction fn(clk);
        EngineConfig cfg = Rig::test_config();
        TickEngine eng(cfg, clk, hub, stop);
        UnitSpec s;
        s.name = "comp";
        for (auto& in : ins) s.in.push_back(InputSpec{in.get(), {}, 1, Need::Required, InMode::Exact});
        s.out.push_back(&o);
        std::uint32_t u = 0;
        std::string why;
        REQUIRE(eng.add_unit(s, u, why));
        std::vector<std::size_t> nin;
        fn.on_process = [&](const Batch& b, TickContext&) {
            for (const UnitTick& ut : b.units) nin.push_back(ut.in.size());
        };
        fn.on_end_tick = [&](Index i, TickContext&) {
            if (i >= 1020) stop = true;
        };
        const RunResult res = eng.run(fn);
        for (std::size_t n : nin) CHECK(n == kK);
        const std::int64_t last = std::int64_t{1'000'000} + 100'000 * (kK - 1);
        CHECK(rel(o.find(1014), 1014) == last);  // paskutinis įėjimas nulemia paleidimą
        const auto* g = o.find(1015);             // įėjimas #1 neatėjo: spraga ties B (gyvas → Late)
        REQUIRE(g != nullptr);
        CHECK(g->kind == Kind::Gap);
        CHECK(rel(g, 1015) == 15'000'000);
        CHECK(hub.in(eng.in_id(u, 1)).lost == 1);
        for (std::uint32_t k = 0; k < kK; ++k)
            if (k != 1) CHECK(hub.in(eng.in_id(u, k)).lost == 0);
        check_run(fn, res, u, &o);
    }
}

TEST_CASE("engine: held_ns nepriklauso nuo to, ar miręs įėjimas — ašis", "[engine][counters]") {
    constexpr std::uint32_t kN = 32;
    constexpr Index kI = 1010;
    for (std::uint32_t kK : {6u, 31u}) {  // 31 — didžiausias numeris: lygybėje ašis
        INFO("k = " << kK);
        Rig r(kN);
        ScriptedInput& k = *r.ins[kK];
        k.arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
        k.alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kI); };
        k.attachable = k.alive;
        r.build();
        r.stop_after(kI + 2);
        r.run();
        const auto ci = r.fn.calls_for(kI);
        REQUIRE(ci.size() == 1);
        CHECK(relc(ci[0], kI) == 5'500'000);
        CHECK(ci[0]->held_ns == 4'300'000);
        CHECK(r.hub.engine().held_ns_max == 4'300'000);
    }
}

TEST_CASE("engine: guard_overrun — B apdorojimas baigėsi po t_open(i+1)", "[engine][counters]") {
    constexpr Index kI = 1010;
    Rig r(2);
    r.ins[1]->arrival = [](Index j) { return j == kI ? std::int64_t{14'900'000} : std::int64_t{1'200'000}; };
    r.fn.on_process = [&](const Batch& b, TickContext&) {
        if (b.phase == 1) r.clk.advance(8'000'000);  // 14,9 → 22,9 ms > 20,5
    };
    r.stop_after(kI + 3);
    r.build();
    r.run();
    const auto c = r.fn.calls_for(kI);
    REQUIRE(c.size() == 2);
    CHECK(relc(c[1], kI) == 14'900'000);
    CHECK(r.hub.engine().guard_overrun == 1);
    CHECK(r.out(0).late_skips == 0);  // kitas tick'as atidarytas pavėlavus 2,4 ms — ne praleistas
}

// --- 6. Kiekvieno indekso apskaita -------------------------------------------------------------------------------

TEST_CASE("engine: pernešimas + laikrodžio žingsnis — abu indeksai su spraga", "[engine][accounting]") {
    constexpr std::uint32_t kK = 1;
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    Rig r(3, c);
    r.ins[kK]->arrival = [](Index j) { return j >= kI ? std::int64_t{26'000'000} : std::int64_t{1'200'000}; };
    r.fn.on_begin_tick = [&](Index i, TickContext&) {
        if (i == kI + 1) r.clk.schedule_step(r.clk.now_mono_raw() + 3'000'000, 6'000'000);  // k neša i, prisijungs prie i+1
    };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= kI + 4) r.stop = true;
    };
    r.build();
    r.run();
    const auto* a = r.outs[kK]->find(kI);
    const auto* b = r.outs[kK]->find(kI + 1);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CHECK(a->kind == Kind::Gap);
    CHECK(b->kind == Kind::Gap);
    CHECK(r.hub.engine().step_gaps == 2);
}

TEST_CASE("engine: sink'o praleidimai suskaičiuoti, any_loss", "[engine][accounting]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput in(clk);
    RecordingFunction fn(clk);
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "enc";
    s.in.push_back(InputSpec{&in, {}, 1, Need::Required, InMode::Exact});
    std::uint32_t u = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, u, why));
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == 1009) clk.oversleep_once(45'000'000);
        if (i >= 1015) stop = true;
    };
    const RunResult res = eng.run(fn);
    CHECK(hub.unit(eng.unit_id(u)).late_skips == 2);
    CHECK(res.any_loss);
    std::uint64_t processed = 0;
    for (const auto& e : fn.ends) processed += e.outcome == Outcome::Ready;
    CHECK(processed + hub.unit(eng.unit_id(u)).late_skips == res.last_index - res.first_index + 1);
    check_run(fn, res, u, nullptr);
}

// --- 7. Smulkūs ------------------------------------------------------------------------------------------------------

TEST_CASE("engine: WaitAll 60 s su 100 ppm dreifu — ne laikrodžio žingsnis", "[engine][clock][start]") {
    EngineConfig c = Rig::test_config();
    c.start = StartMode::WaitAll;
    Rig r(1, c);
    const std::uint64_t t0 = r.clk.now_tai();
    bool drifting = true;
    r.ins[0]->attachable = [&, t0](std::uint64_t now) {
        const bool ok = now >= t0 + 60'000'000'000ull;
        if (ok) drifting = false;
        return ok;
    };
    r.clk.on_sleep = [&](std::uint64_t) {
        if (drifting) r.clk.step(10'000);  // +10 µs kas 100 ms miego = 100 ppm
    };
    Index first = kNoIndex;
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (first == kNoIndex) first = i;
        if (i >= first + 5) r.stop = true;
    };
    r.build();
    const RunResult res = r.run();
    CHECK(r.hub.engine().clock_steps == 0);
    CHECK(r.fn.steps.empty());
    CHECK(static_cast<std::int64_t>(T(res.first_index)) >= static_cast<std::int64_t>(t0) + 60'000'000'000);
}

TEST_CASE("engine: duration matuojama monotoniniu laiku", "[engine][lifetime]") {
    EngineConfig c = Rig::test_config();
    c.duration_s = 2;
    Rig r(1, c);
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == 1010) r.clk.schedule_step(r.clk.now_mono_raw() + 10'000'000, 1'300'000'000);
    };
    r.build();
    const std::int64_t m0 = r.clk.now_mono_raw();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Duration);
    const std::int64_t ran = r.clk.now_mono_raw() - m0;
    CHECK(ran >= 1'980'000'000);
    CHECK(ran <= 2'000'000'000);
}

TEST_CASE("engine: Stalled įėjimas tikrinamas probe — atkurtas flow atjungiamas ir prijungiamas", "[engine][stall]") {
    constexpr Index kI = 1010;
    Rig r(2);
    ScriptedInput& k = *r.ins[1];
    k.arrival = [](Index j) { return j >= kI && j < kI + 20 ? kNever : std::int64_t{1'200'000}; };
    k.gone = [](std::uint64_t now) {
        const auto n = static_cast<std::int64_t>(now);
        return n >= T(kI + 10) && n < T(kI + 11);
    };
    InState at10 = InState::Waiting, at11 = InState::Waiting;
    r.fn.on_begin_tick = [&](Index i, TickContext&) {
        if (i == kI + 10) at10 = r.eng->in_state(1, 0);
        if (i == kI + 11) at11 = r.eng->in_state(1, 0);
    };
    r.stop_after(kI + 24);
    r.build();
    r.run();
    CHECK(at10 == InState::Detached);  // Stalled → probe → Gone
    CHECK(at11 == InState::Live);      // prijungtas iš naujo
    CHECK(r.in(1).detaches == 1);
    CHECK(r.in(1).reattached == 1);
    CHECK(rel(r.outs[1]->find(kI + 22), kI + 22) == 1'200'000);
}

// --- 8. B taisyklė --------------------------------------------------------------------------------------------------

TEST_CASE("engine: pakibęs Straggler laiko paruoštą ne ilgiau kaip vieną tick'ą", "[engine][b_rule]") {
    constexpr Index kI = 1010;
    Rig r(3);
    r.ins[1]->arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };                  // pakibęs, gyvas
    r.ins[2]->arrival = [](Index j) { return j >= kI ? std::int64_t{8'000'000} : std::int64_t{1'200'000}; };  // 8 ms
    r.stop_after(kI + 5);
    r.build();
    r.run();
    CHECK(rel(r.outs[0]->find(kI), kI) == 5'500'000);
    CHECK(rel(r.outs[2]->find(kI), kI) == 13'000'000);  // abu švieži: paruoštas (8 ms) laikomas ≤ Wr → 8 + 5
    for (Index i = kI + 1; i <= kI + 5; ++i) {
        INFO("i = " << i);
        CHECK(rel(r.outs[0]->find(i), i) == 1'200'000);
        const auto* e = r.outs[2]->find(i);
        CHECK(e->kind == Kind::Commit);
        CHECK(rel(e, i) == 8'000'000);  // pakibęs (miss_run ≥ 1) nebelaukiamas
    }
    for (Index i = kI; i <= kI + 2; ++i) CHECK(rel(r.outs[1]->find(i), i) == 15'000'000);
    CHECK(rel(r.outs[1]->find(kI + 3), kI + 3) == 500'000);  // Stalled
    CHECK(r.in(1).stalls == 1);
}

// --- 9. Trūkę scenarijai ---------------------------------------------------------------------------------------------

TEST_CASE("engine: enc forma — video grains 2, stride 2, garsas Wanted pakimba", "[engine][wanted]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput video(clk), audio(clk);
    RecordingFunction fn(clk);
    audio.arrival = [](Index j) { return j >= 1031 ? kNever : std::int64_t{1'300'000}; };
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
    struct Frame {
        std::uint8_t phase;
        std::int64_t at;
        bool v0, v1, a;
    };
    std::map<Index, Frame> frames;
    fn.on_process = [&](const Batch& b, TickContext&) {
        for (const UnitTick& ut : b.units) {
            REQUIRE(ut.in.size() == 3);
            frames[ut.index] = {b.phase, static_cast<std::int64_t>(clk.now_tai()) - T(ut.index), ut.in[0].ptr != nullptr,
                                ut.in[1].ptr != nullptr, ut.in[2].ptr != nullptr};
        }
    };
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1041) stop = true;
    };
    const RunResult res = eng.run(fn);
    check_run(fn, res, u, nullptr, 2, 1);
    for (const auto& [i, f] : frames) CHECK(i % 2 == 1);  // tik apatinis laukas
    REQUIRE(frames.count(1029) == 1);
    CHECK(frames[1029].at == 1'300'000);  // garsas — paskutinis
    CHECK(frames[1029].a);
    // Wanted laukiama kaip Required iki vieneto paskutinės ribos (test_config: b = T + 15, C nėra), tada apdorojama be jo
    // (anksčiau — A fazė ties a(i) = T + 5,5): 1031 — B (šviežias), 1033 / 1035 — C (pakibęs, miss_run ≥ 1); trys
    // praleidimai iš eilės → Stalled
    for (Index i = 1031; i <= 1035; i += 2) {
        INFO("i = " << i);
        REQUIRE(frames.count(i) == 1);
        CHECK(frames[i].phase == (i == 1031 ? 1 : 2));
        CHECK(frames[i].at == 15'000'000);
        CHECK_FALSE(frames[i].a);
    }
    for (Index i = 1037; i <= 1041; i += 2) {
        INFO("i = " << i);
        REQUIRE(frames.count(i) == 1);
        CHECK(frames[i].phase == 0);
        CHECK(frames[i].at == 1'200'000);  // Stalled Wanted nebelaukiama
        CHECK(frames[i].v0);
        CHECK(frames[i].v1);
        CHECK_FALSE(frames[i].a);
    }
    CHECK(hub.in(eng.in_id(u, 1)).stalls == 1);
    CHECK(hub.in(eng.in_id(u, 0)).lost == 0);
}

TEST_CASE("engine: Optional įėjimo nelaukiama", "[engine][wanted]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput main(clk), opt(clk);
    RecordingOutput o(clk);
    RecordingFunction fn(clk);
    opt.arrival = [](Index j) { return j % 3 == 0 ? std::int64_t{3'000'000} : std::int64_t{1'000'000}; };
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "mv";
    s.in.push_back(InputSpec{&main, {}, 1, Need::Required, InMode::Exact});
    s.in.push_back(InputSpec{&opt, {}, 1, Need::Optional, InMode::Exact});
    s.out.push_back(&o);
    std::uint32_t u = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, u, why));
    std::map<Index, bool> has_opt;
    fn.on_process = [&](const Batch& b, TickContext&) {
        for (const UnitTick& ut : b.units) has_opt[ut.index] = ut.in[1].ptr != nullptr;
    };
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1030) stop = true;
    };
    const RunResult res = eng.run(fn);
    for (Index i = 1010; i <= 1030; ++i) {
        INFO("i = " << i);
        CHECK(rel(o.find(i), i) == 1'200'000);
        CHECK(has_opt[i] == (i % 3 != 0));
    }
    check_run(fn, res, u, &o);
}

TEST_CASE("engine: stride 2 išėjimas — tik lyginiai indeksai, apskaita", "[engine][stride]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput in(clk);
    RecordingOutput o(clk);
    RecordingFunction fn(clk);
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "half";
    s.in.push_back(InputSpec{&in, {}, 1, Need::Required, InMode::Exact});
    s.out.push_back(&o);
    s.stride = 2;
    std::uint32_t u = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, u, why));
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == 1011) clk.oversleep_once(45'000'000);  // praleidžiami 1012, 1013 → vienetui tik 1012
        if (i >= 1030) stop = true;
    };
    const RunResult res = eng.run(fn);
    for (const auto& e : o.events) CHECK(e.index % 2 == 0);
    CHECK(o.find(1012) == nullptr);
    CHECK(hub.out(eng.out_id(u, 0)).late_skips == 1);
    check_run(fn, res, u, &o, 2, 0);
}

TEST_CASE("engine: interlaced freeze per variklį — kartojimas ir grįžimas kadro riba", "[engine][freeze]") {
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput in(clk);
    RecordingOutput out(clk, true);
    RecordingFunction fn(clk);
    in.arrival = [](Index j) { return j == 1010 ? kNever : std::int64_t{1'200'000}; };  // trūksta viršutinio lauko
    EngineConfig cfg = Rig::test_config();
    cfg.on_loss = GapPolicy::Freeze;
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "logo";
    s.in.push_back(InputSpec{&in, {}, 1, Need::Required, InMode::Exact});
    s.out.push_back(&out);
    std::uint32_t u = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, u, why));
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1016) stop = true;
    };
    const RunResult res = eng.run(fn);
    const auto* g0 = out.find(1010);
    REQUIRE(g0 != nullptr);
    CHECK(g0->kind == Kind::Gap);
    CHECK(g0->action == GapAction::Repeat);
    CHECK(g0->from == 1008);
    CHECK(rel(g0, 1010) == 15'000'000);
    const auto* g1 = out.find(1011);  // turinys yra, bet nelyginis — kadras baigiamas kartojimu (K-T5)
    REQUIRE(g1 != nullptr);
    CHECK(g1->kind == Kind::Gap);
    CHECK(g1->action == GapAction::Repeat);
    CHECK(g1->from == 1009);
    const auto* c2 = out.find(1012);
    REQUIRE(c2 != nullptr);
    CHECK(c2->kind == Kind::Commit);
    CHECK(hub.out(eng.out_id(u, 0)).gap_frozen == 2);
    CHECK(out.flushes >= 2);
    check_run(fn, res, u, &out);
}

TEST_CASE("engine: ašies lygybė — pakibęs Late įėjimas su didžiausiu numeriu nelaiko A", "[engine][pivot]") {
    constexpr std::uint32_t kN = 8;
    constexpr Index kI = 1010;
    for (int variant = 0; variant < 2; ++variant) {
        INFO("variantas " << variant);
        const std::uint32_t kK = variant == 0 ? kN - 1 : 3;  // 0 — didžiausias numeris; 1 — didžiausias ewma
        Rig r(kN);
        ScriptedInput& k = *r.ins[kK];
        if (variant == 0) k.arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
        else k.arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{3'000'000}; };
        r.stop_after(kI + 2);
        r.build();
        r.run();
        for (Index i = kI + 1; i <= kI + 2; ++i) {
            const auto c = r.fn.calls_for(i);
            REQUIRE(!c.empty());
            CHECK(relc(c[0], i) == 1'200'000);
        }
    }
}

TEST_CASE("engine: pabudimo vėlavimo riba — < P neišmeta indekso", "[engine][skips]") {
    constexpr Index kI = 1010;
    struct Case {
        std::uint64_t over;
        std::uint64_t skips;
    };
    for (const Case cs : {Case{9'600'000, 0}, Case{19'999'999, 0}, Case{20'000'000, 1}, Case{45'000'000, 2}}) {
        INFO("pavėlavimas " << cs.over << " ns");
        Rig r(2);
        r.fn.on_end_tick = [&](Index i, TickContext&) {
            if (i == kI - 1) r.clk.oversleep_once(cs.over);
            if (i >= kI + 3) r.stop = true;
        };
        r.build();
        r.run();
        CHECK(r.out(0).late_skips == cs.skips);
        const auto* e = r.outs[0]->find(kI);
        if (cs.skips == 0) {
            REQUIRE(e != nullptr);
            CHECK(e->kind == Kind::Commit);
            CHECK(rel(e, kI) == 500'000 + static_cast<std::int64_t>(cs.over));  // atidarytas pavėlavus, iškart
        } else {
            CHECK(e == nullptr);
        }
    }
}

TEST_CASE("engine: 59,94 / 29,97 / 25 Hz — 5000 tick'ų tiksliai", "[engine][rates]") {
    for (mxlv::timing::Rational rate :
         {mxlv::timing::Rational{60000, 1001}, mxlv::timing::Rational{30000, 1001}, mxlv::timing::Rational{25, 1}}) {
        INFO("rate " << rate.num << "/" << rate.den);
        EngineConfig c = Rig::test_config();
        c.cuts = default_cuts(rate);
        const Index start = 106'000'000'000ull;  // TAI ≈ 1,77·10¹⁸ ns prie 59,94 Hz
        Rig r(2, c, start);
        r.stop_after(start + 5000);
        r.build();
        const RunResult res = r.run();
        std::uint64_t bad = 0;
        const auto& ev = r.outs[0]->events;
        for (std::size_t n = 0; n < ev.size(); ++n) {
            if (ev[n].index != res.first_index + n || ev[n].kind != Kind::Commit) ++bad;
            if (static_cast<std::int64_t>(ev[n].tai) - T(ev[n].index, rate) != 1'200'000) ++bad;
        }
        CHECK(ev.size() == 5001);
        CHECK(bad == 0);
    }
}

TEST_CASE("engine: rašytojas miršta ir grįžta — sveikas nepaliestas", "[engine][isolation]") {
    constexpr Index kI = 1010;
    Rig r(2);
    ScriptedInput& k = *r.ins[1];
    k.arrival = [](Index j) { return (j >= kI && j < kI + 20) ? kNever : std::int64_t{1'200'000}; };
    k.alive = [](std::uint64_t now) {
        const auto n = static_cast<std::int64_t>(now);
        return n < T(kI) || n >= T(kI + 20);
    };
    k.attachable = k.alive;
    r.stop_after(kI + 40);
    r.build();
    r.run();
    CHECK(r.in(1).detaches == 1);
    CHECK(r.in(1).reattached == 1);
    CHECK(rel(r.outs[0]->find(kI), kI) == 5'500'000);
    for (Index i = kI + 1; i <= kI + 40; ++i) CHECK(rel(r.outs[0]->find(i), i) == 1'200'000);
    for (Index i = kI + 21; i <= kI + 40; ++i) CHECK(r.outs[1]->find(i)->kind == Kind::Commit);
}

TEST_CASE("engine: žingsnis atgal su atvirais vienetais (D = 15 ir 30)", "[engine][clock]") {
    constexpr Index kI = 1010;
    for (std::int64_t dl : {15'000'000ll, 30'000'000ll}) {
        INFO("D = " << dl);
        EngineConfig c = Rig::test_config();
        c.cuts.deadline_ns = dl;
        Rig r(2, c);
        const std::int64_t arr = dl == 15'000'000 ? 12'000'000 : 26'000'000;
        r.ins[1]->arrival = [arr](Index j) { return j >= kI ? arr : std::int64_t{1'200'000}; };
        bool armed = false;
        const std::int64_t at_tai = dl == 15'000'000 ? T(kI) + 7'000'000 : T(kI + 1) + 3'000'000;
        r.fn.on_begin_tick = [&](Index i, TickContext&) {
            if (!armed && i == kI) {
                armed = true;
                r.clk.schedule_step(r.clk.now_mono_raw() + (at_tai - static_cast<std::int64_t>(r.clk.now_tai())),
                                    -1'300'000'000);
            }
        };
        r.fn.on_end_tick = [&](Index i, TickContext&) {
            if (i >= kI + 5) r.stop = true;
        };
        r.build();
        r.run();
        CHECK(r.hub.engine().clock_backward == 1);
        CHECK(r.hub.engine().step_gaps >= 1);
    }
}

TEST_CASE("engine: atsiliekantis rašytojas (~1,05 P) sveikų neliečia", "[engine][isolation]") {
    constexpr Index kI = 1010;
    Rig r(2);
    r.ins[1]->arrival = [](Index j) { return j >= kI ? std::int64_t{21'000'000} : std::int64_t{1'200'000}; };
    r.stop_after(kI + 100);
    r.build();
    r.run();
    CHECK(rel(r.outs[0]->find(kI), kI) == 5'500'000);
    for (Index i = kI + 1; i <= kI + 100; ++i) CHECK(rel(r.outs[0]->find(i), i) == 1'200'000);
}
