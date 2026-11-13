// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Šeštos peržiūros testai (2026-11-13, f61e5b6..f578fc3): neprivalomi įėjimai (composite sluoksniai), aukštupio spragos,
// enc garso laukimo sutartis.
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
// Vienetas = Required (bazė) + Wanted (sluoksnis) → 1 išėjimas (composite forma).
struct LayerRig {
    explicit LayerRig(EngineConfig c = Rig::test_config()) : clk(static_cast<std::uint64_t>(T(1000))), fn(clk), cfg(std::move(c)) {
        base = std::make_unique<ScriptedInput>(clk);
        layer = std::make_unique<ScriptedInput>(clk);
        out = std::make_unique<RecordingOutput>(clk);
    }
    void build() {
        eng = std::make_unique<TickEngine>(cfg, clk, hub, stop);
        UnitSpec s;
        s.name = "comp";
        s.in.push_back(InputSpec{base.get(), {}, 1, Need::Required, InMode::Exact});
        s.in.push_back(InputSpec{layer.get(), {}, 1, Need::Wanted, InMode::Exact});
        s.out.push_back(out.get());
        std::uint32_t id = 0;
        std::string why;
        REQUIRE(eng->add_unit(s, id, why));
        fn.on_end_tick = [this](Index i, TickContext&) {
            if (i >= last) stop = true;
        };
    }
    RunResult run() {
        res = eng->run(fn);
        CHECK(out->strictly_increasing());
        check_run(fn, res, 0, out.get());
        return res;
    }
    VirtualClock clk;
    StatsHub hub;
    std::atomic<bool> stop{false};
    RecordingFunction fn;
    EngineConfig cfg;
    std::unique_ptr<ScriptedInput> base, layer;
    std::unique_ptr<RecordingOutput> out;
    std::unique_ptr<TickEngine> eng;
    Index last = 1100;
    RunResult res;
};
}  // namespace

// Punktas 1: neprivalomo (Wanted) įėjimo trūkumas — ne vieneto praradimas. Sluoksnio rašytojas gyvas, bet nuo kL grain'ų
// nebėra (užstrigęs), vėliau nužudomas: kiekvienas indeksas commit'inamas (Ready), any_loss == false, o sluoksnio `lost`
// skaičiuojamas (tai jo būsena, ne išėjimo praradimas).
TEST_CASE("review6: 1 — Wanted įėjimo trūkumas nėra any_loss; vienetas commit'ina kiekvieną indeksą", "[engine][review6][optional]") {
    constexpr Index kL = 1020, kDead = 1060;
    LayerRig r;
    r.layer->arrival = [](Index j) { return j >= kL ? kNever : std::int64_t{1'200'000}; };
    r.layer->alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kDead); };
    r.build();
    const RunResult res = r.run();
    REQUIRE(res.reason == EndReason::Stopped);
    for (Index i = 1001; i <= r.last; ++i) {
        INFO("i = " << i);
        const auto* e = r.out->find(i);
        REQUIRE(e != nullptr);
        CHECK(e->kind == RecordingOutput::Kind::Commit);
    }
    CHECK(r.hub.in(r.eng->in_id(0, 1)).lost > 0);
    CHECK_FALSE(res.any_loss);
}

// Kontrolė: būtino įėjimo (bazės) trūkumas — praradimas, kaip anksčiau.
TEST_CASE("review6: 1 — Required įėjimo trūkumas lieka any_loss", "[engine][review6][optional]") {
    constexpr Index kL = 1020;
    LayerRig r;
    r.base->arrival = [](Index j) { return j >= kL && j < kL + 3 ? kNever : std::int64_t{1'200'000}; };
    r.build();
    const RunResult res = r.run();
    CHECK(res.any_loss);
}

// --- Punktas 3: aukštupio spraga — ne šios funkcijos praradimas ------------------------------------------------------
// Būtinas įėjimas pateikia INVALID grain'us (aukštupio paleidimas / gedimas): išėjimas — spraga pagal politiką, indeksas
// skaičiuojamas gap_upstream (vieneto gaps_upstream ⊂ gaps), any_loss == false; end_unit — Outcome::GapUpstream.
TEST_CASE("review6: 3 — INVALID būtiname įėjime → gap_upstream, ne praradimas", "[engine][review6][upstream]") {
    constexpr Index kA = 1010, kB = 1030;
    Rig r(2);
    r.ins[0]->invalid = [](Index j) { return j >= kA && j < kB; };
    r.stop_after(1060);
    r.build();
    const RunResult res = r.run();
    REQUIRE(res.reason == EndReason::Stopped);
    CHECK(r.out(0).gap_upstream == kB - kA);
    CHECK(r.out(0).gap_invalid == 0);  // spragos veiksmas — invalid, bet priežastis — aukštupys (kaip gap_startup)
    CHECK(r.hub.unit(r.eng->unit_id(0)).gaps_upstream == kB - kA);
    CHECK(r.hub.unit(r.eng->unit_id(0)).gaps == kB - kA);
    for (Index i = kA; i < kB; ++i) {
        INFO("i = " << i);
        const auto* e = r.outs[0]->find(i);
        REQUIRE(e != nullptr);
        CHECK(e->kind == RecordingOutput::Kind::Gap);
        CHECK(e->action == GapAction::Invalid);
    }
    std::uint64_t up = 0;
    for (const auto& e : r.fn.ends)
        if (e.unit == 0 && e.outcome == Outcome::GapUpstream) ++up;
    CHECK(up == kB - kA);
    CHECK(r.in(0).invalid == kB - kA);
    CHECK_FALSE(res.any_loss);
}

// Kontrolė: tas pats su dingusiu (ne INVALID) įėjimu — praradimas (missing / GapBlocked), kaip anksčiau.
TEST_CASE("review6: 3 — dingęs būtinas įėjimas lieka praradimas", "[engine][review6][upstream]") {
    constexpr Index kA = 1010;
    Rig r(2);
    r.ins[0]->arrival = [](Index j) { return j >= kA ? kNever : std::int64_t{1'200'000}; };
    r.ins[0]->alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kA); };
    r.ins[0]->attachable = r.ins[0]->alive;
    r.stop_after(1040);
    r.build();
    const RunResult res = r.run();
    CHECK(r.out(0).gap_upstream == 0);
    CHECK(res.any_loss);
}

// freeze / black interlaced: turinys, grįžtantis nelyginiame indekse, laikomas iki kadro ribos — gap_align (ne praradimas,
// ne GapBlocked / missing_fields).
TEST_CASE("review6: 3 — interlaced freeze: turinys nelyginiame indekse laikomas iki kadro ribos → gap_align", "[engine][review6][align]") {
    constexpr Index kA = 1010, kB = 1021;  // kB nelyginis: pirmas grįžęs grain'as — apatinis laukas
    EngineConfig c = Rig::test_config();
    c.on_loss = GapPolicy::Freeze;
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    RecordingFunction fn(clk);
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput in(clk);
    in.invalid = [](Index j) { return j >= kA && j < kB; };
    RecordingOutput out(clk, /*interlaced=*/true);
    TickEngine eng(c, clk, hub, stop);
    UnitSpec s;
    s.name = "u";
    s.in.push_back(InputSpec{&in, {}, 1, Need::Required, InMode::Exact});
    s.out.push_back(&out);
    std::uint32_t id = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, id, why));
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1040) stop = true;
    };
    const RunResult res = eng.run(fn);
    check_run(fn, res, 0, &out);
    std::uint64_t align = 0, blocked = 0;
    for (const auto& e : fn.ends) {
        if (e.outcome == Outcome::GapAlign) {
            ++align;
            CHECK(e.index == kB);
        }
        if (e.outcome == Outcome::GapBlocked) ++blocked;
    }
    CHECK(align == 1);
    CHECK(blocked == 0);
    CHECK(hub.out(eng.out_id(id, 0)).gap_align == 1);
    CHECK(hub.unit(eng.unit_id(id)).gaps_align == 1);
    const auto* e = out.find(kB + 1);
    REQUIRE(e != nullptr);
    CHECK(e->kind == RecordingOutput::Kind::Commit);  // turinys — nuo lyginio (kadro ribos)
    CHECK_FALSE(res.any_loss);
}

// --- Punktas 4: enc garso laukimo sutartis — dabar briaunos sutarties atvejis (V24, K-T10) -----------------------------
// Kadro garsas baigiamas ties T(j) + P (realaus laiko rašytojas): garso įėjimas skelbia expect P, todėl variklio fazė E = P
// (viena fazė varikliui) ir tick'as atidaromas ties T + P + ro — VISI vienetai (ir tas, kurio garsas ateina iškart) apdorojami
// A fazėje su garsu, niekas nevėlus, riba D = 3P/4 skaičiuojama nuo T + E (atskiros enc ribos 3P/2 nebėra).
// Pakeistas lūkestis (V24): anksčiau (atskira taisyklė, D = 3P/2) srautas su garsu iškart apdorojamas ties T + 1,3 ms; pagal
// K-T10 skirtingo gylio įėjimai viename variklyje gauna didžiausią vėlinimą (rizika R11, dokumentuota).
TEST_CASE("review6: 4 — garsas ties T + P su expect P: fazė E = P, be tylos, niekas nevėlus", "[engine][review6][audio]") {
    constexpr std::int64_t kP = 20'000'000;
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    RecordingFunction fn(clk);
    StatsHub hub;
    std::atomic<bool> stop{false};
    EngineConfig cfg = Rig::test_config();
    // numatytoji D = 3P/4 (nuo T + E) — atskiros enc ribos nebėra
    std::vector<std::unique_ptr<ScriptedInput>> video, audio;
    for (int u = 0; u < 2; ++u) {
        video.push_back(std::make_unique<ScriptedInput>(clk));
        audio.push_back(std::make_unique<ScriptedInput>(clk));
    }
    audio[0]->arrival = [](Index) { return kP + 300'000; };   // kadro garsas baigiamas ties T + P (+0,3 ms rašymas)
    audio[1]->arrival = [](Index) { return std::int64_t{1'300'000}; };  // garsas iškart po video
    TickEngine eng(cfg, clk, hub, stop);
    for (std::uint32_t u = 0; u < 2; ++u) {
        UnitSpec s;
        s.name = "enc" + std::to_string(u);
        s.in.push_back(InputSpec{video[u].get(), {}, 2, Need::Required, InMode::Exact});
        s.in.push_back(InputSpec{audio[u].get(), EdgeTiming{0, kP}, 1, Need::Wanted, InMode::Exact});
        s.stride = 2;
        s.residue = 1;
        std::uint32_t id = 0;
        std::string why;
        REQUIRE(eng.add_unit(s, id, why));
    }
    struct Seen {
        std::uint8_t phase;
        std::int64_t at;
        bool audio;
    };
    std::map<std::pair<std::uint32_t, Index>, Seen> seen;
    fn.on_process = [&](const Batch& b, TickContext&) {
        for (const UnitTick& ut : b.units)
            seen[{ut.unit, ut.index}] = {b.phase, static_cast<std::int64_t>(clk.now_tai()) - T(ut.index), ut.in[2].ptr != nullptr};
    };
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1101) stop = true;
    };
    const RunResult res = eng.run(fn);
    check_run(fn, res, 0, nullptr, 2, 1);
    check_run(fn, res, 1, nullptr, 2, 1);
    std::uint64_t n0 = 0;
    for (Index i = 1021; i <= 1099; i += 2) {
        INFO("i = " << i);
        REQUIRE(seen.count({0, i}) == 1);
        CHECK(seen[{0, i}].audio);  // garsas yra — ne tyla
        CHECK(seen[{0, i}].phase == 0);
        CHECK(seen[{0, i}].at == kP + 500'000);  // t_open = T + E + ro: garsas jau commit'intas (T + P + 0,3 ms)
        REQUIRE(seen.count({1, i}) == 1);
        CHECK(seen[{1, i}].audio);
        CHECK(seen[{1, i}].phase == 0);
        CHECK(seen[{1, i}].at == kP + 500'000);  // viena fazė varikliui (K-T10)
        ++n0;
    }
    CHECK(n0 == 40);
    const InCounters& a0 = hub.in(eng.in_id(0, 1));
    CHECK(eng.expect_ns() == kP);
    CHECK(a0.late == 0);  // laukiamas ties T + P — laiku
    CHECK(a0.lost == 0);
    CHECK_FALSE(res.any_loss);
}
