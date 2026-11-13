// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Šeštos peržiūros testai (2026-11-13, f61e5b6..f578fc3): neprivalomi įėjimai (composite sluoksniai), aukštupio spragos,
// enc garso laukimo sutartis.
#include <catch2/catch_test_macros.hpp>

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
