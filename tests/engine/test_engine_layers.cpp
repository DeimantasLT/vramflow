// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Peržiūra D2: neprivalomas sluoksnis (Need::UntilA) laukiamas tik iki a(i) — niekada iki paskutinės ribos; vėluojantis
// sluoksnis neperkelia vieneto į B ir nevėlina programos (commit'o riba nuo sluoksnių nepriklauso).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
struct LayerRig {
    explicit LayerRig(std::int64_t layer_arrival, bool with_layer = true) : clk(static_cast<std::uint64_t>(T(1000))), fn(clk), base(clk),
                                                                             layer(clk), out(clk) {
        base.arrival = [](Index) { return std::int64_t{1'200'000}; };
        layer.arrival = [layer_arrival](Index) { return layer_arrival; };
        eng = std::make_unique<TickEngine>(Rig::test_config(), clk, hub, stop);
        UnitSpec s;
        s.name = "comp";
        s.in.push_back(InputSpec{&base, {}, 1, Need::Required, InMode::Exact});
        if (with_layer) s.in.push_back(InputSpec{&layer, {}, 1, Need::UntilA, InMode::Exact});
        s.out.push_back(&out);
        std::string why;
        REQUIRE(eng->add_unit(s, id, why));
        fn.on_process = [this, with_layer](const Batch& b, TickContext&) {
            for (const UnitTick& ut : b.units) seen[ut.index] = {b.phase, static_cast<std::int64_t>(clk.now_tai()) - T(ut.index),
                                                                with_layer && ut.in[1].ptr != nullptr};
        };
        fn.on_end_tick = [this](Index i, TickContext&) {
            if (i >= 1100) stop = true;
        };
    }
    struct Seen {
        std::uint8_t phase;
        std::int64_t at;
        bool layer;
    };
    VirtualClock clk;
    RecordingFunction fn;
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput base, layer;
    RecordingOutput out;
    std::unique_ptr<TickEngine> eng;
    std::uint32_t id = 0;
    std::map<Index, Seen> seen;
};
}  // namespace

TEST_CASE("engine: UntilA sluoksnis vėluoja 1,5 ms po a(i) kiekvieną kadrą — programa nevėluoja", "[engine][layers]") {
    LayerRig ctl(0, false);  // kontrolė be sluoksnio
    const RunResult rc = ctl.eng->run(ctl.fn);
    LayerRig r(7'000'000);   // a(i) = T + 5,5 ms; sluoksnis — T + 7 ms
    const RunResult res = r.eng->run(r.fn);
    check_run(r.fn, res, 0, &r.out, 1, 0);
    std::uint32_t transitions = 0;
    bool prev = true;
    for (Index i = 1002; i <= 1100; ++i) {
        INFO("i = " << i);
        REQUIRE(r.seen.count(i) == 1);
        REQUIRE(ctl.seen.count(i) == 1);
        CHECK(r.seen[i].phase == 0);                 // niekada B / C dėl sluoksnio
        CHECK(r.seen[i].at <= 5'500'000);            // ne vėliau kaip a(i)
        CHECK_FALSE(r.seen[i].layer);                // stabiliai nenaudojamas (niekada nespėja iki a(i))
        if (r.seen[i].layer != prev) ++transitions;
        prev = r.seen[i].layer;
        if (i >= 1004) CHECK(r.seen[i].at == ctl.seen[i].at);  // po pirmųjų (Late) — kaip be sluoksnio
    }
    CHECK(transitions <= 1);
    CHECK(r.hub.out(r.eng->out_id(r.id, 0)).late == 0);
    CHECK(r.hub.out(r.eng->out_id(r.id, 0)).ok == ctl.hub.out(ctl.eng->out_id(ctl.id, 0)).ok);
    CHECK_FALSE(res.any_loss);
    CHECK_FALSE(rc.any_loss);
}

TEST_CASE("engine: UntilA sluoksnis laiku — naudojamas, paleidžiama jo atėjimu", "[engine][layers]") {
    LayerRig r(3'000'000);
    const RunResult res = r.eng->run(r.fn);
    for (Index i = 1002; i <= 1100; ++i) {
        INFO("i = " << i);
        REQUIRE(r.seen.count(i) == 1);
        CHECK(r.seen[i].layer);
        CHECK(r.seen[i].phase == 0);
        CHECK(r.seen[i].at == 3'000'000);
    }
    CHECK_FALSE(res.any_loss);
}

TEST_CASE("engine: UntilA sluoksnis vienam indeksui vėluoja iki B — tik tas indeksas be jo, ties a(i)", "[engine][layers]") {
    LayerRig r(0);
    r.layer.arrival = [](Index j) { return j == 1050 ? std::int64_t{12'000'000} : std::int64_t{1'500'000}; };
    const RunResult res = r.eng->run(r.fn);
    REQUIRE(r.seen.count(1050) == 1);
    CHECK_FALSE(r.seen[1050].layer);
    CHECK(r.seen[1050].phase == 0);
    CHECK(r.seen[1050].at == 5'500'000);  // laukta tik iki a(i), ne iki B = T + 15 ms
    CHECK(r.seen[1049].layer);
    CHECK_FALSE(res.any_loss);
}
