// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Ketvirtos peržiūros (48a8a51..863ff26) radiniai: Wanted įėjimas (enc garsas) laukiamas kaip Required per B / C fazes su
// ta pačia Late histereze, be jo tęsiama tik vieneto paskutinėje riboje; C fazė neišmeta vieneto, paruošto iki savo ribos.
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
// enc forma: N vienetų, video (2 grain'ai, Required) + garsas (Wanted), stride 2, residue 1. D = 30 ms (enc numatytoji):
// a = T + 5,5, b = T + 17,5, c = T + 30.
struct EncRig {
    explicit EncRig(std::uint32_t n) : clk(static_cast<std::uint64_t>(T(1000))), fn(clk) {
        cfg = Rig::test_config();
        cfg.cuts.deadline_ns = 30'000'000;
        for (std::uint32_t u = 0; u < n; ++u) {
            video.push_back(std::make_unique<ScriptedInput>(clk));
            audio.push_back(std::make_unique<ScriptedInput>(clk));
        }
    }
    void build() {
        eng = std::make_unique<TickEngine>(cfg, clk, hub, stop);
        for (std::uint32_t u = 0; u < video.size(); ++u) {
            UnitSpec s;
            s.name = "enc" + std::to_string(u);
            s.in.push_back(InputSpec{video[u].get(), {}, 2, Need::Required, InMode::Exact});
            s.in.push_back(InputSpec{audio[u].get(), {}, 1, Need::Wanted, InMode::Exact});
            s.stride = 2;
            s.residue = 1;
            std::uint32_t id = 0;
            std::string why;
            REQUIRE(eng->add_unit(s, id, why));
        }
        fn.on_process = [this](const Batch& b, TickContext&) {
            for (const UnitTick& ut : b.units)
                seen[{ut.unit, ut.index}] = {b.phase, static_cast<std::int64_t>(clk.now_tai()) - T(ut.index), ut.in[2].ptr != nullptr};
        };
    }
    struct Seen {
        std::uint8_t phase;
        std::int64_t at;
        bool audio;
    };
    VirtualClock clk;
    StatsHub hub;
    std::atomic<bool> stop{false};
    RecordingFunction fn;
    EngineConfig cfg;
    std::vector<std::unique_ptr<ScriptedInput>> video, audio;
    std::unique_ptr<TickEngine> eng;
    std::map<std::pair<std::uint32_t, Index>, Seen> seen;
};
}  // namespace

TEST_CASE("review4: garsas vėluoja 20 ms kiekvieną kadrą — be tylos, kiti vienetai A fazėje savo paruošimu", "[engine][review4][wanted]") {
    // Realaus laiko garso rašytojas kadro garsą baigia ties T(i) + P (+ 0,5 ms). Anksčiau Wanted buvo laukiama tik iki
    // a(i) = T + 5,5: kiekvienas kadras — tyla, o visų vienetų A laukė iki a(i).
    constexpr std::uint32_t kN = 4;
    EncRig r(kN);
    r.audio[0]->arrival = [](Index) { return std::int64_t{20'500'000}; };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1061) r.stop = true;
    };
    r.build();
    const RunResult res = r.eng->run(r.fn);
    for (std::uint32_t u = 0; u < kN; ++u) check_run(r.fn, res, u, nullptr, 2, 1);
    int silent = 0, late_unit = 0;
    for (Index i = 1011; i <= 1061; i += 2) {
        INFO("i = " << i);
        REQUIRE(r.seen.count({0, i}) == 1);
        const auto& s0 = r.seen[{0, i}];
        silent += s0.audio ? 0 : 1;
        CHECK(s0.at == 20'500'000);  // apdorojamas atėjus garsui (C fazė iki T + 30)
        if (s0.phase != 0) ++late_unit;
        for (std::uint32_t u = 1; u < kN; ++u) {
            const auto& s = r.seen[{u, i}];
            CHECK(s.phase == 0);
            CHECK(s.at == 1'200'000);  // sveikų A — jų pačių paruošimas (video ir garsas T + 1,2), ne a(i)
            CHECK(s.audio);
        }
    }
    CHECK(silent == 0);
    CHECK(late_unit == 26);  // nuolat vėluojantis — tik savo vienetas už A fazės
    CHECK(r.hub.in(r.eng->in_id(0, 1)).lost == 0);
    CHECK(r.hub.in(r.eng->in_id(0, 1)).stalls == 0);
    CHECK(r.hub.in(r.eng->in_id(0, 1)).late >= 25);
    CHECK_FALSE(res.any_loss);
    CHECK(r.fn.at_most_three_per_index());
}

TEST_CASE("review4: Wanted be grain'o iki paskutinės ribos — vienetas tęsia be jo (ne spraga)", "[engine][review4][wanted]") {
    constexpr Index kI = 1031;
    EncRig r(2);
    // netikras rašytojas grain'us pateikia eilės tvarka: ir kitas (lyginis) indeksas vėluoja, kad kI nepasirodytų anksčiau
    r.audio[0]->arrival = [](Index j) { return j == kI || j == kI + 1 ? std::int64_t{40'000'000} : std::int64_t{1'300'000}; };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1051) r.stop = true;
    };
    r.build();
    const RunResult res = r.eng->run(r.fn);
    check_run(r.fn, res, 0, nullptr, 2, 1);
    REQUIRE(r.seen.count({0, kI}) == 1);
    CHECK(r.seen[{0, kI}].at == 30'000'000);  // paskutinė riba c = T + 30
    CHECK_FALSE(r.seen[{0, kI}].audio);
    CHECK(r.seen[{1, kI}].at == 5'500'000);  // kitas — tik vienkartinis kvorumo laikymas iki a(i), ne iki c
    CHECK(r.seen[{1, kI}].phase == 0);
    CHECK(r.seen[{1, kI + 2}].at == 1'200'000);  // kitame kadre — vėl savo paruošimu (vėluojantis jau Late → B)
    for (const auto& e : r.fn.ends)
        if (e.unit == 0 && e.index == kI) CHECK(e.outcome == Outcome::Ready);
    CHECK(r.hub.in(r.eng->in_id(0, 1)).lost == 1);
    CHECK(res.any_loss);  // garso grain'as prarastas — praradimas (exit 4 griežtame režime)
}
