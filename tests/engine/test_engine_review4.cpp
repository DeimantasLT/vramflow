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
    CHECK(r.fn.at_most_four_per_index());
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

TEST_CASE("review4: C fazė — po A + B + ankstyvo C atėjęs iki c(i) vienetas apdorojamas paskutiniu batch'u, ne spraga",
          "[engine][review4][phasec]") {
    // D = 30 ms: a = T + 5,5, b = T + 17,5, c = T + 30. Indekse kI: u0 laiku (A ties a — kvorumas laukė), u1 T + 10 (B ties
    // t_r + Wr = T + 15), u2 T + 22 (C ties t_r + Wr = T + 27 — trečias kvietimas), u3 T + 28 (< c). Anksčiau u3 — spraga
    // ties c (kvietimų limitas 3), nors buvo paruoštas iki savo ribos.
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    Rig r(4, c);
    // Netikras rašytojas grain'us pateikia eilės tvarka (vėlesnis padaro ankstesnį matomą), todėl u2 / u3 kito indekso
    // grain'as ateina T(kI + 1) + 9 ms = T(kI) + 29 — po jų vėlavusių kI grain'ų.
    const std::int64_t at[4] = {1'200'000, 10'000'000, 22'000'000, 28'000'000};
    for (int u = 0; u < 4; ++u)
        r.ins[static_cast<std::size_t>(u)]->arrival = [a = at[u], u](Index j) {
            if (j == kI) return a;
            return u >= 2 && j == kI + 1 ? std::int64_t{9'000'000} : std::int64_t{1'200'000};
        };
    r.stop_after(kI + 6);
    r.build();
    const RunResult res = r.run();
    CHECK_FALSE(res.any_loss);
    const std::int64_t want[4] = {5'500'000, 15'000'000, 27'000'000, 30'000'000};
    for (std::uint32_t u = 0; u < 4; ++u) {
        INFO("u" << u);
        const auto* e = r.outs[u]->find(kI);
        REQUIRE(e != nullptr);
        CHECK(e->kind == RecordingOutput::Kind::Commit);
        CHECK(static_cast<std::int64_t>(e->tai) - T(kI) == want[u]);
    }
    const auto calls = r.fn.calls_for(kI);
    REQUIRE(calls.size() == 4);  // A, B, C ir paskutinis batch'as ties c
    CHECK(calls[3]->phase == 2);
    CHECK(calls[3]->units == std::vector<std::uint32_t>{3});
    CHECK(r.hub.engine().cut_batches == 1);
    CHECK(r.hub.engine().carry_overrun == 0);
}

TEST_CASE("review4: fail_unit — vienas išėjimas sugenda, kiti N − 1 nepaliesti (K17, K-T11)", "[engine][review4][service]") {
    constexpr Index kI = 1010;
    Rig r(4);
    r.outs[2]->fail_commit = [](Index i) { return i >= kI; };
    r.stop_after(kI + 30);
    r.build();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Stopped);
    CHECK(res.exit_code == 0);  // exit politika — funkcijos (exit_code_for: griežtai 2, tarnyboje 0)
    CHECK(res.failed_units == 1);
    CHECK(r.outs[2]->released);
    CHECK(r.out(2).commit_errors == 3);
    CHECK(res.any_loss);
    for (std::uint32_t u : {0u, 1u, 3u}) {
        INFO("u" << u);
        CHECK_FALSE(r.outs[u]->released);
        CHECK(r.out(u).commit_errors == 0);
        for (Index i = kI; i <= kI + 30; ++i) {
            const auto* e = r.outs[u]->find(i);
            REQUIRE(e != nullptr);
            CHECK(e->kind == RecordingOutput::Kind::Commit);
            CHECK(static_cast<std::int64_t>(e->tai) - T(i) == 1'200'000);  // kitų laikas nepakito
        }
    }
    // sugedęs vienetas po gedimo nebeatidaromas: jokių įvykių po kI + 2
    for (const auto& e : r.outs[2]->events) CHECK(e.index <= kI + 2);
}

TEST_CASE("review4: TickContext::fail_unit iš funkcijos — vienetas baigiamas, išėjimas atlaisvinamas", "[engine][review4][service]") {
    constexpr Index kI = 1010;
    Rig r(2);
    r.fn.on_begin_tick = [](Index i, TickContext& ctx) {
        if (i == kI) ctx.fail_unit(0, "testas");
    };
    r.stop_after(kI + 10);
    r.build();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Stopped);
    CHECK(res.failed_units == 1);
    CHECK(res.failed_why == "testas");
    CHECK(r.outs[0]->released);
    CHECK(r.outs[0]->find(kI) == nullptr);
    CHECK(r.outs[1]->find(kI + 10) != nullptr);
}
