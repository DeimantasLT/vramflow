// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V24: briaunos laiko sutartis (planas §6 D7, §9 V24; K-T10 viena fazė varikliui, K-T14 skelbiama riba, ne matavimas).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/audio_frames.hpp"
#include "engine/edge.hpp"
#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
constexpr std::int64_t kP = 20'000'000;
using Kind = RecordingOutput::Kind;
}  // namespace

TEST_CASE("engine: index_offset −2", "[engine][edge]") {
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    RecordingFunction fn(clk);
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput in(clk);
    RecordingOutput out(clk);
    TickEngine eng(c, clk, hub, stop);
    UnitSpec s;
    s.name = "u0";
    s.in.push_back(InputSpec{&in, EdgeTiming{-2, 0, 0}, 1, Need::Required, InMode::Exact});
    s.out.push_back(&out);
    std::uint32_t id = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, id, why));
    std::vector<std::pair<Index, Index>> seen;  // (i, paskutinis probe j) — be alokacijų tick'e nesvarbu (testas ne apie tai)
    seen.reserve(64);
    fn.on_process = [&](const Batch& b, TickContext&) { seen.emplace_back(b.index, in.last_probe); };
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= kI) stop = true;
    };
    const RunResult res = eng.run(fn);
    REQUIRE(seen.size() >= 5);
    for (const auto& [i, j] : seen) {
        INFO("i = " << i);
        CHECK(j == i - 2);  // skaitomas j = i − 2
    }
    const auto* e = out.find(kI);
    REQUIRE(e != nullptr);
    CHECK(e->kind == Kind::Commit);  // išėjimo indeksas lieka i
    CHECK(eng.output_timing(id).content_delay_ns == 2 * kP);  // 40 ms
    CHECK_FALSE(res.any_loss);
}

TEST_CASE("engine: expect 9 ms", "[engine][edge]") {
    constexpr Index kI = 1010;
    constexpr std::int64_t kE = 9'000'000;
    EngineConfig c = Rig::test_config();  // ro 0,5 ms, Wr 5 ms, D 15 ms, G 3 ms
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    RecordingFunction fn(clk);
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput up(clk), other(clk);
    up.arrival = [](Index) { return std::int64_t{8'900'000}; };       // aukštupis commit'ina pagal savo ribą (9 ms)
    other.arrival = [](Index) { return std::int64_t{1'200'000}; };    // kitas įėjimas — be žymos (expect 0)
    RecordingOutput o0(clk), o1(clk);
    TickEngine eng(c, clk, hub, stop);
    std::uint32_t u0 = 0, u1 = 0;
    std::string why;
    {
        UnitSpec s;
        s.name = "u0";
        s.in.push_back(InputSpec{&up, EdgeTiming{0, kE, 0}, 1, Need::Required, InMode::Exact});
        s.out.push_back(&o0);
        REQUIRE(eng.add_unit(s, u0, why));
    }
    {
        UnitSpec s;
        s.name = "u1";
        s.in.push_back(InputSpec{&other, EdgeTiming{}, 1, Need::Required, InMode::Exact});
        s.out.push_back(&o1);
        REQUIRE(eng.add_unit(s, u1, why));
    }
    CHECK(eng.expect_ns() == kE);            // E = max(expect_ns) — viena fazė varikliui (K-T10)
    CHECK(eng.phase_ns() == 9'500'000);      // Φ = E + ro
    std::vector<std::int64_t> open_rel;
    open_rel.reserve(64);
    fn.on_begin_tick = [&](Index i, TickContext&) { open_rel.push_back(static_cast<std::int64_t>(clk.now_tai()) - T(i)); };
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= kI) stop = true;
    };
    const RunResult res = eng.run(fn);
    REQUIRE(!open_rel.empty());
    for (const std::int64_t t : open_rel) CHECK(t == 9'500'000);  // t_open = T + 9,5 ms
    const auto calls = fn.calls_for(kI);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0]->phase == 0);
    CHECK(static_cast<std::int64_t>(calls[0]->tai) - T(kI) == 9'500'000);  // sveikas vienetas paleidžiamas be laukimo
    CHECK(calls[0]->units.size() == 2);
    CHECK(up.waits == 0);
    CHECK(other.waits == 0);
    const auto* e = o0.find(kI);
    REQUIRE(e != nullptr);
    CHECK(e->kind == Kind::Commit);
    CHECK(static_cast<std::int64_t>(e->tai) - T(kI) == 9'500'000);
    CHECK(hub.in(eng.in_id(u0, 0)).late == 0);
    CHECK(hub.out(eng.out_id(u0, 0)).late == 0);
    // Skelbiama riba: Φ + Wr + G = 9,5 + 5 + 3 = 17,5 ms; turinio vėlinimas — 0 (index_offset 0)
    CHECK(eng.output_timing(u0).commit_bound_ns == 17'500'000);
    CHECK(eng.output_timing(u0).content_delay_ns == 0);
    CHECK(eng.output_timing(u1).commit_bound_ns == 17'500'000);
    CHECK_FALSE(res.any_loss);
}

TEST_CASE("engine: rinkinio lygiavimas", "[engine][edge]") {
    // video content_delay 40 ms, garso 0 → garsas skaitomas 1920 mėginių anksčiau, išėjimas skelbia 40 ms
    const std::vector<BundleMember> m{{40'000'000, false}, {0, true}};
    const BundleAlign a = bundle_align(m, {50, 1});
    CHECK(a.align_ns == 40'000'000);
    REQUIRE(a.shift.size() == 2);
    CHECK(a.shift[0] == 0);
    CHECK(a.shift[1] == -1920);
    // atvirkščiai: garsas vėluoja 40 ms — video skaitomas 2 grain'ais anksčiau
    const std::vector<BundleMember> m2{{0, false}, {40'000'000, true}};
    const BundleAlign a2 = bundle_align(m2, {50, 1});
    CHECK(a2.align_ns == 40'000'000);
    CHECK(a2.shift[0] == -2);
    CHECK(a2.shift[1] == 0);
    // 1080i59.94 (grain'ai 60000/1001): 1 grain'as = 16 683 333 ns; vienodi — be poslinkių
    const std::vector<BundleMember> m3{{16'683'333, false}, {16'683'333, true}};
    const BundleAlign a3 = bundle_align(m3, {60000, 1001});
    CHECK(a3.align_ns == 16'683'333);
    CHECK(a3.shift[0] == 0);
    CHECK(a3.shift[1] == 0);
}

TEST_CASE("engine: rinkinio lygiavimas — garso mėginių poslinkis", "[engine][edge][audio]") {
    // 1080i50 (kadras — 2 laukai, 1920 mėginių): poslinkis −1920 — kadro F garsas skaitomas iš kadro F − 1 mėginių
    AudioFrames a{{25, 1}, 2};
    const AudioFrames plain = a;
    a.sample_shift = -1920;
    for (const Index j : {Index{2001}, Index{2003}, Index{40001}}) {
        CHECK(a.end(j) == plain.end(j) - 1920);
        CHECK(a.count(j) == plain.count(j));
    }
    // head — paskutinio įrašyto mėginių intervalo galas: su poslinkiu tas pats kadras paruoštas 1920 mėginių anksčiau
    const std::uint64_t head = plain.end(2001);
    CHECK(plain.last_ready(head) == 2001);
    CHECK(a.last_ready(head) == 2003);
    CHECK(a.last_ready(head - 1920) == 2001);
}

TEST_CASE("edge: commit bound ir fazė pagal dažnį", "[engine][edge]") {
    CutConfig c = default_cuts({50, 1});
    CHECK(phase_ns(c) == 500'000);
    CHECK(commit_bound_ns(c, false) == 8'500'000);  // 0,5 + 5 + 3
    c.read_offset_ns = 0;
    CHECK(commit_bound_ns(c, true) == 8'000'000);  // šaltinis (gen, ro 0): t_open + P/4 + G
    c.expect_ns = 11'500'000;
    c.read_offset_ns = 500'000;
    CHECK(phase_ns(c) == 12'000'000);
    CHECK(commit_bound_ns(c, false) == 20'000'000);
    const CutConfig p25 = default_cuts({25, 1});  // 25p: 0,5 + 10 + 6
    CHECK(commit_bound_ns(p25, false) == 16'500'000);
    const CutConfig legacy = legacy_cuts({50, 1}, 0, 15'000'000);  // šaltinis legacy ribose (Wr = D) — t_open + P/4 + G
    CHECK(commit_bound_ns(legacy, true) == 8'000'000);
    // content_delay: max(cd − index_offset · P)
    const std::vector<EdgeTiming> e{{0, 0, 0}, {-2, 0, 0}, {0, 0, 60'000'000}};
    CHECK(content_delay_ns(e, {50, 1}) == 60'000'000);
    const std::vector<EdgeTiming> e2{{-3, 0, 0}};
    CHECK(content_delay_ns(e2, {50, 1}) == 60'000'000);
    CHECK(content_delay_ns({}, {50, 1}) == 0);
}

TEST_CASE("edge: ring'o gylis", "[engine][edge]") {
    CutConfig c = default_cuts({50, 1});  // D 15 ms
    // 5 pakopų grandinė: E = 28,5 ms → lag 43,5 ms; istorija 200 ms = 10 slot'ų, saugu iki 160 ms
    c.expect_ns = 28'500'000;
    RingCheck r = ring_check(c, 0, 200'000'000);
    CHECK(r.fit == RingFit::Ok);
    CHECK(r.lag_ns == 43'500'000);
    CHECK(r.safe_ns == 160'000'000);
    c.expect_ns = 70'000'000;  // lag 85 ms > 80 — įspėjimas
    r = ring_check(c, 0, 200'000'000);
    CHECK(r.fit == RingFit::Warn);
    CHECK(r.text.find("history_duration") != std::string::npos);
    c.expect_ns = 150'000'000;  // lag 165 ms > 160 — atsisakoma
    r = ring_check(c, 0, 200'000'000);
    CHECK(r.fit == RingFit::Refuse);
    CHECK(r.text.find("165") != std::string::npos);
    CHECK(r.text.find("urn:x-mxl:option:history_duration/v1.0") != std::string::npos);
    r = ring_check(c, 0, 500'000'000);  // istorija 500 ms — telpa
    CHECK(r.fit == RingFit::Ok);
    c.expect_ns = 0;
    r = ring_check(c, -8, 200'000'000);  // index_offset −8: lag 15 + 160 = 175 ms
    CHECK(r.lag_ns == 175'000'000);
    CHECK(r.fit == RingFit::Refuse);
}

TEST_CASE("edge: žymų tekstas ir skaitymas", "[engine][edge]") {
    const std::string tags = timing_tags_json({11'500'000, 40'000'000});
    CHECK(tags == R"(,"urn:x-vramflow:tag:commit-bound-ns":["11500000"],"urn:x-vramflow:tag:content-delay-ns":["40000000"])");
    const std::string def = std::string(R"({"id":"x","tags":{"urn:x-nmos:tag:grouphint/v1.0":["g:Video 0"])") + tags + R"(},"format":"v"})";
    OutputTiming t;
    REQUIRE(parse_timing_tags(def, t));
    CHECK(t.commit_bound_ns == 11'500'000);
    CHECK(t.content_delay_ns == 40'000'000);
    OutputTiming u{7, 7};
    CHECK_FALSE(parse_timing_tags(R"({"id":"x","tags":{"urn:x-nmos:tag:grouphint/v1.0":["g:Video 0"]}})", u));  // svetimas flow
    CHECK(u.commit_bound_ns == 7);  // nepakeista
    CHECK_FALSE(parse_timing_tags(R"({"tags":{"urn:x-vramflow:tag:commit-bound-ns":["-5"]}})", u));  // netinkama reikšmė
    CHECK(parse_timing_tags(R"({"tags":{"urn:x-vramflow:tag:commit-bound-ns": [ "3000000" ]}})", u));  // tarpai leidžiami
    CHECK(u.commit_bound_ns == 3'000'000);
    CHECK(u.content_delay_ns == 0);
}
