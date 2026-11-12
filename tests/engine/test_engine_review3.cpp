// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Trečios (migracijos) peržiūros radiniai (kodas 4567b1e): išėjimo klaidos — savas rezultatas ir skaitiklis, ilgalaikė
// klaida — fatalu (exit 2); any_loss — tik tikri praradimai (ne laikrodis, ne šaltinio „nėra turinio“); C fazės
// Straggler'ių ryšys ribojamas kaip B fazėje.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "engine/audio_frames.hpp"
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

namespace {
// Įrašo kiekvieno Ready vieneto Wanted (antrojo) įėjimo GrainRef end_unit metu; apdorojimas „trunka“ proc_ns virtualaus laiko.
struct WantedProbeFn final : TickFunction {
    VirtualClock& clk;
    std::int64_t proc_ns;
    std::map<Index, bool> audio_at_end;  // indeksas → ar end_unit matė garso grain'ą
    std::function<void(Index)> on_end;
    WantedProbeFn(VirtualClock& c, std::int64_t p) : clk(c), proc_ns(p) {}
    bool process(const Batch&, TickContext&) override {
        clk.advance(static_cast<std::uint64_t>(proc_ns));
        return true;
    }
    void end_unit(const UnitTick& u, Outcome o, TickContext&) override {
        if (o == Outcome::Ready && u.in.size() == 3) audio_at_end[u.index] = u.in[2].ptr != nullptr;
    }
    void end_tick(Index i, TickContext&) override {
        if (on_end) on_end(i);
    }
};
}  // namespace

TEST_CASE("review3: Wanted grain'as, atėjęs apdorojimo metu, pasiekia end_unit ir nėra lost", "[engine][review3][wanted]") {
    // enc forma: video (2 grain'ai, Required) + garsas (Wanted), stride 2. Garsas indeksui 1031 ateina T + 6 ms (po a = T + 5,5),
    // apdorojimas trunka 1 ms (A paleidžiama ties a, baigiama T + 6,5) → po apdorojimo garsas jau yra.
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput video(clk), audio(clk);
    audio.arrival = [](Index j) { return j == 1031 ? std::int64_t{6'000'000} : std::int64_t{1'300'000}; };
    WantedProbeFn fn(clk, 1'000'000);
    fn.on_end = [&](Index i) {
        if (i >= 1041) stop = true;
    };
    TickEngine eng(Rig::test_config(), clk, hub, stop);
    UnitSpec s;
    s.name = "enc";
    s.in.push_back(InputSpec{&video, {}, 2, Need::Required, InMode::Exact});
    s.in.push_back(InputSpec{&audio, {}, 1, Need::Wanted, InMode::Exact});
    s.stride = 2;
    s.residue = 1;
    std::uint32_t id = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, id, why));
    const RunResult res = eng.run(fn);
    REQUIRE(fn.audio_at_end.count(1031) == 1);
    CHECK(fn.audio_at_end.at(1031));
    CHECK(fn.audio_at_end.at(1033));
    const InCounters& a = hub.in(eng.in_id(0, 1));
    CHECK(a.lost == 0);
    CHECK(a.late == 1);  // po a(c) — vėlus, bet ne prarastas
    CHECK_FALSE(res.any_loss);
}

TEST_CASE("review3: AudioFrames — kadro mėginiai ir head atvaizdavimas", "[engine][review3][audio]") {
    const AudioFrames i50{{25, 1}, 2};
    CHECK(i50.end(1) == 1920);  // kadras 0 — laukai 0, 1
    CHECK(i50.count(1) == 1920);
    CHECK(i50.end(2001) == 1001 * 1920);
    CHECK(i50.last_ready(1919) == kNoIndex);
    CHECK(i50.last_ready(1920) == 1);
    CHECK(i50.last_ready(1001 * 1920) == 2001);
    CHECK(i50.last_ready(1001 * 1920 + 1919) == 2001);
    CHECK(i50.last_ready(1002 * 1920) == 2003);
    const AudioFrames p2997{{30000, 1001}, 1};  // 1601 / 1602 mėginių kadrui
    std::uint64_t total = 0;
    for (Index j = 0; j < 5; ++j) total += p2997.count(j);
    CHECK(total == 8008);  // 5 kadrai 29,97 Hz = 8008 mėginiai tiksliai
    for (Index j = 1; j < 3000; j += 7) {
        CHECK(p2997.last_ready(p2997.end(j)) == j);
        CHECK(p2997.last_ready(p2997.end(j) - 1) == j - 1);
    }
    const std::uint64_t big = 1'000'000'007ull * 1920;  // ilgas TAI: be slankaus kablelio paklaidos
    CHECK(i50.last_ready(big) == 1'000'000'006ull * 2 + 1);
}

TEST_CASE("review3: C fazė — paruoštas WaitC vienetas nelaukia užstrigusio iki C ribos (riba t_r + Wr)", "[engine][review3][phasec]") {
    // D = 30 ms prie 50 Hz: B = T + 17,5, C = T + 30. Indekse kI abu vienetai pernešti per kito tick'o atidarymą (T + 20,5):
    // u0 ateina T + 22 ms, u1 (gyvas, bet užstrigęs) — niekada. Anksčiau u0 laukė iki C = T + 30, dabar — iki 22 + Wr = 27.
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    Rig r(2, c);
    r.ins[0]->arrival = [](Index j) { return j == kI ? std::int64_t{22'000'000} : std::int64_t{1'200'000}; };
    r.ins[1]->arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    r.stop_after(kI + 6);
    r.build();
    r.run();
    const auto* e0 = r.outs[0]->find(kI);
    REQUIRE(e0 != nullptr);
    CHECK(e0->kind == RecordingOutput::Kind::Commit);
    const std::int64_t at = static_cast<std::int64_t>(e0->tai) - T(kI);
    CHECK(at >= 22'000'000);
    CHECK(at <= 27'000'000);  // ≤ t_r + Wr (anksčiau — 30 ms, C riba)
    // u1 indeksas kI — spraga ties C riba (nesulaukta), ne anksčiau
    const auto* e1 = r.outs[1]->find(kI);
    REQUIRE(e1 != nullptr);
    CHECK(e1->kind == RecordingOutput::Kind::Gap);
    CHECK(static_cast<std::int64_t>(e1->tai) - T(kI) >= 30'000'000);
}

TEST_CASE("review3: C fazė — du paruošti per Wr vienas nuo kito — vienas paleidimas", "[engine][review3][phasec]") {
    constexpr Index kI = 1010;
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    Rig r(3, c);
    r.ins[0]->arrival = [](Index j) { return j == kI ? std::int64_t{22'000'000} : std::int64_t{1'200'000}; };
    r.ins[1]->arrival = [](Index j) { return j == kI ? std::int64_t{24'000'000} : std::int64_t{1'200'000}; };
    r.ins[2]->arrival = [](Index j) { return j >= kI ? kNever : std::int64_t{1'200'000}; };
    r.stop_after(kI + 6);
    r.build();
    r.run();
    const auto calls = r.fn.calls_for(kI);
    REQUIRE(calls.size() == 1);  // A ir B — tušti; vienas C su u0 ir u1
    CHECK(calls[0]->phase == 2);
    CHECK(calls[0]->units.size() == 2);
    CHECK(static_cast<std::int64_t>(calls[0]->tai) - T(kI) <= 27'000'000);
}

TEST_CASE("review3: Blocked vieneto gyvo brolio stebėjimas — Gone atjungia įėjimą", "[engine][review3][observe]") {
    // composite forma: 2 būtini įėjimai; in1 niekada neprisijungia (vienetas Blocked), in0 gyvas — jo grain'as tik stebimas
    // (observe). Nuo T(1010) in0 flow'as „atkurtas“ (probe → Gone): anksčiau stebėjimas tai ignoruodavo (in0 likdavo Live).
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub;
    std::atomic<bool> stop{false};
    ScriptedInput a(clk), b(clk);
    RecordingOutput out(clk);
    RecordingFunction fn(clk);
    b.attachable = [](std::uint64_t) { return false; };
    a.gone = [](std::uint64_t now) { return static_cast<std::int64_t>(now) >= T(1010); };
    a.attachable = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(1010); };
    InState at_end = InState::Waiting;
    fn.on_end_tick = [&](Index i, TickContext&) {
        if (i >= 1015) stop = true;
    };
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec s;
    s.name = "comp";
    s.in.push_back(InputSpec{&a, {}, 1, Need::Required, InMode::Exact});
    s.in.push_back(InputSpec{&b, {}, 1, Need::Required, InMode::Exact});
    s.out.push_back(&out);
    std::uint32_t id = 0;
    std::string why;
    REQUIRE(eng.add_unit(s, id, why));
    eng.run(fn);
    at_end = eng.in_state(0, 0);
    CHECK(at_end != InState::Live);
    CHECK(hub.in(eng.in_id(0, 0)).detaches >= 1);
}

TEST_CASE("review3: įėjimas grįžo kitu formatu (neprisijungia, bet mismatch) — FormatChange, exit 6", "[engine][review3][mismatch]") {
    // MxlVideoInput: InputSlot Mismatch būsenoje attach_poll grąžina false, mismatch() — true. Anksčiau variklis mismatch
    // tikrino tik po sėkmingo prisijungimo — exit 6 neįvykdavo, vienetas laukė be galo (missing_fields).
    Rig r(1);
    auto& k = *r.ins[0];
    k.alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(1010); };
    k.attachable = k.alive;
    const auto base = k.arrival;
    k.arrival = [base](Index j) { return j >= 1010 ? kNever : base(j); };
    r.fn.on_end_tick = [&](Index i, TickContext&) {
        if (i == 1015) k.mismatch_flag = true;  // gamintojas grįžo kitu formatu
        if (i >= 1100) r.stop = true;
    };
    r.build();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::FormatChange);
    CHECK(res.exit_code == 6);
    CHECK(res.last_index < 1030);
}
