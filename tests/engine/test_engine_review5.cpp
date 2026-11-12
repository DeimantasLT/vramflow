// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Penktos peržiūros testai (2026-11-12): izoliacija, kai sugedęs vienetas — PIRMAS eilėje (A1); vėlesni — žr. žemiau.
#include <catch2/catch_test_macros.hpp>

#include <map>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
// enc forma (A1): Isolation ribos su D = 30 ms (enc --deadline-ms numatyta): a = T + 5,5, b = T + 17,5, c = T + 30 ms.
EngineConfig enc_like() {
    EngineConfig c = Rig::test_config();
    c.cuts.deadline_ns = 30'000'000;
    return c;
}
}  // namespace

// A1 (fiso_enc[C]: po gedimo tik sveiki flow'ai PO sugedusio eilėje vėlavo ~11 ms): variklio pusė — sugedusio vieneto vieta
// eilėje sveikų commit'o laiko nekeičia. Paveiktas — PIRMAS (u0, nužudomas ties kK), sveiki — du aukštupiai: u1…u3 atvyksta
// T + 11,9 ms (B fazė), u4…u7 — T + 19 ms (po b = 17,5 ms, C fazė). Tikslūs laikai kiekvienam srautui prieš ir po gedimo.
TEST_CASE("review5: A1 — sugedęs pirmas; sveiki B / C fazėse commit'ina savo paruošimu, prieš ir po gedimo vienodai",
          "[engine][review5][isolation]") {
    constexpr Index kK = 1040;
    constexpr std::int64_t kB = 11'900'000, kC = 19'000'000, kV = 1'200'000;
    Rig r(8, enc_like());
    r.ins[0]->arrival = [](Index j) { return j >= kK ? kNever : kV; };
    r.ins[0]->alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kK); };
    for (std::uint32_t u = 1; u < 8; ++u) r.ins[u]->arrival = [u](Index) { return u <= 3 ? kB : kC; };
    r.stop_after(kK + 60);
    r.build();
    const RunResult res = r.run();
    REQUIRE(res.reason == EndReason::Stopped);
    for (std::uint32_t u = 1; u < 8; ++u) {
        INFO("u" << u);
        const std::int64_t want = u <= 3 ? kB : kC;
        for (Index i = kK - 20; i <= kK + 60; ++i) {
            if (i == kK) continue;  // vienkartinis laikymas iki a(i), kai įėjimas miršta (≤ Wr + ro) — tikrinamas atskirai
            INFO("indeksas " << i);
            const auto* e = r.outs[u]->find(i);
            REQUIRE(e != nullptr);
            CHECK(e->kind == RecordingOutput::Kind::Commit);
            CHECK(static_cast<std::int64_t>(e->tai) - T(i) == want);
        }
        const auto* e = r.outs[u]->find(kK);
        REQUIRE(e != nullptr);
        CHECK(e->kind == RecordingOutput::Kind::Commit);
        CHECK(static_cast<std::int64_t>(e->tai) - T(kK) == want);  // vėluojantys sveiki ir taip laukė ilgiau nei a(i)
        CHECK(r.out(u).commit_errors == 0);
        CHECK(r.out(u).late_skips == 0);
    }
    CHECK(r.outs[0]->find(kK + 10) != nullptr);  // paveikto spraga užpildyta pagal politiką (Invalid)
    CHECK(r.outs[0]->find(kK + 10)->kind == RecordingOutput::Kind::Gap);
}

// --- B5: TickContext::fail_unit iš process() / end_unit — vienetas užbaigiamas LYGIAI vieną kartą -------------------------
// Anksčiau fail_unit užbaigdavo vykdomą vienetą (end_unit + resolve), o tada jį užbaigdavo ir kviečiantysis: pending
// persivyniodavo (--duration išleidimas pakibdavo — virtualus laikrodis „nušokdavo“ į begalybę), iš process() — rašytojas
// atlaisvinamas su atidarytu grain'u. Tikrinamos visos trys kvietimo vietos.
namespace {
constexpr Index kF = 1010;
void check_fail_once(Rig& r, Outcome want) {
    const RunResult res = r.run();
    INFO("priežastis " << static_cast<int>(res.reason));
    CHECK(res.reason == EndReason::Duration);
    CHECK(res.failed_units == 1);
    // išleidimas baigėsi laiku (anksčiau — laukimas iki INT64_MAX)
    CHECK(static_cast<std::int64_t>(r.clk.now_tai()) < T(kF + 200));
    std::map<Index, int> n0;
    Outcome got = Outcome::Ready;
    for (const auto& e : r.fn.ends)
        if (e.unit == 0) {
            ++n0[e.index];
            if (e.index == kF) got = e.outcome;
        }
    for (const auto& [i, n] : n0) {
        INFO("u0 indeksas " << i);
        CHECK(n == 1);  // kiekvienas indeksas — lygiai vienas end_unit
        CHECK(i <= kF);  // po gedimo nebedalyvauja
    }
    CHECK(n0.count(kF) == 1);
    CHECK(got == want);
    CHECK(r.outs[0]->released);
    CHECK_FALSE(r.outs[0]->released_open);
    CHECK(r.outs[0]->opens == r.outs[0]->commits + r.outs[0]->cancels + r.outs[0]->failed);
    const UnitCounters& c = r.hub.unit(r.eng->unit_id(0));
    CHECK(c.ok + c.late + c.gaps + c.startup == static_cast<std::uint64_t>(n0.size()));  // apskaita — vieną kartą
    for (Index i = kF; i <= kF + 100; ++i) CHECK(r.outs[1]->find(i) != nullptr);  // kitas vienetas tęsia
}
}  // namespace

TEST_CASE("review5: B5 — fail_unit iš process() (vienetas batch'e)", "[engine][review5][service]") {
    EngineConfig c = Rig::test_config();
    c.duration_s = 3;
    Rig r(2, c);
    r.fn.on_process = [](const Batch& b, TickContext& ctx) {
        if (b.index != kF) return;
        for (const UnitTick& u : b.units)
            if (u.unit == 0) ctx.fail_unit(0, "process");
    };
    r.build();
    check_fail_once(r, Outcome::OutputError);
    CHECK(r.outs[0]->find(kF) == nullptr);  // slot'as atšauktas, ne commit'intas
}

TEST_CASE("review5: B5 — fail_unit iš end_unit po commit'o", "[engine][review5][service]") {
    EngineConfig c = Rig::test_config();
    c.duration_s = 3;
    Rig r(2, c);
    r.fn.on_end_unit = [](const UnitTick& u, Outcome o, TickContext& ctx) {
        if (u.unit == 0 && u.index == kF && o == Outcome::Ready) ctx.fail_unit(0, "end_unit");
    };
    r.build();
    check_fail_once(r, Outcome::Ready);
    CHECK(r.outs[0]->find(kF) != nullptr);  // commit'as įvyko prieš gedimą
}

TEST_CASE("review5: B5 — fail_unit iš end_unit spragos kelyje", "[engine][review5][service]") {
    EngineConfig c = Rig::test_config();
    c.duration_s = 3;
    Rig r(2, c);
    r.ins[0]->arrival = [](Index j) { return j == kF ? kNever : std::int64_t{1'200'000}; };  // vienas grain'as neateina
    r.fn.on_end_unit = [](const UnitTick& u, Outcome o, TickContext& ctx) {
        if (u.unit == 0 && u.index == kF && o != Outcome::Ready) ctx.fail_unit(0, "gap");
    };
    r.build();
    check_fail_once(r, Outcome::GapLate);
}
