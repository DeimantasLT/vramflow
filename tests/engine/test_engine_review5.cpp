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
