// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V3: spragų politikos būsena (K3; planas §6 D4, sprendimai K-T4, K-T5, K-T6).
#include <catch2/catch_test_macros.hpp>

#include "engine/gap_state.hpp"
#include "timing/index.hpp"

using namespace mxlv::engine;
using mxlv::timing::time_of;

namespace {
constexpr mxlv::timing::Rational k50{50, 1};
std::int64_t T(Index i) { return time_of(i, k50); }

// Vienas spragos žingsnis: decide + on_gap (kaip variklis), grąžina veiksmą ir šaltinį.
GapAction gap(GapState& g, Index i, bool interlaced, std::uint32_t slots, Index& from) {
    from = kNoIndex;
    const GapAction a = g.decide(i, T(i), interlaced, slots, from);
    g.on_gap(i, a);
    return a;
}
}  // namespace

TEST_CASE("gap_state: invalid → visada Invalid", "[engine][gap]") {
    GapState g;
    g.policy = GapPolicy::Invalid;
    Index from = 0;
    CHECK(gap(g, 10, true, 10, from) == GapAction::Invalid);  // be istorijos
    g.on_content(11, T(11));
    CHECK(gap(g, 12, true, 10, from) == GapAction::Invalid);
    CHECK(gap(g, 500, false, 10, from) == GapAction::Invalid);
}

TEST_CASE("gap_state: leave → None, black → Black", "[engine][gap]") {
    GapState l;
    l.policy = GapPolicy::Leave;
    Index from = 0;
    l.on_content(1, T(1));
    CHECK(gap(l, 2, false, 10, from) == GapAction::None);
    GapState b;
    b.policy = GapPolicy::Black;
    b.on_content(1, T(1));
    CHECK(gap(b, 2, false, 10, from) == GapAction::Black);
}

TEST_CASE("gap_state: freeze, interlaced — to paties pariteto kartojimas, po hold → Black", "[engine][gap]") {
    GapState g;
    g.policy = GapPolicy::Freeze;
    g.hold_ns = 1'000'000'000;
    g.on_content(100, T(100));
    g.on_content(101, T(101));
    Index from = 0;
    CHECK(gap(g, 102, true, 10, from) == GapAction::Repeat);
    CHECK(from == 100);
    CHECK(gap(g, 103, true, 10, from) == GapAction::Repeat);
    CHECK(from == 101);
    CHECK(gap(g, 104, true, 10, from) == GapAction::Repeat);
    CHECK(from == 102);  // kartojama iš ankstesnio kartojimo — tas pats turinys
    for (Index i = 105; i <= 151; ++i) {
        INFO("i = " << i);
        REQUIRE(gap(g, i, true, 10, from) == GapAction::Repeat);
        CHECK(from == i - 2);
    }
    CHECK(T(151) - T(101) == 1'000'000'000);  // dar ne „> hold“
    CHECK(gap(g, 152, true, 10, from) == GapAction::Black);  // T(152) − T(101) = 1,02 s > 1 s
    CHECK(gap(g, 153, true, 10, from) == GapAction::Black);
}

TEST_CASE("gap_state: freeze be istorijos → Black (K-T4)", "[engine][gap]") {
    GapState g;
    g.policy = GapPolicy::Freeze;
    Index from = 0;
    CHECK(gap(g, 7, true, 10, from) == GapAction::Black);
    CHECK(gap(g, 8, false, 10, from) == GapAction::Black);
}

TEST_CASE("gap_state: freeze, šaltinio nebėra ring'e → Black", "[engine][gap]") {
    GapState g;
    g.policy = GapPolicy::Freeze;
    g.on_content(100, T(100));
    Index from = 0;
    // 101…104 — variklio praleisti (K-T6: nepildomi); 105 — spraga, 100 jau per toli (slot_count = 5)
    CHECK(gap(g, 105, false, 5, from) == GapAction::Black);
    GapState h;
    h.policy = GapPolicy::Freeze;
    h.on_content(100, T(100));
    CHECK(gap(h, 101, false, 5, from) == GapAction::Repeat);  // progresyvus: i − 1
    CHECK(from == 100);
}

TEST_CASE("gap_state: Invalid / None neperduoda turinio kartojimui", "[engine][gap]") {
    GapState g;
    g.policy = GapPolicy::Freeze;
    g.on_content(100, T(100));
    g.on_gap(101, GapAction::Invalid);  // pvz. politika pakeista vidury — slot'e 101 nėra galiojančio turinio
    Index from = 0;
    CHECK(gap(g, 102, false, 10, from) == GapAction::Black);
}

TEST_CASE("gap_state: may_resume (K-T5)", "[engine][gap]") {
    GapState g;
    g.policy = GapPolicy::Freeze;
    g.on_content(100, T(100));
    g.on_content(101, T(101));
    CHECK(g.may_resume(103, true));  // spragos nėra — tęsiama
    Index from = 0;
    gap(g, 102, true, 10, from);
    CHECK_FALSE(g.may_resume(103, true));  // nelyginis — tik kadro riba
    CHECK(g.may_resume(104, true));
    CHECK(g.may_resume(103, false));  // progresyvus — iškart
    GapState b;
    b.policy = GapPolicy::Black;
    gap(b, 5, true, 10, from);
    CHECK_FALSE(b.may_resume(7, true));
    CHECK(b.may_resume(8, true));
    GapState v;
    v.policy = GapPolicy::Invalid;
    gap(v, 5, true, 10, from);
    CHECK(v.may_resume(7, true));
    CHECK(v.may_resume(8, true));
}
