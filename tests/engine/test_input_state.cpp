// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V2: įėjimo būsenų automatas — po vieną atvejį kiekvienai §6 D2 perėjimų lentelės eilutei.
#include <catch2/catch_test_macros.hpp>

#include "engine/input_state.hpp"

using namespace mxlv::engine;

namespace {
InputState live(std::uint32_t m = 3) {  // prijungtas ir jau įrodęs, kad ateina laiku (8 iš eilės)
    InputState s(m, 8);
    InCounters c{};
    s.on_attach(c);
    for (int k = 0; k < 8; ++k) s.on_ready(c, true);
    return s;
}
}  // namespace

TEST_CASE("input_state: Waiting → Live + Late (prijungtas)", "[engine][input_state]") {
    InputState s(3);
    CHECK(s.state() == InState::Waiting);
    InCounters c{};
    s.on_attach(c);
    CHECK(s.state() == InState::Live);
    CHECK(s.late());  // naujai prijungtas kvorumo nelaiko
    CHECK(c.reattached == 0);  // tik iš Detached
}

TEST_CASE("input_state: Detached → Live + Late, reattached++", "[engine][input_state]") {
    InputState s = live();
    InCounters c{};
    s.on_unready_at_a(false, c);
    REQUIRE(s.state() == InState::Detached);
    s.on_attach(c);
    CHECK(s.state() == InState::Live);
    CHECK(s.late());
    CHECK(c.reattached == 1);
}

TEST_CASE("input_state: Waiting / Detached → Mismatch", "[engine][input_state]") {
    InputState s(3);
    s.on_mismatch();
    CHECK(s.state() == InState::Mismatch);
    InputState d = live();
    InCounters c{};
    d.on_unready_at_a(false, c);
    d.on_mismatch();
    CHECK(d.state() == InState::Mismatch);
}

TEST_CASE("input_state: naujai prijungtas — Late nuimama po pirmo laiku atėjusio", "[engine][input_state]") {
    InputState s(3, 8);
    InCounters c{};
    s.on_attach(c);
    REQUIRE(s.late());
    s.on_ready(c, true);
    CHECK_FALSE(s.late());  // bandomasis laikotarpis — vienas grain'as
    CHECK(c.ok == 1);
}

TEST_CASE("input_state: po vėlavimo Late nuimama po 8 laiku iš eilės", "[engine][input_state]") {
    InputState s = live();
    InCounters c{};
    s.on_unready_at_a(true, c);
    REQUIRE(s.late());
    for (int k = 0; k < 7; ++k) s.on_ready(c, true);
    CHECK(s.late());  // 7 — dar ne
    CHECK(s.on_time_run() == 7);
    s.on_ready(c, true);
    CHECK(s.state() == InState::Live);
    CHECK_FALSE(s.late());
    CHECK(s.miss_run() == 0);
    CHECK(c.ok == 8);
    CHECK(c.late == 0);
}

TEST_CASE("input_state: plazdėjimas — vienas vėlus grain'as nuline seriją", "[engine][input_state]") {
    InputState s(3, 8);
    InCounters c{};
    s.on_attach(c);
    for (int k = 0; k < 50; ++k) s.on_ready(c, k % 2 == 0);  // laiku / vėlai pakaitomis
    CHECK(s.late());
    CHECK(c.ok == 25);
    CHECK(c.late == 25);
    InputState q = live();
    q.on_ready(c, false);  // vienas vėlus — vėl Late
    CHECK(q.late());
    CHECK(q.on_time_run() == 0);
}

TEST_CASE("input_state: Live neparuoštas ties A, rašytojas negyvas → Detached", "[engine][input_state]") {
    InputState s = live();
    InCounters c{};
    s.on_unready_at_a(false, c);
    CHECK(s.state() == InState::Detached);
    CHECK(c.detaches == 1);
    CHECK(c.lost == 1);
}

TEST_CASE("input_state: Live neparuoštas ties A, rašytojas gyvas → Live + Late", "[engine][input_state]") {
    InputState s = live();
    InCounters c{};
    s.on_unready_at_a(true, c);
    CHECK(s.state() == InState::Live);
    CHECK(s.late());
    CHECK(c.lost == 0);
    CHECK(c.detaches == 0);
}

TEST_CASE("input_state: Live + Late paruoštas po a(i), iki paskutinės ribos → late++", "[engine][input_state]") {
    InputState s = live();
    InCounters c{};
    s.on_unready_at_a(true, c);
    s.on_missed(c);
    REQUIRE(s.miss_run() == 1);
    s.on_ready(c, false);
    CHECK(s.state() == InState::Live);
    CHECK(s.late());
    CHECK(s.miss_run() == 0);
    CHECK(c.late == 1);
}

TEST_CASE("input_state: neparuoštas iki paskutinės ribos → lost++; M = 3 → Stalled", "[engine][input_state]") {
    InputState s = live(3);
    InCounters c{};
    s.on_unready_at_a(true, c);
    CHECK_FALSE(s.on_missed(c));
    CHECK_FALSE(s.on_missed(c));
    CHECK(s.state() == InState::Live);
    CHECK(s.on_missed(c));  // trečias iš eilės
    CHECK(s.state() == InState::Stalled);
    CHECK(c.lost == 3);
    CHECK(c.stalls == 1);
}

TEST_CASE("input_state: du praleidimai + vienas sėkmingas → miss_run == 0", "[engine][input_state]") {
    InputState s = live(3);
    InCounters c{};
    s.on_unready_at_a(true, c);
    s.on_missed(c);
    s.on_missed(c);
    s.on_ready(c, false);
    CHECK(s.miss_run() == 0);
    s.on_missed(c);
    s.on_missed(c);
    CHECK(s.state() == InState::Live);  // vėl tik 2 iš eilės
    CHECK(c.stalls == 0);
}

TEST_CASE("input_state: get → FLOW_INVALID → Detached", "[engine][input_state]") {
    InputState s = live();
    InCounters c{};
    s.on_gone(c);
    CHECK(s.state() == InState::Detached);
    CHECK(c.detaches == 1);
}

TEST_CASE("input_state: TOO_LATE ir OK + INVALID → Live, skaitikliai, miss_run = 0", "[engine][input_state]") {
    InputState s = live();
    InCounters c{};
    s.on_too_late(c);
    s.on_invalid_grain(c);
    CHECK(s.state() == InState::Live);
    CHECK(c.too_late == 1);
    CHECK(c.invalid == 1);
    // Atėjęs INVALID / TOO_LATE grain'as — rašytojas gamina: ne praleidimas ir nutraukia praleidimų seriją
    s.on_missed(c);
    s.on_missed(c);
    s.on_invalid_grain(c);
    CHECK(s.miss_run() == 0);
    CHECK_FALSE(s.on_missed(c));
    s.on_missed(c);
    s.on_too_late(c);
    CHECK(s.miss_run() == 0);
    CHECK(s.state() == InState::Live);
    CHECK(c.lost == 4);
    CHECK(c.stalls == 0);
}

TEST_CASE("input_state: Stalled, head padidėjo → Live + Late", "[engine][input_state]") {
    InputState s = live(3);
    InCounters c{};
    s.on_unready_at_a(true, c);
    for (int k = 0; k < 3; ++k) s.on_missed(c);
    REQUIRE(s.state() == InState::Stalled);
    CHECK_FALSE(s.on_stalled_head(100));  // pirmas skaitymas — tik atskaita
    CHECK_FALSE(s.on_stalled_head(100));
    CHECK(s.state() == InState::Stalled);
    CHECK(s.on_stalled_head(101));
    CHECK(s.state() == InState::Live);
    CHECK(s.late());
    CHECK(s.miss_run() == 0);
}

TEST_CASE("input_state: Stalled, rašytojas negyvas → Detached", "[engine][input_state]") {
    InputState s = live(3);
    InCounters c{};
    s.on_unready_at_a(true, c);
    for (int k = 0; k < 3; ++k) s.on_missed(c);
    REQUIRE(s.state() == InState::Stalled);
    s.on_stalled_dead(c);
    CHECK(s.state() == InState::Detached);
    CHECK(c.detaches == 1);
}

TEST_CASE("input_state: set_enabled ir retarget", "[engine][input_state]") {
    InputState s = live();
    s.set_enabled(false);
    CHECK(s.state() == InState::Disabled);
    s.set_enabled(true);
    CHECK(s.state() == InState::Waiting);
    InCounters c{};
    s.on_attach(c);
    s.on_retarget();
    CHECK(s.state() == InState::Waiting);
    s.on_attach(c);
    CHECK(c.reattached == 0);  // retarget reattached nedidina
}

TEST_CASE("input_state: M = 0 — be Late žymių ir be pertraukiklio (legacy)", "[engine][input_state]") {
    InputState s(0);
    InCounters c{};
    s.on_attach(c);
    CHECK_FALSE(s.late());
    s.on_unready_at_a(true, c);
    CHECK_FALSE(s.late());
    for (int k = 0; k < 10; ++k) CHECK_FALSE(s.on_missed(c));
    CHECK(s.state() == InState::Live);
    CHECK(c.lost == 10);
}
