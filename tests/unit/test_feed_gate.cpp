// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Dekoderio tiekimo vartai (planas §P3 „feed_gate: Hold / Send ribos“) ir perkrovos praleidimas iki rakto kadro (pamoka 4).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "ingest/feed_gate.hpp"

using namespace mxlv::ingest;

TEST_CASE("feed_gate: Hold / Send ribos", "[ingest][feed_gate]") {
    const FeedGateConfig c{400'000'000, 0};
    constexpr std::int64_t now = 1'000'000'000'000;
    CHECK(feed_decision(now + 400'000'001, now, c) == Feed::Hold);
    CHECK(feed_decision(now + 400'000'000, now, c) == Feed::Send);
    CHECK(feed_decision(now - 1, now, c) == Feed::Send);       // jau praeityje
    CHECK(feed_decision(now - 5'000'000'000, now, c) == Feed::Send);
    const FeedGateConfig c200{200'000'000, 0};
    CHECK(feed_decision(now + 300'000'000, now, c200) == Feed::Hold);
    CHECK(feed_decision(now + 200'000'000, now, c200) == Feed::Send);
}

TEST_CASE("feed_gate: pavėlavęs kadras — praleidžiama iki rakto kadro, kuris dar spėja", "[ingest][feed_gate]") {
    const FeedGateConfig c{400'000'000, 40'000'000};  // kadras turi turėti ≥ 40 ms iki savo laiko
    constexpr std::int64_t now = 1'000'000'000'000;
    SkipToKey g;
    bool flush = false;
    CHECK(g.admit(now + 100'000'000, true, now, c, flush));   // spėja
    CHECK_FALSE(flush);
    CHECK(g.admit(now + 40'000'000, false, now, c, flush));   // lygiai riba — spėja
    CHECK_FALSE(g.admit(now + 39'999'999, false, now, c, flush));  // nebespėja → praleidimas prasideda
    CHECK(g.skipping());
    CHECK(g.skip_runs() == 1);
    CHECK_FALSE(g.admit(now + 500'000'000, false, now, c, flush));  // ne rakto — metama net jei spėtų (be atramos)
    CHECK_FALSE(g.admit(now + 10'000'000, true, now, c, flush));    // rakto, bet nebespėja — metama
    CHECK(g.admit(now + 200'000'000, true, now, c, flush));          // rakto ir spėja — tęsiama, dekoderis išvalomas
    CHECK(flush);
    CHECK_FALSE(g.skipping());
    CHECK(g.admit(now + 240'000'000, false, now, c, flush));
    CHECK_FALSE(flush);
    CHECK(g.dropped() == 3);
    CHECK(g.skip_runs() == 1);
}
