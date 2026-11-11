// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// SourceClock ir AudioCursor (plano §P1 testai). Numatyta: tb = 1/90000, kadras 3600 tikų = 40 ms, L = 1 s,
// align_rate = 25/1. Viskas sintetika: atvykimo laikas paduodamas parametru.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "timing/index.hpp"
#include "timing/source_clock.hpp"

using namespace mxlv::timing;

namespace {

constexpr Rational kTb{1, 90000};
constexpr Rational k25{25, 1};
constexpr Rational k48k{48000, 1};
constexpr std::int64_t kWrap = std::int64_t{1} << 33;
constexpr std::int64_t kFrame = 3600;  // tikai
constexpr std::int64_t kTf = 40'000'000;
constexpr std::int64_t kL = 1'000'000'000;
constexpr std::int64_t kA0 = 1'700'000'000'123'456'789;

SourceClockConfig cfg25() {
    SourceClockConfig c;
    c.align_rate = k25;
    return c;
}

// Vienas video srautas (0), pts = dts.
Stamp video(SourceClock& sc, std::int64_t raw, std::int64_t arrival) { return sc.observe(0, raw, raw, kFrame, arrival); }

}  // namespace

TEST_CASE("source_clock: fiksavimas ir kvantavimas", "[timing][source_clock]") {
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    CHECK_FALSE(sc.locked());
    const Stamp s = video(sc, 900'000, kA0);
    CHECK(s.ev == Event::Locked);
    CHECK(sc.locked());
    CHECK(s.pts_tai_ns % kTf == 0);
    CHECK(std::abs(s.pts_tai_ns - (kA0 + kL)) <= kTf / 2);
    CHECK(s.dts_tai_ns == s.pts_tai_ns);
    CHECK(sc.offset_ns() == s.pts_tai_ns - 10'000'000'000);  // ns(900 000) = 10 s
}

TEST_CASE("source_clock: wrap", "[timing][source_clock]") {
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    const std::int64_t raw[] = {kWrap - kFrame, 0, kFrame};
    std::vector<Stamp> st;
    for (int k = 0; k < 3; ++k) st.push_back(video(sc, raw[k], kA0 + k * kTf));
    CHECK(st[0].ev == Event::Locked);
    CHECK(st[1].ev == Event::None);
    CHECK(st[2].ev == Event::None);
    CHECK(st[1].dts_tai_ns - st[0].dts_tai_ns == 40'000'000);
    CHECK(st[2].dts_tai_ns - st[1].dts_tai_ns == 40'000'000);
    const ClockStats cs = sc.stats();
    CHECK(cs.wraps == 1);
    CHECK(cs.discontinuities == 0);
}

TEST_CASE("source_clock: PTS ir DTS skirtingose wrap pusėse", "[timing][source_clock]") {
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    const Stamp p = sc.observe(0, kFrame, kWrap - kFrame, kFrame, kA0);
    REQUIRE(p.ev == Event::Locked);
    CHECK(p.pts_tai_ns - p.dts_tai_ns == 80'000'000);
    const Stamp b = sc.observe(0, kWrap - kFrame, 0, kFrame, kA0 + kTf);  // B kadras
    CHECK(b.ev == Event::None);
    CHECK(b.pts_tai_ns - b.dts_tai_ns == -40'000'000);
    CHECK(b.dts_tai_ns - p.dts_tai_ns == 40'000'000);
}

TEST_CASE("source_clock: 90 kHz drebėjimas", "[timing][source_clock]") {
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> jitter(-1, 1);
    std::uint64_t prev = 0;
    int bad = 0;
    for (int k = 0; k < 1000; ++k) {
        const std::int64_t raw = 900'000 + k * kFrame + jitter(rng);
        const Stamp s = video(sc, raw, kA0 + k * kTf);
        REQUIRE(s.ev != Event::Dropped);
        const std::uint64_t idx = index_of(s.pts_tai_ns, k25);
        if (k > 0 && idx != prev + 1) ++bad;
        prev = idx;
    }
    CHECK(bad == 0);
}

TEST_CASE("source_clock: nutrūkimas pirmyn / atgal, tęsinys", "[timing][source_clock]") {
    const std::int64_t jump = GENERATE(std::int64_t{900'000}, std::int64_t{-450'000});
    INFO("DTS šuolis " << jump);
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    Stamp prev{};
    for (int k = 0; k < 200; ++k) {
        const std::int64_t raw = 900'000 + k * kFrame + (k >= 100 ? jump : 0);
        const Stamp s = video(sc, raw, kA0 + k * kTf);  // atvykimas tolydus
        if (k == 100) {
            CHECK(s.ev == Event::Relocked);
            CHECK(s.dts_tai_ns == prev.dts_tai_ns + 40'000'000);
        } else if (k > 0) {
            CHECK(s.ev == Event::None);
            CHECK(s.dts_tai_ns == prev.dts_tai_ns + 40'000'000);
        }
        prev = s;
    }
    const ClockStats cs = sc.stats();
    CHECK(cs.discontinuities == 1);
    CHECK(cs.relock_cont == 1);
    CHECK(cs.relock_fresh == 0);
}

TEST_CASE("source_clock: šaltinis dingo 10 s", "[timing][source_clock]") {
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    for (int k = 0; k < 100; ++k) REQUIRE(video(sc, 900'000 + k * kFrame, kA0 + k * kTf).ev != Event::Dropped);
    const std::int64_t a = kA0 + 100 * kTf + 10'000'000'000;
    const Stamp s = video(sc, 900'000 + 100 * kFrame + 900'000, a);
    CHECK(s.ev == Event::Relocked);
    CHECK(std::abs(s.dts_tai_ns - (a + kL)) <= kTf / 2);
    CHECK(s.pts_tai_ns % kTf == 0);
    const ClockStats cs = sc.stats();
    CHECK(cs.relock_fresh == 1);
    CHECK(cs.relock_cont == 0);
}

TEST_CASE("source_clock: du flow'ai lieka sulygiuoti", "[timing][source_clock]") {
    // Video (0) ir garsas (1) su tuo pačiu PTS: prieš wrap, per jį ir po nutrūkimo (+10 s ties k = 120).
    const bool audio_first = GENERATE(false, true);
    INFO("po nutrūkimo pirmas atkeliauja " << (audio_first ? "garsas" : "video"));
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    sc.add_stream(1, kTb, false);
    const std::int64_t start = kWrap - 50 * kFrame;
    int pairs = 0, bad = 0;
    for (int k = 0; k < 200; ++k) {
        const std::int64_t raw = (start + k * kFrame + (k >= 120 ? 900'000 : 0)) % kWrap;
        const std::int64_t a = kA0 + k * kTf;
        Stamp v{}, au{};
        if (audio_first && k >= 120) {
            au = sc.observe(1, raw, raw, 1920, a);
            v = sc.observe(0, raw, raw, kFrame, a + 1000);
        } else {
            v = sc.observe(0, raw, raw, kFrame, a);
            au = sc.observe(1, raw, raw, 1920, a + 1000);
        }
        if (v.ev == Event::Dropped || au.ev == Event::Dropped) continue;
        ++pairs;
        const std::int64_t tv = time_of(index_of(v.pts_tai_ns, k25), k25);
        const std::int64_t ta = static_cast<std::int64_t>(i128{index_of(au.pts_tai_ns, k48k)} * 1'000'000'000 / 48000);
        if (std::abs(tv - ta) > 10'417) ++bad;
    }
    CHECK(bad == 0);
    CHECK(pairs >= 199);  // tik garsas ties k = 120 „garsas pirmas“ atveju laukia perfiksavimo
    const ClockStats cs = sc.stats();
    CHECK(cs.wraps >= 1);
    CHECK(cs.discontinuities == 1);
    CHECK(cs.relock_cont == 1);
}

TEST_CASE("source_clock: garsas prieš fiksavimą metamas", "[timing][source_clock]") {
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    sc.add_stream(1, kTb, false);
    for (int k = 0; k < 5; ++k) CHECK(sc.observe(1, 900'000 + k * 1920, 900'000 + k * 1920, 1920, kA0 + k * 21'333'333).ev == Event::Dropped);
    CHECK_FALSE(sc.locked());
    CHECK(sc.stats().prelock_dropped == 5);
    CHECK(video(sc, 909'600, kA0 + 5 * 21'333'333).ev == Event::Locked);
    CHECK(sc.observe(1, 909'600, 909'600, 1920, kA0 + 6 * 21'333'333).ev == Event::None);
    CHECK(sc.stats().prelock_dropped == 5);
}

TEST_CASE("source_clock: laikrodžio šuolis", "[timing][source_clock]") {
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    REQUIRE(video(sc, 900'000, kA0).ev == Event::Locked);
    const std::int64_t off0 = sc.offset_ns();
    sc.on_clock_step(-1'290'000'000);
    CHECK(sc.offset_ns() - off0 == -1'280'000'000);  // 32 kadrai
    sc.on_clock_step(+15'000'000);
    CHECK(sc.offset_ns() - off0 == -1'280'000'000);  // < pusė kadro — nekeičiama
    CHECK(sc.stats().clock_steps == 2);
    // Po šuolio žymos tęsiasi tinklelyje, be perfiksavimo.
    const Stamp s = video(sc, 900'000 + kFrame, kA0 + kTf - 1'290'000'000);
    CHECK(s.ev == Event::None);
    CHECK(s.pts_tai_ns % kTf == 0);
}

TEST_CASE("source_clock: dreifas +200 ppm", "[timing][source_clock]") {
    // Atvykimas 200 ppm lėtesnis nei DTS (šaltinio laikrodis greitesnis): m = dts_tai − a auga 0,2 ms/s.
    // Atvirkštinis atvejis (−200 ppm) — simetriškas. 1 val. medijos laiko = 90 000 kadrų.
    const int sign = GENERATE(+1, -1);
    INFO("dreifas " << sign * 200 << " ppm");
    SourceClock sc(cfg25());
    sc.add_stream(0, kTb, true);
    const double rate = 1.0 - sign * 200e-6;  // atvykimo ns vienai medijos ns
    std::uint64_t slipped = 0, relocked = 0;
    std::int64_t worst = 0;
    const std::int64_t off0 = [&] {
        REQUIRE(video(sc, 0, kA0).ev == Event::Locked);
        return sc.offset_ns();
    }();
    for (int k = 1; k < 90'000; ++k) {
        const std::int64_t a = kA0 + static_cast<std::int64_t>(static_cast<double>(k) * kTf * rate);
        const Stamp s = video(sc, k * kFrame, a);
        if (s.ev == Event::Slipped) ++slipped;
        if (s.ev == Event::Relocked) ++relocked;
        worst = std::max(worst, std::abs(s.dts_tai_ns - a - kL));
    }
    const ClockStats cs = sc.stats();
    // Planas sako 17–19 (= 0,72 s / 40 ms), bet pinned algoritmas (9 žingsnis) slysta tik kai m_min išeina už
    // L ± drift_limit (200 ms): pirmi ~200 ms dreifo sugeriami be slydimo → (720 − 200 ± 20) / 40 ≈ 12–14.
    CHECK(cs.drift_slips >= 12);
    CHECK(cs.drift_slips <= 14);
    CHECK(slipped == cs.drift_slips);
    CHECK(relocked == 0);
    CHECK(cs.discontinuities == 0);
    CHECK(worst <= 240'000'000);
    CHECK(sc.offset_ns() - off0 == -sign * static_cast<std::int64_t>(cs.drift_slips) * kTf);
    CHECK(std::abs(cs.drift_ppm - sign * 200.0) <= 1.0);
}

TEST_CASE("audio_cursor: tolydumas", "[timing][audio_cursor]") {
    // AAC kadrai po 1024 mėginius, PTS 90 kHz: 1920 tikų = 21,333… ms (ns žyma nukirsta → apvalinimo paklaida).
    constexpr std::int64_t t0 = 1'700'000'001'000'000'000;
    const std::uint64_t s0 = index_of(t0, k48k);
    AudioCursor c;
    int bad = 0;
    for (std::int64_t k = 0; k < 1000; ++k) {
        const std::int64_t pts = t0 + k * 1920 * 1'000'000'000 / 90000;
        std::int64_t gap = 123;
        const std::uint64_t at = c.place(pts, 1024, 48000, 48, gap);
        if (at != s0 + 1024 * static_cast<std::uint64_t>(k) || gap != 0) ++bad;
    }
    CHECK(bad == 0);
}

TEST_CASE("audio_cursor: spraga ir persidengimas", "[timing][audio_cursor]") {
    constexpr std::int64_t t0 = 1'700'000'001'000'000'000;
    const std::uint64_t s0 = index_of(t0, k48k);
    auto ns_of = [](std::uint64_t s) { return static_cast<std::int64_t>(i128{s} * 1'000'000'000 / 48000); };
    AudioCursor c;
    std::int64_t gap = 0;
    CHECK(c.place(t0, 1024, 48000, 48, gap) == s0);
    CHECK(gap == 0);
    // +100 ms spraga
    const std::uint64_t s1 = s0 + 1024 + 4800;
    CHECK(c.place(ns_of(s0 + 1024) + 100'000'000, 1024, 48000, 48, gap) == s1);
    CHECK(gap == 4800);
    CHECK(c.next == s1 + 1024);
    // −50 ms persidengimas
    const std::uint64_t s2 = s1 + 1024 - 2400;
    CHECK(c.place(ns_of(s1 + 1024) - 50'000'000, 1024, 48000, 48, gap) == s2);
    CHECK(gap == -2400);
    CHECK(c.next == s1 + 1024);  // atgal nejuda: visas paketas persidengė
}

TEST_CASE("source_clock: tik garso šaltinis", "[timing][source_clock]") {
    SourceClock sc(SourceClockConfig{});  // align_rate {0,1}
    sc.add_stream(0, kTb, false);
    const Stamp s = sc.observe(0, 900'000, 900'000, 1920, kA0);
    CHECK(s.ev == Event::Locked);
    CHECK(s.pts_tai_ns == kA0 + kL);
    const Stamp s2 = sc.observe(0, 901'920, 901'920, 1920, kA0 + 21'333'333);
    CHECK(s2.ev == Event::None);
    CHECK(s2.pts_tai_ns == kA0 + kL + 21'333'333);
}
