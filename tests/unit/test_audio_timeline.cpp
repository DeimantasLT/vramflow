// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// timing/audio_timeline.hpp: AAC PTS iš kadro indekso, spragos — tyla arba perinkaravimas, PTS niekada atgal.
// Modelis vietoje FIFO: mėginio „vertė“ = jo indeksas (tyla = −1), todėl tikrinama, kad kiekvienas mėginys gauna savo PTS.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

#include "common/format.hpp"
#include "timing/audio_timeline.hpp"

using mxlv::timing::AudioPlan;
using mxlv::timing::AudioTimeline;

namespace {

// Koderio darbininko modelis: FIFO su mėginių indeksais, išstumti kadrai — (PTS, mėginiai).
struct Sim {
    AudioTimeline tl;
    std::vector<std::int64_t> fifo;
    std::int64_t last_pts = -1;
    std::uint64_t frames = 0, mismatches = 0, backwards = 0;

    explicit Sim(std::int64_t max_silence = 48000) : tl(1024, max_silence) {}

    void emit(std::int64_t pts) {
        if (last_pts >= 0 && pts < last_pts + 1024) ++backwards;
        last_pts = pts;
        for (std::int64_t k = 0; k < 1024; ++k)
            if (fifo[static_cast<std::size_t>(k)] >= 0 && fifo[static_cast<std::size_t>(k)] != pts + k) ++mismatches;
        fifo.erase(fifo.begin(), fifo.begin() + 1024);
        ++frames;
    }
    AudioPlan push(std::int64_t begin, std::int64_t count) {
        const AudioPlan p = tl.submit(begin, count);
        if (p.flush_frames) {
            fifo.insert(fifo.end(), static_cast<std::size_t>(p.flush_pad), -1);
            for (std::int64_t k = 0; k < p.flush_frames; ++k) emit(p.flush_pts + k * 1024);
        }
        fifo.insert(fifo.end(), static_cast<std::size_t>(p.silence), -1);
        for (std::int64_t s = begin + p.skip; s < begin + count; ++s) fifo.push_back(s);
        while (tl.ready()) emit(tl.take());
        return p;
    }
    // kadro F mėginiai pagal formatą
    AudioPlan frame(vramflow::Rational rate, std::uint64_t f) {
        const auto b = static_cast<std::int64_t>(vramflow::sample_begin(rate, f));
        const auto e = static_cast<std::int64_t>(vramflow::sample_begin(rate, f + 1));
        return push(b, e - b);
    }
    bool aligned(vramflow::Rational rate, std::uint64_t next_f) const {
        return tl.head() + tl.queued() == static_cast<std::int64_t>(vramflow::sample_begin(rate, next_f)) &&
               static_cast<std::int64_t>(fifo.size()) == tl.queued();
    }
};

constexpr vramflow::Rational k25{25, 1}, k50{50, 1}, k2997{30000, 1001}, k5994{60000, 1001};
constexpr std::uint64_t kF0 = 44'000'000'000ull;  // ~ šiandienos TAI kadro indeksas 25 Hz (dideli skaičiai — be perpildymo)

}  // namespace

TEST_CASE("audio timeline: tolydūs kadrai — PTS = sample_begin, be tylos", "[timing][audio]") {
    for (const auto rate : {k25, k50, k2997, k5994}) {
        Sim s;
        for (std::uint64_t f = kF0; f < kF0 + 500; ++f) s.frame(rate, f);
        INFO("rate " << rate.num << "/" << rate.den);
        CHECK(s.frames > 0);
        CHECK(s.mismatches == 0);
        CHECK(s.backwards == 0);
        CHECK(s.tl.silence_samples() == 0);
        CHECK(s.tl.reanchors() == 0);
        CHECK(s.aligned(rate, kF0 + 500));
        CHECK(s.tl.head() >= static_cast<std::int64_t>(vramflow::sample_begin(rate, kF0)));
    }
}

TEST_CASE("audio timeline: pirmo kadro inkaras — sample_begin(F)", "[timing][audio]") {
    Sim s;
    s.frame(k25, kF0 + 7);  // 1920 mėginių → vienas kadras išstumtas
    REQUIRE(s.frames == 1);
    CHECK(s.last_pts == static_cast<std::int64_t>(vramflow::sample_begin(k25, kF0 + 7)));
}

TEST_CASE("audio timeline: atmestas kadras — tyla tiksliai spragai, toliau PTS teisingas", "[timing][audio]") {
    // radinys 2: senas skaitiklis po kiekvieno atmesto kadro vėluodavo 1920 mėginių (40 ms) visam laikui
    Sim s;
    for (std::uint64_t f = kF0; f < kF0 + 10; ++f) s.frame(k25, f);
    const AudioPlan p = s.frame(k25, kF0 + 11);  // kF0 + 10 atmestas
    CHECK(p.silence == 1920);
    CHECK(p.flush_frames == 0);
    CHECK(p.skip == 0);
    for (std::uint64_t f = kF0 + 12; f < kF0 + 100; ++f) s.frame(k25, f);
    CHECK(s.aligned(k25, kF0 + 100));
    CHECK(s.mismatches == 0);
    CHECK(s.backwards == 0);
    CHECK(s.tl.silence_samples() == 1920);
}

TEST_CASE("audio timeline: atsitiktiniai atmetimai visais formatais — kiekvienas mėginys savo vietoje", "[timing][audio]") {
    std::mt19937_64 rng(20261111);
    for (const auto rate : {k25, k50, k2997, k5994}) {
        Sim s;
        std::uint64_t f = kF0, sil = 0;
        for (int i = 0; i < 5000; ++i) {
            const std::uint64_t gap = rng() % 10 == 0 ? 1 + rng() % 5 : 0;  // ~10 % — 1..5 atmestų kadrų
            const std::uint64_t prev_end = vramflow::sample_begin(rate, f);
            f += gap;
            sil += vramflow::sample_begin(rate, f) - prev_end;
            s.frame(rate, f);
            ++f;
        }
        INFO("rate " << rate.num << "/" << rate.den);
        CHECK(s.mismatches == 0);
        CHECK(s.backwards == 0);
        CHECK(s.tl.reanchors() == 0);  // 5 kadrai < 1 s
        CHECK(s.tl.silence_samples() == sil);
        CHECK(s.aligned(rate, f));
    }
}

TEST_CASE("audio timeline: ilga spraga — likutis išstumiamas su senu PTS, tada perinkaravimas", "[timing][audio]") {
    Sim s(48000);
    s.frame(k25, kF0);  // 1920 → 1 kadras, likutis 896
    REQUIRE(s.tl.queued() == 896);
    const std::int64_t old_head = s.tl.head();
    const AudioPlan p = s.frame(k25, kF0 + 1 + 33);  // 1,32 s laikrodžio šuolis (> 1 s)
    CHECK(p.flush_frames == 1);
    CHECK(p.flush_pad == 1024 - 896);
    CHECK(p.flush_pts == old_head);
    CHECK(p.silence == 0);
    CHECK(s.tl.reanchors() == 1);
    // likutis (896 + 128 tylos) išstumtas su senu PTS, o pirmas naujas kadras prasideda tiksliai sample_begin(kF0 + 34)
    CHECK(s.frames == 3);
    CHECK(s.last_pts == static_cast<std::int64_t>(vramflow::sample_begin(k25, kF0 + 34)));
    s.frame(k25, kF0 + 35);
    CHECK(s.mismatches == 0);
    CHECK(s.backwards == 0);
    CHECK(s.aligned(k25, kF0 + 36));
}

TEST_CASE("audio timeline: spraga lygi max_silence — dar tyla; +1 — perinkaravimas", "[timing][audio]") {
    {
        AudioTimeline t(1024, 4800);
        t.submit(0, 1000);
        const AudioPlan p = t.submit(1000 + 4800, 100);
        CHECK(p.silence == 4800);
        CHECK(t.reanchors() == 0);
    }
    {
        AudioTimeline t(1024, 4800);
        t.submit(0, 1000);
        const AudioPlan p = t.submit(1000 + 4801, 100);
        CHECK(p.silence == 0);
        CHECK(p.flush_frames == 1);
        CHECK(p.flush_pad == 24);
        CHECK(t.reanchors() == 1);
        CHECK(t.head() == 5801);
        CHECK(t.queued() == 100);
    }
    {  // max_silence < kadro dydžio — pakeliamas iki kadro (išstumtas likutis negali užlipti ant naujo segmento)
        AudioTimeline t(1024, 10);
        t.submit(0, 1000);
        const AudioPlan p = t.submit(1000 + 500, 10);
        CHECK(p.silence == 500);
    }
}

TEST_CASE("audio timeline: persidengimas — praeities mėginiai išmetami, PTS neina atgal", "[timing][audio]") {
    Sim s;
    for (std::uint64_t f = kF0; f < kF0 + 5; ++f) s.frame(k25, f);
    const AudioPlan p = s.frame(k25, kF0 + 3);  // tas pats kadras dar kartą (laikrodis atgal)
    CHECK(p.skip == 1920);
    const AudioPlan q = s.push(static_cast<std::int64_t>(vramflow::sample_begin(k25, kF0 + 5)) - 100, 1920 + 100);  // dalinis
    CHECK(q.skip == 100);
    for (std::uint64_t f = kF0 + 6; f < kF0 + 20; ++f) s.frame(k25, f);
    CHECK(s.mismatches == 0);
    CHECK(s.backwards == 0);
    CHECK(s.aligned(k25, kF0 + 20));
    CHECK(s.tl.skipped_samples() == 2020);
}

TEST_CASE("audio timeline: du srautai vienoje gijoje — būsena nepriklausoma", "[timing][audio]") {
    // senas `static thread_local` skaitiklis buvo bendras visiems vienos gijos srautams
    Sim a, b;
    std::uint64_t b_next = 0;
    for (std::uint64_t f = kF0; f < kF0 + 50; ++f) {
        a.frame(k25, f);
        if (f % 7) {
            b.frame(k25, f + 1000);
            b_next = f + 1001;
        }
    }
    CHECK(a.mismatches == 0);
    CHECK(b.mismatches == 0);
    CHECK(a.aligned(k25, kF0 + 50));
    CHECK(b.aligned(k25, b_next));
    CHECK(a.tl.silence_samples() == 0);
    CHECK(b.tl.silence_samples() > 0);
}
