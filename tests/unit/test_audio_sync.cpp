// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Garso sekimas mėginio tikslumu (K6): AudioSync kompensacija be trūkių iki max_ppm, perinkaravimas virš hard_samples; AudioRing.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#include "ingest/audio_sync.hpp"

using namespace mxlv::ingest;

namespace {
// swr emuliacija: kompensacija comp mėginių per distance išvesties — pridedama proporcingai šio kadro daliai.
struct SwrEmu {
    double carry = 0;
    std::int64_t out(std::int64_t in, const AudioSyncStep& s) {
        double n = static_cast<double>(in) + carry;
        if (s.comp != 0 && s.distance > 0) n += static_cast<double>(s.comp) * std::min<double>(1.0, static_cast<double>(in) / static_cast<double>(s.distance));
        const auto k = static_cast<std::int64_t>(std::floor(n));
        carry = n - static_cast<double>(k);
        return k;
    }
};
}  // namespace

TEST_CASE("audio_sync: šaltinis greitesnis / lėtesnis per biudžetą — sekama be perinkaravimo", "[ingest][audio_sync]") {
    const double ppm = GENERATE(-900.0, -200.0, 200.0, 900.0, 5000.0);
    INFO("dreifas " << ppm << " ppm");
    AudioSyncConfig c;
    c.max_ppm = ppm > 1000 ? 6000 : 1000;
    AudioSync s(c);
    SwrEmu swr;
    // Kadras = 1024 įėjimo mėginių; jo PTS TAI (su off_a rampa) išsidėsto 1024 · (1 − ppm·10⁻⁶) išvesties mėginiais.
    std::int64_t worst = 0;
    for (int k = 0; k < 30'000; ++k) {  // ~640 s
        const auto expected = static_cast<std::int64_t>(std::llround(1'000'000.0 + k * 1024.0 * (1.0 - ppm * 1e-6)));
        const AudioSyncStep st = s.on_frame(expected);
        if (k > 0) {
            REQUIRE_FALSE(st.resync);
            if (k > 2000) worst = std::max<std::int64_t>(worst, std::abs(st.err));
        }
        if (st.comp != 0) CHECK(std::abs(static_cast<double>(st.comp)) / static_cast<double>(st.distance) <= c.max_ppm * 1e-6 + 1e-9);
        s.produced(swr.out(1024, st));
    }
    CHECK(s.resyncs() == 0);
    CHECK(worst <= 48);  // nusistovėjus — ≤ 1 ms
}

TEST_CASE("audio_sync: šuolis > hard_samples — perinkaravimas; mažas — kompensacija", "[ingest][audio_sync]") {
    AudioSync s;
    AudioSyncStep st = s.on_frame(48'000);
    CHECK(st.resync);
    CHECK(s.next() == 48'000);
    s.produced(1024);
    st = s.on_frame(48'000 + 1024 + 1);  // 1 mėginys — deadband
    CHECK_FALSE(st.resync);
    CHECK(st.comp == 0);
    s.produced(1024);
    st = s.on_frame(48'000 + 2048 + 100);  // 100 mėginių — kompensacija per ≥ 100 000 mėginių (≤ 0,1 %)
    CHECK(st.comp == 100);
    CHECK(st.distance == 100'000);
    s.produced(1024);
    st = s.on_frame(48'000 + 3072 + 961);  // > 960 — perinkaravimas
    CHECK(st.resync);
    CHECK(s.next() == 48'000 + 3072 + 961);
    CHECK(s.resyncs() == 1);
    s.reset();
    CHECK(s.on_frame(5).resync);  // nauja sesija — perinkaravimas neskaičiuojamas
    CHECK(s.resyncs() == 1);
}

TEST_CASE("audio_ring: pozicijos, tarpas — tyla, vėlavimas, perrašymas", "[ingest][audio_ring]") {
    AudioRing r;
    r.init(2, 1000);  // → 1024
    std::vector<float> a(100, 0.5f), b(100, -0.5f), l(200), rr(200);
    const float* in[2] = {a.data(), b.data()};
    float* out[2] = {l.data(), rr.data()};
    std::uint32_t late = 0;
    CHECK(r.read(0, 10, out, late) == 0);  // dar nieko
    r.write(1000, in, 100);
    CHECK(r.read(1000, 100, out, late) == 100);
    CHECK(late == 0);
    CHECK(l[0] == 0.5f);
    CHECK(rr[99] == -0.5f);
    CHECK(r.read(1050, 100, out, late) == 50);  // 50 dar neparašyta
    CHECK(late == 50);
    CHECK(l[50] == 0.0f);
    r.write(1200, in, 100);  // tarpas [1100, 1200) — tyla, galioja
    CHECK(r.read(1100, 100, out, late) == 100);
    CHECK(l[0] == 0.0f);
    CHECK(r.silence_written() == 100);
    CHECK(r.read(900, 100, out, late) == 0);  // prieš pradžią
    r.write(3000, in, 100);  // > 1024 į priekį — nauja sritis
    CHECK(r.read(1200, 10, out, late) == 0);
    CHECK(r.read(3000, 100, out, late) == 100);
    r.write(2990, in, 20);  // atgal — perrašoma, sritis prasiplečia
    CHECK(r.read(2990, 10, out, late) == 10);
    CHECK(r.head() == 3100);
}

TEST_CASE("audio_ring: B4 — perinkaravimas su perėjimu: be spragtelėjimo (ribotas mėginio šuolis)", "[ingest][audio_ring]") {
    // 1 kHz tonas 0,5: natūralus didžiausias žingsnis 0,5 · 2π · 1000 / 48 000 ≈ 0,065. Šuolis pirmyn (+3000 mėginių, tarpas — tyla)
    // ir atgal (−2000, persidengimas); naujos ašies tono fazė kita. Su write_resync (5 ms perėjimas) žingsnis ≤ 0,08; be jo — ~0,5.
    const std::int64_t jump = GENERATE(3000, -2000);
    INFO("šuolis " << jump << " mėginių");
    auto run = [&](bool fade) {
        AudioRing r;
        r.init(1, 1 << 17);
        std::vector<float> buf(1024);
        const float* in[1] = {buf.data()};
        auto tone = [&](std::int64_t pos, double phase) {
            for (std::size_t k = 0; k < buf.size(); ++k) buf[k] = static_cast<float>(0.5 * std::sin(phase + 2 * std::numbers::pi * 1000.0 * static_cast<double>(pos + static_cast<std::int64_t>(k)) / 48000.0));
        };
        std::int64_t pos = 0;
        for (; pos < 48 * 1024; pos += 1024) {
            tone(pos, 0);
            r.write(pos, in, 1024);
        }
        pos += jump;
        tone(pos, 1.3);  // nauja ašis — kita fazė
        if (fade) r.write_resync(pos, in, 1024, 240);
        else r.write(pos, in, 1024);
        for (int c = 0; c < 8; ++c) {
            pos += 1024;
            tone(pos, 1.3);
            r.write(pos, in, 1024);
        }
        std::vector<float> out(static_cast<std::size_t>(pos - 40 * 1024));
        float* o[1] = {out.data()};
        std::uint32_t late = 0;
        r.read(40 * 1024, static_cast<std::uint32_t>(out.size()), o, late);
        float worst = 0;
        for (std::size_t k = 1; k < out.size(); ++k) worst = std::max(worst, std::fabs(out[k] - out[k - 1]));
        return worst;
    };
    const float with = run(true), without = run(false);
    INFO("didžiausias žingsnis: su perėjimu " << with << ", be jo " << without);
    CHECK(with <= 0.08f);
    CHECK(without > 0.2f);  // patikra, kad bandymas tikrai turi sandūrą
}

TEST_CASE("audio_ring: B4 — senos ašies garsas atmetamas nuo „dabar“ su užgesinimu", "[ingest][audio_ring]") {
    AudioRing r;
    r.init(1, 1 << 15);
    std::vector<float> buf(4800, 0.5f);
    const float* in[1] = {buf.data()};
    r.write(0, in, 4800);
    CHECK(r.truncate(3000, 240) == 1800);
    CHECK(r.head() == 3000);
    std::vector<float> out(3000);
    float* o[1] = {out.data()};
    std::uint32_t late = 0;
    CHECK(r.read(0, 3000, o, late) == 3000);
    CHECK(out[2759] == 0.5f);         // prieš užgesinimą — nepakeista
    CHECK(out[2999] < 0.01f);          // ties riba — ~0
    float worst = 0;
    for (std::size_t k = 1; k < out.size(); ++k) worst = std::max(worst, std::fabs(out[k] - out[k - 1]));
    CHECK(worst <= 0.5f / 240 + 1e-4f);  // tolygus užgesinimas, be šuolio
    CHECK(r.truncate(5000, 240) == 0);  // už head — nieko
}
TEST_CASE("audio_sync: M2 — perfiksavimo žymė: kitas kadras pririšamas tiksliai, ne kompensuojamas", "[ingest][audio_sync]") {
    AudioSync s;
    CHECK(s.on_frame(48000).resync);
    s.produced(1024);
    // be žymės: 500 mėginių (≤ hard_samples 960) — kompensacija, garsas liktų atsilikęs ~10 ms ilgai (1000 ppm — ~10 s)
    {
        AudioSync t = s;
        const AudioSyncStep st = t.on_frame(49024 + 500);
        CHECK_FALSE(st.resync);
        CHECK(st.comp == 500);
    }
    s.realign();
    const AudioSyncStep st = s.on_frame(49024 + 500);
    CHECK(st.resync);
    CHECK(s.next() == 49524);
    CHECK(s.realigns() == 1);
    CHECK(s.resyncs() == 0);  // perfiksavimo perinkaravimas skaičiuojamas atskirai (audio_resyncs — tik paties sekimo)
    s.produced(1024);
    // žymė, bet garsas tolydus (|err| ≤ deadband) — nieko nekeičiama, perėjimo nėra
    s.realign();
    const AudioSyncStep st2 = s.on_frame(50548);
    CHECK_FALSE(st2.resync);
    CHECK(st2.comp == 0);
    CHECK(s.realigns() == 1);
    // žymė vienkartinė
    s.produced(1024);
    CHECK_FALSE(s.on_frame(51572 + 300).resync);
}
