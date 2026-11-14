// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Garso sekimas mėginio tikslumu (K6): AudioSync kompensacija be trūkių iki max_ppm, perinkaravimas virš hard_samples; AudioRing.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
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
