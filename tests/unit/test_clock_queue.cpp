// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// dec atkūrimo eilės perpildymų priskyrimas laikrodžiui (timing::ClockQueueBudget): tik tai, ką paaiškina pats šuolis atgal,
// vieną kartą; pirmyn — nieko. Modelis: 25 fps (kadras 40 ms), gylis 6, riba gylis + 25 (1 s), darbininkas kadrus pateikia
// realiu laiku (po vieną kadrui), tick'as ima po vieną.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "timing/clock_queue.hpp"

using mxlv::timing::ClockQueueBudget;

namespace {
constexpr std::int64_t kFrame = 40'000'000;
constexpr std::uint64_t kDepth = 6, kCap = 6 + 25;

struct Model {
    ClockQueueBudget b{kFrame};
    std::uint64_t q = kDepth - 1;  // eilė po atkurto kadro (kadro tick'o metu — gylis, kaip dec atkūrimo režime)
    std::uint64_t clock = 0, overrun = 0, worker_drops = 0, worker_clock = 0, since = 0;
    // n kadro tick'ų: prieš kiekvieną darbininkas prideda add kadrų (eilė apribota 60 — apsauginiai išmetimai), tick'as ima vieną
    void ticks(int n, std::uint64_t add = 1) {
        for (int k = 0; k < n; ++k) {
            q += add;
            if (q > 60) {
                since += q - 60;
                worker_drops += q - 60;
                q = 60;
            }
            const auto t = b.on_frame_tick(q, kDepth, kCap, since);
            since = 0;
            clock += t.clock_drop;
            overrun += t.overrun_drop;
            worker_clock += t.reclassify;
            worker_drops -= t.reclassify;
            q -= t.clock_drop + t.overrun_drop;
            if (q > 0) --q;  // atkurtas kadras
        }
    }
    // Tick'as laukia wait_ns realaus laiko (laikrodis atgal): darbininkas tuo metu prideda wait / kadras kadrų.
    void wait(std::int64_t wait_ns) {
        q += static_cast<std::uint64_t>(wait_ns / kFrame);
        if (q > 60) {
            since += q - 60;
            worker_drops += q - 60;
            q = 60;
        }
    }
};
}  // namespace

TEST_CASE("clock_queue: žingsnis atgal −1,3 s — paaiškinti kadrai išmetami vieną kartą, vėlesnis pliūpsnis ne perpildymas",
          "[timing][clock_queue]") {
    Model m;
    m.ticks(100);
    m.b.on_step(-1'300'000'000);
    CHECK(m.b.pending() == 33);  // ceil(1300 / 40)
    m.wait(1'300'000'000);       // 32 kadrai susikaupė laukiant
    m.ticks(1);
    CHECK(m.clock == 32);        // visi susikaupę (ne daugiau nei paaiškina žingsnis)
    CHECK(m.overrun == 0);
    CHECK(m.b.pending() == 0);   // vienas kartas
    CHECK(m.q == kDepth - 1);    // eilė grįžo į gylį (be ilgalaikio +1,3 s vėlinimo)
    m.ticks(50);
    // vėlesnis dekoderio pliūpsnis (+24 kadrai vienu metu) telpa į ribą — 0 perpildymų (seni kodas be išmetimo: eilė ties
    // riba, toks pliūpsnis duodavo overrun_frames ir exit 4)
    m.ticks(1, 25);
    m.ticks(30);
    CHECK(m.overrun == 0);
    CHECK(m.clock == 32);
}

TEST_CASE("clock_queue: žingsnis pirmyn +10 ms nepaaiškina nieko — tikras perpildymas lieka perpildymu", "[timing][clock_queue]") {
    Model m;
    m.ticks(20);
    m.b.on_step(+10'000'000);
    CHECK(m.b.pending() == 0);
    m.ticks(1, 40);  // tikra perkrova: eilė virš ribos
    CHECK(m.overrun == kDepth - 1 + 40 - kCap);
    CHECK(m.clock == 0);
}

TEST_CASE("clock_queue: žingsnis −40 ms 1-ą minutę, perpildymas 50-ą — perpildymas (ne laikrodis)", "[timing][clock_queue]") {
    Model m;
    m.ticks(1500);
    m.b.on_step(-40'000'000);
    m.wait(40'000'000);
    m.ticks(1);
    CHECK(m.clock == 1);
    m.ticks(50 * 60 * 25);
    m.ticks(1, 40);
    CHECK(m.overrun == kDepth - 1 + 40 - kCap);
    CHECK(m.clock == 1);
}

TEST_CASE("clock_queue: šuolis didesnis už eilės ribą — darbininko išmetimai perklasifikuojami iš to paties likučio",
          "[timing][clock_queue]") {
    Model m;
    m.ticks(10);
    m.b.on_step(-3'000'000'000);  // 75 kadrai; eilė 5 + 75 > 60 → darbininkas laukiant išmeta 20, tick'e dar 1
    m.wait(3'000'000'000);
    CHECK(m.worker_drops == 20);
    m.ticks(1);
    CHECK(m.clock == 54);         // 60 − gylis 6
    CHECK(m.worker_clock == 21);  // likutis 75 − 54 = 21 ≥ 21
    CHECK(m.worker_drops == 0);
    CHECK(m.overrun == 0);
}

TEST_CASE("clock_queue: du žingsniai atgal prieš kadro tick'ą sumuojami, likutis nebegalioja", "[timing][clock_queue]") {
    ClockQueueBudget b(kFrame);
    b.on_step(-50'000'000);  // 2
    b.on_step(-41'000'000);  // 2
    CHECK(b.pending() == 4);
    const auto t = b.on_frame_tick(kDepth + 1, kDepth, kCap, 0);
    CHECK(t.clock_drop == 1);
    CHECK(b.pending() == 0);
    const auto t2 = b.on_frame_tick(kDepth + 3, kDepth, kCap, 0);
    CHECK(t2.clock_drop == 0);
}
