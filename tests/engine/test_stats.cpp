// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V6: statistikos centras — skirtumai intervalui, BCP-008 būsenos, neblokuojantis publish, 0 alokacijų tick'e
// (planas §6 D6, §9 V6).
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "engine/stats.hpp"
#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

// --- Skaičiuojantis alokatorius: skaičiuoja tik gijoje, kurioje g_count == true (variklio gija) ------------------
namespace {
thread_local bool g_count = false;
std::atomic<std::uint64_t> g_allocs{0};
std::vector<int>* g_escape = nullptr;
}  // namespace

void* operator new(std::size_t n) {
    if (g_count) g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n != 0 ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {
// Sink'as, kuris įsimena skirtumus; publish kviečia hub'o gija.
class CollectSink final : public StatsSink {
public:
    void publish(const Snapshot& s) override {
        std::lock_guard<std::mutex> lk(m);
        ok.push_back(s.out.empty() ? 0 : s.out[0].d.ok);
        health.push_back(s.health);
        cv.notify_all();
    }
    bool wait_count(std::size_t n) {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, std::chrono::seconds(5), [&] { return ok.size() >= n; });
    }
    std::mutex m;
    std::condition_variable cv;
    std::vector<std::uint64_t> ok;
    std::vector<Health> health;
};

// Sink'as, kurio publish negrįžta, kol testas neatleidžia.
class StuckSink final : public StatsSink {
public:
    void publish(const Snapshot&) override {
        std::unique_lock<std::mutex> lk(m);
        ++entered;
        cv.notify_all();
        cv.wait(lk, [&] { return released; });
    }
    void release() {
        {
            std::lock_guard<std::mutex> lk(m);
            released = true;
        }
        cv.notify_all();
    }
    std::mutex m;
    std::condition_variable cv;
    bool released = false;
    int entered = 0;
};
constexpr std::uint64_t kSec = 1'000'000'000;
}  // namespace

TEST_CASE("stats: skirtumai, ne sumos", "[engine][stats]") {
    CollectSink sink;
    StatsHub hub(1.0, &sink);
    const std::uint32_t o = hub.add_output("out0");
    hub.add_unit("u0");
    hub.prepare();
    const std::uint64_t t0 = 100 * kSec;
    hub.end_tick(t0);  // intervalo pradžia
    for (int k = 1; k <= 3; ++k) {
        hub.out(o).ok = 50 * static_cast<std::uint64_t>(k);  // kaupiamieji 50, 100, 150
        hub.end_tick(t0 + static_cast<std::uint64_t>(k) * kSec);
        REQUIRE(sink.wait_count(static_cast<std::size_t>(k)));
    }
    hub.stop();
    REQUIRE(sink.ok.size() == 3);
    CHECK(sink.ok[0] == 50);
    CHECK(sink.ok[1] == 50);
    CHECK(sink.ok[2] == 50);
    CHECK(hub.dropped_lines() == 0);
}

TEST_CASE("stats: būsenos", "[engine][stats]") {
    CHECK(std::string(to_string(Health::Healthy)) == "Healthy");
    CHECK(std::string(to_string(Health::PartiallyHealthy)) == "PartiallyHealthy");
    CHECK(std::string(to_string(Health::Unhealthy)) == "Unhealthy");
    CHECK(std::string(to_string(Health::Inactive)) == "Inactive");

    SECTION("įėjimas") {
        InCounters h{};
        h.ok = 5;
        CHECK(input_health(InState::Live, h) == Health::Healthy);
        InCounters p = h;
        p.late = 1;
        CHECK(input_health(InState::Live, p) == Health::PartiallyHealthy);
        p = h;
        p.lost = 1;
        CHECK(input_health(InState::Live, p) == Health::PartiallyHealthy);
        p = h;
        p.invalid = 1;
        CHECK(input_health(InState::Live, p) == Health::PartiallyHealthy);
        for (InState s : {InState::Waiting, InState::Stalled, InState::Detached, InState::Mismatch})
            CHECK(input_health(s, h) == Health::Unhealthy);
        InCounters z{};
        z.lost = 3;
        CHECK(input_health(InState::Live, z) == Health::Unhealthy);  // ok = 0
        CHECK(input_health(InState::Disabled, h) == Health::Inactive);
    }
    SECTION("išėjimas") {
        OutCounters h{};
        h.ok = 50;
        CHECK(output_health(h, true) == Health::Healthy);
        OutCounters l{};
        l.ok = 40;
        l.late = 10;  // B / C fazėje — vis tiek pateikta
        CHECK(output_health(l, true) == Health::Healthy);
        OutCounters g{};
        g.ok = 40;
        g.gap_invalid = 10;
        CHECK(output_health(g, true) == Health::PartiallyHealthy);
        OutCounters f{};
        f.ok = 40;
        f.gap_frozen = 5;
        f.gap_black = 5;  // spragą dengia freeze / black
        CHECK(output_health(f, true) == Health::PartiallyHealthy);
        OutCounters z{};
        z.gap_invalid = 50;
        CHECK(output_health(z, true) == Health::Unhealthy);
        CHECK(output_health(h, false) == Health::Inactive);
    }
    SECTION("procesas") {
        const Health all_ok[] = {Health::Healthy, Health::Healthy};
        const Health mixed[] = {Health::Healthy, Health::Unhealthy};
        const Health all_bad[] = {Health::Unhealthy, Health::Unhealthy};
        const Health none[] = {Health::Inactive};
        CHECK(process_health(all_ok, all_ok) == Health::Healthy);
        CHECK(process_health(all_ok, mixed) == Health::PartiallyHealthy);
        CHECK(process_health(mixed, all_ok) == Health::PartiallyHealthy);
        CHECK(process_health(all_ok, all_bad) == Health::Unhealthy);
        CHECK(process_health({}, {}) == Health::Inactive);
        CHECK(process_health(none, none) == Health::Inactive);
    }
}

TEST_CASE("stats: tick'as neblokuojamas", "[engine][stats]") {
    StuckSink sink;
    {
        StatsHub hub(1.0, &sink);
        const std::uint32_t o = hub.add_output("out0");
        hub.add_unit("u0");
        hub.prepare();
        // Be sieninio laikrodžio: publish užstringa ir laiko vieną kopiją; variklio gija turi baigti visus 1000
        // end_tick (jei laukų — testas pakibtų), o kiekviena kita eilutė suskaičiuojama dropped_lines.
        hub.end_tick(0);  // intervalo pradžia
        hub.out(o).ok += 50;
        hub.end_tick(kSec);  // 1-a eilutė → publikavimo gija įstringa publish'e
        {
            std::unique_lock<std::mutex> lk(sink.m);
            REQUIRE(sink.cv.wait_for(lk, std::chrono::seconds(5), [&] { return sink.entered == 1; }));
        }
        for (int k = 2; k <= 1000; ++k) {
            hub.out(o).ok += 50;
            hub.end_tick(static_cast<std::uint64_t>(k) * kSec);  // kiekvienas kvietimas — naujas intervalas
        }
        CHECK(sink.entered == 1);
        CHECK(hub.dropped_lines() == 998);  // 2-a eilutė laukia buferyje, 3…1000 — praleistos
        sink.release();
        hub.stop();
    }
    CHECK(sink.entered == 2);  // atleidus — išleista ir laukusi eilutė
}

namespace {
// Funkcija be įrašų (RecordingFunction alokuoja); skaičiavimas įjungiamas pirmame begin_tick.
class QuietFunction final : public TickFunction {
public:
    void begin_tick(Index, TickContext&) override { g_count = true; }
    bool process(const Batch& b, TickContext&) override {
        units += b.units.size();
        return true;
    }
    void end_tick(Index i, TickContext&) override {
        if (i >= last) stop->store(true);
    }
    void finish(TickContext&) override { g_count = false; }
    std::uint64_t units = 0;
    Index last = 0;
    std::atomic<bool>* stop = nullptr;
};
}  // namespace

TEST_CASE("stats: be alokacijų tick'e", "[engine][stats][alloc]") {
    constexpr std::uint32_t kN = 8;
    CollectSink sink;
    VirtualClock clk(static_cast<std::uint64_t>(T(1000)));
    StatsHub hub(0.1, &sink);
    std::atomic<bool> stop{false};
    std::vector<std::unique_ptr<ScriptedInput>> ins;
    std::vector<std::unique_ptr<RecordingOutput>> outs;
    EngineConfig cfg = Rig::test_config();
    TickEngine eng(cfg, clk, hub, stop);
    for (std::uint32_t u = 0; u < kN; ++u) {
        ins.push_back(std::make_unique<ScriptedInput>(clk));
        outs.push_back(std::make_unique<RecordingOutput>(clk));
        if (u == 3) ins[u]->arrival = [](Index j) { return j % 7 == 0 ? std::int64_t{8'000'000} : std::int64_t{1'200'000}; };
        if (u == 5) ins[u]->arrival = [](Index j) { return j >= 1100 ? kNever : std::int64_t{1'200'000}; };  // užstringa
        UnitSpec s;
        s.name = "u" + std::to_string(u);
        s.in.push_back(InputSpec{ins[u].get(), {}, 1, Need::Required, InMode::Exact});
        s.out.push_back(outs[u].get());
        std::uint32_t id = 0;
        std::string why;
        REQUIRE(eng.add_unit(s, id, why));
    }
    hub.register_counter("frames_queued");
    QuietFunction fn;
    fn.last = 1299;
    fn.stop = &stop;
    g_count = true;  // kontrolė: skaitiklis tikrai mato alokacijas šioje gijoje (globalus — neeliminuojama)
    g_escape = new std::vector<int>(100);
    g_count = false;
    delete g_escape;
    REQUIRE(g_allocs.load() == 2);  // vektoriaus objektas + jo buferis
    g_allocs = 0;
    const RunResult res = eng.run(fn);
    const std::uint64_t allocs = g_allocs.load();
    CHECK(res.last_index == 1299);
    CHECK(fn.units > 0);
    CHECK(allocs == 0);
    // 6 s virtualaus laiko, intervalas 0,1 s → ≈ 59 intervalų; virtualus laikas bėga greičiau, nei publikuojama,
    // todėl dalis eilučių praleidžiama (dropped_lines), bet ne prarandama: skirtumai pereina į kitą eilutę.
    REQUIRE(sink.wait_count(1));
    hub.stop();
    const std::size_t lines = sink.ok.size() + hub.dropped_lines();
    CHECK(lines >= 58);
    CHECK(lines <= 60);
    CHECK(hub.in(eng.in_id(5, 0)).stalls == 1);
    CHECK(hub.out(eng.out_id(3, 0)).late > 0);
}
