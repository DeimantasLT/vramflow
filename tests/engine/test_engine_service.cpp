// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Tarnybos stebėsena ir atkūrimas (B1; planas V21, D8): įvykiai operatoriui iš karto, sugedusio vieneto atkūrimas su
// atidėjimu 1, 2, 4 … ≤ 30 s, Σ eilučių == skaitikliai, užstrigęs publikavimas tick'o nevėlina.
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "fakes.hpp"

using namespace mxlv::engine;
using namespace mxlv::engine::test;

namespace {
// Sink'as: įvykiai ir vieneto skirtumų sumos (publish / event kviečia hub'o gija).
class LogSink final : public StatsSink {
public:
    void publish(const Snapshot& s) override {
        std::lock_guard<std::mutex> lk(m);
        ++lines;
        if (ok.size() < s.unit.size()) ok.resize(s.unit.size(), 0);
        for (std::size_t u = 0; u < s.unit.size(); ++u) ok[u] += s.unit[u].d.ok + s.unit[u].d.late;
        states.push_back(s.unit.empty() ? Health::Inactive : s.unit[0].health);
        tai.push_back(s.t_end_tai);
    }
    void event(const Event& e, const StatsNames& n) override {
        std::lock_guard<std::mutex> lk(m);
        std::string who;
        if (e.kind == EventKind::UnitFailed || e.kind == EventKind::UnitRecovered || e.kind == EventKind::UnitRetryFailed) who = (*n.unit)[e.id];
        else if (e.kind == EventKind::OutputError || e.kind == EventKind::OutputRecovered) who = (*n.out)[e.id];
        else if (e.kind != EventKind::ClockStep && e.kind != EventKind::Warning) who = (*n.in)[e.id];
        ev.push_back({e.kind, who, e.index, e.tai, e.text});
    }
    struct Ev {
        EventKind kind;
        std::string who;
        std::uint64_t index, tai;
        std::string text;
    };
    std::vector<const Ev*> of(EventKind k) const {
        std::vector<const Ev*> r;
        for (const Ev& e : ev)
            if (e.kind == k) r.push_back(&e);
        return r;
    }
    std::mutex m;
    std::vector<Ev> ev;
    std::vector<std::uint64_t> ok, tai;
    std::vector<Health> states;
    std::uint64_t lines = 0;
};

// Kaip Rig, bet StatsHub su sink'u (Rig'o hub'as — be jo) ir atkūrimas variklio gijoje (virtualus laikas).
struct SvcRig {
    explicit SvcRig(std::uint32_t n, StatsSink* sink = nullptr, double interval_s = 1.0)
        : clk(static_cast<std::uint64_t>(T(1000))), fn(clk), hub(interval_s, sink) {
        cfg = Rig::test_config();
        cfg.retry_async = false;
        for (std::uint32_t u = 0; u < n; ++u) {
            ins.push_back(std::make_unique<ScriptedInput>(clk));
            outs.push_back(std::make_unique<RecordingOutput>(clk));
        }
    }
    TickEngine& build() {
        eng = std::make_unique<TickEngine>(cfg, clk, hub, stop);
        for (std::uint32_t u = 0; u < ins.size(); ++u) {
            UnitSpec s;
            s.name = "u" + std::to_string(u);
            s.in.push_back(InputSpec{ins[u].get(), {}, 1, Need::Required, InMode::Exact});
            s.out.push_back(outs[u].get());
            std::uint32_t id = 0;
            std::string why;
            if (!eng->add_unit(s, id, why)) throw std::runtime_error(why);
        }
        return *eng;
    }
    void stop_after(Index last) {
        fn.on_end_tick = [this, last](Index i, TickContext&) {
            if (i >= last) stop = true;
        };
    }
    const OutCounters& out(std::uint32_t u) const { return hub.out(eng->out_id(u, 0)); }
    RunResult run() {
        res = eng->run(fn);
        hub.flush(clk.now_tai());
        hub.stop();
        return res;
    }
    VirtualClock clk;
    RecordingFunction fn;
    StatsHub hub;
    EngineConfig cfg;
    std::atomic<bool> stop{false};
    std::vector<std::unique_ptr<ScriptedInput>> ins;
    std::vector<std::unique_ptr<RecordingOutput>> outs;
    std::unique_ptr<TickEngine> eng;
    RunResult res;
};

Index first_commit_after(const RecordingOutput& o, Index after) {
    for (const auto& e : o.events)
        if (e.kind == RecordingOutput::Kind::Commit && e.index > after) return e.index;
    return kNoIndex;
}
}  // namespace

TEST_CASE("svc: išėjimo gedimas — įvykiai iš karto, atkūrimas po 1 s, kai priežastis praėjo", "[engine][svc][retry]") {
    constexpr Index kI = 1010;
    LogSink sink;
    SvcRig r(2, &sink);
    r.outs[1]->fail_commit = [](Index i) { return i >= kI && i < kI + 30; };  // praeina po 30 indeksų (0,6 s)
    r.stop_after(kI + 120);
    r.build();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Stopped);
    CHECK(res.failed_units == 0);  // atsigavo — sustabdant nebėra sugedusių
    CHECK(res.unit_failures == 1);
    CHECK(res.unit_recoveries == 1);
    CHECK(r.hub.unit(r.eng->unit_id(1)).failures == 1);
    CHECK(r.hub.unit(r.eng->unit_id(1)).recoveries == 1);
    CHECK_FALSE(r.eng->unit_failed(1));
    CHECK(r.eng->unit_ever_failed(1));
    // Gedimas — po 3 klaidų iš eilės (kI + 2); vienintelis atkūrimo bandymas — pirmo tick'o atidaryme po +1 s.
    REQUIRE(r.outs[1]->reopen_tai.size() == 1);
    const auto fail_at = static_cast<std::int64_t>(T(kI + 2));
    const std::int64_t dt = static_cast<std::int64_t>(r.outs[1]->reopen_tai[0]) - fail_at;
    CHECK(dt >= 1'000'000'000);
    CHECK(dt <= 1'000'000'000 + 2 * 20'000'000);
    const Index back = first_commit_after(*r.outs[1], kI + 2);
    REQUIRE(back != kNoIndex);
    CHECK(static_cast<std::int64_t>(T(back)) >= static_cast<std::int64_t>(r.outs[1]->reopen_tai[0]) - 20'000'000);
    for (Index i = back; i <= kI + 120; ++i) {  // nuo atkūrimo — kiekvienas indeksas commit'intas
        const auto* e = r.outs[1]->find(i);
        REQUIRE(e != nullptr);
        CHECK(e->kind == RecordingOutput::Kind::Commit);
    }
    for (Index i = kI; i <= kI + 120; ++i) CHECK(r.outs[0]->find(i) != nullptr);  // sveikas — nepaliestas
    CHECK(r.out(0).commit_errors == 0);
    // Įvykiai: išėjimo klaida (pirmoji), vienetas sugedo (su priežastimi ir išėjimo vardu), atsigavo.
    const auto oe = sink.of(EventKind::OutputError);
    const auto uf = sink.of(EventKind::UnitFailed);
    const auto ur = sink.of(EventKind::UnitRecovered);
    REQUIRE(oe.size() == 1);
    REQUIRE(uf.size() == 1);
    REQUIRE(ur.size() == 1);
    CHECK(oe[0]->who == "u1/out0");
    CHECK(oe[0]->index == kI);
    CHECK(uf[0]->who == "u1");
    CHECK(uf[0]->text.find("u1/out0") != std::string::npos);
    CHECK(uf[0]->text.find("bandymas po 1 s") != std::string::npos);
    CHECK(uf[0]->tai - static_cast<std::uint64_t>(fail_at) < 20'000'000);  // pranešta tame pačiame tick'e
    CHECK(ur[0]->who == "u1");
}

TEST_CASE("svc: neatkuriamas išėjimas — atidėjimas 1, 2, 4 … ≤ 30 s, sustabdant vis dar sugedęs", "[engine][svc][retry]") {
    constexpr Index kI = 1010;
    LogSink sink;
    SvcRig r(2, &sink);
    r.outs[1]->fail_open = [](Index i) { return i >= kI; };
    r.outs[1]->reopen_ok = [] { return false; };
    r.stop_after(kI + 50 * 130);  // 130 s
    r.build();
    const RunResult res = r.run();
    CHECK(res.reason == EndReason::Stopped);
    CHECK(res.failed_units == 1);  // funkcijos: exit 2 abiem režimais
    CHECK(res.unit_recoveries == 0);
    const auto& t = r.outs[1]->reopen_tai;
    // bandymai: +1, +3, +7, +15, +31, +61, +91, +121 s nuo gedimo (atidėjimas 1, 2, 4, 8, 16, 30, 30, 30)
    const double want[] = {1, 3, 7, 15, 31, 61, 91, 121};
    REQUIRE(t.size() == 8);
    for (std::size_t k = 0; k < t.size(); ++k) {
        INFO("bandymas " << k);
        const double s = static_cast<double>(static_cast<std::int64_t>(t[k]) - static_cast<std::int64_t>(T(kI + 2))) / 1e9;
        CHECK(s >= want[k]);
        CHECK(s <= want[k] + 0.05);
    }
    CHECK(res.unit_retries == 8);
    CHECK(sink.of(EventKind::UnitRetryFailed).size() == 8);
    CHECK(sink.of(EventKind::UnitFailed).size() == 1);
    CHECK(r.out(0).commit_errors == 0);
}

TEST_CASE("svc: įėjimo būsenos — atsijungė / prisijungė iš naujo pranešama per ≤ 1 periodą", "[engine][svc][events]") {
    constexpr Index kD = 1020, kR = 1080;
    LogSink sink;
    SvcRig r(2, &sink);
    r.ins[1]->arrival = [](Index j) { return j >= kD && j < kR ? kNever : std::int64_t{1'200'000}; };
    r.ins[1]->alive = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kD) || static_cast<std::int64_t>(now) >= T(kR); };
    r.ins[1]->attachable = [](std::uint64_t now) { return static_cast<std::int64_t>(now) < T(kD) || static_cast<std::int64_t>(now) >= T(kR); };
    r.stop_after(kR + 40);
    r.build();
    r.run();
    const auto att = sink.of(EventKind::InputAttached);
    const auto det = sink.of(EventKind::InputDetached);
    const auto rea = sink.of(EventKind::InputReattached);
    CHECK(att.size() == 2);  // abu įėjimai paleidžiant
    REQUIRE(det.size() == 1);
    REQUIRE(rea.size() == 1);
    CHECK(det[0]->who == "u1/in0");
    CHECK(rea[0]->who == "u1/in0");
    // atsijungta ties a(kD) (rašytojas negyvas) — pranešta kito tick'o atidaryme
    CHECK(static_cast<std::int64_t>(det[0]->tai) - T(kD) <= 2 * 20'000'000);
    CHECK(static_cast<std::int64_t>(rea[0]->tai) - T(kR) <= 100'000'000 + 2 * 20'000'000);  // prisijungimo bandymai kas 100 ms
}

namespace {
// publish negrįžta, kol testas neatleidžia (užstrigęs stebėsenos skaitytojas / pilnas diskas).
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
}  // namespace

TEST_CASE("svc: užstrigęs publikavimas (10 s) tick'o nevėlina ir nekeičia", "[engine][svc][stats]") {
    constexpr Index kLast = 1000 + 50 * 12;  // 12 s virtualaus laiko, intervalas 1 s
    std::vector<std::uint64_t> base;
    {  // kontrolė: publikavimas nestringa
        LogSink sink;
        SvcRig r(4, &sink);
        r.stop_after(kLast);
        r.build();
        r.run();
        for (const auto& e : r.outs[2]->events) base.push_back(e.tai);
    }
    StuckSink stuck;
    SvcRig r(4, &stuck);
    r.stop_after(kLast);
    r.build();
    const RunResult res = r.eng->run(r.fn);  // variklis baigia, nors publikavimo gija įstrigusi pirmoje eilutėje
    {
        std::lock_guard<std::mutex> lk(stuck.m);
        CHECK(stuck.entered == 1);
        CHECK_FALSE(stuck.released);
    }
    CHECK(res.last_index == kLast);
    CHECK(r.hub.dropped_lines() >= 9);  // 12 intervalų: 1 įstrigusi, 1 laukia buferyje, kitos praleistos (skirtumai pereina)
    std::vector<std::uint64_t> got;
    for (const auto& e : r.outs[2]->events) got.push_back(e.tai);
    CHECK(got == base);  // tie patys commit'ai tais pačiais laikais
    stuck.release();
    r.hub.stop();
}

TEST_CASE("svc: Σ ok per eilutes == skaitikliai (su paskutine nepilna eilute)", "[engine][svc][stats]") {
    LogSink sink;
    SvcRig r(3, &sink, 0.5);
    r.ins[2]->arrival = [](Index j) { return j >= 1100 && j < 1140 ? kNever : std::int64_t{1'200'000}; };  // spraga
    r.stop_after(1000 + 50 * 7 + 13);  // ne intervalų kartotinis
    r.build();
    r.run();
    REQUIRE(sink.ok.size() == 3);
    for (std::uint32_t u = 0; u < 3; ++u) {
        INFO("u" << u);
        const UnitCounters& c = r.hub.unit(r.eng->unit_id(u));
        CHECK(sink.ok[u] == c.ok + c.late);
        CHECK(c.ok + c.late > 300);
    }
    // virtualus laikas bėga greičiau nei publikuojama: dalis eilučių praleidžiama, bet jų skirtumai pereina į kitą
    CHECK(sink.lines + r.hub.dropped_lines() >= 14);
}

TEST_CASE("svc: vieneto būsena — Healthy, praradimai PartiallyHealthy, sugedęs Unhealthy", "[engine][svc][stats]") {
    UnitCounters h{};
    h.ok = 40;
    h.late = 10;
    CHECK(unit_health(h, false) == Health::Healthy);  // B / C fazė — ne praradimas
    UnitCounters p = h;
    p.gaps = 2;
    CHECK(unit_health(p, false) == Health::PartiallyHealthy);
    UnitCounters c = h;
    c.gaps = 2;
    c.gaps_clock = 2;  // laikrodžio spragos — ne praradimas
    CHECK(unit_health(c, false) == Health::Healthy);
    UnitCounters z{};
    z.gaps = 50;
    CHECK(unit_health(z, false) == Health::Unhealthy);
    CHECK(unit_health(h, true) == Health::Unhealthy);
    CHECK(unit_health(UnitCounters{}, false) == Health::Inactive);
    const Health a[] = {Health::Healthy, Health::Healthy};
    const Health m[] = {Health::Healthy, Health::Unhealthy};
    const Health b[] = {Health::Unhealthy, Health::Inactive};
    CHECK(units_health(a) == Health::Healthy);
    CHECK(units_health(m) == Health::PartiallyHealthy);
    CHECK(units_health(b) == Health::Unhealthy);
}
