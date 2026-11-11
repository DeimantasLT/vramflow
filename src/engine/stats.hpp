// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variklio statistika (planas §6 D6). Skaitikliai — fiksuotos struktūros masyvuose, užregistruoti prieš run();
// rašo tik variklio gija. end_tick(now): kai praėjo intervalas, variklio gija į iš anksto paruoštą buferį nukopijuoja
// intervalo skirtumus ir būsenas ir pažadina savo publikavimo giją, kuri kviečia StatsSink::publish. Tick'as niekada
// nedaro I/O ir nelaukia: jei ankstesnė kopija dar nepaimta — dropped_lines++ (skirtumai pereina į kitą eilutę).
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "engine/counters.hpp"
#include "engine/input_state.hpp"

namespace mxlv::engine {

using CounterId = std::uint32_t;
using HistId = std::uint32_t;

// BCP-008 būsenos (planas nurodo runtime::Health iš [P] E0 — jo dar nėra, todėl kol kas čia).
enum class Health : std::uint8_t { Healthy, PartiallyHealthy, Unhealthy, Inactive };
const char* to_string(Health);

// §6 D6 lentelė; skaitikliai — intervalo skirtumai.
Health input_health(InState, const InCounters&);
Health output_health(const OutCounters&, bool enabled);
Health process_health(std::span<const Health> inputs, std::span<const Health> outputs);

struct EngineCounters {
    std::uint64_t ticks = 0, process_calls = 0, guard_overrun = 0, carry_overrun = 0;
    std::uint64_t clock_steps = 0, clock_backward = 0, step_gaps = 0;
    std::uint64_t held_ns_max = 0;  // kiek ilgiausiai paruoštus vienetus laikė neparuošti (A fazė); maksimumas
};

// 64 kibirų log-histograma (¼ oktavos, nuo 1 µs iki ~65 ms), 256 B. Vėlavimams p50 / p99.
class Hist64 {
public:
    static int bucket_of(std::uint64_t ns);
    static std::uint64_t value_of(int b);  // kibiro apatinė riba ns
    void add(std::uint64_t ns) {
        ++c_[static_cast<std::size_t>(bucket_of(ns))];
        ++n_;
    }
    std::uint64_t quantile(double q) const;
    std::uint32_t count() const { return n_; }
    void reset() { *this = Hist64{}; }

private:
    std::array<std::uint32_t, 64> c_{};
    std::uint32_t n_ = 0;
};

struct InSnap {
    InCounters d;  // intervalo skirtumai
    InState state = InState::Waiting;
    Health health = Health::Unhealthy;
    std::uint64_t arrival_p50_ns = 0, arrival_p99_ns = 0;
};
struct OutSnap {
    OutCounters d;
    bool enabled = true;
    Health health = Health::Unhealthy;
    std::uint64_t commit_p50_ns = 0, commit_p99_ns = 0;
};
struct HistSnap {
    std::uint64_t count = 0, p50_ns = 0, p99_ns = 0;
};
// POD masyvai + intervalas + būsenos. Vardai — rodyklės į hub'o sąrašus (nekinta po prepare()).
struct Snapshot {
    std::uint64_t seq = 0, t_begin_tai = 0, t_end_tai = 0, dropped_lines = 0;
    Health health = Health::Inactive;
    EngineCounters engine;  // skirtumai (held_ns_max — kaupiamasis maksimumas)
    std::vector<InSnap> in;
    std::vector<OutSnap> out;
    std::vector<std::uint64_t> fn;  // units × counters, skirtumai
    std::vector<HistSnap> hist;
    const std::vector<std::string>* in_names = nullptr;
    const std::vector<std::string>* out_names = nullptr;
    const std::vector<std::string>* unit_names = nullptr;
    const std::vector<std::string>* counter_names = nullptr;
    const std::vector<std::string>* hist_names = nullptr;
};

class StatsSink {
public:
    virtual ~StatsSink() = default;
    virtual void publish(const Snapshot&) = 0;  // kviečia hub'o publikavimo gija; gali užtrukti
};

class StatsHub {
public:
    explicit StatsHub(double interval_s = 1.0, StatsSink* sink = nullptr);
    ~StatsHub();
    StatsHub(const StatsHub&) = delete;
    StatsHub& operator=(const StatsHub&) = delete;

    // Funkcijos skaitikliai (vienetui) ir histogramos — prieš run().
    CounterId register_counter(std::string name);
    HistId register_hist(std::string name);

    // Variklis registruoja vienetus / įėjimus / išėjimus add_unit metu.
    std::uint32_t add_unit(std::string name);
    std::uint32_t add_input(std::string name);
    std::uint32_t add_output(std::string name);
    void prepare();  // run() pradžioje: dydžiai fiksuojami, paleidžiama publikavimo gija (po to — be alokacijų)

    InCounters& in(std::uint32_t id) { return in_[id]; }
    const InCounters& in(std::uint32_t id) const { return in_[id]; }
    OutCounters& out(std::uint32_t id) { return out_[id]; }
    const OutCounters& out(std::uint32_t id) const { return out_[id]; }
    EngineCounters& engine() { return eng_; }
    const EngineCounters& engine() const { return eng_; }
    Hist64& in_arrival(std::uint32_t id) { return in_hist_[id]; }
    Hist64& out_commit(std::uint32_t id) { return out_hist_[id]; }

    void count(std::uint32_t unit, CounterId id, std::uint64_t n) { fn_[unit * counter_names_.size() + id] += n; }
    std::uint64_t counter(std::uint32_t unit, CounterId id) const { return fn_[unit * counter_names_.size() + id]; }
    void hist(HistId id, std::uint64_t ns) { hist_[id].add(ns); }

    // Variklio gija: būsenos, kurių skaitikliai neturi (nustatomos prieš end_tick, kai due()).
    void set_in_state(std::uint32_t id, InState s) { in_state_[id] = s; }
    void set_out_enabled(std::uint32_t id, bool on) { out_on_[id] = on ? 1 : 0; }
    bool due(std::uint64_t now_tai) const { return !started_ || now_tai >= next_due_; }
    void end_tick(std::uint64_t now_tai);  // variklio gija; neblokuoja
    Health health() const { return health_; }  // paskutinio intervalo proceso būsena (variklio gija)
    std::uint64_t dropped_lines() const { return dropped_; }
    void stop();  // išleidžia laukiančią kopiją ir sustabdo publikavimo giją (po run())

    std::size_t inputs() const { return in_.size(); }
    std::size_t outputs() const { return out_.size(); }
    std::size_t units() const { return unit_names_.size(); }

private:
    void publisher();

    std::uint64_t interval_ns_;
    StatsSink* sink_;
    std::vector<std::string> counter_names_, hist_names_, unit_names_, in_names_, out_names_;
    std::vector<InCounters> in_, in_prev_;
    std::vector<OutCounters> out_, out_prev_;
    std::vector<std::uint64_t> fn_, fn_prev_;
    std::vector<Hist64> in_hist_, out_hist_, hist_;
    std::vector<InState> in_state_;
    std::vector<std::uint8_t> out_on_;
    std::vector<Health> in_h_, out_h_;
    EngineCounters eng_, eng_prev_;
    bool started_ = false;
    std::uint64_t next_due_ = 0, t_begin_ = 0, seq_ = 0, dropped_ = 0;
    Health health_ = Health::Inactive;

    Snapshot buf_[2];
    std::atomic<int> full_{0};   // 1 — buf_[w_] laukia publikavimo
    std::atomic<int> w_{0};      // kurį buferį pildo variklio gija
    std::atomic<bool> quit_{false};
    std::mutex m_;
    std::condition_variable cv_;
    std::thread th_;
};

}  // namespace mxlv::engine
