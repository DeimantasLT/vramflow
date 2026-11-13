// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variklio statistika (planas §6 D6). Skaitikliai — fiksuotos struktūros masyvuose, užregistruoti prieš run();
// rašo tik variklio gija. end_tick(now): kai praėjo intervalas, variklio gija į iš anksto paruoštą buferį nukopijuoja
// intervalo skirtumus ir būsenas ir pažadina savo publikavimo giją, kuri kviečia StatsSink::publish. Tick'as niekada
// nedaro I/O ir nelaukia: jei ankstesnė kopija dar nepaimta — dropped_lines++ (skirtumai pereina į kitą eilutę).
// Įvykiai (vienetas sugedo / atsigavo, įėjimas atsijungė / prisijungė / užstrigo, išėjimo klaida, formato pokytis, laikrodžio
// žingsnis) — iš anksto paruoštame žiede (be alokacijų); publikavimo gija juos perduoda StatsSink::event per ≤ 50 ms.
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
// Vienetas (flow): sugedęs — Unhealthy; per intervalą nė vieno pateikto indekso (ok + late == 0) — Unhealthy, nebent
// vienetas nedalyvavo (stride / sustabdytas) — Inactive; praradimų (spragos be laikrodžio / „nėra turinio“, lap'ai,
// late_skips) — PartiallyHealthy; kitaip Healthy. Vėlavimas B / C fazėje (late) — ne praradimas.
Health unit_health(const UnitCounters& d, bool failed);
// Proceso būsena eilutėje — iš vienetų: visi Healthy → Healthy; visi aktyvūs Unhealthy → Unhealthy; kitaip PartiallyHealthy.
Health units_health(std::span<const Health> units);

struct EngineCounters {
    std::uint64_t ticks = 0, process_calls = 0, guard_overrun = 0, carry_overrun = 0;
    // Paskutinis batch'as ties c(i): vienetai, paruošti po trečio process kvietimo, bet iki savo ribos (4-as kvietimas
    // indeksui; retas — A + B + ankstyvas C jau buvo). Ne praradimas: tokie vienetai niekada neišmetami.
    std::uint64_t cut_batches = 0;
    std::uint64_t clock_steps = 0, clock_backward = 0, step_gaps = 0, clock_back_ns = 0;  // back — atgalinių šuolių suma
    std::uint64_t held_ns_max = 0;  // kiek ilgiausiai paruoštus vienetus laikė neparuošti (A fazė); maksimumas
    // Saugikliai (sutarties pažeidimai, niekada neturėtų didėti): open_wait grįžo true prieš t_open; laukimo ciklas be
    // būsimo įvykio (ev ≤ dabar) — miegama 100 µs, ne sukamasi.
    std::uint64_t open_wait_early = 0, spin_guard = 0;
};

// 64 kibirų log-histograma (¼ oktavos, nuo 1 µs iki ~65 ms), 256 B. Vėlavimams p50 / p99.
class Hist64 {
public:
    static int bucket_of(std::uint64_t ns);
    static std::uint64_t value_of(int b);  // kibiro apatinė riba ns
    void add(std::uint64_t ns) {
        ++c_[static_cast<std::size_t>(bucket_of(ns))];
        ++n_;
        if (ns > max_) max_ = ns;
    }
    std::uint64_t quantile(double q) const;
    std::uint64_t max() const { return max_; }  // tikslus maksimumas (ne kibiro riba)
    std::uint32_t count() const { return n_; }
    void reset() { *this = Hist64{}; }

private:
    std::array<std::uint32_t, 64> c_{};
    std::uint32_t n_ = 0;
    std::uint64_t max_ = 0;
};

// Įvykis operatoriui (stderr eilutė su laiku iš karto, ne intervalo gale). id — hub'o numeris pagal rūšį (vienetas /
// įėjimas / išėjimas; ClockStep, Warning — nenaudojamas). text — nukopijuojamas (be alokacijų), ilgesnis — nukerpamas.
enum class EventKind : std::uint8_t {
    UnitFailed, UnitRetryFailed, UnitRecovered, InputAttached, InputReattached, InputDetached, InputStalled, InputResumed,
    OutputError, OutputRecovered, FormatChange, ClockStep, Warning,
    // Neprivalomas įėjimas (composite sluoksnis): nebenaudojamas (skaidrus) / vėl naudojamas — praneša funkcija, ne variklis
    LayerAbsent, LayerPresent
};
const char* to_string(EventKind);
struct Event {
    std::uint64_t tai = 0;
    std::uint64_t index = ~std::uint64_t{0};
    std::int64_t value = 0;  // ClockStep — Δ ns; UnitRetryFailed — kitas bandymas po ms; kitur 0
    std::uint32_t id = 0;
    EventKind kind = EventKind::Warning;
    char text[160]{};
};
struct StatsNames {
    const std::vector<std::string>* unit = nullptr;
    const std::vector<std::string>* in = nullptr;
    const std::vector<std::string>* out = nullptr;
};

struct InSnap {
    InCounters d;  // intervalo skirtumai
    InState state = InState::Waiting;
    Health health = Health::Unhealthy;
    std::uint64_t arrival_p50_ns = 0, arrival_p99_ns = 0, arrival_max_ns = 0;
    std::uint32_t unit = 0;  // vieneto hub numeris
};
struct OutSnap {
    OutCounters d;
    bool enabled = true;
    Health health = Health::Unhealthy;
    std::uint64_t commit_p50_ns = 0, commit_p99_ns = 0, commit_max_ns = 0;
    std::uint32_t unit = 0;
};
struct UnitSnap {
    UnitCounters d;  // intervalo skirtumai
    bool failed = false;
    Health health = Health::Inactive;
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
    std::vector<UnitSnap> unit;
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
    virtual void event(const Event&, const StatsNames&) {}  // ta pati gija; prieš to intervalo eilutę
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

    // Variklis registruoja vienetus / įėjimus / išėjimus add_unit metu: pirma vienetą, tada jo įėjimus ir išėjimus (jie
    // priskiriami paskutiniam užregistruotam vienetui).
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
    UnitCounters& unit(std::uint32_t id) { return unit_[id]; }
    const UnitCounters& unit(std::uint32_t id) const { return unit_[id]; }
    Hist64& in_arrival(std::uint32_t id) { return in_hist_[id]; }
    Hist64& out_commit(std::uint32_t id) { return out_hist_[id]; }

    void count(std::uint32_t unit, CounterId id, std::uint64_t n) { fn_[unit * counter_names_.size() + id] += n; }
    std::uint64_t counter(std::uint32_t unit, CounterId id) const { return fn_[unit * counter_names_.size() + id]; }
    void hist(HistId id, std::uint64_t ns) { hist_[id].add(ns); }

    // Variklio gija: būsenos, kurių skaitikliai neturi (nustatomos prieš end_tick, kai due()).
    void set_in_state(std::uint32_t id, InState s) { in_state_[id] = s; }
    void set_out_enabled(std::uint32_t id, bool on) { out_on_[id] = on ? 1 : 0; }
    void set_unit_failed(std::uint32_t id, bool f) { unit_failed_[id] = f ? 1 : 0; }
    bool due(std::uint64_t now_tai) const { return !started_ || now_tai >= next_due_; }
    void end_tick(std::uint64_t now_tai);  // variklio gija; neblokuoja
    // Po run() (ne tick'e): paskutinė, nepilno intervalo eilutė — kad Σ eilučių skirtumų == santraukos skaitikliai. Laukia
    // ≤ 2 s, kol publikavimo gija paims ankstesnę kopiją (kitaip — dropped_lines++, skirtumai lieka neišleisti).
    void flush(std::uint64_t now_tai);
    // Variklio gija: įvykis į žiedą (neblokuoja, nealokuoja; pilnas — events_dropped++).
    void event(EventKind k, std::uint32_t id, std::uint64_t tai, std::uint64_t index, std::int64_t value, const char* text);
    std::uint64_t events_dropped() const { return ev_dropped_; }
    Health health() const { return health_; }  // paskutinio intervalo proceso būsena (variklio gija)
    std::uint64_t dropped_lines() const { return dropped_; }
    void stop();  // išleidžia laukiančią kopiją ir įvykius, sustabdo publikavimo giją (po run())

    std::size_t inputs() const { return in_.size(); }
    std::size_t outputs() const { return out_.size(); }
    std::size_t units() const { return unit_names_.size(); }

private:
    void publisher();
    void drain_events();
    void snapshot(std::uint64_t now);

    std::uint64_t interval_ns_;
    StatsSink* sink_;
    std::vector<std::string> counter_names_, hist_names_, unit_names_, in_names_, out_names_;
    std::vector<std::uint32_t> in_owner_, out_owner_;
    std::vector<InCounters> in_, in_prev_;
    std::vector<OutCounters> out_, out_prev_;
    std::vector<UnitCounters> unit_, unit_prev_;
    std::vector<std::uint8_t> unit_failed_;
    std::vector<Health> unit_h_;
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

    static constexpr std::uint32_t kEvents = 256;
    std::array<Event, kEvents> ev_{};
    std::atomic<std::uint32_t> ev_w_{0}, ev_r_{0};  // SPSC: rašo variklio gija, skaito publikavimo gija
    std::uint64_t ev_dropped_ = 0;
};

}  // namespace mxlv::engine
