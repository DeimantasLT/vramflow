// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bendras tick variklis (planas docs/planai/2026-11-12-tick-variklis.md §5, §6 D1–D4, §7). Viena gija valdo visus
// įėjimus ir išėjimus; funkcija (TickFunction) gauna paruoštus vienetus vienu batch'u kiekvienai fazei (A, B, C).
// Izoliacijos taisyklė T1–T13 — §6 D2; laukimas — §6 D3 (ašinis); spragos — §6 D4.
// Po run() pradžios variklis karštame kelyje nealokuoja (tikrina testas „stats: be alokacijų tick'e“).
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/clock.hpp"
#include "engine/cuts.hpp"
#include "engine/gap_state.hpp"
#include "engine/ports.hpp"
#include "engine/stats.hpp"

namespace mxlv::engine {

struct InputSpec {
    InputPort* port = nullptr;
    EdgeTiming timing;
    std::uint8_t grains = 1;  // > 1: skaitomi j − grains + 1 … j (enc interlaced: 2)
    Need need = Need::Required;
    InMode mode = InMode::Exact;
};
struct Demand {
    double nvdec_mpix_s = 0, nvenc_mpix_s = 0;
    std::uint64_t vram_bytes = 0;
};
struct UnitSpec {
    std::string name;               // statistikai (flow ID arba rinkinys)
    std::vector<InputSpec> in;      // tuščia — šaltinis
    std::vector<OutputPort*> out;   // tuščia — sink
    std::uint32_t stride = 1, residue = 0;  // dalyvauja, kai i % stride == residue
    Demand demand;
};

// Skipped — išėjimo indeksas jau commit'intas (laikrodis atgal, M11): ne klaida ir ne praradimas.
// OutputError — išėjimo open / commit klaida: indeksas be įrašo, commit_errors++ (praradimas); out_fail_fatal kartų iš
// eilės tam pačiam išėjimui — sugenda TIK tas vienetas (fail_unit: jo išėjimai atlaisvinami, kiti N − 1 tęsia); kai sugenda
// visi vienetai — fatalu (exit 2).
enum class Outcome : std::uint8_t { Ready, GapBlocked, GapLate, GapNoContent, GapClock, Lapped, Skipped, OutputError };
const char* to_string(Outcome);

struct UnitTick {
    std::uint32_t unit;
    Index index;
    std::uint8_t phase;              // 0 = A, 1 = B, 2 = C
    std::span<const GrainRef> in;    // Σ grains, įėjimų tvarka; nesulaukto Wanted / Optional ptr == nullptr
    std::span<void* const> out;      // atidaryti išėjimo slot'ai
};
struct Batch {
    Index index;
    std::uint8_t phase;
    std::span<const UnitTick> units;
};

class TickEngine;

class TickContext {
public:
    Clock& clock();
    Index index() const;
    void no_content(std::uint32_t unit);                  // šaltinis: šiam indeksui turinio nėra → K3
    void fail_unit(std::uint32_t unit, std::string why);  // K17: vienetas sustabdomas, kiti tęsia
    void fatal(int exit_code, std::string why);
    void count(std::uint32_t unit, CounterId id, std::uint64_t n = 1);
    void hist(HistId id, std::uint64_t ns);
    bool clock_event_recent(std::uint64_t window_ns) const;  // A17
    // Šio indekso A paleidimo „laikymas“: dabar − momentas, kai paskutinis į batch'ą patekęs vienetas tapo
    // paruoštas (0 — niekas nelaikė). Ne §7 sąsajos dalis — diagnostikai ir testams.
    std::int64_t held_ns() const;
    // Tik on_skip metu: true — praleidimas yra praradimas (late_skips), false — laikrodžio žingsnis arba įšilimas.
    bool skip_is_loss() const;

    explicit TickContext(void* engine_impl) : e_(engine_impl) {}  // kuria tik variklis

private:
    void* e_;
};

class TickFunction {
public:
    virtual ~TickFunction() = default;
    virtual void begin_tick(Index, TickContext&) {}  // t_open: šaltiniai paruošia turinį
    // Visi paruošti vienetai vienu batch'u; ≤ 3 kvietimai indeksui (A, B, C) ir retas 4-as — paskutinis batch'as ties c(i)
    // (vienetai, paruošti po trečio kvietimo, bet iki savo ribos; EngineCounters::cut_batches). Funkcija pati daro vieną
    // GPU sync prieš grįždama.
    virtual bool process(const Batch&, TickContext&) = 0;  // false → fatal
    virtual void end_unit(const UnitTick&, Outcome, TickContext&) {}  // po commit'o / spragos
    // Kai visi šio indekso vienetai išspręsti, įskaitant prisijungusius po pernešimo (tada — po kito begin_tick).
    virtual void end_tick(Index, TickContext&) {}
    virtual void on_clock_step(std::int64_t, TickContext&) {}
    // Variklio praleisti indeksai [from, to) vienetui (T11; nepildomi — K-T6). Neperduodami tik stride nedalyvaujantys.
    virtual void on_skip(std::uint32_t /*unit*/, Index /*from*/, Index /*to*/, TickContext&) {}
    virtual void finish(TickContext&) {}  // sustabdant: eilių išleidimas
};

enum class StartMode : std::uint8_t { Immediate, WaitAll };
enum class WaitMode : std::uint8_t { Pivot, Poll };

struct EngineConfig {
    CutConfig cuts = default_cuts({50, 1});  // dažnis, ro, Wr, D, G
    std::uint32_t stall_after = 3;           // 0 — pertraukiklis ir Late žymės išjungti (legacy)
    std::uint32_t late_clear_run = 8;        // Late nuimama po tiek laiku (≤ a(i)) atėjusių grain'ų iš eilės
    GapPolicy on_loss = GapPolicy::Invalid;
    std::int64_t hold_ns = 1'000'000'000;
    StartMode start = StartMode::Immediate;
    WaitMode wait = WaitMode::Pivot;
    std::int64_t input_wait_ns = 0;  // 0 — neribotai
    double duration_s = 0, warmup_s = 5;
    std::uint32_t skip_align = 1;    // dec interlaced: 2 (po praleidimo tęsti nuo viršutinio lauko)
    std::uint32_t out_fail_fatal = 3;  // tiek open / commit klaidų iš eilės vienam išėjimui → vienetas sugenda; 0 — niekada
    // Variklio gija (planas A19, D8): SCHED_FIFO prioritetas (0 — nekeisti) ir CPU rinkinys (tuščias — nekeisti). Taikoma
    // run() pradžioje PO statistikos gijos paleidimo — ji (ir kitos jau sukurtos gijos) RT nepaveldi. Nepavykus (nėra
    // teisių) — įspėjimas RunResult::warn, darbas tęsiamas.
    int rt_priority = 0;
    std::vector<int> cpu_affinity;
    std::function<bool(const UnitSpec&, std::string& why)> admit;  // P4; tuščias — priimti
    // Bibliotekos PacingHooks::wait: jei nustatytas, tick'as atidaromas juo, ne Clock::sleep_until.
    std::function<bool(Index, std::int64_t offset_ns, const std::atomic<bool>& stop)> open_wait;

    static EngineConfig legacy(timing::Rational rate, std::int64_t read_offset_ns, std::int64_t deadline_ns);
};

enum class EndReason : std::uint8_t { Stopped, Duration, Fatal, InputsTimeout, FormatChange };
struct RunResult {
    EndReason reason = EndReason::Stopped;
    int exit_code = 0;
    std::string fatal;
    Index first_index = kNoIndex, last_index = kNoIndex;
    // Bent vienas tikras praradimas: vieneto spraga (be laikrodžio ir šaltinio „nėra turinio“ spragų), lap'as, late_skips,
    // įėjimo lost, išėjimo commit_errors. Laikrodžio įvykiai (clock_skips, GapClock, AlreadyCommitted) — ne praradimas.
    // Funkcijos jį naudoja exit kodui 4 griežtame režime (--strict-exit).
    bool any_loss = false;
    std::uint32_t failed_units = 0;  // fail_unit (ilgalaikė išėjimo klaida arba funkcijos sprendimas); visi — Fatal, exit 2
    std::string failed_why;          // pirmo sugedusio vieneto priežastis (santraukai)
    std::string warn;                // ne fatalūs įspėjimai (pvz. --rt-priority be teisių)
};

class TickEngine {
public:
    TickEngine(const EngineConfig&, Clock&, StatsHub&, const std::atomic<bool>& stop);
    ~TickEngine();
    TickEngine(const TickEngine&) = delete;
    TickEngine& operator=(const TickEngine&) = delete;

    bool add_unit(const UnitSpec&, std::uint32_t& unit, std::string& why);  // prieš run(); false — atmesta
    RunResult run(TickFunction&);                                            // kviečianti gija — variklio gija

    // Statistikos ir testų prieiga (StatsHub numeriai; įėjimo būsena).
    std::uint32_t in_id(std::uint32_t unit, std::uint32_t input) const;
    std::uint32_t out_id(std::uint32_t unit, std::uint32_t output) const;
    std::uint32_t unit_id(std::uint32_t unit) const;
    InState in_state(std::uint32_t unit, std::uint32_t input) const;
    bool unit_failed(std::uint32_t unit) const;
    bool in_late(std::uint32_t unit, std::uint32_t input) const;

private:
    friend class TickContext;
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace mxlv::engine
