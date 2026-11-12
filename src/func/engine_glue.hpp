// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Funkcijų ↔ bendro tick variklio jungtis (planas docs/planai/2026-11-12-tick-variklis.md §7 „mxlv::func“, V8+):
// parinktys → EngineConfig (viena vieta) ir variklio skaitikliai → santraukos raktai (tie patys kaip prieš migraciją).
#include <cstdint>
#include <vector>

#include "engine/engine.hpp"
#include "engine/stats.hpp"
#include "func/func_common.hpp"

namespace mxlv::func {

// Variklio ribos funkcijai:
//   Legacy    — šiandieninis ciklas: viena riba ties T + --deadline-ms, apklausa 0,5 ms, be pertraukiklio, leave, wait-all;
//   Isolation — planas V10: Wr = P/4, M = 3, Late histerezė 8, futex (ašinis) laukimas; D = --deadline-ms (apkarpoma iki
//               2P − G), leave, wait-all (spragų politika ir startas keičiami vėliau — V17, V18).
enum class EngineTiming { Legacy, Isolation };

// read_offset_ns — ro (logo: --read-offset-us; gen: 0). Trukmė, warmup — iš parinkčių.
engine::EngineConfig engine_config(const Opts& o, EngineTiming t, std::int64_t read_offset_ns);

// Kur santraukoje rašyti variklio praleidimus: proceso lygiu (gen) ar kiekvienam flow'ui (logo, composite…).
enum class SkipScope { Process, PerFlow, PerFlowFrames };  // PerFlowFrames — enc: *_skips_frames

// Variklio rezultatas ir skaitikliai → Stats (first / last_index, clock_steps, clock_backward, *_skips). Raktai, kurių
// reikšmė 0, nerašomi (kaip anksčiau — „įvykio“ skaitikliai atsiranda tik padidinti); clock_steps — tik Process (gen
// santraukoje jis visada buvo, kitų funkcijų — ne).
void to_summary(const engine::StatsHub& hub, const engine::TickEngine& eng, const engine::RunResult& r, std::uint32_t units,
                SkipScope scope, Stats& st);

// Tick'o laikai funkcijų histogramoms — ATSKIRAI kiekvienam iš dviejų galimai persidengiančių indeksų (pernešimas:
// i − 1 baigiasi po i atidarymo; slot = index & 1), kad vieno indekso būsena nepasiimtų kito laikų.
//   proc  — tik apdorojimas: Σ per fazes (process pradžia → paskutinis tos fazės vieneto commit / lap / klaida);
//   wait  — tick'o atidarymas → pirmo apdorojimo pradžia (įėjimų laukimas);
//   held  — laikas tarp fazių (paruošti / pavėluoję vienetai laukė B / C fazės): (paskutinis commit − pirma pradžia) − proc;
//   end   — paskutinis commit (be commit'ų — dabar).
class TickTimes {
public:
    struct Result {
        bool processed = false;
        std::uint64_t proc = 0, wait = 0, held = 0, end = 0, first = 0;
        std::uint32_t phases = 0;
    };
    void wake(engine::Index i, std::uint64_t now) { s_[i & 1] = S{now}; }
    void phase_begin(engine::Index i, std::uint64_t now) {
        S& t = s_[i & 1];
        close(t);
        if (t.phases++ == 0) t.first = now;
        t.phase0 = t.last = now;
        t.open = true;
    }
    // Vieneto pabaiga po šios fazės apdorojimo (Ready / Lapped / OutputError); committed — tikras commit.
    void unit_done(engine::Index i, std::uint64_t now, bool committed) {
        S& t = s_[i & 1];
        if (t.open) t.last = now;
        if (committed) t.commit = now;
    }
    std::uint64_t first(engine::Index i) const { return s_[i & 1].first; }
    Result end(engine::Index i, std::uint64_t now) {
        S& t = s_[i & 1];
        close(t);
        Result r;
        r.processed = t.phases > 0;
        r.phases = t.phases;
        r.end = t.commit ? t.commit : now;
        r.first = r.processed ? t.first : r.end;
        r.proc = t.proc;
        r.wait = r.first > t.wake ? r.first - t.wake : 0;
        const std::uint64_t span = r.end > r.first ? r.end - r.first : 0;
        r.held = span > t.proc ? span - t.proc : 0;
        return r;
    }

private:
    struct S {
        std::uint64_t wake = 0, first = 0, phase0 = 0, last = 0, commit = 0, proc = 0;
        std::uint32_t phases = 0;
        bool open = false;
    };
    static void close(S& t) {
        if (!t.open) return;
        t.proc += t.last > t.phase0 ? t.last - t.phase0 : 0;
        t.open = false;
    }
    S s_[2];
};

// Testinė išėjimo klaidų injekcija (--test-out-error open|commit, paslėpta): apgaubia tikrą išėjimą; nuo pirmo open po
// `after_s` (TAI) `count` indeksų iš eilės (0 — visam laikui) open grąžina Error arba commit — false (slot'as atšaukiamas).
// Tikrina, kad išėjimo klaida nėra tyli (skaitiklis output_errors, ilgalaikė — exit 2 su pranešimu).
class FaultyOutput final : public engine::OutputPort {
public:
    FaultyOutput(engine::OutputPort& inner, const Opts& o);
    engine::OpenResult open(engine::Index i, void*& slot) override;
    bool commit() override;
    void cancel() override { in_.cancel(); }
    bool fill_gap(engine::Index i, engine::GapAction a, engine::Index from) override { return in_.fill_gap(i, a, from); }
    bool flush_gaps() override { return in_.flush_gaps(); }
    bool interlaced() const override { return in_.interlaced(); }
    std::uint32_t slot_count() const override { return in_.slot_count(); }

private:
    bool active(engine::Index i);
    engine::OutputPort& in_;
    bool open_mode_;
    std::uint64_t from_tai_;
    std::uint32_t count_;
    engine::Index first_ = engine::kNoIndex, cur_ = engine::kNoIndex;
};

// Išėjimų prievadai varikliui: be --test-out-error — patys prievadai, su ja — FaultyOutput apvalkalai (laikomi `keep`).
std::vector<engine::OutputPort*> output_ports(std::vector<engine::OutputPort*> ports, const Opts& o,
                                              std::vector<FaultyOutput>& keep);

}  // namespace mxlv::func
