// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variklio statistika (planas §6 D6). Skaitikliai — fiksuotos struktūros masyvuose, užregistruoti prieš run();
// rašo tik variklio gija.
#include <cstdint>
#include <string>
#include <vector>

#include "engine/counters.hpp"

namespace mxlv::engine {

using CounterId = std::uint32_t;
using HistId = std::uint32_t;

struct EngineCounters {
    std::uint64_t ticks = 0, process_calls = 0, guard_overrun = 0, carry_overrun = 0;
    std::uint64_t clock_steps = 0, clock_backward = 0, step_gaps = 0;
    std::uint64_t held_ns_max = 0;  // kiek ilgiausiai paruoštus vienetus laikė neparuošti (A fazė)
};

class StatsHub {
public:
    StatsHub() = default;

    // Funkcijos skaitikliai (vienetui) ir histogramos — prieš run().
    CounterId register_counter(std::string name);
    HistId register_hist(std::string name);

    // Variklis registruoja vienetus / įėjimus / išėjimus add_unit metu.
    std::uint32_t add_unit(std::string name);
    std::uint32_t add_input(std::string name);
    std::uint32_t add_output(std::string name);
    void prepare();  // run() pradžioje: matricos dydžiai fiksuojami (po to — be alokacijų)

    InCounters& in(std::uint32_t id) { return in_[id]; }
    const InCounters& in(std::uint32_t id) const { return in_[id]; }
    OutCounters& out(std::uint32_t id) { return out_[id]; }
    const OutCounters& out(std::uint32_t id) const { return out_[id]; }
    EngineCounters& engine() { return eng_; }
    const EngineCounters& engine() const { return eng_; }

    void count(std::uint32_t unit, CounterId id, std::uint64_t n) { fn_[unit * counter_names_.size() + id] += n; }
    std::uint64_t counter(std::uint32_t unit, CounterId id) const { return fn_[unit * counter_names_.size() + id]; }
    void hist(HistId, std::uint64_t) {}  // V6

    std::size_t inputs() const { return in_.size(); }
    std::size_t outputs() const { return out_.size(); }
    std::size_t units() const { return unit_names_.size(); }

private:
    std::vector<std::string> counter_names_, hist_names_, unit_names_, in_names_, out_names_;
    std::vector<InCounters> in_;
    std::vector<OutCounters> out_;
    std::vector<std::uint64_t> fn_;
    EngineCounters eng_;
};

}  // namespace mxlv::engine
