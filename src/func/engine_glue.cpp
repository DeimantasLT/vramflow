// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/engine_glue.hpp"

#include <cstdio>
#include <string>

namespace mxlv::func {

engine::EngineConfig engine_config(const Opts& o, EngineTiming t, std::int64_t read_offset_ns) {
    const vramflow::Rational rate = o.fmt.grain_rate();
    const auto deadline_ns = static_cast<std::int64_t>(o.deadline_ms) * 1'000'000;
    engine::EngineConfig c;
    if (t == EngineTiming::Legacy) {
        c = engine::EngineConfig::legacy(rate, read_offset_ns, deadline_ns);
    } else {
        c.cuts = engine::default_cuts(rate);
        c.cuts.read_offset_ns = read_offset_ns;
        c.cuts.deadline_ns = deadline_ns;
        c.stall_after = 3;
        c.late_clear_run = 8;
        c.wait = engine::WaitMode::Pivot;
        c.on_loss = engine::GapPolicy::Leave;
        c.start = engine::StartMode::WaitAll;
    }
    std::string warn;
    if (engine::clamp_deadline(c.cuts, warn)) std::fprintf(stderr, "[vramflow] %s\n", warn.c_str());
    c.duration_s = o.duration_s;
    c.warmup_s = o.warmup_s;
    c.input_wait_ns = o.input_wait_ms * 1'000'000;
    return c;
}

void to_summary(const engine::StatsHub& hub, const engine::TickEngine& eng, const engine::RunResult& r, std::uint32_t units,
                SkipScope scope, Stats& st) {
    if (r.first_index != engine::kNoIndex) st.first_index = r.first_index;
    if (r.last_index != engine::kNoIndex) st.last_index = r.last_index;
    const engine::EngineCounters& e = hub.engine();
    st.proc["clock_steps"] = e.clock_steps;
    if (e.clock_backward) st.proc["clock_backward"] += e.clock_backward;
    auto put = [](std::map<std::string, std::uint64_t>& m, const char* k, std::uint64_t v) {
        if (v) m[k] += v;
    };
    if (scope == SkipScope::Process) {
        // Visi šaltinio vienetai praleidžia tuos pačius indeksus — proceso skaitiklis = vieno vieneto
        if (units == 0) return;
        const engine::UnitCounters& u = hub.unit(eng.unit_id(0));
        put(st.proc, "late_skips", u.late_skips);
        put(st.proc, "warmup_skips", u.warmup_skips);
        put(st.proc, "clock_skips", u.clock_skips);
        return;
    }
    for (std::uint32_t n = 0; n < units && n < st.flows.size(); ++n) {
        const engine::UnitCounters& u = hub.unit(eng.unit_id(n));
        put(st.flows[n].c, "late_skips", u.late_skips);
        put(st.flows[n].c, "warmup_skips", u.warmup_skips);
        put(st.flows[n].c, "clock_skips", u.clock_skips);
    }
}

}  // namespace mxlv::func
