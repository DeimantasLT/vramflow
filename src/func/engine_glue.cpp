// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/engine_glue.hpp"

#include <mxl/time.h>

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

FaultyOutput::FaultyOutput(engine::OutputPort& inner, const Opts& o)
    : in_(inner), open_mode_(o.test_out_error == "open"),
      from_tai_(mxlGetTime() + static_cast<std::uint64_t>(o.test_out_error_after_s * 1e9)), count_(o.test_out_error_count) {}

bool FaultyOutput::active(engine::Index i) {
    if (first_ == engine::kNoIndex) {
        if (mxlGetTime() < from_tai_) return false;
        first_ = i;
    }
    return i >= first_ && (count_ == 0 || i < first_ + count_);
}

engine::OpenResult FaultyOutput::open(engine::Index i, void*& slot) {
    cur_ = i;
    if (open_mode_ && active(i)) return engine::OpenResult::Error;
    return in_.open(i, slot);
}

bool FaultyOutput::commit() {
    if (!open_mode_ && active(cur_)) {
        in_.cancel();
        return false;
    }
    return in_.commit();
}

std::vector<engine::OutputPort*> output_ports(std::vector<engine::OutputPort*> ports, const Opts& o,
                                              std::vector<FaultyOutput>& keep) {
    if (o.test_out_error.empty()) return ports;
    keep.clear();
    keep.reserve(ports.size());
    for (auto*& p : ports) {
        keep.emplace_back(*p, o);
        p = &keep.back();
    }
    return ports;
}

void to_summary(const engine::StatsHub& hub, const engine::TickEngine& eng, const engine::RunResult& r, std::uint32_t units,
                SkipScope scope, Stats& st) {
    if (r.first_index != engine::kNoIndex) st.first_index = r.first_index;
    if (r.last_index != engine::kNoIndex) st.last_index = r.last_index;
    const engine::EngineCounters& e = hub.engine();
    if (scope == SkipScope::Process) st.proc["clock_steps"] = e.clock_steps;  // gen visada turėjo; logo ir kt. — ne (raktų aibė)
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
    const bool fr = scope == SkipScope::PerFlowFrames;
    for (std::uint32_t n = 0; n < units && n < st.flows.size(); ++n) {
        const engine::UnitCounters& u = hub.unit(eng.unit_id(n));
        put(st.flows[n].c, fr ? "late_skips_frames" : "late_skips", u.late_skips);
        put(st.flows[n].c, fr ? "warmup_skips_frames" : "warmup_skips", u.warmup_skips);
        put(st.flows[n].c, fr ? "clock_skips_frames" : "clock_skips", u.clock_skips);
    }
}

}  // namespace mxlv::func
