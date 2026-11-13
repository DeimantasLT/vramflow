// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/engine_glue.hpp"

#include <cuda_runtime_api.h>
#include <mxl/time.h>

#include <cstdio>
#include <cstdlib>
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
        c.start = engine::StartMode::WaitAll;
    }
    c.on_loss = gap_policy(o.on_loss);  // K3: abiem ribų režimais (gen, dec, gfxgen — Legacy; logo, composite, enc — Isolation)
    c.hold_ns = static_cast<std::int64_t>(o.hold_ms) * 1'000'000;
    std::string warn;
    if (engine::clamp_deadline(c.cuts, warn)) std::fprintf(stderr, "[vramflow] %s\n", warn.c_str());
    c.duration_s = o.duration_s;
    c.warmup_s = o.warmup_s;
    c.input_wait_ns = o.input_wait_ms * 1'000'000;
    c.rt_priority = o.rt_priority;
    parse_cpu_list(o.cpu_affinity, c.cpu_affinity);  // patikrinta CLI
    return c;
}

engine::GapPolicy gap_policy(const std::string& s) {
    if (s == "freeze") return engine::GapPolicy::Freeze;
    if (s == "black") return engine::GapPolicy::Black;
    if (s == "leave") return engine::GapPolicy::Leave;
    return engine::GapPolicy::Invalid;
}

GapFill::GapFill(engine::GapPolicy p, bool cuda) : policy_(p), cuda_(cuda) {
    if (!cuda_) return;
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) stream_ = s;
    else cudaGetLastError();  // be stream'o kopija nepavyks (copy grąžina false → commit_errors), ne tyliai
}

GapFill::~GapFill() {
    for (Black& b : black_)
        if (cuda_ && b.p) cudaFree(b.p);
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

bool GapFill::prepare(const vramflow::gfx::Raster& r, std::string& err) {
    if (policy_ != engine::GapPolicy::Freeze && policy_ != engine::GapPolicy::Black) return true;  // invalid / leave — nereikia
    if (black(r)) return true;
    Black b;
    b.r = r;
    b.host = engine::black_grain(r);
    if (cuda_) {
        if (!stream_) {
            err = "GapFill: cudaStreamCreate nepavyko";
            return false;
        }
        const cudaError_t e1 = cudaMalloc(&b.p, b.host.size());
        const cudaError_t e2 = e1 == cudaSuccess ? cudaMemcpy(b.p, b.host.data(), b.host.size(), cudaMemcpyHostToDevice) : e1;
        if (e2 != cudaSuccess) {
            cudaGetLastError();
            if (b.p) cudaFree(b.p);
            err = std::string("GapFill: juodo grain'o VRAM buferis: ") + cudaGetErrorString(e2);
            return false;
        }
        b.host.clear();
        b.host.shrink_to_fit();
        black_bytes_ += vramflow::gfx::grain_bytes(r);
    } else {
        b.p = b.host.data();
        black_bytes_ += b.host.size();
    }
    black_.push_back(std::move(b));
    if (!cuda_) black_.back().p = black_.back().host.data();  // vektorius perkeltas — rodyklė iš naujo
    return true;
}

engine::GrainCopy GapFill::copy() const {
    if (!cuda_) return engine::host_copy();
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    return [s](void* dst, const void* src, std::size_t n) {
        if (!s) return false;
        if (cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s) == cudaSuccess) return true;
        cudaGetLastError();
        return false;
    };
}

std::function<bool()> GapFill::flush() const {
    if (!cuda_) return {};
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    return [s] {
        if (s && cudaStreamSynchronize(s) == cudaSuccess) return true;
        cudaGetLastError();
        return false;
    };
}

const void* GapFill::black(const vramflow::gfx::Raster& r) const {
    for (const Black& b : black_)
        if (b.r == r) return b.p;
    return nullptr;
}

bool parse_cpu_list(const std::string& s, std::vector<int>& out) {
    out.clear();
    std::size_t p = 0;
    while (p < s.size()) {
        const std::size_t e = s.find(',', p);
        const std::string tok = s.substr(p, e == std::string::npos ? std::string::npos : e - p);
        p = e == std::string::npos ? s.size() : e + 1;
        if (tok.empty()) return false;
        char* end = nullptr;
        const long a = std::strtol(tok.c_str(), &end, 10);
        long b = a;
        if (*end == '-') b = std::strtol(end + 1, &end, 10);
        if (*end != '\0' || a < 0 || b < a || b > 1023) return false;
        for (long c = a; c <= b; ++c) out.push_back(static_cast<int>(c));
    }
    return true;
}

bool strict_exit(const Opts& o) { return o.strict_exit == "on" || (o.strict_exit == "auto" && o.duration_s > 0); }

int exit_code_for(const Opts& o, int rc, bool loss, std::uint32_t failed_units, Stats& st) {
    const bool strict = strict_exit(o);
    st.strict = strict ? 1 : 0;
    st.any_loss = loss;
    if (rc != 0) return exit_code(rc);
    if (failed_units) return 2;  // abiem režimais (D8): sustabdant dar sugedęs vienetas
    if (!strict) return 0;
    return loss ? 4 : 0;
}

void set_release(std::vector<engine::MxlVideoOutput>& ports, std::vector<VideoOut>& outs, mxlInstance inst,
                 std::function<bool(std::size_t)> reopen) {
    for (std::size_t n = 0; n < ports.size() && n < outs.size(); ++n) {
        ports[n].release_fn = [&outs, n, inst] { outs[n].release(inst); };
        if (reopen) ports[n].reopen_fn = [reopen, n] { return reopen(n); };
    }
}

EngineStats::EngineStats(const Opts& o, const char* role)
    : lines(o, role), hub(o.stats_interval_s > 0 ? o.stats_interval_s : 1.0, &lines) {}

void EngineStats::finish() {
    hub.flush(mxlGetTime());
    hub.stop();
}

int engine_rc(const engine::RunResult& r, Stats& st) {
    switch (r.reason) {
        case engine::EndReason::Fatal:
            if (st.fatal.empty()) st.fatal = r.fatal;
            return r.exit_code ? r.exit_code : 2;
        case engine::EndReason::FormatChange:
            st.fatal = r.fatal;
            return kExitFormatChange;
        case engine::EndReason::InputsTimeout:  // būtinas įėjimas neprisijungė per --input-wait-ms (exit 4, kaip dokumentuota)
            st.fatal = r.fatal;
            return kExitInputs;
        case engine::EndReason::Stopped:
        case engine::EndReason::Duration: return 0;
    }
    return 0;
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
    for (std::size_t n = 0; n < ports.size(); ++n) {
        if (o.test_out_error_flow >= 0 && n != static_cast<std::size_t>(o.test_out_error_flow)) continue;
        keep.emplace_back(*ports[n], o);
        ports[n] = &keep.back();
    }
    return ports;
}

void to_summary(const engine::StatsHub& hub, const engine::TickEngine& eng, const engine::RunResult& r, std::uint32_t units,
                SkipScope scope, Stats& st) {
    if (r.first_index != engine::kNoIndex) st.first_index = r.first_index;
    if (r.last_index != engine::kNoIndex) st.last_index = r.last_index;
    // Įspėjimai ir vienetų gedimai — jau pranešti iš karto (variklio įvykiai, StatsLines); čia — santraukos raktai.
    if (r.unit_failures) {  // K17: sugedę vienetai (kiti tęsė) — gedimai, atkūrimai, likę sugedę (įvykio raktai)
        st.proc["unit_failures"] = r.unit_failures;
        if (r.unit_recoveries) st.proc["unit_recoveries"] = r.unit_recoveries;
        if (r.unit_retries) st.proc["unit_retries"] = r.unit_retries;
        if (r.failed_units) {
            st.proc["failed_units"] = r.failed_units;
            std::fprintf(stderr, "[vramflow] sustabdant sugedusių vienetų: %u (pirmas gedimas: %s)\n", r.failed_units, r.failed_why.c_str());
        }
        for (std::uint32_t n = 0; n < units && n < st.flows.size(); ++n) {
            const engine::UnitCounters& u = hub.unit(eng.unit_id(n));
            if (u.failures) st.flows[n].c["unit_failures"] = u.failures;
            if (u.recoveries) st.flows[n].c["unit_recoveries"] = u.recoveries;
            if (eng.unit_failed(n)) st.flows[n].c["unit_failed"] = 1;
        }
    }
    const engine::EngineCounters& e = hub.engine();
    if (scope == SkipScope::Process) st.proc["clock_steps"] = e.clock_steps;  // gen visada turėjo; logo ir kt. — ne (raktų aibė)
    if (e.clock_backward) st.proc["clock_backward"] += e.clock_backward;
    auto put = [](auto& m, const char* k, std::uint64_t v) {
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
        if (fr) continue;  // enc — be MXL išėjimų (spragų politika taikoma tik MXL rašytojams)
        // K3: kiek spragų commit'inta kokiu veiksmu (išėjimo skaitikliai; vienas išėjimas vienetui). gap_startup — paleidimo
        // spragos (įėjimas dar niekada neprisijungė, --start immediate) — ne praradimas.
        const engine::OutCounters& oc = hub.out(eng.out_id(n, 0));
        put(st.flows[n].c, "gap_invalid", oc.gap_invalid);
        put(st.flows[n].c, "gap_frozen", oc.gap_frozen);
        put(st.flows[n].c, "gap_black", oc.gap_black);
        put(st.flows[n].c, "gap_startup", oc.gap_startup);
    }
}

}  // namespace mxlv::func
