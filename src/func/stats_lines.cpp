// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/stats_lines.hpp"

#include <dlfcn.h>
#include <nvml.h>

#include <mxl/time.h>

#include <cinttypes>
#include <cstring>
#include <ctime>

#include "func/func_common.hpp"
#include "mxlio/common_io.hpp"

namespace mxlv::func {

namespace {
constexpr std::uint64_t kTaiUtcNs = 37'000'000'000ull;  // TAI − UTC (nuo 2017-01-01)

using FnInit = nvmlReturn_t (*)();
using FnByUuid = nvmlReturn_t (*)(const char*, nvmlDevice_t*);
using FnByIndex = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using FnUtil = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*);
using FnCodec = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, unsigned int*);
using FnMem = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
using FnShutdown = nvmlReturn_t (*)();

void ms(std::string& j, const char* k, std::uint64_t ns, bool comma = true) {
    char b[64];
    std::snprintf(b, sizeof b, "%s\"%s\":%.3f", comma ? "," : "", k, static_cast<double>(ns) / 1e6);
    j += b;
}
void num(std::string& j, const char* k, std::uint64_t v, bool comma = true) {
    char b[64];
    std::snprintf(b, sizeof b, "%s\"%s\":%" PRIu64, comma ? "," : "", k, v);
    j += b;
}
void str(std::string& j, const char* k, const std::string& v, bool comma = true) {
    if (comma) j += ",";
    j += "\"";
    j += k;
    j += "\":";
    mxlv::json_escape_into(j, v);
}
const std::string& name_of(const std::vector<std::string>* v, std::uint32_t id) {
    static const std::string empty;
    return v != nullptr && id < v->size() ? (*v)[id] : empty;
}
}  // namespace

std::string utc_of_tai(std::uint64_t tai_ns) {
    const std::uint64_t utc = tai_ns > kTaiUtcNs ? tai_ns - kTaiUtcNs : 0;
    const auto sec = static_cast<std::time_t>(utc / 1'000'000'000ull);
    std::tm tm{};
    gmtime_r(&sec, &tm);
    char b[48];
    const std::size_t n = std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", &tm);
    std::snprintf(b + n, sizeof b - n, ".%03uZ", static_cast<unsigned>((utc / 1'000'000ull) % 1000));
    return b;
}

void log_line(const char* role, const std::string& text) {
    std::fprintf(stderr, "[vramflow %s] %s: %s\n", utc_of_tai(mxlGetTime()).c_str(), role, text.c_str());
}

// --- GpuProbe -------------------------------------------------------------------------------------------------------
GpuProbe::~GpuProbe() {
    if (f_shutdown_) reinterpret_cast<FnShutdown>(f_shutdown_)();
    if (lib_) dlclose(lib_);
}

bool GpuProbe::open(const std::string& device_uuid, std::string& err) {
    lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib_) {
        err = "libnvidia-ml.so.1 nerasta";
        return false;
    }
    auto sym = [&](const char* n) { return dlsym(lib_, n); };
    auto init = reinterpret_cast<FnInit>(sym("nvmlInit_v2"));
    auto by_uuid = reinterpret_cast<FnByUuid>(sym("nvmlDeviceGetHandleByUUID"));
    auto by_index = reinterpret_cast<FnByIndex>(sym("nvmlDeviceGetHandleByIndex_v2"));
    f_util_ = sym("nvmlDeviceGetUtilizationRates");
    f_enc_ = sym("nvmlDeviceGetEncoderUtilization");
    f_dec_ = sym("nvmlDeviceGetDecoderUtilization");
    f_mem_ = sym("nvmlDeviceGetMemoryInfo");
    if (!init || !by_index || init() != NVML_SUCCESS) {
        err = "nvmlInit nepavyko";
        return false;
    }
    f_shutdown_ = sym("nvmlShutdown");
    nvmlDevice_t d{};
    bool got = false;
    if (!device_uuid.empty() && by_uuid) got = by_uuid(device_uuid.c_str(), &d) == NVML_SUCCESS;
    if (!got) got = by_index(0, &d) == NVML_SUCCESS;
    if (!got) {
        err = "NVML įrenginys nerastas";
        return false;
    }
    dev_ = d;
    return true;
}

GpuSample GpuProbe::sample() {
    GpuSample s;
    if (!dev_) return s;
    auto d = static_cast<nvmlDevice_t>(dev_);
    s.valid = true;
    nvmlUtilization_t u{};
    if (f_util_ && reinterpret_cast<FnUtil>(f_util_)(d, &u) == NVML_SUCCESS) {
        s.sm_pct = static_cast<int>(u.gpu);
        s.mem_pct = static_cast<int>(u.memory);
    }
    unsigned int v = 0, per = 0;
    if (f_enc_ && reinterpret_cast<FnCodec>(f_enc_)(d, &v, &per) == NVML_SUCCESS) s.nvenc_pct = static_cast<int>(v);
    if (f_dec_ && reinterpret_cast<FnCodec>(f_dec_)(d, &v, &per) == NVML_SUCCESS) s.nvdec_pct = static_cast<int>(v);
    nvmlMemory_t m{};
    if (f_mem_ && reinterpret_cast<FnMem>(f_mem_)(d, &m) == NVML_SUCCESS) {
        s.vram_used = m.used;
        s.vram_total = m.total;
    }
    return s;
}

// --- StatsLines -----------------------------------------------------------------------------------------------------
StatsLines::StatsLines(const Opts& o, std::string role)
    : role_(std::move(role)), run_id_(o.run_id), device_uuid_(o.device_uuid), interval_s_(o.stats_interval_s) {
    if (o.stats_file == "-") {
        f_ = stderr;
    } else if (!o.stats_file.empty()) {
        f_ = std::fopen(o.stats_file.c_str(), "a");
        own_ = f_ != nullptr;
        if (!f_) log_line(role_.c_str(), "--stats-file " + o.stats_file + ": nepavyko atidaryti (" + std::strerror(errno) + ")");
    }
    gpu_wanted_ = f_ != nullptr;
    buf_.reserve(1 << 16);
}

StatsLines::~StatsLines() {
    if (own_) std::fclose(f_);
}

void StatsLines::event(const engine::Event& e, const engine::StatsNames& n) {
    using engine::EventKind;
    const std::vector<std::string>* names = nullptr;
    switch (e.kind) {
        case EventKind::UnitFailed:
        case EventKind::UnitRetryFailed:
        case EventKind::UnitRecovered: names = n.unit; break;
        case EventKind::InputAttached:
        case EventKind::InputReattached:
        case EventKind::InputDetached:
        case EventKind::InputStalled:
        case EventKind::InputResumed:
        case EventKind::FormatChange:
        case EventKind::LayerAbsent:
        case EventKind::LayerPresent: names = n.in; break;
        case EventKind::OutputError:
        case EventKind::OutputRecovered: names = n.out; break;
        case EventKind::ClockStep:
        case EventKind::Warning: break;
    }
    const std::string& who = name_of(names, e.id);
    const std::string t = utc_of_tai(e.tai);
    std::string idx = e.index == ~std::uint64_t{0} ? std::string{} : " (indeksas " + std::to_string(e.index) + ")";
    std::fprintf(stderr, "[vramflow %s] %s: %s%s%s: %s%s\n", t.c_str(), role_.c_str(), engine::to_string(e.kind),
                 who.empty() ? "" : " ", who.c_str(), e.text, idx.c_str());
    if (!f_) return;
    std::string j = "{\"v\":1";
    str(j, "event", engine::to_string(e.kind));
    str(j, "t", t);
    num(j, "tai", e.tai);
    str(j, "role", role_);
    str(j, "run", run_id_);
    if (!who.empty()) str(j, "id", who);
    if (e.index != ~std::uint64_t{0}) num(j, "index", e.index);
    if (e.value != 0) j += ",\"value\":" + std::to_string(e.value);
    str(j, "text", e.text);
    j += "}\n";
    std::fputs(j.c_str(), f_);
    std::fflush(f_);
}

void StatsLines::publish(const engine::Snapshot& s) {
    if (!f_) return;
    if (gpu_wanted_ && !gpu_tried_) {
        gpu_tried_ = true;
        std::string err;
        gpu_ok_ = gpu_.open(device_uuid_, err);
    }
    std::string& j = buf_;
    j.clear();
    j += "{\"v\":1";
    str(j, "t", utc_of_tai(s.t_end_tai));
    num(j, "tai", s.t_end_tai);
    num(j, "seq", s.seq);
    str(j, "role", role_);
    str(j, "run", run_id_);
    {
        char b[64];
        std::snprintf(b, sizeof b, ",\"interval_s\":%.3f", static_cast<double>(s.t_end_tai - s.t_begin_tai) / 1e9);
        j += b;
    }
    str(j, "state", engine::to_string(s.health));
    num(j, "dropped_lines", s.dropped_lines);
    const std::size_t ncnt = s.counter_names ? s.counter_names->size() : 0;
    j += ",\"flows\":[";
    for (std::size_t u = 0; u < s.unit.size(); ++u) {
        const engine::UnitSnap& x = s.unit[u];
        const engine::UnitCounters& d = x.d;
        if (u) j += ",";
        j += "{";
        num(j, "n", u, false);
        str(j, "id", name_of(s.unit_names, static_cast<std::uint32_t>(u)));
        str(j, "state", engine::to_string(x.health));
        j += std::string(",\"failed\":") + (x.failed ? "true" : "false");
        num(j, "ok", d.ok + d.late);  // pateikta turinio (A + B / C fazės) — Σ eilutėse == santraukos fields_ok
        num(j, "late_commit", d.late);
        num(j, "lost", d.gaps - d.gaps_clock - d.gaps_no_content - d.gaps_upstream - d.gaps_align + d.lapped + d.late_skips);
        num(j, "gaps", d.gaps);
        num(j, "startup", d.startup);
        num(j, "upstream", d.gaps_upstream);
        num(j, "align", d.gaps_align);
        num(j, "lapped", d.lapped);
        num(j, "late_skips", d.late_skips);
        num(j, "clock_skips", d.clock_skips + d.gaps_clock);
        num(j, "warmup_skips", d.warmup_skips);
        num(j, "no_content", d.gaps_no_content);
        num(j, "failures", d.failures);
        num(j, "retries", d.retries);
        num(j, "recoveries", d.recoveries);
        j += ",\"inputs\":[";
        bool first = true;
        for (std::size_t k = 0; k < s.in.size(); ++k) {
            const engine::InSnap& in = s.in[k];
            if (in.unit != u) continue;
            if (!first) j += ",";
            first = false;
            j += "{";
            str(j, "id", name_of(s.in_names, static_cast<std::uint32_t>(k)), false);
            str(j, "state", engine::to_string(in.state));
            str(j, "health", engine::to_string(in.health));
            num(j, "ok", in.d.ok);
            num(j, "late", in.d.late);
            num(j, "lost", in.d.lost);
            num(j, "invalid", in.d.invalid);
            num(j, "too_late", in.d.too_late);
            num(j, "read_errors", in.d.read_errors);
            num(j, "stalls", in.d.stalls);
            num(j, "detaches", in.d.detaches);
            num(j, "reattached", in.d.reattached);
            ms(j, "lat_p50_ms", in.arrival_p50_ns);
            ms(j, "lat_p99_ms", in.arrival_p99_ns);
            ms(j, "lat_max_ms", in.arrival_max_ns);
            j += "}";
        }
        j += "],\"outputs\":[";
        first = true;
        for (std::size_t k = 0; k < s.out.size(); ++k) {
            const engine::OutSnap& o = s.out[k];
            if (o.unit != u) continue;
            if (!first) j += ",";
            first = false;
            j += "{";
            str(j, "id", name_of(s.out_names, static_cast<std::uint32_t>(k)), false);
            str(j, "health", engine::to_string(o.health));
            num(j, "ok", o.d.ok);
            num(j, "late", o.d.late);
            num(j, "gaps", o.d.gap_invalid + o.d.gap_frozen + o.d.gap_black + o.d.gap_left);
            num(j, "gap_startup", o.d.gap_startup);
            num(j, "gap_upstream", o.d.gap_upstream);
            num(j, "gap_align", o.d.gap_align);
            num(j, "lapped", o.d.lapped);
            num(j, "late_skips", o.d.late_skips);
            num(j, "commit_errors", o.d.commit_errors);
            ms(j, "lat_p50_ms", o.commit_p50_ns);
            ms(j, "lat_p99_ms", o.commit_p99_ns);
            ms(j, "lat_max_ms", o.commit_max_ns);
            j += "}";
        }
        j += "]";
        if (ncnt) {
            j += ",\"fn\":{";
            for (std::size_t c = 0; c < ncnt; ++c) num(j, (*s.counter_names)[c].c_str(), s.fn[u * ncnt + c], c != 0);
            j += "}";
        }
        j += "}";
    }
    j += "]";
    if (!s.hist.empty() && s.hist_names) {
        j += ",\"hist\":{";
        for (std::size_t k = 0; k < s.hist.size(); ++k) {
            if (k) j += ",";
            j += "\"" + (*s.hist_names)[k] + "\":{";
            num(j, "count", s.hist[k].count, false);
            ms(j, "p50_ms", s.hist[k].p50_ns);
            ms(j, "p99_ms", s.hist[k].p99_ns);
            j += "}";
        }
        j += "}";
    }
    const engine::EngineCounters& e = s.engine;
    j += ",\"engine\":{";
    num(j, "ticks", e.ticks, false);
    num(j, "process_calls", e.process_calls);
    num(j, "guard_overrun", e.guard_overrun);
    num(j, "carry_overrun", e.carry_overrun);
    num(j, "cut_batches", e.cut_batches);
    ms(j, "held_ms_max", e.held_ns_max);
    j += "},\"clock\":{";
    num(j, "steps", e.clock_steps, false);
    num(j, "backward", e.clock_backward);
    j += "}";
    const GpuSample g = gpu_ok_ ? gpu_.sample() : GpuSample{};
    if (!g.valid) {
        j += ",\"gpu\":null";
    } else {
        auto pct = [&](const char* k, int v) { j += std::string(",\"") + k + "\":" + (v < 0 ? std::string("null") : std::to_string(v)); };
        j += ",\"gpu\":{\"valid\":true";
        pct("sm_pct", g.sm_pct);
        pct("mem_pct", g.mem_pct);
        pct("nvenc_pct", g.nvenc_pct);
        pct("nvdec_pct", g.nvdec_pct);
        num(j, "vram_used_mb", g.vram_used >> 20);
        num(j, "vram_total_mb", g.vram_total >> 20);
        j += "}";
    }
    j += "}\n";
    std::fputs(j.c_str(), f_);
    std::fflush(f_);
    lines_.fetch_add(1, std::memory_order_release);
}

}  // namespace mxlv::func
