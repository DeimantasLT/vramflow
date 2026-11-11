// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "mxlio/common_io.hpp"

#include <cuda_runtime_api.h>
#include <dirent.h>
#include <sys/resource.h>
#include <sys/timex.h>
#include <unistd.h>

#include <mxl/mxl.h>
#include <mxl/time.h>

#include <chrono>
#include <cinttypes>
#include <thread>

#include "common/pattern.hpp"

namespace mxlv {

std::string flow_id(std::uint32_t n) {
    char b[64];
    std::snprintf(b, sizeof b, "4d584c56-0000-4000-8000-%012x", n);
    return b;
}

std::uint64_t flow_seed(std::uint64_t run_seed, std::uint32_t n) { return splitmix64(run_seed ^ n); }

std::uint32_t v210_grain_bytes(std::uint32_t w, std::uint32_t h) { return (w + 47) / 48 * 128 * h; }

std::string flow_def_json(const Options& o, std::uint32_t n) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow-verify flow %u","label":"vramflow-verify flow %u",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-verify-%016llx:Video %u"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210",)"
                  R"("grain_rate":{"numerator":%lld,"denominator":%lld},"frame_width":%u,"frame_height":%u,)"
                  R"("interlace_mode":"progressive","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":%u,"height":%u,"bit_depth":10},)"
                  R"({"name":"Cb","width":%u,"height":%u,"bit_depth":10},)"
                  R"({"name":"Cr","width":%u,"height":%u,"bit_depth":10}]})",
                  flow_id(n).c_str(), n, n, static_cast<unsigned long long>(o.run_seed), n, static_cast<long long>(o.rate_num),
                  static_cast<long long>(o.rate_den), o.width, o.height, o.width, o.height, o.width / 2, o.height,
                  o.width / 2, o.height);
    return b;
}

std::string writer_options(const Options& o, const std::string& uuid) {
    if (o.storage == "host") return "";
    return R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + uuid + R"("}})";
}

std::uint64_t now_tai_ns() { return mxlGetTime(); }
std::uint64_t now_mono_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void json_escape_into(std::string& out, const std::string& s) {
    out.push_back('"');
    for (char ch : s) {
        if (ch == '"' || ch == '\\') {
            out.push_back('\\');
            out.push_back(ch);
        } else if (static_cast<unsigned char>(ch) < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", ch);
            out += b;
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
}

std::string versions_json() {
    std::string j = "{";
    mxlVersionType v{};
    if (mxlGetVersion(&v) == MXL_STATUS_OK) {
        char b[128];
        std::snprintf(b, sizeof b, "\"mxl\":\"%u.%u.%u\",", v.major, v.minor, v.bugfix);
        j += b;
    }
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    j += "\"cuda_runtime\":" + std::to_string(rt) + ",\"cuda_driver\":" + std::to_string(drv);
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        j += ",\"gpu\":";
        json_escape_into(j, p.name);
        j += ",\"gpu_uuid\":\"" + cuda_uuid_string(0) + "\"";
        j += ",\"gpu_cc\":\"" + std::to_string(p.major) + "." + std::to_string(p.minor) + "\"";
    }
#ifdef VRAMFLOW_CUDA_ARCH
    j += ",\"cuda_arch_build\":";
    json_escape_into(j, VRAMFLOW_CUDA_ARCH);
#endif
    return j + "}";
}

namespace {

int count_fds() {
    int n = 0;
    if (DIR* d = opendir("/proc/self/fd")) {
        while (dirent* e = readdir(d))
            if (e->d_name[0] != '.') ++n;
        closedir(d);
    }
    return n - 1;
}

long rss_kb() {
    long pages = 0, resident = 0;
    if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
        std::fclose(f);
    }
    return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

const char* kFlowCsvHeader =
    "ts_tai_ns,ts_mono_ns,flow,role,head_index,cur_index,grains_ok,grains_invalid,gap_grains,too_late_events,"
    "too_early_timeouts,stalls,lapped_discarded,reopens,index_mismatch,slot_mismatch,size_mismatch,hdr_fail,"
    "content_index_mismatch,mismatch_grains,mismatch_words,checksum_fail,dup,seq_break,host_crosscheck_fail,"
    "cuda_errors,commits,writer_late_skips,clock_backward,injections,lat_c2v_p50_us,lat_c2v_p99_us,lat_c2v_max_us,"
    "lat_e2e_p50_us,lat_e2e_p99_us,gpu_op_p50_us,gpu_op_p99_us,jitter_p99_us,stale_max_ms\n";

double us(std::uint64_t ns) { return static_cast<double>(ns) / 1000.0; }

std::string hist_json(const Histogram& h) {
    char b[256];
    std::snprintf(b, sizeof b, "{\"count\":%" PRIu64 ",\"p50_us\":%.1f,\"p99_us\":%.1f,\"p999_us\":%.1f,\"max_us\":%.1f}",
                  h.count(), us(h.quantile(0.5)), us(h.quantile(0.99)), us(h.quantile(0.999)), us(h.max()));
    return b;
}

}  // namespace

Reporter::Reporter(const Options& o, std::string role, std::vector<FlowState*> flows, int cuda_device)
    : o_(o), role_(std::move(role)), flows_(std::move(flows)), dev_(cuda_device) {
    for (std::size_t i = 0; i < flows_.size(); ++i) {
        const std::string p = o_.out + "/flow-" + std::to_string(o_.flow_base + i) + "-" + role_ + ".csv";
        FILE* f = std::fopen(p.c_str(), "w");
        if (f) std::fputs(kFlowCsvHeader, f);
        flow_csv_.push_back(f);
    }
    const std::string pp = o_.out + "/proc-" + role_ + ".csv";
    proc_csv_ = std::fopen(pp.c_str(), "w");
    if (proc_csv_)
        std::fputs("ts_tai_ns,ts_mono_ns,rss_kb,fd_count,threads,cpu_user_ms,cpu_sys_ms,vram_free_mb,vram_total_mb,"
                   "tai_minus_mono_ns,tai_offset_s\n",
                   proc_csv_);
    start_mono_ = now_mono_ns();
}

void Reporter::run(const std::atomic<bool>& stop) {
    if (dev_ >= 0) cudaSetDevice(dev_);
    auto next = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!stop.load()) {
        std::this_thread::sleep_until(next);
        next += std::chrono::seconds(1);
        tick(false);
    }
    tick(true);
}

void Reporter::tick(bool) {
    const auto tai = now_tai_ns();
    const auto mono = now_mono_ns();
    for (std::size_t i = 0; i < flows_.size(); ++i) {
        FlowState& s = *flows_[i];
        Histogram c2v, e2e, gpu, jit;
        std::uint64_t stale = 0;
        {
            std::lock_guard g(s.lat.m);
            c2v = s.lat.sec_c2v;
            e2e = s.lat.sec_e2e;
            gpu = s.lat.sec_gpu;
            jit = s.lat.sec_jitter;
            stale = s.lat.stale_max_ns;
            s.lat.sec_c2v.reset();
            s.lat.sec_e2e.reset();
            s.lat.sec_gpu.reset();
            s.lat.sec_jitter.reset();
        }
        if (FILE* f = flow_csv_[i]) {
            const Counters& c = s.c;
            std::fprintf(f,
                         "%" PRIu64 ",%" PRIu64 ",%u,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                         ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                         ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                         ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f\n",
                         tai, mono, o_.flow_base + static_cast<std::uint32_t>(i), role_.c_str(), c.head_index.load(),
                         c.cur_index.load(), c.grains_ok.load(), c.grains_invalid.load(), c.gap_grains.load(),
                         c.too_late_events.load(), c.too_early_timeouts.load(), c.stalls.load(),
                         c.lapped_discarded.load(), c.reopens.load(), c.index_mismatch.load(), c.slot_mismatch.load(),
                         c.size_mismatch.load(), c.hdr_fail.load(), c.content_index_mismatch.load(),
                         c.mismatch_grains.load(), c.mismatch_words.load(), c.checksum_fail.load(), c.dup.load(),
                         c.seq_break.load(), c.host_crosscheck_fail.load(), c.cuda_errors.load(), c.commits.load(),
                         c.writer_late_skips.load(), c.clock_backward.load(), c.injections.load(),
                         us(c2v.quantile(0.5)), us(c2v.quantile(0.99)), us(c2v.max()), us(e2e.quantile(0.5)),
                         us(e2e.quantile(0.99)), us(gpu.quantile(0.5)), us(gpu.quantile(0.99)), us(jit.quantile(0.99)),
                         static_cast<double>(stale) / 1e6);
            std::fflush(f);
        }
    }
    if (proc_csv_) {
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        std::size_t fr = 0, tot = 0;
        if (dev_ >= 0) cudaMemGetInfo(&fr, &tot);
        timex tx{};
        adjtimex(&tx);
        long threads = 0;
        if (FILE* f = std::fopen("/proc/self/status", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, f))
                if (std::sscanf(line, "Threads: %ld", &threads) == 1) break;
            std::fclose(f);
        }
        std::fprintf(proc_csv_, "%" PRIu64 ",%" PRIu64 ",%ld,%d,%ld,%ld,%ld,%zu,%zu,%" PRId64 ",%d\n", tai, mono, rss_kb(),
                     count_fds(), threads, ru.ru_utime.tv_sec * 1000 + ru.ru_utime.tv_usec / 1000,
                     ru.ru_stime.tv_sec * 1000 + ru.ru_stime.tv_usec / 1000, fr >> 20, tot >> 20,
                     static_cast<std::int64_t>(tai - mono), tx.tai);
        std::fflush(proc_csv_);
    }
}

void Reporter::write_summary(const std::string& version_json, int exit_code) {
    std::string j = "{\"role\":\"" + role_ + "\",\"versions\":" + version_json;
    j += ",\"args\":{\"domain\":";
    json_escape_into(j, o_.domain);
    j += ",\"storage\":\"" + o_.storage + "\",\"width\":" + std::to_string(o_.width) +
         ",\"height\":" + std::to_string(o_.height) + ",\"rate\":\"" + std::to_string(o_.rate_num) + "/" +
         std::to_string(o_.rate_den) + "\",\"flows\":" + std::to_string(o_.flows) +
         ",\"duration_s\":" + std::to_string(o_.duration_s) + ",\"warmup_s\":" + std::to_string(o_.warmup_s) + "}";
    j += ",\"exit_code\":" + std::to_string(exit_code) + ",\"flows\":[";
    std::uint64_t verify = 0, capacity = 0;
    Histogram m_c2v, m_e2e, m_gpu, m_jit;
    for (std::size_t i = 0; i < flows_.size(); ++i) {
        FlowState& s = *flows_[i];
        const Counters& c = s.c;
        verify += c.verification_errors();
        capacity += c.capacity_errors();
        if (i) j += ",";
        char b[4096];
        std::snprintf(
            b, sizeof b,
            "{\"flow\":%u,\"id\":\"%s\",\"storage\":\"%s\",\"slot_count\":%u,\"grain_size\":%u,\"first_index\":%" PRIu64
            ",\"last_index\":%" PRIu64 ",\"counters\":{\"grains_ok\":%" PRIu64 ",\"grains_invalid\":%" PRIu64
            ",\"gap_grains\":%" PRIu64 ",\"too_late_events\":%" PRIu64 ",\"too_early_timeouts\":%" PRIu64
            ",\"stalls\":%" PRIu64 ",\"lapped_discarded\":%" PRIu64 ",\"reopens\":%" PRIu64 ",\"index_mismatch\":%" PRIu64
            ",\"slot_mismatch\":%" PRIu64 ",\"size_mismatch\":%" PRIu64 ",\"hdr_fail\":%" PRIu64
            ",\"content_index_mismatch\":%" PRIu64 ",\"mismatch_grains\":%" PRIu64 ",\"mismatch_words\":%" PRIu64
            ",\"checksum_fail\":%" PRIu64 ",\"dup\":%" PRIu64 ",\"seq_break\":%" PRIu64 ",\"host_crosschecks\":%" PRIu64
            ",\"host_crosscheck_fail\":%" PRIu64 ",\"cuda_errors\":%" PRIu64 ",\"commits\":%" PRIu64
            ",\"writer_late_skips\":%" PRIu64 ",\"clock_backward\":%" PRIu64 ",\"injections\":%" PRIu64
            ",\"coverage_low\":%" PRIu64 "},\"coverage\":%.6f,\"exit_code\":%d",
            o_.flow_base + static_cast<unsigned>(i), s.id.c_str(), s.layout_storage.c_str(), s.slot_count, s.grain_size,
            s.first_index == ~0ull ? 0 : s.first_index, s.last_index, c.grains_ok.load(), c.grains_invalid.load(),
            c.gap_grains.load(), c.too_late_events.load(), c.too_early_timeouts.load(), c.stalls.load(),
            c.lapped_discarded.load(), c.reopens.load(), c.index_mismatch.load(), c.slot_mismatch.load(),
            c.size_mismatch.load(), c.hdr_fail.load(), c.content_index_mismatch.load(), c.mismatch_grains.load(),
            c.mismatch_words.load(), c.checksum_fail.load(), c.dup.load(), c.seq_break.load(), c.host_crosschecks.load(),
            c.host_crosscheck_fail.load(), c.cuda_errors.load(), c.commits.load(), c.writer_late_skips.load(),
            c.clock_backward.load(), c.injections.load(), c.coverage_low.load(), s.coverage, s.exit_code);
        j += b;
        {
            std::lock_guard g(s.lat.m);
            j += ",\"latency\":{\"commit_to_verified\":" + hist_json(s.lat.tot_c2v) +
                 ",\"origin_to_verified\":" + hist_json(s.lat.tot_e2e) + ",\"gpu_op\":" + hist_json(s.lat.tot_gpu) +
                 ",\"wake\":" + hist_json(s.lat.tot_wake) + ",\"period_jitter\":" + hist_json(s.lat.tot_jitter) +
                 ",\"stale_max_ms\":" + std::to_string(static_cast<double>(s.lat.stale_max_ns) / 1e6) + "}";
            // lag_grains: kiek grain'ų head buvo priekyje patikros metu (vienetai — grain'ai, ne laikas)
            char lb[160];
            std::snprintf(lb, sizeof lb, ",\"lag_grains\":{\"count\":%" PRIu64 ",\"p50\":%" PRIu64 ",\"p99\":%" PRIu64 ",\"max\":%" PRIu64 "}",
                          s.lat.tot_lag.count(), s.lat.tot_lag.quantile(0.5), s.lat.tot_lag.quantile(0.99), s.lat.tot_lag.max());
            j += lb;
            m_c2v.merge(s.lat.tot_c2v);
            m_e2e.merge(s.lat.tot_e2e);
            m_gpu.merge(s.lat.tot_gpu);
            m_jit.merge(s.lat.tot_jitter);
        }
        {
            std::lock_guard g(s.rec_m);
            j += ",\"errors\":[";
            for (std::size_t k = 0; k < s.errors.size(); ++k) {
                if (k) j += ",";
                j += "[" + std::to_string(s.errors[k].index) + "," + std::to_string(s.errors[k].bits) + "," +
                     std::to_string(s.errors[k].first_bad_word) + "]";
            }
            j += "],\"injected\":[";
            for (std::size_t k = 0; k < s.injected.size(); ++k) {
                if (k) j += ",";
                j += "[" + std::to_string(s.injected[k]) + ",\"" + s.injected_kind[k] + "\"]";
            }
            j += "],\"committed\":[";
            for (std::size_t k = 0; k < s.committed.size(); ++k) {
                if (k) j += ",";
                j += "[" + std::to_string(s.committed[k].first) + "," + std::to_string(s.committed[k].second) + "]";
            }
            j += "]";
        }
        if (!s.fatal.empty()) {
            j += ",\"fatal\":";
            json_escape_into(j, s.fatal);
        }
        j += "}";
    }
    // Sujungtos (ne per-flow medianų) latencijos visiems šio proceso flow'ams
    j += "],\"merged_latency\":{\"commit_to_verified\":" + hist_json(m_c2v) + ",\"origin_to_verified\":" + hist_json(m_e2e) +
         ",\"gpu_op\":" + hist_json(m_gpu) + ",\"period_jitter\":" + hist_json(m_jit) + "}";
    j += ",\"totals\":{\"verification_errors\":" + std::to_string(verify) +
         ",\"capacity_errors\":" + std::to_string(capacity) + "}}";
    const std::string p = o_.out + "/summary-" + role_ + ".json";
    if (FILE* f = std::fopen(p.c_str(), "w")) {
        std::fputs(j.c_str(), f);
        std::fputc('\n', f);
        std::fclose(f);
    }
    for (FILE* f : flow_csv_)
        if (f) std::fclose(f);
    if (proc_csv_) std::fclose(proc_csv_);
    flow_csv_.clear();
    proc_csv_ = nullptr;
}

}  // namespace mxlv
