// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/func_common.hpp"

#include <cuda_runtime_api.h>
#include <signal.h>

#include <mxl/time.h>

#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <thread>

#include "mxlio/common_io.hpp"

namespace mxlv::func {

std::atomic<bool>& stop_flag() {
    static std::atomic<bool> s{false};
    return s;
}

void install_signals() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    std::thread([set] {
        int sig = 0;
        sigwait(&set, &sig);
        stop_flag() = true;
    }).detach();
}

int cuda_device(const std::string& uuid) { return mxlv::cuda_device_by_uuid(uuid); }

std::string storage_options(const Opts& o, int dev) { return o.storage == "host" ? std::string{} : cuda_options(dev); }

ThreadPool::ThreadPool(int threads) {
    for (int t = 1; t < threads; ++t) workers_.emplace_back([this] { worker(); });  // kviečianti gija — taip pat darbininkas
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard g(m_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::dispatch(int n) {
    {
        std::lock_guard g(m_);
        n_ = n;
        next_ = 0;
        active_ = static_cast<int>(workers_.size());
        ++gen_;
    }
    cv_.notify_all();
    for (int i; (i = next_.fetch_add(1)) < n;) job_(i);
    std::unique_lock lk(m_);
    done_cv_.wait(lk, [this] { return active_ == 0; });
}

void ThreadPool::worker() {
    std::uint64_t seen = 0;
    for (;;) {
        int n = 0;
        {
            std::unique_lock lk(m_);
            cv_.wait(lk, [&] { return quit_ || gen_ != seen; });
            if (quit_) return;
            seen = gen_;
            n = n_;
        }
        for (int i; (i = next_.fetch_add(1)) < n;) job_(i);
        {
            std::lock_guard g(m_);
            if (--active_ == 0) done_cv_.notify_one();
        }
    }
}

std::string cuda_options(int dev) {
    return R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + mxlv::cuda_uuid_string(dev) + R"("}})";
}

double warm_tai_after(double warmup_s) { return static_cast<double>(mxlGetTime()) + warmup_s * 1e9; }

namespace {
std::string hist_json(const Histogram& h) {
    char b[256];
    std::snprintf(b, sizeof b, "{\"count\":%" PRIu64 ",\"p50_ms\":%.3f,\"p99_ms\":%.3f,\"p999_ms\":%.3f,\"max_ms\":%.3f}", h.count(),
                  h.quantile(0.5) / 1e6, h.quantile(0.99) / 1e6, h.quantile(0.999) / 1e6, h.max() / 1e6);
    return b;
}
std::string map_json(const std::map<std::string, std::uint64_t>& m) {
    std::string j = "{";
    bool first = true;
    for (const auto& [k, v] : m) {
        if (!first) j += ",";
        first = false;
        j += "\"" + k + "\":" + std::to_string(v);
    }
    return j + "}";
}
}  // namespace

void write_summary(const Opts& o, const std::string& role, const Stats& s, const std::string& extra_json) {
    std::error_code ec;
    std::filesystem::create_directories(o.out, ec);
    std::string j = "{\"role\":\"" + role + "\",\"versions\":" + mxlv::versions_json() + ",\"flows_n\":" + std::to_string(o.flows) +
                    ",\"duration_s\":" + std::to_string(o.duration_s) + ",\"exit_code\":" + std::to_string(s.exit_code) +
                    ",\"first_index\":" + std::to_string(s.first_index == ~0ull ? 0 : s.first_index) +
                    ",\"last_index\":" + std::to_string(s.last_index) + ",\"proc\":" + map_json(s.proc) + ",\"hist\":{";
    bool first = true;
    for (const auto& [k, h] : s.hist) {
        if (!first) j += ",";
        first = false;
        j += "\"" + k + "\":" + hist_json(h);
    }
    j += "},\"flows\":[";
    for (std::size_t i = 0; i < s.flows.size(); ++i) {
        if (i) j += ",";
        j += map_json(s.flows[i].c);
    }
    j += "]";
    if (!s.fatal.empty()) {
        j += ",\"fatal\":";
        mxlv::json_escape_into(j, s.fatal);
    }
    if (!extra_json.empty()) j += "," + extra_json;
    j += "}";
    const std::string p = o.out + "/summary-" + role + ".json";
    if (FILE* f = std::fopen(p.c_str(), "w")) {
        std::fputs(j.c_str(), f);
        std::fputc('\n', f);
        std::fclose(f);
    }
}

}  // namespace mxlv::func
