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

std::string cuda_options(int dev) {
    return R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + mxlv::cuda_uuid_string(dev) + R"("}})";
}

double warm_tai_after(double warmup_s) { return static_cast<double>(mxlGetTime()) + warmup_s * 1e9; }

runtime::OpenPolicy open_policy(const Opts& o) { return runtime::policy_from_cli(o.input_wait_ms); }

runtime::FormatChange format_change(const Opts& o) {
    return o.on_format_change == "wait" ? runtime::FormatChange::Wait : runtime::FormatChange::Exit;
}

runtime::DomainLease& domain_lease() {
    static runtime::DomainLease l;
    return l;
}

void lease_output(const std::string& flow_id) {
    if (domain_lease().held() && !domain_lease().add_flow(flow_id))
        std::fprintf(stderr, "[vramflow] nepavyko užregistruoti flow %s nuomoje\n", flow_id.c_str());
}

std::string owner_note(const Opts& o, const std::string& flow_id) {
    const std::string who = runtime::DomainLease::flow_owner(o.domain, flow_id);
    return " — flow " + flow_id + " jau turi gyvą writer'į" + (who.empty() ? " (savininkas nežinomas: ne vramflow-func arba be nuomos)" : " (run_id " + who + ")");
}

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
    std::string j = "{\"role\":\"" + role + "\",\"versions\":" + mxlv::versions_json() + ",\"format\":\"" + vramflow::format_name(o.fmt) + "\",\"run_id\":\"" + o.run_id + "\",\"flows_n\":" + std::to_string(o.flows) +
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
