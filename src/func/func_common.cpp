// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/func_common.hpp"

#include <cuda_runtime_api.h>
#include <signal.h>

#include <mxl/time.h>

#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <future>
#include <thread>

#include "mxlio/common_io.hpp"

namespace mxlv::func {

std::atomic<bool>& stop_flag() {
    static std::atomic<bool> s{false};
    return s;
}

std::atomic<bool>& fast_stop_flag() {
    static std::atomic<bool> s{false};
    return s;
}

// Signalai — tik sigwait atskiroje gijoje cikle (A25): pirmas SIGINT / SIGTERM / SIGHUP / SIGQUIT — tvarkingas sustabdymas
// (stop_flag; flow'ai atlaisvinami, santrauka parašoma), antras — greitas (fast_stop_flag: eilių išleidimas nutraukiamas).
// SIGHUP (terminalas uždarytas, supervizorius) ir SIGQUIT anksčiau nužudydavo procesą be flow'ų atlaisvinimo. SIGPIPE
// ignoruojamas: uždarytas --stats-file skaitytuvas (FIFO, `| head`) ar UDP gavėjas — rašymo klaida, ne proceso mirtis.
// Variklio gija signalų negauna (užblokuoti visose gijose).
void install_signals() {
    std::signal(SIGPIPE, SIG_IGN);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    sigaddset(&set, SIGHUP);
    sigaddset(&set, SIGQUIT);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    std::thread([set] {
        for (int n = 0; n < 2; ++n) {
            int sig = 0;
            if (sigwait(&set, &sig) != 0) return;
            if (n == 0) stop_flag() = true;
            else fast_stop_flag() = true;
        }
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

std::string exists_note(mxlInstance inst, const Opts& o, const std::string& flow_id) {
    const std::string who = runtime::DomainLease::flow_owner(o.domain, flow_id);
    if (!who.empty()) return " — flow " + flow_id + " jau turi gyvą writer'į (run_id " + who + ")";
    bool active = false;
    if (mxlIsFlowActive(inst, flow_id.c_str(), &active) != MXL_STATUS_OK) return {};  // tokio flow nėra — klaida kita
    return active ? " — flow " + flow_id + " jau turi gyvą writer'į (savininkas nežinomas: ne vramflow-func arba be nuomos)"
                  : " — flow " + flow_id + " pasenęs: liko po nutrūkusio paleidimo, gyvo writer'io nėra (pašalina mxlGarbageCollectFlows)";
}

runtime::InputSlot make_input(const Opts& o, const std::string& flow_id, const vramflow::VideoFormat* expect) {
    return runtime::InputSlot(flow_id, expect, format_change(o));
}

std::vector<runtime::InputSlot*> slot_ptrs(std::vector<runtime::InputSlot>& v) {
    std::vector<runtime::InputSlot*> p;
    p.reserve(v.size());
    for (auto& s : v) p.push_back(&s);
    return p;
}

int attach_inputs(const std::vector<runtime::InputSlot*>& in, mxlInstance inst, const Opts& o, const char* who, std::string& fatal) {
    std::string err;
    if (runtime::wait_attached(in, inst, open_policy(o), stop_flag(), who, err)) return 0;
    if (stop_flag().load()) return kStoppedWaiting;  // sustabdyta belaukiant — ne klaida
    fatal = err;
    return kExitInputs;
}

int poll_inputs(const std::vector<runtime::InputSlot*>& in, mxlInstance inst, std::string& fatal) {
    const std::uint64_t now = mxlGetTime();
    for (runtime::InputSlot* s : in)
        if (s->poll(inst, now) == runtime::InputSlot::State::Mismatch) {
            fatal = s->reason();
            return kExitFormatChange;
        }
    return 0;
}

bool create_output(mxlInstance inst, VideoOut& out, const std::string& id, const std::string& def, const std::string& options,
                   const vramflow::VideoFormat& f, const Opts& o, std::string& err) {
    if (out.create(inst, def, options, err, f)) {
        lease_output(id);
        return true;
    }
    const bool exists = out.existed || out.create_status == MXL_ERR_CONFLICT;
    out.release(inst);  // created=false — prisijungta prie svetimo flow; atlaisvinam prieš tikrindami, kas jį laiko
    if (exists) err += exists_note(inst, o, id);
    return false;
}

bool create_output(mxlInstance inst, AudioOut& out, const std::string& id, const std::string& def, vramflow::Rational grain_rate,
                   const Opts& o, std::string& err) {
    if (out.create(inst, def, err, grain_rate)) {
        lease_output(id);
        return true;
    }
    const bool exists = out.existed;
    out.release(inst);
    if (exists) err += exists_note(inst, o, id);
    return false;
}

namespace {
std::string hist_json(const Histogram& h) {
    char b[256];
    std::snprintf(b, sizeof b, "{\"count\":%" PRIu64 ",\"p50_ms\":%.3f,\"p99_ms\":%.3f,\"p999_ms\":%.3f,\"max_ms\":%.3f}", h.count(),
                  h.quantile(0.5) / 1e6, h.quantile(0.99) / 1e6, h.quantile(0.999) / 1e6, h.max() / 1e6);
    return b;
}
template <class Map>
std::string map_json(const Map& m) {
    std::string j = "{";
    bool first = true;
    for (const auto& [k, v] : m) {
        if (!first) j += ",";
        first = false;
        j += "\"" + k + "\":" + std::to_string(v);
    }
    return j + "}";
}
// [[sekundė, reikšmė], …] — lateness_s / commit_lateness_s (µs), loss_s (skaičius)
std::string seconds_json(const SecondRing& m) {
    std::string j = "[";
    m.for_each([&](std::uint64_t k, std::uint64_t v) {
        if (j.size() > 1) j += ",";
        j += "[" + std::to_string(k) + "," + std::to_string(v) + "]";
    });
    return j + "]";
}
}  // namespace

namespace {
std::shared_future<std::string>& versions_future() {
    static std::shared_future<std::string> f;
    return f;
}
}  // namespace

void prefetch_versions(const Opts& o) {
    if (o.impl == "cpu") versions_future() = std::async(std::launch::async, [] { return mxlv::versions_json(false); }).share();
}

void write_summary(const Opts& o, const std::string& role, const Stats& s, const std::string& extra_json) {
    std::error_code ec;
    std::filesystem::create_directories(o.out, ec);
    // CPU variantas (C) CUDA nenaudoja: versijos — iš paleidimo metu fone paruoštų (prefetch_versions), be įrenginio laukų (A3)
    const std::string vj = versions_future().valid() ? versions_future().get() : mxlv::versions_json(o.impl != "cpu");
    std::string j = "{\"role\":\"" + role + "\",\"versions\":" + vj + ",\"format\":\"" + vramflow::format_name(o.fmt) + "\",\"run_id\":\"" + o.run_id + "\",\"flows_n\":" + std::to_string(o.flows) +
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
        std::string f = map_json(s.flows[i].c);
        if (!s.flows[i].lateness_s.empty()) {
            f.pop_back();
            f += std::string(f.size() > 1 ? "," : "") + "\"lateness_s\":" + seconds_json(s.flows[i].lateness_s) + "}";
        }
        if (!s.flows[i].p99_s.empty() || s.flows[i].cur_n() != 0) {  // + einamoji (neužbaigta) sekundė
            SecondRing r = s.flows[i].p99_s;
            if (s.flows[i].cur_n() != 0) r.max(s.flows[i].cur_sec(), s.flows[i].cur_p99());
            f.pop_back();
            f += std::string(f.size() > 1 ? "," : "") + "\"p99_s\":" + seconds_json(r) + "}";
        }
        if (!s.flows[i].loss_s.empty()) {
            f.pop_back();
            f += std::string(f.size() > 1 ? "," : "") + "\"loss_s\":" + seconds_json(s.flows[i].loss_s) + "}";
        }
        j += f;
    }
    j += "]";
    if (s.strict >= 0)  // variklio funkcijos: exit politika ir ar buvo praradimų (tarnybos režime exit 0 jų neparodo)
        j += std::string(",\"strict_exit\":") + (s.strict ? "true" : "false") + ",\"any_loss\":" + (s.any_loss ? "true" : "false");
    if (!s.commit_lateness_s.empty()) j += ",\"commit_lateness_s\":" + seconds_json(s.commit_lateness_s);
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
