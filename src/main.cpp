// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// vramflow-verify: nepriklausomas MXL (PR #653 cuda-linear) teisingumo ir stabilumo patikros įrankis.
// Subkomandos: gpu-info, writer, reader, hog. --flows N = N flow'ų viename procese, gija kiekvienam handle'ui.
#include <CLI/CLI.hpp>
#include <cuda_runtime_api.h>
#include <signal.h>

#include <mxl/mxl.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "mxlio/common_io.hpp"
#include "mxlio/verify.hpp"

using namespace mxlv;

namespace {

std::atomic<bool> g_stop{false};

void start_signal_thread() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);  // paveldi visos vėliau sukurtos gijos
    std::thread([set] {
        int sig = 0;
        sigwait(&set, &sig);
        g_stop = true;
    }).detach();
}

void add_common(CLI::App* c, Options& o) {
    c->add_option("--domain", o.domain, "MXL domain (tmpfs)")->required();
    c->add_option("--storage", o.storage, "cuda-linear | host")->check(CLI::IsMember({"cuda-linear", "host"}));
    c->add_option("--device-uuid", o.device_uuid, "GPU-… (numatyta: įrenginys 0)");
    c->add_option("--width", o.width);
    c->add_option("--height", o.height);
    c->add_option("--rate-num", o.rate_num);
    c->add_option("--rate-den", o.rate_den);
    c->add_option("--flows", o.flows, "flow'ų skaičius šiame procese")->check(CLI::Range(1u, 256u));
    c->add_option("--flow-base", o.flow_base);
    c->add_option("--run-seed", o.run_seed);
    c->add_option("--duration", o.duration_s, "sekundės");
    c->add_option("--warmup", o.warmup_s, "sekundės be latencijos statistikos");
    c->add_option("--out", o.out, "išvesties katalogas");
}

int run_role(const Options& o, const std::string& role) {
    std::error_code ec;
    std::filesystem::create_directories(o.out, ec);
    bool tmpfs = false;
    if (mxlIsTmpFs(o.domain.c_str(), &tmpfs) != MXL_STATUS_OK || !tmpfs) {
        std::fprintf(stderr, "--domain %s nėra tmpfs\n", o.domain.c_str());
        return kExitUsage;
    }
    const int dev = cuda_device_by_uuid(o.device_uuid);
    if (dev < 0) {
        std::fprintf(stderr, "CUDA įrenginys nerastas (%s)\n", o.device_uuid.c_str());
        return kExitFatal;
    }
    cudaSetDevice(dev);
    cudaFree(nullptr);  // kontekstas sukuriamas iš anksto, kad nepatektų į pirmų grain'ų latenciją

    std::vector<std::unique_ptr<FlowState>> states;
    std::vector<FlowState*> raw;
    for (std::uint32_t i = 0; i < o.flows; ++i) {
        auto s = std::make_unique<FlowState>();
        s->id = flow_id(o.flow_base + i);
        s->seed = flow_seed(o.run_seed, o.flow_base + i);
        raw.push_back(s.get());
        states.push_back(std::move(s));
    }
    Reporter rep(o, role, raw, dev);
    std::atomic<bool> rep_stop{false};
    std::thread rep_thread([&] { rep.run(rep_stop); });

    const std::uint64_t end_tai = now_tai_ns() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    std::vector<std::thread> ts;
    for (std::uint32_t i = 0; i < o.flows; ++i)
        ts.emplace_back([&, i] {
            if (role == "writer") run_writer_flow(o, o.flow_base + i, *states[i], dev, g_stop, end_tai);
            else run_reader_flow(o, o.flow_base + i, *states[i], dev, g_stop, end_tai);
        });
    for (auto& t : ts) t.join();
    rep_stop = true;
    rep_thread.join();

    // Proceso kodas = svarbiausias flow'ų kodas. Svarba: verifikacija (3) > fatali (2) > konfliktas (5) > pajėgumas (4) > 0
    auto rank = [](int c) {
        switch (c) {
            case kExitVerify: return 5;
            case kExitFatal: return 4;
            case kExitConflict: return 3;
            case kExitCapacity: return 2;
            case kExitOk: return 0;
            default: return 1;
        }
    };
    int code = kExitOk;
    for (auto& s : states)
        if (rank(s->exit_code) > rank(code)) code = s->exit_code;
    rep.write_summary(versions_json(), code);
    std::fprintf(stderr, "[%s] baigta, exit %d\n", role.c_str(), code);
    return code;
}

int gpu_info() {
    int count = 0;
    cudaGetDeviceCount(&count);
    std::printf("[");
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, i);
        cudaSetDevice(i);
        std::size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        // VMM ir POSIX FD palaikymą tikrina probe/cuda_fd_probe (driver API), čia — tik runtime informacija.
        std::printf("%s{\"ordinal\":%d,\"name\":\"%s\",\"uuid\":\"%s\",\"total_mb\":%zu,\"free_mb\":%zu,\"cc\":\"%d.%d\"}",
                    i ? "," : "", i, p.name, cuda_uuid_string(i).c_str(), tot >> 20, fr >> 20, p.major, p.minor);
    }
    std::printf("]\n");
    return count > 0 ? 0 : kExitFatal;
}

// VRAM spaudimas (F8): užima atmintį blokais po 64 MiB, palieka --leave-free-mb, laiko iki SIGTERM.
int hog(std::size_t leave_free_mb) {
    cudaFree(nullptr);
    std::vector<void*> blocks;
    for (;;) {
        std::size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        if ((fr >> 20) <= leave_free_mb + 64) break;
        void* p = nullptr;
        if (cudaMalloc(&p, 64ull << 20) != cudaSuccess) break;
        cudaMemset(p, 0xA5, 64ull << 20);
        blocks.push_back(p);
    }
    std::size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    std::printf("{\"hog_blocks\":%zu,\"held_mb\":%zu,\"free_mb\":%zu}\n", blocks.size(), blocks.size() * 64, fr >> 20);
    std::fflush(stdout);
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (void* p : blocks) cudaFree(p);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    start_signal_thread();
    CLI::App app{"vramflow-verify — MXL GPU payload teisingumo ir stabilumo patikra"};
    app.require_subcommand(1);
    Options wo, ro;
    std::size_t leave_free = 512;

    app.add_subcommand("gpu-info", "GPU sąrašas JSON");
    auto* w = app.add_subcommand("writer", "rašo deterministinį turinį į N flow'ų");
    add_common(w, wo);
    w->add_flag("--gc-before-create", wo.gc_before_create);
    w->add_option("--corrupt-every", wo.inj.corrupt_every);
    w->add_option("--corrupt-header-every", wo.inj.corrupt_header_every);
    w->add_option("--skip-every", wo.inj.skip_every);
    w->add_option("--wrong-index-every", wo.inj.wrong_index_every);
    w->add_option("--stale-every", wo.inj.stale_every);
    w->add_option("--commit-before-sync-every", wo.inj.commit_before_sync_every);
    w->add_option("--late-write-ms", wo.inj.late_write_ms);
    w->add_option("--dup-every", wo.inj.dup_every);

    auto* r = app.add_subcommand("reader", "tikrina kiekvieną N flow'ų grain'ą");
    add_common(r, ro);
    r->add_option("--read-timeout-ms", ro.read_timeout_ms);
    r->add_option("--stall-timeout-ms", ro.stall_timeout_ms);
    r->add_option("--reopen-timeout-ms", ro.reopen_timeout_ms);
    r->add_option("--wait-flow-ms", ro.wait_flow_ms);
    r->add_option("--host-crosscheck-every", ro.host_crosscheck_every, "CPU patikra kas K-tą grain'ą (0 = ne)");
    r->add_option("--min-coverage", ro.min_coverage, "min. įvertintų indeksų dalis lange (mažiau → exit 4)")
        ->check(CLI::Range(0.0, 1.0));

    auto* h = app.add_subcommand("hog", "užima VRAM iki SIGTERM");
    h->add_option("--leave-free-mb", leave_free);

    CLI11_PARSE(app, argc, argv);
    if (app.got_subcommand("gpu-info")) return gpu_info();
    if (app.got_subcommand("writer")) return run_role(wo, "writer");
    if (app.got_subcommand("reader")) return run_role(ro, "reader");
    if (app.got_subcommand("hog")) return hog(leave_free);
    return kExitUsage;
}
