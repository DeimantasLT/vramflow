// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// keepwarm — laiko GPU virš tuščiosios eigos P-būsenos (P8) lengva periodine apkrova.
// Kodėl: kai nešiojamo kompiuterio GPU (WSL2 / WDDM) tuščiąja eiga nuslenka į P8, bet kurio proceso platus GPU darbo
// pliūpsnis sukelia P8 → P3 perėjimą, kurio metu VISŲ procesų GPU darbas sustoja 50–350 ms (įrodyta:
// docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md §4–§6; su nuolatine lengva apkrova 0 / 105 pliūpsnių, be jos 29 / 55).
// Ką daro: kas --interval-ms (absoliutus periodas, be dreifo) užpildo --size-mb VRAM bloką (cudaMemsetAsync — device
// užpildymo kernel'is, ta pati darbo klasė kaip tyrimo gpu_ctx_probe busy) ir palaukia jo pabaigos.
// Kol veikia — nespausdina nieko (išskyrus --event-ms: ilgesni užpildymai — „GPU įvykis“ su TAI pradžia); pabaigoje (trukmė arba
// SIGINT / SIGTERM) — viena santraukos eilutė į stderr.
// Numatytosios reikšmės — iš tyrimo (64 MB kas 5 ms). --duration 0 — iki sustabdymo. --parent-pid P — baigti ir tada, kai
// proceso P nebėra (harness'as, nužudytas SIGKILL, neturi kaip sustabdyti „amžino“ keepwarm): PR_SET_PDEATHSIG (SIGTERM, kai
// miršta tėvas) ir patikra kiekviename cikle (tiesioginiam tėvui — getppid pasikeitimas: nužudytas tėvas gali likti zombiu,
// kurį kill(P, 0) dar „mato“).
#include <cuda_runtime_api.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

#include "func/func_common.hpp"

namespace mxlv::func {

namespace {
bool parent_alive(int pid, pid_t ppid0) {
    if (pid <= 0) return true;
    if (pid == ppid0) return getppid() == ppid0;  // tiesioginis tėvas: mirus — perimamas kito (init / subreaper)
    return kill(pid, 0) == 0 || errno == EPERM;
}
}  // namespace

int run_keepwarm(const KeepWarmOpts& k, const std::string& device_uuid) {
    auto& stop = stop_flag();
    const pid_t ppid0 = getppid();
    if (k.parent_pid > 0) {
        if (k.parent_pid == ppid0) prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (!parent_alive(k.parent_pid, ppid0)) {
            std::fprintf(stderr, "[keepwarm] --parent-pid %d: proceso nėra\n", k.parent_pid);
            return 0;
        }
    }
    const int dev = cuda_device(device_uuid);
    if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) {
        std::fprintf(stderr, "[keepwarm] CUDA įrenginys nerastas\n");
        return 2;
    }
    const std::size_t bytes = static_cast<std::size_t>(k.size_mb) << 20;
    void* p = nullptr;
    cudaStream_t s{};
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess || cudaMalloc(&p, bytes) != cudaSuccess) {
        std::fprintf(stderr, "[keepwarm] CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (s) cudaStreamDestroy(s);
        return 2;
    }
    using clk = std::chrono::steady_clock;
    const auto period = std::chrono::duration_cast<clk::duration>(std::chrono::duration<double, std::milli>(k.interval_ms));
    const auto t0 = clk::now();
    const auto end = k.duration_s > 0
                         ? t0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(k.duration_s))
                         : clk::time_point::max();
    std::uint64_t fills = 0, late = 0;
    double fill_max_ms = 0, fill_sum_ms = 0;
    cudaError_t ce = cudaSuccess;
    auto next = t0;
    bool orphan = false;
    while (!stop.load(std::memory_order_relaxed) && clk::now() < end) {
        if (!parent_alive(k.parent_pid, ppid0)) {  // harness'as dingo (SIGKILL) — nelikti veikti be galo
            orphan = true;
            break;
        }
        const auto a = clk::now();
        timespec tai{};
        if (k.event_ms >= 0) clock_gettime(CLOCK_TAI, &tai);  // įvykio pradžia — ta pati laiko skalė kaip mxlGetTime
        ce = cudaMemsetAsync(p, static_cast<int>(fills & 0xff), bytes, s);
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(s);
        if (ce != cudaSuccess) break;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - a).count();
        ++fills;
        fill_sum_ms += ms;
        fill_max_ms = std::max(fill_max_ms, ms);
        if (k.event_ms >= 0 && ms > k.event_ms)
            std::fprintf(stderr, "[keepwarm] GPU įvykis: TAI %lld%09ld trukmė %.3f ms\n", static_cast<long long>(tai.tv_sec), tai.tv_nsec, ms);
        next += period;
        const auto now = clk::now();
        if (next <= now) {  // užpildymas ilgesnis už periodą — nepasivejama (kitas — nuo dabar)
            ++late;
            next = now;
            continue;
        }
        std::this_thread::sleep_until(std::min(next, end));
    }
    const double secs = std::chrono::duration<double>(clk::now() - t0).count();
    cudaFree(p);
    cudaStreamDestroy(s);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "[keepwarm] CUDA: %s (po %llu užpildymų)\n", cudaGetErrorString(ce), static_cast<unsigned long long>(fills));
        return 2;
    }
    std::fprintf(stderr,
                 "[keepwarm] baigta: %llu užpildymų po %u MB kas %.1f ms, %.1f s; užpildymo trukmė vid. %.3f ms, maks. %.3f ms, "
                 "ilgesnių už periodą %llu%s\n",
                 static_cast<unsigned long long>(fills), k.size_mb, k.interval_ms, secs, fills ? fill_sum_ms / static_cast<double>(fills) : 0.0,
                 fill_max_ms, static_cast<unsigned long long>(late), orphan ? " (tėvinio proceso nebėra)" : "");
    return 0;
}

int run_hostwatch(const HostWatchOpts& h) {
    auto& stop = stop_flag();
    const pid_t ppid0 = getppid();
    if (h.parent_pid > 0) {
        if (h.parent_pid == ppid0) prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (!parent_alive(h.parent_pid, ppid0)) {
            std::fprintf(stderr, "[hostwatch] --parent-pid %d: proceso nėra\n", h.parent_pid);
            return 0;
        }
    }
    auto ns = [](clockid_t c) {
        timespec t{};
        clock_gettime(c, &t);
        return static_cast<std::int64_t>(t.tv_sec) * 1'000'000'000 + t.tv_nsec;
    };
    const auto period = static_cast<std::int64_t>(h.period_ms * 1e6);
    const std::int64_t limit = h.event_ms >= 0 ? period + static_cast<std::int64_t>(h.event_ms * 1e6) : INT64_MAX;
    const std::int64_t t0 = ns(CLOCK_MONOTONIC);
    const std::int64_t end = h.duration_s > 0 ? t0 + static_cast<std::int64_t>(h.duration_s * 1e9) : INT64_MAX;
    // TAI — ta pati laiko skalė kaip mxlGetTime ir keepwarm įvykiai
    std::int64_t prev = t0, prev_tai = ns(CLOCK_TAI), next = t0, gap_max = 0;
    std::uint64_t wakes = 0, events = 0;
    bool orphan = false;
    while (!stop.load(std::memory_order_relaxed)) {
        next += period;
        const timespec ts{static_cast<time_t>(next / 1'000'000'000), static_cast<long>(next % 1'000'000'000)};
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR && !stop.load(std::memory_order_relaxed)) {
        }
        const std::int64_t now = ns(CLOCK_MONOTONIC), now_tai = ns(CLOCK_TAI);
        ++wakes;
        const std::int64_t gap = now - prev;
        gap_max = std::max(gap_max, gap);
        if (gap > limit) {
            ++events;
            std::fprintf(stderr, "[hostwatch] host įvykis: TAI %lld trukmė %.3f ms\n", static_cast<long long>(prev_tai),
                         static_cast<double>(gap) / 1e6);
        }
        prev = now;
        prev_tai = now_tai;
        if (next <= now) next = now;  // pauzė ilgesnė už periodą — nepasivejama (kitas — nuo dabar)
        if (now >= end) break;
        if (!parent_alive(h.parent_pid, ppid0)) {  // harness'as dingo (SIGKILL) — nelikti veikti be galo
            orphan = true;
            break;
        }
    }
    std::fprintf(stderr, "[hostwatch] baigta: %llu pabudimų kas %.1f ms, %.1f s; įvykių %llu, maks. tarpas %.3f ms%s\n",
                 static_cast<unsigned long long>(wakes), h.period_ms, static_cast<double>(ns(CLOCK_MONOTONIC) - t0) / 1e9,
                 static_cast<unsigned long long>(events), static_cast<double>(gap_max) / 1e6, orphan ? " (tėvinio proceso nebėra)" : "");
    return 0;
}

}  // namespace mxlv::func
