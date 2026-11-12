// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// keepwarm — laiko GPU virš tuščiosios eigos P-būsenos (P8) lengva periodine apkrova.
// Kodėl: kai nešiojamo kompiuterio GPU (WSL2 / WDDM) tuščiąja eiga nuslenka į P8, bet kurio proceso platus GPU darbo
// pliūpsnis sukelia P8 → P3 perėjimą, kurio metu VISŲ procesų GPU darbas sustoja 50–350 ms (įrodyta:
// docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md §4–§6; su nuolatine lengva apkrova 0 / 105 pliūpsnių, be jos 29 / 55).
// Ką daro: kas --interval-ms (absoliutus periodas, be dreifo) užpildo --size-mb VRAM bloką (cudaMemsetAsync — device
// užpildymo kernel'is, ta pati darbo klasė kaip tyrimo gpu_ctx_probe busy) ir palaukia jo pabaigos.
// Kol veikia — nespausdina nieko; pabaigoje (trukmė arba SIGINT / SIGTERM) — viena santraukos eilutė į stderr.
// Numatytosios reikšmės — iš tyrimo (64 MB kas 5 ms). --duration 0 — iki sustabdymo.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

#include "func/func_common.hpp"

namespace mxlv::func {

int run_keepwarm(const KeepWarmOpts& k, const std::string& device_uuid) {
    auto& stop = stop_flag();
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
    while (!stop.load(std::memory_order_relaxed) && clk::now() < end) {
        const auto a = clk::now();
        ce = cudaMemsetAsync(p, static_cast<int>(fills & 0xff), bytes, s);
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(s);
        if (ce != cudaSuccess) break;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - a).count();
        ++fills;
        fill_sum_ms += ms;
        fill_max_ms = std::max(fill_max_ms, ms);
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
                 "ilgesnių už periodą %llu\n",
                 static_cast<unsigned long long>(fills), k.size_mb, k.interval_ms, secs, fills ? fill_sum_ms / static_cast<double>(fills) : 0.0,
                 fill_max_ms, static_cast<unsigned long long>(late));
    return 0;
}

}  // namespace mxlv::func
