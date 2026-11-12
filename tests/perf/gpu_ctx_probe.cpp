// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// gpu_ctx_probe — mažiausias CUDA „įvykių“ šaltinis tyrimui tests/perf/gpu_event_hiccup.py: ar KITO proceso CUDA konteksto
// sukūrimas / sunaikinimas, didelis cudaMalloc / cudaFree ar vien proceso startas / pabaiga vėlina toje pačioje GPU
// veikiantį realaus laiko gamintoją (WSL2 / WDDM)?
//
// Kiekviena fazė spausdinama į stdout viena eilute: `<fazė> <TAI ns pradžia> <TAI ns pabaiga> [rc]` (CLOCK_TAI, kaip
// vramflow-func `vėlus commit` eilutėse) — harness'as pagal jas tiksliai sulygina įvykio langą su aukos commit'ų laikais.
//
// Režimai:
//   ctx <hold_s>   cudaFree(0) (pirminis kontekstas) → laukti hold_s → cudaDeviceReset → išeiti;
//   serve [ctx]    skaityti komandas iš stdin: `ctx` (sukurti kontekstą), `malloc <MB>` (cudaMalloc + cudaMemset + sync),
//                  `mallocraw <MB>` (tik cudaMalloc), `memset <MB>` (paskutiniam blokui), `spin <ms>` (vienas kernel'is,
//                  besisukantis ms milisekundžių pagal globaltimer), `spinw <ms>` (tas pats per visą GPU: 4096 × 256),
//                  `memsetc <MB> <dalis MB>` (memset dalimis su sinchronizacija — mitigacijos bandymas),
//                  `fill <MB>` (užpildymas savo kernel'iu vietoj cudaMemset),
//                  `free` (paskutinis blokas), `reset` (cudaDeviceReset), `quit`; be argumento `ctx` kontekstas kuriamas
//                  tik pirmos komandos metu;
//   busy <MB> <ms> nuolatinė lengva apkrova (užpildymas MB kas ms), kad GPU neišeitų į P8 (mitigacijos bandymas);
//   host <MB>      tik CPU: malloc + memset MB, 1 s, išeiti (CUDA nekraunama — kontrolė „proceso startas / pabaiga“).
// Kompiliavimas (WSL2): /usr/local/cuda-13.2/bin/nvcc -O2 -x cu -o gpu_ctx_probe tests/perf/gpu_ctx_probe.cpp
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
// Vienas blokas sukasi `ns` nanosekundžių pagal GPU globaltimer (nepriklauso nuo takto): „svetimas ilgas GPU darbas“.
__global__ void spin_kernel(unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}

__global__ void fill_kernel(uint4* p, unsigned long long n) {
    const uint4 v = make_uint4(0x66666666u, 0x66666666u, 0x66666666u, 0x66666666u);
    for (unsigned long long k = blockIdx.x * 1ull * blockDim.x + threadIdx.x; k < n; k += 1ull * gridDim.x * blockDim.x) p[k] = v;
}

unsigned long long tai() {
    timespec ts{};
    clock_gettime(CLOCK_TAI, &ts);
    return static_cast<unsigned long long>(ts.tv_sec) * 1'000'000'000ull + static_cast<unsigned long long>(ts.tv_nsec);
}

void line(const char* phase, unsigned long long a, unsigned long long b, int rc = 0) {
    std::printf("%s %llu %llu %d\n", phase, a, b, rc);
    std::fflush(stdout);
}

int ctx_create() {
    const auto a = tai();
    const cudaError_t e = cudaFree(nullptr);
    line("ctx_create", a, tai(), static_cast<int>(e));
    return e == cudaSuccess ? 0 : 2;
}

int ctx_destroy() {
    const auto a = tai();
    const cudaError_t e = cudaDeviceReset();
    line("ctx_destroy", a, tai(), static_cast<int>(e));
    return e == cudaSuccess ? 0 : 2;
}

int mode_ctx(double hold_s) {
    line("exec", tai(), tai());
    if (ctx_create()) return 2;
    std::this_thread::sleep_for(std::chrono::duration<double>(hold_s));
    if (ctx_destroy()) return 2;
    line("exit", tai(), tai());
    return 0;
}

int mode_serve(bool ctx_now) {
    line("exec", tai(), tai());
    if (ctx_now && ctx_create()) return 2;
    std::vector<void*> blocks;
    std::string cmd;
    while (std::cin >> cmd) {
        if (cmd == "ctx") {
            ctx_create();
        } else if (cmd == "malloc") {
            unsigned long long mb = 0;
            std::cin >> mb;
            void* p = nullptr;
            const auto a = tai();
            cudaError_t e = cudaMalloc(&p, mb << 20);
            const auto b = tai();
            line("malloc", a, b, static_cast<int>(e));
            if (e == cudaSuccess) {
                const auto c = tai();
                e = cudaMemset(p, 0x5a, mb << 20);
                if (e == cudaSuccess) e = cudaDeviceSynchronize();
                line("memset", c, tai(), static_cast<int>(e));
                blocks.push_back(p);
            }
        } else if (cmd == "mallocraw") {  // tik cudaMalloc (be memset): ar vėlina pats rezervavimas?
            unsigned long long mb = 0;
            std::cin >> mb;
            void* p = nullptr;
            const auto a = tai();
            const cudaError_t e = cudaMalloc(&p, mb << 20);
            line("mallocraw", a, tai(), static_cast<int>(e));
            if (e == cudaSuccess) blocks.push_back(p);
        } else if (cmd == "memset") {  // memset jau rezervuotam paskutiniam blokui (tik GPU darbas, be rezervavimo)
            unsigned long long mb = 0;
            std::cin >> mb;
            cudaError_t e = cudaErrorInvalidValue;
            const auto a = tai();
            if (!blocks.empty()) {
                e = cudaMemset(blocks.back(), 0x33, mb << 20);
                if (e == cudaSuccess) e = cudaDeviceSynchronize();
            }
            line("memset", a, tai(), static_cast<int>(e));
        } else if (cmd == "memsetc") {  // memset dalimis: <MB> po <chunk MB>, sinchronizuojant po kiekvienos dalies
            unsigned long long mb = 0, chunk = 0;
            std::cin >> mb >> chunk;
            cudaError_t e = cudaErrorInvalidValue;
            const auto a = tai();
            if (!blocks.empty() && chunk > 0) {
                auto* base = static_cast<unsigned char*>(blocks.back());
                for (unsigned long long off = 0; off < mb; off += chunk) {
                    e = cudaMemset(base + (off << 20), 0x44, std::min(chunk, mb - off) << 20);
                    if (e == cudaSuccess) e = cudaDeviceSynchronize();
                    if (e != cudaSuccess) break;
                }
            }
            line("memsetc", a, tai(), static_cast<int>(e));
        } else if (cmd == "fill") {  // tas pats užpildymas savo kernel'iu (ne cudaMemset): ar svarbus būdas, ar pralaidumas?
            unsigned long long mb = 0;
            std::cin >> mb;
            cudaError_t e = cudaErrorInvalidValue;
            const auto a = tai();
            if (!blocks.empty()) {
                fill_kernel<<<4096, 256>>>(static_cast<uint4*>(blocks.back()), (mb << 20) / sizeof(uint4));
                e = cudaDeviceSynchronize();
            }
            line("fill", a, tai(), static_cast<int>(e));
        } else if (cmd == "spin" || cmd == "spinw") {  // spin — 1 gija; spinw — 4096 blokų × 256 gijų (visa GPU)
            double ms = 0;
            std::cin >> ms;
            const auto a = tai();
            if (cmd == "spin") spin_kernel<<<1, 1>>>(static_cast<unsigned long long>(ms * 1e6));
            else spin_kernel<<<4096, 256>>>(static_cast<unsigned long long>(ms * 1e6));
            const cudaError_t e = cudaDeviceSynchronize();
            line(cmd.c_str(), a, tai(), static_cast<int>(e));
        } else if (cmd == "free") {
            if (blocks.empty()) { line("free", tai(), tai(), -1); continue; }
            const auto a = tai();
            const cudaError_t e = cudaFree(blocks.back());
            line("free", a, tai(), static_cast<int>(e));
            blocks.pop_back();
        } else if (cmd == "reset") {
            blocks.clear();
            ctx_destroy();
        } else if (cmd == "quit") {
            break;
        } else {
            line("unknown", tai(), tai(), -1);
        }
    }
    line("exit", tai(), tai());
    return 0;
}

// Nuolatinė lengva GPU apkrova (mitigacijos bandymas): kas sleep_ms užpildoma MB — GPU neišleidžiama į P8, todėl
// svetimas darbo pliūpsnis nebesukelia P8 → P3 perėjimo. Sukasi, kol nužudomas (SIGTERM).
int mode_busy(unsigned long long mb, double sleep_ms) {
    line("exec", tai(), tai());
    if (ctx_create()) return 2;
    void* p = nullptr;
    if (cudaMalloc(&p, mb << 20) != cudaSuccess) return 2;
    line("busy", tai(), tai());
    for (;;) {
        fill_kernel<<<4096, 256>>>(static_cast<uint4*>(p), (mb << 20) / sizeof(uint4));
        if (cudaDeviceSynchronize() != cudaSuccess) return 2;
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sleep_ms));
    }
}

int mode_host(unsigned long long mb) {
    line("exec", tai(), tai());
    const auto a = tai();
    auto* p = static_cast<unsigned char*>(std::malloc(mb << 20));
    if (!p) return 2;
    std::memset(p, 0x5a, mb << 20);
    volatile unsigned char sink = 0;  // kad kompiliatorius neišmestų memset (blokas kitaip nenaudojamas)
    for (unsigned long long k = 0; k < (mb << 20); k += 4096) sink = sink + p[k];
    line("host_touch", a, tai(), sink == 1 ? 1 : 0);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::free(p);
    line("exit", tai(), tai());
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string m = argc > 1 ? argv[1] : "";
    if (m == "ctx") return mode_ctx(argc > 2 ? std::atof(argv[2]) : 5.0);
    if (m == "serve") return mode_serve(argc > 2 && std::string(argv[2]) == "ctx");
    if (m == "host") return mode_host(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 512);
    if (m == "busy") return mode_busy(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 64, argc > 3 ? std::atof(argv[3]) : 5.0);
    std::fprintf(stderr, "naudojimas: gpu_ctx_probe ctx <hold_s> | serve [ctx] | host <MB> | busy <MB> <sleep_ms>\n");
    return 1;
}
