// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// R20 regresija PRIEŠ PRISTATOMĄ libvramflow-gfx.so (auditas 9): programa linkuojama su .so ir su ATSKIRA (dinamine)
// libcudart.so — kaip tikras naudotojas; biblioteka turi savo privačią statinę cudart (eksportuoja tik vfgfx_*). Tikrina,
// kad kviečiančiojo stream'e dar neįrašytas CUDA kadras (host funkcija miega 200 ms, tik tada kopijuoja kadrą):
//   numatytas kelias (cudaDeviceSynchronize bibliotekos cudart'e) ir cuda_stream kelias (įvykis kviečiančiojo stream'e,
//   rankena iš KITOS cudart) -> grain'as == kadras; VFGFX_FRAME_CALLER_SYNCED -> grain'as PRIVALO skirtis (testas
//   atskiria atvejus). Abi saugyklos (host, cuda-linear). Etalonas — to paties kadro grain'as, įrašytas jau paruoštam
//   kadrui (bitų tikslumą prieš gfx::ref tikrina test_gfx_writer.cpp).
// --bench: vfgfx_write kaina (ms/kvietimą) 1080i50 cuda-linear, CUDA kadras, trims keliams, laisvoje GPU ir kai kitame
//   stream'e veikia ~100 ms kernel'is. Grąžinimo kodas 77 — nėra GPU (ctest SKIP).
#include <cuda_runtime.h>
#include <link.h>
#include <mxl/mxl.h>
#include <mxl/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "func/flows.hpp"
#include "vramflow/gfx.h"

namespace {

int g_fail = 0;
#define EXPECT(c)                                                            \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

std::string g_cudart;  // įkeltos dinaminės libcudart.so kelias

int find_cudart(dl_phdr_info* info, std::size_t, void*) {
    if (info->dlpi_name && std::strstr(info->dlpi_name, "libcudart.so")) g_cudart = info->dlpi_name;
    return 0;
}

std::string device_uuid() {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return {};
    static const char* hex = "0123456789abcdef";
    std::string r = "GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) r.push_back('-');
        const auto v = static_cast<std::uint8_t>(p.uuid.bytes[i]);
        r.push_back(hex[v >> 4]);
        r.push_back(hex[v & 15]);
    }
    return r;
}

std::string flow_def(int n, int w, int h, bool interlaced) {
    char id[64];
    std::snprintf(id, sizeof id, "7a210a00-0000-4000-8000-%012x", 0x700000 + (static_cast<int>(::getpid()) & 0xFFFF) * 16 + n);
    vfgfx_flow_params p = VFGFX_FLOW_PARAMS_INIT;
    p.flow_id = id;
    p.label = "gfx so cuda check";
    p.width = static_cast<std::uint32_t>(w);
    p.height = static_cast<std::uint32_t>(h);
    p.scan = interlaced ? VFGFX_SCAN_INTERLACED_TFF : VFGFX_SCAN_PROGRESSIVE;
    p.frame_rate_num = interlaced ? 25 : 50;
    p.frame_rate_den = 1;
    char b[2048];
    if (vfgfx_flow_def(&p, b, sizeof b, nullptr) != VFGFX_OK) {
        std::fprintf(stderr, "vfgfx_flow_def: %s\n", vfgfx_last_error(nullptr));
        return {};
    }
    return b;
}

std::string id_of(const std::string& def) {
    const std::size_t k = def.find("\"id\":\"");
    return def.substr(k + 6, 36);
}

// Grain'o baitai (host arba device saugykla) į host buferį.
bool read_grain(mxlv::func::VideoIn& in, std::uint64_t idx, std::vector<std::uint8_t>& out) {
    mxlGrainInfo info{};
    const std::uint32_t* ptr = nullptr;
    if (in.get(idx, 0, info, ptr) != MXL_STATUS_OK) return false;
    out.resize(info.grainSize);
    if (in.cuda) return cudaMemcpy(out.data(), ptr, out.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
    std::memcpy(out.data(), ptr, out.size());
    return true;
}

// Užimtumo kernel'is: sukasi ns laiko pagal %globaltimer (sieninis laikas, nepriklauso nuo SM dažnio kitimo).
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__global__ void spin(unsigned long long ns) {
    const unsigned long long t0 = gtime();
    while (gtime() - t0 < ns) {
    }
}

struct Pattern {
    int w, h;
    std::size_t stride;
    std::vector<std::uint8_t> host;
    Pattern(int w_, int h_) : w(w_), h(h_), stride(static_cast<std::size_t>(w_) * 4), host(stride * static_cast<std::size_t>(h_)) {
        std::uint32_t s = 12345;
        for (auto& b : host) {
            s = s * 1664525u + 1013904223u;
            b = static_cast<std::uint8_t>(s >> 24);
        }
    }
};

// Viena R20 patikra: saugykla (cuda_store), kelias (0 numatytas, 1 CALLER_SYNCED, 2 cuda_stream).
void r20_case(mxlInstance inst, int n, bool cuda_store, int mode, const std::string& opts) {
    const int W = 382, H = 40;
    const std::string def = flow_def(n, W, H, false);
    vfgfx_writer w = nullptr;
    if (vfgfx_writer_create(inst, def.c_str(), cuda_store ? opts.c_str() : nullptr, &w) != 0) {
        std::fprintf(stderr, "create: %s\n", vfgfx_last_error(nullptr));
        ++g_fail;
        return;
    }
    mxlv::func::VideoIn in;
    std::atomic<bool> stop{false};
    std::string err;
    const vramflow::VideoFormat fmt{{W, H, false, true}, {50, 1}};
    EXPECT(in.open(inst, id_of(def), 2000, stop, err, &fmt));
    Pattern pat(W, H);
    void *good = nullptr, *late = nullptr;
    EXPECT(cudaMalloc(&good, pat.host.size()) == cudaSuccess);
    EXPECT(cudaMalloc(&late, pat.host.size()) == cudaSuccess);
    EXPECT(cudaMemcpy(good, pat.host.data(), pat.host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    EXPECT(cudaMemset(late, 0, pat.host.size()) == cudaSuccess);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
    vfgfx_frame f = VFGFX_FRAME_INIT;
    f.width = W;
    f.height = H;
    f.stride = static_cast<std::uint32_t>(pat.stride);
    f.mem = VFGFX_MEM_CUDA;
    f.pixels = good;
    f.flags = VFGFX_FRAME_CALLER_SYNCED;  // etalonas: kadras tikrai paruoštas
    const std::uint64_t base = vfgfx_current_index(w);
    EXPECT(vfgfx_write(w, base, &f) == 0);
    std::vector<std::uint8_t> want, got;
    EXPECT(read_grain(in, base, want));

    cudaStream_t s = nullptr;
    EXPECT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    EXPECT(cudaLaunchHostFunc(s, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }, nullptr) == cudaSuccess);
    EXPECT(cudaMemcpyAsync(late, good, pat.host.size(), cudaMemcpyDeviceToDevice, s) == cudaSuccess);
    f.pixels = late;
    f.flags = mode == 1 ? VFGFX_FRAME_CALLER_SYNCED : 0u;
    f.cuda_stream = mode == 2 ? s : nullptr;
    const int rc = vfgfx_write(w, base + 1, &f);  // be cudaStreamSynchronize(s) — tyčia
    EXPECT(rc == 0);
    EXPECT(cudaStreamSynchronize(s) == cudaSuccess);
    EXPECT(read_grain(in, base + 1, got));
    const bool same = got == want && !want.empty();
    std::printf("  R20 %-11s %-14s grain == kadras: %s\n", cuda_store ? "cuda-linear" : "host",
                mode == 0 ? "numatytas" : mode == 1 ? "CALLER_SYNCED" : "cuda_stream", same ? "taip" : "ne");
    if (mode == 1) EXPECT(!same);  // biblioteka nelaukė -> nuliai
    else EXPECT(same);
    if (rc) std::fprintf(stderr, "  vfgfx_write: %s\n", vfgfx_last_error(w));
    in.release(inst);
    vfgfx_writer_destroy(w);
    cudaStreamDestroy(s);
    cudaFree(good);
    cudaFree(late);
}

struct Times {
    std::vector<double> ms;
    void print(const char* name) {
        std::sort(ms.begin(), ms.end());
        double sum = 0;
        for (double v : ms) sum += v;
        std::printf("  %-34s n=%2zu  vid. %8.3f  p50 %8.3f  max %8.3f ms\n", name, ms.size(), sum / ms.size(), ms[ms.size() / 2],
                    ms.back());
    }
};

void bench(mxlInstance inst, const std::string& opts) {
    const int W = 1920, H = 1080;
    const std::string def = flow_def(15, W, H, true);
    vfgfx_writer w = nullptr;
    if (vfgfx_writer_create(inst, def.c_str(), opts.c_str(), &w) != 0) {
        std::fprintf(stderr, "create: %s\n", vfgfx_last_error(nullptr));
        ++g_fail;
        return;
    }
    Pattern pat(W, H);
    void* d = nullptr;
    EXPECT(cudaMalloc(&d, pat.host.size()) == cudaSuccess);
    EXPECT(cudaMemcpy(d, pat.host.data(), pat.host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    cudaStream_t mine = nullptr, other = nullptr;
    EXPECT(cudaStreamCreateWithFlags(&mine, cudaStreamNonBlocking) == cudaSuccess);
    EXPECT(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
    const unsigned long long cycles100 = 100'000'000ull;  // 100 ms
    // Kalibravimas: kiek iš tikrųjų trunka spin kernel'is
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, other);
    spin<<<1, 1, 0, other>>>(cycles100);
    cudaEventRecord(e1, other);
    cudaEventSynchronize(e1);
    float kms = 0;
    cudaEventElapsedTime(&kms, e0, e1);
    std::printf("bench: 1920x1080i50 cuda-linear, CUDA kadras, vfgfx_write (1 laukas); spin kernel'is %.1f ms\n", kms);
    std::uint64_t idx = vfgfx_current_index(w);
    vfgfx_frame f = VFGFX_FRAME_INIT;
    f.width = W;
    f.height = H;
    f.stride = static_cast<std::uint32_t>(pat.stride);
    f.mem = VFGFX_MEM_CUDA;
    f.pixels = d;
    for (int k = 0; k < 5; ++k) EXPECT(vfgfx_write(w, idx++, &f) == 0);  // apšilimas
    const char* names[3] = {"numatytas (cudaDeviceSynchronize)", "cuda_stream (įvykis)", "CALLER_SYNCED (nelaukia)"};
    for (int busy = 0; busy < 2; ++busy)
        for (int mode = 0; mode < 3; ++mode) {
            Times t;
            const int iters = busy ? 10 : 50;
            for (int k = 0; k < iters; ++k) {
                f.flags = mode == 2 ? VFGFX_FRAME_CALLER_SYNCED : 0u;
                f.cuda_stream = mode == 1 ? mine : nullptr;
                if (busy) {
                    spin<<<1, 1, 0, other>>>(cycles100);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // kernel'is tikrai prasidėjo
                }
                const auto t0 = std::chrono::steady_clock::now();
                EXPECT(vfgfx_write(w, idx++, &f) == 0);
                t.ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                if (busy) EXPECT(cudaStreamSynchronize(other) == cudaSuccess);
            }
            char label[96];
            std::snprintf(label, sizeof label, "%s, %s", busy ? "100 ms kernel" : "laisva GPU", names[mode]);
            t.print(label);
        }
    vfgfx_writer_destroy(w);
    cudaFree(d);
    cudaStreamDestroy(mine);
    cudaStreamDestroy(other);
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::puts("SKIP: nėra CUDA įrenginio");
        return 77;
    }
    dl_iterate_phdr(find_cudart, nullptr);
    std::printf("libvramflow-gfx %s, kviečiančiojo cudart: %s\n", vfgfx_version_string(), g_cudart.empty() ? "(statinė?)" : g_cudart.c_str());
    EXPECT(!g_cudart.empty());  // testas prasmingas tik su atskira dinamine cudart
    const std::filesystem::path dir = std::filesystem::path("/dev/shm") / ("vf-gfx-so-" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    mxlInstance inst = mxlCreateInstance(dir.c_str(), "");
    if (!inst) {
        std::fputs("mxlCreateInstance\n", stderr);
        return 1;
    }
    const std::string opts = R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + device_uuid() + R"("}})";
    if (do_bench) {
        bench(inst, opts);
    } else {
        int k = 0;
        for (const bool cuda_store : {false, true})
            for (int mode = 0; mode < 3; ++mode) r20_case(inst, ++k, cuda_store, mode, opts);
    }
    mxlDestroyInstance(inst);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (g_fail) {
        std::printf("FAILED: %d\n", g_fail);
        return 1;
    }
    std::puts("OK");
    return 0;
}
