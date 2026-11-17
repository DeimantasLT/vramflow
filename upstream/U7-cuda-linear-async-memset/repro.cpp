// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// U7 minimal repro: mxlCreateFlowWriter with the cuda-linear payload backend returns while the backend's cudaMemset of the
// new ring is still executing. cudaMemset is asynchronous with respect to the host for device memory, and the backend does
// not wait for it. Usage: u7-repro [history_ms]   (default 200, the MXL default ring history)
#include <cuda_runtime_api.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

static const char* kDef =
    R"({"id":"5fbec3b1-1b0f-417d-9059-8b94a47197f7","description":"u7","label":"u7",)"
    R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["u7:Video"]},"format":"urn:x-nmos:format:video","parents":[],)"
    R"("media_type":"video/v210","grain_rate":{"numerator":50,"denominator":1},"frame_width":1920,"frame_height":1080,)"
    R"("interlace_mode":"progressive","colorspace":"BT709","components":[{"name":"Y","width":1920,"height":1080,"bit_depth":10},)"
    R"({"name":"Cb","width":960,"height":1080,"bit_depth":10},{"name":"Cr","width":960,"height":1080,"bit_depth":10}]})";

int main(int argc, char** argv) {
    const int history_ms = argc > 1 ? std::atoi(argv[1]) : 200;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    char uuid[64];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(p.uuid.bytes);
    std::snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2], b[3],
                  b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);

    const std::string dir = "/dev/shm/u7-repro";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/options.json") << "{\"urn:x-mxl:option:history_duration/v1.0\": " << history_ms << "000000}";
    mxlInstance inst = mxlCreateInstance(dir.c_str(), "");
    const std::string opts = std::string(R"({"payload":{"backend":"cuda-linear","deviceUuid":")") + uuid + R"("}})";
    mxlFlowWriter w = nullptr;
    mxlFlowConfigInfo cfg{};
    bool created = false;
    const mxlStatus st = mxlCreateFlowWriter(inst, kDef, opts.c_str(), &w, &cfg, &created);
    const cudaError_t q0 = cudaStreamQuery(nullptr);  // cudaSuccess: nothing pending; cudaErrorNotReady: work still running
    std::printf("history %d ms (%u grains, %.0f MiB); mxlCreateFlowWriter status %d; default stream right after: %d (%s)\n", history_ms,
                cfg.discrete.grainCount, cfg.discrete.grainCount * 5529600.0 / 1048576.0, static_cast<int>(st), static_cast<int>(q0),
                cudaGetErrorName(q0));
    const cudaError_t sy = cudaStreamSynchronize(nullptr);
    const cudaError_t q1 = cudaStreamQuery(nullptr);
    std::printf("after cudaStreamSynchronize(default stream) = %d: %d (%s)\n", static_cast<int>(sy), static_cast<int>(q1), cudaGetErrorName(q1));
    if (w) mxlReleaseFlowWriter(inst, w);
    mxlDestroyInstance(inst);
    std::filesystem::remove_all(dir);
    return q0 == cudaSuccess ? 0 : 1;
}
