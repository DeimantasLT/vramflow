// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// U6 minimal repro: mxlCreateFlowWriter with the cuda-linear payload backend changes the calling thread's current CUDA
// context from a context the application created (cuCtxCreate) to the device's primary context, and does not restore it.
#include <cuda.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>

#include <cstdio>
#include <filesystem>
#include <string>

static const char* kDef =
    R"({"id":"5fbec3b1-1b0f-417d-9059-8b94a47197ff","description":"u6","label":"u6",)"
    R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["u6:Video"]},"format":"urn:x-nmos:format:video","parents":[],)"
    R"("media_type":"video/v210","grain_rate":{"numerator":50,"denominator":1},"frame_width":1920,"frame_height":1080,)"
    R"("interlace_mode":"progressive","colorspace":"BT709","components":[{"name":"Y","width":1920,"height":1080,"bit_depth":10},)"
    R"({"name":"Cb","width":960,"height":1080,"bit_depth":10},{"name":"Cr","width":960,"height":1080,"bit_depth":10}]})";

int main() {
    if (cuInit(0) != CUDA_SUCCESS) return 77;
    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);
    CUuuid u{};
    cuDeviceGetUuid(&u, dev);
    char uuid[64];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
    std::snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2], b[3],
                  b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    CUcontext mine = nullptr, primary = nullptr, cur = nullptr;
    cuCtxCreate(&mine, nullptr, 0, dev);  // application's own context, current on this thread
    cuDevicePrimaryCtxRetain(&primary, dev);
    cuCtxGetCurrent(&cur);
    std::printf("before mxlCreateFlowWriter: current == own context: %s\n", cur == mine ? "yes" : "no");

    const std::string dir = "/dev/shm/u6-repro";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    mxlInstance inst = mxlCreateInstance(dir.c_str(), "");
    const std::string opts = std::string(R"({"payload":{"backend":"cuda-linear","deviceUuid":")") + uuid + R"("}})";
    mxlFlowWriter w = nullptr;
    mxlFlowConfigInfo cfg{};
    bool created = false;
    const mxlStatus st = mxlCreateFlowWriter(inst, kDef, opts.c_str(), &w, &cfg, &created);
    cuCtxGetCurrent(&cur);
    std::printf("mxlCreateFlowWriter status %d; after: current == own context: %s, == primary: %s\n", static_cast<int>(st),
                cur == mine ? "yes" : "no", cur == primary ? "yes" : "no");
    if (w) mxlReleaseFlowWriter(inst, w);
    mxlDestroyInstance(inst);
    std::filesystem::remove_all(dir);
    cuDevicePrimaryCtxRelease(dev);
    cuCtxDestroy(mine);
    return cur == mine ? 0 : 1;
}
