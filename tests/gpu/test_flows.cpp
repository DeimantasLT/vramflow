// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// flows.cpp testai ant tikro MXL: v210a plokštumų išdėstymas (host ir cuda-linear), check_layout atmetimai, grainRate 50/1
// visada (R9) VideoOut ir VideoIn. Host dalis nereikalauja GPU; cuda-linear dalis be GPU praleidžiama (SKIP).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "func/flows.hpp"
#include "gfx/raster.hpp"
#include "mxlio/slots.hpp"

using namespace mxlv;
using namespace mxlv::func;
namespace gfx = vramflow::gfx;

namespace {

bool have_gpu() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string device_uuid() {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) throw std::runtime_error("cudaGetDeviceProperties");
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

// Laikinas domenas /dev/shm/vf-flows-<pid>-<n> + instancija.
struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n) {
        dir = std::filesystem::path("/dev/shm") / ("vf-flows-" + std::to_string(::getpid()) + "-" + std::to_string(n));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        inst = mxlCreateInstance(dir.c_str(), "");
        if (!inst) throw std::runtime_error("mxlCreateInstance");
    }
    ~Domain() {
        if (inst) mxlDestroyInstance(inst);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

// Flow aprašas 1920x1080; rate — grain_rate numeratorius (interlaced 25 -> grain 50/1; progresyvus: grain_rate = pats).
std::string def(const char* id, const char* media_type, bool interlaced, int rate) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow flows test","label":"vramflow flows test",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-flows-test:Video"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"%s",)"
                  R"("grain_rate":{"numerator":%d,"denominator":1},"frame_width":1920,"frame_height":1080,)"
                  R"("interlace_mode":"%s","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":1920,"height":1080,"bit_depth":10},)"
                  R"({"name":"Cb","width":960,"height":1080,"bit_depth":10},)"
                  R"({"name":"Cr","width":960,"height":1080,"bit_depth":10}]})",
                  id, media_type, rate, interlaced ? "interlaced_tff" : "progressive");
    return b;
}

gfx::Raster v210a_1080i() { return gfx::Raster{1920, 1080, true, true}; }
gfx::Raster v210a_1080p() { return gfx::Raster{1920, 1080, false, true}; }

mxlGrainStorageLayout synth_layout(const gfx::Raster& r) {
    mxlGrainStorageLayout l{};
    l.planeCount = r.alpha ? 2 : 1;
    l.planes[0] = {0, gfx::fill_bytes(r), gfx::fill_line_bytes(r.w), 0};
    if (r.alpha) l.planes[1] = {gfx::fill_bytes(r), gfx::key_bytes(r), gfx::key_line_bytes(r.w), 0};
    return l;
}

}  // namespace

TEST_CASE("flows: check_layout atmeta netinkamą plokštumų išdėstymą", "[flows]") {
    const gfx::Raster r = v210a_1080i();
    CHECK(check_layout(synth_layout(r), r).empty());
    const gfx::Raster v = gfx::kHd1080i;
    CHECK(check_layout(synth_layout(v), v).empty());

    mxlGrainStorageLayout l = synth_layout(r);
    l.planes[1].offset += 64;  // tarpas tarp fill ir key: ref/cpu/cuda adresuotų neteisingą vietą
    CHECK_FALSE(check_layout(l, r).empty());
    l = synth_layout(r);
    l.planes[1].offset = 0;
    CHECK_FALSE(check_layout(l, r).empty());
    l = synth_layout(r);
    l.planes[0].offset = 4096;
    CHECK_FALSE(check_layout(l, r).empty());
    l = synth_layout(r);
    l.planes[1].pitch += 4;
    CHECK_FALSE(check_layout(l, r).empty());
    l = synth_layout(r);
    l.planes[0].pitch += 128;
    CHECK_FALSE(check_layout(l, r).empty());
    l = synth_layout(r);
    l.planes[1].size -= 4;
    CHECK_FALSE(check_layout(l, r).empty());
    l = synth_layout(r);
    l.planeCount = 1;
    CHECK_FALSE(check_layout(l, r).empty());
}

TEST_CASE("flows: host v210a writer — key iškart po fill", "[flows][mxl]") {
    Domain d(1);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, def("7a210a00-0000-4000-8000-0000000000f1", "video/v210a", true, 25), "", err, v210a_1080i()));
    CHECK(is_host_layout(o.layout));
    REQUIRE(o.layout.planeCount == 2);
    CHECK(o.layout.planes[0].offset == 0);
    CHECK(o.layout.planes[0].size == gfx::fill_bytes(v210a_1080i()));
    CHECK(o.layout.planes[1].offset == o.layout.planes[0].size);  // key iškart po fill
    CHECK(o.layout.planes[1].size == gfx::key_bytes(v210a_1080i()));
    mxlStatus st = MXL_STATUS_OK;
    const mxlRational rate = kFieldRate;
    REQUIRE(o.open(mxlGetCurrentIndex(&rate), st) != nullptr);
    CHECK(o.info.grainSize == o.layout.planes[0].size + o.layout.planes[1].size);
    CHECK(o.info.grainSize == gfx::grain_bytes(v210a_1080i()));
    o.cancel();
    o.release(d.inst);
}

TEST_CASE("flows: cuda-linear v210a writer — key iškart po fill", "[flows][mxl][gpu]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(2);
    VideoOut o;
    std::string err;
    const std::string opts = R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + device_uuid() + R"("}})";
    REQUIRE(o.create(d.inst, def("7a210a00-0000-4000-8000-0000000000f2", "video/v210a", true, 25), opts, err, v210a_1080i()));
    CHECK(is_cuda_layout(o.layout));
    REQUIRE(o.layout.planeCount == 2);
    CHECK(o.layout.planes[0].offset == 0);
    CHECK(o.layout.planes[1].offset == o.layout.planes[0].size);
    CHECK(o.layout.planes[1].size == gfx::key_bytes(v210a_1080i()));
    o.release(d.inst);
}

TEST_CASE("flows: VideoOut grainRate 50/1 visada (R9)", "[flows][mxl]") {
    std::string err;
    {  // progresyvinis 50p — tinka
        Domain d(3);
        VideoOut o;
        CHECK(o.create(d.inst, def("7a210a00-0000-4000-8000-0000000000f3", "video/v210a", false, 50), "", err, v210a_1080p()));
        o.release(d.inst);
    }
    {  // progresyvinis 25p — netinka (anksčiau tikrinta tik interlaced)
        Domain d(4);
        VideoOut o;
        CHECK_FALSE(o.create(d.inst, def("7a210a00-0000-4000-8000-0000000000f4", "video/v210a", false, 25), "", err, v210a_1080p()));
        CHECK(err.find("grainRate") != std::string::npos);
        o.release(d.inst);
    }
}

TEST_CASE("flows: VideoIn grainRate 50/1 visada, v210a rastras iš aprašo (R9)", "[flows][mxl]") {
    std::atomic<bool> stop{false};
    std::string err;
    {  // 50p v210a: atidaro; rastras iš aprašo (nullptr expect)
        Domain d(6);
        VideoOut o;
        REQUIRE(o.create(d.inst, def("7a210a00-0000-4000-8000-0000000000f6", "video/v210a", false, 50), "", err, v210a_1080p()));
        VideoIn in;
        REQUIRE(in.open(d.inst, "7a210a00-0000-4000-8000-0000000000f6", 1000, stop, err));
        CHECK(in.raster == v210a_1080p());
        in.release(d.inst);
        o.release(d.inst);
    }
    {  // 25p: writer'į kuriam tiesiogiai (VideoOut atmestų), VideoIn privalo atmesti
        Domain d(7);
        mxlFlowWriter w = nullptr;
        mxlFlowConfigInfo cfg{};
        bool created = false;
        REQUIRE(mxlCreateFlowWriter(d.inst, def("7a210a00-0000-4000-8000-0000000000f7", "video/v210a", false, 25).c_str(), nullptr, &w,
                                    &cfg, &created) == MXL_STATUS_OK);
        VideoIn in;
        CHECK_FALSE(in.open(d.inst, "7a210a00-0000-4000-8000-0000000000f7", 1000, stop, err));
        CHECK(err.find("grainRate") != std::string::npos);
        in.release(d.inst);
        mxlReleaseFlowWriter(d.inst, w);
    }
}
