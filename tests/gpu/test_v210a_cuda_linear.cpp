// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Atitikties testas (alfa kompozitoriaus vartai): MXL `video/v210a` (fill + key) ant PR #653 `cuda-linear` backend'o.
// Tikrina: planeCount == 2 ir dydžiai, abiejų plokštumų perdavimą per device atmintį, slice'us / minValidSlices.
// Reikia GPU ir MXL instaliacijos su cuda-linear įskiepiu; be GPU testai praleidžiami (SKIP), ne PASS.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mxlio/slots.hpp"

using namespace mxlv;

namespace {

constexpr std::size_t kWidth = 1920, kFrameHeight = 1080, kFieldHeight = kFrameHeight / 2;
constexpr std::size_t kFillPitch = (kWidth + 47) / 48 * 128;  // 5120 B
constexpr std::size_t kKeyPitch = (kWidth + 2) / 3 * 4;       // 2560 B
constexpr std::size_t kFillSize = kFillPitch * kFieldHeight;   // 2 764 800 B
constexpr std::size_t kKeySize = kKeyPitch * kFieldHeight;     // 1 382 400 B

#define CUDA_OK(x)                                                                                         \
    do {                                                                                                   \
        const cudaError_t _e = (x);                                                                        \
        if (_e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(_e)); \
    } while (0)

bool have_gpu() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string device_uuid() {
    cudaDeviceProp p{};
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
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

// 1080i50: 25 kadrai/s interlaced -> grain = laukas, grain_rate gaunamas 50/1.
std::string v210a_def(const char* id) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow v210a cuda-linear test","label":"vramflow v210a test",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-v210a-test:Video"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210a",)"
                  R"("grain_rate":{"numerator":25,"denominator":1},"frame_width":1920,"frame_height":1080,)"
                  R"("interlace_mode":"interlaced_tff","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":1920,"height":1080,"bit_depth":10},)"
                  R"({"name":"Cb","width":960,"height":1080,"bit_depth":10},)"
                  R"({"name":"Cr","width":960,"height":1080,"bit_depth":10}]})",
                  id);
    return b;
}

// Laikinas domenas /dev/shm/vf-v210a-<pid>-<n> + instancija; tvarkosi pats.
struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n) {
        dir = std::filesystem::path("/dev/shm") / ("vf-v210a-" + std::to_string(::getpid()) + "-" + std::to_string(n));
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

// Writer + reader su savo slot'ų aprašais (kaip VideoOut/VideoIn).
struct Pair {
    Domain& d;
    mxlFlowWriter w = nullptr;
    mxlFlowReader r = nullptr;
    mxlFlowConfigInfo cfg{};
    mxlGrainStorageLayout wl{}, rl{};
    std::vector<mxlGrainStorage> wslots, rslots;

    Pair(Domain& dom, const char* id) : d(dom) {
        // REQUIRE meta išimtį — destruktorius nebus iškviestas, todėl jau sukurtus handle'us atleidžiam patys.
        try {
            const std::string opts = R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + device_uuid() + R"("}})";
            bool created = false;
            REQUIRE(mxlCreateFlowWriter(d.inst, v210a_def(id).c_str(), opts.c_str(), &w, &cfg, &created) == MXL_STATUS_OK);
            REQUIRE(created);
            REQUIRE(mxlFlowWriterGetStorageLayout(w, &wl) == MXL_STATUS_OK);
            wslots.assign(wl.slotCount, mxlGrainStorage{});
            REQUIRE(mxlFlowWriterMapSlots(w, wl.slotCount, wslots.data()) == MXL_STATUS_OK);
            REQUIRE(mxlFlowWriterGetStorageLayout(w, &wl) == MXL_STATUS_OK);
            REQUIRE(mxlCreateFlowReader(d.inst, id, "", &r) == MXL_STATUS_OK);
            REQUIRE(mxlFlowReaderGetStorageLayout(r, &rl) == MXL_STATUS_OK);
            rslots.assign(rl.slotCount, mxlGrainStorage{});
            REQUIRE(mxlFlowReaderMapSlots(r, rl.slotCount, rslots.data()) == MXL_STATUS_OK);
            REQUIRE(mxlFlowReaderGetStorageLayout(r, &rl) == MXL_STATUS_OK);
        } catch (...) {
            release();
            throw;
        }
    }
    ~Pair() { release(); }
    void release() {
        if (r) mxlReleaseFlowReader(d.inst, r);
        if (w) mxlReleaseFlowWriter(d.inst, w);
        r = nullptr;
        w = nullptr;
    }
    // Plokštumos p adresas slot'e (device).
    std::uint8_t* wplane(std::uint32_t slot, std::uint32_t p) const {
        return static_cast<std::uint8_t*>(slot_ptr(wl, wslots[slot])) + wl.planes[p].offset;
    }
    const std::uint8_t* rplane(std::uint32_t slot, std::uint32_t p) const {
        return static_cast<const std::uint8_t*>(slot_ptr(rl, rslots[slot])) + rl.planes[p].offset;
    }
};

std::vector<std::uint8_t> random_bytes(std::size_t n, std::uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<std::uint8_t> v(n);
    for (auto& b : v) b = static_cast<std::uint8_t>(g());
    return v;
}

std::size_t count_diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    REQUIRE(a.size() == b.size());
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}

}  // namespace

TEST_CASE("v210a cuda-linear: dvi plokštumos, dydžiai", "[v210a][gpu][mxl]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(1);
    Pair p(d, "7a210a00-0000-4000-8000-000000000001");

    CHECK(std::string(p.wl.storageType) == MXL_CUDA_LINEAR_STORAGE_TYPE);
    REQUIRE(p.wl.planeCount == 2);
    CHECK(p.wl.planes[0].size == kFillSize);
    CHECK(p.wl.planes[1].size == kKeySize);
    CHECK(p.wl.planes[0].size == 2764800);
    CHECK(p.wl.planes[1].size == 1382400);
    CHECK(p.wl.planes[0].pitch == kFillPitch);
    CHECK(p.wl.planes[1].pitch == kKeyPitch);
    CHECK(p.wl.planes[0].offset == 0);
    CHECK(p.wl.planes[1].offset == kFillSize);  // key iš karto po fill
    // reader mato tą patį išdėstymą
    CHECK(p.rl.planeCount == 2);
    CHECK(p.rl.planes[0].size == kFillSize);
    CHECK(p.rl.planes[1].size == kKeySize);
    CHECK(p.rl.planes[1].offset == kFillSize);
    // grain'as = laukas (50/1); slice dydžiai = eilučių ilgiai
    CHECK(p.cfg.common.grainRate.numerator == 50);
    CHECK(p.cfg.common.grainRate.denominator == 1);
    CHECK(p.cfg.discrete.sliceSizes[0] == kFillPitch);
    CHECK(p.cfg.discrete.sliceSizes[1] == kKeyPitch);

    const mxlRational rate = p.cfg.common.grainRate;
    mxlGrainInfo info{};
    std::uint32_t slot = 0;
    REQUIRE(mxlFlowWriterOpenGrainSlot(p.w, mxlGetCurrentIndex(&rate), &info, &slot) == MXL_STATUS_OK);
    CHECK(info.grainSize == kFillSize + kKeySize);
    CHECK(info.totalSlices == kFieldHeight);
    REQUIRE(mxlFlowWriterCancelGrain(p.w) == MXL_STATUS_OK);
}

TEST_CASE("v210a cuda-linear: abi plokštumos perduodamos", "[v210a][gpu][mxl]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(2);
    Pair p(d, "7a210a00-0000-4000-8000-000000000002");
    REQUIRE(p.wl.planeCount == 2);

    const auto fill = random_bytes(kFillSize, 1);
    const auto key = random_bytes(kKeySize, 2);

    const mxlRational rate = p.cfg.common.grainRate;
    const std::uint64_t index = mxlGetCurrentIndex(&rate);
    mxlGrainInfo info{};
    std::uint32_t wslot = 0;
    REQUIRE(mxlFlowWriterOpenGrainSlot(p.w, index, &info, &wslot) == MXL_STATUS_OK);
    CUDA_OK(cudaMemcpy(p.wplane(wslot, 0), fill.data(), fill.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(p.wplane(wslot, 1), key.data(), key.size(), cudaMemcpyHostToDevice));
    info.validSlices = info.totalSlices;
    REQUIRE(mxlFlowWriterCommitGrain(p.w, &info) == MXL_STATUS_OK);

    mxlGrainInfo rinfo{};
    std::uint32_t rslot = 0;
    REQUIRE(mxlFlowReaderGetGrainSlot(p.r, index, MXL_GRAIN_VALID_SLICES_ALL, 0, &rinfo, &rslot) == MXL_STATUS_OK);
    REQUIRE(rslot == wslot);
    CHECK(rinfo.index == index);
    CHECK(rinfo.validSlices == rinfo.totalSlices);

    std::vector<std::uint8_t> got_fill(kFillSize), got_key(kKeySize);
    CUDA_OK(cudaMemcpy(got_fill.data(), p.rplane(rslot, 0), got_fill.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(got_key.data(), p.rplane(rslot, 1), got_key.size(), cudaMemcpyDeviceToHost));
    CHECK(count_diff(fill, got_fill) == 0);
    CHECK(count_diff(key, got_key) == 0);
}

TEST_CASE("v210a cuda-linear: slice'ai", "[v210a][gpu][mxl]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(3);
    Pair p(d, "7a210a00-0000-4000-8000-000000000003");
    REQUIRE(p.wl.planeCount == 2);

    const auto fill = random_bytes(kFillSize, 3);
    const auto key = random_bytes(kKeySize, 4);

    const mxlRational rate = p.cfg.common.grainRate;
    const std::uint64_t index = mxlGetCurrentIndex(&rate);
    mxlGrainInfo info{};
    std::uint32_t wslot = 0;
    REQUIRE(mxlFlowWriterOpenGrainSlot(p.w, index, &info, &wslot) == MXL_STATUS_OK);
    REQUIRE(info.totalSlices == kFieldHeight);

    constexpr std::uint16_t kBatch = 135;
    for (std::uint16_t valid = kBatch; valid <= info.totalSlices; valid = static_cast<std::uint16_t>(valid + kBatch)) {
        const std::size_t row0 = valid - kBatch;
        // eilutės [row0, valid) abiejose plokštumose (kiekviena su savo pitch)
        CUDA_OK(cudaMemcpy(p.wplane(wslot, 0) + row0 * kFillPitch, fill.data() + row0 * kFillPitch, kBatch * kFillPitch,
                           cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(p.wplane(wslot, 1) + row0 * kKeyPitch, key.data() + row0 * kKeyPitch, kBatch * kKeyPitch,
                           cudaMemcpyHostToDevice));
        info.validSlices = valid;
        REQUIRE(mxlFlowWriterCommitGrain(p.w, &info) == MXL_STATUS_OK);

        mxlGrainInfo rinfo{};
        std::uint32_t rslot = 0;
        if (valid < info.totalSlices) {
            // viena eilute daugiau nei commit'inta — dar nėra
            CHECK(mxlFlowReaderGetGrainSlot(p.r, index, static_cast<std::uint16_t>(valid + 1), 0, &rinfo, &rslot) ==
                  MXL_ERR_OUT_OF_RANGE_TOO_EARLY);
        }
        REQUIRE(mxlFlowReaderGetGrainSlot(p.r, index, valid, 0, &rinfo, &rslot) == MXL_STATUS_OK);
        REQUIRE(rslot == wslot);
        CHECK(rinfo.validSlices == valid);

        // Pirmos `valid` eilutės abiejose plokštumose sutampa.
        std::vector<std::uint8_t> gf(valid * kFillPitch), gk(valid * kKeyPitch);
        CUDA_OK(cudaMemcpy(gf.data(), p.rplane(rslot, 0), gf.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(gk.data(), p.rplane(rslot, 1), gk.size(), cudaMemcpyDeviceToHost));
        CHECK(std::memcmp(gf.data(), fill.data(), gf.size()) == 0);
        CHECK(std::memcmp(gk.data(), key.data(), gk.size()) == 0);
    }
    CHECK(info.validSlices == 540);
    mxlGrainInfo rinfo{};
    std::uint32_t rslot = 0;
    CHECK(mxlFlowReaderGetGrainSlot(p.r, index, MXL_GRAIN_VALID_SLICES_ALL, 0, &rinfo, &rslot) == MXL_STATUS_OK);
}
