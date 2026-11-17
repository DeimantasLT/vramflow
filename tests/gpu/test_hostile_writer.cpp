// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Piktybiškas arba sugedęs rašytojas tame pačiame domene (planas docs/planai/2026-11-17-production-ready.md §1.4,
// kriterijus S3). Skaitytojas niekada neturi tikėti svetimo rašytojo antraštės dydžiais: klaida, ne skaitymas už ribų.
// Rašytojas čia — tiesiai per MXL API (antraštę jis gali įrašyti bet kokią); skaitytojas — func::VideoIn.
// Host saugykla GPU nereikalauja; cuda-linear dalis be GPU praleidžiama (SKIP).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime_api.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "common/format.hpp"
#include "func/flows.hpp"
#include "gfx/raster.hpp"

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

// Laikinas domenas /dev/shm/vf-host-<pid>-<n> + instancija.
struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n) {
        dir = std::filesystem::path("/dev/shm") / ("vf-host-" + std::to_string(::getpid()) + "-" + std::to_string(n));
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

// v210 flow aprašas w × h; rate — grain_rate numeratorius (interlaced 25 → grain 50/1; progresyvus — pats dažnis).
std::string def(const char* id, int w, int h, bool interlaced, int rate) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow hostile test","label":"vramflow hostile test",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-hostile-test:Video"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210",)"
                  R"("grain_rate":{"numerator":%d,"denominator":1},"frame_width":%d,"frame_height":%d,)"
                  R"("interlace_mode":"%s","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cb","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cr","width":%d,"height":%d,"bit_depth":10}]})",
                  id, rate, w, h, interlaced ? "interlaced_tff" : "progressive", w, h, w / 2, h, w / 2, h);
    return b;
}

// Rašytojas atidaro indeksą, perrašo antraštę (tamper) ir commit'ina tiesiai per MXL — be VideoOut patikrų.
template <class Tamper>
mxlStatus hostile_commit(VideoOut& o, std::uint64_t index, Tamper&& tamper) {
    mxlStatus st = MXL_STATUS_OK;
    REQUIRE(o.open(index, st) != nullptr);
    o.info.validSlices = o.info.totalSlices;
    tamper(o.info);
    st = mxlFlowWriterCommitGrain(o.w, &o.info);
    o.open_grain = false;
    return st;
}

// Skaitytojas negali grąžinti rodyklės į grain'ą su netinkama antrašte: statusas ≠ OK ir ptr == nullptr.
void check_rejected(VideoIn& in, std::uint64_t index) {
    mxlGrainInfo gi{};
    const std::uint32_t* p = reinterpret_cast<const std::uint32_t*>(std::uintptr_t{1});
    const mxlStatus rst = in.get(index, 50'000'000, gi, p);
    CHECK(rst == MXL_ERR_INTERNAL);  // mūsų antraštės patikra (ne MXL laukimo riba ar kita klaida)
    CHECK(p == nullptr);
}

void hostile_headers(const std::string& options, int dom, const char* id) {
    Domain d(dom);
    std::atomic<bool> stop{false};
    std::string err;
    VideoOut o;
    REQUIRE(o.create(d.inst, def(id, 1920, 1080, true, 25), options, err, gfx::kHd1080i));
    VideoIn in;
    REQUIRE(in.open(d.inst, id, 1000, stop, err));
    const mxlRational rate = kFieldRate;
    std::uint64_t idx = mxlGetCurrentIndex(&rate);
    const std::uint32_t real = static_cast<std::uint32_t>(gfx::grain_bytes(gfx::kHd1080i));
    int committed = 0;  // kiek netikrų antraščių MXL iš tikrųjų priėmė

    SECTION("grainSize ne rastro dydžio") {
        for (const std::uint32_t bad : {std::uint32_t{0}, real + 1, real - 1, std::uint32_t{0xFFFFFFFF}}) {
            ++idx;
            const mxlStatus cst = hostile_commit(o, idx, [&](mxlGrainInfo& g) { g.grainSize = bad; });
            INFO("grainSize " << bad << " (tikras " << real << "), commit statusas " << cst);
            if (cst != MXL_STATUS_OK) continue;  // MXL pats atmetė — tokios antraštės skaitytojas nepamatys
            ++committed;
            check_rejected(in, idx);
        }
        CHECK(committed > 0);  // kitaip šis atvejis nieko netikrintų
    }
    SECTION("validSlices ir totalSlices") {
        struct S {
            std::uint16_t total, valid;
        };
        for (const S s : {S{0, 0}, S{0, 7}, S{540, 541}, S{540, 0xFFFF}}) {
            ++idx;
            const mxlStatus cst = hostile_commit(o, idx, [&](mxlGrainInfo& g) {
                g.totalSlices = s.total;
                g.validSlices = s.valid;
            });
            INFO("totalSlices " << s.total << ", validSlices " << s.valid << ", commit statusas " << cst);
            if (cst != MXL_STATUS_OK) continue;
            ++committed;
            check_rejected(in, idx);
        }
        CHECK(committed > 0);
    }
    // Po netinkamų antraščių sveikas grain'as tebeskaitomas (skaitytojas nesugadintas)
    ++idx;
    mxlStatus st = MXL_STATUS_OK;
    REQUIRE(o.open(idx, st) != nullptr);
    REQUIRE(o.commit() == MXL_STATUS_OK);
    mxlGrainInfo gi{};
    const std::uint32_t* p = nullptr;
    CHECK(in.get(idx, 50'000'000, gi, p) == MXL_STATUS_OK);
    CHECK(p != nullptr);
    CHECK(gi.grainSize == real);
    in.release(d.inst);
    o.release(d.inst);
}

}  // namespace

TEST_CASE("hostile: netikra grain'o antraštė (host) — VideoIn::get grąžina klaidą, ne rodyklę", "[flows][mxl][hostile]") {
    hostile_headers("", 21, "7a210a00-0000-4000-8000-000000000a21");
}

TEST_CASE("hostile: netikra grain'o antraštė (cuda-linear) — VideoIn::get grąžina klaidą, ne rodyklę", "[flows][mxl][gpu][hostile]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    hostile_headers(R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + device_uuid() + R"("}})", 22,
                    "7a210a00-0000-4000-8000-000000000a22");
}

TEST_CASE("hostile: flow aprašas didesnis už tikrą saugyklą — VideoIn::open atmeta", "[flows][mxl][hostile]") {
    Domain d(23);
    const char* id = "7a210a00-0000-4000-8000-000000000a23";
    // Tikras flow'as — 1280×720p50 (grain'as 2 488 320 B)
    mxlFlowWriter w = nullptr;
    mxlFlowConfigInfo cfg{};
    bool created = false;
    REQUIRE(mxlCreateFlowWriter(d.inst, def(id, 1280, 720, false, 50).c_str(), nullptr, &w, &cfg, &created) == MXL_STATUS_OK);
    REQUIRE(created);
    // Rašytojas perrašo aprašą į 1920×1080p50 (5 529 600 B): skaitytojas, patikėjęs aprašu, skaitytų už saugyklos ribų
    const std::filesystem::path fd = d.dir / (std::string(id) + ".mxl-flow") / "flow_def.json";
    REQUIRE(std::filesystem::exists(fd));
    {
        std::ofstream f(fd, std::ios::trunc);
        f << def(id, 1920, 1080, false, 50);
    }
    std::atomic<bool> stop{false};
    std::string err;
    VideoIn in;
    CHECK_FALSE(in.open(d.inst, id, 200, stop, err));
    CHECK(err.find("fill") != std::string::npos);  // check_layout: „fill dydis …“ arba „fill pitch …“
    in.release(d.inst);
    mxlReleaseFlowWriter(d.inst, w);
}

TEST_CASE("hostile: neįmanomi matmenys flow apraše atmetami", "[flows][hostile]") {
    const char* id = "7a210a00-0000-4000-8000-000000000a24";
    vramflow::VideoFormat f{};
    std::string err;
    REQUIRE(vramflow::format_from_flow_def(def(id, 1920, 1080, false, 50), f, err));
    for (const char* w : {"0", "-1", "65537", "2147483647", "99999999999999999999", "1e9", "\"1920\"", ""}) {
        std::string j = def(id, 1920, 1080, false, 50);
        const std::string key = "\"frame_width\":1920";
        const auto at = j.find(key);
        REQUIRE(at != std::string::npos);
        j.replace(at, key.size(), std::string("\"frame_width\":") + w);
        INFO("frame_width " << w);
        CHECK_FALSE(vramflow::format_from_flow_def(j, f, err));
    }
    for (const char* h : {"0", "-2", "65537", "4294967297"}) {
        std::string j = def(id, 1920, 1080, false, 50);
        const std::string key = "\"frame_height\":1080";
        const auto at = j.find(key);
        REQUIRE(at != std::string::npos);
        j.replace(at, key.size(), std::string("\"frame_height\":") + h);
        INFO("frame_height " << h);
        CHECK_FALSE(vramflow::format_from_flow_def(j, f, err));
    }
    // Didžiausias priimamas rastras (65536 × 65536) neperpildo dydžių aritmetikos (64 bitai): fill 1366 · 128 B ir key
    // 21846 · 4 B eilutei
    gfx::Raster big{65536, 65536, false, true};
    CHECK(gfx::grain_bytes(big) == (std::size_t{1366} * 128 + std::size_t{21846} * 4) * 65536);
}
