// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// flows.cpp testai ant tikro MXL: v210a plokštumų išdėstymas (host ir cuda-linear), check_layout atmetimai, grainRate 50/1
// visada (R9) VideoOut ir VideoIn. Host dalis nereikalauja GPU; cuda-linear dalis be GPU praleidžiama (SKIP).
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cuda_runtime_api.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

// Laikinas domenas /dev/shm/vf-flows-<pid>-<n> + instancija. history_ms > 0 — domeno ring'o istorija (options.json).
struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n, int history_ms = 0) {
        dir = std::filesystem::path("/dev/shm") / ("vf-flows-" + std::to_string(::getpid()) + "-" + std::to_string(n));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        if (history_ms > 0)
            std::ofstream(dir / "options.json") << "{\"urn:x-mxl:option:history_duration/v1.0\": " << history_ms << "000000}";
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

// MXL cuda-linear įskiepis naują žiedą nulina asinchroniškai (cudaMemset numatytajame sraute) ir pabaigos nelaukia
// (CudaLinearBackend.cpp, prepare). Funkcijos į žiedą rašo kernel'iais non-blocking sraute, kuris su numatytuoju
// nesinchronizuojamas: kol nulinimas nebaigtas, ką tik įrašyti duomenys gali būti perrašyti nuliais, o greitas
// atlaisvinimas palieka GPU darbą atlaisvintai atminčiai (compute-sanitizer: „failed to handle a hardware exception“,
// 3 paleidimai iš 5). VideoOut::create privalo grįžti tik baigus nulinimą.
TEST_CASE("flows: cuda-linear VideoOut::create grįžta tik baigus žiedo nulinimą — kitame sraute įrašyti duomenys nedingsta", "[flows][mxl][gpu]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(8, 2000);  // 2 s istorija: 100 × 5,5 MiB 1080p50 grain'ų = 527 MiB — nulinimas trunka pakankamai ilgai
    VideoOut o;
    std::string err;
    const std::string opts = R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + device_uuid() + R"("}})";
    const gfx::Raster r{1920, 1080, false, false};
    REQUIRE(o.create(d.inst, def("7a210a00-0000-4000-8000-0000000000f8", "video/v210", false, 50), opts, err, vramflow::VideoFormat{r, {50, 1}}));
    // Tiesioginis požymis: numatytajame sraute (ten įskiepis paleido cudaMemset) nebėra nebaigto darbo
    const cudaError_t pending = cudaStreamQuery(nullptr);
    CHECK(pending == cudaSuccess);  // cudaErrorNotReady (600) — create() grįžo nulinimui dar vykstant
    REQUIRE(is_cuda_layout(o.layout));
    REQUIRE(o.layout.slotCount >= 50);
    cudaStream_t s{};
    REQUIRE(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    // Rašoma iškart po create(): pirmo, vidurinio ir paskutinio slot'o pabaigoje po 64 KiB baitų 0xA5
    constexpr std::size_t kN = 64 * 1024;
    const std::size_t gb = gfx::grain_bytes(r);
    std::vector<std::uint8_t*> at;
    for (const std::uint32_t k : {std::uint32_t{0}, o.layout.slotCount / 2, o.layout.slotCount - 1})
        at.push_back(static_cast<std::uint8_t*>(slot_ptr(o.layout, o.slots[k])) + gb - kN);
    for (std::uint8_t* p : at) REQUIRE(cudaMemsetAsync(p, 0xA5, kN, s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);  // dabar baigtas ir bet koks dar vykęs nulinimas
    std::vector<std::uint8_t> h(kN);
    std::size_t lost = 0;
    for (std::uint8_t* p : at) {
        REQUIRE(cudaMemcpy(h.data(), p, kN, cudaMemcpyDeviceToHost) == cudaSuccess);
        lost += static_cast<std::size_t>(std::count_if(h.begin(), h.end(), [](std::uint8_t v) { return v != 0xA5; }));
    }
    CHECK(lost == 0);
    cudaStreamDestroy(s);
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

// Regresija (capacity G, N >= 2): cudaMemcpy H2D iš pageable atminties gali grįžti prieš DMA pabaigą (CUDA „API
// synchronization behavior“), o libvramflow-gfx kadrą skaito savo non-blocking srautu (kitas cudart, jokios tvarkos su
// legacy default stream) — gfxgen pirmas (lyginis) laukas buvo konvertuojamas iš dalinai įkelto kadro. upload_h2d turi
// grąžinti tik baigus: iškart po jo kitas non-blocking srautas privalo matyti VISUS naujus baitus.
#include <vector>

#include "func/cuda_upload.hpp"

TEST_CASE("upload_h2d: grįžus duomenys matomi kitam non-blocking srautui (pageable šaltinis)", "[flows][upload]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    // logo RGBA kadras (384x216x4, kaip gfxgen) ir 4 MiB: be sinchronizavimo išmatuota ~50 % / ~90 % iteracijų neatitinka
    const std::size_t kBytes = GENERATE(std::size_t{384 * 216 * 4}, std::size_t{4} << 20);
    std::vector<std::uint8_t> host(kBytes), back(kBytes);
    void *dev = nullptr, *dev2 = nullptr;
    REQUIRE(cudaMalloc(&dev, kBytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&dev2, kBytes) == cudaSuccess);
    cudaStream_t s{};
    REQUIRE(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    int bad = 0;
    for (int it = 1; it <= 200; ++it) {
        std::fill(host.begin(), host.end(), static_cast<std::uint8_t>(it));
        REQUIRE(mxlv::func::upload_h2d(dev, host.data(), kBytes) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(dev2, dev, kBytes, cudaMemcpyDeviceToDevice, s) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(back.data(), dev2, kBytes, cudaMemcpyDeviceToHost, s) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
        if (std::memcmp(back.data(), host.data(), kBytes) != 0) ++bad;
    }
    INFO("baitų " << kBytes << ", neatitikusių iteracijų " << bad << " iš 200");
    CHECK(bad == 0);
    cudaStreamDestroy(s);
    cudaFree(dev);
    cudaFree(dev2);
}

// --- P2: formatui neutralūs flow'ai -----------------------------------------------------------------------------------
#include "common/format.hpp"

TEST_CASE("flows: create/open tikrina grainRate prieš formatą", "[flows][mxl][format]") {
    std::atomic<bool> stop{false};
    std::string err;
    vramflow::VideoFormat p25{}, p50{};
    REQUIRE(vramflow::parse_format("1080p25", p25, err));
    REQUIRE(vramflow::parse_format("1080p50", p50, err));
    Domain d(20);
    const std::string id = flow_id(kRoleGen, 0);
    VideoOut o;
    REQUIRE(o.create(d.inst, video_def(kRoleGen, 0, 1, p25), "", err, p25));
    CHECK(o.fmt == p25);
    VideoIn in;
    REQUIRE(in.open(d.inst, id, 1000, stop, err, &p25));
    CHECK(in.cfg.common.grainRate.numerator == 25);
    CHECK(in.cfg.common.grainRate.denominator == 1);
    CHECK(in.fmt == p25);
    in.release(d.inst);
    VideoIn any;  // be lūkesčio: formatas iš flow aprašo
    REQUIRE(any.open(d.inst, id, 1000, stop, err, static_cast<const vramflow::VideoFormat*>(nullptr)));
    CHECK(any.fmt == p25);
    CHECK(any.raster == p25.raster);
    any.release(d.inst);
    VideoIn bad;  // lūkestis 1080p50 → klaida su abiem dažniais
    CHECK_FALSE(bad.open(d.inst, id, 1000, stop, err, &p50));
    INFO(err);
    CHECK(err.find("25/1") != std::string::npos);
    CHECK(err.find("50/1") != std::string::npos);
    bad.release(d.inst);
    o.release(d.inst);
    // writer'is: aprašo dažnis neatitinka formato → klaida
    Domain d2(21);
    VideoOut o2;
    CHECK_FALSE(o2.create(d2.inst, video_def(kRoleGen, 0, 1, p25), "", err, p50));
    CHECK(err.find("grainRate") != std::string::npos);
    o2.release(d2.inst);
    // 1080i50: grain'o ritmas 50/1, flow apraše 25/1
    Domain d3(22);
    VideoOut o3;
    REQUIRE(o3.create(d3.inst, video_def(kRoleGen, 0, 1, vramflow::kFmt1080i50), "", err, vramflow::kFmt1080i50));
    VideoIn i3;
    REQUIRE(i3.open(d3.inst, id, 1000, stop, err, &vramflow::kFmt1080i50));
    CHECK(i3.cfg.common.grainRate.numerator == 50);
    i3.release(d3.inst);
    o3.release(d3.inst);
}

TEST_CASE("flows: AudioOut priima ilgiausią grain'ą (25/1 → 1920, 59,94 → 801 mėginių)", "[flows][mxl][format]") {
    std::string err;
    for (const vramflow::Rational r : {vramflow::Rational{50, 1}, vramflow::Rational{25, 1}, vramflow::Rational{60000, 1001}}) {
        Domain d(23);
        AudioOut a;
        INFO(r.num << "/" << r.den);
        CHECK(a.create(d.inst, audio_def(0, 1), err, r));
        a.release(d.inst);
    }
}

TEST_CASE("flows: wait_until(rate) laukia indekso laiko pagal dažnį", "[flows][format]") {
    std::atomic<bool> stop{false};
    const mxlRational r25{25, 1};
    const std::uint64_t i = mxlGetCurrentIndex(&r25) + 1;  // ≤ 1,5 grain'o iki tikslo (indeksas apvalinamas) — ne šuolis
    CHECK_FALSE(wait_until(r25, i, 0, stop));
    CHECK(now_tai() >= mxlIndexToTimestamp(&r25, i));
    CHECK(mxl_rate(vramflow::Rational{60000, 1001}).numerator == 60000);
}
