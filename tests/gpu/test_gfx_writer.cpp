// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// libvramflow-gfx testai ant tikro MXL (host ir cuda-linear): rankinis rašymas == gfx::ref, ritmo režimas [PF-4],
// laikrodžio šuolis atgal [PF-5], vėlavimo taisyklė, klaidų kodai. cuda-linear dalis be GPU praleidžiama (SKIP).
// Test case pavadinimuose „gfx_writer“, kad veiktų `ctest -R gfx_writer`.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <mxl/flow.h>
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
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "func/flows.hpp"
#include "gfx/gfx_ref.hpp"
#include "gfx/gfx_writer_internal.hpp"
#include "gfx/raster.hpp"
#include "vramflow/gfx.h"

namespace gfx = vramflow::gfx;
namespace gi = vramflow::gfx::internal;
using mxlv::func::VideoIn;
using namespace std::chrono_literals;

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

std::string storage_opts(bool cuda) {
    return cuda ? R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + device_uuid() + R"("}})" : std::string{};
}

struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n) {
        dir = std::filesystem::path("/dev/shm") / ("vf-gfxw-" + std::to_string(::getpid()) + "-" + std::to_string(n));
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

int g_flow_n = 0;

struct FlowId {
    std::string id;
    FlowId() {
        char b[64];
        std::snprintf(b, sizeof b, "7a210a00-0000-4000-8000-%012x", 0x600000 + ++g_flow_n);
        id = b;
    }
};

// v210a flow aprašas; interlaced: grain_rate 25 (-> grain 50/1), progresyvus: 50 (kitaip VideoOut atmeta).
std::string def(const std::string& id, int w, int h, bool interlaced, int rate = 0) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow gfx writer test","label":"vramflow gfx writer test",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-gfxw-test:Video"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210a",)"
                  R"("grain_rate":{"numerator":%d,"denominator":1},"frame_width":%d,"frame_height":%d,)"
                  R"("interlace_mode":"%s","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cb","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cr","width":%d,"height":%d,"bit_depth":10}]})",
                  id.c_str(), rate ? rate : (interlaced ? 25 : 50), w, h, interlaced ? "interlaced_tff" : "progressive", w, h, w / 2, h,
                  w / 2, h);
    return b;
}

// RGBA kadras (host + pasirinktinai device kopija) su stride tarpu; tarpas užpildytas 0xAB (privalo būti ignoruojamas).
struct Frame {
    int w, h;
    std::size_t stride;
    std::vector<std::uint8_t> host;
    void* dev = nullptr;
    vfgfx_frame f{};
    Frame(int w_, int h_, std::uint32_t seed, vfgfx_alpha alpha, bool cuda_mem, std::size_t pad = 16)
        : w(w_), h(h_), stride(static_cast<std::size_t>(w_) * 4 + pad), host(stride * static_cast<std::size_t>(h_), 0xAB) {
        std::uint32_t s = seed * 2654435761u + 12345u;
        auto rnd = [&] { s = s * 1664525u + 1013904223u; return s >> 24; };
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                std::uint8_t* px = &host[static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * 4];
                std::uint8_t a = static_cast<std::uint8_t>(rnd());
                if ((x + y) % 11 == 0) a = 0;
                if ((x + y) % 13 == 0) a = 255;
                std::uint8_t c[3] = {static_cast<std::uint8_t>(rnd()), static_cast<std::uint8_t>(rnd()), static_cast<std::uint8_t>(rnd())};
                if (alpha == VFGFX_ALPHA_PREMULTIPLIED)
                    for (auto& v : c) v = static_cast<std::uint8_t>(v * a / 255);  // v <= a: teisingas premultiplied
                px[0] = c[0]; px[1] = c[1]; px[2] = c[2]; px[3] = a;
            }
        f.width = static_cast<std::uint32_t>(w);
        f.height = static_cast<std::uint32_t>(h);
        f.stride = static_cast<std::uint32_t>(stride);
        f.alpha = alpha;
        f.mem = cuda_mem ? VFGFX_MEM_CUDA : VFGFX_MEM_HOST;
        if (cuda_mem) {
            REQUIRE(cudaMalloc(&dev, host.size()) == cudaSuccess);
            REQUIRE(cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
            REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);  // pageable H2D gali grįžti prieš DMA pabaigą (gfx.h)
            f.rgba = dev;
        } else {
            f.rgba = host.data();
        }
    }
    ~Frame() {
        if (dev) cudaFree(dev);
    }
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    // Etalonas: kadro lauko (paritetas) grain'as.
    std::vector<std::uint8_t> ref(const gfx::Raster& r, int parity) const {
        std::vector<std::uint8_t> g(gfx::grain_bytes(r), 0xCD);
        const gfx::AlphaIn m = f.alpha == VFGFX_ALPHA_PREMULTIPLIED ? gfx::AlphaIn::Premultiplied : gfx::AlphaIn::Straight;
        REQUIRE(gfx::ref::rgba_to_v210a(host.data(), w, h, stride, m, r, parity, g.data()));
        return g;
    }
};

// Skaitytojas: lygina flow grain'ą su etalonu.
struct Checker {
    VideoIn in;
    mxlInstance inst;
    gfx::Raster r;
    std::vector<std::uint8_t> buf;
    Checker(mxlInstance i, const std::string& flow, const gfx::Raster& rr) : inst(i), r(rr), buf(gfx::grain_bytes(rr)) {
        std::atomic<bool> stop{false};
        std::string err;
        REQUIRE(in.open(inst, flow, 2000, stop, err, &r));
    }
    ~Checker() { in.release(inst); }
    // true — grain'as egzistuoja ir == want. mismatch — grain'as yra, bet skiriasi.
    bool matches(std::uint64_t idx, const std::vector<std::uint8_t>& want, bool* present = nullptr) {
        mxlGrainInfo info{};
        const std::uint32_t* ptr = nullptr;
        const mxlStatus st = in.get(idx, 0, info, ptr);
        if (present) *present = st == MXL_STATUS_OK;
        if (st != MXL_STATUS_OK) return false;
        if (in.cuda) {
            if (cudaMemcpy(buf.data(), ptr, buf.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        } else {
            std::memcpy(buf.data(), ptr, buf.size());
        }
        return want.size() == buf.size() && std::memcmp(buf.data(), want.data(), buf.size()) == 0;
    }
};

struct Recorder {
    std::mutex m;
    std::vector<std::uint64_t> idx;
    void add(std::uint64_t i) {
        std::lock_guard<std::mutex> l(m);
        idx.push_back(i);
    }
    std::vector<std::uint64_t> snapshot() {
        std::lock_guard<std::mutex> l(m);
        return idx;
    }
    std::size_t size() {
        std::lock_guard<std::mutex> l(m);
        return idx.size();
    }
};

constexpr mxlRational kRate{50, 1};

// Laukimas iki kontroliuojamo laiko šaltinio laiko (PF-5): „dabar“ = mxlGetTime() + shift.
struct FakeClock {
    std::atomic<std::int64_t> shift{0};
    std::uint64_t now() const { return static_cast<std::uint64_t>(static_cast<std::int64_t>(mxlGetTime()) + shift.load()); }
    bool wait(std::uint64_t index, std::int64_t offset, const std::atomic<bool>& stop) const {
        const std::uint64_t target = mxlIndexToTimestamp(&kRate, index) + static_cast<std::uint64_t>(offset);
        std::uint64_t n = now();
        bool stepped = target > n && target - n > 40'000'000ull;
        while (n < target && !stop.load()) {
            const std::uint64_t d = target - n;
            mxlSleepForNs(d < 5'000'000ull ? d : 5'000'000ull);
            const std::uint64_t prev = n;
            n = now();
            if (n + 5'000'000ull < prev) stepped = true;  // šuolis atgal laukimo metu
        }
        return stepped;
    }
};

const gfx::Raster kR1080i{1920, 1080, true, true};

void expect_strictly_consecutive(const std::vector<std::uint64_t>& v) {
    for (std::size_t k = 1; k < v.size(); ++k) REQUIRE(v[k] == v[k - 1] + 1);
}

}  // namespace

TEST_CASE("gfx_writer: rankinis vfgfx_write 4 laukams == ref (host/cuda-linear x host/CUDA kadras)", "[gfx_writer][mxl]") {
    const bool cuda_store = GENERATE(false, true);
    const bool cuda_mem = GENERATE(false, true);
    const vfgfx_alpha alpha = GENERATE(VFGFX_ALPHA_STRAIGHT, VFGFX_ALPHA_PREMULTIPLIED);
    if ((cuda_store || cuda_mem) && !have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(1);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 1920, 1080, true).c_str(), storage_opts(cuda_store).c_str(), &w) == 0);
    Checker chk(d.inst, fid.id, kR1080i);
    Frame fr(1920, 1080, 7, alpha, cuda_mem);
    const std::uint64_t base = mxlGetCurrentIndex(&kRate) & ~1ull;
    for (std::uint64_t k = 0; k < 4; ++k) {
        INFO("indeksas " << base + k << " " << vfgfx_last_error(w));
        REQUIRE(vfgfx_write(w, base + k, &fr.f) == 0);
    }
    CHECK(vfgfx_committed(w) == 4);
    for (std::uint64_t k = 0; k < 4; ++k) {
        bool present = false;
        const bool ok = chk.matches(base + k, fr.ref(kR1080i, static_cast<int>((base + k) & 1)), &present);
        CHECK(present);
        CHECK(ok);
    }
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: progresyvus 382x40 (plotis nesidalija iš 3/6/48), stride su tarpu", "[gfx_writer][mxl]") {
    const bool cuda_store = GENERATE(false, true);
    const bool cuda_mem = GENERATE(false, true);
    if ((cuda_store || cuda_mem) && !have_gpu()) SKIP("nėra CUDA įrenginio");
    const gfx::Raster r{382, 40, false, true};
    Domain d(2);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 382, 40, false).c_str(), storage_opts(cuda_store).c_str(), &w) == 0);
    Checker chk(d.inst, fid.id, r);
    Frame fr(382, 40, 3, VFGFX_ALPHA_STRAIGHT, cuda_mem, 20);
    const std::uint64_t base = mxlGetCurrentIndex(&kRate);
    REQUIRE(vfgfx_write(w, base, &fr.f) == 0);
    REQUIRE(vfgfx_write(w, base + 1, &fr.f) == 0);  // progresyvus: paritetas ignoruojamas, abu indeksai = visas kadras
    CHECK(chk.matches(base, fr.ref(r, 0)));
    CHECK(chk.matches(base + 1, fr.ref(r, 0)));
    vfgfx_writer_destroy(w, d.inst);
}

// R20 / L18: kviečiantysis įrašo CUDA kadrą asinchroniškai savo stream'e ir NEsinchronizuoja prieš vfgfx_write.
// Biblioteka privalo pati palaukti, kol kadras įrašytas (kitaip konvertuoja pusiau įkeltą kadrą — būtent taip krito
// 382x40 testas apkrautoje GPU). Deterministiška: kviečiančiojo stream'as 200 ms miega host funkcijoje, tik tada kopijuoja
// teisingą kadrą. Su vfgfx_set_caller_synced(w, 1) biblioteka nelaukia — tada grain'as PRIVALO skirtis (įrodo, kad
// testas atskiria abu atvejus).
TEST_CASE("gfx_writer: CUDA kadras be kviečiančiojo sinchronizavimo (R20)", "[gfx_writer][mxl]") {
    const bool cuda_store = GENERATE(false, true);
    const bool caller_synced = GENERATE(false, true);
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    const gfx::Raster r{382, 40, false, true};
    Domain d(20);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 382, 40, false).c_str(), storage_opts(cuda_store).c_str(), &w) == 0);
    REQUIRE(vfgfx_set_caller_synced(w, caller_synced ? 1 : 0) == 0);
    Checker chk(d.inst, fid.id, r);
    Frame fr(382, 40, 9, VFGFX_ALPHA_STRAIGHT, true, 20);  // fr.dev — teisingas kadras (jau įkeltas)
    void* late = nullptr;
    REQUIRE(cudaMalloc(&late, fr.host.size()) == cudaSuccess);
    REQUIRE(cudaMemset(late, 0, fr.host.size()) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaStream_t s = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaLaunchHostFunc(s, [](void*) { std::this_thread::sleep_for(200ms); }, nullptr) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(late, fr.dev, fr.host.size(), cudaMemcpyDeviceToDevice, s) == cudaSuccess);
    vfgfx_frame f = fr.f;
    f.rgba = late;
    const std::uint64_t base = mxlGetCurrentIndex(&kRate);
    INFO("cuda_store=" << cuda_store << " caller_synced=" << caller_synced << " " << vfgfx_last_error(w));
    REQUIRE(vfgfx_write(w, base, &f) == 0);  // be cudaStreamSynchronize(s) — tyčia
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    const bool ok = chk.matches(base, fr.ref(r, 0));
    if (caller_synced) CHECK_FALSE(ok);  // biblioteka nelaukė → nuliai, ne kadras
    else CHECK(ok);
    vfgfx_writer_destroy(w, d.inst);
    cudaStreamDestroy(s);
    cudaFree(late);
}

TEST_CASE("gfx_writer: klaidų kodai (INVALID_ARG, GEOMETRY, flow aprašas)", "[gfx_writer][mxl]") {
    Domain d(3);
    FlowId fid;
    vfgfx_writer w = nullptr;
    CHECK(vfgfx_writer_create(nullptr, "{}", nullptr, &w) == 2001);
    CHECK(w == nullptr);
    CHECK(std::strlen(vfgfx_last_error(nullptr)) > 0);
    CHECK(vfgfx_writer_create(d.inst, "nesąmonė", nullptr, &w) == 2001);
    // v210 (be alfa) — ne v210a
    std::string v210 = def(fid.id, 1920, 1080, true);
    v210.replace(v210.find("video/v210a"), 11, "video/v210");
    CHECK(vfgfx_writer_create(d.inst, v210.c_str(), nullptr, &w) == 2001);
    // progresyvus 25p: grainRate != 50/1 -> GEOMETRY
    CHECK(vfgfx_writer_create(d.inst, def(fid.id, 64, 16, false, 25).c_str(), nullptr, &w) == 2002);
    CHECK(w == nullptr);

    FlowId fid2;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid2.id, 64, 16, false).c_str(), nullptr, &w) == 0);
    Frame ok(64, 16, 1, VFGFX_ALPHA_STRAIGHT, false);
    vfgfx_frame bad = ok.f;
    CHECK(vfgfx_write(w, 1, nullptr) == 2001);
    bad.rgba = nullptr;
    CHECK(vfgfx_write(w, 1, &bad) == 2001);
    bad = ok.f;
    bad.width = 62;
    CHECK(vfgfx_write(w, 1, &bad) == 2002);
    CHECK(std::strlen(vfgfx_last_error(w)) > 0);
    bad = ok.f;
    bad.height = 18;
    CHECK(vfgfx_submit(w, &bad) == 2002);
    bad = ok.f;
    bad.stride = 64 * 4 - 4;
    CHECK(vfgfx_write(w, 1, &bad) == 2002);
    bad = ok.f;
    bad.mem = static_cast<vfgfx_mem>(7);
    CHECK(vfgfx_write(w, 1, &bad) == 2001);
    CHECK(vfgfx_committed(w) == 0);
    CHECK(vfgfx_stop_pacing(w) == 0);  // ne ritmo režime: nieko nedaro
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: ritmo režimas — be submit grain'ų nėra, po submit >= 48/s, indeksai iš eilės == ref [PF-4]",
          "[gfx_writer][mxl][pf4]") {
    const bool cuda_store = GENERATE(false, true);
    if (cuda_store && !have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(4);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 1920, 1080, true).c_str(), storage_opts(cuda_store).c_str(), &w) == 0);
    Checker chk(d.inst, fid.id, kR1080i);
    Recorder rec;
    gi::PacingHooks hk;
    hk.on_commit = [&](std::uint64_t i) { rec.add(i); };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame a(1920, 1080, 11, VFGFX_ALPHA_STRAIGHT, cuda_store);
    Frame b(1920, 1080, 22, VFGFX_ALPHA_PREMULTIPLIED, cuda_store);
    const std::vector<std::uint8_t> ra[2] = {a.ref(kR1080i, 0), a.ref(kR1080i, 1)};
    const std::vector<std::uint8_t> rb[2] = {b.ref(kR1080i, 0), b.ref(kR1080i, 1)};

    REQUIRE(vfgfx_start_pacing(w, 0) == 0);
    CHECK(vfgfx_start_pacing(w, 0) == 2004);  // jau paleista
    std::this_thread::sleep_for(500ms);
    CHECK(vfgfx_committed(w) == 0);  // [PF-4]: iki pirmo submit grain'ų nėra
    CHECK(rec.size() == 0);
    CHECK(gi::pacing_stats(w).no_frame >= 20);

    REQUIRE(vfgfx_submit(w, &a.f) == 0);
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t verified = 0, bad = 0, missing = 0;
    auto verify_new = [&](const std::vector<std::uint8_t>* want) {
        const auto snap = rec.snapshot();
        for (; verified < snap.size(); ++verified) {
            bool present = false;
            if (!chk.matches(snap[verified], want[snap[verified] & 1], &present)) ++(present ? bad : missing);
        }
    };
    while (std::chrono::steady_clock::now() - t0 < 1000ms) {
        verify_new(ra);
        std::this_thread::sleep_for(2ms);
    }
    const std::size_t n1 = rec.size();
    CHECK(n1 >= 48);  // [PF-4]
    CHECK(vfgfx_committed(w) >= 48);
    CHECK(vfgfx_write(w, 1, &a.f) == 2004);  // rankinis ritmo metu
    CHECK(std::strlen(vfgfx_last_error(w)) > 0);

    // Naujas kadras: po poros laukų visi indeksai == b atitinkamo pariteto laukas (swap tik tarp kadrų, ne per pusę lauko).
    REQUIRE(vfgfx_submit(w, &b.f) == 0);
    std::this_thread::sleep_for(100ms);
    verified = rec.size();  // per swap'ą išleisti grain'ai gali būti seno ARBA naujo kadro — netikrinami
    const auto t1 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t1 < 300ms) {  // lango (ring) dydis ribotas: tikrinam sekdami
        verify_new(rb);
        std::this_thread::sleep_for(2ms);
    }
    vfgfx_stop_pacing(w);
    const auto all = rec.snapshot();
    expect_strictly_consecutive(all);
    CHECK(all.size() >= 48 + 15);
    CHECK(bad == 0);
    CHECK(missing == 0);
    const auto st = gi::pacing_stats(w);
    CHECK(st.errors == 0);
    CHECK(st.committed == vfgfx_committed(w));
    CHECK(vfgfx_write(w, all.back() + 5, &a.f) == 0);  // po stop — vėl galima
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: ritmas — kryžminės atminties poros (host kadras -> cuda-linear, CUDA kadras -> host), progresyvus",
          "[gfx_writer][mxl]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    const bool cuda_store = GENERATE(false, true);
    const gfx::Raster r{382, 40, false, true};
    Domain d(5);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 382, 40, false).c_str(), storage_opts(cuda_store).c_str(), &w) == 0);
    Checker chk(d.inst, fid.id, r);
    Recorder rec;
    gi::PacingHooks hk;
    hk.on_commit = [&](std::uint64_t i) { rec.add(i); };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame f(382, 40, 5, VFGFX_ALPHA_STRAIGHT, !cuda_store);  // priešinga atmintis nei saugykla
    REQUIRE(vfgfx_start_pacing(w, 0) == 0);
    REQUIRE(vfgfx_submit(w, &f.f) == 0);
    std::this_thread::sleep_for(300ms);
    vfgfx_stop_pacing(w);
    const auto all = rec.snapshot();
    REQUIRE(all.size() >= 10);
    expect_strictly_consecutive(all);
    const auto want = f.ref(r, 0);
    for (std::size_t k = all.size() - 5; k < all.size(); ++k) CHECK(chk.matches(all[k], want));
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: laikrodžio šuolis atgal ritmo gijoje — joks indeksas du kartus, jokio catch-up [PF-5]",
          "[gfx_writer][mxl][pf5]") {
    const bool cuda_store = GENERATE(false, true);
    if (cuda_store && !have_gpu()) SKIP("nėra CUDA įrenginio");
    Domain d(6);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 1920, 1080, true).c_str(), storage_opts(cuda_store).c_str(), &w) == 0);
    Checker chk(d.inst, fid.id, kR1080i);
    FakeClock clk;
    Recorder rec;
    gi::PacingHooks hk;
    hk.wait = [&](std::uint64_t i, std::int64_t off, const std::atomic<bool>& stop) { return clk.wait(i, off, stop); };
    hk.now = [&] { return clk.now(); };
    hk.on_commit = [&](std::uint64_t i) { rec.add(i); };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame f(1920, 1080, 9, VFGFX_ALPHA_STRAIGHT, cuda_store);
    const std::vector<std::uint8_t> want[2] = {f.ref(kR1080i, 0), f.ref(kR1080i, 1)};
    REQUIRE(vfgfx_submit(w, &f.f) == 0);
    REQUIRE(vfgfx_start_pacing(w, 0) == 0);

    std::this_thread::sleep_for(400ms);
    const std::size_t before = rec.size();
    CHECK(before >= 15);
    clk.shift.store(-400'000'000);  // laikrodis šoko 400 ms atgal (~20 laukų)
    std::this_thread::sleep_for(300ms);  // gija laukia, kol laikas pasivys: per tą laiką <= 1 (jau vykdomas) commit
    const std::size_t during = rec.size();
    CHECK(during - before <= 1);  // jokio „runaway“ / pakartotinio rašymo šuolio metu
    std::this_thread::sleep_for(600ms);  // laikas pasivijo; ritmas atsistato
    vfgfx_stop_pacing(w);

    const auto all = rec.snapshot();
    expect_strictly_consecutive(all);  // vienas po kito, be dublių ir be praleidimų (praleidimas = pavėlavimas, čia neturi būti)
    CHECK(all.size() > during + 5);    // po šuolio ritmas tęsiasi
    // Viso laiko biudžetas: negali būti daugiau grain'ų, nei leidžia laikas (50/s) — catch-up burst'o nėra.
    CHECK(all.size() <= static_cast<std::size_t>(50.0 * 1.35) + 4);
    CHECK(gi::pacing_stats(w).clock_backward >= 1);
    CHECK(gi::pacing_stats(w).errors == 0);
    CHECK(chk.matches(all.back(), want[all.back() & 1]));
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: pavėlavusi gija — vėluojantis indeksas įrašomas, praleisti laukai skaičiuojami (kaip gen)", "[gfx_writer][mxl]") {
    Domain d(7);
    FlowId fid;
    const gfx::Raster r{128, 72, true, true};
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 128, 72, true).c_str(), nullptr, &w) == 0);
    Recorder rec;
    std::atomic<int> calls{0};
    gi::PacingHooks hk;
    hk.wait = [&](std::uint64_t i, std::int64_t off, const std::atomic<bool>& stop) {
        const bool b = mxlv::func::wait_until(i, off, stop);
        if (calls.fetch_add(1) == 12) std::this_thread::sleep_for(70ms);  // gija „pasimetė“ ~3,5 lauko
        return b;
    };
    hk.on_commit = [&](std::uint64_t i) { rec.add(i); };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame f(128, 72, 4, VFGFX_ALPHA_STRAIGHT, false);
    REQUIRE(vfgfx_submit(w, &f.f) == 0);
    REQUIRE(vfgfx_start_pacing(w, 0) == 0);
    std::this_thread::sleep_for(700ms);
    vfgfx_stop_pacing(w);
    const auto all = rec.snapshot();
    REQUIRE(all.size() > 20);
    // Lygiai viena pertrauka: vėluojantis indeksas (13-as iškvietimas) įrašytas, po jo praleista >= 2 laukų.
    int gaps = 0;
    for (std::size_t k = 1; k < all.size(); ++k) {
        REQUIRE(all[k] > all[k - 1]);
        if (all[k] != all[k - 1] + 1) {
            ++gaps;
            CHECK(all[k] - all[k - 1] >= 3);
            CHECK(all[k] - all[k - 1] <= 6);
        }
    }
    CHECK(gaps == 1);
    CHECK(gi::pacing_stats(w).late_skips >= 2);
    vfgfx_writer_destroy(w, d.inst);
}

// --- Fix round 1 ---------------------------------------------------------------------------------------------------------

TEST_CASE("gfx_writer: offset_ns +60 ms / -20 ms — indeksai iš eilės, be praleidimų pastovioje būsenoje", "[gfx_writer][mxl][offset]") {
    const std::int64_t off = GENERATE(60'000'000, -20'000'000);
    Domain d(8);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 128, 72, true).c_str(), nullptr, &w) == 0);
    Recorder rec;
    gi::PacingHooks hk;
    hk.on_commit = [&](std::uint64_t i) { rec.add(i); };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame f(128, 72, 4, VFGFX_ALPHA_STRAIGHT, false);
    REQUIRE(vfgfx_submit(w, &f.f) == 0);
    REQUIRE(vfgfx_start_pacing(w, off) == 0);
    std::this_thread::sleep_for(700ms);
    vfgfx_stop_pacing(w);
    const auto all = rec.snapshot();
    CHECK(all.size() >= 25);
    expect_strictly_consecutive(all);
    CHECK(gi::pacing_stats(w).late_skips == 0);
    CHECK(gi::pacing_stats(w).errors == 0);
    // commit'as ties index laiku + offset: +60 ms -> grain'as pasirodo ne anksčiau nei laikas
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: stop -> start su neigiamu offset — indeksai nesikartoja", "[gfx_writer][mxl][offset]") {
    Domain d(9);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 128, 72, true).c_str(), nullptr, &w) == 0);
    Recorder rec;
    gi::PacingHooks hk;
    hk.on_commit = [&](std::uint64_t i) { rec.add(i); };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame f(128, 72, 4, VFGFX_ALPHA_STRAIGHT, false);
    REQUIRE(vfgfx_submit(w, &f.f) == 0);
    for (int round = 0; round < 3; ++round) {
        REQUIRE(vfgfx_start_pacing(w, -30'000'000) == 0);
        std::this_thread::sleep_for(150ms);
        REQUIRE(vfgfx_stop_pacing(w) == 0);
    }
    const auto all = rec.snapshot();
    REQUIRE(all.size() >= 6);
    for (std::size_t k = 1; k < all.size(); ++k) REQUIRE(all[k] > all[k - 1]);
    CHECK(gi::pacing_stats(w).errors == 0);
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: lygiagretūs vfgfx_write ir vfgfx_submit (bendri staging buferiai)", "[gfx_writer][mxl][race]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio (kryžminė atmintis)");
    const bool cuda_store = GENERATE(false, true);
    const gfx::Raster r{640, 360, true, true};
    Domain d(10);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 640, 360, true).c_str(), storage_opts(cuda_store).c_str(), &w) == 0);
    Checker chk(d.inst, fid.id, r);
    Frame f(640, 360, 21, VFGFX_ALPHA_STRAIGHT, !cuda_store);  // priešinga atmintis: naudojamas staging
    std::atomic<int> sub_bad{0}, wr_bad{0};
    const std::uint64_t base = mxlGetCurrentIndex(&kRate) & ~1ull;
    std::thread ts([&] {
        for (int k = 0; k < 40; ++k)
            if (vfgfx_submit(w, &f.f) != 0) ++sub_bad;
    });
    std::thread tw([&] {
        for (int k = 0; k < 8; ++k)
            if (vfgfx_write(w, base + k, &f.f) != 0) ++wr_bad;
    });
    ts.join();
    tw.join();
    CHECK(sub_bad == 0);
    CHECK(wr_bad == 0);
    for (int k = 0; k < 8; ++k) {
        bool present = false;
        CHECK(chk.matches(base + k, f.ref(r, (base + k) & 1), &present));
        CHECK(present);
    }
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: flow jau egzistuoja -> STATE; nežinomas deviceUuid -> INVALID_ARG", "[gfx_writer][mxl]") {
    Domain d(11);
    FlowId fid;
    vfgfx_writer w = nullptr, w2 = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 64, 16, false).c_str(), nullptr, &w) == 0);
    CHECK(vfgfx_writer_create(d.inst, def(fid.id, 64, 16, false).c_str(), nullptr, &w2) == VFGFX_ERR_STATE);
    CHECK(w2 == nullptr);
    FlowId f2;
    CHECK(vfgfx_writer_create(d.inst, def(f2.id, 64, 16, false).c_str(),
                              R"({"payload":{"backend":"cuda-linear","deviceUuid":"GPU-00000000-0000-0000-0000-000000000000"}})",
                              &w2) == VFGFX_ERR_INVALID_ARG);
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: išimtis ritmo gijoje neužmuša proceso, klaida užfiksuota", "[gfx_writer][mxl]") {
    Domain d(12);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 128, 72, true).c_str(), nullptr, &w) == 0);
    gi::PacingHooks hk;
    hk.on_commit = [](std::uint64_t) { throw std::runtime_error("testinė išimtis"); };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame f(128, 72, 4, VFGFX_ALPHA_STRAIGHT, false);
    REQUIRE(vfgfx_submit(w, &f.f) == 0);
    REQUIRE(vfgfx_start_pacing(w, 0) == 0);
    std::this_thread::sleep_for(200ms);
    CHECK(vfgfx_stop_pacing(w) == 0);
    CHECK(std::string(vfgfx_last_error(w)).find("ritmo gija nutrūko") != std::string::npos);
    CHECK(gi::pacing_stats(w).errors >= 1);
    vfgfx_writer_destroy(w, d.inst);
}

TEST_CASE("gfx_writer: ritmo gijos commit klaida eina į ritmo klaidos lauką, ne į kviečiančiosios", "[gfx_writer][mxl]") {
    Domain d(13);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 128, 72, true).c_str(), nullptr, &w) == 0);
    gi::PacingHooks hk;
    hk.fail_commit = [] { return true; };
    REQUIRE(gi::set_pacing_hooks(w, hk) == 0);
    Frame f(128, 72, 4, VFGFX_ALPHA_STRAIGHT, false);
    Frame bad(64, 16, 4, VFGFX_ALPHA_STRAIGHT, false);
    CHECK(vfgfx_submit(w, &bad.f) == 2002);  // kviečiančiosios klaida
    const std::string caller = vfgfx_last_error(w);
    REQUIRE_FALSE(caller.empty());
    REQUIRE(vfgfx_submit(w, &f.f) == 0);
    REQUIRE(vfgfx_start_pacing(w, 0) == 0);
    std::this_thread::sleep_for(200ms);
    vfgfx_stop_pacing(w);
    CHECK(gi::pacing_stats(w).errors >= 5);
    CHECK(vfgfx_committed(w) == 0);
    CHECK(std::string(vfgfx_last_error(w)) == caller);  // ritmo klaida kviečiančiosios nepakeitė
    CHECK(gi::pacing_error(w).find("CommitGrain") != std::string::npos);
    vfgfx_writer_destroy(w, d.inst);
}

// Apsauga (capacity G, N >= 2 tyrimas): keli cuda-linear rašytojai viename procese, ritmo režimas, kaip gfxgen — visi sukuriami,
// kiekvienam submit'inamas CUDA kadras (iškart atlaisvinamas), tada paleidžiamos visos ritmo gijos. Kiekvieno rašytojo ABU
// paritetai turi būti == ref (bibliotekos būsena per rašytoją; capacity G N >= 2 klaidos šaltinis buvo gfxgen įkėlimas).
TEST_CASE("gfx_writer: keli cuda-linear rašytojai viename procese, ritmas — visų paritetai == ref", "[gfx_writer][mxl][multi]") {
    if (!have_gpu()) SKIP("nėra CUDA įrenginio");
    constexpr int kW = 3;
    const gfx::Raster r{384, 216, true, true};
    Domain d(14);
    std::vector<FlowId> fid(kW);
    std::vector<vfgfx_writer> w(kW, nullptr);
    for (int k = 0; k < kW; ++k)
        REQUIRE(vfgfx_writer_create(d.inst, def(fid[k].id, r.w, r.h, true).c_str(), storage_opts(true).c_str(), &w[k]) == 0);
    std::vector<std::vector<std::uint8_t>> want[2];
    std::vector<std::unique_ptr<Recorder>> rec;
    for (int k = 0; k < kW; ++k) {
        rec.push_back(std::make_unique<Recorder>());
        gi::PacingHooks hk;
        Recorder* rp = rec.back().get();
        hk.on_commit = [rp](std::uint64_t i) { rp->add(i); };
        REQUIRE(gi::set_pacing_hooks(w[k], hk) == 0);
        auto f = std::make_unique<Frame>(r.w, r.h, 100 + k, (k & 1) ? VFGFX_ALPHA_PREMULTIPLIED : VFGFX_ALPHA_STRAIGHT, true, 0);
        want[0].push_back(f->ref(r, 0));
        want[1].push_back(f->ref(r, 1));
        REQUIRE(vfgfx_submit(w[k], &f->f) == 0);
    }  // kadrai atlaisvinti (biblioteka jų neišsaugo)
    for (int k = 0; k < kW; ++k) REQUIRE(vfgfx_start_pacing(w[k], 0) == 0);
    std::vector<std::unique_ptr<Checker>> chk;
    for (int k = 0; k < kW; ++k) chk.push_back(std::make_unique<Checker>(d.inst, fid[k].id, r));
    std::vector<std::size_t> verified(kW, 0), ok(kW, 0), bad_even(kW, 0), bad_odd(kW, 0);
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < 500ms) {
        for (int k = 0; k < kW; ++k) {
            const auto snap = rec[k]->snapshot();
            for (; verified[k] < snap.size(); ++verified[k]) {
                const std::uint64_t i = snap[verified[k]];
                bool present = false;
                if (chk[k]->matches(i, want[i & 1][k], &present)) ++ok[k];
                else if (present) ++((i & 1) ? bad_odd : bad_even)[k];
            }
        }
        std::this_thread::sleep_for(2ms);
    }
    for (int k = 0; k < kW; ++k) vfgfx_stop_pacing(w[k]);
    for (int k = 0; k < kW; ++k) {
        INFO("rašytojas " << k);
        CHECK(ok[k] >= 15);
        CHECK(bad_even[k] == 0);
        CHECK(bad_odd[k] == 0);
    }
    chk.clear();
    for (int k = 0; k < kW; ++k) vfgfx_writer_destroy(w[k], d.inst);
}

// Host saugyklos rašytuvas su host kadrais neturi sukurti CUDA konteksto (srautas s_conv kuriamas tingiai, tik CUDA kadrui).
// Tikrinama per vairuotojo API (dlopen, be tiesioginio linkinimo): pirminis konteksto būsena po create + write + ritmo.
// catch_discover_tests kiekvieną test case paleidžia atskirame procese, todėl kiti testai kontekstu neužteršia.
TEST_CASE("gfx_writer: host rašytuvas nekuria CUDA konteksto", "[gfx_writer][mxl][nocontext]") {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) SKIP("nėra libcuda.so.1");
    using InitFn = int (*)(unsigned);
    using StateFn = int (*)(int, unsigned*, int*);
    auto cu_init = reinterpret_cast<InitFn>(dlsym(lib, "cuInit"));
    auto cu_state = reinterpret_cast<StateFn>(dlsym(lib, "cuDevicePrimaryCtxGetState"));
    if (!cu_init || !cu_state || cu_init(0) != 0) SKIP("vairuotojo API neprieinamas (nėra GPU)");
    const auto active = [&] {
        unsigned flags = 0;
        int act = -1;
        REQUIRE(cu_state(0, &flags, &act) == 0);
        return act;
    };
    if (active() != 0) SKIP("CUDA kontekstą jau sukūrė ankstesnis testas šiame procese (paleisk atskirai: ctest -R nekuria)");
    Domain d(40);
    FlowId fid;
    vfgfx_writer w = nullptr;
    REQUIRE(vfgfx_writer_create(d.inst, def(fid.id, 64, 16, true).c_str(), nullptr, &w) == 0);
    Frame fr(64, 16, 5, VFGFX_ALPHA_STRAIGHT, false);
    const std::uint64_t base = mxlGetCurrentIndex(&kRate) & ~1ull;
    REQUIRE(vfgfx_write(w, base, &fr.f) == 0);
    REQUIRE(vfgfx_write(w, base + 1, &fr.f) == 0);
    CHECK(active() == 0);  // host -> host: jokio CUDA konteksto
    vfgfx_writer_destroy(w, d.inst);
    CHECK(active() == 0);
}
