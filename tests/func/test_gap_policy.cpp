// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V17 (planas docs/planai/2026-11-12-tick-variklis.md §6 D4, §9 „4 etapas“; production-grade §6 K3): funkcijų spragų politika.
//   * parinktys → EngineConfig (--on-loss, --hold-ms) abiem ribų režimais, numatyta invalid / 1000 ms;
//   * GapFill: Repeat / Black kopija ir juodas grain'as pagal saugyklą (host ir cuda-linear) per MxlVideoOutput;
//   * žemupio blokuojantis skaitymas spragos indekse grįžta per < 1 ms nuo spragos commit'o (ne po periodo);
//   * [.][measure] — vienos spragos kaina (C / H: host memcpy, G: D2D) 1080i50 grain'ui.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cuda_runtime_api.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "engine/system_clock.hpp"
#include "func/engine_glue.hpp"
#include "func/flows.hpp"
#include "mxlio/cuda_uuid.hpp"
#include "mxlio/slots.hpp"
#include "runtime/input_slot.hpp"

using namespace mxlv;
using namespace mxlv::engine;
using mxlv::func::GapFill;
using mxlv::func::VideoIn;
using mxlv::func::VideoOut;
namespace gfx = vramflow::gfx;

namespace {

struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n) {
        dir = std::filesystem::path("/dev/shm") / ("vgp-" + std::to_string(::getpid()) + "-" + std::to_string(n));
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

bool have_gpu() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string opts_for(bool cuda) {
    return cuda ? R"({"payload":{"backend":"cuda-linear","deviceUuid":")" + mxlv::cuda_uuid_string(0) + R"("}})" : std::string{};
}

constexpr vramflow::VideoFormat k1080i = vramflow::kFmt1080i50;

// Grain'o i turinys į host buferį (cuda — D2H).
std::vector<std::uint8_t> read_grain(VideoIn& in, std::uint64_t i, std::uint32_t* flags = nullptr) {
    mxlGrainInfo info{};
    const std::uint32_t* p = nullptr;
    REQUIRE(in.get(i, 0, info, p) == MXL_STATUS_OK);
    if (flags) *flags = info.flags;
    std::vector<std::uint8_t> b(gfx::grain_bytes(gfx::kHd1080i));
    if (in.cuda) REQUIRE(cudaMemcpy(b.data(), p, b.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    else std::memcpy(b.data(), p, b.size());
    return b;
}

// Commit'ina turinį: kiekvienas baitas = (i * 7 + k) & 0xff (cuda — H2D).
bool commit_pattern(MxlVideoOutput& port, std::uint64_t i, bool cuda) {
    void* slot = nullptr;
    if (port.open(i, slot) != OpenResult::Ok) return false;
    std::vector<std::uint8_t> b(gfx::grain_bytes(gfx::kHd1080i));
    for (std::size_t k = 0; k < b.size(); ++k) b[k] = static_cast<std::uint8_t>((i * 7 + k) & 0xff);
    if (cuda) {
        // cudaMemcpy iš pageable atminties gali grįžti prieš DMA pabaigą; GapFill D2D eina kitu (non-blocking) stream'u
        if (cudaMemcpy(slot, b.data(), b.size(), cudaMemcpyHostToDevice) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
            return false;
    } else {
        std::memcpy(slot, b.data(), b.size());
    }
    return port.commit();
}

struct Reader {
    VideoIn in;
    mxlInstance inst;
    Reader(mxlInstance i, const std::string& id) : inst(i) {
        std::atomic<bool> stop{false};
        std::string err;
        if (!in.open(i, id, runtime::OpenPolicy{0, 0}, stop, err, static_cast<const vramflow::VideoFormat*>(nullptr)))
            throw std::runtime_error(err);
    }
    ~Reader() { in.release(inst); }
};

}  // namespace

TEST_CASE("gap_policy: --on-loss / --hold-ms → EngineConfig abiem ribų režimais; numatyta invalid, 1000 ms", "[func][gap]") {
    func::Opts o;
    for (const auto t : {func::EngineTiming::Legacy, func::EngineTiming::Isolation}) {
        const EngineConfig d = func::engine_config(o, t, 500'000);
        CHECK(d.on_loss == GapPolicy::Invalid);
        CHECK(d.hold_ns == 1'000'000'000);
    }
    o.on_loss = "freeze";
    o.hold_ms = 400;
    for (const auto t : {func::EngineTiming::Legacy, func::EngineTiming::Isolation}) {
        const EngineConfig c = func::engine_config(o, t, 500'000);
        CHECK(c.on_loss == GapPolicy::Freeze);
        CHECK(c.hold_ns == 400'000'000);
    }
    o.on_loss = "black";
    CHECK(func::engine_config(o, func::EngineTiming::Isolation, 0).on_loss == GapPolicy::Black);
    CHECK(func::gap_policy("invalid") == GapPolicy::Invalid);
    CHECK(func::gap_policy("leave") == GapPolicy::Leave);
}

// V20 (K-T2): --deadline-ms 0 (numatyta) — auto 3·P/4 pagal grain'ų dažnį; aiški reikšmė — tokia pati.
TEST_CASE("timing: --deadline-ms numatyta = 3·P/4 (50 Hz — 15 ms, 25 Hz — 30 ms), aiški — nepakitusi", "[func][timing]") {
    func::Opts o;
    CHECK(o.deadline_ms == 0u);
    o.fmt = vramflow::kFmt1080i50;  // 50 laukų/s
    CHECK(func::deadline_ns(o) == 15'000'000);
    CHECK(func::engine_config(o, func::EngineTiming::Isolation, 500'000).cuts.deadline_ns == 15'000'000);
    CHECK(func::engine_config(o, func::EngineTiming::Legacy, 0).cuts.deadline_ns == 15'000'000);
    std::string why;
    REQUIRE(vramflow::parse_format("1080p25", o.fmt, why));
    CHECK(func::deadline_ns(o) == 30'000'000);
    CHECK(func::engine_config(o, func::EngineTiming::Isolation, 500'000).cuts.deadline_ns == 30'000'000);
    REQUIRE(vramflow::parse_format("1080i59.94", o.fmt, why));
    CHECK(func::deadline_ns(o) == engine::default_cuts({60000, 1001}).deadline_ns);  // 12,51 ms
    o.fmt = vramflow::kFmt1080i50;
    o.deadline_ms = 30;  // WSL2 kampanijos — aiškiai
    CHECK(func::deadline_ns(o) == 30'000'000);
    CHECK(func::engine_config(o, func::EngineTiming::Isolation, 500'000).cuts.deadline_ns == 30'000'000);
}

// Peržiūra 2026-11-13 p. 4: enc numatytoji riba — 3·P/2 (kadro garsas baigiamas ties T + P), garsas laukiamas ties T + P;
// aiški --deadline-ms nepakitusi.
TEST_CASE("timing: enc --deadline-ms numatyta = 3·P/2 (50 Hz — 30 ms, 25p — 60 ms), garsas laukiamas ties T + P", "[func][timing]") {
    func::Opts o;
    o.fmt = vramflow::kFmt1080i50;
    CHECK(func::enc_deadline_ns(o) == 30'000'000);
    CHECK(func::enc_audio_expect_ns(o) == 20'000'000);
    std::string why;
    REQUIRE(vramflow::parse_format("1080p25", o.fmt, why));
    CHECK(func::enc_deadline_ns(o) == 60'000'000);
    CHECK(func::enc_audio_expect_ns(o) == 40'000'000);
    o.deadline_ms = 25;
    CHECK(func::enc_deadline_ns(o) == 25'000'000);
}

TEST_CASE("gap_policy: GapFill — Repeat == to paties pariteto šaltinis, Black == Y 64 / C 512, Invalid — žymė (host, cuda-linear)",
          "[func][gap][mxl]") {
    const bool cuda = GENERATE(false, true);
    if (cuda && !have_gpu()) SKIP("nėra CUDA įrenginio");
    if (cuda) REQUIRE(cudaSetDevice(0) == cudaSuccess);
    Domain d(cuda ? 2 : 1);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 1, 1, k1080i), opts_for(cuda), err, k1080i));
    o.id = func::flow_id(func::kRoleLogo, 1);
    GapFill gf(GapPolicy::Freeze, cuda);
    REQUIRE(gf.prepare(o.raster, err));
    REQUIRE(gf.black(o.raster) != nullptr);
    CHECK(gf.black_bytes() == gfx::grain_bytes(o.raster));
    MxlVideoOutput port(o, gf.copy(), gf.flush(), gf.black(o.raster));
    REQUIRE(commit_pattern(port, 100, cuda));
    REQUIRE(commit_pattern(port, 101, cuda));
    REQUIRE(port.fill_gap(102, GapAction::Repeat, 100));
    REQUIRE(port.fill_gap(103, GapAction::Repeat, 101));
    REQUIRE(port.fill_gap(104, GapAction::Black, kNoIndex));
    REQUIRE(port.fill_gap(105, GapAction::Invalid, kNoIndex));
    REQUIRE(port.flush_gaps());
    Reader r(d.inst, o.id);
    std::uint32_t fl = 0;
    CHECK(read_grain(r.in, 102, &fl) == read_grain(r.in, 100));
    CHECK((fl & MXL_GRAIN_FLAG_INVALID) == 0);
    CHECK(read_grain(r.in, 103) == read_grain(r.in, 101));
    const std::vector<std::uint8_t> blk = read_grain(r.in, 104, &fl);
    CHECK(blk == black_grain(o.raster));
    CHECK((fl & MXL_GRAIN_FLAG_INVALID) == 0);
    // Peržiūra 2026-11-13 p. 2: INVALID grain'as neša juodą turinį (anksčiau — slot'o turinys neliečiamas)
    CHECK(read_grain(r.in, 105, &fl) == black_grain(o.raster));
    CHECK((fl & MXL_GRAIN_FLAG_INVALID) != 0);
    o.release(d.inst);
}

// Peržiūra 2026-11-13 p. 2 (pakeistas lūkestis): invalid politikai juodas grain'as RUOŠIAMAS — tai INVALID grain'o turinys
// (anksčiau: neruošiamas, 0 papildomos atminties). Tik leave — be jo.
TEST_CASE("gap_policy: invalid politikai juodas grain'as paruošiamas (INVALID turinys), leave — ne", "[func][gap]") {
    GapFill gf(GapPolicy::Invalid, false);
    std::string err;
    REQUIRE(gf.prepare(gfx::kHd1080i, err));
    CHECK(gf.black(gfx::kHd1080i) != nullptr);
    CHECK(gf.black_bytes() == gfx::grain_bytes(gfx::kHd1080i));
    GapFill gl(GapPolicy::Leave, false);
    REQUIRE(gl.prepare(gfx::kHd1080i, err));
    CHECK(gl.black(gfx::kHd1080i) == nullptr);
    CHECK(gl.black_bytes() == 0);
}

// Peržiūra 2026-11-13 p. 2: INVALID grain'as neša juodą turinį — žymės nepaisantis skaitytojas nemato pasenusio ring'o kadro.
// Slot'ai užpildomi turiniu (pilnas ratas), tada visas ratas INVALID: kiekvieno grain'o turinys == juodas, juoda rašyta lygiai
// slotCount kartų (pirmas ratas); antras INVALID ratas juodos nebeperrašo (0 rašymų), turinys vis tiek juodas; tikras turinys
// slot'ui žymę nuima — kitas INVALID tame slot'e vėl rašo juodą.
TEST_CASE("gap_policy: INVALID grain'as — žymė IR juodas turinys, juoda kartą slot'ui per gedimą (host, cuda-linear)",
          "[func][gap][mxl]") {
    const bool cuda = GENERATE(false, true);
    if (cuda && !have_gpu()) SKIP("nėra CUDA įrenginio");
    if (cuda) REQUIRE(cudaSetDevice(0) == cudaSuccess);
    Domain d(cuda ? 22 : 21);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 2, 1, k1080i), opts_for(cuda), err, k1080i));
    o.id = func::flow_id(func::kRoleLogo, 2);
    GapFill gf(GapPolicy::Invalid, cuda);
    REQUIRE(gf.prepare(o.raster, err));
    REQUIRE(gf.black(o.raster) != nullptr);
    MxlVideoOutput port(o, gf.copy(), gf.flush(), gf.black(o.raster));
    const std::uint32_t n = o.layout.slotCount;
    REQUIRE(n >= 4);
    Index i = 1000;
    for (std::uint32_t k = 0; k < n; ++k) REQUIRE(commit_pattern(port, i++, cuda));  // visi slot'ai — turinys
    const std::vector<std::uint8_t> blk = black_grain(o.raster);
    Reader r(d.inst, o.id);
    // 1 ratas INVALID — kiekviename slot'e vietoj seno kadro juoda
    for (std::uint32_t k = 0; k < n; ++k) REQUIRE(port.fill_gap(i++, GapAction::Invalid, kNoIndex));
    CHECK(o.black_writes == n);
    for (Index j = i - n + 2; j < i; ++j) {  // ≤ slotCount − 2 atgal — dar skaitomas
        INFO("j = " << j);
        std::uint32_t fl = 0;
        CHECK(read_grain(r.in, j, &fl) == blk);
        CHECK((fl & MXL_GRAIN_FLAG_INVALID) != 0);
    }
    // 2 ratas — juoda jau yra, nerašoma
    for (std::uint32_t k = 0; k < n; ++k) REQUIRE(port.fill_gap(i++, GapAction::Invalid, kNoIndex));
    CHECK(o.black_writes == n);
    std::uint32_t fl = 0;
    CHECK(read_grain(r.in, i - 1, &fl) == blk);
    CHECK((fl & MXL_GRAIN_FLAG_INVALID) != 0);
    // tikras turinys nuima žymę: kitas INVALID tame slot'e (po rato) vėl rašo juodą
    REQUIRE(commit_pattern(port, i++, cuda));
    for (std::uint32_t k = 0; k + 1 < n; ++k) REQUIRE(port.fill_gap(i++, GapAction::Invalid, kNoIndex));
    CHECK(o.black_writes == n);
    REQUIRE(port.fill_gap(i++, GapAction::Invalid, kNoIndex));  // tas pats slot'as, kuriame buvo turinys
    CHECK(o.black_writes == n + 1);
    CHECK(read_grain(r.in, i - 1, &fl) == blk);
    o.release(d.inst);
}


namespace {
// Šaltinis be turinio (dec underrun / ritmas be submit modelis): kiekvienam indeksui no_content → spraga atidarymo metu;
// end_unit užrašo spragos commit'o momentą (TAI).
class NoContent final : public TickFunction {
public:
    std::vector<std::pair<Index, std::uint64_t>> gaps;
    std::atomic<Index> last{kNoIndex};
    NoContent() { gaps.reserve(4096); }
    void begin_tick(Index, TickContext& ctx) override { ctx.no_content(0); }
    bool process(const Batch&, TickContext&) override { return true; }
    void end_unit(const UnitTick& ut, Outcome oc, TickContext&) override {
        if (oc == Outcome::GapNoContent) gaps.emplace_back(ut.index, mxlGetTime());
        last.store(ut.index);
    }
};
}  // namespace

TEST_CASE("gap_policy: žemupio blokuojantis skaitymas spragos indekse grįžta per < 1 ms nuo spragos commit'o", "[func][gap][mxl][realtime]") {
    Domain d(3);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 3, 1, k1080i), "", err, k1080i));
    o.id = func::flow_id(func::kRoleLogo, 3);
    MxlVideoOutput port(o, engine::host_copy(), nullptr, nullptr);
    SystemClock clk;
    StatsHub hub;
    std::atomic<bool> stop{false};
    EngineConfig cfg;
    cfg.cuts = default_cuts({50, 1});
    cfg.warmup_s = 0;
    cfg.duration_s = 1.6;
    cfg.on_loss = GapPolicy::Invalid;
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec u;
    u.name = "src";
    u.out.push_back(&port);
    std::uint32_t id = 0;
    REQUIRE(eng.add_unit(u, id, err));
    NoContent fn;
    // Skaitytojas: kitam indeksui — blokuojantis get (riba 100 ms ≫ P); grįžimo momentas (TAI) kiekvienam indeksui.
    std::vector<std::pair<Index, std::uint64_t>> got;
    got.reserve(4096);
    std::atomic<bool> rstop{false};
    std::thread rd([&] {
        const mxlRational r{50, 1};
        std::unique_ptr<Reader> rr;
        while (!rr && !rstop.load()) {
            try {
                rr = std::make_unique<Reader>(d.inst, o.id);
            } catch (...) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (!rr) return;
        Index i = mxlGetCurrentIndex(&r) + 2;
        while (!rstop.load()) {
            mxlGrainInfo info{};
            const std::uint32_t* p = nullptr;
            const mxlStatus st = rr->in.get(i, 100'000'000, info, p);
            const std::uint64_t t = mxlGetTime();
            if (st == MXL_STATUS_OK && (info.flags & MXL_GRAIN_FLAG_INVALID)) got.emplace_back(i, t);
            if (st == MXL_ERR_OUT_OF_RANGE_TOO_LATE) i = mxlGetCurrentIndex(&r) + 1;
            else ++i;
        }
    });
    const RunResult res = eng.run(fn);
    rstop = true;
    rd.join();
    CHECK(res.reason == EndReason::Duration);
    REQUIRE(fn.gaps.size() >= 60);
    REQUIRE(got.size() >= 50);
    std::vector<std::int64_t> lat;
    std::size_t k = 0;
    for (const auto& [i, t] : got) {
        while (k < fn.gaps.size() && fn.gaps[k].first < i) ++k;
        if (k == fn.gaps.size() || fn.gaps[k].first != i) continue;
        lat.push_back(static_cast<std::int64_t>(t) - static_cast<std::int64_t>(fn.gaps[k].second));
    }
    REQUIRE(lat.size() >= 50);
    std::sort(lat.begin(), lat.end());
    UNSCOPED_INFO("spragos commit → skaitytojo grįžimas: p50 " << lat[lat.size() / 2] / 1000 << " µs, max " << lat.back() / 1000
                                                            << " µs, n = " << lat.size());
    // end_unit kviečiamas PO commit'o, todėl skirtumas gali būti ir neigiamas (skaitytojas pabudo anksčiau nei užrašyta).
    CHECK(lat.back() < 1'000'000);
    CHECK(hub.out(eng.out_id(id, 0)).gap_invalid == fn.gaps.size());
    o.release(d.inst);
}

TEST_CASE("gap_policy: matavimas — vienos spragos kaina (1080i50 grain'as)", "[.][measure]") {
    for (int v = 0; v < 2; ++v) {
        const bool cuda = v == 1;
        if (cuda && !have_gpu()) continue;
        if (cuda) REQUIRE(cudaSetDevice(0) == cudaSuccess);
        Domain d(10 + v);
        VideoOut o;
        std::string err;
        REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 9, 1, k1080i), opts_for(cuda), err, k1080i));
        GapFill gf(GapPolicy::Freeze, cuda);
        REQUIRE(gf.prepare(o.raster, err));
        MxlVideoOutput port(o, gf.copy(), gf.flush(), gf.black(o.raster));
        Index i = 1000;
        // Peržiūra 2026-11-13 p. 2: INVALID — pirmas gedimo ratas (kiekviename slot'e — turinys, rašoma juoda) ir vėlesni
        // ratai (slot'as jau juodas — tik žymė). Prieš kiekvieną pirmo rato matavimą slot'ai vėl užpildomi turiniu.
        const std::uint32_t nslot = o.layout.slotCount;
        for (int lap = 0; lap < 2; ++lap) {
            std::vector<double> us;
            for (int rep = 0; rep < 40; ++rep) {
                if (lap == 0)
                    for (std::uint32_t k = 0; k < nslot; ++k) REQUIRE(commit_pattern(port, i++, cuda));
                for (std::uint32_t k = 0; k < nslot; ++k) {
                    const std::uint64_t w0 = o.black_writes;
                    const auto t0 = std::chrono::steady_clock::now();
                    REQUIRE(port.fill_gap(i++, GapAction::Invalid, kNoIndex));
                    const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    REQUIRE(o.black_writes == w0 + (lap == 0 ? 1 : 0));
                    us.push_back(t);
                }
            }
            std::sort(us.begin(), us.end());
            std::printf("MEASURE gap %s Invalid %s: p50 %.1f us, p99 %.1f us, max %.1f us (n = %zu, grain %zu B)\n",
                        cuda ? "cuda-linear" : "host", lap == 0 ? "pirmas ratas (juoda rašoma)" : "vėlesni ratai (tik žymė)",
                        us[us.size() / 2], us[us.size() * 99 / 100], us.back(), us.size(), gfx::grain_bytes(o.raster));
        }
        REQUIRE(commit_pattern(port, i++, cuda));
        REQUIRE(commit_pattern(port, i++, cuda));
        for (const GapAction a : {GapAction::Repeat, GapAction::Black}) {
            std::vector<double> us;
            for (int n = 0; n < 400; ++n) {
                const auto t0 = std::chrono::steady_clock::now();
                REQUIRE(port.fill_gap(i, a, i - 2));
                us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
                ++i;
            }
            std::sort(us.begin(), us.end());
            std::printf("MEASURE gap %s %s: p50 %.1f us, p99 %.1f us, max %.1f us (grain %zu B)\n", cuda ? "cuda-linear" : "host",
                        to_string(a), us[us.size() / 2], us[us.size() * 99 / 100], us.back(), gfx::grain_bytes(o.raster));
        }
        o.release(d.inst);
    }
}

namespace {
// Įėjimo prievadas, užrašantis kiekvieno blokuojančio laukimo (wait) grąžinimo momentą — kur dingsta laikas tarp ribos ir
// spragos commit'o (futex pabudimas vs apdorojimas iki commit'o).
class RecordingInput final : public InputPort {
public:
    struct Wait {
        Index j;
        std::uint64_t asked_until, returned;
        Probe p;
    };
    explicit RecordingInput(InputPort& in) : in_(in) { waits.reserve(1 << 18); }
    std::vector<Wait> waits;
    bool attach_poll(std::uint64_t now) override { return in_.attach_poll(now); }
    bool mismatch(std::string& why) const override { return in_.mismatch(why); }
    Index head() override { return in_.head(); }
    std::uint64_t last_write_tai() override { return in_.last_write_tai(); }
    Probe probe(Index j, GrainRef& out) override { return in_.probe(j, out); }
    Probe wait(Index j, std::uint64_t timeout_ns, GrainRef& out) override {
        const std::uint64_t t0 = mxlGetTime();
        const Probe p = in_.wait(j, timeout_ns, out);
        if (waits.size() < waits.capacity()) waits.push_back({j, t0 + timeout_ns, mxlGetTime(), p});
        return p;
    }
    bool writer_alive() override { return in_.writer_alive(); }
    bool lapped(Index j, Index now_idx) override { return in_.lapped(j, now_idx); }
    void detach() override { in_.detach(); }
    void retarget(const FlowRef& f) override { in_.retarget(f); }

private:
    InputPort& in_;
};

class GapTimes final : public TickFunction {
public:
    std::vector<std::pair<Index, std::uint64_t>> gaps;  // (indeksas, end_unit momentas — iškart po spragos commit'o)
    GapTimes() { gaps.reserve(1 << 14); }
    bool process(const Batch&, TickContext&) override { return true; }
    void end_unit(const UnitTick& ut, Outcome oc, TickContext&) override {
        if (oc == Outcome::GapLate) gaps.emplace_back(ut.index, mxlGetTime());
    }
};

std::string pct(std::vector<std::int64_t> v) {
    if (v.empty()) return "n = 0";
    std::sort(v.begin(), v.end());
    char b[160];
    std::snprintf(b, sizeof b, "p50 %.3f ms, p99 %.3f ms, max %.3f ms, min %.3f ms (n = %zu)", static_cast<double>(v[v.size() / 2]) / 1e6,
                  static_cast<double>(v[v.size() * 99 / 100]) / 1e6, static_cast<double>(v.back()) / 1e6, static_cast<double>(v.front()) / 1e6,
                  v.size());
    return b;
}
}  // namespace

// Peržiūra 2026-11-13 p. 7a: spragos commit'as ties riba (D = 3·P/4 = 15 ms, numatytosios ribos kaip logo). Rašytojas gyvas:
// lyginiai grain'ai ties T + 1 ms, nelyginiai — ties T + 18 ms (po ribos), todėl kiekvienas nelyginis indeksas — riba praėjo,
// spraga commit'inama (GapLate), o įėjimas niekada neužstringa (miss_run nusinulina). VRAMFLOW_GAP_MEASURE_N (numatyta 2000)
// spragų. Spausdinama: commit − T(i); futex laukimo pabudimas − prašytas laikas (overshoot); commit − pabudimas (apdorojimas).
TEST_CASE("gap_policy: matavimas — spragos commit'o laikas ties riba (2000 spragų)", "[.][measure][gapcommit]") {
    const char* env = std::getenv("VRAMFLOW_GAP_MEASURE_N");
    const std::size_t want = env ? static_cast<std::size_t>(std::strtoul(env, nullptr, 10)) : 2000;
    Domain d(30);
    VideoOut src, dst;
    std::string err;
    REQUIRE(src.create(d.inst, func::video_def(func::kRoleGen, 30, 1, k1080i), "", err, k1080i));
    REQUIRE(dst.create(d.inst, func::video_def(func::kRoleLogo, 30, 1, k1080i), "", err, k1080i));
    std::atomic<bool> wstop{false};
    std::thread writer([&] {
        const mxlRational r{50, 1};
        std::uint64_t i = mxlGetCurrentIndex(&r) + 1;
        while (!wstop.load()) {
            const std::uint64_t at = mxlIndexToTimestamp(&r, i) + ((i & 1) ? 18'000'000ull : 1'000'000ull);
            while (mxlGetTime() < at && !wstop.load()) mxlSleepUntil(at);
            mxlStatus st{};
            if (src.open(i, st)) src.commit();
            ++i;
        }
    });
    runtime::InputSlot slot(func::flow_id(func::kRoleGen, 30), &k1080i, runtime::FormatChange::Exit);
    MxlVideoInput min(slot, d.inst);
    RecordingInput rin(min);
    GapFill gf(GapPolicy::Invalid, false);
    REQUIRE(gf.prepare(dst.raster, err));
    MxlVideoOutput port(dst, gf.copy(), gf.flush(), gf.black(dst.raster));
    SystemClock clk;
    StatsHub hub;
    std::atomic<bool> stop{false};
    EngineConfig cfg;
    cfg.cuts = default_cuts({50, 1});
    cfg.warmup_s = 0;
    cfg.duration_s = static_cast<double>(want) / 25.0 + 3;  // kas antras indeksas — spraga
    cfg.on_loss = GapPolicy::Invalid;
    TickEngine eng(cfg, clk, hub, stop);
    UnitSpec u;
    u.name = "logo-like";
    u.in.push_back(InputSpec{&rin, {}, 1, Need::Required, InMode::Exact});
    u.out.push_back(&port);
    std::uint32_t id = 0;
    REQUIRE(eng.add_unit(u, id, err));
    GapTimes fn;
    const RunResult res = eng.run(fn);
    wstop = true;
    writer.join();
    CHECK(res.reason == EndReason::Duration);
    const mxlRational r{50, 1};
    std::vector<std::int64_t> commit, overshoot, work;
    std::size_t w = 0;
    for (const auto& [i, t] : fn.gaps) {
        const std::int64_t ti = static_cast<std::int64_t>(mxlIndexToTimestamp(&r, i));
        commit.push_back(static_cast<std::int64_t>(t) - ti);
        // paskutinis šio indekso laukimas, grįžęs be grain'o (riba)
        const RecordingInput::Wait* last = nullptr;
        while (w < rin.waits.size() && rin.waits[w].j <= i) {
            if (rin.waits[w].j == i && rin.waits[w].p == Probe::NotYet) last = &rin.waits[w];
            ++w;
        }
        if (last == nullptr) continue;
        overshoot.push_back(static_cast<std::int64_t>(last->returned) - static_cast<std::int64_t>(last->asked_until));
        work.push_back(static_cast<std::int64_t>(t) - static_cast<std::int64_t>(last->returned));
    }
    std::printf("MEASURE gap commit − T(i): %s\n", pct(commit).c_str());
    std::printf("MEASURE futex pabudimas − prašytas laikas: %s\n", pct(overshoot).c_str());
    std::printf("MEASURE commit − pabudimas (apdorojimas): %s\n", pct(work).c_str());
    std::printf("MEASURE juodo turinio rašymų (pirmas ratas): %llu\n", static_cast<unsigned long long>(dst.black_writes));
    CHECK(fn.gaps.size() >= want * 9 / 10);
    slot.release();
    src.release(d.inst);
    dst.release(d.inst);
}
