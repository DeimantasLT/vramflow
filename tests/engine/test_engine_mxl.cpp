// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// V7: variklio MXL prievadai ant tikro MXL, host saugykla (GPU nereikia) — planas §9 „2 etapas“.
// Laiko testai (tikras laikas) tvirtina tik ribas; ctest juos serializuoja (RESOURCE_LOCK realtime).
#include <catch2/catch_test_macros.hpp>

#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/format.hpp"
#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "func/flows.hpp"
#include "gfx/raster.hpp"
#include "mxlio/slots.hpp"
#include "runtime/input_slot.hpp"

using namespace mxlv;
using namespace mxlv::engine;
using mxlv::func::VideoIn;
using mxlv::func::VideoOut;
using mxlv::runtime::FormatChange;
using mxlv::runtime::InputSlot;
namespace gfx = vramflow::gfx;

namespace {

// Trumpas laikinas domenas /dev/shm/vem-<pid>-<n>.
struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n) {
        dir = std::filesystem::path("/dev/shm") / ("vem-" + std::to_string(::getpid()) + "-" + std::to_string(n));
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

constexpr vramflow::VideoFormat k1080i = vramflow::kFmt1080i50;
constexpr vramflow::VideoFormat k720p{{1280, 720, false, false}, {50, 1}};

std::uint8_t* grain_ptr(VideoOut& o, std::uint64_t i) {
    return static_cast<std::uint8_t*>(slot_ptr(o.layout, o.slots[i % o.layout.slotCount]));
}
// Commit'ina turinį: kiekvienas baitas = (i * 7 + k) & 0xff.
bool commit_pattern(MxlVideoOutput& port, VideoOut& o, std::uint64_t i) {
    void* slot = nullptr;
    if (port.open(i, slot) != OpenResult::Ok) return false;
    auto* p = static_cast<std::uint8_t*>(slot);
    const std::size_t n = gfx::grain_bytes(o.raster);
    for (std::size_t k = 0; k < n; ++k) p[k] = static_cast<std::uint8_t>((i * 7 + k) & 0xff);
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

TEST_CASE("engine_mxl: INVALID skaitomas kaip OK + INVALID iškart", "[engine_mxl]") {
    Domain d(1);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 1, 1, k1080i), "", err, k1080i));
    o.id = func::flow_id(func::kRoleLogo, 1);
    MxlVideoOutput port(o, host_copy(), nullptr, nullptr);
    REQUIRE(commit_pattern(port, o, 100));
    REQUIRE(port.fill_gap(101, GapAction::Invalid, kNoIndex));
    REQUIRE(port.flush_gaps());
    Reader r(d.inst, o.id);
    // Prieš commit'inant 102 — skirtumas nuo leave (TOO_EARLY iki kito commit'o, M5)
    mxlGrainInfo info{};
    const std::uint32_t* ptr = nullptr;
    CHECK(r.in.get(101, 0, info, ptr) == MXL_STATUS_OK);
    CHECK(info.index == 101);
    CHECK((info.flags & MXL_GRAIN_FLAG_INVALID) != 0);
    CHECK(info.validSlices == info.totalSlices);
    // Per variklio prievadą: Probe::Invalid
    InputSlot slot(o.id, &k1080i, FormatChange::Exit);
    REQUIRE(slot.poll(d.inst, mxlGetTime()) == InputSlot::State::Attached);
    MxlVideoInput in(slot, d.inst);
    GrainRef g;
    CHECK(in.head() == 101);
    CHECK(in.probe(101, g) == Probe::Invalid);
    CHECK(in.probe(100, g) == Probe::Ready);
    CHECK(g.ptr != nullptr);
    CHECK(in.probe(102, g) == Probe::NotYet);
    REQUIRE(commit_pattern(port, o, 102));
    CHECK(r.in.get(101, 0, info, ptr) == MXL_STATUS_OK);
    CHECK((info.flags & MXL_GRAIN_FLAG_INVALID) != 0);
    CHECK(in.probe(102, g) == Probe::Ready);
    slot.release();
    o.release(d.inst);
}

TEST_CASE("engine_mxl: Repeat — baitas į baitą to paties pariteto, be žymės", "[engine_mxl]") {
    Domain d(2);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 2, 1, k1080i), "", err, k1080i));
    o.id = func::flow_id(func::kRoleLogo, 2);
    REQUIRE(o.raster.interlaced);
    MxlVideoOutput port(o, host_copy(), nullptr, nullptr);
    CHECK(port.interlaced());
    CHECK(port.slot_count() == o.layout.slotCount);
    REQUIRE(commit_pattern(port, o, 100));
    REQUIRE(commit_pattern(port, o, 101));
    REQUIRE(port.fill_gap(102, GapAction::Repeat, 100));
    REQUIRE(port.fill_gap(103, GapAction::Repeat, 101));
    REQUIRE(port.flush_gaps());
    const std::size_t n = gfx::grain_bytes(o.raster);
    CHECK(std::memcmp(grain_ptr(o, 102), grain_ptr(o, 100), n) == 0);
    CHECK(std::memcmp(grain_ptr(o, 103), grain_ptr(o, 101), n) == 0);
    CHECK(std::memcmp(grain_ptr(o, 102), grain_ptr(o, 101), n) != 0);
    Reader r(d.inst, o.id);
    mxlGrainInfo info{};
    const std::uint32_t* ptr = nullptr;
    REQUIRE(r.in.get(102, 0, info, ptr) == MXL_STATUS_OK);
    CHECK((info.flags & MXL_GRAIN_FLAG_INVALID) == 0);
    CHECK(info.validSlices == info.totalSlices);
    o.release(d.inst);
}

TEST_CASE("engine_mxl: Black — Y 64, C 512 visame grain'e; v210a key 0", "[engine_mxl]") {
    for (const bool alpha : {false, true}) {
        INFO("alpha = " << alpha);
        Domain d(alpha ? 4 : 3);
        vramflow::VideoFormat f = k720p;
        f.raster.alpha = alpha;
        VideoOut o;
        std::string err;
        const std::string def = alpha ? func::comp_def(3, f, 1) : func::video_def(func::kRoleLogo, 3, 1, f);
        REQUIRE(o.create(d.inst, def, "", err, f));
        o.id = alpha ? func::flow_id(func::kRoleComp, 3) : func::flow_id(func::kRoleLogo, 3);
        const std::vector<std::uint8_t> black = black_grain(o.raster);
        REQUIRE(black.size() == gfx::grain_bytes(o.raster));
        MxlVideoOutput port(o, host_copy(), nullptr, black.data());
        REQUIRE(commit_pattern(port, o, 50));
        REQUIRE(port.fill_gap(51, GapAction::Black, kNoIndex));
        REQUIRE(port.flush_gaps());
        const std::uint8_t* p = grain_ptr(o, 51);
        // v210 žodžiai: (Cb Y Cr) (Y Cb Y) (Cr Y Cb) (Y Cr Y), 10 bitų laukai
        std::uint64_t bad_fill = 0, bad_key = 0;
        const std::size_t fill = gfx::fill_bytes(o.raster);
        for (std::size_t w = 0; w < fill / 4; ++w) {
            std::uint32_t v;
            std::memcpy(&v, p + 4 * w, 4);
            const std::uint32_t a = v & 0x3ff, b = (v >> 10) & 0x3ff, c = (v >> 20) & 0x3ff;
            const bool chroma_first = (w % 4) % 2 == 0;  // 0, 2: C Y C; 1, 3: Y C Y
            const bool ok = chroma_first ? (a == 512 && b == 64 && c == 512) : (a == 64 && b == 512 && c == 64);
            if (!ok || (v >> 30) != 0) ++bad_fill;
        }
        for (std::size_t k = fill; k < gfx::grain_bytes(o.raster); ++k) bad_key += p[k] != 0;
        CHECK(bad_fill == 0);
        CHECK(bad_key == 0);
        Reader r(d.inst, o.id);
        mxlGrainInfo info{};
        const std::uint32_t* ptr = nullptr;
        REQUIRE(r.in.get(51, 0, info, ptr) == MXL_STATUS_OK);
        CHECK((info.flags & MXL_GRAIN_FLAG_INVALID) == 0);
        o.release(d.inst);
    }
}

TEST_CASE("engine_mxl: head() be sisteminių kvietimų (matavimas)", "[engine_mxl][realtime]") {
    // Sisteminių kvietimų skaitiklio procesui Linux neduoda be ptrace / seccomp; matuojama kaina: head() turi būti
    // pigesnis už vieną sisteminį kvietimą (writer_alive = open + flock + close) bent 10 kartų.
    Domain d(5);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 5, 1, k720p), "", err, k720p));
    o.id = func::flow_id(func::kRoleLogo, 5);
    MxlVideoOutput port(o, host_copy(), nullptr, nullptr);
    REQUIRE(commit_pattern(port, o, 10));
    InputSlot slot(o.id, &k720p, FormatChange::Exit);
    REQUIRE(slot.poll(d.inst, mxlGetTime()) == InputSlot::State::Attached);
    MxlVideoInput in(slot, d.inst);
    using C = std::chrono::steady_clock;
    constexpr int kHead = 100'000, kAlive = 2'000;
    std::uint64_t sum = 0;
    const auto t0 = C::now();
    for (int k = 0; k < kHead; ++k) sum += in.head();
    const double head_ns = std::chrono::duration<double, std::nano>(C::now() - t0).count() / kHead;
    const auto t1 = C::now();
    int alive = 0;
    for (int k = 0; k < kAlive; ++k) alive += in.writer_alive() ? 1 : 0;
    const double alive_ns = std::chrono::duration<double, std::nano>(C::now() - t1).count() / kAlive;
    std::printf("[engine_mxl] head(): %.1f ns/kvietimą; writer_alive(): %.0f ns/kvietimą\n", head_ns, alive_ns);
    CHECK(sum == 10ull * kHead);
    CHECK(alive == kAlive);
    CHECK(head_ns * 10 < alive_ns);
    slot.release();
    o.release(d.inst);
}

TEST_CASE("engine_mxl: blokuojantis laukimas pabunda nuo commit'o", "[engine_mxl][realtime]") {
    Domain d(6);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 6, 1, k1080i), "", err, k1080i));
    o.id = func::flow_id(func::kRoleLogo, 6);
    MxlVideoOutput port(o, host_copy(), nullptr, nullptr);
    REQUIRE(commit_pattern(port, o, 200));
    InputSlot slot(o.id, &k1080i, FormatChange::Exit);
    REQUIRE(slot.poll(d.inst, mxlGetTime()) == InputSlot::State::Attached);
    MxlVideoInput in(slot, d.inst);
    std::thread w([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        commit_pattern(port, o, 201);
    });
    GrainRef g;
    const auto t0 = std::chrono::steady_clock::now();
    const Probe p = in.wait(201, 1'000'000'000, g);  // 1 s riba — pabudimas turi būti nuo commit'o
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    w.join();
    CHECK(p == Probe::Ready);
    CHECK(g.ptr != nullptr);
    CHECK(ms < 20.0);  // < P (50 Hz)
    // Be commit'o: grįžta NotYet po timeout (M1: niekada MXL_ERR_TIMEOUT)
    CHECK(in.wait(202, 3'000'000, g) == Probe::NotYet);
    slot.release();
    o.release(d.inst);
}

TEST_CASE("engine_mxl: matavimai — futex pabudimas, head, gyvumas", "[.][measure]") {
    Domain d(7);
    VideoOut o;
    std::string err;
    REQUIRE(o.create(d.inst, func::video_def(func::kRoleLogo, 7, 1, k720p), "", err, k720p));
    o.id = func::flow_id(func::kRoleLogo, 7);
    MxlVideoOutput port(o, host_copy(), nullptr, nullptr);
    REQUIRE(commit_pattern(port, o, 1));
    InputSlot slot(o.id, &k720p, FormatChange::Exit);
    REQUIRE(slot.poll(d.inst, mxlGetTime()) == InputSlot::State::Attached);
    MxlVideoInput in(slot, d.inst);
    constexpr int kN = 2000;
    std::vector<std::int64_t> lat;
    lat.reserve(kN);
    std::atomic<std::uint64_t> committed_at{0};
    std::thread w([&] {
        for (int k = 0; k < kN; ++k) {
            std::this_thread::sleep_for(std::chrono::microseconds(1500));
            void* s = nullptr;
            port.open(static_cast<std::uint64_t>(k + 2), s);
            committed_at = mxlGetTime();
            port.commit();
        }
    });
    for (int k = 0; k < kN; ++k) {
        GrainRef g;
        if (in.wait(static_cast<std::uint64_t>(k + 2), 100'000'000, g) != Probe::Ready) continue;
        lat.push_back(static_cast<std::int64_t>(mxlGetTime() - committed_at.load()));
    }
    w.join();
    std::sort(lat.begin(), lat.end());
    std::printf("[measure] futex pabudimas: n=%zu p50=%lld ns p99=%lld ns max=%lld ns\n", lat.size(),
                static_cast<long long>(lat[lat.size() / 2]), static_cast<long long>(lat[lat.size() * 99 / 100]),
                static_cast<long long>(lat.back()));
    slot.release();
    o.release(d.inst);
}

namespace {
// Išėjimo prievadas, kuris įrašo, ką variklis su juo darė (ir perduoda tikram MxlVideoOutput).
class Tap final : public OutputPort {
public:
    struct Ev {
        Index i;
        bool gap;
        GapAction a;
    };
    explicit Tap(OutputPort& p) : p_(p) { ev.reserve(4096); }
    std::vector<Ev> ev;
    OpenResult open(Index i, void*& s) override {
        cur_ = i;
        return p_.open(i, s);
    }
    bool commit() override {
        ev.push_back({cur_, false, GapAction::None});
        return p_.commit();
    }
    void cancel() override { p_.cancel(); }
    bool fill_gap(Index i, GapAction a, Index from) override {
        ev.push_back({i, true, a});
        return p_.fill_gap(i, a, from);
    }
    bool flush_gaps() override { return p_.flush_gaps(); }
    bool interlaced() const override { return p_.interlaced(); }
    std::uint32_t slot_count() const override { return p_.slot_count(); }

private:
    OutputPort& p_;
    Index cur_ = kNoIndex;
};

class Nop final : public TickFunction {
public:
    bool process(const Batch&, TickContext&) override { return true; }
};

// Gamintojas: kiekvienam indeksui commit'ina T(i) + 1 ms (mxlSleepUntil), kol stop.
void produce(std::vector<VideoOut*> outs, const std::atomic<bool>& stop) {
    const mxlRational r{50, 1};
    std::uint64_t i = mxlGetCurrentIndex(&r) + 1;
    while (!stop.load()) {
        mxlSleepUntil(mxlIndexToTimestamp(&r, i) + 1'000'000);
        for (VideoOut* o : outs) {
            mxlStatus st{};
            if (o->open(i, st)) o->commit();
        }
        ++i;
    }
}
}  // namespace

TEST_CASE("engine_mxl: 8 flow'ai, vienas rašytojas nužudytas", "[engine_mxl][realtime]") {
    constexpr std::uint32_t kN = 8, kDead = 7;
    Domain d(8);
    const vramflow::VideoFormat f = k720p;
    // Aštuntojo gamintojas — atskiras procesas (fork prieš bet kokias gijas), nužudomas kill -9.
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        mxlInstance ci = mxlCreateInstance(d.dir.c_str(), "");
        VideoOut o;
        std::string e;
        if (!ci || !o.create(ci, func::video_def(func::kRoleGen, kDead, 1, f), "", e, f)) ::_exit(3);
        std::atomic<bool> never{false};
        produce({&o}, never);
        ::_exit(0);
    }
    std::vector<std::unique_ptr<VideoOut>> src(kN - 1);
    std::vector<VideoOut*> srcp;
    std::string err;
    for (std::uint32_t u = 0; u < kN - 1; ++u) {
        src[u] = std::make_unique<VideoOut>();
        REQUIRE(src[u]->create(d.inst, func::video_def(func::kRoleGen, u, 1, f), "", err, f));
        srcp.push_back(src[u].get());
    }
    std::atomic<bool> stop_prod{false};
    std::thread prod(produce, srcp, std::cref(stop_prod));

    std::vector<std::unique_ptr<InputSlot>> slots;
    std::vector<std::unique_ptr<MxlVideoInput>> ins;
    std::vector<std::unique_ptr<VideoOut>> outs;
    std::vector<std::unique_ptr<MxlVideoOutput>> ports;
    std::vector<std::unique_ptr<Tap>> taps;
    for (std::uint32_t u = 0; u < kN; ++u) {
        slots.push_back(std::make_unique<InputSlot>(func::flow_id(func::kRoleGen, u), &f, FormatChange::Exit));
        ins.push_back(std::make_unique<MxlVideoInput>(*slots[u], d.inst));
        outs.push_back(std::make_unique<VideoOut>());
        REQUIRE(outs[u]->create(d.inst, func::video_def(func::kRoleLogo, u, 1, f), "", err, f));
        ports.push_back(std::make_unique<MxlVideoOutput>(*outs[u], host_copy(), nullptr, nullptr));
        taps.push_back(std::make_unique<Tap>(*ports[u]));
    }
    SystemClock clk;
    StatsHub hub;
    std::atomic<bool> stop{false};
    EngineConfig cfg;  // ne legacy: Wr = P/4, D = 3P/4, M = 3, invalid, immediate
    cfg.cuts = default_cuts({50, 1});
    cfg.start = StartMode::WaitAll;
    cfg.warmup_s = 0;
    cfg.duration_s = 2.5;
    TickEngine eng(cfg, clk, hub, stop);
    for (std::uint32_t u = 0; u < kN; ++u) {
        UnitSpec s;
        s.name = "u" + std::to_string(u);
        s.in.push_back(InputSpec{ins[u].get(), {}, 1, Need::Required, InMode::Exact});
        s.out.push_back(taps[u].get());
        std::uint32_t id = 0;
        REQUIRE(eng.add_unit(s, id, err));
    }
    std::atomic<std::uint64_t> death_tai{0};
    std::thread killer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        ::kill(child, SIGKILL);
        death_tai = mxlGetTime();
    });
    Nop fn;
    const RunResult res = eng.run(fn);
    killer.join();
    int status = 0;
    ::waitpid(child, &status, 0);
    stop_prod = true;
    prod.join();
    CHECK(res.reason == EndReason::Duration);
    const mxlRational r{50, 1};
    const Index death_idx = mxlTimestampToIndex(&r, death_tai.load());
    for (std::uint32_t u = 0; u < kN; ++u) {
        INFO("u = " << u);
        const auto& ev = taps[u]->ev;
        REQUIRE(ev.size() > 50);
        std::uint64_t holes = 0, commits = 0, invalid_after = 0, after = 0;
        for (std::size_t n = 1; n < ev.size(); ++n) holes += ev[n].i != ev[n - 1].i + 1;
        for (const auto& e : ev) {
            commits += !e.gap;
            if (e.i > death_idx + 1) {
                ++after;
                invalid_after += e.gap && e.a == GapAction::Invalid;
            }
        }
        const OutCounters& oc = hub.out(eng.out_id(u, 0));
        if (u != kDead) {
            CHECK(holes == 0);  // indeksai iš eilės
            CHECK(oc.late_skips == 0);
            UNSCOPED_INFO("u" << u << ": commit'ų " << commits << " iš " << ev.size());
        } else {
            CHECK(holes == 0);
            CHECK(after > 30);
            CHECK(invalid_after == after);  // kiekvienas indeksas po mirties — INVALID
        }
    }
    CHECK(eng.in_state(kDead, 0) == InState::Detached);
    for (auto& s : slots) s->release();
    for (auto& o : outs) o->release(d.inst);
    for (auto& o : src) o->release(d.inst);
}
