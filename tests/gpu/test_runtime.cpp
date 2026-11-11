// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Planas §P6 — tvirtas paleidimas ir sustabdymas ant tikro MXL (host saugykla, GPU nereikia): įėjimo laukimo politika
// (OpenPolicy), InputSlot (vartotojas prieš gamintoją, gamintojas nužudytas ir grįžo, formatas pasikeitė [PF-3]) ir domeno
// nuoma (DomainLease: du savininkai, SIGKILL atlaisvina [PF-4]).
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
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/format.hpp"
#include "func/flows.hpp"
#include "runtime/domain_lease.hpp"
#include "runtime/input_slot.hpp"
#include "runtime/open_policy.hpp"

using namespace mxlv;
using namespace mxlv::func;
using mxlv::runtime::DomainLease;
using mxlv::runtime::FormatChange;
using mxlv::runtime::InputSlot;
using mxlv::runtime::OpenPolicy;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// Laikinas domenas /dev/shm/vf-rt-<pid>-<n> + instancija; nuomos katalogas šalinamas kartu.
struct Domain {
    std::filesystem::path dir;
    mxlInstance inst = nullptr;
    explicit Domain(int n) {
        dir = std::filesystem::path("/dev/shm") / ("vf-rt-" + std::to_string(::getpid()) + "-" + std::to_string(n));
        std::filesystem::remove_all(dir);
        std::filesystem::remove_all(runtime::lease_dir(dir.string()));
        std::filesystem::create_directories(dir);
        inst = mxlCreateInstance(dir.c_str(), "");
        if (!inst) throw std::runtime_error("mxlCreateInstance");
    }
    ~Domain() {
        if (inst) mxlDestroyInstance(inst);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::remove_all(runtime::lease_dir(dir.string()), ec);
    }
};

// Gamintojas (host saugykla): flow_id(kRoleGen, n) nurodytu formatu, grain'ai commit'inami dabartiniu indeksu.
struct Producer {
    VideoOut out;
    mxlInstance inst = nullptr;
    vramflow::VideoFormat fmt{};
    bool start(mxlInstance i, std::uint32_t n, const vramflow::VideoFormat& f) {
        inst = i;
        fmt = f;
        std::string err;
        mxlGarbageCollectFlows(inst);  // kaip gen: nužudyto gamintojo pasenęs flow pašalinamas
        if (!out.create(inst, video_def(kRoleGen, n, 1, f), "", err, f)) {
            UNSCOPED_INFO("create: " << err);
            return false;
        }
        return commit_now();
    }
    bool commit_now() {
        const mxlRational r = mxl_rate(fmt.grain_rate());
        const std::uint64_t idx = mxlGetCurrentIndex(&r);
        if (idx <= out.last_committed && out.last_committed) return true;
        mxlStatus st{};
        if (!out.open(idx, st)) return false;
        return out.commit() == MXL_STATUS_OK;
    }
    void stop() { out.release(inst); }
};

constexpr vramflow::VideoFormat kP25{{1920, 1080, false, false}, {25, 1}};

// Kaip vartotojo tick'as: poll, o prijungtam — get(dabartinis indeksas) (dingimą / perkūrimą slot'as aptinka per get).
// Kartoja, kol būsena != s arba praeina ms; grąžina galutinę būseną.
InputSlot::State poll_while(InputSlot& slot, mxlInstance inst, InputSlot::State s, double ms, Producer* p = nullptr) {
    const auto t0 = Clock::now();
    const mxlRational r = mxl_rate(vramflow::kFmt1080i50.grain_rate());
    while (ms_since(t0) < ms) {
        if (p) p->commit_now();
        if (slot.poll(inst, mxlGetTime()) == InputSlot::State::Attached && s == InputSlot::State::Attached) {
            mxlGrainInfo info{};
            const std::uint32_t* ptr = nullptr;
            (void)slot.get(mxlGetCurrentIndex(&r) + 1, 0, info, ptr);
        }
        if (slot.state() != s) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return slot.state();
}

}  // namespace

TEST_CASE("open_policy: 0 → klaida iškart (< 100 ms)", "[runtime][mxl]") {
    Domain d(1);
    std::atomic<bool> stop{false};
    VideoIn in;
    std::string err;
    const auto t0 = Clock::now();
    CHECK_FALSE(in.open(d.inst, flow_id(kRoleGen, 1), OpenPolicy{0, 0}, stop, err, &vramflow::kFmt1080i50));
    CHECK(ms_since(t0) < 100);
    CHECK(err.find(flow_id(kRoleGen, 1)) != std::string::npos);
}

TEST_CASE("open_policy: baigtinė riba → aiški klaida su riba ir parinktimi", "[runtime][mxl]") {
    Domain d(2);
    std::atomic<bool> stop{false};
    VideoIn in;
    std::string err;
    const auto t0 = Clock::now();
    CHECK_FALSE(in.open(d.inst, flow_id(kRoleGen, 2), OpenPolicy{300, 0}, stop, err, &vramflow::kFmt1080i50));
    const double ms = ms_since(t0);
    CHECK(ms >= 300);
    CHECK(ms < 700);
    INFO(err);
    CHECK(err.find("300 ms") != std::string::npos);
    CHECK(err.find("--input-wait-ms") != std::string::npos);
}

TEST_CASE("open_policy: −1 vis dar laukia po 20 s (> buvusios 15 s ribos), tada prisijungia", "[runtime][mxl][slow]") {
    Domain d(3);
    std::atomic<bool> stop{false}, done{false};
    bool ok = false;
    std::string err;
    VideoIn in;
    std::thread th([&] {
        ok = in.open(d.inst, flow_id(kRoleGen, 3), OpenPolicy{-1, 0}, stop, err, &vramflow::kFmt1080i50);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::seconds(20));
    CHECK_FALSE(done.load());
    Producer p;
    REQUIRE(p.start(d.inst, 3, vramflow::kFmt1080i50));
    const auto t0 = Clock::now();
    while (!done.load() && ms_since(t0) < 2000) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    stop = true;
    th.join();
    CHECK(ok);
    CHECK(ms_since(t0) < 500);
    in.release(d.inst);
    p.stop();
}

TEST_CASE("open_policy: −1 — sustabdymo signalas nutraukia per ≤ 100 ms", "[runtime][mxl]") {
    Domain d(4);
    std::atomic<bool> stop{false}, done{false};
    bool ok = true;
    std::string err;
    VideoIn in;
    std::thread th([&] {
        ok = in.open(d.inst, flow_id(kRoleGen, 4), OpenPolicy{-1, 0}, stop, err, &vramflow::kFmt1080i50);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    CHECK_FALSE(done.load());
    const auto t0 = Clock::now();
    stop = true;
    while (!done.load() && ms_since(t0) < 1000) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(ms_since(t0) <= 100);
    th.join();
    CHECK_FALSE(ok);
}

TEST_CASE("input_slot: vartotojas prieš gamintoją", "[runtime][mxl]") {
    Domain d(5);
    InputSlot slot(flow_id(kRoleGen, 5), &vramflow::kFmt1080i50, FormatChange::Exit);
    CHECK(poll_while(slot, d.inst, InputSlot::State::Waiting, 3000) == InputSlot::State::Waiting);
    Producer p;
    REQUIRE(p.start(d.inst, 5, vramflow::kFmt1080i50));
    const auto t0 = Clock::now();
    CHECK(poll_while(slot, d.inst, InputSlot::State::Waiting, 1000, &p) == InputSlot::State::Attached);
    CHECK(ms_since(t0) <= 200);
    CHECK(slot.reattached() == 0);
    CHECK(slot.format().raster == vramflow::kFmt1080i50.raster);
    slot.release();
    p.stop();
}

TEST_CASE("input_slot: gamintojas nužudytas ir grįžo", "[runtime][mxl]") {
    Domain d(6);
    Producer p;
    REQUIRE(p.start(d.inst, 6, vramflow::kFmt1080i50));
    InputSlot slot(flow_id(kRoleGen, 6), &vramflow::kFmt1080i50, FormatChange::Exit);
    REQUIRE(poll_while(slot, d.inst, InputSlot::State::Waiting, 1000, &p) == InputSlot::State::Attached);
    // writer'is uždaromas (kaip kill -9: flow failai lieka, užraktas atlaisvintas)
    const mxlRational r = mxl_rate(vramflow::kFmt1080i50.grain_rate());
    const std::uint64_t next = p.out.last_committed + 1;
    p.stop();
    const auto t0 = Clock::now();
    mxlStatus last = MXL_STATUS_OK;
    while (slot.state() == InputSlot::State::Attached && ms_since(t0) < 1000) {
        mxlGrainInfo info{};
        const std::uint32_t* ptr = nullptr;
        last = slot.get(std::max(next, mxlGetCurrentIndex(&r)), 0, info, ptr);
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    const double det_ms = ms_since(t0);
    CHECK(slot.state() == InputSlot::State::Detached);
    CHECK(last == MXL_ERR_FLOW_INVALID);
    CHECK(det_ms <= 2 * 20.0 + 20.0);  // ≤ 2 grain'ai (1080i50 — 20 ms) + 1 grain'o atsarga laikui
    // pasenęs flow (be writer'io) neprijungiamas
    CHECK(poll_while(slot, d.inst, InputSlot::State::Detached, 300) == InputSlot::State::Detached);
    // naujas writer'is tuo pačiu ID
    Producer p2;
    REQUIRE(p2.start(d.inst, 6, vramflow::kFmt1080i50));
    CHECK(poll_while(slot, d.inst, InputSlot::State::Detached, 1000, &p2) == InputSlot::State::Attached);
    CHECK(slot.reattached() == 1);
    mxlGrainInfo info{};
    const std::uint32_t* ptr = nullptr;
    CHECK(slot.get(p2.out.last_committed, 0, info, ptr) == MXL_STATUS_OK);
    slot.release();
    p2.stop();
}

TEST_CASE("input_slot: formatas pasikeitė po re-attach [PF-3]", "[runtime][mxl]") {
    SECTION("Exit → Mismatch, tekste abu formatai") {
        Domain d(7);
        Producer p;
        REQUIRE(p.start(d.inst, 7, vramflow::kFmt1080i50));
        InputSlot slot(flow_id(kRoleGen, 7), nullptr, FormatChange::Exit);  // laukiamas — pirmas prijungtas
        REQUIRE(poll_while(slot, d.inst, InputSlot::State::Waiting, 1000, &p) == InputSlot::State::Attached);
        p.stop();
        Producer q;
        REQUIRE(q.start(d.inst, 7, kP25));
        CHECK(poll_while(slot, d.inst, InputSlot::State::Attached, 1000, &q) == InputSlot::State::Detached);  // perkūrimas pastebėtas
        CHECK(poll_while(slot, d.inst, InputSlot::State::Detached, 1000, &q) == InputSlot::State::Mismatch);
        INFO(slot.reason());
        CHECK(slot.reason().find("1080i50") != std::string::npos);
        CHECK(slot.reason().find("1080p25") != std::string::npos);
        // Mismatch galutinė
        CHECK(poll_while(slot, d.inst, InputSlot::State::Mismatch, 300, &q) == InputSlot::State::Mismatch);
        q.stop();
    }
    SECTION("Wait → lieka Detached, kol grįžta 1080i50") {
        Domain d(8);
        Producer p;
        REQUIRE(p.start(d.inst, 8, vramflow::kFmt1080i50));
        InputSlot slot(flow_id(kRoleGen, 8), &vramflow::kFmt1080i50, FormatChange::Wait);
        REQUIRE(poll_while(slot, d.inst, InputSlot::State::Waiting, 1000, &p) == InputSlot::State::Attached);
        p.stop();
        Producer q;
        REQUIRE(q.start(d.inst, 8, kP25));
        const auto st = poll_while(slot, d.inst, InputSlot::State::Attached, 1000, &q);
        CHECK((st == InputSlot::State::Attached || st == InputSlot::State::Detached));
        if (st == InputSlot::State::Attached) {  // dar nepastebėjo dingimo — get turi pastebėti
            mxlGrainInfo info{};
            const std::uint32_t* ptr = nullptr;
            (void)slot.get(q.out.last_committed + 1000, 0, info, ptr);
        }
        CHECK(poll_while(slot, d.inst, InputSlot::State::Detached, 500, &q) == InputSlot::State::Detached);
        CHECK(slot.reason().find("1080p25") != std::string::npos);
        q.stop();
        Producer r;
        REQUIRE(r.start(d.inst, 8, vramflow::kFmt1080i50));
        CHECK(poll_while(slot, d.inst, InputSlot::State::Detached, 1000, &r) == InputSlot::State::Attached);
        CHECK(slot.reattached() == 1);
        slot.release();
        r.stop();
    }
}

TEST_CASE("input_slot: poll kaina, kai Attached, < 5 µs", "[runtime][mxl][perf]") {
    Domain d(9);
    Producer p;
    REQUIRE(p.start(d.inst, 9, vramflow::kFmt1080i50));
    InputSlot slot(flow_id(kRoleGen, 9), &vramflow::kFmt1080i50, FormatChange::Exit);
    REQUIRE(poll_while(slot, d.inst, InputSlot::State::Waiting, 1000, &p) == InputSlot::State::Attached);
    constexpr int kN = 100000;
    const auto t0 = Clock::now();
    int attached = 0;
    for (int k = 0; k < kN; ++k) attached += slot.poll(d.inst, mxlGetTime()) == InputSlot::State::Attached;
    const double us = ms_since(t0) * 1000.0 / kN;
    std::printf("[measure] input_slot poll (Attached): %.3f µs\n", us);
    CHECK(attached == kN);
    CHECK(us < 5.0);
    slot.release();
    p.stop();
}

TEST_CASE("input_slot: wait_attached — baigtinė riba ir neribotas laukimas", "[runtime][mxl]") {
    Domain d(10);
    std::atomic<bool> stop{false};
    InputSlot a(flow_id(kRoleGen, 10), &vramflow::kFmt1080i50, FormatChange::Exit);
    std::vector<InputSlot*> v{&a};
    std::string err;
    CHECK_FALSE(runtime::wait_attached(v, d.inst, OpenPolicy{300, 0}, stop, "test", err));
    INFO(err);
    CHECK(err.find(flow_id(kRoleGen, 10)) != std::string::npos);
    CHECK(err.find("--input-wait-ms") != std::string::npos);
    Producer p;
    std::thread th([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (p.start(d.inst, 10, vramflow::kFmt1080i50))
            for (int k = 0; k < 100 && !stop.load(); ++k) { p.commit_now(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
    });
    err.clear();
    CHECK(runtime::wait_attached(v, d.inst, OpenPolicy{-1, 0}, stop, "test", err));
    stop = true;
    th.join();
    a.release();
    p.stop();
}

TEST_CASE("domain_lease: lease_dir ir run_id", "[runtime][lease]") {
    CHECK(runtime::lease_dir("/dev/shm/x") == "/dev/shm/x.vramflow");
    CHECK(runtime::lease_dir("/dev/shm/x/") == "/dev/shm/x.vramflow");
    CHECK(runtime::valid_run_id(runtime::make_run_id("gen")));
    CHECK(runtime::make_run_id("gen") != runtime::make_run_id("gen"));
    CHECK_FALSE(runtime::valid_run_id(""));
    CHECK_FALSE(runtime::valid_run_id("a/b"));
    CHECK_FALSE(runtime::valid_run_id(std::string(65, 'a')));
}

TEST_CASE("domain_lease: du savininkai", "[runtime][lease]") {
    Domain d(11);
    const std::string dom = d.dir.string();
    DomainLease a, b, dup;
    std::string err;
    REQUIRE(DomainLease::acquire(dom, "run-A", a, err));
    REQUIRE(DomainLease::acquire(dom, "run-B", b, err));
    CHECK(DomainLease::live_owners(dom) == std::vector<std::string>{"run-A", "run-B"});
    // tas pats run_id gyvam savininkui — klaida
    CHECK_FALSE(DomainLease::acquire(dom, "run-A", dup, err));
    CHECK(err.find("run-A") != std::string::npos);
    // nuomos failai šalia domeno, ne jame
    CHECK(std::filesystem::exists(runtime::lease_dir(dom) + "/run-A.lease"));
    for (const auto& e : std::filesystem::directory_iterator(d.dir)) CHECK(e.path().extension() != ".lease");
    REQUIRE(b.add_flow(flow_id(kRoleGen, 42)));
    CHECK(DomainLease::flow_owner(dom, flow_id(kRoleGen, 42)) == "run-B");
    CHECK(DomainLease::flow_owner(dom, flow_id(kRoleGen, 43)).empty());
    a.release();
    CHECK(DomainLease::live_owners(dom) == std::vector<std::string>{"run-B"});
    b = DomainLease{};  // perkėlimas atlaisvina
    CHECK(DomainLease::live_owners(dom).empty());
}

TEST_CASE("domain_lease: SIGKILL atlaisvina nuomą [PF-4]", "[runtime][lease]") {
    Domain d(12);
    const std::string dom = d.dir.string();
    int fds[2];
    REQUIRE(::pipe(fds) == 0);
    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        DomainLease l;
        std::string err;
        const char ok = DomainLease::acquire(dom, "child-run", l, err) ? 'y' : 'n';
        (void)!::write(fds[1], &ok, 1);
        for (;;) ::pause();
    }
    char c = 0;
    const bool got = ::read(fds[0], &c, 1) == 1 && c == 'y';
    const auto owners = DomainLease::live_owners(dom);
    ::kill(pid, SIGKILL);  // visada — net jei nuoma nepavyko (kitaip vaikas liktų)
    ::waitpid(pid, nullptr, 0);
    REQUIRE(got);
    CHECK(owners == std::vector<std::string>{"child-run"});
    const auto t0 = Clock::now();
    while (!DomainLease::live_owners(dom).empty() && ms_since(t0) < 1000) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(ms_since(t0) <= 100);
    CHECK(DomainLease::live_owners(dom).empty());
    CHECK_FALSE(std::filesystem::exists(runtime::lease_dir(dom) + "/child-run.lease"));  // negyvo nuoma pašalinta
    ::close(fds[0]);
    ::close(fds[1]);
}

TEST_CASE("domain path: cuda-linear reader'io payload.sock riba — domeno kelias ≤ 48 simboliai", "[runtime][lease]") {
    // <domenas>/<uuid>.mxl-flow/payload.sock turi tilpti į sockaddr_un::sun_path (107 simboliai): 107 − 59 = 48.
    CHECK(runtime::kMaxCudaDomainPath == 48);
    const std::string base = "/dev/shm/vf-rt-len-" + std::to_string(::getpid()) + "-";
    const std::string ok = base + std::string(48 - base.size(), 'a');
    const std::string bad = base + std::string(49 - base.size(), 'b');
    REQUIRE(ok.size() == 48);
    std::filesystem::create_directories(ok);
    std::filesystem::create_directories(bad);
    CHECK(runtime::cuda_domain_path_error(ok).empty());
    CHECK(runtime::cuda_domain_path_error(ok + "/").empty());  // galinis '/' neskaičiuojamas (MXL kelią kanonizuoja)
    const std::string e = runtime::cuda_domain_path_error(bad);
    INFO(e);
    CHECK(e.find("payload.sock") != std::string::npos);
    CHECK(e.find("49") != std::string::npos);
    CHECK(e.find("48") != std::string::npos);
    CHECK(runtime::cuda_domain_path_error("/dev/shm/vf-rt-nera-tokio-katalogo").empty());  // nėra domeno — klaidą praneš MXL
    std::filesystem::remove_all(ok);
    std::filesystem::remove_all(bad);
}

TEST_CASE("input_slot: wait_attached klaidoje — kodėl įėjimas neprijungtas (kitas formatas, politika Wait)", "[runtime][mxl]") {
    Domain d(13);
    std::atomic<bool> stop{false};
    Producer p;
    REQUIRE(p.start(d.inst, 13, kP25));
    InputSlot a(flow_id(kRoleGen, 13), &vramflow::kFmt1080i50, FormatChange::Wait);
    std::vector<InputSlot*> v{&a};
    std::string err;
    CHECK_FALSE(runtime::wait_attached(v, d.inst, OpenPolicy{400, 0}, stop, "test", err));
    INFO(err);
    CHECK(err.find("1080p25") != std::string::npos);
    CHECK(err.find("1080i50") != std::string::npos);
    p.stop();
}

TEST_CASE("input_slot: vėluojantis, bet gyvas gamintojas neatjungiamas", "[runtime][mxl]") {
    // Gamintojas commit'ina tik kas 60 ms (3 grain'ai, ilgiau nei gyvumo patikros slenkstis): skaitymai grąžina TOO_EARLY,
    // gyvumo patikra vyksta, bet slot'as lieka Attached — atjungiama tik tada, kai writer'io tikrai nebėra.
    Domain d(14);
    Producer p;
    REQUIRE(p.start(d.inst, 14, vramflow::kFmt1080i50));
    InputSlot slot(flow_id(kRoleGen, 14), &vramflow::kFmt1080i50, FormatChange::Exit);
    REQUIRE(poll_while(slot, d.inst, InputSlot::State::Waiting, 1000, &p) == InputSlot::State::Attached);
    const mxlRational r = mxl_rate(vramflow::kFmt1080i50.grain_rate());
    const auto t0 = Clock::now();
    auto last_commit = t0;
    int early = 0;
    while (ms_since(t0) < 600) {
        if (ms_since(last_commit) >= 60) { p.commit_now(); last_commit = Clock::now(); }
        mxlGrainInfo info{};
        const std::uint32_t* ptr = nullptr;
        if (slot.get(mxlGetCurrentIndex(&r) + 1, 0, info, ptr) == MXL_ERR_OUT_OF_RANGE_TOO_EARLY) ++early;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(early > 100);
    CHECK(slot.state() == InputSlot::State::Attached);
    CHECK(slot.reattached() == 0);
    slot.release();
    p.stop();
}
