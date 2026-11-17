// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Priėmimo kontrolė (planas §P4): sprendimo ribos (NVDEC, NVENC sesijos, VRAM), talpos failas (schema 1, UUID, tvarkyklė,
// lower_bound), ingest srauto paklausa ir rezervacijų knyga tarp procesų (flock; fork / kill -9).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/admission.hpp"

using namespace mxlv::runtime;
using Catch::Approx;

namespace {
const std::string kUuid = "GPU-be7b6e9b-27ec-ac11-92d2-d91c03de8d4f";  // tests/unit/data/capacity-<uuid>.json (RTX 4070 Laptop, P8)
const std::string kData = MXLV_TEST_DATA_DIR;
constexpr double k1080 = 1920.0 * 1080 * 25 / 1e6;  // 51,84 Mpix/s

std::string fixture() {
    std::ifstream f(kData + "/capacity-" + kUuid + ".json");
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
std::string replace(std::string s, const std::string& a, const std::string& b) {
    const std::size_t p = s.find(a);
    REQUIRE(p != std::string::npos);
    return s.replace(p, a.size(), b);
}
}  // namespace

TEST_CASE("admission: ribos — 16-tas 1080p25 priimamas, 17-tas atmetamas dėl nvdec", "[admission]") {
    Capacity c;
    c.nvdec_mpix_s = 1000;
    c.headroom = 0.85;
    Demand one;
    one.nvdec_mpix_s = k1080;
    Demand used;
    for (int n = 1; n <= 16; ++n) {
        const Decision d = decide(used, one, c);
        INFO(n);
        REQUIRE(d.admit);
        used = used + one;
    }
    CHECK(used.nvdec_mpix_s == Approx(829.44));
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "nvdec");
    CHECK(d.limit == Approx(850));
    CHECK(d.used == Approx(829.44));
    CHECK(d.requested == Approx(51.84));
    const std::string t = describe(d);
    CHECK(t.find("nvdec") != std::string::npos);
    CHECK(t.find("829,44") != std::string::npos);
    CHECK(t.find("51,84") != std::string::npos);
    CHECK(t.find("850,00") != std::string::npos);
}

TEST_CASE("admission: VRAM riboja pirmiau — 6 GiB, 135 MiB srautui: 38-tas priimamas, 39-tas atmetamas", "[admission]") {
    Capacity c;
    c.nvdec_mpix_s = 1e6;  // NVDEC neriboja
    c.vram_bytes = std::uint64_t{6} << 30;
    Demand one;
    one.nvdec_mpix_s = k1080;
    one.vram_bytes = std::uint64_t{135} << 20;
    Demand used;
    for (int n = 1; n <= 38; ++n) {
        INFO(n);
        REQUIRE(decide(used, one, c).admit);
        used = used + one;
    }
    CHECK(used.vram_bytes == (std::uint64_t{5130} << 20));  // 5130 MB ≤ 5222 MB
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "vram");
    CHECK(d.limit == Approx(0.85 * static_cast<double>(std::uint64_t{6} << 30)));
    CHECK(describe(d).find("MB") != std::string::npos);
}

TEST_CASE("admission: NVENC sesijos — atskiras resursas, 13-ta atmetama, kai Mpix/s dar leistų", "[admission]") {
    Capacity c;
    c.nvenc_mpix_s = 100000;  // pralaidumas neriboja
    c.nvenc_sessions = 12;
    Demand one;
    one.nvenc_mpix_s = k1080;
    one.nvenc_sessions = 1;
    Demand used;
    for (int n = 1; n <= 12; ++n) {
        INFO(n);
        REQUIRE(decide(used, one, c).admit);  // headroom sesijoms netaikomas: 12, ne floor(0,85 · 12) = 10
        used = used + one;
    }
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "nvenc_sessions");
    CHECK(d.limit == Approx(12));
    CHECK(d.used == Approx(12));
    CHECK(d.requested == Approx(1));
}

TEST_CASE("admission: nvenc Mpix/s riboja prieš sesijas; nežinoma talpa (0) netikrinama", "[admission]") {
    Capacity c;
    c.nvenc_mpix_s = 200;  // riba 170: 3 srautai (155,52) telpa, 4-as — ne
    c.nvenc_sessions = 12;
    Demand one;
    one.nvenc_mpix_s = k1080;
    one.nvenc_sessions = 1;
    one.nvdec_mpix_s = 1e9;          // NVDEC talpa nežinoma — netikrinama
    one.vram_bytes = ~std::uint64_t{0} >> 1;  // VRAM talpa nežinoma — netikrinama
    Demand used;
    for (int n = 1; n <= 3; ++n) {
        REQUIRE(decide(used, one, c).admit);
        used = used + one;
    }
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "nvenc");
    CHECK(d.limit == Approx(170));
}

TEST_CASE("capacity: P8 failas (RTX 4070 Laptop) — varikliai, sesijos, funkcijų VRAM, bound", "[admission][capacity]") {
    Capacity c;
    std::string err;
    REQUIRE(load_capacity(kData + "/capacity-" + kUuid + ".json", kUuid, c, err));
    CHECK(err.empty());
    CHECK(c.nvdec_mpix_s == Approx(1140.48));
    CHECK_FALSE(c.nvdec_lower_bound);
    CHECK(c.nvenc_mpix_s == Approx(622.08));
    CHECK(c.nvenc_lower_bound);
    CHECK(c.nvenc_sessions == 12);
    CHECK(c.gpu_uuid == kUuid);
    CHECK(c.driver == "596.49");
    CHECK(c.format == "1080i50");
    CHECK(c.vram_bytes == 0);  // faile nėra — iš įrenginio
    CHECK(c.source.find("capacity-" + kUuid) != std::string::npos);
    REQUIRE(c.functions.count("dec"));
    CHECK(c.functions["dec"].vram_per_flow_bytes == 191029576u);
    CHECK(c.functions["dec"].vram_offset_bytes == 102760448u);
    CHECK(c.functions["dec"].knee == 22);
    REQUIRE(c.functions.count("enc-nvenc-ll"));
    CHECK(c.functions["enc-nvenc-ll"].vram_per_flow_bytes == 132340777u);
}

TEST_CASE("capacity: katalogas → capacity-<uuid>.json; UUID didžiosiomis tinka", "[admission][capacity]") {
    Capacity c;
    std::string err;
    std::string up = kUuid;
    for (std::size_t i = 4; i < up.size(); ++i) up[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(up[i])));
    REQUIRE(load_capacity(kData, kUuid, c, err));
    CHECK(c.nvenc_sessions == 12);
    CHECK(load_capacity(kData + "/capacity-" + kUuid + ".json", up, c, err));
}

TEST_CASE("capacity: kita GPU (UUID nesutampa) → failas nenaudojamas, klaidoje abu UUID", "[admission][capacity]") {
    Capacity c;
    std::string err;
    const std::string other = "GPU-00000000-1111-2222-3333-444444444444";
    CHECK_FALSE(load_capacity(kData + "/capacity-" + kUuid + ".json", other, c, err));
    CHECK(err.find(kUuid) != std::string::npos);
    CHECK(err.find(other) != std::string::npos);
}

TEST_CASE("capacity: nėra failo → „talpa neišmatuota“", "[admission][capacity]") {
    Capacity c;
    std::string err;
    CHECK_FALSE(load_capacity(kData, "GPU-00000000-1111-2222-3333-444444444444", c, err));
    CHECK(err.find("talpa neišmatuota") != std::string::npos);
    CHECK_FALSE(load_capacity("/nonexistent/capacity.json", "", c, err));
    CHECK(err.find("talpa neišmatuota") != std::string::npos);
}

TEST_CASE("capacity: schema, trūkstami laukai, netinkamas JSON — klaida", "[admission][capacity]") {
    const std::string f = fixture();
    Capacity c;
    std::string err;
    REQUIRE(parse_capacity(f, c, err));
    CHECK_FALSE(parse_capacity(replace(f, "\"schema\": 1", "\"schema\": 2"), c, err));
    CHECK(err.find("schema") != std::string::npos);
    CHECK_FALSE(parse_capacity(replace(f, "\"driver\": \"596.49\"", "\"drv\": \"596.49\""), c, err));
    CHECK(err.find("gpu.driver") != std::string::npos);
    CHECK_FALSE(parse_capacity(f.substr(0, f.size() / 2), c, err));
    CHECK_FALSE(parse_capacity("[1,2]", c, err));
    CHECK_FALSE(parse_capacity(replace(f, "\"mpix_s\": 1140.48", "\"mpix_s\": -5"), c, err));
    // vram_per_flow_bytes: null (VRAM neišmatuotas) — leidžiama, 0
    Capacity n;
    REQUIRE(parse_capacity(replace(f, "\"vram_per_flow_bytes\": 191029576", "\"vram_per_flow_bytes\": null"), n, err));
    CHECK(n.functions["dec"].vram_per_flow_bytes == 0);
    // gpu.vram_bytes (neprivalomas) — naudojamas
    Capacity v;
    REQUIRE(parse_capacity(replace(f, "\"driver\": \"596.49\"", "\"driver\": \"596.49\", \"vram_bytes\": 8585216000"), v, err));
    CHECK(v.vram_bytes == 8585216000u);
}

TEST_CASE("capacity: įspėjimai — kita tvarkyklė (abi versijos), lower_bound", "[admission][capacity]") {
    Capacity c;
    std::string err;
    REQUIRE(parse_capacity(fixture(), c, err));
    auto has = [](const std::vector<std::string>& v, const std::string& a, const std::string& b = {}) {
        for (const std::string& s : v)
            if (s.find(a) != std::string::npos && (b.empty() || s.find(b) != std::string::npos)) return true;
        return false;
    };
    const auto same = capacity_warnings(c, "596.49");
    CHECK_FALSE(has(same, "tvarkyklės versija"));
    CHECK(has(same, "nvenc", "lower_bound"));  // NVENC Mpix/s — apatinė riba (sesijų riba nulėmė kelią)
    CHECK_FALSE(has(same, "nvdec"));
    const auto other = capacity_warnings(c, "600.10");
    CHECK(has(other, "tvarkyklės versija", "596.49"));
    CHECK(has(other, "596.49", "600.10"));
    CHECK(has(capacity_warnings(c, ""), "tvarkyklės versija nežinoma"));  // dabartinė versija nežinoma — irgi įspėjimas
}

TEST_CASE("ingest_demand: Mpix/s pagal formatą; VRAM — NV12 paviršiai + v210 ring'as", "[admission]") {
    CHECK(mpix_s(vramflow::kFmt1080i50) == Approx(51.84));
    vramflow::VideoFormat p50{{1920, 1080, false, false}, {50, 1}};
    CHECK(mpix_s(p50) == Approx(103.68));
    const Demand d = ingest_demand(vramflow::kFmt1080i50, 10, 11, true);
    CHECK(d.nvdec_mpix_s == Approx(51.84));
    CHECK(d.nvenc_mpix_s == 0);
    CHECK(d.nvenc_sessions == 0);
    // 10 × 1920 × 1088 × 1,5 + 11 × (40 · 128 B × 540 eilučių lauke)
    CHECK(d.vram_bytes == 10ull * 1920 * 1088 * 3 / 2 + 11ull * 5120 * 540);
    const Demand sw = ingest_demand(p50, 4, 10, false);
    CHECK(sw.nvdec_mpix_s == 0);
    CHECK(sw.vram_bytes == 4ull * 1920 * 1088 * 3 / 2 + 10ull * 5120 * 1080);
}

// --- Ledger (rezervacijų knyga tarp procesų) --------------------------------------------------------------------------
namespace {
using Clock = std::chrono::steady_clock;

// Laikinas domenas /dev/shm/vf-adm-<pid>-<n> (knyga — šalia: <domenas>.vramflow/); pašalinamas.
struct TmpDomain {
    std::string dir;
    explicit TmpDomain(int n) : dir("/dev/shm/vf-adm-" + std::to_string(::getpid()) + "-" + std::to_string(n)) { clean(); }
    ~TmpDomain() { clean(); }
    void clean() const {
        std::error_code ec;
        std::filesystem::remove_all(dir + ".vramflow", ec);
    }
};

Capacity nvdec_cap(double mpix, double headroom = 1.0) {
    Capacity c;
    c.nvdec_mpix_s = mpix;
    c.headroom = headroom;
    return c;
}
Demand nvdec(double mpix) {
    Demand d;
    d.nvdec_mpix_s = mpix;
    return d;
}
// Vaikų procesai nužudomi ir surenkami ir nepavykus REQUIRE (kitaip pause() vaikas laikytų testo išvesties kanalą).
struct Kids {
    std::vector<pid_t> v;
    void kill_all() {
        for (pid_t p : v) {
            ::kill(p, SIGKILL);
            ::waitpid(p, nullptr, 0);
        }
        v.clear();
    }
    ~Kids() { kill_all(); }
};
}  // namespace

TEST_CASE("ledger: du procesai — A rezervuoja 10, B mato 10 ir gauna tik likutį; A release → B mato tik savo", "[admission][ledger]") {
    TmpDomain dom(1);
    Ledger a, b;
    std::string err;
    REQUIRE(Ledger::open(dom.dir, kUuid, a, err));
    REQUIRE(Ledger::open(dom.dir, kUuid, b, err));
    CHECK(a.path() != b.path());
    const Capacity c = nvdec_cap(20);
    REQUIRE(a.reserve("a0", nvdec(10), c).admit);
    CHECK(b.total().nvdec_mpix_s == Approx(10));
    const Decision over = b.reserve("b0", nvdec(15), c);
    CHECK_FALSE(over.admit);
    CHECK(over.engine == "nvdec");
    CHECK(over.used == Approx(10));
    CHECK(over.limit == Approx(20));
    CHECK(b.total().nvdec_mpix_s == Approx(10));  // atmesta — neįrašyta
    REQUIRE(b.reserve("b0", nvdec(10), c).admit);
    CHECK(a.total().nvdec_mpix_s == Approx(20));
    a.release("a0");
    CHECK(b.total().nvdec_mpix_s == Approx(10));
    CHECK(b.own().nvdec_mpix_s == Approx(10));
    b.release("b0");
    CHECK(a.total().nvdec_mpix_s == Approx(0));
}

TEST_CASE("ledger: talpa 8 srautų, pirmas procesas 5, antras prašo 5 → gauna 1 (riba floor(0,85 · 8) = 6)", "[admission][ledger]") {
    TmpDomain dom(2);
    Ledger p1, p2;
    std::string err;
    REQUIRE(Ledger::open(dom.dir, kUuid, p1, err));
    REQUIRE(Ledger::open(dom.dir, kUuid, p2, err));
    const Capacity c = nvdec_cap(8 * k1080, 0.85);
    for (int n = 0; n < 5; ++n) REQUIRE(p1.reserve("p1-" + std::to_string(n), nvdec(k1080), c).admit);
    int got = 0;
    Decision last;
    for (int n = 0; n < 5; ++n) {
        const Decision d = p2.reserve("p2-" + std::to_string(n), nvdec(k1080), c);
        if (d.admit) ++got;
        else last = d;
    }
    CHECK(got == 1);
    CHECK(last.engine == "nvdec");
    CHECK(last.used == Approx(6 * k1080));
    CHECK(last.limit == Approx(0.85 * 8 * k1080));
}

TEST_CASE("ledger: force (warn režimas) įrašo ir netilpusią rezervaciją; kita GPU nesiskaičiuoja", "[admission][ledger]") {
    TmpDomain dom(3);
    Ledger a, other;
    std::string err;
    REQUIRE(Ledger::open(dom.dir, kUuid, a, err));
    REQUIRE(Ledger::open(dom.dir, "GPU-00000000-1111-2222-3333-444444444444", other, err));
    const Capacity c = nvdec_cap(10);
    REQUIRE(other.reserve("x", nvdec(100), c, true).admit == false);  // kita GPU: įrašyta (force), bet čia nesimato
    CHECK(other.total().nvdec_mpix_s == Approx(100));
    CHECK(a.total().nvdec_mpix_s == Approx(0));
    const Decision d = a.reserve("y", nvdec(15), c, true);
    CHECK_FALSE(d.admit);
    CHECK(a.total().nvdec_mpix_s == Approx(15));
}

TEST_CASE("ledger: mirusio proceso (kill -9) rezervacija nebeskaičiuojama per ≤ 100 ms [PF-4]", "[admission][ledger]") {
    TmpDomain dom(4);
    int pfd[2];
    REQUIRE(::pipe(pfd) == 0);
    Kids kids;
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        ::close(pfd[0]);
        Ledger l;
        std::string e;
        const bool ok = Ledger::open(dom.dir, kUuid, l, e) && l.reserve("child", nvdec(10), nvdec_cap(100)).admit;
        const char c = ok ? '1' : '0';
        if (::write(pfd[1], &c, 1) != 1) ::_exit(3);
        for (;;) ::pause();
    }
    kids.v.push_back(child);
    ::close(pfd[1]);
    char c = 0;
    REQUIRE(::read(pfd[0], &c, 1) == 1);
    ::close(pfd[0]);
    REQUIRE(c == '1');
    Ledger me;
    std::string err;
    REQUIRE(Ledger::open(dom.dir, kUuid, me, err));
    CHECK(me.total().nvdec_mpix_s == Approx(10));
    kids.kill_all();  // SIGKILL + waitpid
    const auto t0 = Clock::now();
    const Demand t = me.total();
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    CHECK(t.nvdec_mpix_s == Approx(0));
    CHECK(ms <= 100);
    REQUIRE(me.reserve("new", nvdec(95), nvdec_cap(100)).admit);  // naujas srautas priimamas
    // mirusio failas pašalintas: kataloge liko tik šio Ledger failas
    int adm = 0;
    for (const auto& e : std::filesystem::directory_iterator(dom.dir + ".vramflow"))
        if (e.path().extension() == ".adm") ++adm;
    CHECK(adm == 1);
}

TEST_CASE("ledger: 8 procesai vienu metu, talpa 4 srautams → priimti lygiai 4 (atominė rezervacija)", "[admission][ledger]") {
    TmpDomain dom(5);
    const Capacity c = nvdec_cap(270, 0.85);  // riba 229,5: 4 × 51,84 = 207,36 telpa, 5 — ne
    constexpr int kN = 8;
    int go[2], res[2];
    REQUIRE(::pipe(go) == 0);
    REQUIRE(::pipe(res) == 0);
    Kids kids;
    for (int k = 0; k < kN; ++k) {
        const pid_t p = ::fork();
        REQUIRE(p >= 0);
        if (p == 0) {
            ::close(go[1]);
            ::close(res[0]);
            Ledger l;
            std::string e;
            if (!Ledger::open(dom.dir, kUuid, l, e)) ::_exit(3);
            char x;
            if (::read(go[0], &x, 1) != 0) ::_exit(4);  // laukia, kol tėvas uždarys go (visi startuoja kartu)
            const char r = l.reserve("k", nvdec(k1080), c).admit ? 'A' : 'R';
            if (::write(res[1], &r, 1) != 1) ::_exit(5);
            for (;;) ::pause();
        }
        kids.v.push_back(p);
    }
    ::close(go[0]);
    ::close(res[1]);
    ::close(go[1]);  // startas
    int admitted = 0, refused = 0;
    for (int k = 0; k < kN; ++k) {
        char r = 0;
        REQUIRE(::read(res[0], &r, 1) == 1);
        (r == 'A' ? admitted : refused) += 1;
    }
    ::close(res[0]);
    Ledger me;
    std::string err;
    REQUIRE(Ledger::open(dom.dir, kUuid, me, err));
    CHECK(me.total().nvdec_mpix_s == Approx(4 * k1080));
    kids.kill_all();
    CHECK(admitted == 4);
    CHECK(refused == 4);
    CHECK(me.total().nvdec_mpix_s == Approx(0));
}

TEST_CASE("ledger: close() — failas pašalintas, rezervacijos dingsta", "[admission][ledger]") {
    TmpDomain dom(6);
    Ledger a, b;
    std::string err;
    REQUIRE(Ledger::open(dom.dir, kUuid, a, err));
    REQUIRE(Ledger::open(dom.dir, kUuid, b, err));
    REQUIRE(a.reserve("a", nvdec(5), nvdec_cap(10)).admit);
    const std::string p = a.path();
    CHECK(std::filesystem::exists(p));
    a.close();
    CHECK_FALSE(a.held());
    CHECK_FALSE(std::filesystem::exists(p));
    CHECK(b.total().nvdec_mpix_s == Approx(0));
}
