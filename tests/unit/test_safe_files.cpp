// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Saugus nuomos ir priėmimo knygos failų kūrimas (planas docs/planai/2026-11-17-production-ready.md §1.2, kriterijus S4,
// peržiūros fokusas RF-2). Nuomos katalogas `<domenas>.vramflow/` yra šalia domeno, dažniausiai /dev/shm (1777): kitas
// vietinis vartotojas gali jį sukurti pirmas arba padėti jame simbolinę nuorodą ar FIFO. Testai — per viešą sąsają
// (DomainLease, Ledger); GPU ir MXL nereikia.
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "runtime/admission.hpp"
#include "runtime/domain_lease.hpp"

using mxlv::runtime::Capacity;
using mxlv::runtime::Demand;
using mxlv::runtime::DomainLease;
using mxlv::runtime::Ledger;
using mxlv::runtime::lease_dir;
namespace fs = std::filesystem;

namespace {
constexpr const char* kUuid = "GPU-be7b6e9b-27ec-ac11-92d2-d91c03de8d4f";

// Laikina bazė /dev/shm (tmpfs — flock veikia; tėvinis katalogas 1777, kaip gamyboje) su domenu `dom` nurodytomis teisėmis.
struct Base {
    fs::path root, dom;
    explicit Base(mode_t dom_mode = 0755) {
        char t[] = "/dev/shm/vf-safe-XXXXXX";
        REQUIRE(::mkdtemp(t) != nullptr);
        root = t;
        dom = root / "dom";
        REQUIRE(::mkdir(dom.c_str(), 0700) == 0);
        REQUIRE(::chmod(dom.c_str(), dom_mode) == 0);  // nepriklausomai nuo umask
    }
    ~Base() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    std::string d() const { return dom.string(); }
    fs::path lease() const { return lease_dir(dom.string()); }
};

mode_t mode_of(const fs::path& p) {
    struct stat st {};
    REQUIRE(::lstat(p.c_str(), &st) == 0);
    return st.st_mode & 0777;
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void spit(const fs::path& p, const std::string& s) {
    std::ofstream f(p);
    f << s;
}

// Vykdo fn vaiko procese ir grąžina jo exit kodą; -1 — nebaigė per timeout_ms (pakibo) ir buvo nužudytas.
template <class F>
int child_exit_within(F&& fn, int timeout_ms) {
    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) ::_exit(fn());
    for (int waited = 0; waited < timeout_ms; waited += 5) {
        int st = 0;
        if (::waitpid(pid, &st, WNOHANG) == pid) return WIFEXITED(st) ? WEXITSTATUS(st) : -2;
        ::usleep(5000);
    }
    ::kill(pid, SIGKILL);
    ::waitpid(pid, nullptr, 0);
    return -1;
}

Capacity cap1000() {
    Capacity c;
    c.nvdec_mpix_s = 1000;
    return c;
}

Demand nvdec(double v) {
    Demand d;
    d.nvdec_mpix_s = v;
    return d;
}
}  // namespace

TEST_CASE("safe_files: simbolinė nuoroda vietoje nuomos katalogo atmetama", "[safe_files][lease]") {
    Base b;
    const fs::path elsewhere = b.root / "elsewhere";
    REQUIRE(::mkdir(elsewhere.c_str(), 0755) == 0);
    REQUIRE(::symlink(elsewhere.c_str(), b.lease().c_str()) == 0);
    DomainLease l;
    std::string err;
    CHECK_FALSE(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(err.find("nuomos katalogas") != std::string::npos);
    CHECK(fs::is_empty(elsewhere));  // per nuorodą nieko nesukurta
    Ledger g;
    CHECK_FALSE(Ledger::open(b.d(), kUuid, g, err));
    CHECK(fs::is_empty(elsewhere));
}

TEST_CASE("safe_files: pasenęs laikinas nuomos failas kaip nuoroda — taikinys neliečiamas, nuoma gaunama", "[safe_files][lease]") {
    Base b;
    const fs::path victim = b.root / "victim";
    spit(victim, "A");
    REQUIRE(::mkdir(b.lease().c_str(), 0755) == 0);
    // senas kodas laikiną failą atidarydavo O_CREAT | O_TRUNC nuspėjamu vardu `.<run_id>.<pid>.tmp` — nuoroda būtų sekama
    const fs::path tmp = b.lease() / (".run-A." + std::to_string(::getpid()) + ".tmp");
    REQUIRE(::symlink(victim.c_str(), tmp.c_str()) == 0);
    DomainLease l;
    std::string err;
    CHECK(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(slurp(victim) == "A");
    CHECK(DomainLease::live_owners(b.d()) == std::vector<std::string>{"run-A"});
}

TEST_CASE("safe_files: nuoroda vietoje admission.lock nesekama", "[safe_files][ledger]") {
    Base b;
    const fs::path victim = b.root / "victim";
    spit(victim, "A");
    REQUIRE(::mkdir(b.lease().c_str(), 0755) == 0);
    REQUIRE(::symlink(victim.c_str(), (b.lease() / "admission.lock").c_str()) == 0);
    Ledger g;
    std::string err;
    CHECK_FALSE(Ledger::open(b.d(), kUuid, g, err));
    CHECK(err.find("admission.lock") != std::string::npos);
    CHECK(slurp(victim) == "A");
}

TEST_CASE("safe_files: FIFO vietoje .lease nepakabina live_owners", "[safe_files][lease]") {
    Base b;
    REQUIRE(::mkdir(b.lease().c_str(), 0755) == 0);
    REQUIRE(::mkfifo((b.lease() / "evil.lease").c_str(), 0644) == 0);
    const std::string dom = b.d();
    // vaiko exit: 0 — sąrašas tuščias; 3 — FIFO palaikytas gyvu savininku; -1 — pakibo (atidarymas be O_NONBLOCK)
    CHECK(child_exit_within([&] { return DomainLease::live_owners(dom).empty() ? 0 : 3; }, 1000) == 0);
    CHECK(child_exit_within([&] { return DomainLease::flow_owner(dom, "x").empty() ? 0 : 3; }, 1000) == 0);
}

TEST_CASE("safe_files: FIFO vietoje .adm nepakabina Ledger::total", "[safe_files][ledger]") {
    Base b;
    REQUIRE(::mkdir(b.lease().c_str(), 0755) == 0);
    REQUIRE(::mkfifo((b.lease() / "evil.adm").c_str(), 0644) == 0);
    const std::string dom = b.d();
    const int rc = child_exit_within(
        [&] {
            Ledger g;
            std::string err;
            if (!Ledger::open(dom, kUuid, g, err)) return 4;
            return g.total().nvdec_mpix_s == 0 ? 0 : 3;
        },
        1000);
    CHECK(rc == 0);
}

TEST_CASE("safe_files: teisės paveldimos iš domeno katalogo, nepriklausomai nuo umask", "[safe_files][lease]") {
    const mode_t old = ::umask(022);
    struct Case {
        mode_t dom, dir, file;
    };
    for (const Case c : {Case{0750, 0750, 0640}, Case{0770, 0770, 0660}, Case{0700, 0700, 0600}, Case{0755, 0755, 0644}}) {
        Base b(c.dom);
        DomainLease l;
        Ledger g;
        std::string err;
        REQUIRE(DomainLease::acquire(b.d(), "run-A", l, err));
        REQUIRE(Ledger::open(b.d(), kUuid, g, err));
        INFO("domeno teisės " << std::oct << c.dom);
        CHECK(mode_of(b.lease()) == c.dir);
        CHECK(mode_of(b.lease() / "run-A.lease") == c.file);
        CHECK(mode_of(g.path()) == c.file);
        CHECK(mode_of(b.lease() / "admission.lock") == c.file);
    }
    ::umask(old);
}

TEST_CASE("safe_files: savas nuomos katalogas su platesnėmis nei domeno teisėmis susiaurinamas", "[safe_files][lease]") {
    Base b(0750);
    REQUIRE(::mkdir(b.lease().c_str(), 0700) == 0);
    REQUIRE(::chmod(b.lease().c_str(), 0777) == 0);
    DomainLease l;
    std::string err;
    REQUIRE(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(mode_of(b.lease()) == 0750);
}

TEST_CASE("safe_files: svetimam priklausantis nuomos katalogas atmetamas", "[safe_files][lease]") {
    if (::geteuid() != 0) SKIP("reikia root (chown į kitą vartotoją)");
    Base b(0755);  // domenas — root, nerašomas nei grupei, nei kitiems
    REQUIRE(::mkdir(b.lease().c_str(), 0755) == 0);
    REQUIRE(::chown(b.lease().c_str(), 65534, 65534) == 0);
    DomainLease l;
    std::string err;
    CHECK_FALSE(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(err.find("savininkas") != std::string::npos);
    Ledger g;
    CHECK_FALSE(Ledger::open(b.d(), kUuid, g, err));
    CHECK(DomainLease::live_owners(b.d()).empty());
}

// [RF-2] Tas pats domenas kitu keliu (simbolinė nuoroda, `/./`, galinis `/`) turi turėti vieną nuomos katalogą ir vieną
// knygą — kitaip du procesai nemato vienas kito rezervacijų ir priėmimo kontrolė talpą išdalija du kartus.
TEST_CASE("ledger: tas pats domenas per simbolinę nuorodą — viena knyga [RF-2]", "[safe_files][ledger]") {
    Base b;
    const fs::path link = b.root / "link";
    REQUIRE(::symlink(b.dom.c_str(), link.c_str()) == 0);
    CHECK(lease_dir(link.string()) == lease_dir(b.d()));
    CHECK(lease_dir(b.root.string() + "/./dom/") == lease_dir(b.d()));
    Ledger a, viaLink;
    std::string err;
    REQUIRE(Ledger::open(b.d(), kUuid, a, err));
    REQUIRE(Ledger::open(link.string(), kUuid, viaLink, err));
    const Capacity c = cap1000();
    REQUIRE(a.reserve("f0", nvdec(100), c).admit);
    CHECK(viaLink.total().nvdec_mpix_s == 100);
    DomainLease la, lb;
    REQUIRE(DomainLease::acquire(b.d(), "run-A", la, err));
    CHECK_FALSE(DomainLease::acquire(link.string(), "run-A", lb, err));  // tas pats run_id tame pačiame domene
    CHECK(DomainLease::live_owners(link.string()) == std::vector<std::string>{"run-A"});
}
