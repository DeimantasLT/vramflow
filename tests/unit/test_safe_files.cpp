// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Saugus nuomos ir priėmimo knygos failų kūrimas (planas docs/planai/2026-11-17-production-ready.md §1.2, kriterijus S4,
// peržiūros fokusas RF-2). Nuomos katalogas `<domenas>.vramflow/` yra šalia domeno, dažniausiai /dev/shm (1777): kitas
// vietinis vartotojas gali jį sukurti pirmas arba padėti jame simbolinę nuorodą ar FIFO. Testai — per viešą sąsają
// (DomainLease, Ledger); GPU ir MXL nereikia. Atvejai su kitu vartotoju vykdomi tik kaip root (kitaip SKIP).
#include <fcntl.h>
#include <grp.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
// Išgalvoti vartotojai ir grupės (pagal numerį; sistemoje jų nėra): kiti vartotojai A ir B, bendra grupė, pirminės grupės
constexpr uid_t kUserA = 54322, kUserB = 54324;
constexpr gid_t kShared = 54321, kPrimaryA = 54323, kPrimaryB = 54325;

// Laikina bazė /dev/shm (tmpfs — flock veikia) su domenu `dom` nurodytomis teisėmis. Bazė — 1777, kaip /dev/shm: šalia
// domeno kurti gali visi.
struct Base {
    fs::path root, dom;
    explicit Base(mode_t dom_mode = 0755) {
        char t[] = "/dev/shm/vf-safe-XXXXXX";
        REQUIRE(::mkdtemp(t) != nullptr);
        root = t;
        REQUIRE(::chmod(root.c_str(), 01777) == 0);
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

// Vaiko procese: tapti kitu vartotoju (pirminė grupė primary, papildoma — bendra arba jokios). false — nepavyko.
bool become(uid_t uid, gid_t primary, bool in_shared) {
    const gid_t groups[] = {kShared};
    return ::setgroups(in_shared ? 1 : 0, groups) == 0 && ::setgid(primary) == 0 && ::setuid(uid) == 0;
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
    REQUIRE(::mkdir(b.lease().c_str(), 0700) == 0);
    // senas kodas laikiną failą atidarydavo O_CREAT | O_TRUNC nuspėjamu vardu `.<run_id>.<pid>.tmp` — nuoroda būtų sekama
    const fs::path tmp = b.lease() / (".run-A." + std::to_string(::getpid()) + ".tmp");
    REQUIRE(::symlink(victim.c_str(), tmp.c_str()) == 0);
    DomainLease l;
    std::string err;
    CHECK(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(slurp(victim) == "A");
    CHECK(DomainLease::live_owners(b.d()) == std::vector<std::string>{"run-A"});
}

TEST_CASE("safe_files: nuoroda vietoje užrakto failo nesekama", "[safe_files][ledger]") {
    Base b;
    const fs::path victim = b.root / "victim";
    spit(victim, "A");
    REQUIRE(::mkdir(b.lease().c_str(), 0700) == 0);
    std::string err;
    SECTION("admission.lock") {
        REQUIRE(::symlink(victim.c_str(), (b.lease() / "admission.lock").c_str()) == 0);
        Ledger g;
        CHECK_FALSE(Ledger::open(b.d(), kUuid, g, err));
        CHECK(err.find("admission.lock") != std::string::npos);
    }
    SECTION("lease.lock") {
        REQUIRE(::symlink(victim.c_str(), (b.lease() / "lease.lock").c_str()) == 0);
        DomainLease l;
        CHECK_FALSE(DomainLease::acquire(b.d(), "run-A", l, err));
        CHECK(err.find("lease.lock") != std::string::npos);
    }
}

TEST_CASE("safe_files: FIFO vietoje .lease nepakabina live_owners", "[safe_files][lease]") {
    Base b;
    REQUIRE(::mkdir(b.lease().c_str(), 0700) == 0);
    REQUIRE(::mkfifo((b.lease() / "evil.lease").c_str(), 0644) == 0);
    const std::string dom = b.d();
    // vaiko exit: 0 — sąrašas tuščias; 3 — FIFO palaikytas gyvu savininku; -1 — pakibo (atidarymas be O_NONBLOCK)
    CHECK(child_exit_within([&] { return DomainLease::live_owners(dom).empty() ? 0 : 3; }, 3000) == 0);
    CHECK(child_exit_within([&] { return DomainLease::flow_owner(dom, "x").empty() ? 0 : 3; }, 3000) == 0);
}

TEST_CASE("safe_files: FIFO vietoje .adm nepakabina Ledger::total", "[safe_files][ledger]") {
    Base b;
    REQUIRE(::mkdir(b.lease().c_str(), 0700) == 0);
    REQUIRE(::mkfifo((b.lease() / "evil.adm").c_str(), 0644) == 0);
    const std::string dom = b.d();
    const int rc = child_exit_within(
        [&] {
            Ledger g;
            std::string err;
            if (!Ledger::open(dom, kUuid, g, err)) return 4;
            return g.total().nvdec_mpix_s == 0 ? 0 : 3;
        },
        3000);
    CHECK(rc == 0);
}

// Prieiga prie nuomos katalogo = rašymo teisė į domeną: kas negali rašyti į domeną, tas negali atidaryti (taigi ir
// užrakinti) nė vieno nuomos ar knygos failo. Nuo umask nepriklauso.
TEST_CASE("safe_files: teisės — tik tiems, kam domenas rašomas; nepriklausomai nuo umask", "[safe_files][lease]") {
    const mode_t old = ::umask(022);
    struct Case {
        mode_t dom, dir, file;
    };
    for (const Case c : {Case{0755, 0700, 0600}, Case{0750, 0700, 0600}, Case{0700, 0700, 0600}, Case{0770, 0770, 0660},
                         Case{0775, 0770, 0660}, Case{0777, 0777, 0666}}) {
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
        CHECK(mode_of(b.lease() / "lease.lock") == c.file);
    }
    ::umask(old);
}

TEST_CASE("safe_files: savas nuomos katalogas su platesnėmis teisėmis susiaurinamas", "[safe_files][lease]") {
    Base b(0750);
    REQUIRE(::mkdir(b.lease().c_str(), 0700) == 0);
    REQUIRE(::chmod(b.lease().c_str(), 0777) == 0);
    DomainLease l;
    std::string err;
    REQUIRE(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(mode_of(b.lease()) == 0700);
}

// Kol domeno katalogo nėra (pvz. jis perkuriamas), apie teises spręsti negalima: esamas katalogas nekeičiamas.
TEST_CASE("safe_files: dingus domenui esamas nuomos katalogas nesiaurinamas", "[safe_files][lease]") {
    Base b(0770);
    std::string err;
    {
        DomainLease l;
        REQUIRE(DomainLease::acquire(b.d(), "run-A", l, err));
    }
    REQUIRE(mode_of(b.lease()) == 0770);
    REQUIRE(::rmdir(b.dom.c_str()) == 0);
    CHECK(DomainLease::live_owners(b.d()).empty());
    CHECK(mode_of(b.lease()) == 0770);
    DomainLease l;
    CHECK(DomainLease::acquire(b.d(), "run-B", l, err));  // savas katalogas — priimamas ir be domeno
    CHECK(mode_of(b.lease()) == 0770);
}

TEST_CASE("safe_files: svetimam priklausantis nuomos katalogas atmetamas", "[safe_files][lease]") {
    if (::geteuid() != 0) SKIP("reikia root (chown į kitą vartotoją)");
    Base b(0755);  // domenas — root, nerašomas nei grupei, nei kitiems
    std::string err;
    // Kataloge yra GYVA nuoma — bet katalogas (po chown) svetimas, todėl ja nepasitikima
    DomainLease live;
    REQUIRE(DomainLease::acquire(b.d(), "run-live", live, err));
    REQUIRE(DomainLease::live_owners(b.d()) == std::vector<std::string>{"run-live"});
    REQUIRE(::chown(b.lease().c_str(), kUserA, kPrimaryA) == 0);
    DomainLease l;
    CHECK_FALSE(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(err.find("savininkas") != std::string::npos);
    Ledger g;
    CHECK_FALSE(Ledger::open(b.d(), kUuid, g, err));
    CHECK(DomainLease::live_owners(b.d()).empty());
}

TEST_CASE("safe_files: domeno savininko katalogas su platesnėmis teisėmis, nei priklauso, atmetamas", "[safe_files][lease]") {
    if (::geteuid() != 0) SKIP("reikia root (chown į kitą vartotoją)");
    Base b(0750);
    REQUIRE(::chown(b.dom.c_str(), kUserA, kPrimaryA) == 0);  // domenas — vartotojo A, mes (root) — kitas vartotojas
    REQUIRE(::mkdir(b.lease().c_str(), 0700) == 0);
    REQUIRE(::chown(b.lease().c_str(), kUserA, kPrimaryA) == 0);
    REQUIRE(::chmod(b.lease().c_str(), 0777) == 0);
    DomainLease l;
    std::string err;
    CHECK_FALSE(DomainLease::acquire(b.d(), "run-A", l, err));
    CHECK(err.find("teisės") != std::string::npos);
}

// flock užtenka skaitymo aprašo: jei pašalietis galėtų atidaryti užrakto ar nuomos failą, jis galėtų pakabinti visus
// domeno procesus (LOCK_EX ant admission.lock) arba laikyti mirusio proceso nuomą „gyvą“.
TEST_CASE("safe_files: pašalietis negali atidaryti (taigi ir užrakinti) nuomos katalogo failų", "[safe_files][lease]") {
    if (::geteuid() != 0) SKIP("reikia root (kito vartotojo imitacijai)");
    Base b(0755);  // įprastas domenas: skaitomas visiems, rašomas tik savininkui
    DomainLease l;
    Ledger g;
    std::string err;
    REQUIRE(DomainLease::acquire(b.d(), "run-A", l, err));
    REQUIRE(Ledger::open(b.d(), kUuid, g, err));
    const std::string dir = b.lease().string();
    const int rc = child_exit_within(
        [&] {
            if (!become(kUserA, kPrimaryA, false)) return 9;
            for (const char* f : {"admission.lock", "lease.lock", "run-A.lease"}) {
                const int fd = ::open((dir + "/" + f).c_str(), O_RDONLY);
                if (fd >= 0) return 3;  // atidarė — galėtų flock
            }
            return 0;
        },
        3000);
    CHECK(rc == 0);
}

// Kitas procesas laiko knygos užraktą be galo (pakibęs ar piktybiškas grupės narys): paleidimas negali pakibti —
// rezervacija atmetama per ribotą laiką.
TEST_CASE("safe_files: užimtas knygos užraktas nepakabina — rezervacija atmetama per ribotą laiką", "[safe_files][ledger]") {
    Base b;
    const std::string dom = b.d();
    std::string err;
    Ledger g;
    REQUIRE(Ledger::open(dom, kUuid, g, err));
    int rd[2];
    REQUIRE(::pipe(rd) == 0);
    const fs::path lockf = b.lease() / "admission.lock";
    const pid_t holder = ::fork();
    REQUIRE(holder >= 0);
    if (holder == 0) {
        const int fd = ::open(lockf.c_str(), O_RDONLY);
        const char ch = fd >= 0 && ::flock(fd, LOCK_EX) == 0 ? 'y' : 'n';
        (void)!::write(rd[1], &ch, 1);
        for (;;) ::pause();
    }
    char ch = 0;
    const bool held = ::read(rd[0], &ch, 1) == 1 && ch == 'y';
    const Capacity c = cap1000();
    // vaiko exit: 0 — rezervacija atmesta su „ledger“; 3 — priimta be užrakto; -1 — pakibo
    const int rc = child_exit_within(
        [&] {
            Ledger mine;
            std::string e;
            if (Ledger::open(dom, kUuid, mine, e)) return 4;  // atidaryti be užrakto negalima — klaida, ne pakibimas
            return 0;
        },
        8000);
    const auto dec = g.reserve("f0", nvdec(10), c);
    ::kill(holder, SIGKILL);
    ::waitpid(holder, nullptr, 0);
    ::close(rd[0]);
    ::close(rd[1]);
    REQUIRE(held);
    CHECK(rc == 0);
    CHECK_FALSE(dec.admit);
    CHECK(dec.engine == "ledger");
    CHECK(g.reserve("f0", nvdec(10), c).admit);  // užraktą atlaisvinus — vėl veikia
}

// Grupei bendras domenas: skirtingi vartotojai, viena grupė (pvz. konteineriai su bendru fsGroup). Nuomos katalogą gali
// sukurti bet kuris iš jų; kitas turi jį priimti, gauti savo nuomą ir matyti pirmojo rezervaciją. Vartotojų pirminės grupės —
// ne domeno (domeno grupė tik papildoma): be fchown į domeno grupę katalogas liktų svetimos grupės.
TEST_CASE("safe_files: grupei bendras domenas — kiti tos grupės vartotojai gauna nuomą ir mato knygą", "[safe_files][lease]") {
    if (::geteuid() != 0) SKIP("reikia root (kitų vartotojų ir grupės imitacijai)");
    const Capacity c = cap1000();
    // Vaikas kito vartotojo teisėmis: nuoma + knyga; exit 0 — pavyko, knygoje matė `expect` Mpix/s. hold_ms > 0 — laiko
    // nuomą ir rezervaciją tiek ms (kad kitas vaikas jas matytų); ready_fd — vamzdis, į kurį parašoma, kai viskas gauta.
    auto other = [&](uid_t uid, gid_t primary, const std::string& dom, const char* run, double expect, double reserve, int hold_ms,
                     int ready_fd) {
        if (!become(uid, primary, true)) return 9;
        DomainLease l;
        Ledger g;
        std::string err;
        if (!DomainLease::acquire(dom, run, l, err)) return 3;
        if (!Ledger::open(dom, kUuid, g, err)) return 4;
        if (g.total().nvdec_mpix_s != expect) return 5;
        if (reserve > 0 && !g.reserve("f", nvdec(reserve), c).admit) return 6;
        if (ready_fd >= 0) (void)!::write(ready_fd, "y", 1);
        if (hold_ms > 0) ::usleep(static_cast<useconds_t>(hold_ms) * 1000);
        return 0;
    };
    SECTION("katalogą pirmas sukuria root, kitas vartotojas prisijungia") {
        Base b(0770);
        REQUIRE(::chown(b.dom.c_str(), 0, kShared) == 0);
        DomainLease l;
        Ledger g;
        std::string err;
        REQUIRE(DomainLease::acquire(b.d(), "run-root", l, err));
        REQUIRE(Ledger::open(b.d(), kUuid, g, err));
        REQUIRE(g.reserve("f0", nvdec(100), c).admit);
        struct stat st {};
        REQUIRE(::stat(b.lease().c_str(), &st) == 0);
        CHECK(st.st_gid == kShared);         // domeno grupė
        CHECK((st.st_mode & S_ISGID) != 0);  // failai paveldi grupę
        const std::string dom = b.d();
        CHECK(child_exit_within([&] { return other(kUserA, kPrimaryA, dom, "run-other", 100, 0, 0, -1); }, 3000) == 0);
    }
    SECTION("katalogą pirmas sukuria kitas vartotojas, root prisijungia") {
        Base b(0770);
        REQUIRE(::chown(b.dom.c_str(), 0, kShared) == 0);
        const std::string dom = b.d();
        REQUIRE(child_exit_within([&] { return other(kUserA, kPrimaryA, dom, "run-other", 0, 0, 0, -1); }, 3000) == 0);
        struct stat st {};
        REQUIRE(::stat(b.lease().c_str(), &st) == 0);
        CHECK(st.st_uid == kUserA);
        CHECK(st.st_gid == kShared);
        DomainLease l;
        std::string err;
        CHECK(DomainLease::acquire(b.d(), "run-root", l, err));
        INFO(err);
        CHECK(DomainLease::live_owners(b.d()) == std::vector<std::string>{"run-root"});
    }
    SECTION("du neprivilegijuoti vartotojai, nė vienas ne domeno savininkas") {
        Base b(0770);
        REQUIRE(::chown(b.dom.c_str(), 0, kShared) == 0);
        const std::string dom = b.d();
        int ready[2];
        REQUIRE(::pipe(ready) == 0);
        // A sukuria katalogą, rezervuoja 100 ir laiko 1,5 s; B (kitas uid, ta pati grupė) prisijungia ir turi matyti 100
        const pid_t a = ::fork();
        REQUIRE(a >= 0);
        if (a == 0) ::_exit(other(kUserA, kPrimaryA, dom, "run-a", 0, 100, 1500, ready[1]));
        char ch = 0;
        const bool a_ready = ::read(ready[0], &ch, 1) == 1;
        const int rb = a_ready ? child_exit_within([&] { return other(kUserB, kPrimaryB, dom, "run-b", 100, 50, 0, -1); }, 3000) : -3;
        int st = 0;
        ::waitpid(a, &st, 0);
        ::close(ready[0]);
        ::close(ready[1]);
        CHECK(a_ready);
        CHECK(rb == 0);
        CHECK((WIFEXITED(st) && WEXITSTATUS(st) == 0));
    }
}

// Tas pats run_id vienu metu: nuomą gauna lygiai vienas, ir ji lieka matoma, kol jis gyvas — nors kataloge yra pasenęs to
// paties vardo failas, o kiti procesai tuo metu skaito savininkų sąrašą (ir šalina negyvas nuomas).
TEST_CASE("safe_files: tas pats run_id lygiagrečiai — nuomą gauna lygiai vienas ir ji lieka matoma", "[safe_files][lease]") {
    Base b;
    const std::string dom = b.d();
    // Pasenęs failas: vaikas gauna nuomą ir nužudomas (kill -9) — failas lieka, užraktas atlaisvintas
    {
        int rd[2];
        REQUIRE(::pipe(rd) == 0);
        const pid_t dead = ::fork();
        REQUIRE(dead >= 0);
        if (dead == 0) {
            DomainLease l;
            std::string e;
            const char ok = DomainLease::acquire(dom, "run-X", l, e) ? 'y' : 'n';
            (void)!::write(rd[1], &ok, 1);
            for (;;) ::pause();
        }
        char ch = 0;
        REQUIRE(::read(rd[0], &ch, 1) == 1);
        ::kill(dead, SIGKILL);
        ::waitpid(dead, nullptr, 0);
        ::close(rd[0]);
        ::close(rd[1]);
        REQUIRE(ch == 'y');
        REQUIRE(fs::exists(b.lease() / "run-X.lease"));
    }
    constexpr int kAcquirers = 8, kReaders = 4;
    int go[2], res[2];
    REQUIRE(::pipe(go) == 0);
    REQUIRE(::pipe(res) == 0);
    std::vector<pid_t> kids;
    for (int i = 0; i < kAcquirers + kReaders; ++i) {
        const pid_t p = ::fork();
        REQUIRE(p >= 0);
        if (p == 0) {
            ::close(go[1]);
            char ch = 0;
            (void)!::read(go[0], &ch, 1);  // visi startuoja kartu (EOF, kai tėvas uždaro rašymo galą)
            if (i < kAcquirers) {
                DomainLease l;
                std::string e;
                const bool got = DomainLease::acquire(dom, "run-X", l, e);
                const char c = got ? 'W' : 'L';
                (void)!::write(res[1], &c, 1);
                if (got) ::usleep(400 * 1000);  // laimėtojas laiko nuomą
            } else {
                // skaitytojas: sukasi 250 ms — trumpiau, nei laimėtojas laiko nuomą (400 ms); kartą pamatęs gyvą nuomą,
                // turi ją matyti iki galo
                bool seen = false, lost = false;
                const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
                while (std::chrono::steady_clock::now() < until) {
                    const bool has = !DomainLease::live_owners(dom).empty();
                    if (seen && !has) lost = true;
                    seen = seen || has;
                    ::usleep(100);
                }
                const char c = lost ? 'X' : 'r';
                (void)!::write(res[1], &c, 1);
            }
            ::_exit(0);
        }
        kids.push_back(p);
    }
    ::close(go[0]);
    ::close(go[1]);  // startas
    ::usleep(150 * 1000);
    const auto owners_mid = DomainLease::live_owners(dom);  // laimėtojas dar laiko nuomą
    for (const pid_t p : kids) ::waitpid(p, nullptr, 0);
    ::close(res[1]);
    std::string out;
    char ch = 0;
    while (::read(res[0], &ch, 1) == 1) out.push_back(ch);
    ::close(res[0]);
    INFO("rezultatai: " << out);
    CHECK(std::count(out.begin(), out.end(), 'W') == 1);
    CHECK(std::count(out.begin(), out.end(), 'L') == kAcquirers - 1);
    CHECK(std::count(out.begin(), out.end(), 'X') == 0);  // nė vienas skaitytojas nematė, kad gyva nuoma dingtų
    CHECK(owners_mid == std::vector<std::string>{"run-X"});
    CHECK(DomainLease::live_owners(dom).empty());  // visi baigė — nuoma atlaisvinta
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

// Kito proceso knygos failas su neigiama ar sugadinta eilute negali sumažinti „užimta“ (talpa būtų išdalyta du kartus).
TEST_CASE("ledger: neigiama ar sugadinta svetimo failo eilutė neskaičiuojama", "[safe_files][ledger]") {
    Base b;
    const std::string dom = b.d();
    std::string err;
    Ledger mine;
    REQUIRE(Ledger::open(dom, kUuid, mine, err));
    const Capacity c = cap1000();
    REQUIRE(mine.reserve("f0", nvdec(100), c).admit);
    // „Kitas procesas“: vaikas laiko užrakintą .adm failą su vienu sveiku ir keliais sugadintais įrašais
    int rd[2];
    REQUIRE(::pipe(rd) == 0);
    const fs::path evil = b.lease() / "adm-evil.adm";
    const pid_t p = ::fork();
    REQUIRE(p >= 0);
    if (p == 0) {
        const int fd = ::open(evil.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        const std::string t = std::string("device=") + kUuid + "\npid=1\n" + "r ok 50.000000 0.000000 0 0\n" +
                              "r neg -500.000000 0.000000 0 0\n" + "r nan nan 0.000000 0 0\n" + "r sess 0.000000 0.000000 -3 0\n" +
                              "r vram 0.000000 0.000000 0 -5\n";
        const bool ok = fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) == 0 && ::write(fd, t.data(), t.size()) == static_cast<ssize_t>(t.size());
        const char ch = ok ? 'y' : 'n';
        (void)!::write(rd[1], &ch, 1);
        for (;;) ::pause();
    }
    char ch = 0;
    const bool ready = ::read(rd[0], &ch, 1) == 1 && ch == 'y';
    const Demand t = mine.total();
    ::kill(p, SIGKILL);
    ::waitpid(p, nullptr, 0);
    ::close(rd[0]);
    ::close(rd[1]);
    REQUIRE(ready);
    CHECK(t.nvdec_mpix_s == 150);  // 100 savo + 50 sveika eilutė; −500 ir nan neskaičiuojami
    CHECK(t.nvenc_sessions == 0);
    CHECK(t.vram_bytes == 0);
}
