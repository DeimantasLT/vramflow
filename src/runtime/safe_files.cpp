// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/safe_files.hpp"

#include "common/textf.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>

namespace mxlv::runtime {
namespace {
constexpr mode_t kPerm = 0777;

std::string octal(mode_t m) {
    return strf("%04o", static_cast<unsigned>(m & kPerm));
}

// Nuomos katalogo teisės pagal domeno katalogo teises: savininkas — visada; grupė ir „kiti“ — tik jei domenas jiems rašomas.
mode_t lease_dir_mode(mode_t dom) {
    mode_t m = 0700;
    if (dom & S_IWGRP) m |= 0070;
    if (dom & S_IWOTH) m |= 0007;
    return m;
}

// Paprastas failas? Kitaip uždaro fd ir grąžina -1 su errno = EINVAL.
int only_regular(int fd) {
    if (fd < 0) return -1;
    struct stat st {};
    if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) return fd;
    ::close(fd);
    errno = EINVAL;
    return -1;
}

// open / openat su trumpu kartojimu, kai gaunama EACCES: kito vartotojo ką tik sukurtas katalogas ar failas trumpai būna
// 0700 / 0600 — iki jo fchown ir fchmod (≤ 100 ms).
template <class Open>
int open_retry_eacces(Open&& open) {
    int fd = -1;
    for (int attempt = 0; attempt < 6; ++attempt) {
        fd = open();
        if (fd >= 0 || errno != EACCES) break;
        const int e = errno;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        errno = e;
    }
    return fd;
}
}  // namespace

std::string canonical_domain(const std::string& domain) {
    std::error_code ec;
    std::string d = std::filesystem::weakly_canonical(domain, ec).string();
    if (ec || d.empty()) d = domain;
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    return d;
}

bool open_lease_dir(const std::string& domain, bool create, LeaseDir& out, std::string& err) {
    err.clear();
    out = LeaseDir{};
    const std::string dom = canonical_domain(domain);
    const std::string dir = dom + ".vramflow";
    const std::string what = "nuomos katalogas " + dir + ": ";
    struct stat ds {};
    const bool have_dom = ::stat(dom.c_str(), &ds) == 0 && S_ISDIR(ds.st_mode);
    mode_t want = have_dom ? lease_dir_mode(ds.st_mode) : mode_t{0700};
    // Naujas katalogas — pirma 0700; galutinė grupė ir teisės nustatomos per aprašą (mkdir teisės priklausytų nuo umask)
    const bool made = create && ::mkdir(dir.c_str(), 0700) == 0;
    if (create && !made && errno != EEXIST) {
        err = what + errno_text(errno);
        return false;
    }
    const int fd = open_retry_eacces([&] { return ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC); });
    if (fd < 0) {
        const int oerr = errno;
        if (oerr == ENOENT && !create) return false;  // katalogo nėra — ne klaida (err tuščias)
        err = what + (oerr == ELOOP || oerr == ENOTDIR ? "ne katalogas arba simbolinė nuoroda (atmetama)" : errno_text(oerr));
        return false;
    }
    auto fail = [&](const std::string& e) {
        err = what + e;
        ::close(fd);
        return false;
    };
    struct stat ls {};
    if (::fstat(fd, &ls) != 0) return fail(errno_text(errno));
    const uid_t me = ::geteuid();
    const bool owner_ok = ls.st_uid == me || (have_dom && (ls.st_uid == ds.st_uid || (ds.st_mode & S_IWOTH) != 0 ||
                                                           ((ds.st_mode & S_IWGRP) != 0 && ls.st_gid == ds.st_gid)));
    if (!owner_ok)
        return fail("savininkas (uid " + std::to_string(ls.st_uid) + ") nėra nei šis procesas, nei domeno katalogo savininkas (atmetama)");
    gid_t group = static_cast<gid_t>(-1);
    if (made) {
        // Grupei rašomas domenas (skirtingi vartotojai, viena grupė — pvz. konteineriai su bendru fsGroup): katalogas gauna
        // domeno grupę ir setgid bitą — kiti grupės nariai jį priima, o jame kuriami failai paveldi tą pačią grupę.
        // fchown nepavykus (procesas ne tos grupės narys) grupės teisės nesuteikiamos: jos atitektų ne tai grupei.
        if ((want & 0070) != 0 && ::fchown(fd, static_cast<uid_t>(-1), ds.st_gid) != 0 && ls.st_gid != ds.st_gid) want &= ~mode_t{0070};
        if (::fchmod(fd, want | ((want & 0070) != 0 ? mode_t{S_ISGID} : mode_t{0})) != 0) return fail(std::string("fchmod: ") + errno_text(errno));
    } else if (have_dom) {
        // Esamas katalogas: teisės ne platesnės, nei priklauso šiam domenui — savą susiaurinam (setgid bitą paliekam),
        // svetimo nepriimam. Kai domeno nėra, apie teises spręsti negalima — katalogas nekeičiamas.
        const mode_t have = ls.st_mode & kPerm;
        if ((have & ~want) != 0) {
            if (ls.st_uid != me) return fail("teisės " + octal(have) + " platesnės nei leidžia domenas (" + octal(want) + ") (atmetama)");
            if (::fchmod(fd, (ls.st_mode & S_ISGID) | (have & want)) != 0) return fail(std::string("fchmod: ") + errno_text(errno));
        }
    }
    if (have_dom && (want & 0070) != 0) group = ds.st_gid;
    out.fd = fd;
    out.file_mode = want & 0666;
    out.group = group;
    return true;
}

int create_file_at(const LeaseDir& d, const std::string& name) {
    const int fd = ::openat(d.fd, name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    mode_t mode = d.file_mode;
    // Grupė — domeno (jei katalogas dar be setgid bito arba kūrėjo pirminė grupė kita); nepavykus grupės teisės nesuteikiamos
    if (d.group != static_cast<gid_t>(-1) && ::fchown(fd, static_cast<uid_t>(-1), d.group) != 0) {
        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_gid != d.group) mode &= ~mode_t{0060};
    }
    if (::fchmod(fd, mode) != 0) {  // nuo umask nepriklauso
        const int e = errno;
        ::close(fd);
        ::unlinkat(d.fd, name.c_str(), 0);
        errno = e;
        return -1;
    }
    return fd;
}

int open_regular_at(int dirfd, const std::string& name, int flags) {
    return only_regular(::openat(dirfd, name.c_str(), flags | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
}

bool same_file_at(int dirfd, const std::string& name, int fd) {
    struct stat a {}, b {};
    return ::fstat(fd, &a) == 0 && ::fstatat(dirfd, name.c_str(), &b, AT_SYMLINK_NOFOLLOW) == 0 && a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

DirLock::DirLock(const LeaseDir& d, const char* name, int timeout_ms) {
    const auto open_existing = [&] { return open_regular_at(d.fd, name, O_RDWR); };
    fd_ = open_retry_eacces(open_existing);
    if (fd_ < 0 && errno == ENOENT) {
        fd_ = create_file_at(d, name);
        if (fd_ < 0 && errno == EEXIST) fd_ = open_retry_eacces(open_existing);  // kitas procesas sukūrė tuo pačiu metu
    }
    if (fd_ < 0) return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) {
            locked_ = true;
            return;
        }
        if (errno != EWOULDBLOCK && errno != EINTR) return;
        if (std::chrono::steady_clock::now() >= deadline) {
            errno = ETIMEDOUT;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

DirLock::~DirLock() {
    if (fd_ >= 0) ::close(fd_);  // close atlaisvina flock
}

}  // namespace mxlv::runtime
