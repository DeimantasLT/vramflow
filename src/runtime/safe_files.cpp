// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/safe_files.hpp"

#include <fcntl.h>
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
    char b[8];
    std::snprintf(b, sizeof b, "%04o", static_cast<unsigned>(m & kPerm));
    return b;
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
}  // namespace

std::string canonical_domain(const std::string& domain) {
    std::error_code ec;
    std::string d = std::filesystem::weakly_canonical(domain, ec).string();
    if (ec || d.empty()) d = domain;
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    return d;
}

int open_lease_dir(const std::string& domain, bool create, std::string& err) {
    err.clear();
    const std::string dom = canonical_domain(domain);
    const std::string dir = dom + ".vramflow";
    const std::string what = "nuomos katalogas " + dir + ": ";
    struct stat ds {};
    const bool have_dom = ::stat(dom.c_str(), &ds) == 0 && S_ISDIR(ds.st_mode);
    // Naujas katalogas — pirma 0700, galutinės teisės nustatomos fchmod (mkdir teisės priklausytų nuo umask)
    const bool made = create && ::mkdir(dir.c_str(), 0700) == 0;
    if (create && !made && errno != EEXIST) {
        err = what + std::strerror(errno);
        return -1;
    }
    int fd = -1, oerr = 0;
    // Kito vartotojo ką tik sukurtas katalogas trumpai būna 0700 (iki jo fchmod) — EACCES kartojamas iki 100 ms
    for (int attempt = 0; attempt < 6; ++attempt) {
        fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        oerr = fd < 0 ? errno : 0;
        if (fd >= 0 || oerr != EACCES) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (fd < 0) {
        if (oerr == ENOENT && !create) return -1;  // katalogo nėra — ne klaida (err tuščias)
        err = what + (oerr == ELOOP || oerr == ENOTDIR ? "ne katalogas arba simbolinė nuoroda (atmetama)" : std::strerror(oerr));
        return -1;
    }
    struct stat ls {};
    if (::fstat(fd, &ls) != 0) {
        err = what + std::strerror(errno);
        ::close(fd);
        return -1;
    }
    const uid_t me = ::geteuid();
    const mode_t dom_mode = have_dom ? (ds.st_mode & kPerm) : mode_t{0700};
    const bool owner_ok = ls.st_uid == me || (have_dom && (ls.st_uid == ds.st_uid || (ds.st_mode & S_IWOTH) != 0 ||
                                                           ((ds.st_mode & S_IWGRP) != 0 && ls.st_gid == ds.st_gid)));
    if (!owner_ok) {
        err = what + "savininkas (uid " + std::to_string(ls.st_uid) + ") nėra nei šis procesas, nei domeno katalogo savininkas (atmetama)";
        ::close(fd);
        return -1;
    }
    const mode_t have = ls.st_mode & kPerm;
    if (made || (have & ~dom_mode) != 0) {
        // Teisės ne platesnės nei domeno: savą katalogą sutvarkom, svetimo — nepriimam
        if (ls.st_uid != me) {
            err = what + "teisės " + octal(have) + " platesnės nei domeno katalogo " + octal(dom_mode) + " (atmetama)";
            ::close(fd);
            return -1;
        }
        mode_t want = made ? dom_mode : (have & dom_mode);
        // Grupei bendras domenas (skirtingi vartotojai, viena grupė — pvz. konteineriai su bendru fsGroup): naujas nuomos
        // katalogas gauna domeno grupę ir setgid bitą — kiti grupės nariai jį priima, o jame kuriami failai paveldi tą pačią
        // grupę. fchown nepavykus (procesas ne tos grupės narys) lieka kūrėjo grupė: kitas vartotojas katalogo nepriims
        // (aiški klaida), bet niekas nesugadinama.
        if (made && have_dom && (dom_mode & 0070) != 0 && ::fchown(fd, static_cast<uid_t>(-1), ds.st_gid) == 0) want |= S_ISGID;
        if (::fchmod(fd, want) != 0) {
            err = what + "fchmod: " + std::strerror(errno);
            ::close(fd);
            return -1;
        }
    }
    return fd;
}

int create_file_at(int dirfd, const std::string& name) {
    struct stat ds {};
    if (::fstat(dirfd, &ds) != 0) return -1;
    const int fd = ::openat(dirfd, name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    if (::fchmod(fd, ds.st_mode & 0666) != 0) {  // katalogo teisės be vykdymo bitų; nuo umask nepriklauso
        const int e = errno;
        ::close(fd);
        ::unlinkat(dirfd, name.c_str(), 0);
        errno = e;
        return -1;
    }
    return fd;
}

int open_regular_at(int dirfd, const std::string& name, int flags) {
    return only_regular(::openat(dirfd, name.c_str(), flags | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
}

int open_lock_file_at(int dirfd, const std::string& name) {
    int fd = only_regular(::openat(dirfd, name.c_str(), O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
    if (fd >= 0 || errno != ENOENT) return fd;
    fd = create_file_at(dirfd, name);
    if (fd >= 0 || errno != EEXIST) return fd;
    // kitas procesas sukūrė tuo pačiu metu — atidarom jo failą
    return only_regular(::openat(dirfd, name.c_str(), O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
}

}  // namespace mxlv::runtime
