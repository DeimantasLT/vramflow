// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/domain_lease.hpp"

#include "runtime/safe_files.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

namespace mxlv::runtime {

std::string cuda_domain_path_error(const std::string& domain) {
    std::error_code ec;
    const std::string canon = std::filesystem::canonical(domain, ec).string();  // MXL domeno kelią kanonizuoja taip pat
    if (ec || canon.size() <= kMaxCudaDomainPath) return {};
    return "domeno kelias " + canon + " — " + std::to_string(canon.size()) + " simboliai, cuda-linear flow'ams leidžiama ne daugiau kaip " +
           std::to_string(kMaxCudaDomainPath) + ": reader'io lizdo kelias <domenas>/<uuid>.mxl-flow/payload.sock netelpa į 107 simbolius "
           "(reader'iai neprisijungtų). Sutrumpinkite domeno kelią arba naudokite --storage host";
}

namespace {
constexpr const char* kExt = ".lease";

std::string read_all(int fd) {
    std::string s;
    char b[4096];
    if (::lseek(fd, 0, SEEK_SET) < 0) return s;
    for (ssize_t n; (n = ::read(fd, b, sizeof b)) > 0;) s.append(b, static_cast<std::size_t>(n));
    return s;
}

bool write_all(int fd, const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::write(fd, s.data() + off, s.size() - off);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        off += static_cast<std::size_t>(n);
    }
    return true;
}

// Nuomos failo būsena: 1 — gyvas savininkas (užraktas laikomas), 0 — negyvas (failas pašalintas), -1 — nėra / klaida / ne
// paprastas failas (simbolinė nuoroda ar FIFO nuomos kataloge — ne nuoma; atidarymas jų neseka ir nepakimba).
// text != nullptr — gyvo savininko failo turinys.
int probe(int dirfd, const std::string& name, std::string* text) {
    const int fd = open_regular_at(dirfd, name, O_RDONLY);
    if (fd < 0) return -1;
    int r;
    if (::flock(fd, LOCK_SH | LOCK_NB) == 0) {
        // užraktą gavom — savininkas negyvas (kill -9 / išėjo be release); jo failas nebereikalingas
        ::unlinkat(dirfd, name.c_str(), 0);
        r = 0;
    } else {
        r = errno == EWOULDBLOCK ? 1 : -1;
        if (r == 1 && text) *text = read_all(fd);
    }
    ::close(fd);
    return r;
}

// Visi katalogo `*.lease` vardai be plėtinio (surūšiuoti); laikini `.<run_id>.*.tmp` praleidžiami.
std::vector<std::string> lease_names(int dirfd) {
    std::vector<std::string> v;
    const int fd = ::openat(dirfd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);  // atskiras aprašas: savas skaitymo poslinkis
    DIR* d = fd >= 0 ? ::fdopendir(fd) : nullptr;
    if (!d) {
        if (fd >= 0) ::close(fd);
        return v;
    }
    const std::size_t el = std::strlen(kExt);
    while (const dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > el && n[0] != '.' && n.compare(n.size() - el, el, kExt) == 0) v.push_back(n.substr(0, n.size() - el));
    }
    ::closedir(d);
    std::sort(v.begin(), v.end());
    return v;
}
}  // namespace

std::string lease_dir(const std::string& domain) { return canonical_domain(domain) + ".vramflow"; }

std::string make_run_id(const std::string& role) {
    std::random_device rd;
    char b[96];
    std::snprintf(b, sizeof b, "%.40s-%d-%08x", role.c_str(), static_cast<int>(::getpid()), static_cast<unsigned>(rd()));
    return b;
}

bool valid_run_id(const std::string& id) {
    if (id.empty() || id.size() > 64 || id[0] == '.') return false;
    return std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    });
}

DomainLease::DomainLease(DomainLease&& o) noexcept
    : fd_(o.fd_), dirfd_(o.dirfd_), path_(std::move(o.path_)), name_(std::move(o.name_)), run_id_(std::move(o.run_id_)) {
    o.fd_ = o.dirfd_ = -1;
}

DomainLease& DomainLease::operator=(DomainLease&& o) noexcept {
    if (this != &o) {
        release();
        fd_ = o.fd_;
        dirfd_ = o.dirfd_;
        path_ = std::move(o.path_);
        name_ = std::move(o.name_);
        run_id_ = std::move(o.run_id_);
        o.fd_ = o.dirfd_ = -1;
    }
    return *this;
}

DomainLease::~DomainLease() { release(); }

void DomainLease::release() {
    if (fd_ < 0) return;
    ::unlinkat(dirfd_, name_.c_str(), 0);  // pirma pašalinam, tada atlaisvinam: niekas nemato atrakinto mūsų failo
    ::close(fd_);                          // close atlaisvina flock
    ::close(dirfd_);
    fd_ = dirfd_ = -1;
}

bool DomainLease::acquire(const std::string& domain, const std::string& run_id, DomainLease& out, std::string& err) {
    out.release();
    if (!valid_run_id(run_id)) { err = "run_id '" + run_id + "' netinkamas (1…64 simboliai A-Z a-z 0-9 . _ -)"; return false; }
    // Katalogas — patikrintas (ne nuoroda, savininkas, teisės ne platesnės nei domeno); toliau viskas per jo fd (openat)
    const int dirfd = open_lease_dir(domain, true, err);
    if (dirfd < 0) return false;
    const std::string dir = lease_dir(domain), name = run_id + kExt, path = dir + "/" + name;
    if (probe(dirfd, name, nullptr) == 1) {
        err = "run_id " + run_id + " jau naudoja gyvas procesas (" + path + ")";
        ::close(dirfd);
        return false;
    }
    // Laikinas failas užrakinamas PRIEŠ paskelbiant: live_owners niekada nemato paskelbto, bet neužrakinto failo.
    // Kuriamas O_EXCL | O_NOFOLLOW; pasenęs to paties vardo įrašas (ir simbolinė nuoroda) pašalinamas, taikinys neliečiamas.
    const std::string tmp = "." + run_id + "." + std::to_string(::getpid()) + ".tmp";
    int fd = create_file_at(dirfd, tmp);
    if (fd < 0 && errno == EEXIST) {
        ::unlinkat(dirfd, tmp.c_str(), 0);
        fd = create_file_at(dirfd, tmp);
    }
    if (fd < 0) {
        err = "nuomos failas " + dir + "/" + tmp + ": " + std::strerror(errno);
        ::close(dirfd);
        return false;
    }
    auto fail = [&](const std::string& e) {
        err = e;
        ::close(fd);
        ::unlinkat(dirfd, tmp.c_str(), 0);
        ::close(dirfd);
        return false;
    };
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) return fail("flock " + dir + "/" + tmp + ": " + std::strerror(errno));
    if (!write_all(fd, "run_id=" + run_id + "\npid=" + std::to_string(::getpid()) + "\n"))
        return fail("nuomos failo rašymas: " + std::string(std::strerror(errno)));
    if (::renameat2(dirfd, tmp.c_str(), dirfd, name.c_str(), RENAME_NOREPLACE) != 0)
        return fail(errno == EEXIST ? "run_id " + run_id + " jau naudojamas (" + path + ")"
                                    : "nuomos paskelbimas: " + std::string(std::strerror(errno)));
    out.fd_ = fd;
    out.dirfd_ = dirfd;
    out.path_ = path;
    out.name_ = name;
    out.run_id_ = run_id;
    return true;
}

std::vector<std::string> DomainLease::live_owners(const std::string& domain) {
    std::vector<std::string> v;
    std::string err;
    const int dirfd = open_lease_dir(domain, false, err);  // nėra arba nepatikimas katalogas — savininkų nėra
    if (dirfd < 0) return v;
    for (const std::string& n : lease_names(dirfd))
        if (probe(dirfd, n + kExt, nullptr) == 1) v.push_back(n);
    ::close(dirfd);
    return v;
}

std::string DomainLease::flow_owner(const std::string& domain, const std::string& flow_id) {
    std::string err, owner;
    const int dirfd = open_lease_dir(domain, false, err);
    if (dirfd < 0) return owner;
    const std::string line = "\nflow=" + flow_id + "\n";
    for (const std::string& n : lease_names(dirfd)) {
        std::string text;
        if (probe(dirfd, n + kExt, &text) == 1 && ("\n" + text).find(line) != std::string::npos) {
            owner = n;
            break;
        }
    }
    ::close(dirfd);
    return owner;
}

bool DomainLease::add_flow(const std::string& flow_id) {
    if (fd_ < 0) return false;
    if (::lseek(fd_, 0, SEEK_END) < 0) return false;
    return write_all(fd_, "flow=" + flow_id + "\n");
}

}  // namespace mxlv::runtime
