// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/admission.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "runtime/domain_lease.hpp"
#include "runtime/safe_files.hpp"

namespace mxlv::runtime {

namespace {

// --- Minimalus JSON skaitytuvas (talpos failui; projekte kito JSON skaitytuvo nėra, nauja priklausomybė — netinka) ------
// Objektai, masyvai, eilutės (\" \\ \/ \b \f \n \r \t; \uXXXX — tik ASCII), skaičiai, true / false / null. Gylis ribotas.
struct JVal {
    enum class T { Null, Bool, Num, Str, Obj, Arr } t = T::Null;
    double num = 0;
    bool b = false;
    std::string str;
    std::vector<std::string> keys;  // objektas: raktai (reikšmės — arr tomis pačiomis pozicijomis; vector — leidžia nepilną tipą)
    std::vector<JVal> arr;
    const JVal* get(const std::string& k) const {
        for (std::size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == k) return &arr[i];
        return nullptr;
    }
};

class JParser {
public:
    explicit JParser(const std::string& s) : s_(s) {}
    bool parse(JVal& out, std::string& err) {
        if (!value(out, 0)) {
            err = "netinkamas JSON (pozicija " + std::to_string(p_) + ")";
            return false;
        }
        ws();
        if (p_ != s_.size()) {
            err = "netinkamas JSON: perteklinis tekstas (pozicija " + std::to_string(p_) + ")";
            return false;
        }
        return true;
    }

private:
    const std::string& s_;
    std::size_t p_ = 0;
    void ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) ++p_;
    }
    bool lit(const char* w) {
        const std::size_t n = std::char_traits<char>::length(w);
        if (s_.compare(p_, n, w) != 0) return false;
        p_ += n;
        return true;
    }
    bool string(std::string& o) {
        if (p_ >= s_.size() || s_[p_] != '"') return false;
        ++p_;
        while (p_ < s_.size()) {
            const char c = s_[p_++];
            if (c == '"') return true;
            if (c != '\\') {
                o += c;
                continue;
            }
            if (p_ >= s_.size()) return false;
            const char e = s_[p_++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    if (p_ + 4 > s_.size()) return false;
                    const unsigned long cp = std::strtoul(s_.substr(p_, 4).c_str(), nullptr, 16);
                    p_ += 4;
                    o += cp < 0x80 ? static_cast<char>(cp) : '?';  // talpos faile ne ASCII reikšmių nėra (GPU vardas — ASCII)
                    break;
                }
                default: return false;
            }
        }
        return false;
    }
    bool value(JVal& v, int depth) {
        if (depth > 32) return false;
        ws();
        if (p_ >= s_.size()) return false;
        const char c = s_[p_];
        if (c == '{') {
            ++p_;
            v.t = JVal::T::Obj;
            ws();
            if (p_ < s_.size() && s_[p_] == '}') { ++p_; return true; }
            for (;;) {
                ws();
                std::string k;
                if (!string(k)) return false;
                ws();
                if (p_ >= s_.size() || s_[p_++] != ':') return false;
                if (v.get(k)) return false;  // pasikartojantis raktas — netinkamas talpos failas
                v.keys.push_back(std::move(k));
                v.arr.emplace_back();
                if (!value(v.arr.back(), depth + 1)) return false;
                ws();
                if (p_ >= s_.size()) return false;
                if (s_[p_] == ',') { ++p_; continue; }
                if (s_[p_] == '}') { ++p_; return true; }
                return false;
            }
        }
        if (c == '[') {
            ++p_;
            v.t = JVal::T::Arr;
            ws();
            if (p_ < s_.size() && s_[p_] == ']') { ++p_; return true; }
            for (;;) {
                v.arr.emplace_back();
                if (!value(v.arr.back(), depth + 1)) return false;
                ws();
                if (p_ >= s_.size()) return false;
                if (s_[p_] == ',') { ++p_; continue; }
                if (s_[p_] == ']') { ++p_; return true; }
                return false;
            }
        }
        if (c == '"') {
            v.t = JVal::T::Str;
            return string(v.str);
        }
        if (lit("true")) { v.t = JVal::T::Bool; v.b = true; return true; }
        if (lit("false")) { v.t = JVal::T::Bool; return true; }
        if (lit("null")) { v.t = JVal::T::Null; return true; }
        const char* b = s_.c_str() + p_;
        char* e = nullptr;
        v.num = std::strtod(b, &e);
        if (e == b || !std::isfinite(v.num)) return false;
        v.t = JVal::T::Num;
        p_ += static_cast<std::size_t>(e - b);
        return true;
    }
};

// Kelias „a.b.c“ objektuose; nullptr — nėra.
const JVal* at(const JVal& root, const std::string& path) {
    const JVal* cur = &root;
    std::size_t s = 0;
    while (s <= path.size()) {
        const std::size_t e = std::min(path.find('.', s), path.size());
        if (cur->t != JVal::T::Obj) return nullptr;
        cur = cur->get(path.substr(s, e - s));
        if (!cur) return nullptr;
        s = e + 1;
    }
    return cur;
}

std::string upper_uuid(std::string u) {
    for (char& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return u;
}

std::string fmt2(double v) {  // „829,44“ — lietuviškas dešimtainis kablelis (pranešimai operatoriui)
    char b[64];
    std::snprintf(b, sizeof b, "%.2f", v);
    std::string s = b;
    for (char& c : s)
        if (c == '.') c = ',';
    return s;
}

// Bendras sprendimo žingsnis: telpa, jei used + req ≤ limit (santykinė paklaida 1e-9 — sumuojamos trupmenos)
bool over(double used, double req, double limit) { return used + req > limit * (1 + 1e-9); }

}  // namespace

Demand operator+(const Demand& a, const Demand& b) {
    Demand r;
    r.nvdec_mpix_s = a.nvdec_mpix_s + b.nvdec_mpix_s;
    r.nvenc_mpix_s = a.nvenc_mpix_s + b.nvenc_mpix_s;
    r.nvenc_sessions = a.nvenc_sessions + b.nvenc_sessions;
    r.vram_bytes = a.vram_bytes + b.vram_bytes;
    return r;
}

Decision decide(const Demand& used, const Demand& req, const Capacity& c) {
    Decision d;
    auto check = [&](const char* engine, double u, double r, double limit) {
        if (!d.admit || r <= 0 || limit <= 0) return;
        if (over(u, r, limit)) d = Decision{false, engine, u, r, limit};
    };
    check("nvdec", used.nvdec_mpix_s, req.nvdec_mpix_s, c.nvdec_mpix_s * c.headroom);
    check("nvenc", used.nvenc_mpix_s, req.nvenc_mpix_s, c.nvenc_mpix_s * c.headroom);
    check("nvenc_sessions", used.nvenc_sessions, req.nvenc_sessions, c.nvenc_sessions);  // kieta tvarkyklės riba: be atsargos
    check("vram", static_cast<double>(used.vram_bytes), static_cast<double>(req.vram_bytes), static_cast<double>(c.vram_bytes) * c.headroom);
    return d;
}

std::string describe(const Decision& d) {
    if (d.admit) return "priimta";
    if (d.engine == "vram") {
        const double mb = 1024.0 * 1024.0;
        return "vram: prašoma " + fmt2(d.requested / mb) + ", užimta " + fmt2(d.used / mb) + ", leidžiama " + fmt2(d.limit / mb) + " MB";
    }
    if (d.engine == "nvenc_sessions")
        return "nvenc_sessions: prašoma " + std::to_string(std::llround(d.requested)) + ", užimta " + std::to_string(std::llround(d.used)) +
               ", leidžiama " + std::to_string(std::llround(d.limit)) + " sesijų (tvarkyklės riba)";
    return d.engine + ": prašoma " + fmt2(d.requested) + ", užimta " + fmt2(d.used) + ", leidžiama " + fmt2(d.limit) + " Mpix/s";
}

bool parse_capacity(const std::string& json, Capacity& out, std::string& err) {
    JVal root;
    if (!JParser(json).parse(root, err)) return false;
    if (root.t != JVal::T::Obj) {
        err = "talpos failas — ne JSON objektas";
        return false;
    }
    static const char* kRequired[] = {"schema", "gpu.name", "gpu.uuid", "gpu.driver", "format", "measured_utc", "engines.nvdec.mpix_s",
                                      "engines.nvdec.bound", "engines.nvenc.mpix_s", "engines.nvenc.bound", "functions"};
    for (const char* k : kRequired)
        if (!at(root, k)) {
            err = std::string("talpos faile trūksta lauko ") + k;
            return false;
        }
    const JVal& schema = *at(root, "schema");
    if (schema.t != JVal::T::Num || schema.num != 1) {
        err = "talpos failo schema " + (schema.t == JVal::T::Num ? std::to_string(std::llround(schema.num)) : std::string("?")) +
              ", palaikoma 1";
        return false;
    }
    auto str = [&](const char* k, std::string& o) {
        const JVal* v = at(root, k);
        if (v->t != JVal::T::Str) {
            err = std::string("talpos failo laukas ") + k + " — ne tekstas";
            return false;
        }
        o = v->str;
        return true;
    };
    // Neneigiamas skaičius; null / nėra — 0 (neišmatuota), jei leidžiama
    auto num = [&](const JVal* v, const std::string& k, double& o, bool null_ok) {
        if ((!v || v->t == JVal::T::Null) && null_ok) {
            o = 0;
            return true;
        }
        if (!v || v->t != JVal::T::Num || !std::isfinite(v->num) || v->num < 0) {
            err = "talpos failo laukas " + k + " — ne neneigiamas baigtinis skaičius";
            return false;
        }
        o = v->num;
        return true;
    };
    // Sveikasis 0…max (imtinai). Trupmena arba per didelė reikšmė — klaida su lauko vardu: anksčiau static_cast ją tyliai
    // apkarpydavo (fuzzerio radinys: "knee": 2232340777 → neigiamas int; neigiama sesijų riba reikštų „be ribos“).
    auto whole = [&](const JVal* v, const std::string& k, double max, double& o, bool null_ok) {
        if (!num(v, k, o, null_ok)) return false;
        if (o > max || o != std::floor(o)) {
            char b[64];
            std::snprintf(b, sizeof b, "%.0f", max);
            err = "talpos failo laukas " + k + " — ne sveikasis skaičius 0…" + b;
            return false;
        }
        return true;
    };
    constexpr double kMaxCount = 1e6;                 // sesijos, kelias (srautų skaičius)
    constexpr double kMaxBytes = 9007199254740992.0;  // 2^53 — didžiausias sveikasis, kurį double atvaizduoja tiksliai
    Capacity c;
    c.headroom = out.headroom;
    c.source = out.source;
    if (!str("gpu.name", c.gpu_name) || !str("gpu.uuid", c.gpu_uuid) || !str("gpu.driver", c.driver) || !str("format", c.format) ||
        !str("measured_utc", c.measured_utc))
        return false;
    double v = 0;
    if (!num(at(root, "engines.nvdec.mpix_s"), "engines.nvdec.mpix_s", c.nvdec_mpix_s, false)) return false;
    if (!num(at(root, "engines.nvenc.mpix_s"), "engines.nvenc.mpix_s", c.nvenc_mpix_s, false)) return false;
    if (!whole(at(root, "engines.nvenc.sessions"), "engines.nvenc.sessions", kMaxCount, v, true)) return false;
    c.nvenc_sessions = static_cast<int>(v);
    if (!whole(at(root, "gpu.vram_bytes"), "gpu.vram_bytes", kMaxBytes, v, true)) return false;
    c.vram_bytes = static_cast<std::uint64_t>(v);
    std::string bd, be;
    if (!str("engines.nvdec.bound", bd) || !str("engines.nvenc.bound", be)) return false;
    c.nvdec_lower_bound = bd == "lower_bound";
    c.nvenc_lower_bound = be == "lower_bound";
    const JVal& fn = *at(root, "functions");
    if (fn.t != JVal::T::Obj) {
        err = "talpos failo laukas functions — ne objektas";
        return false;
    }
    for (std::size_t i = 0; i < fn.keys.size(); ++i) {
        const std::string& tag = fn.keys[i];
        const JVal& f = fn.arr[i];
        FunctionCapacity fc;
        const std::string p = "functions." + tag + ".";
        if (f.t != JVal::T::Obj) {
            err = "talpos failo laukas functions." + tag + " — ne objektas";
            return false;
        }
        auto field = [&](const char* k) { return f.get(k); };
        if (!whole(field("vram_per_flow_bytes"), p + "vram_per_flow_bytes", kMaxBytes, v, true)) return false;
        fc.vram_per_flow_bytes = static_cast<std::uint64_t>(v);
        if (!whole(field("vram_offset_bytes"), p + "vram_offset_bytes", kMaxBytes, v, true)) return false;
        fc.vram_offset_bytes = static_cast<std::uint64_t>(v);
        if (!whole(field("knee"), p + "knee", kMaxCount, v, true)) return false;
        fc.knee = static_cast<int>(v);
        c.functions[tag] = fc;
    }
    out = std::move(c);
    return true;
}

bool load_capacity(const std::string& path, const std::string& device_uuid, Capacity& out, std::string& err) {
    std::string file = path;
    struct stat sb {};
    if (::stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) file = path + "/capacity-" + device_uuid + ".json";
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        err = "talpa neišmatuota: nėra talpos failo " + file;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    Capacity c;
    c.headroom = out.headroom;
    c.source = file;
    if (!parse_capacity(ss.str(), c, err)) {
        err = file + ": " + err;
        return false;
    }
    if (!device_uuid.empty() && upper_uuid(c.gpu_uuid) != upper_uuid(device_uuid)) {
        err = "talpos failas " + file + " išmatuotas kitai GPU (failo " + c.gpu_uuid + ", ši " + device_uuid + ") — nenaudojamas";
        return false;
    }
    out = std::move(c);
    return true;
}

std::vector<std::string> capacity_warnings(const Capacity& c, const std::string& running_driver) {
    std::vector<std::string> w;
    if (running_driver.empty())
        w.push_back("tvarkyklės versija nežinoma (be NVML) — negalima patikrinti, ar talpos failas (išmatuotas su " + c.driver +
                    ") dar galioja");
    else if (running_driver != c.driver)
        w.push_back("tvarkyklės versija: talpos failas išmatuotas su " + c.driver + ", dabar — " + running_driver +
                    ": talpa galėjo pasikeisti, permatuokite (PROFILE=knee scripts/linux/70-capacity.sh)");
    auto lb = [&](bool on, const char* eng, double v) {
        if (on)
            w.push_back(std::string(eng) + ": talpa " + fmt2(v) + " Mpix/s faile pažymėta bound = lower_bound (tikra talpa didesnė — " +
                        "kelią nulėmė paieškos ar tvarkyklės riba); ji vis tiek naudojama kaip riba (atsargiai)");
    };
    lb(c.nvdec_lower_bound, "nvdec", c.nvdec_mpix_s);
    lb(c.nvenc_lower_bound, "nvenc", c.nvenc_mpix_s);
    return w;
}

double mpix_s(const vramflow::VideoFormat& f) {
    return static_cast<double>(f.raster.w) * f.raster.h * static_cast<double>(f.frame_rate.num) / static_cast<double>(f.frame_rate.den) / 1e6;
}

Demand ingest_demand(const vramflow::VideoFormat& f, int surfaces, std::uint32_t ring_grains, bool nvdec) {
    Demand d;
    if (nvdec) d.nvdec_mpix_s = mpix_s(f);
    const std::uint64_t w = (static_cast<std::uint64_t>(f.raster.w) + 63) / 64 * 64, h = (static_cast<std::uint64_t>(f.raster.h) + 31) / 32 * 32;
    const std::uint64_t nv12 = w * h * 3 / 2;
    vramflow::gfx::Raster r = f.raster;
    r.alpha = false;
    d.vram_bytes = static_cast<std::uint64_t>(surfaces > 0 ? surfaces : 0) * nv12 + std::uint64_t{ring_grains} * vramflow::gfx::grain_bytes(r);
    return d;
}

// --- Ledger --------------------------------------------------------------------------------------------------------
namespace {
constexpr const char* kAdmExt = ".adm";

bool write_all(int fd, const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::pwrite(fd, s.data() + off, s.size() - off, static_cast<off_t>(off));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        off += static_cast<std::size_t>(n);
    }
    return true;
}

std::string read_fd(int fd) {
    std::string s;
    char b[4096];
    off_t off = 0;
    for (ssize_t n; (n = ::pread(fd, b, sizeof b, off)) > 0; off += n) s.append(b, static_cast<std::size_t>(n));
    return s;
}

// Bendras knygos užraktas `<katalogas>/admission.lock` (LOCK_EX, blokuojantis): skaitymas + sprendimas + įrašymas atominiai.
// Laikomas tik kelias ms (failų skaitymas), todėl blokuojantis laukimas priimtinas (paleidimo metu, ne tick gijoje).
class DirLock {
public:
    explicit DirLock(int dirfd) {
        fd_ = open_lock_file_at(dirfd, "admission.lock");  // nuoroda ar ne paprastas failas — atmetama
        if (fd_ >= 0)
            while (::flock(fd_, LOCK_EX) != 0 && errno == EINTR) {
            }
    }
    ~DirLock() {
        if (fd_ >= 0) ::close(fd_);
    }
    DirLock(const DirLock&) = delete;
    DirLock& operator=(const DirLock&) = delete;
    bool ok() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

// Rezervacijos ID faile — be tarpų ir eilučių pabaigų.
std::string clean_id(std::string id) {
    for (char& c : id)
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') c = '_';
    return id.empty() ? "_" : id;
}

// Kito proceso failo turinys → jo rezervacijų suma (0, jei kita GPU).
Demand parse_ledger(const std::string& text, const std::string& device) {
    Demand sum;
    bool same_device = false;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("device=", 0) == 0) same_device = upper_uuid(line.substr(7)) == upper_uuid(device);
        if (line.rfind("r ", 0) != 0) continue;
        std::istringstream ls(line.substr(2));
        std::string id;
        Demand d;
        if (ls >> id >> d.nvdec_mpix_s >> d.nvenc_mpix_s >> d.nvenc_sessions >> d.vram_bytes) sum = sum + d;
    }
    return same_device ? sum : Demand{};
}
}  // namespace

Ledger::Ledger(Ledger&& o) noexcept
    : fd_(o.fd_), dirfd_(o.dirfd_), dir_(std::move(o.dir_)), path_(std::move(o.path_)), name_(std::move(o.name_)),
      device_(std::move(o.device_)), res_(std::move(o.res_)) {
    o.fd_ = o.dirfd_ = -1;
}

Ledger& Ledger::operator=(Ledger&& o) noexcept {
    if (this != &o) {
        close();
        fd_ = o.fd_;
        dirfd_ = o.dirfd_;
        dir_ = std::move(o.dir_);
        path_ = std::move(o.path_);
        name_ = std::move(o.name_);
        device_ = std::move(o.device_);
        res_ = std::move(o.res_);
        o.fd_ = o.dirfd_ = -1;
    }
    return *this;
}

Ledger::~Ledger() { close(); }

void Ledger::close() {
    res_.clear();
    if (fd_ < 0) return;
    ::unlinkat(dirfd_, name_.c_str(), 0);  // pirma pašalinam, tada atlaisvinam: niekas nemato atrakinto mūsų failo
    ::close(fd_);
    ::close(dirfd_);
    fd_ = dirfd_ = -1;
}

bool Ledger::open(const std::string& domain, const std::string& device_uuid, Ledger& out, std::string& err) {
    out.close();
    // Katalogas — patikrintas (ne nuoroda, savininkas, teisės ne platesnės nei domeno); toliau viskas per jo fd
    const int dirfd = open_lease_dir(domain, true, err);
    if (dirfd < 0) {
        err = "priėmimo knyga: " + err;
        return false;
    }
    const std::string dir = lease_dir(domain);
    auto fail = [&](const std::string& e) {
        err = e;
        ::close(dirfd);
        return false;
    };
    DirLock lock(dirfd);
    if (!lock.ok()) return fail("priėmimo knyga: " + dir + "/admission.lock: " + std::strerror(errno));
    // Failas sukuriamas ir užrakinamas po bendru užraktu: kitas skaitytojas (irgi po juo) niekada nemato neužrakinto failo
    std::string name;
    int fd = -1;
    for (int attempt = 0; attempt < 8 && fd < 0; ++attempt) {
        name = make_run_id("adm") + kAdmExt;
        fd = create_file_at(dirfd, name);  // O_EXCL | O_NOFOLLOW: esamo vardo (ir nuorodos) neperrašo
        if (fd < 0 && errno != EEXIST) break;
    }
    if (fd < 0) return fail("priėmimo knyga: " + dir + "/" + name + ": " + std::strerror(errno));
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        const std::string e = "priėmimo knyga: flock " + dir + "/" + name + ": " + std::strerror(errno);
        ::close(fd);
        ::unlinkat(dirfd, name.c_str(), 0);
        return fail(e);
    }
    out.fd_ = fd;
    out.dirfd_ = dirfd;
    out.dir_ = dir;
    out.name_ = name;
    out.path_ = dir + "/" + name;
    out.device_ = device_uuid;
    if (!out.write_own()) {
        err = "priėmimo knyga: rašymas " + out.path_ + ": " + std::strerror(errno);
        out.close();
        return false;
    }
    return true;
}

bool Ledger::write_own() {
    std::string t = "device=" + device_ + "\npid=" + std::to_string(::getpid()) + "\n";
    char b[160];
    for (const auto& [id, d] : res_) {
        std::snprintf(b, sizeof b, " %.6f %.6f %d %llu\n", d.nvdec_mpix_s, d.nvenc_mpix_s, d.nvenc_sessions,
                      static_cast<unsigned long long>(d.vram_bytes));
        t += "r " + id + b;
    }
    if (::ftruncate(fd_, 0) != 0) return false;
    return write_all(fd_, t);
}

Demand Ledger::others() const {
    Demand sum;
    const int lfd = ::openat(dirfd_, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);  // atskiras aprašas: savas skaitymo poslinkis
    DIR* d = lfd >= 0 ? ::fdopendir(lfd) : nullptr;
    if (!d) {
        if (lfd >= 0) ::close(lfd);
        return sum;
    }
    const std::size_t el = std::strlen(kAdmExt);
    std::vector<std::string> names;
    while (const dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > el && n[0] != '.' && n.compare(n.size() - el, el, kAdmExt) == 0) names.push_back(n);
    }
    ::closedir(d);
    for (const std::string& n : names) {
        if (n == name_) continue;
        const int fd = open_regular_at(dirfd_, n, O_RDONLY);  // nuoroda ar FIFO — ne rezervacija; atidarymas nepakimba
        if (fd < 0) continue;
        if (::flock(fd, LOCK_SH | LOCK_NB) == 0)
            ::unlinkat(dirfd_, n.c_str(), 0);  // užraktą gavom — savininkas negyvas (kill -9): jo rezervacija nebeskaičiuojama
        else if (errno == EWOULDBLOCK)
            sum = sum + parse_ledger(read_fd(fd), device_);
        ::close(fd);
    }
    return sum;
}

Demand Ledger::own() const {
    Demand s;
    for (const auto& r : res_) s = s + r.second;
    return s;
}

Demand Ledger::total() const {
    if (fd_ < 0) return {};
    DirLock lock(dirfd_);
    return others() + own();
}

Decision Ledger::reserve(const std::string& id_in, const Demand& d, const Capacity& c, bool force) {
    if (fd_ < 0) return Decision{false, "ledger", 0, 0, 0};
    const std::string id = clean_id(id_in);
    DirLock lock(dirfd_);
    res_.erase(std::remove_if(res_.begin(), res_.end(), [&](const auto& r) { return r.first == id; }), res_.end());
    const Decision dec = decide(others() + own(), d, c);
    if (dec.admit || force) {
        res_.emplace_back(id, d);
        if (!write_own()) return Decision{false, "ledger", 0, 0, 0};
    }
    return dec;
}

void Ledger::release(const std::string& id_in) {
    if (fd_ < 0) return;
    const std::string id = clean_id(id_in);
    DirLock lock(dirfd_);
    res_.erase(std::remove_if(res_.begin(), res_.end(), [&](const auto& r) { return r.first == id; }), res_.end());
    write_own();
}

// --- Priėmimas iš eilės ----------------------------------------------------------------------------------------------
bool parse_admission_mode(const std::string& s, AdmissionMode& out) {
    if (s == "off") out = AdmissionMode::Off;
    else if (s == "warn") out = AdmissionMode::Warn;
    else if (s == "enforce") out = AdmissionMode::Enforce;
    else return false;
    return true;
}

bool resolve_admission_mode(const std::string& s, bool have_capacity, AdmissionMode& out) {
    if (s == "auto") {
        out = have_capacity ? AdmissionMode::Enforce : AdmissionMode::Warn;
        return true;
    }
    return parse_admission_mode(s, out);
}

AdmitResult admit_in_order(Ledger& l, const std::string& prefix, const Demand& process, const std::vector<Demand>& flows,
                           const Capacity& c, AdmissionMode mode) {
    AdmitResult r;
    r.decisions.resize(flows.size());
    if (mode == AdmissionMode::Off) {
        r.admitted = flows.size();
        return r;
    }
    const bool force = mode == AdmissionMode::Warn;
    const bool any_process = process.nvdec_mpix_s > 0 || process.nvenc_mpix_s > 0 || process.nvenc_sessions > 0 || process.vram_bytes > 0;
    if (any_process) {
        const Decision d = l.reserve(prefix + "-proc", process, c, force);
        if (!d.admit && !force) {
            r.process_refused = true;
            for (Decision& x : r.decisions) x = d;
            return r;
        }
        if (!d.admit) ++r.over;
    }
    bool stopped = false;  // enforce: po pirmo atmetimo — visi tolesni atmetami
    for (std::size_t k = 0; k < flows.size(); ++k) {
        const std::string id = prefix + "-" + std::to_string(k);
        if (stopped) {
            Decision d = decide(l.total(), flows[k], c);
            if (d.admit) d = Decision{false, "order", 0, 0, 0};
            r.decisions[k] = d;
            continue;
        }
        const Decision d = l.reserve(id, flows[k], c, force);
        r.decisions[k] = d;
        if (d.admit || force) {
            ++r.admitted;
            if (!d.admit) ++r.over;
        } else {
            stopped = true;
        }
    }
    return r;
}

}  // namespace mxlv::runtime
