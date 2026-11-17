// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/admission.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

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
        if (!v || v->t != JVal::T::Num || v->num < 0) {
            err = "talpos failo laukas " + k + " — ne neneigiamas skaičius";
            return false;
        }
        o = v->num;
        return true;
    };
    Capacity c;
    c.headroom = out.headroom;
    c.source = out.source;
    if (!str("gpu.name", c.gpu_name) || !str("gpu.uuid", c.gpu_uuid) || !str("gpu.driver", c.driver) || !str("format", c.format) ||
        !str("measured_utc", c.measured_utc))
        return false;
    double v = 0;
    if (!num(at(root, "engines.nvdec.mpix_s"), "engines.nvdec.mpix_s", c.nvdec_mpix_s, false)) return false;
    if (!num(at(root, "engines.nvenc.mpix_s"), "engines.nvenc.mpix_s", c.nvenc_mpix_s, false)) return false;
    if (!num(at(root, "engines.nvenc.sessions"), "engines.nvenc.sessions", v, true)) return false;
    c.nvenc_sessions = static_cast<int>(std::llround(v));
    if (!num(at(root, "gpu.vram_bytes"), "gpu.vram_bytes", v, true)) return false;
    c.vram_bytes = static_cast<std::uint64_t>(std::llround(v));
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
        if (!num(field("vram_per_flow_bytes"), p + "vram_per_flow_bytes", v, true)) return false;
        fc.vram_per_flow_bytes = static_cast<std::uint64_t>(std::llround(v));
        if (!num(field("vram_offset_bytes"), p + "vram_offset_bytes", v, true)) return false;
        fc.vram_offset_bytes = static_cast<std::uint64_t>(std::llround(v));
        if (!num(field("knee"), p + "knee", v, true)) return false;
        fc.knee = static_cast<int>(std::llround(v));
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
                    ": talpa galėjo pasikeisti, permatuokite (scripts/linux/70-capacity.sh, tests/perf/capacity_file.py)");
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

}  // namespace mxlv::runtime
