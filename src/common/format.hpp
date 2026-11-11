// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Flow'o video formatas (planas docs/planai/2026-11-11-production-grade.md §P2, sprendimas K5), header-only, be MXL:
// rastras (gfx::Raster: plotis, aukštis, skenavimas, alfa) + kadrų dažnis + 4:2:0 chromos padėtis.
//  - grain'o dažnis: interlaced = 2 × kadrų dažnis (grain = laukas), progresyvus = kadrų dažnis (grain = kadras);
//  - MXL flow apraše `grain_rate` interlaced atveju yra KADRŲ dažnis (MXL grain'ų ritmą dvigubina pats);
//  - palaikoma: 1280×720, 1920×1080, 3840×2160 progresyviai p25 / p50 / p29,97 / p59,94; interlaced (tik TFF) —
//    tik 1080i50 ir 1080i59,94. Kita — atmetama su aiškia klaida (netestuojama = nepalaikoma).
// Dažnių aritmetika — timing/index.hpp (index_of / time_of), čia tik garso mėginių ribos grain'ui.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "gfx/raster.hpp"
#include "timing/index.hpp"

namespace vramflow {

using Rational = mxlv::timing::Rational;

// 4:2:0 chromos padėtis (H.264 / HEVC chroma_sample_loc_type). Įgyvendintas tik 0 tipas (kairė, tarp eilučių);
// kiti — format_supported klaida.
enum class ChromaSiting420 { Type0Left, Type1Center, Type2TopLeft };

struct VideoFormat {
    gfx::Raster raster;   // w, h, interlaced (TFF), alpha
    Rational frame_rate;  // 25/1, 50/1, 30000/1001, 60000/1001
    ChromaSiting420 siting = ChromaSiting420::Type0Left;
    constexpr Rational grain_rate() const {
        return raster.interlaced ? Rational{2 * frame_rate.num, frame_rate.den} : frame_rate;
    }
};

constexpr bool same_rate(Rational a, Rational b) { return a.num * b.den == b.num * a.den && a.den != 0 && b.den != 0; }
constexpr bool operator==(const VideoFormat& a, const VideoFormat& b) {
    return a.raster == b.raster && same_rate(a.frame_rate, b.frame_rate) && a.siting == b.siting;
}

constexpr VideoFormat kFmt1080i50{{1920, 1080, true, false}, {25, 1}};

namespace detail_fmt {
struct RateName {
    const char* name;
    Rational frame;  // kadrų dažnis
};
// Progresyvūs vardai: kadrų dažnis; interlaced vardai: LAUKŲ dažnis (1080i50 = 25 kadrai/s).
inline constexpr RateName kProgRates[] = {{"25", {25, 1}}, {"50", {50, 1}}, {"29.97", {30000, 1001}}, {"59.94", {60000, 1001}}};
inline constexpr RateName kIntRates[] = {{"50", {25, 1}}, {"59.94", {30000, 1001}}};
struct Size {
    int w, h;
    bool interlaced_ok;
};
inline constexpr Size kSizes[] = {{1280, 720, false}, {1920, 1080, true}, {3840, 2160, false}};
struct Rates {
    const RateName* b;
    const RateName* e;
    constexpr const RateName* begin() const { return b; }
    constexpr const RateName* end() const { return e; }
};
constexpr Rates rates(bool interlaced) {
    return interlaced ? Rates{kIntRates, kIntRates + 2} : Rates{kProgRates, kProgRates + 4};
}
}  // namespace detail_fmt

// Garso mėginio indeksas, kuriuo prasideda grain'as `index`: round(index · sr · den / num), pusė į viršų, sveikaisiais
// skaičiais (be slankaus kablelio dreifo). Grain'o mėginiai — [sample_begin(i), sample_begin(i + 1)).
constexpr std::uint64_t sample_begin(Rational grain_rate, std::uint64_t index, std::uint32_t sample_rate = 48000) {
    using mxlv::timing::i128;
    if (grain_rate.num <= 0 || grain_rate.den <= 0) return 0;
    return static_cast<std::uint64_t>((i128{index} * sample_rate * grain_rate.den * 2 + grain_rate.num) / (i128{2} * grain_rate.num));
}
// Ilgiausias grain'as mėginiais (rašymo buferio dydžiui): ceil(sr · den / num).
constexpr std::uint32_t max_samples_per_grain(Rational grain_rate, std::uint32_t sample_rate = 48000) {
    if (grain_rate.num <= 0 || grain_rate.den <= 0) return 0;
    return static_cast<std::uint32_t>((std::int64_t{sample_rate} * grain_rate.den + grain_rate.num - 1) / grain_rate.num);
}

// Vardas: "<h><i|p><dažnis>", pvz. "1080i50", "720p59.94". Nepalaikomam formatui — "<w>x<h><i|p><num>/<den>".
inline std::string format_name(const VideoFormat& f) {
    const auto tbl = detail_fmt::rates(f.raster.interlaced);
    for (const auto& s : detail_fmt::kSizes)
        if (s.w == f.raster.w && s.h == f.raster.h)
            for (const auto& r : tbl)
                if (same_rate(r.frame, f.frame_rate)) return std::to_string(f.raster.h) + (f.raster.interlaced ? "i" : "p") + r.name;
    return std::to_string(f.raster.w) + "x" + std::to_string(f.raster.h) + (f.raster.interlaced ? "i" : "p") +
           std::to_string(f.frame_rate.num) + "/" + std::to_string(f.frame_rate.den);
}

// Ar formatas palaikomas (K5 sąrašas, 0 tipo chromos padėtis). why — priežastis.
inline bool format_supported(const VideoFormat& f, std::string& why) {
    if (f.siting != ChromaSiting420::Type0Left) {
        why = "nepalaikoma 4:2:0 chromos padėtis (įgyvendintas tik 0 tipas — kairė, tarp eilučių)";
        return false;
    }
    if (!gfx::raster_ok(f.raster)) { why = "netinkamas rastras"; return false; }
    const detail_fmt::Size* size = nullptr;
    for (const auto& s : detail_fmt::kSizes)
        if (s.w == f.raster.w && s.h == f.raster.h) size = &s;
    if (!size) {
        why = "nepalaikomas rastras " + std::to_string(f.raster.w) + "x" + std::to_string(f.raster.h) +
              " (palaikoma 1280x720, 1920x1080, 3840x2160)";
        return false;
    }
    if (f.raster.interlaced && !size->interlaced_ok) { why = "interlaced palaikomas tik 1080 eilučių rastrui"; return false; }
    const auto tbl = detail_fmt::rates(f.raster.interlaced);
    for (const auto& r : tbl)
        if (same_rate(r.frame, f.frame_rate)) return true;
    why = "nepalaikomas kadrų dažnis " + std::to_string(f.frame_rate.num) + "/" + std::to_string(f.frame_rate.den) +
          (f.raster.interlaced ? " (interlaced: 25/1, 30000/1001)" : " (progresyvus: 25/1, 50/1, 30000/1001, 60000/1001)");
    return false;
}

// "1080i50", "1080p25", "720p50", "2160p50", "1080p59.94" … (alpha = false). Klaida → false ir tekstas į err.
inline bool parse_format(const std::string& s, VideoFormat& out, std::string& err) {
    const auto bad = [&](const std::string& why) {
        err = "netinkamas formatas '" + s + "': " + why +
              " (palaikoma: 720p25|50|29.97|59.94, 1080i50|59.94, 1080p25|50|29.97|59.94, 2160p25|50|29.97|59.94)";
        return false;
    };
    std::size_t k = 0;
    while (k < s.size() && s[k] >= '0' && s[k] <= '9') ++k;
    if (k == 0 || k >= s.size() || k > 4) return bad("tikimasi <aukštis><i|p><dažnis>");
    const char scan = s[k];
    if (scan != 'i' && scan != 'p') return bad("skenavimas turi būti i arba p");
    const int h = std::atoi(s.substr(0, k).c_str());
    const std::string rate = s.substr(k + 1);
    const detail_fmt::Size* size = nullptr;
    for (const auto& z : detail_fmt::kSizes)
        if (z.h == h) size = &z;
    if (!size) return bad("nepalaikomas aukštis " + std::to_string(h));
    const bool inter = scan == 'i';
    if (inter && !size->interlaced_ok) return bad("interlaced tik 1080 eilučių");
    const auto tbl = detail_fmt::rates(inter);
    for (const auto& r : tbl)
        if (rate == r.name) {
            out = VideoFormat{{size->w, size->h, inter, false}, r.frame};
            err.clear();
            return true;
        }
    return bad("nepalaikomas dažnis '" + rate + "'");
}

namespace detail_fmt {
// "grain_rate":{"numerator":N,"denominator":D} — ieškoma objekto viduje (ne pirmo "numerator" visame tekste).
inline bool grain_rate_of(const std::string& j, Rational& r) {
    const std::size_t g = j.find("\"grain_rate\"");
    if (g == std::string::npos) return false;
    const std::size_t open = j.find('{', g), close = j.find('}', g);
    if (open == std::string::npos || close == std::string::npos || open > close) return false;
    const std::string obj = j.substr(open, close - open + 1);
    const char* n = gfx::detail::json_value(obj, "numerator");
    if (!n) return false;
    char* e = nullptr;
    const long long num = std::strtoll(n, &e, 10);
    if (e == n || num <= 0) return false;
    long long den = 1;
    if (const char* d = gfx::detail::json_value(obj, "denominator")) {
        den = std::strtoll(d, &e, 10);
        if (e == d || den <= 0) return false;
    }
    r = Rational{num, den};
    return true;
}
}  // namespace detail_fmt

// Formatas iš MXL video flow aprašo: rastras (raster_from_flow_def) + grain_rate (kadrų dažnis). Palaikymo NEtikrina
// (grafikos sluoksniai gali būti bet kokio dydžio) — tam format_supported.
inline bool format_from_flow_def(const std::string& json, VideoFormat& out, std::string& err) {
    VideoFormat f{};
    if (!gfx::raster_from_flow_def(json, f.raster, err)) return false;
    if (!detail_fmt::grain_rate_of(json, f.frame_rate)) { err = "flow apraše nėra tinkamo grain_rate"; return false; }
    out = f;
    return true;
}

// MXL video flow aprašas (v210, arba v210a, jei f.raster.alpha). label — description ir label; group_hint —
// urn:x-nmos:tag:grouphint/v1.0 reikšmė (pvz. "vramflow-func-<seed>-1:Video 0").
inline std::string video_flow_def(const std::string& id, const std::string& label, const VideoFormat& f, const std::string& group_hint) {
    const gfx::Raster& r = f.raster;
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"%s","label":"%s",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["%s"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"%s",)"
                  R"("grain_rate":{"numerator":%lld,"denominator":%lld},"frame_width":%d,"frame_height":%d,)"
                  R"("interlace_mode":"%s","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cb","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cr","width":%d,"height":%d,"bit_depth":10}]})",
                  id.c_str(), label.c_str(), label.c_str(), group_hint.c_str(), r.alpha ? "video/v210a" : "video/v210",
                  static_cast<long long>(f.frame_rate.num), static_cast<long long>(f.frame_rate.den), r.w, r.h,
                  r.interlaced ? "interlaced_tff" : "progressive", r.w, r.h, r.w / 2, r.h, r.w / 2, r.h);
    return b;
}

}  // namespace vramflow
