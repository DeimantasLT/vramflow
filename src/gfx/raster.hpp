// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bendras rastro aprašas alfa grafikai ir kompozitoriui (header-only): v210 / v210a grain'o dydžiai, key pakavimas
// (MXL docs/Architecture.md: 32 bitų LE blokai po 3 10 bitų mėginius, bitai 30–31 = 0) ir rastro skaitymas iš flow aprašo.
// Grain'as: fill (v210) iš karto, po jo key (jei alpha). Interlaced: grain = laukas (h/2 eilučių).
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vramflow::gfx {

struct Raster {
    int w, h;         // h — kadro aukštis
    bool interlaced;  // grain = laukas
    bool alpha;       // v210a (fill + key)
};
constexpr Raster kHd1080i{1920, 1080, true, false};

constexpr std::size_t fill_line_bytes(int w) { return static_cast<std::size_t>((w + 47) / 48) * 128; }
constexpr std::size_t key_line_bytes(int w) { return static_cast<std::size_t>((w + 2) / 3) * 4; }
constexpr int grain_lines(const Raster& r) { return r.interlaced ? r.h / 2 : r.h; }
constexpr std::size_t fill_bytes(const Raster& r) { return fill_line_bytes(r.w) * static_cast<std::size_t>(grain_lines(r)); }
constexpr std::size_t key_bytes(const Raster& r) { return r.alpha ? key_line_bytes(r.w) * static_cast<std::size_t>(grain_lines(r)) : 0; }
constexpr std::size_t grain_bytes(const Raster& r) { return fill_bytes(r) + key_bytes(r); }

constexpr bool operator==(const Raster& a, const Raster& b) {
    return a.w == b.w && a.h == b.h && a.interlaced == b.interlaced && a.alpha == b.alpha;
}
constexpr bool operator!=(const Raster& a, const Raster& b) { return !(a == b); }

// Key mėginys x (10 bitų): žodis x/3, postūmis 10·(x%3). key_set palieka kitus bitus nepaliestus.
constexpr std::uint32_t key_get(const std::uint32_t* key_line, int x) {
    return (key_line[x / 3] >> (10 * (x % 3))) & 0x3FFu;
}
constexpr void key_set(std::uint32_t* key_line, int x, std::uint32_t v) {
    const int s = 10 * (x % 3);
    std::uint32_t& wd = key_line[x / 3];
    wd = (wd & ~(0x3FFu << s)) | ((v & 0x3FFu) << s);
}

// Bendri tipai visiems įgyvendinimams (etalonas, CPU, CUDA).
enum class AlphaIn { Straight, Premultiplied };
struct Layer {
    const std::uint8_t* grain;  // host arba device
    Raster r;
    int x, y;
    float opacity;
};
struct CompositeJob {
    const std::uint8_t* base;
    Raster base_r;
    const Layer* layers;
    int k;
    Raster out_r;
    std::uint8_t* out;
};

namespace detail {
// Randa `"key"` ir grąžina nuorodą į reikšmės pradžią (po ':' ir tarpų) arba nullptr. Pakanka plokščių MXL flow aprašų
// raktų (frame_width, media_type, ...); neieško įdėtuose objektuose pagal kelią.
inline const char* json_value(const std::string& j, const char* key) {
    const std::string needle = std::string("\"") + key + "\"";
    std::size_t p = j.find(needle);
    while (p != std::string::npos) {
        std::size_t q = p + needle.size();
        while (q < j.size() && (j[q] == ' ' || j[q] == '\t' || j[q] == '\n' || j[q] == '\r')) ++q;
        if (q < j.size() && j[q] == ':') {
            ++q;
            while (q < j.size() && (j[q] == ' ' || j[q] == '\t' || j[q] == '\n' || j[q] == '\r')) ++q;
            return j.c_str() + q;
        }
        p = j.find(needle, p + 1);
    }
    return nullptr;
}
inline bool json_int(const std::string& j, const char* key, int& out) {
    const char* v = json_value(j, key);
    if (!v) return false;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || n <= 0 || n > 65536) return false;
    out = static_cast<int>(n);
    return true;
}
inline bool json_str(const std::string& j, const char* key, std::string& out) {
    const char* v = json_value(j, key);
    if (!v || *v != '"') return false;
    const char* e = std::strchr(v + 1, '"');
    if (!e) return false;
    out.assign(v + 1, e);
    return true;
}
}  // namespace detail

// Rastras iš MXL video flow aprašo (media_type video/v210 | video/v210a, frame_width/height, interlace_mode).
inline bool raster_from_flow_def(const std::string& json, Raster& out, std::string& err) {
    std::string mt;
    if (!detail::json_str(json, "media_type", mt)) { err = "flow apraše nėra media_type"; return false; }
    if (mt != "video/v210" && mt != "video/v210a") { err = "nepalaikomas media_type " + mt; return false; }
    Raster r{};
    r.alpha = mt == "video/v210a";
    if (!detail::json_int(json, "frame_width", r.w)) { err = "flow apraše nėra frame_width"; return false; }
    if (!detail::json_int(json, "frame_height", r.h)) { err = "flow apraše nėra frame_height"; return false; }
    std::string im;
    r.interlaced = detail::json_str(json, "interlace_mode", im) && im != "progressive";
    if (r.interlaced && (r.h % 2) != 0) { err = "interlaced rastro aukštis nelyginis"; return false; }
    out = r;
    return true;
}

}  // namespace vramflow::gfx
