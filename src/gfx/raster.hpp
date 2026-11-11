// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bendras rastro aprašas alfa grafikai ir kompozitoriui (header-only): v210 / v210a grain'o dydžiai, key pakavimas
// (MXL docs/Architecture.md: 32 bitų LE blokai po 3 10 bitų mėginius, bitai 30–31 = 0) ir rastro skaitymas iš flow aprašo.
// Grain'as: fill (v210) iš karto, po jo key (jei alpha). Interlaced: grain = laukas (h/2 eilučių).
#include <cmath>
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

// Rastro geometrija: plotis ir aukštis > 0, plotis lyginis (4:2:2 chroma poros, R7), interlaced aukštis lyginis.
constexpr bool raster_ok(const Raster& r) {
    if (r.w <= 0 || r.h <= 0 || (r.w & 1)) return false;
    if (r.interlaced && (r.h & 1)) return false;
    return true;
}

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
// 8 bitų kanalų tvarka atmintyje: RGBA (baitai R, G, B, A) arba BGRA (B, G, R, A — Chromium / naršyklės, GStreamer BGRA).
enum class PixelOrder { RGBA, BGRA };
struct Layer {
    const std::uint8_t* grain;  // host arba device
    Raster r;
    int x, y;
    float opacity;
};
// Sluoksnio opacity -> o8 (0…255). VIENINTELIS apibrėžimas visiems variantams (etalonas, CPU, CUDA host pusė):
// daugyba VIENO tikslumo (float) aritmetika, tada lround, tada apkirpimas į 0…255. Todėl opacity už [0,1] apkerpama
// (<0 -> 0, >1 -> 255). 0.7f·255.0f = 178.5f -> 179 (double duotų 178.49999… -> 178; tai NEleidžiama).
inline int opacity_to_o8(float op) {
    const float p = op * 255.0f;  // laikinas float kintamasis: jokio išplėsto tikslumo
    const long v = std::lround(p);
    return v < 0 ? 0 : (v > 255 ? 255 : static_cast<int>(v));
}

struct CompositeJob {
    const std::uint8_t* base;
    Raster base_r;
    const Layer* layers;
    int k;
    Raster out_r;
    std::uint8_t* out;
};

// VIENINTELIS composite užduoties patikrinimas visiems variantams (etalonas, CPU, CUDA; R13). Semantika vienoda (R11):
// bent vienas netinkamas job'as -> visas batch'as atmetamas, nieko nerašoma. Taisyklės:
//  - out != nullptr, išėjimo rastras tinkamas, k >= 0, k > 0 -> layers != nullptr;
//  - interlaced išėjimui parity 0 arba 1 (progresyviam ignoruojamas);
//  - bazė (jei yra): tinkamas rastras, tas pats w / h / interlaced kaip išėjimo; base == nullptr leidžiama TIK v210a
//    išėjimui (R12: be bazės fill neapibrėžtas v210 išėjime, nes nėra alfa jam parodyti);
//  - kiekvienas sluoksnis: grain != nullptr, tinkamas rastras, tas pats interlaced, x >= 0 ir lyginis, y >= 0
//    (interlaced -> y lyginis, R1), opacity baigtinis (ne NaN / ±inf).
// Įrenginio specifiniai reikalavimai (lygiavimas, kMaxBatchLayers) tikrinami atskirai paties varianto. why (nebūtinas) —
// pirmos rastos klaidos aprašas.
inline bool composite_job_valid(const CompositeJob& j, int parity, std::string* why = nullptr) {
    auto fail = [&](const char* m) {
        if (why) *why = m;
        return false;
    };
    if (!j.out) return fail("išėjimas nullptr");
    if (!raster_ok(j.out_r)) return fail("netinkamas išėjimo rastras");
    if (j.k < 0) return fail("k < 0");
    if (j.k > 0 && !j.layers) return fail("sluoksniai nullptr");
    if (j.out_r.interlaced && parity != 0 && parity != 1) return fail("paritetas ne 0/1");
    if (j.base) {
        if (!raster_ok(j.base_r)) return fail("netinkamas bazės rastras");
        if (j.base_r.w != j.out_r.w || j.base_r.h != j.out_r.h || j.base_r.interlaced != j.out_r.interlaced)
            return fail("bazės rastras neatitinka išėjimo");
    } else if (!j.out_r.alpha) {
        return fail("be bazės galimas tik v210a išėjimas");  // R12
    }
    for (int l = 0; l < j.k; ++l) {
        const Layer& L = j.layers[l];
        if (!L.grain) return fail("sluoksnio grain nullptr");
        if (!raster_ok(L.r)) return fail("netinkamas sluoksnio rastras");
        if (L.r.interlaced != j.out_r.interlaced) return fail("sluoksnio interlace neatitinka išėjimo");
        if (L.x < 0 || (L.x & 1) || L.y < 0) return fail("sluoksnio x / y neleistinas");
        if (j.out_r.interlaced && (L.y & 1)) return fail("nelyginis y interlaced rastre");  // R1
        if (!std::isfinite(L.opacity)) return fail("opacity ne baigtinis (NaN / ±inf)");
    }
    return true;
}

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
    // Palaikoma TIK progressive (arba nenurodyta) ir interlaced_tff (lyginis indeksas = viršutinis laukas). BFF ir kitos
    // reikšmės atmetamos aiškiai: tylus TFF laukų tvarkos priskyrimas BFF turiniui sukeistų laukus.
    const bool has_im = detail::json_str(json, "interlace_mode", im);
    if (has_im && im != "progressive" && im != "interlaced_tff") {
        err = "nepalaikomas interlace_mode " + im + " (palaikoma: progressive, interlaced_tff)";
        return false;
    }
    r.interlaced = has_im && im == "interlaced_tff";
    if (r.interlaced && (r.h % 2) != 0) { err = "interlaced rastro aukštis nelyginis"; return false; }
    out = r;
    return true;
}

}  // namespace vramflow::gfx
