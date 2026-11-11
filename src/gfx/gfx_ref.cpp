// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Nepriklausomas etalonas (žr. gfx_ref.hpp). Viskas eina per int / uint64_t mėginius; jokių SIMD ar kešų.
#include "gfx/gfx_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "media/media_ref.hpp"
#include "media/v210.hpp"

namespace vramflow::gfx::ref {
namespace {

using u64 = std::uint64_t;
constexpr int A = kAlphaScale;

// Vienos eilutės mėginiai: y[w], cb/cr[w/2], key[w] (key tik v210a).
struct Row {
    std::vector<int> y, cb, cr, key;
    explicit Row(int w) : y(w, 64), cb(w / 2, 512), cr(w / 2, 512), key(w, 64) {}
};

std::uint32_t load_word(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
void store_word(std::uint8_t* p, std::uint32_t v) { std::memcpy(p, &v, 4); }

// v210 fill eilutė -> mėginiai (tik tikri: x < w, chroma < w/2).
void unpack_fill_line(const std::uint8_t* line, int w, Row& row) {
    const int groups = (w + 5) / 6;
    for (int g = 0; g < groups; ++g) {
        std::uint32_t words[4];
        for (int i = 0; i < 4; ++i) words[i] = load_word(line + g * 16 + i * 4);
        mxlv::media::Group grp;
        mxlv::media::unpack_group(words, grp);
        for (int k = 0; k < 6; ++k)
            if (g * 6 + k < w) row.y[g * 6 + k] = grp.y[k];
        for (int j = 0; j < 3; ++j)
            if (g * 3 + j < w / 2) {
                row.cb[g * 3 + j] = grp.cb[j];
                row.cr[g * 3 + j] = grp.cr[j];
            }
    }
}

void unpack_key_line(const std::uint8_t* line, int w, Row& row) {
    for (int x = 0; x < w; ++x) {
        const std::uint32_t wd = load_word(line + (x / 3) * 4);
        row.key[x] = static_cast<int>(key_get(&wd, x % 3));
    }
}

// Mėginiai -> v210 fill eilutė. Nenaudoti mėginiai lieka 0 (buferis iš anksto nunulintas).
void pack_fill_line(std::uint8_t* line, int w, const Row& row) {
    const int groups = (w + 5) / 6;
    for (int g = 0; g < groups; ++g) {
        mxlv::media::Group grp{};
        for (int k = 0; k < 6; ++k)
            if (g * 6 + k < w) grp.y[k] = static_cast<std::uint16_t>(row.y[g * 6 + k]);
        for (int j = 0; j < 3; ++j)
            if (g * 3 + j < w / 2) {
                grp.cb[j] = static_cast<std::uint16_t>(row.cb[g * 3 + j]);
                grp.cr[j] = static_cast<std::uint16_t>(row.cr[g * 3 + j]);
            }
        std::uint32_t words[4];
        mxlv::media::pack_group(grp, words);
        for (int i = 0; i < 4; ++i) store_word(line + g * 16 + i * 4, words[i]);
    }
}

void pack_key_line(std::uint8_t* line, int w, const Row& row) {
    for (int wi = 0; wi < (w + 2) / 3; ++wi) {
        std::uint32_t wd = 0;  // bitai 30–31 ir nenaudoti mėginiai = 0
        for (int s = 0; s < 3; ++s)
            if (wi * 3 + s < w) key_set(&wd, s, static_cast<std::uint32_t>(row.key[wi * 3 + s]));
        store_word(line + wi * 4, wd);
    }
}

bool raster_ok(const Raster& r) {
    if (r.w <= 0 || r.h <= 0 || (r.w & 1)) return false;  // R7
    if (r.interlaced && (r.h & 1)) return false;
    return true;
}

int clamp_fill(int v) { return std::clamp(v, 4, 1019); }

}  // namespace

bool rgba_to_v210a(const std::uint8_t* rgba, int w, int h, std::size_t stride, AlphaIn mode, const Raster& r,
                   int parity, std::uint8_t* grain) {
    if (!rgba || !grain || !raster_ok(r) || !r.alpha) return false;
    if (r.w != w || r.h != h) return false;
    if (stride < static_cast<std::size_t>(w) * 4) return false;
    if (r.interlaced && parity != 0 && parity != 1) return false;

    std::memset(grain, 0, grain_bytes(r));
    std::uint8_t* const key_plane = grain + fill_bytes(r);
    const int lines = grain_lines(r);
    for (int j = 0; j < lines; ++j) {
        const int frame_row = r.interlaced ? 2 * j + parity : j;
        const std::uint8_t* src = rgba + static_cast<std::size_t>(frame_row) * stride;
        Row row(w);
        std::vector<int> a8(w), cb(w), cr(w);  // cb/cr — pikselio (ne poros) reikšmės
        for (int x = 0; x < w; ++x) {
            int c[3] = {src[4 * x], src[4 * x + 1], src[4 * x + 2]};
            const int a = src[4 * x + 3];
            a8[x] = a;
            if (mode == AlphaIn::Premultiplied)
                for (int& v : c) v = (a == 0) ? 0 : std::min(255, (v * 255 + a / 2) / a);
            int y;
            mxlv::media::ref::rgb_to_ycbcr10(c[0], c[1], c[2], y, cb[x], cr[x]);
            row.y[x] = y;
            row.key[x] = 64 + (a * A + 127) / 255;
        }
        // Chroma pora (2i, 2i+1): alfa svertinis vidurkis; a0 + a1 == 0 -> 512.
        for (int i = 0; i < w / 2; ++i) {
            const int a0 = a8[2 * i], a1 = a8[2 * i + 1];
            if (a0 + a1 == 0) {
                row.cb[i] = 512;
                row.cr[i] = 512;
            } else {
                row.cb[i] = (a0 * cb[2 * i] + a1 * cb[2 * i + 1] + (a0 + a1) / 2) / (a0 + a1);
                row.cr[i] = (a0 * cr[2 * i] + a1 * cr[2 * i + 1] + (a0 + a1) / 2) / (a0 + a1);
            }
        }
        pack_fill_line(grain + static_cast<std::size_t>(j) * fill_line_bytes(w), w, row);
        pack_key_line(key_plane + static_cast<std::size_t>(j) * key_line_bytes(w), w, row);
    }
    return true;
}

namespace {

// Vienas dedamasis taškas: out = f su apačios (b, ab) pagal v210a formulę. Grąžina {fill, ao}.
struct Mixed { int v; int ao; };
Mixed mix_v210a(int f, int ae, int b, int ab, int def) {
    const u64 n = static_cast<u64>(ae) * A + static_cast<u64>(ab) * (A - ae);
    const int ao = static_cast<int>((n + A / 2) / A);
    if (n == 0) return {def, ao};
    const u64 num = static_cast<u64>(f) * ae * A + static_cast<u64>(b) * ab * (A - ae) + n / 2;
    return {clamp_fill(static_cast<int>(num / n)), ao};
}
int mix_v210(int f, int ae, int b) {
    return clamp_fill(static_cast<int>((static_cast<u64>(b) * (A - ae) + static_cast<u64>(f) * ae + A / 2) / A));
}

}  // namespace

bool composite(const std::uint8_t* base, const Raster& base_r, const Layer* layers, int k, const Raster& out_r,
               int parity, std::uint8_t* out) {
    if (!out || !raster_ok(out_r) || k < 0 || (k > 0 && !layers)) return false;
    if (out_r.interlaced && parity != 0 && parity != 1) return false;
    if (base && (!raster_ok(base_r) || base_r.w != out_r.w || base_r.h != out_r.h || base_r.interlaced != out_r.interlaced))
        return false;
    for (int l = 0; l < k; ++l) {
        const Layer& L = layers[l];
        if (!L.grain || !raster_ok(L.r) || L.r.interlaced != out_r.interlaced) return false;
        if (L.x < 0 || (L.x & 1) || L.y < 0) return false;
        if (out_r.interlaced && (L.y & 1)) return false;  // R1
        if (std::isnan(L.opacity)) return false;
    }

    const int w = out_r.w;
    const int lines = grain_lines(out_r);
    std::memset(out, 0, grain_bytes(out_r));
    std::uint8_t* const out_key = out + fill_bytes(out_r);

    for (int r = 0; r < lines; ++r) {
        // 1. Pradinė būsena: fill (Y/Cb/Cr) ir sukaupta alfa ab[x] ∈ 0…A.
        Row st(w);  // numatyta: Y 64, C 512 (bazė none)
        std::vector<int> ab(w, 0);
        if (base) {
            unpack_fill_line(base + static_cast<std::size_t>(r) * fill_line_bytes(w), w, st);
            if (base_r.alpha && out_r.alpha) {
                unpack_key_line(base + fill_bytes(base_r) + static_cast<std::size_t>(r) * key_line_bytes(w), w, st);
                for (int x = 0; x < w; ++x) ab[x] = std::clamp(st.key[x] - 64, 0, A);
            } else {
                std::fill(ab.begin(), ab.end(), A);  // v210 bazė nepermatoma; v210 išėjimui ab nenaudojamas
            }
        }

        // 2. Sluoksniai iš apačios į viršų.
        for (int l = 0; l < k; ++l) {
            const Layer& L = layers[l];
            const int ly = r - (out_r.interlaced ? L.y / 2 : L.y);  // sluoksnio (lauko) eilutė
            if (ly < 0 || ly >= grain_lines(L.r)) continue;
            const int cw = std::min(L.r.w, w - L.x);  // nukirpimas; L.x, L.r.w, w lyginiai -> cw lyginis
            if (cw <= 0) continue;

            Row lr(L.r.w);
            unpack_fill_line(L.grain + static_cast<std::size_t>(ly) * fill_line_bytes(L.r.w), L.r.w, lr);
            if (L.r.alpha)
                unpack_key_line(L.grain + fill_bytes(L.r) + static_cast<std::size_t>(ly) * key_line_bytes(L.r.w), L.r.w, lr);
            const int o8 = std::clamp(static_cast<int>(std::lround(L.opacity * 255.0f)), 0, 255);
            std::vector<int> ae(L.r.w);
            for (int x = 0; x < L.r.w; ++x) {
                const int a = L.r.alpha ? std::clamp(lr.key[x] - 64, 0, A) : A;
                ae[x] = (a * o8 + 127) / 255;
            }

            // Chroma pirma, su ab PRIEŠ šio sluoksnio atnaujinimą (R8).
            for (int i = 0; i < cw / 2; ++i) {
                const int oi = (L.x + 2 * i) / 2;
                const int ac = (ae[2 * i] + ae[2 * i + 1] + 1) >> 1;
                const int abc = (ab[L.x + 2 * i] + ab[L.x + 2 * i + 1] + 1) >> 1;
                if (out_r.alpha) {
                    st.cb[oi] = mix_v210a(lr.cb[i], ac, st.cb[oi], abc, 512).v;
                    st.cr[oi] = mix_v210a(lr.cr[i], ac, st.cr[oi], abc, 512).v;
                } else {
                    st.cb[oi] = mix_v210(lr.cb[i], ac, st.cb[oi]);
                    st.cr[oi] = mix_v210(lr.cr[i], ac, st.cr[oi]);
                }
            }
            for (int x = 0; x < cw; ++x) {
                const int ox = L.x + x;
                if (out_r.alpha) {
                    const Mixed m = mix_v210a(lr.y[x], ae[x], st.y[ox], ab[ox], 64);
                    st.y[ox] = m.v;
                    ab[ox] = m.ao;
                } else {
                    st.y[ox] = mix_v210(lr.y[x], ae[x], st.y[ox]);
                }
            }
        }

        // 3. Supakuoti.
        pack_fill_line(out + static_cast<std::size_t>(r) * fill_line_bytes(w), w, st);
        if (out_r.alpha) {
            for (int x = 0; x < w; ++x) st.key[x] = ab[x] + 64;
            pack_key_line(out_key + static_cast<std::size_t>(r) * key_line_bytes(w), w, st);
        }
    }
    return true;
}

}  // namespace vramflow::gfx::ref
