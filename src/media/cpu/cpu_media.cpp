// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "media/cpu/cpu_media.hpp"

#include <algorithm>
#include <cstring>

namespace mxlv::media::cpu {
namespace {

inline std::uint16_t blend(std::uint16_t s, std::uint16_t l, std::uint8_t a) {
    return static_cast<std::uint16_t>((static_cast<std::uint32_t>(s) * (255u - a) + static_cast<std::uint32_t>(l) * a + 127u) / 255u);
}
inline std::uint8_t to8(int v10) { return static_cast<std::uint8_t>(std::min(255, (v10 + 2) >> 2)); }

// Pilnų grupių skaičius (visi 6 pikseliai pločio viduje) ir naudojamų grupių skaičius (pvz. 1280 = 213 · 6 + 2 → 213 ir
// 214). Pilnos grupės apdorojamos be šakų (kompiliatorius vektorizuoja), dalinė — su x < w sąlyga, lygiavimo — nuliais.
struct Cols {
    int full, used;
    explicit Cols(int w) : full(w / 6), used((w + 5) / 6) {}
};

// Dvi v210 eilutės r0, r1 → jų luma (y0, y1) ir viena NV12 chromos eilė c = mix(a, b) (a — r0, b — r1 mėginys).
template <class Mix>
void pair_rows(const std::uint32_t* r0, const std::uint32_t* r1, std::uint8_t* y0, std::uint8_t* y1, std::uint8_t* c, const Cols& cols,
               int w, Mix mix) {
    for (int g = 0; g < cols.full; ++g) {
        Group a, b;
        unpack_group(r0 + 4 * g, a);
        unpack_group(r1 + 4 * g, b);
        for (int i = 0; i < 6; ++i) {
            y0[6 * g + i] = to8(a.y[i]);
            y1[6 * g + i] = to8(b.y[i]);
        }
        for (int i = 0; i < 3; ++i) {
            c[6 * g + 2 * i] = mix(a.cb[i], b.cb[i]);
            c[6 * g + 2 * i + 1] = mix(a.cr[i], b.cr[i]);
        }
    }
    for (int g = cols.full; g < cols.used; ++g) {  // dalinė grupė: tik x < w
        Group a, b;
        unpack_group(r0 + 4 * g, a);
        unpack_group(r1 + 4 * g, b);
        for (int i = 0; i < 6 && 6 * g + i < w; ++i) {
            y0[6 * g + i] = to8(a.y[i]);
            y1[6 * g + i] = to8(b.y[i]);
        }
        for (int i = 0; i < 3 && 6 * g + 2 * i < w; ++i) {
            c[6 * g + 2 * i] = mix(a.cb[i], b.cb[i]);
            c[6 * g + 2 * i + 1] = mix(a.cr[i], b.cr[i]);
        }
    }
}

}  // namespace

void logo_field(const std::uint32_t* in, std::uint32_t* out, const LogoPlanes& L, int parity) {
    logo_field(in, out, L, kDim1080, parity);
}

void logo_field(const std::uint32_t* in, std::uint32_t* out, const LogoPlanes& L, Dim d, int field) {
    const Geom G = geom_of(d, field);
    // 1) visas grain'as — viena kopija (už logotipo ribų bitas į bitą, įskaitant bitus 30–31)
    if (out != in) std::memcpy(out, in, G.grain_words() * 4);
    // 2) blend tik logotipo stačiakampyje: grain'o eilutės, kurių kadro eilutė fy = G.row(j) patenka į [Y, Y+lh)
    const int fy0 = std::max(L.y, 0), fy1 = std::min(L.y + L.lh, d.h);
    const int x0 = std::max(L.x, 0), x1 = std::min(L.x + L.lw, d.w);
    if (fy0 >= fy1 || x0 >= x1) return;
    const int g0 = x0 / 6, g1 = (x1 + 5) / 6;
    const int lcw = L.lw / 2;
    // pirmoji kadro eilutė >= fy0, priklausanti šiam grain'ui (laukui — tinkamo pariteto, kadrui — bet kuri)
    const int first = G.step == 1 ? fy0 : fy0 + ((fy0 - G.par) & 1);
    for (int fy = first; fy < fy1; fy += G.step) {
        const int j = (fy - G.par) / G.step;
        const int ly = fy - L.y;
        const std::uint16_t* ly10 = L.y10 + ly * L.lw;
        const std::uint8_t* la8 = L.a8 + ly * L.lw;
        const std::uint16_t* lcb = L.cb10 + ly * lcw;
        const std::uint16_t* lcr = L.cr10 + ly * lcw;
        const std::uint8_t* lac = L.ac8 + ly * lcw;
        std::uint32_t* row = out + static_cast<std::size_t>(j) * G.words;
        for (int g = g0; g < g1; ++g) {
            std::uint32_t* w = row + 4 * g;
            Group grp;
            unpack_group(w, grp);
            const std::uint32_t hi[4] = {w[0], w[1], w[2], w[3]};
            for (int k = 0; k < 6; ++k) {
                const int x = 6 * g + k;
                if (x < x0 || x >= x1) continue;
                const int lx = x - L.x;
                grp.y[k] = blend(grp.y[k], ly10[lx], la8[lx]);
            }
            for (int c = 0; c < 3; ++c) {
                const int x = 6 * g + 2 * c;
                if (x < x0 || x >= x1) continue;
                const int cx = (x - L.x) / 2;
                grp.cb[c] = blend(grp.cb[c], lcb[cx], lac[cx]);
                grp.cr[c] = blend(grp.cr[c], lcr[cx], lac[cx]);
            }
            pack_group(grp, w, hi);
        }
    }
}

void weave_nv12(const std::uint32_t* top, const std::uint32_t* bottom, std::uint8_t* y, std::uint8_t* uv, int pitch_y,
                int pitch_uv) {
    weave_nv12(top, bottom, y, uv, pitch_y, pitch_uv, kDim1080, true);
}

void weave_nv12(const std::uint32_t* top, const std::uint32_t* bottom, std::uint8_t* y, std::uint8_t* uv, int pitch_y,
                int pitch_uv, Dim d, bool interlaced) {
    const int words = v210_words_per_line(d.w);
    const Cols cols(d.w);
    if (!interlaced) {
        // progresyvus kadras: UV eilė k = (c[2k] + c[2k+1] + 4) >> 3 (10 → 8 bitų kartu su vidurkiu; 0 tipo padėtis)
        const auto mix = [](int a, int b) { return static_cast<std::uint8_t>(std::min(255, (a + b + 4) >> 3)); };
        for (int k = 0; k < d.h / 2; ++k) {
            const std::uint32_t* r0 = top + static_cast<std::size_t>(2 * k) * words;
            pair_rows(r0, r0 + words, y + static_cast<std::size_t>(2 * k) * pitch_y, y + static_cast<std::size_t>(2 * k + 1) * pitch_y,
                      uv + static_cast<std::size_t>(k) * pitch_uv, cols, d.w, mix);
        }
        return;
    }
    const std::uint32_t* f[2] = {top, bottom};
    for (int p = 0; p < 2; ++p) {
        // lauko chroma: viršutinis (3,1), apatinis (1,3), dvi apvalinimo pakopos (10 bitų, tada 8)
        const int w0 = p == 0 ? 3 : 1, w1 = p == 0 ? 1 : 3;
        const auto mix = [w0, w1](int a, int b) { return to8((w0 * a + w1 * b + 2) >> 2); };
        for (int k = 0; k < d.h / 4; ++k) {  // lauko chromos eilė k: lauko eilutės 2k, 2k+1
            const std::uint32_t* r0 = f[p] + static_cast<std::size_t>(2 * k) * words;
            pair_rows(r0, r0 + words, y + static_cast<std::size_t>(2 * (2 * k) + p) * pitch_y,
                      y + static_cast<std::size_t>(2 * (2 * k + 1) + p) * pitch_y, uv + static_cast<std::size_t>(2 * k + p) * pitch_uv,
                      cols, d.w, mix);
        }
    }
}

namespace {
// Bendras split'as: chroma mėginys (UV eilė r, chroma pikselis cx) pateikiamas funkcijomis — NV12 ir planar keliams.
// Kiekvienai grain'o eilutei: c = (A·c[ra] + B·c[rb] + rnd) >> sh.
template <class CbAt, class CrAt>
void split_impl(const std::uint8_t* y, int pitch_y, Dim d, int field, ChromaMode m, std::uint32_t* out, CbAt cb_at, CrAt cr_at) {
    static const int wa[2][2] = {{1, 5}, {3, 7}};  // [p][j nelyginis?] koeficientas „prieš" (k−1 arba k)
    static const int wb[2][2] = {{7, 3}, {5, 1}};  // koeficientas „po" (k arba k+1)
    const Geom G = geom_of(d, field);
    const Cols cols(d.w);
    const int K = d.h / 2;  // kadro chromos eilių
    for (int j = 0; j < G.lines; ++j) {
        const int f = G.row(j);
        const std::uint8_t* ys = y + static_cast<std::size_t>(f) * pitch_y;
        int ra = 0, rb = 0, A = 0, B = 0, rnd = 0, sh = 0;
        if (m == ChromaMode::FrameSited) {
            // kadro chroma (0 tipas): 3·c[k] + c[kaimynė], 10 bitų be apvalinimo
            const int k = f >> 1;
            ra = k;
            rb = (f & 1) ? std::min(k + 1, K - 1) : std::max(k - 1, 0);
            A = 3;
            B = 1;
        } else {
            // lauko chroma: lauko chromos eilė c[k] = UV eilė 2k + p; svoriai aštuntosiomis (§6.4)
            const int kc = G.lines / 2;
            const int k = j >> 1, odd = j & 1;
            ra = 2 * (odd ? k : std::max(k - 1, 0)) + field;
            rb = 2 * (odd ? std::min(k + 1, kc - 1) : k) + field;
            A = wa[field][odd];
            B = wb[field][odd];
            rnd = 1;
            sh = 1;
        }
        std::uint32_t* row = out + static_cast<std::size_t>(j) * G.words;
        for (int g = 0; g < cols.full; ++g) {
            Group grp;
            for (int i = 0; i < 6; ++i) grp.y[i] = static_cast<std::uint16_t>(ys[6 * g + i] << 2);
            for (int i = 0; i < 3; ++i) {
                const int cx = 3 * g + i;
                grp.cb[i] = static_cast<std::uint16_t>((A * cb_at(ra, cx) + B * cb_at(rb, cx) + rnd) >> sh);
                grp.cr[i] = static_cast<std::uint16_t>((A * cr_at(ra, cx) + B * cr_at(rb, cx) + rnd) >> sh);
            }
            pack_group(grp, row + 4 * g);
        }
        for (int g = cols.full; g < G.groups; ++g) {  // dalinė grupė (x < w) ir lygiavimo grupės (0)
            Group grp{};
            for (int i = 0; i < 6 && 6 * g + i < d.w; ++i) grp.y[i] = static_cast<std::uint16_t>(ys[6 * g + i] << 2);
            for (int i = 0; i < 3 && 6 * g + 2 * i < d.w; ++i) {
                const int cx = 3 * g + i;
                grp.cb[i] = static_cast<std::uint16_t>((A * cb_at(ra, cx) + B * cb_at(rb, cx) + rnd) >> sh);
                grp.cr[i] = static_cast<std::uint16_t>((A * cr_at(ra, cx) + B * cr_at(rb, cx) + rnd) >> sh);
            }
            pack_group(grp, row + 4 * g);
        }
    }
}
bool split_args_ok(Dim d, int field, ChromaMode m) {
    if (!dim_ok(d, field >= 0) || field > 1) return false;
    return !(m == ChromaMode::FieldSited && field < 0);
}
}  // namespace

void split_yuv420p(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v, int pitch_y, int pitch_c, int parity,
                   std::uint32_t* out) {
    split_yuv420p(y, u, v, pitch_y, pitch_c, kDim1080, parity, ChromaMode::FieldSited, out);
}

void split_yuv420p(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v, int pitch_y, int pitch_c, Dim d, int field,
                   ChromaMode m, std::uint32_t* out) {
    if (!split_args_ok(d, field, m)) return;
    split_impl(y, pitch_y, d, field, m, out, [&](int r, int cx) { return int{u[static_cast<std::size_t>(r) * pitch_c + cx]}; },
               [&](int r, int cx) { return int{v[static_cast<std::size_t>(r) * pitch_c + cx]}; });
}

void split_nv12(const std::uint8_t* y, const std::uint8_t* uv, int pitch_y, int pitch_uv, int parity, std::uint32_t* out) {
    split_nv12(y, uv, pitch_y, pitch_uv, kDim1080, parity, ChromaMode::FieldSited, out);
}

void split_nv12(const std::uint8_t* y, const std::uint8_t* uv, int pitch_y, int pitch_uv, Dim d, int field, ChromaMode m,
                std::uint32_t* out) {
    if (!split_args_ok(d, field, m)) return;
    split_impl(y, pitch_y, d, field, m, out, [&](int r, int cx) { return int{uv[static_cast<std::size_t>(r) * pitch_uv + 2 * cx]}; },
               [&](int r, int cx) { return int{uv[static_cast<std::size_t>(r) * pitch_uv + 2 * cx + 1]}; });
}

void split_nv12_progressive(const std::uint8_t* y, const std::uint8_t* uv, int pitch_y, int pitch_uv, Dim d, std::uint32_t* out) {
    split_nv12(y, uv, pitch_y, pitch_uv, d, -1, ChromaMode::FrameSited, out);
}

void gen_field(const GenParams& g, std::uint64_t index, std::uint32_t* out) { gen_field(g, index, kDim1080, true, out); }

void gen_field(const GenParams& g, std::uint64_t index, Dim d, bool interlaced, std::uint32_t* out) {
    const Geom G = geom_of(d, interlaced ? static_cast<int>(index & 1) : -1);
    const Cols cols(d.w);
    for (int j = 0; j < G.lines; ++j) {
        const int fy = G.row(j);
        std::uint32_t* row = out + static_cast<std::size_t>(j) * G.words;
        for (int gi = 0; gi < cols.full; ++gi) {
            Group grp;
            for (int i = 0; i < 6; ++i) grp.y[i] = gen_y(g, index, 6 * gi + i, fy);
            for (int i = 0; i < 3; ++i) gen_c(g, index, 6 * gi + 2 * i, fy, grp.cb[i], grp.cr[i]);
            pack_group(grp, row + 4 * gi);
        }
        for (int gi = cols.full; gi < G.groups; ++gi) {  // dalinė grupė ir lygiavimo grupės: už pločio — 0
            Group grp{};
            for (int i = 0; i < 6 && 6 * gi + i < d.w; ++i) grp.y[i] = gen_y(g, index, 6 * gi + i, fy);
            for (int i = 0; i < 3 && 6 * gi + 2 * i < d.w; ++i) gen_c(g, index, 6 * gi + 2 * i, fy, grp.cb[i], grp.cr[i]);
            pack_group(grp, row + 4 * gi);
        }
    }
}

}  // namespace mxlv::media::cpu
