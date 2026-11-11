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

}  // namespace

void logo_field(const std::uint32_t* in, std::uint32_t* out, const LogoPlanes& L, int parity) {
    // 1) visas laukas — viena kopija (už logotipo ribų bitas į bitą, įskaitant bitus 30–31)
    if (out != in) std::memcpy(out, in, kFieldBytes);
    // 2) blend tik logotipo stačiakampyje: lauko eilutės, kurių kadro eilutė fy = 2j+parity patenka į [Y, Y+lh)
    const int fy0 = std::max(L.y, 0), fy1 = std::min(L.y + L.lh, kFrameHeight);
    const int x0 = std::max(L.x, 0), x1 = std::min(L.x + L.lw, kWidth);
    if (fy0 >= fy1 || x0 >= x1) return;
    const int g0 = x0 / 6, g1 = (x1 + 5) / 6;
    const int lcw = L.lw / 2;
    for (int fy = fy0 + ((fy0 - parity) & 1); fy < fy1; fy += 2) {
        const int j = (fy - parity) / 2;
        const int ly = fy - L.y;
        const std::uint16_t* ly10 = L.y10 + ly * L.lw;
        const std::uint8_t* la8 = L.a8 + ly * L.lw;
        const std::uint16_t* lcb = L.cb10 + ly * lcw;
        const std::uint16_t* lcr = L.cr10 + ly * lcw;
        const std::uint8_t* lac = L.ac8 + ly * lcw;
        std::uint32_t* row = out + j * kWordsPerLine;
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
    const std::uint32_t* f[2] = {top, bottom};
    for (int p = 0; p < 2; ++p) {
        for (int k = 0; k < kFieldHeight / 2; ++k) {  // chroma eilė k: lauko eilutės 2k, 2k+1
            const std::uint32_t* r0 = f[p] + (2 * k) * kWordsPerLine;
            const std::uint32_t* r1 = r0 + kWordsPerLine;
            std::uint8_t* y0 = y + (2 * (2 * k) + p) * pitch_y;
            std::uint8_t* y1 = y + (2 * (2 * k + 1) + p) * pitch_y;
            std::uint8_t* c = uv + (2 * k + p) * pitch_uv;
            const int w0 = p == 0 ? 3 : 1, w1 = p == 0 ? 1 : 3;
            for (int g = 0; g < kGroupsPerLine; ++g) {
                Group a, b;
                unpack_group(r0 + 4 * g, a);
                unpack_group(r1 + 4 * g, b);
                for (int i = 0; i < 6; ++i) {
                    y0[6 * g + i] = to8(a.y[i]);
                    y1[6 * g + i] = to8(b.y[i]);
                }
                for (int i = 0; i < 3; ++i) {
                    c[6 * g + 2 * i] = to8((w0 * a.cb[i] + w1 * b.cb[i] + 2) >> 2);
                    c[6 * g + 2 * i + 1] = to8((w0 * a.cr[i] + w1 * b.cr[i] + 2) >> 2);
                }
            }
        }
    }
}

namespace {
// Bendras split'as: chroma mėginys (eilutė r, chroma pikselis cx) pateikiamas funkcijomis — NV12 ir planar keliams.
template <class CbAt, class CrAt>
void split_impl(const std::uint8_t* y, int pitch_y, int parity, std::uint32_t* out, CbAt cb_at, CrAt cr_at) {
    static const int wa[2][2] = {{1, 5}, {3, 7}};
    static const int wb[2][2] = {{7, 3}, {5, 1}};
    constexpr int kc = kFieldHeight / 2;
    for (int j = 0; j < kFieldHeight; ++j) {
        const std::uint8_t* ys = y + (2 * j + parity) * pitch_y;
        const int k = j >> 1, odd = j & 1;
        const int ra = 2 * (odd ? k : std::max(k - 1, 0)) + parity;
        const int rb = 2 * (odd ? std::min(k + 1, kc - 1) : k) + parity;
        const int A = wa[parity][odd], B = wb[parity][odd];
        std::uint32_t* row = out + j * kWordsPerLine;
        for (int g = 0; g < kGroupsPerLine; ++g) {
            Group grp;
            for (int i = 0; i < 6; ++i) grp.y[i] = static_cast<std::uint16_t>(ys[6 * g + i] << 2);
            for (int i = 0; i < 3; ++i) {
                const int cx = 3 * g + i;
                grp.cb[i] = static_cast<std::uint16_t>((A * cb_at(ra, cx) + B * cb_at(rb, cx) + 1) >> 1);
                grp.cr[i] = static_cast<std::uint16_t>((A * cr_at(ra, cx) + B * cr_at(rb, cx) + 1) >> 1);
            }
            pack_group(grp, row + 4 * g);
        }
    }
}
}  // namespace

void split_yuv420p(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v, int pitch_y, int pitch_c, int parity,
                   std::uint32_t* out) {
    split_impl(y, pitch_y, parity, out, [&](int r, int cx) { return int{u[r * pitch_c + cx]}; },
               [&](int r, int cx) { return int{v[r * pitch_c + cx]}; });
}

void split_nv12(const std::uint8_t* y, const std::uint8_t* uv, int pitch_y, int pitch_uv, int parity, std::uint32_t* out) {
    // lauko p chroma eilės c[k] = UV eilė 2k+p; svoriai aštuntosiomis (§6.4)
    static const int wa[2][2] = {{1, 5}, {3, 7}};  // [p][j nelyginis?] koeficientas „prieš" (k−1 arba k)
    static const int wb[2][2] = {{7, 3}, {5, 1}};  // koeficientas „po" (k arba k+1)
    constexpr int kc = kFieldHeight / 2;           // 270 chroma eilių lauke
    for (int j = 0; j < kFieldHeight; ++j) {
        const std::uint8_t* ys = y + (2 * j + parity) * pitch_y;
        const int k = j >> 1, odd = j & 1;
        const int ka = odd ? k : std::max(k - 1, 0);
        const int kb = odd ? std::min(k + 1, kc - 1) : k;
        const std::uint8_t* ca = uv + (2 * ka + parity) * pitch_uv;
        const std::uint8_t* cb = uv + (2 * kb + parity) * pitch_uv;
        const int A = wa[parity][odd], B = wb[parity][odd];
        std::uint32_t* row = out + j * kWordsPerLine;
        for (int g = 0; g < kGroupsPerLine; ++g) {
            Group grp;
            for (int i = 0; i < 6; ++i) grp.y[i] = static_cast<std::uint16_t>(ys[6 * g + i] << 2);
            for (int i = 0; i < 3; ++i) {
                const int x = 6 * g + 2 * i;
                grp.cb[i] = static_cast<std::uint16_t>((A * ca[x] + B * cb[x] + 1) >> 1);
                grp.cr[i] = static_cast<std::uint16_t>((A * ca[x + 1] + B * cb[x + 1] + 1) >> 1);
            }
            pack_group(grp, row + 4 * g);
        }
    }
}

void gen_field(const GenParams& g, std::uint64_t index, std::uint32_t* out) {
    const int parity = static_cast<int>(index & 1);
    for (int j = 0; j < kFieldHeight; ++j) {
        const int fy = 2 * j + parity;
        std::uint32_t* row = out + j * kWordsPerLine;
        for (int gi = 0; gi < kGroupsPerLine; ++gi) {
            Group grp;
            for (int i = 0; i < 6; ++i) grp.y[i] = gen_y(g, index, 6 * gi + i, fy);
            for (int i = 0; i < 3; ++i) gen_c(g, index, 6 * gi + 2 * i, fy, grp.cb[i], grp.cr[i]);
            pack_group(grp, row + 4 * gi);
        }
    }
}

}  // namespace mxlv::media::cpu
