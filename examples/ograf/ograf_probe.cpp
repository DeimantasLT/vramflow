// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// ograf_probe — demonstracijos patikra: skaito composite `v210a` išėjimą (host saugykla), skaičiuoja key (alfa)
// pasiskirstymą ir išsaugo vieną lauką kaip PNG peržiūrą (key ir Y). Naudoja tik viešą MXL C API.
// Sėkmės taisyklė (exit 0): perskaityta >= 1 laukas, ir key turi reikšmių griežtai tarp 64 ir 940 (alfa tikrai perduodama,
// ne vien permatoma/nepermatoma). Naudojimas: ograf_probe --domain D --flow-id UUID --out DIR [--seconds 10]
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>
#include <vector>

namespace {
constexpr int W = 1920, H = 1080, LH = H / 2;  // grain = laukas 1920x540
constexpr size_t FILL_LINE = ((W + 47) / 48) * 128, KEY_LINE = ((W + 2) / 3) * 4, FILL_BYTES = FILL_LINE * LH;

std::uint32_t crc_table[256];
std::uint32_t crc(const std::uint8_t* d, size_t n, std::uint32_t c = 0xFFFFFFFFu) {
    for (size_t i = 0; i < n; ++i) c = crc_table[(c ^ d[i]) & 255] ^ (c >> 8);
    return c;
}
void be32(std::vector<std::uint8_t>& o, std::uint32_t v) { for (int s = 24; s >= 0; s -= 8) o.push_back(static_cast<std::uint8_t>(v >> s)); }
void chunk(std::vector<std::uint8_t>& o, const char* type, const std::vector<std::uint8_t>& data) {
    be32(o, static_cast<std::uint32_t>(data.size()));
    std::vector<std::uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    o.insert(o.end(), td.begin(), td.end());
    be32(o, ~crc(td.data(), td.size()));
}
// 8 bitų pilkas PNG, nesuspausti (stored) deflate blokai — be zlib priklausomybės.
bool write_png(const std::string& path, const std::vector<std::uint8_t>& gray, int w, int h) {
    for (std::uint32_t n = 0; n < 256; ++n) { std::uint32_t c = n; for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; crc_table[n] = c; }
    std::vector<std::uint8_t> raw;
    for (int y = 0; y < h; ++y) { raw.push_back(0); raw.insert(raw.end(), gray.begin() + static_cast<size_t>(y) * w, gray.begin() + static_cast<size_t>(y + 1) * w); }
    std::vector<std::uint8_t> z{0x78, 0x01};
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t p = 0; p < raw.size(); p += 65535) {
        const size_t n = std::min<size_t>(65535, raw.size() - p);
        z.push_back(p + n >= raw.size());
        z.push_back(n & 255); z.push_back(n >> 8); z.push_back(~n & 255); z.push_back((~n >> 8) & 255);
        z.insert(z.end(), raw.begin() + p, raw.begin() + p + n);
    }
    be32(z, (b << 16) | a);
    std::vector<std::uint8_t> out{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}, ihdr;
    be32(ihdr, w); be32(ihdr, h);
    for (std::uint8_t v : {8, 0, 0, 0, 0}) ihdr.push_back(v);
    chunk(out, "IHDR", ihdr); chunk(out, "IDAT", z); chunk(out, "IEND", {});
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    std::string domain, id, outdir = ".";
    double seconds = 10;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--domain") domain = v; else if (k == "--flow-id") id = v; else if (k == "--out") outdir = v;
        else if (k == "--seconds") seconds = std::atof(v.c_str()); else return 2;
    }
    if (domain.empty() || id.empty()) { std::fprintf(stderr, "naudojimas: ograf_probe --domain D --flow-id UUID --out DIR [--seconds N]\n"); return 2; }
    mxlInstance inst = mxlCreateInstance(domain.c_str(), "");
    mxlFlowReader r = nullptr;
    for (int i = 0; i < 100 && mxlCreateFlowReader(inst, id.c_str(), "", &r) != MXL_STATUS_OK; ++i) {
        r = nullptr; struct timespec ts{0, 100000000}; nanosleep(&ts, nullptr);
    }
    mxlGrainStorageLayout layout{};
    if (!r || mxlFlowReaderGetStorageLayout(r, &layout) != MXL_STATUS_OK || layout.planeCount != 2) { std::fprintf(stderr, "flow neatsidarė / ne v210a\n"); return 1; }
    std::vector<mxlGrainStorage> slots(layout.slotCount);
    if (mxlFlowReaderMapSlots(r, layout.slotCount, slots.data()) != MXL_STATUS_OK) { std::fprintf(stderr, "MapSlots nepavyko\n"); return 1; }
    mxlFlowReaderGetStorageLayout(r, &layout);
    if (layout.planes[1].offset != FILL_BYTES || layout.planes[1].pitch != KEY_LINE) { std::fprintf(stderr, "netikėtas layout\n"); return 1; }

    const mxlRational rate{50, 1};
    std::uint64_t idx = mxlGetCurrentIndex(&rate), fields = 0, k64 = 0, k940 = 0, mid = 0, total = 0;
    std::vector<std::uint8_t> key_png(static_cast<size_t>(W) * LH), y_png(static_cast<size_t>(W) * LH);
    std::uint64_t best_mid = 0;
    const std::uint64_t end = mxlGetTime() + static_cast<std::uint64_t>(seconds * 1e9);
    while (mxlGetTime() < end) {
        mxlGrainInfo info{};
        std::uint32_t slot = 0;
        const mxlStatus st = mxlFlowReaderGetGrainSlot(r, idx, MXL_GRAIN_VALID_SLICES_ALL, 50'000'000, &info, &slot);
        if (st == MXL_ERR_OUT_OF_RANGE_TOO_LATE) { idx = mxlGetCurrentIndex(&rate); continue; }
        if (st != MXL_STATUS_OK) continue;  // TOO_EARLY: grain'o dar nėra (arba nebus commit'intas)
        ++idx;
        if (info.flags & MXL_GRAIN_FLAG_INVALID) continue;
        const auto* base = static_cast<const std::uint8_t*>(slots[slot].host.pointer);
        std::vector<std::uint8_t> kp(key_png.size()), yp(y_png.size());
        std::uint64_t m = 0;
        for (int y = 0; y < LH; ++y) {
            const auto* kl = reinterpret_cast<const std::uint32_t*>(base + FILL_BYTES + static_cast<size_t>(y) * KEY_LINE);
            for (int x = 0; x < W; ++x) {
                const std::uint32_t k = (kl[x / 3] >> (10 * (x % 3))) & 1023;
                ++total; if (k == 64) ++k64; else if (k == 940) ++k940; else ++mid;
                if (k > 64 && k < 940) ++m;
                kp[static_cast<size_t>(y) * W + x] = static_cast<std::uint8_t>(std::min(255, std::max(0, (int(k) - 64) * 255 / 876)));
            }
            const auto* fl = reinterpret_cast<const std::uint32_t*>(base + static_cast<size_t>(y) * FILL_LINE);
            for (int x = 0; x < W; ++x) {  // v210: 6 pikseliai / 4 žodžiai; Y: w0[10:19], w1[0:9], w1[20:29], w2[10:19], w3[0:9], w3[20:29]
                const std::uint32_t* w = fl + (x / 6) * 4;
                static const int wi[6] = {0, 1, 1, 2, 3, 3}, sh[6] = {10, 0, 20, 10, 0, 20};
                const std::uint32_t Y = (w[wi[x % 6]] >> sh[x % 6]) & 1023;
                yp[static_cast<size_t>(y) * W + x] = static_cast<std::uint8_t>(std::min(255, std::max(0, (int(Y) - 64) * 255 / 876)));
            }
        }
        ++fields;
        if (m >= best_mid) { best_mid = m; key_png = kp; y_png = yp; }
    }
    std::printf("ograf_probe: fields=%llu key==64: %llu, key==940: %llu, 64<key<940: %llu (of %llu samples); best field has %llu intermediate\n",
                (unsigned long long)fields, (unsigned long long)k64, (unsigned long long)k940, (unsigned long long)mid,
                (unsigned long long)total, (unsigned long long)best_mid);
    const bool pngs = write_png(outdir + "/ograf-key.png", key_png, W, LH) && write_png(outdir + "/ograf-fill-y.png", y_png, W, LH);
    std::printf("ograf_probe: PNG %s (%s/ograf-key.png, ograf-fill-y.png)\n", pngs ? "OK" : "FAIL", outdir.c_str());
    mxlReleaseFlowReader(inst, r);
    mxlDestroyInstance(inst);
    return fields >= 1 && mid > 0 && pngs ? 0 : 1;
}
