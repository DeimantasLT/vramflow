// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// mxl_hop_probe — gryno libmxl šuolio vėlavimas be vramflow variklio: vienas rašytojas 50 Hz commit'ina grain'us laiku,
// skaitytojo gija blokuojančiai laukia kito indekso. Matuojama: skaitytojo grįžimas (CLOCK_TAI) − rašytojo laikas tuoj po
// commit'o. Tai ta pati atkarpa, kurią matuoja `ctest gap_policy … < 1 ms` (planas 3.1, hipotezės 3 ir 4 atskyrimas:
// jei gryno šuolio uodega tokia pati, vėlavimą lemia platforma, o ne vramflow kodo kelias).
//
// Naudojimas: mxl_hop_probe <domenas> <sekundės> [invalid] [window]
//   invalid — commit'inti su MXL_GRAIN_FLAG_INVALID ir validSlices = 0 (kaip spragos grain'as);
//   window  — kiek imčių į vieną langą (numatyta 80 = tiek, kiek gap_policy testas surenka per 1,6 s).
// Išvestis: viena eilutė su p50 / p99 / p999 / max µs ir langų maksimumų skirstiniu (kiek langų max ≥ 1 ms).
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr const char* kFlowId = "6b1d2f3a-0c4e-4d5f-9a6b-7c8d9e0f1a2b";
const std::string kDef = R"({"id":"6b1d2f3a-0c4e-4d5f-9a6b-7c8d9e0f1a2b","description":"mxl_hop_probe","label":"mxl_hop_probe",)"
                         R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["mxl_hop_probe:Video"]},)"
                         R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210",)"
                         R"("grain_rate":{"numerator":50,"denominator":1},"frame_width":1920,"frame_height":1080,)"
                         R"("interlace_mode":"progressive","colorspace":"BT709","components":[)"
                         R"({"name":"Y","width":1920,"height":1080,"bit_depth":10},)"
                         R"({"name":"Cb","width":960,"height":1080,"bit_depth":10},)"
                         R"({"name":"Cr","width":960,"height":1080,"bit_depth":10}]})";

std::int64_t pct(const std::vector<std::int64_t>& v, double p) {
    return v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size())))];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "naudojimas: mxl_hop_probe <domenas> <sekundės> [invalid] [window]\n");
        return 2;
    }
    const double seconds = std::atof(argv[2]);
    const bool invalid = argc > 3 && std::strcmp(argv[3], "invalid") == 0;
    const std::size_t window = argc > 4 ? static_cast<std::size_t>(std::atoi(argv[4])) : 80;
    mxlInstance inst = mxlCreateInstance(argv[1], "");
    if (!inst) return std::fprintf(stderr, "mxlCreateInstance nepavyko\n"), 2;
    mxlFlowWriter w = nullptr;
    mxlFlowConfigInfo ci{};
    bool created = false;
    if (mxlCreateFlowWriter(inst, kDef.c_str(), "", &w, &ci, &created) != MXL_STATUS_OK) return std::fprintf(stderr, "writer nepavyko\n"), 2;
    const mxlRational rate{50, 1};
    const std::uint64_t first = mxlGetCurrentIndex(&rate) + 3;
    const std::uint64_t count = static_cast<std::uint64_t>(seconds * 50);
    std::vector<std::uint64_t> committed(count, 0), returned(count, 0);  // indeksas − first
    std::atomic<bool> stop{false};
    std::thread rd([&] {
        mxlFlowReader r = nullptr;
        while (mxlCreateFlowReader(inst, kFlowId, "", &r) != MXL_STATUS_OK) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        for (std::uint64_t i = first; i < first + count && !stop.load();) {
            mxlGrainInfo gi{};
            std::uint8_t* p = nullptr;
            const mxlStatus st = mxlFlowReaderGetGrain(r, i, 100'000'000, &gi, &p);
            const std::uint64_t t = mxlGetTime();
            if (st == MXL_STATUS_OK) returned[i - first] = t, ++i;
            else if (st == MXL_ERR_OUT_OF_RANGE_TOO_LATE) ++i;
        }
        mxlReleaseFlowReader(inst, r);
    });
    for (std::uint64_t i = first; i < first + count; ++i) {
        const std::uint64_t ns = mxlGetNsUntilIndex(i, &rate);
        if (ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
        mxlGrainInfo gi{};
        std::uint8_t* p = nullptr;
        if (mxlFlowWriterOpenGrain(w, i, &gi, &p) != MXL_STATUS_OK) continue;
        if (invalid) gi.flags |= MXL_GRAIN_FLAG_INVALID, gi.validSlices = 0;
        else gi.validSlices = gi.totalSlices;
        mxlFlowWriterCommitGrain(w, &gi);
        committed[i - first] = mxlGetTime();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    stop = true;
    rd.join();
    std::vector<std::int64_t> lat;
    std::size_t windows = 0, windows_over = 0;
    std::vector<std::int64_t> wmax;
    std::int64_t cur = INT64_MIN;
    std::size_t in_win = 0;
    for (std::uint64_t k = 0; k < count; ++k) {
        if (!committed[k] || !returned[k]) continue;
        const std::int64_t d = static_cast<std::int64_t>(returned[k]) - static_cast<std::int64_t>(committed[k]);
        lat.push_back(d);
        cur = std::max(cur, d);
        if (++in_win == window) {
            ++windows;
            windows_over += cur >= 1'000'000 ? 1 : 0;
            wmax.push_back(cur);
            cur = INT64_MIN, in_win = 0;
        }
    }
    mxlReleaseFlowWriter(inst, w);
    mxlDestroyInstance(inst);
    if (lat.size() < 2) return std::fprintf(stderr, "per mažai imčių: %zu\n", lat.size()), 2;
    std::sort(lat.begin(), lat.end());
    std::sort(wmax.begin(), wmax.end());
    std::printf("hop %s n=%zu p50=%lld p99=%lld p999=%lld max=%lld us | langai(%zu)=%zu, max>=1ms: %zu, lango max p50=%lld p90=%lld us\n",
                invalid ? "invalid" : "valid", lat.size(), static_cast<long long>(pct(lat, 0.5) / 1000),
                static_cast<long long>(pct(lat, 0.99) / 1000), static_cast<long long>(pct(lat, 0.999) / 1000),
                static_cast<long long>(lat.back() / 1000), window, windows, windows_over,
                static_cast<long long>(wmax.empty() ? 0 : pct(wmax, 0.5) / 1000),
                static_cast<long long>(wmax.empty() ? 0 : pct(wmax, 0.9) / 1000));
    return 0;
}
