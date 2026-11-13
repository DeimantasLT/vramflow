// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/edge.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace mxlv::engine {

std::int64_t phase_ns(const CutConfig& k) { return k.expect_ns + k.read_offset_ns; }

std::int64_t commit_bound_ns(const CutConfig& k, bool source) {
    const Cuts c = cuts_for(0, k);  // T(0) = 0: ribos — santykiniai laikai
    // šaltinis įėjimų nelaukia, bet skelbia tą pačią formulę su numatytuoju Wr = P/4 (planas D7: Φ + Wr + G) — atsarga
    // pabudimo drebėjimui (WSL2), legacy ribų Wr = D į ją neįeina
    if (source) return c.open + period_ns(k.rate) / 4 + k.guard_ns;
    return c.a + k.guard_ns;
}

std::int64_t content_delay_ns(std::span<const EdgeTiming> in, timing::Rational rate) {
    const std::int64_t p = period_ns(rate);
    std::int64_t cd = 0;
    for (const EdgeTiming& e : in) cd = std::max(cd, e.content_delay_ns - static_cast<std::int64_t>(e.index_offset) * p);
    return cd;
}

BundleAlign bundle_align(std::span<const BundleMember> m, timing::Rational video_rate, std::int64_t sample_rate) {
    BundleAlign r;
    for (const BundleMember& b : m) r.align_ns = std::max(r.align_ns, b.content_delay_ns);
    const std::int64_t p = period_ns(video_rate);
    r.shift.reserve(m.size());
    for (const BundleMember& b : m) {
        const std::int64_t d = r.align_ns - b.content_delay_ns;  // ≥ 0
        // video — artimiausias grain'ų skaičius (vėlinimai — grain'ų kartotiniai; 59,94 Hz periodas suapvalintas iki ns);
        // garsas — mėginiai: d · sr / 1e9 (40 ms · 48 kHz = 1920)
        const std::int64_t s = b.audio ? (d * sample_rate + 500'000'000) / 1'000'000'000 : (p > 0 ? (d + p / 2) / p : 0);
        r.shift.push_back(-s);
    }
    return r;
}

RingCheck ring_check(const CutConfig& k, std::int32_t index_offset, std::int64_t history_ns) {
    RingCheck r;
    const std::int64_t p = period_ns(k.rate);
    const std::int64_t d = std::min(k.deadline_ns, max_deadline_ns(k));
    r.lag_ns = k.expect_ns + d - static_cast<std::int64_t>(index_offset) * p;
    r.history_ns = history_ns;
    const std::int64_t slots = p > 0 ? history_ns / p : 0;
    r.safe_ns = std::max<std::int64_t>(0, slots - 2) * p;
    if (r.lag_ns <= r.safe_ns / 2) return r;
    r.fit = r.lag_ns > r.safe_ns ? RingFit::Refuse : RingFit::Warn;
    char b[512];
    std::snprintf(b, sizeof b,
                  "%s: įėjimas skaitomas iki T + %.1f ms (fazė E %.1f ms + riba D %.1f ms%s), o ring'as saugus iki %.1f ms "
                  "(domeno istorija %.0f ms = %lld grain'ų, 2 — atsarga). Padidinkite domeno istoriją: <domenas>/options.json "
                  "{\"urn:x-mxl:option:history_duration/v1.0\": %lld} (ns) PRIEŠ kuriant flow'us, arba trumpinkite grandinę "
                  "(--deadline-ms, mažiau pakopų viename laikrodžio domene)",
                  r.fit == RingFit::Refuse ? "ring'as per seklus" : "ring'o atsarga maža", r.lag_ns / 1e6, k.expect_ns / 1e6,
                  d / 1e6, index_offset < 0 ? " + --input-offset" : "", r.safe_ns / 1e6, history_ns / 1e6,
                  static_cast<long long>(slots), static_cast<long long>(((r.lag_ns * 2) / p + 3) * p));
    r.text = b;
    return r;
}

std::string timing_tags_json(const OutputTiming& t) {
    char b[256];
    std::snprintf(b, sizeof b, R"(,"%s":["%lld"],"%s":["%lld"])", kTagCommitBound, static_cast<long long>(t.commit_bound_ns),
                  kTagContentDelay, static_cast<long long>(t.content_delay_ns));
    return b;
}

namespace {
// "<tag>" : [ "N" ] → N (≥ 0); false — nėra arba netinkama.
bool tag_value(const std::string& def, const char* tag, std::int64_t& out) {
    const std::string key = std::string("\"") + tag + "\"";
    std::size_t p = def.find(key);
    if (p == std::string::npos) return false;
    p += key.size();
    auto skip = [&] {
        while (p < def.size() && std::isspace(static_cast<unsigned char>(def[p]))) ++p;
    };
    for (const char want : {':', '[', '"'}) {
        skip();
        if (p >= def.size() || def[p] != want) return false;
        ++p;
    }
    const std::size_t q = def.find('"', p);
    if (q == std::string::npos || q == p) return false;
    const std::string v = def.substr(p, q - p);
    for (const char ch : v)
        if (!std::isdigit(static_cast<unsigned char>(ch))) return false;  // be ženklo: riba ≥ 0
    if (v.size() > 18) return false;
    out = std::strtoll(v.c_str(), nullptr, 10);
    return true;
}
}  // namespace

bool parse_timing_tags(const std::string& def, OutputTiming& out) {
    std::int64_t b = 0, cd = 0;
    if (!tag_value(def, kTagCommitBound, b)) return false;
    if (def.find(kTagContentDelay) != std::string::npos && !tag_value(def, kTagContentDelay, cd)) return false;
    out.commit_bound_ns = b;
    out.content_delay_ns = cd;
    return true;
}

}  // namespace mxlv::engine
