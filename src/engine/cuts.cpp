// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/cuts.hpp"

#include <algorithm>

namespace mxlv::engine {

CutConfig default_cuts(timing::Rational rate) {
    const std::int64_t p = period_ns(rate);
    CutConfig c;
    c.rate = rate;
    c.read_offset_ns = 500'000;
    c.ready_wait_ns = p / 4;
    c.deadline_ns = 3 * p / 4;
    c.guard_ns = 3 * p / 20;
    return c;
}

CutConfig legacy_cuts(timing::Rational rate, std::int64_t read_offset_ns, std::int64_t deadline_ns) {
    CutConfig c = default_cuts(rate);
    c.read_offset_ns = read_offset_ns;
    c.deadline_ns = deadline_ns;
    c.ready_wait_ns = deadline_ns;
    c.single_phase = true;
    return c;
}

std::int64_t max_deadline_ns(const CutConfig& c) { return 2 * period_ns(c.rate) - c.guard_ns; }

bool clamp_deadline(CutConfig& c, std::string& warn) {
    const std::int64_t mx = max_deadline_ns(c);
    if (c.deadline_ns <= mx) return false;
    warn = "--deadline-ms: " + std::to_string(c.deadline_ns) + " ns > 2P - G = " + std::to_string(mx) +
           " ns; apkarpyta iki " + std::to_string(mx) + " ns";
    c.deadline_ns = mx;
    return true;
}

Cuts cuts_for(Index c, const CutConfig& k) {
    const std::int64_t t = timing::time_of(c, k.rate) + k.expect_ns;
    const std::int64_t d = std::min(k.deadline_ns, max_deadline_ns(k));
    Cuts r{};
    r.open = t + k.read_offset_ns;
    if (k.single_phase) {
        r.a = r.b = r.c = t + d;
        r.has_b = r.has_c = false;
        return r;
    }
    const std::int64_t open_next = timing::time_of(c + 1, k.rate) + k.expect_ns + k.read_offset_ns;
    r.b = std::min(t + d, open_next - k.guard_ns);
    r.a = std::min(r.open + k.ready_wait_ns, r.b);
    r.c = t + d;
    r.has_b = true;
    r.has_c = r.c > r.b;
    if (!r.has_c) r.c = r.b;
    return r;
}

Index due_index(std::uint64_t now_tai, const CutConfig& k) {
    return timing::index_of(static_cast<std::int64_t>(now_tai) - k.expect_ns, k.rate);
}

}  // namespace mxlv::engine
