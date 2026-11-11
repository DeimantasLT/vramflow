// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Tick'o ribos — gryna matematika (planas docs/planai/2026-11-12-tick-variklis.md §6 D2). Visi laikai — TAI ns.
//   t_open(c) = T(c) + E + ro
//   A(c)      = t_open(c) + Wr                              (ne vėliau kaip B(c))
//   B(c)      = min(T(c) + E + D_eff, t_open(c+1) − G)
//   C(c)      = T(c) + E + D_eff, tik jei > B(c)            (pernešimas; K-T1)
//   D_eff     = min(D, 2P − G)
// single_phase (legacy): viena riba A = T(c) + E + D_eff; B ir C nėra.
#include <cstdint>
#include <string>

#include "engine/clock.hpp"
#include "timing/index.hpp"

namespace mxlv::engine {

struct Cuts {
    std::int64_t open, a, b, c;
    bool has_b, has_c;  // has_b == false — viena riba (b == a)
};

struct CutConfig {
    timing::Rational rate{50, 1};
    std::int64_t read_offset_ns = 500'000;  // ro
    std::int64_t expect_ns = 0;             // E (briaunos sutartis; kol neįjungta — 0)
    std::int64_t ready_wait_ns = 0;         // Wr
    std::int64_t deadline_ns = 0;           // D (nuo T(c) + E)
    std::int64_t guard_ns = 0;              // G
    bool single_phase = false;
};

constexpr std::int64_t period_ns(timing::Rational r) { return timing::time_of(1, r) - timing::time_of(0, r); }

CutConfig default_cuts(timing::Rational rate);  // Wr = P/4, D = 3P/4, G = 3P/20, ro = 500 µs
// Šiandieninis ciklas: viena riba ties T + deadline; Wr = D.
CutConfig legacy_cuts(timing::Rational rate, std::int64_t read_offset_ns, std::int64_t deadline_ns);
std::int64_t max_deadline_ns(const CutConfig&);   // 2P − G
bool clamp_deadline(CutConfig&, std::string& warn);  // D_eff = min(D, 2P − G); true — apkarpyta (warn — abu skaičiai)
Cuts cuts_for(Index c, const CutConfig&);
Index due_index(std::uint64_t now_tai, const CutConfig&);  // index_of(now − expect_ns, rate) — T11

}  // namespace mxlv::engine
