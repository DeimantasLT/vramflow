// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Fiksuotos struktūros skaitikliai (planas §6 D6, §7 stats.hpp): indeksuojami lauku, ne vardu; vardai — tik išvedant.
// Rašo tik variklio gija.
#include <cstdint>

namespace mxlv::engine {

struct InCounters {
    std::uint64_t ok = 0, late = 0, lost = 0, invalid = 0, too_late = 0, read_errors = 0, reattached = 0, stalls = 0,
                  detaches = 0;
};

struct OutCounters {
    std::uint64_t ok = 0;  // tikras turinys commit'intas A fazėje
    std::uint64_t late = 0;  // tikras turinys commit'intas B / C fazėje
    std::uint64_t gap_invalid = 0, gap_frozen = 0, gap_black = 0, gap_left = 0, gap_startup = 0;
    std::uint64_t lapped = 0, late_skips = 0, clock_skips = 0, warmup_skips = 0, commit_errors = 0;
};

}  // namespace mxlv::engine
