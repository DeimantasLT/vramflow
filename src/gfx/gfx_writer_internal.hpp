// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// VIDINĖ libvramflow-gfx sąsaja testams (nėra viešoje C API ir neeksportuojama iš .so — žr. cmake/vramflow-gfx.map):
// ritmo gijos laiko šaltinio / laukimo kabliukai (laikrodžio šuolių modeliavimui, PF-5) ir skaitikliai.
#include <atomic>
#include <cstdint>
#include <functional>

#include "vramflow/gfx.h"

namespace vramflow::gfx::internal {

// Tušti (nullptr / tuščias std::function) laukai — numatytoji elgsena (mxlv::func::wait_until / now_tai).
struct PacingHooks {
    // Kaip func::wait_until: laukia iki indekso laiko + offset; true — laikrodis šoko atgal.
    std::function<bool(std::uint64_t index, std::int64_t offset_ns, const std::atomic<bool>& stop)> wait;
    // „Dabar“ (TAI ns) — iš jo skaičiuojamas einamasis indeksas.
    std::function<std::uint64_t()> now;
    // Kviečiamas ritmo gijoje po sėkmingo commit'o.
    std::function<void(std::uint64_t index)> on_commit;
};

struct PacingStats {
    std::uint64_t committed = 0;
    std::uint64_t clock_backward = 0;  // wait kabliukas pranešė apie šuolį atgal
    std::uint64_t late_skips = 0;      // praleisti (neįrašyti) laukai, nes gija pavėlavo > 1 lauką
    std::uint64_t no_frame = 0;        // indeksai be submit kadro (nieko necommit'inta)
    std::uint64_t errors = 0;          // open / kopijavimo / commit klaidos
};

// Kabliukai taikomi sekančiam vfgfx_start_pacing; w turi būti ne ritmo režime.
int set_pacing_hooks(vfgfx_writer w, PacingHooks hooks);
PacingStats pacing_stats(vfgfx_writer w);

}  // namespace vramflow::gfx::internal
