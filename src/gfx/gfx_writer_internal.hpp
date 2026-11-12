// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// VIDINĖ libvramflow-gfx sąsaja testams (nėra viešoje C API ir neeksportuojama iš .so — žr. cmake/vramflow-gfx.map):
// ritmo gijos laiko šaltinio / laukimo kabliukai (laikrodžio šuolių modeliavimui, PF-5) ir skaitikliai.
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "vramflow/gfx.h"

namespace vramflow::gfx::internal {

// Tušti (nullptr / tuščias std::function) laukai — numatytoji elgsena (bendro variklio SystemClock: mxlGetTime /
// mxlSleepUntil). Rašytojas su laiko kabliukais (now arba wait) gauna savo variklį (savo ritmo giją): now -> variklio Clock,
// wait -> EngineConfig::open_wait (planas V14). on_commit / fail_commit / after_convert laiko nekeičia — rašytojas lieka
// bendroje (dažnis, offset, GPU) ritmo gijoje; jie kviečiami ritmo gijoje, laikant VISŲ grupės narių lat_m (kabliukas
// neturi kviesti šios bibliotekos funkcijų).
struct PacingHooks {
    // Laukia iki indekso laiko + offset (tick'o atidarymas); true — laikrodis šoko atgal.
    std::function<bool(std::uint64_t index, std::int64_t offset_ns, const std::atomic<bool>& stop)> wait;
    // „Dabar“ (TAI ns) — iš jo skaičiuojamas einamasis indeksas.
    std::function<std::uint64_t()> now;
    // Kviečiamas ritmo gijoje po sėkmingo commit'o.
    std::function<void(std::uint64_t index)> on_commit;
    // Jei grąžina true, ritmo gijos commit'as laikomas nepavykusiu (klaidos kelio testams).
    std::function<bool()> fail_commit;
    // vfgfx_submit_at: kviečiamas po kadro konvertavimo, prieš įdedant į eilę (lenktynių su ritmo gija testams).
    std::function<void()> after_convert;
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
std::string pacing_error(vfgfx_writer w);  // paskutinė ritmo gijos klaida
int prepared_sets(vfgfx_writer w);         // išskirtų paruoštų kadrų aibių skaičius (atminties kaina)

}  // namespace vramflow::gfx::internal
