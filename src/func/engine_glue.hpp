// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Funkcijų ↔ bendro tick variklio jungtis (planas docs/planai/2026-11-12-tick-variklis.md §7 „mxlv::func“, V8+):
// parinktys → EngineConfig (viena vieta) ir variklio skaitikliai → santraukos raktai (tie patys kaip prieš migraciją).
#include <cstdint>

#include "engine/engine.hpp"
#include "engine/stats.hpp"
#include "func/func_common.hpp"

namespace mxlv::func {

// Variklio ribos funkcijai:
//   Legacy    — šiandieninis ciklas: viena riba ties T + --deadline-ms, apklausa 0,5 ms, be pertraukiklio, leave, wait-all;
//   Isolation — planas V10: Wr = P/4, M = 3, Late histerezė 8, futex (ašinis) laukimas; D = --deadline-ms (apkarpoma iki
//               2P − G), leave, wait-all (spragų politika ir startas keičiami vėliau — V17, V18).
enum class EngineTiming { Legacy, Isolation };

// read_offset_ns — ro (logo: --read-offset-us; gen: 0). Trukmė, warmup — iš parinkčių.
engine::EngineConfig engine_config(const Opts& o, EngineTiming t, std::int64_t read_offset_ns);

// Kur santraukoje rašyti variklio praleidimus: proceso lygiu (gen) ar kiekvienam flow'ui (logo, composite…).
enum class SkipScope { Process, PerFlow };

// Variklio rezultatas ir skaitikliai → Stats (first / last_index, clock_steps, clock_backward, *_skips). Raktai, kurių
// reikšmė 0, nerašomi (kaip anksčiau — „įvykio“ skaitikliai atsiranda tik padidinti), išskyrus clock_steps.
void to_summary(const engine::StatsHub& hub, const engine::TickEngine& eng, const engine::RunResult& r, std::uint32_t units,
                SkipScope scope, Stats& st);

}  // namespace mxlv::func
