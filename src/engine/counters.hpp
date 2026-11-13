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
    // Spragos ne dėl šios funkcijos (veiksmas — pagal politiką, bet skaičiuojama čia, ne gap_invalid / frozen / black):
    // gap_upstream — būtinas įėjimas pateikė INVALID (aukštupio spraga); gap_align — interlaced freeze / black: grįžęs turinys
    // laikomas iki kadro ribos (nelyginis indeksas). Ne praradimas.
    std::uint64_t gap_upstream = 0, gap_align = 0;
    std::uint64_t lapped = 0, late_skips = 0, clock_skips = 0, warmup_skips = 0, commit_errors = 0;
    std::uint64_t already_committed = 0;  // open → AlreadyCommitted (laikrodis atgal, M11): ne klaida, ne praradimas
};

// Vienetui (ir sink'ui, kuris išėjimų neturi): kiekvienas dalyvaujantis indeksas baigiasi lygiai vienu iš
// ok / late / gaps / lapped / *_skips.
// startup — spragos, kol būtinas įėjimas dar niekada nebuvo prijungtas (ne praradimas).
// gaps_clock, gaps_no_content — gaps poaibiai (laikrodžio žingsnis / jau commit'intas indeksas; šaltinis pranešė „nėra
// turinio“): ne praradimas. gaps_upstream (būtinas įėjimas — INVALID), gaps_align (turinys laikomas iki kadro ribos) — taip
// pat gaps poaibiai ir ne praradimas: šis vienetas tų indeksų neprarado — jų nebuvo.
// failures — kiek kartų vienetas sugedo (fail_unit); retries — nepavykę atkūrimo bandymai; recoveries — sėkmingi (vienetas
// vėl dirba). Tai įvykių skaitikliai, ne indeksų rezultatai.
struct UnitCounters {
    std::uint64_t ok = 0, late = 0, gaps = 0, startup = 0, lapped = 0, late_skips = 0, clock_skips = 0, warmup_skips = 0;
    std::uint64_t gaps_clock = 0, gaps_no_content = 0, gaps_upstream = 0, gaps_align = 0;
    std::uint64_t failures = 0, retries = 0, recoveries = 0;
};

}  // namespace mxlv::engine
