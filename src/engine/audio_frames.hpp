// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Garso įėjimo indeksavimas varikliui (MxlAudioInput): vieneto indeksas j — kadro F = j / gpf paskutinio lauko indeksas
// (enc: stride = gpf, residue = gpf − 1); kadro F garsas — mėginiai [sample_begin(F), sample_begin(F + 1)) (sveikaisiais
// skaičiais, kaip AudioTimeline ir gen). MXL tęstinio flow'o skaitymo „head“ indeksas — mėginių intervalo pabaiga
// (mxlFlowReaderGetSamples index, count → [index − count, index)); intervalas skaitomas, kai pabaiga ≤ headIndex.
// Gryna logika (be MXL) — tikrinama tests/engine.
#include <cstdint>

#include "common/format.hpp"
#include "engine/clock.hpp"

namespace mxlv::engine {

struct AudioFrames {
    vramflow::Rational frame_rate{25, 1};
    std::uint32_t gpf = 2;  // grain'ų (laukų) kadre

    std::uint64_t frame_of(Index j) const { return j / gpf; }
    std::uint64_t end(Index j) const { return vramflow::sample_begin(frame_rate, frame_of(j) + 1); }
    std::uint32_t count(Index j) const {
        return static_cast<std::uint32_t>(end(j) - vramflow::sample_begin(frame_rate, frame_of(j)));
    }
    // Didžiausias vieneto indeksas j (≡ gpf − 1 mod gpf), kurio garsas jau yra, kai paskutinis įrašytas mėginio intervalo
    // galas — head; kNoIndex — nė vieno kadro.
    Index last_ready(std::uint64_t head) const {
        if (head == kNoIndex || head < vramflow::sample_begin(frame_rate, 1)) return kNoIndex;
        // apytikslis F, tada tikslinama sveikaisiais skaičiais (sample_begin monotoniškas)
        std::uint64_t f = static_cast<std::uint64_t>(static_cast<double>(head) * static_cast<double>(frame_rate.num) /
                                                     (48000.0 * static_cast<double>(frame_rate.den)));
        while (f > 0 && vramflow::sample_begin(frame_rate, f + 1) > head) --f;
        while (vramflow::sample_begin(frame_rate, f + 2) <= head) ++f;
        return f * gpf + (gpf - 1);
    }
};

}  // namespace mxlv::engine
