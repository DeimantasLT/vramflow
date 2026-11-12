// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Atkūrimo eilės (dec) perpildymų priskyrimas laikrodžiui — taisyklė (peržiūros radinys po V13, pakeičia 3 s langą):
//
//   Laikrodžio žingsnis ATGAL per S paaiškina ne daugiau kaip ceil(S / kadro trukmė) eilėje susikaupusių kadrų, VIENĄ
//   kartą: pirmame kadro tick'e po žingsnio tiek kadrų (bet ne daugiau, nei eilė viršija atkūrimo gylį) išmetama kaip
//   laikrodžio pasekmė (clock_overrun_frames), ir eilė grįžta į įprastą gylį. Žingsnis PIRMYN eilės nepadidina (variklis
//   indeksus praleidžia — clock_skips — realiu laiku tick'ai eina tuo pačiu tempu), todėl nepaaiškina nė vieno kadro.
//   Visa kita, kas viršija ribą (gylis + 1 s), — tikras perpildymas (overrun_frames → exit 4).
//
// Kodėl ne laiko langas: 3 s langas po BET KURIO ±5 ms žingsnio (ir pirmyn) slėpė tikrą perkrovą; be išmetimo po −1,3 s
// žingsnio eilė likdavo ties riba, ir vėlesni dekoderio pliūpsniai duodavo overrun_frames (exit 4) be jokios perkrovos.
// Darbininko apsauginiai išmetimai (eilė > 60) laukimo metu, kurių dar nepaaiškino išmetimas pirmame tick'e, —
// perklasifikuojami iš to paties likučio (reclassify).
#include <algorithm>
#include <cstdint>

namespace mxlv::timing {

class ClockQueueBudget {
public:
    explicit ClockQueueBudget(std::int64_t frame_ns) : frame_ns_(frame_ns > 0 ? frame_ns : 1) {}

    // Variklio aptiktas žingsnis (TAI − MONOTONIC_RAW pokytis), ns; < 0 — atgal.
    void on_step(std::int64_t delta_ns) {
        if (delta_ns < 0) pending_ += static_cast<std::uint64_t>((-delta_ns + frame_ns_ - 1) / frame_ns_);
    }
    std::uint64_t pending() const { return pending_; }

    struct Tick {
        std::uint64_t clock_drop = 0;    // išmesti dabar: laikrodžio pasekmė
        std::uint64_t overrun_drop = 0;  // išmesti dabar: tikras perpildymas (virš cap)
        std::uint64_t reclassify = 0;    // darbininko išmetimai nuo praeito tick'o, priskiriami laikrodžiui
    };
    // Kadro tick'as: q — eilės ilgis, depth — atkūrimo gylis, cap — riba (gylis + 1 s), worker_drops — darbininko
    // apsauginių išmetimų skaičius nuo praeito kadro tick'o. Likutis po šio tick'o nebegalioja (vienas kartas).
    Tick on_frame_tick(std::uint64_t q, std::uint64_t depth, std::uint64_t cap, std::uint64_t worker_drops) {
        Tick t;
        if (pending_ > 0) {
            t.clock_drop = q > depth ? std::min(pending_, q - depth) : 0;
            t.reclassify = std::min(pending_ - t.clock_drop, worker_drops);
            pending_ = 0;
        }
        const std::uint64_t left = q - t.clock_drop;
        t.overrun_drop = left > cap ? left - cap : 0;
        return t;
    }

private:
    std::int64_t frame_ns_;
    std::uint64_t pending_ = 0;
};

}  // namespace mxlv::timing
