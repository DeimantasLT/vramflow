// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Briaunos laiko sutartis (planas docs/planai/2026-11-12-tick-variklis.md §6 D7, V24; K-T10, K-T14) — gryna matematika ir
// flow aprašo žymų tekstas, be MXL.
//
//   Įėjimas:  EdgeTiming{index_offset, expect_ns, content_delay_ns} — grain'as j = i + index_offset laukiamas commit'intas iki
//             T(j) + expect_ns; expect_ns — aukštupio skelbiama commit'o riba (žyma commit-bound-ns) arba --input-expect-ms.
//   Fazė:     E = max(expect_ns) per visus variklio įėjimus (viena fazė varikliui, K-T10); Φ = E + ro; t_open = T + Φ.
//   Išėjimas: commit_bound_ns = Φ + Wr + G (šaltinis — be įėjimų laukimo: Φ + G) — RIBA, ne matavimas (K-T14);
//             content_delay_ns = max(content_delay_in − index_offset · P) — turinio vėlinimas nuo kilmės.
// Grandinėje kiekviena pakopa prideda ro + Wr + G (50 Hz: 0,5 + 5 + 3 = 8,5 ms): žemupys atsidaro, kai aukštupio grain'as
// pagal sutartį jau commit'intas, todėl niekas nevėluoja „pagal konstrukciją“, o vėlinimas kaupiasi aiškiai ir matomai.
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "engine/cuts.hpp"
#include "engine/ports.hpp"

namespace mxlv::engine {

inline constexpr const char* kTagCommitBound = "urn:x-vramflow:tag:commit-bound-ns";
inline constexpr const char* kTagContentDelay = "urn:x-vramflow:tag:content-delay-ns";

struct OutputTiming {
    std::int64_t commit_bound_ns = 0;   // viršutinė commit'o riba po T(i)
    std::int64_t content_delay_ns = 0;  // turinio vėlinimas nuo kilmės
};

// Φ = E + ro (tick'o atidarymas po T(i)).
std::int64_t phase_ns(const CutConfig&);
// Išėjimo commit'o riba po T(i): a(i) + G (a(i) = t_open + Wr, ne vėliau kaip B); šaltinis (vienetas be įėjimų; dažnai legacy
// ribose, kur Wr = D) — t_open + P/4 + G (ta pati formulė su numatytuoju Wr). Spragos commit'inamos ties t_open arba A —
// taip pat ribose.
std::int64_t commit_bound_ns(const CutConfig&, bool source);
// max(content_delay_in − index_offset · P) per įėjimus; be įėjimų — 0.
std::int64_t content_delay_ns(std::span<const EdgeTiming>, timing::Rational rate);

// Rinkinio (video + garsas + key; tas pats grouphint rinkinys) lygiavimas: align = max(content_delay); kiekvienas narys
// skaitomas anksčiau per (align − content_delay_k): video — grain'ais (index_offset priedas, ≤ 0), garsas — mėginiais
// (sample_begin poslinkis, ≤ 0). Išėjimas skelbia content_delay = align.
struct BundleMember {
    std::int64_t content_delay_ns = 0;
    bool audio = false;
};
struct BundleAlign {
    std::int64_t align_ns = 0;
    std::vector<std::int64_t> shift;  // nariui: video — grain'ai, garsas — mėginiai (≤ 0)
};
BundleAlign bundle_align(std::span<const BundleMember>, timing::Rational video_rate, std::int64_t sample_rate = 48000);

// Ring'o gylis: skaitytojas grain'ą j skaito ne vėliau kaip T(j) + lag, lag = E + D_eff − index_offset · P. Rašytojas grain'ą j
// perrašo ties T(j + slots); MXL skaitymas grąžina TOO_LATE, kai j + slots < head + 2 — saugu, kol lag ≤ (slots − 2) · P.
// Refuse — netelpa (funkcija atsisako startuoti, exit 2); Warn — lag > pusė saugaus gylio (mažai atsargos laikrodžio šuoliams).
enum class RingFit : std::uint8_t { Ok, Warn, Refuse };
struct RingCheck {
    RingFit fit = RingFit::Ok;
    std::int64_t lag_ns = 0, safe_ns = 0, history_ns = 0;
    std::string text;  // Warn / Refuse: abu skaičiai ir kaip padidinti domeno istoriją
};
RingCheck ring_check(const CutConfig&, std::int32_t index_offset, std::int64_t history_ns);

// Flow aprašo žymos: ,"urn:x-vramflow:tag:commit-bound-ns":["N"],"urn:x-vramflow:tag:content-delay-ns":["M"] (prieš jas —
// kablelis: dedama po grouphint žymos tame pačiame "tags" objekte).
std::string timing_tags_json(const OutputTiming&);
// Iš flow aprašo JSON: true — rasta commit-bound-ns (content-delay-ns neprivaloma, nėra — 0). Netinkama reikšmė — false.
bool parse_timing_tags(const std::string& flow_def, OutputTiming& out);

}  // namespace mxlv::engine
