// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Įėjimo flow laukimo politika (planas §P6): paleidimo tvarka nesvarbi — vartotojas gali startuoti prieš gamintoją.
#include <cstdint>

namespace mxlv::runtime {

// wait_ms: < 0 — laukti neribotai (numatyta), 0 — vienas bandymas, > 0 — riba milisekundėmis.
// log_every_ms: kas kiek ms rašoma į stderr „laukiu flow X, N s“ (0 — nerašoma).
struct OpenPolicy {
    std::int64_t wait_ms = -1;
    std::uint32_t log_every_ms = 5000;
};

// CLI `--input-wait-ms N`: 0 — neribotai (numatyta), N > 0 — riba (pasibaigus — aiški klaida ir exit 4).
constexpr OpenPolicy policy_from_cli(std::int64_t input_wait_ms) { return OpenPolicy{input_wait_ms > 0 ? input_wait_ms : -1, 5000}; }

// Įėjimo formatas pasikeitė po re-attach (PF-3): Exit — slot'as tampa Mismatch (funkcija baigiasi exit 6),
// Wait — slot'as lieka atjungtas, kol gamintojas grįžta su tuo pačiu formatu.
enum class FormatChange { Exit, Wait };

}  // namespace mxlv::runtime
