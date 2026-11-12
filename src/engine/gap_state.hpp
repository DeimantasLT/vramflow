// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Spragų politika K3 — gryna logika vienam išėjimui (planas §6 D4; sprendimai K-T4, K-T5, K-T6).
//   invalid — INVALID grain'as (turinys neliečiamas);
//   freeze  — paskutinio galiojančio to paties pariteto grain'o kopija iš savo ring'o (interlaced: i − 2,
//             progresyvus: i − 1), ne ilgiau kaip hold_ns nuo paskutinio tikro turinio; paskui — black;
//             be istorijos arba šaltinio nebėra ring'e (i − šaltinis > slot_count − 2) — black;
//   black   — iš anksto paruoštas juodas grain'as;
//   leave   — nieko (tik migracijai).
// Variklio paties praleisti indeksai (T11) čia nepatenka — jų nepildome (K-T6), todėl jie „nutraukia“ kartojimo
// grandinę tik per atstumo patikrą.
#include <cstdint>

#include "engine/clock.hpp"

namespace mxlv::engine {

enum class GapPolicy : std::uint8_t { Leave, Invalid, Freeze, Black };
enum class GapAction : std::uint8_t { None, Invalid, Repeat, Black };

const char* to_string(GapPolicy);
const char* to_string(GapAction);

struct GapState {
    GapPolicy policy = GapPolicy::Invalid;
    std::int64_t hold_ns = 1'000'000'000;

    // Veiksmas indeksui i; repeat_from — šaltinio indeksas (tik Repeat).
    GapAction decide(Index i, std::int64_t t_index_ns, bool interlaced, std::uint32_t slot_count, Index& repeat_from);
    // freeze / black + interlaced → po spragos tik lyginis i (kadro riba, K-T5); kitaip — visada.
    bool may_resume(Index i, bool interlaced) const;
    void on_content(Index i, std::int64_t t_index_ns);  // commit'intas tikras turinys
    void on_gap(Index i, GapAction done);                // commit'inta spraga (veiksmas, kurį tikrai atliko)

private:
    bool has_good_ = false;
    bool in_gap_ = false;
    std::int64_t t_good_ = 0;
    Index last_[2] = {kNoIndex, kNoIndex};  // paskutinis galiojantis (turinys ar kopija) pagal paritetą
    Index last_any_ = kNoIndex;
    Index last_gap_ = kNoIndex;  // paskutinė spraga ir jos veiksmas (kadro ribos taisyklei)
    GapAction last_act_ = GapAction::None;
};

}  // namespace mxlv::engine
