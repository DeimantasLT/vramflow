// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/gap_state.hpp"

namespace mxlv::engine {

GapAction GapState::decide(Index i, std::int64_t t_index_ns, bool interlaced, std::uint32_t slot_count,
                           Index& repeat_from) {
    switch (policy) {
        case GapPolicy::Leave: return GapAction::None;
        case GapPolicy::Invalid: return GapAction::Invalid;
        case GapPolicy::Black: return GapAction::Black;
        case GapPolicy::Freeze: break;
    }
    if (!has_good_) return GapAction::Black;  // K-T4
    const Index src = interlaced ? last_[i & 1] : last_any_;
    if (t_index_ns - t_good_ > hold_ns) {
        // hold pasibaigė. Interlaced: perjungiama tik kadro riba — apatinis laukas (nelyginis i) kartojamas, jei
        // viršutinis (i − 1) dar buvo kartojimas; kitaip kadras būtų sustingęs viršus + juodas apačia.
        const bool finish_frame = interlaced && (i & 1) == 1 && last_gap_ == i - 1 && last_act_ == GapAction::Repeat;
        if (!finish_frame) return GapAction::Black;
    }
    if (src == kNoIndex || src >= i) return GapAction::Black;
    // Šaltinis turi būti dar ring'e su atsarga (planas D4: step ≤ 2 ≤ slotCount − 2)
    if (slot_count < 2 || i - src > slot_count - 2) return GapAction::Black;
    repeat_from = src;
    return GapAction::Repeat;
}

bool GapState::may_resume(Index i, bool interlaced) const {
    if (!in_gap_ || !interlaced) return true;
    if (policy != GapPolicy::Freeze && policy != GapPolicy::Black) return true;
    return (i & 1) == 0;
}

void GapState::on_content(Index i, std::int64_t t_index_ns) {
    has_good_ = true;
    in_gap_ = false;
    t_good_ = t_index_ns;
    last_[i & 1] = i;
    last_any_ = i;
}

void GapState::on_gap(Index i, GapAction done) {
    in_gap_ = true;
    last_gap_ = i;
    last_act_ = done;
    if (done == GapAction::Repeat || done == GapAction::Black) {
        last_[i & 1] = i;
        last_any_ = i;
    } else {
        last_[i & 1] = kNoIndex;
        last_any_ = kNoIndex;
    }
}

const char* to_string(GapPolicy p) {
    switch (p) {
        case GapPolicy::Leave: return "leave";
        case GapPolicy::Invalid: return "invalid";
        case GapPolicy::Freeze: return "freeze";
        case GapPolicy::Black: return "black";
    }
    return "?";
}

const char* to_string(GapAction a) {
    switch (a) {
        case GapAction::None: return "None";
        case GapAction::Invalid: return "Invalid";
        case GapAction::Repeat: return "Repeat";
        case GapAction::Black: return "Black";
    }
    return "?";
}

}  // namespace mxlv::engine
