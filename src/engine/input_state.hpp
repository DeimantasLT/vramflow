// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Įėjimo būsenų automatas (planas §6 D2 „Perėjimai“) — gryna logika, be laiko ir be MXL. Kada kviečiamas kuris
// perėjimas, sprendžia variklis (ribos A / B / C); čia — tik būsena, žymė Late, miss_run ir skaitikliai.
// stall_after == 0 — pertraukiklis ir Late žymės išjungti (legacy).
#include <cstdint>

#include "engine/clock.hpp"
#include "engine/counters.hpp"

namespace mxlv::engine {

enum class InState : std::uint8_t { Disabled, Waiting, Live, Stalled, Detached, Mismatch };

class InputState {
public:
    explicit InputState(std::uint32_t stall_after = 3) : m_(stall_after) {}

    InState state() const { return st_; }
    bool late() const { return late_; }
    std::uint32_t miss_run() const { return miss_; }
    bool live() const { return st_ == InState::Live; }

    // Waiting / Detached → Live + Late (reattached++ tik iš Detached).
    void on_attach(InCounters& c) {
        if (st_ == InState::Detached) ++c.reattached;
        st_ = InState::Live;
        late_ = m_ != 0;
        miss_ = 0;
        head_ = kNoIndex;
    }
    // Waiting / Detached, formatas kitas, politika exit → Mismatch (galutinė).
    void on_mismatch() { st_ = InState::Mismatch; }
    // Live: paruoštas A paleidimo metu → be Late, miss_run = 0, ok++.
    void on_ready_at_a(InCounters& c) {
        late_ = false;
        miss_ = 0;
        ++c.ok;
    }
    // Live: neparuoštas ties A. Negyvas rašytojas → Detached (detaches++, lost++); gyvas → Late.
    void on_unready_at_a(bool writer_alive, InCounters& c) {
        if (!writer_alive) {
            detach(c);
            ++c.lost;
            return;
        }
        if (m_ != 0) late_ = true;
    }
    // Live (+ Late): paruoštas po A, iki paskutinės ribos (B arba C) → late++, miss_run = 0.
    void on_ready_late(InCounters& c) {
        miss_ = 0;
        ++c.late;
    }
    // Live (+ Late): neparuoštas iki paskutinės ribos → lost++; ++miss_run ≥ M → Stalled (stalls++). true — Stalled.
    bool on_missed(InCounters& c) {
        ++c.lost;
        if (m_ == 0) return false;
        late_ = true;
        if (++miss_ < m_) return false;
        st_ = InState::Stalled;
        head_ = kNoIndex;
        ++c.stalls;
        return true;
    }
    // get → FLOW_INVALID (Gone) → Detached.
    void on_gone(InCounters& c) { detach(c); }
    void on_too_late(InCounters& c) { ++c.too_late; }
    void on_invalid_grain(InCounters& c) { ++c.invalid; }
    void on_read_error(InCounters& c) { ++c.read_errors; }
    // Stalled: head padidėjo nuo praėjusio skaitymo → Live + Late, miss_run = 0. Pirmas skaitymas — tik atskaita
    // (nebent jau buvo žinomas head prieš Stalled). true — perėjo į Live.
    bool on_stalled_head(Index head) {
        if (st_ != InState::Stalled) return false;
        const Index prev = head_;
        head_ = head;
        if (head == kNoIndex || prev == kNoIndex || head <= prev) return false;
        st_ = InState::Live;
        late_ = m_ != 0;
        miss_ = 0;
        return true;
    }
    // Paskutinis žinomas head (variklis jį nustato užstringant, kad pirmas Stalled skaitymas jau lygintų).
    void note_head(Index head) { head_ = head; }
    // Stalled: rašytojas negyvas → Detached.
    void on_stalled_dead(InCounters& c) { detach(c); }
    void set_enabled(bool on) {
        st_ = on ? InState::Waiting : InState::Disabled;
        late_ = false;
        miss_ = 0;
    }
    void on_retarget() {
        st_ = InState::Waiting;
        late_ = false;
        miss_ = 0;
    }

private:
    void detach(InCounters& c) {
        st_ = InState::Detached;
        late_ = false;
        miss_ = 0;
        ++c.detaches;
    }

    std::uint32_t m_;
    InState st_ = InState::Waiting;
    bool late_ = false;
    std::uint32_t miss_ = 0;
    Index head_ = kNoIndex;
};

inline const char* to_string(InState s) {
    switch (s) {
        case InState::Disabled: return "Disabled";
        case InState::Waiting: return "Waiting";
        case InState::Live: return "Live";
        case InState::Stalled: return "Stalled";
        case InState::Detached: return "Detached";
        case InState::Mismatch: return "Mismatch";
    }
    return "?";
}

}  // namespace mxlv::engine
