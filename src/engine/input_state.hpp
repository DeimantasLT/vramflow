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

// „Laiku“ — įėjimo grain'as paruoštas ne vėliau kaip A riba a(i) = T(i) + Φ + Wr (nesvarbu, kada A iš tikrųjų
// paleista). Late žymė nuimama tik po late_clear_run laiku atėjusių grain'ų iš eilės (histerezė prieš „plazdėjimą“).
class InputState {
public:
    explicit InputState(std::uint32_t stall_after = 3, std::uint32_t late_clear_run = 8)
        : m_(stall_after), clear_(late_clear_run == 0 ? 1 : late_clear_run) {}

    InState state() const { return st_; }
    bool late() const { return late_; }
    std::uint32_t miss_run() const { return miss_; }
    std::uint32_t on_time_run() const { return ontime_; }
    bool live() const { return st_ == InState::Live; }

    // Waiting / Detached → Live + Late (reattached++ tik iš Detached).
    void on_attach(InCounters& c) {
        if (st_ == InState::Detached) ++c.reattached;
        st_ = InState::Live;
        set_late();
        miss_ = 0;
        head_ = kNoIndex;
    }
    // Waiting / Detached, formatas kitas, politika exit → Mismatch (galutinė).
    void on_mismatch() { st_ = InState::Mismatch; }
    // Live: grain'as gautas. on_time → ok++ (Late nuimama po late_clear_run iš eilės); kitaip late++ ir Late.
    // Abiem atvejais miss_run = 0.
    void on_ready(InCounters& c, bool on_time) {
        miss_ = 0;
        if (!on_time) {
            ++c.late;
            set_late();
            return;
        }
        ++c.ok;
        if (late_ && ++ontime_ >= clear_) {
            late_ = false;
            ontime_ = 0;
        }
    }
    // Live: neparuoštas ties A (įėjimas A laukė). Negyvas rašytojas → Detached (detaches++, lost++); gyvas → Late.
    void on_unready_at_a(bool writer_alive, InCounters& c) {
        if (!writer_alive) {
            detach(c);
            ++c.lost;
            return;
        }
        set_late();
    }
    // Live (+ Late): neparuoštas iki paskutinės ribos → lost++; ++miss_run ≥ M → Stalled (stalls++). true — Stalled.
    bool on_missed(InCounters& c) {
        ++c.lost;
        if (m_ == 0) return false;
        set_late();
        if (++miss_ < m_) return false;
        st_ = InState::Stalled;
        head_ = kNoIndex;
        ++c.stalls;
        return true;
    }
    // get → FLOW_INVALID (Gone) → Detached.
    void on_gone(InCounters& c) { detach(c); }
    // TOO_LATE: vėlavome mes, ne įėjimas — ne praradimas įėjimui, miss_run = 0 (rašytojas gamina).
    void on_too_late(InCounters& c) {
        ++c.too_late;
        miss_ = 0;
    }
    // OK + INVALID: rašytojas gyvas ir pateikė grain'ą (aukštupio spraga) — ne praleidimas, miss_run = 0.
    void on_invalid_grain(InCounters& c) {
        ++c.invalid;
        miss_ = 0;
    }
    void on_read_error(InCounters& c) { ++c.read_errors; }
    // Stalled: head padidėjo nuo praėjusio skaitymo → Live + Late, miss_run = 0. Pirmas skaitymas — tik atskaita
    // (nebent jau buvo žinomas head prieš Stalled). true — perėjo į Live.
    bool on_stalled_head(Index head) {
        if (st_ != InState::Stalled) return false;
        const Index prev = head_;
        head_ = head;
        if (head == kNoIndex || prev == kNoIndex || head <= prev) return false;
        st_ = InState::Live;
        set_late();
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
        miss_ = ontime_ = 0;
    }
    void on_retarget() {
        st_ = InState::Waiting;
        late_ = false;
        miss_ = ontime_ = 0;
    }

private:
    void set_late() {
        late_ = m_ != 0;
        ontime_ = 0;
    }
    void detach(InCounters& c) {
        st_ = InState::Detached;
        late_ = false;
        miss_ = ontime_ = 0;
        ++c.detaches;
    }

    std::uint32_t m_, clear_;
    InState st_ = InState::Waiting;
    bool late_ = false;
    std::uint32_t miss_ = 0, ontime_ = 0;
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
