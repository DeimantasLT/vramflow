// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// InputSlot — vienas video įėjimas, kuris gali atsirasti, dingti ir grįžti (planas §P6; anksčiau — composite [PF-3] logika).
//
// Būsenos: Waiting (dar niekada neprijungtas), Attached, Detached (buvo prijungtas, gamintojas dingo), Mismatch (grįžo kitu
// formatu / saugykla ir politika Exit — galutinė). Prisijungiama tik prie flow, kuris turi gyvą writer'į (mxlIsFlowActive) ir
// jau gamina (head žinomas): pasenęs nužudyto gamintojo flow ignoruojamas.
// Iš naujo (po Detached) — tik jei nuo atjungimo flow'as iš tikrųjų rašomas: head pajudėjo už atjungimo head'ą ARBA
// lastWriteTime vėlesnis už atjungimo momentą (peržiūra 2026-11-13 p. 5). mxlIsFlowActive pats ima flock(LOCK_EX | LOCK_NB):
// kitas procesas, tuo pat metu tikrinantis tą patį NEGYVĄ flow'ą, atrodo kaip „gyvas writer'is“ — be šios sąlygos slot'as
// prisijungdavo prie pasenusio flow'o ir skaičiuodavo klaidingą re-attach.
//
// Gamintojo dingimas aptinkamas dviem keliais: MXL_ERR_FLOW_INVALID (flow perkurtas / ištrintas) ir gyvumo patikra, kai
// grain'o nėra (TOO_EARLY) jau ilgiau kaip kAliveNs iš eilės (writer'is nužudytas, flow failai liko), tada ne dažniau kaip
// kas kAliveNs. Kol gamintojas vėluoja mažiau nei kAliveNs, patikros (sisteminio kvietimo) nėra visai — tick'ui tai nieko
// nekainuoja. Abiem atvejais get() grąžina MXL_ERR_FLOW_INVALID, reader'is atlaisvinamas, būsena Detached; atjungtas įėjimas
// kitų flow'ų nebevėlina (get iškart grąžina MXL_ERR_FLOW_NOT_FOUND).
//
// Formatas: expect != nullptr — laukiamas formatas (rastras ir grain'o ritmas); nullptr — pirmas prijungtas formatas tampa
// laukiamu. Saugykla (cuda / host) visada užrakinama pagal pirmą prijungimą. Viena gija (tick gija) — be sinchronizacijos.
#include <mxl/flow.h>
#include <mxl/mxl.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "common/format.hpp"
#include "func/flows.hpp"
#include "runtime/open_policy.hpp"

namespace mxlv::runtime {

class InputSlot {
public:
    enum class State { Waiting, Attached, Detached, Mismatch };
    static constexpr std::uint64_t kRetryNs = 100'000'000;  // prisijungimo bandymas ne dažniau kaip kas 100 ms
    static constexpr std::uint64_t kAliveNs = 10'000'000;   // gyvumo patikra po 10 ms TOO_EARLY iš eilės, tada kas 10 ms

    InputSlot(std::string flow_id, const vramflow::VideoFormat* expect, FormatChange on_change);
    InputSlot(InputSlot&& o) noexcept;  // perkeltas slot'as reader'io nebeturi (neatlaisvina dar kartą)
    InputSlot& operator=(InputSlot&& o) noexcept;
    InputSlot(const InputSlot&) = delete;
    InputSlot& operator=(const InputSlot&) = delete;
    ~InputSlot();

    // Neblokuoja. Kai neprijungtas — bandymas prisijungti (ne dažniau kaip kas kRetryNs pagal now_tai_ns). Attached — tik
    // grąžina būseną (be sisteminių kvietimų).
    State poll(mxlInstance inst, std::uint64_t now_tai_ns);
    // Kaip VideoIn::get; ne Attached → MXL_ERR_FLOW_NOT_FOUND. Gamintojas dingo → MXL_ERR_FLOW_INVALID ir būsena Detached.
    mxlStatus get(std::uint64_t index, std::uint64_t timeout_ns, mxlGrainInfo& info, const std::uint32_t*& ptr);
    std::uint64_t head();  // MXL_UNDEFINED_INDEX, jei neprijungtas / nežinomas

    State state() const { return state_; }
    const std::string& id() const { return id_; }
    // Laukiamas formatas (po pirmo prijungimo — faktinis).
    const vramflow::VideoFormat& format() const { return want_; }
    std::uint64_t reattached() const { return reattached_; }
    // Laukiama saugykla iš parinkčių (--start immediate: išėjimai sukuriami prieš įėjimus, V18): kita saugykla atmetama jau
    // pirmą kartą (Mismatch / politika Wait — laukti), ne užrakinama pagal pirmą prijungimą. Kviesti prieš pirmą poll.
    void expect_storage(bool cuda) {
        want_cuda_ = cuda;
        lock_storage_ = true;
    }
    // Paskutinio atmetimo priežastis (Mismatch tekstas su abiem formatais); tuščia, jei nebuvo.
    const std::string& reason() const { return reason_; }
    // Gyvas writer'is (mxlIsFlowActive: open + flock + close, M9) — sisteminiai kvietimai; kviesti retai (variklis:
    // ties A riba ir kas 100 ms Stalled būsenoje). Neprijungtas — false.
    bool writer_alive() const;
    // Atjungia iškart (kaip gamintojo dingimas): reader'is atlaisvinamas, būsena Detached, kitas poll bando iškart.
    void detach_now() { detach(); }
    func::VideoIn& in() { return in_; }
    const func::VideoIn& in() const { return in_; }
    void release();

private:
    void detach();
    std::string id_;
    vramflow::VideoFormat want_{};
    bool have_want_ = false;
    bool want_cuda_ = false;
    bool lock_storage_ = false;  // expect_storage: want_cuda_ žinomas iš parinkčių
    bool ever_ = false;  // buvo prijungtas bent kartą
    FormatChange on_change_ = FormatChange::Exit;
    State state_ = State::Waiting;
    func::VideoIn in_;
    mxlInstance inst_ = nullptr;
    std::uint64_t next_try_ = 0;
    std::uint64_t next_alive_ = 0;
    std::uint64_t early_since_ = 0;  // nuo kada grain'o nėra (TOO_EARLY iš eilės); 0 — paskutinis skaitymas pavyko
    std::uint64_t reattached_ = 0;
    std::uint64_t detach_head_ = 0, detach_tai_ = 0;  // paskutinio atjungimo head ir momentas (re-attach sąlygai)
    std::string reason_;
};

const char* state_name(InputSlot::State s);

// Pradinis prisijungimas: laukia, kol VISI slot'ai Attached, pagal politiką (kas p.log_every_ms — eilutė į stderr su `who`).
// false: err — laiko riba (su --input-wait-ms paaiškinimu), Mismatch priežastis arba sustabdymas.
bool wait_attached(const std::vector<InputSlot*>& slots, mxlInstance inst, const OpenPolicy& p, const std::atomic<bool>& stop,
                   const char* who, std::string& err);
// Neprijungtų slot'ų ID (per kablelį) ir, jei yra, kodėl flow'as neprijungtas („ [priežastis]“) — laukimo eilutei ir klaidai.
std::string missing_inputs(const std::vector<InputSlot*>& slots);
// „įėjimai neatsirado per N ms (--input-wait-ms N; 0 — laukti neribotai): <missing_inputs>“ — wait_attached ir variklio
// InputsTimeout (--start immediate) tekstas.
std::string inputs_timeout_text(const std::vector<InputSlot*>& slots, std::int64_t wait_ms);

}  // namespace mxlv::runtime
