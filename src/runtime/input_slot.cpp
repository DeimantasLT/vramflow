// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/input_slot.hpp"

#include <mxl/time.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace mxlv::runtime {

namespace {
// Gyvas writer'is (MXL: writer'io flock ant flow duomenų failo). Flow nėra / klaida — false.
bool flow_active(mxlInstance inst, const std::string& id) {
    bool active = false;
    return mxlIsFlowActive(inst, id.c_str(), &active) == MXL_STATUS_OK && active;
}

std::string storage_name(bool cuda) { return cuda ? "cuda-linear" : "host"; }
}  // namespace

const char* state_name(InputSlot::State s) {
    switch (s) {
        case InputSlot::State::Waiting: return "Waiting";
        case InputSlot::State::Attached: return "Attached";
        case InputSlot::State::Detached: return "Detached";
        case InputSlot::State::Mismatch: return "Mismatch";
    }
    return "?";
}

InputSlot::InputSlot(std::string flow_id, const vramflow::VideoFormat* expect, FormatChange on_change)
    : id_(std::move(flow_id)), have_want_(expect != nullptr), on_change_(on_change) {
    if (expect) want_ = *expect;
}

InputSlot::InputSlot(InputSlot&& o) noexcept { *this = std::move(o); }

InputSlot& InputSlot::operator=(InputSlot&& o) noexcept {
    if (this == &o) return *this;
    release();
    id_ = std::move(o.id_);
    want_ = o.want_;
    have_want_ = o.have_want_;
    want_cuda_ = o.want_cuda_;
    lock_storage_ = o.lock_storage_;
    ever_ = o.ever_;
    on_change_ = o.on_change_;
    state_ = o.state_;
    in_ = std::move(o.in_);
    inst_ = o.inst_;
    next_try_ = o.next_try_;
    next_alive_ = o.next_alive_;
    early_since_ = o.early_since_;
    reattached_ = o.reattached_;
    detach_head_ = o.detach_head_;
    detach_tai_ = o.detach_tai_;
    reason_ = std::move(o.reason_);
    o.in_.r = nullptr;
    o.state_ = State::Waiting;
    return *this;
}

InputSlot::~InputSlot() { release(); }

void InputSlot::release() {
    if (in_.r && inst_) in_.release(inst_);
    in_.r = nullptr;
}

void InputSlot::detach() {
    // Re-attach sąlygai: head ir momentas atjungimo metu (flow perkurtas — head neskaitomas: 0, tada lemia lastWriteTime)
    const std::uint64_t h = in_.r ? in_.head() : MXL_UNDEFINED_INDEX;
    detach_head_ = h == MXL_UNDEFINED_INDEX ? 0 : h;
    detach_tai_ = mxlGetTime();
    release();
    early_since_ = 0;
    state_ = State::Detached;
    next_try_ = 0;  // kitas poll bando iškart (gamintojas gali jau būti grįžęs)
}

InputSlot::State InputSlot::poll(mxlInstance inst, std::uint64_t now_tai_ns) {
    if (state_ == State::Attached || state_ == State::Mismatch) return state_;
    if (now_tai_ns < next_try_) return state_;
    next_try_ = now_tai_ns + kRetryNs;
    inst_ = inst;
    // Tik gyvas writer'is: nužudyto gamintojo pasenęs flow (failai liko) neprijungiamas.
    if (!flow_active(inst, id_)) {
        reason_.clear();  // flow nėra (arba be writer'io) — ankstesnė atmetimo priežastis nebegalioja
        return state_;
    }
    func::VideoIn fresh;
    std::string err;
    static const std::atomic<bool> no_stop{false};
    if (!fresh.open(inst, id_, OpenPolicy{0, 0}, no_stop, err, static_cast<const vramflow::VideoFormat*>(nullptr))) {
        fresh.release(inst);
        reason_ = "flow " + id_ + " yra, bet reader'is neatsidaro: " + err;  // matoma laukimo eilutėje ir ribos klaidoje
        return state_;
    }
    // Formatas / saugykla: expect (arba pirmas prijungtas formatas) ir pirmo prijungimo saugykla.
    std::string bad;
    if (have_want_ && !(fresh.fmt.raster == want_.raster && vramflow::same_rate(fresh.fmt.grain_rate(), want_.grain_rate())))
        bad = "flow " + id_ + ": formatas " + vramflow::format_name(fresh.fmt) + " != laukiamas " + vramflow::format_name(want_);
    else if ((ever_ || lock_storage_) && fresh.cuda != want_cuda_)
        bad = "flow " + id_ + ": saugykla " + storage_name(fresh.cuda) + " != " + (ever_ ? "buvusi " : "laukiama ") + storage_name(want_cuda_);
    if (!bad.empty()) {
        fresh.release(inst);
        if (reason_ != bad) std::fprintf(stderr, "[vramflow] %s%s\n", bad.c_str(), on_change_ == FormatChange::Exit ? "" : " — laukiama");
        reason_ = bad;
        if (on_change_ == FormatChange::Exit) state_ = State::Mismatch;
        return state_;
    }
    // Gamintojas jau gamina (head žinomas) — kitaip pirmi tick'ai būtų vėlyvi; bandoma vėl po kRetryNs.
    const std::uint64_t h = fresh.head();
    if (h == MXL_UNDEFINED_INDEX || h == 0) {
        fresh.release(inst);
        return state_;
    }
    // Po atjungimo — tik jei flow'as nuo tada rašomas (head pajudėjo arba lastWriteTime šviežias): kitaip „gyvas“ buvo tik
    // kito skaitytojo flock (mxlIsFlowActive), o flow'as — pasenęs.
    if (state_ == State::Detached) {
        mxlFlowRuntimeInfo rt{};
        const bool info = mxlFlowReaderGetRuntimeInfo(fresh.r, &rt) == MXL_STATUS_OK;
        if (!(h > detach_head_ || (info && rt.lastWriteTime > detach_tai_))) {
            fresh.release(inst);
            return state_;
        }
    }
    // Pirmas prisijungimas (peržiūra D6): ta pati taisyklė — tik flow'as, kuris RAŠOMAS (lastWriteTime ne senesnis kaip
    // kStaleNs arba head pajudėjo nuo ankstesnio bandymo). Kitaip nužudyto gamintojo pasenęs flow + kito skaitytojo flock
    // („gyvas“ mxlIsFlowActive) būtų prijungtas ir pirmi tick'ai skaitytų seną ring'ą.
    if (state_ == State::Waiting) {
        mxlFlowRuntimeInfo rt{};
        const bool info = mxlFlowReaderGetRuntimeInfo(fresh.r, &rt) == MXL_STATUS_OK;
        const bool recent = info && rt.lastWriteTime + kStaleNs >= now_tai_ns;
        const bool moved = waiting_head_ != 0 && h > waiting_head_;
        waiting_head_ = h;
        if (!(recent || moved)) {
            fresh.release(inst);
            return state_;
        }
    }
    in_ = std::move(fresh);
    if (!have_want_) {
        want_ = in_.fmt;
        have_want_ = true;
    }
    if (!ever_ && !lock_storage_) want_cuda_ = in_.cuda;
    if (ever_) ++reattached_;
    ever_ = true;
    reason_.clear();
    next_alive_ = 0;
    early_since_ = 0;
    state_ = State::Attached;
    return state_;
}

mxlStatus InputSlot::get(std::uint64_t index, std::uint64_t timeout_ns, mxlGrainInfo& info, const std::uint32_t*& ptr) {
    ptr = nullptr;
    if (state_ != State::Attached) return MXL_ERR_FLOW_NOT_FOUND;
    const mxlStatus st = in_.get(index, timeout_ns, info, ptr);
    if (st == MXL_ERR_FLOW_INVALID) {  // flow perkurtas / ištrintas
        detach();
        return st;
    }
    if (st != MXL_ERR_OUT_OF_RANGE_TOO_EARLY) {
        early_since_ = 0;
        return st;
    }
    // Grain'o nėra jau kAliveNs iš eilės: ar writer'is gyvas? (nužudytas gamintojas palieka flow failus — MXL to nemato.)
    // Trumpiau vėluojantis gamintojas netikrinamas — įprastame tick'e sisteminio kvietimo nėra.
    const std::uint64_t now = mxlGetTime();
    if (!early_since_) early_since_ = now;
    if (now - early_since_ >= kAliveNs && now >= next_alive_) {
        next_alive_ = now + kAliveNs;
        if (!flow_active(inst_, id_)) {
            detach();
            return MXL_ERR_FLOW_INVALID;
        }
    }
    return st;
}

bool InputSlot::writer_alive() const { return inst_ != nullptr && flow_active(inst_, id_); }

std::uint64_t InputSlot::head() { return state_ == State::Attached ? in_.head() : MXL_UNDEFINED_INDEX; }

bool wait_attached(const std::vector<InputSlot*>& slots, mxlInstance inst, const OpenPolicy& p, const std::atomic<bool>& stop,
                   const char* who, std::string& err) {
    using namespace std::chrono;
    const auto t0 = steady_clock::now();
    auto next_log = t0 + milliseconds(p.log_every_ms);
    for (;;) {
        bool all = true;
        const std::uint64_t now = mxlGetTime();
        for (InputSlot* s : slots) {
            const InputSlot::State st = s->poll(inst, now);
            if (st == InputSlot::State::Mismatch) { err = s->reason(); return false; }
            all = all && st == InputSlot::State::Attached;
        }
        if (all) return true;
        const std::string missing = missing_inputs(slots);
        if (stop.load()) { err = "sustabdyta laukiant įėjimų: " + missing; return false; }
        const auto t = steady_clock::now();
        if (p.wait_ms == 0 || (p.wait_ms > 0 && t - t0 >= milliseconds(p.wait_ms))) {
            err = inputs_timeout_text(slots, p.wait_ms);
            return false;
        }
        if (p.log_every_ms && t >= next_log) {
            std::fprintf(stderr, "[%s] laukiu flow %s, %lld s\n", who, missing.c_str(),
                         static_cast<long long>(duration_cast<seconds>(t - t0).count()));
            next_log += milliseconds(p.log_every_ms);
        }
        std::this_thread::sleep_for(milliseconds(20));
    }
}

std::string missing_inputs(const std::vector<InputSlot*>& slots) {
    std::string missing;
    for (InputSlot* s : slots)
        if (s->state() != InputSlot::State::Attached) missing += (missing.empty() ? "" : ", ") + s->id();
    // Kodėl neprijungta, jei flow yra, bet netinka (kitas formatas, reader'io klaida): be to laukimas atrodytų kaip „nėra flow“.
    for (InputSlot* s : slots)
        if (s->state() != InputSlot::State::Attached && !s->reason().empty()) missing += " [" + s->reason() + "]";
    return missing;
}

std::string inputs_timeout_text(const std::vector<InputSlot*>& slots, std::int64_t wait_ms) {
    return "įėjimai neatsirado per " + std::to_string(wait_ms) + " ms (--input-wait-ms " + std::to_string(wait_ms) +
           "; 0 — laukti neribotai): " + missing_inputs(slots);
}

}  // namespace mxlv::runtime
