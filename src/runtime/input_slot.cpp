// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/input_slot.hpp"

#include <mxl/time.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <utility>

namespace mxlv::runtime {

namespace {
// Gyvas writer'is (MXL: writer'io flock ant flow duomenų failo). Flow nėra / klaida — false.
bool flow_active(mxlInstance inst, const std::string& id) {
    bool active = false;
    return mxlIsFlowActive(inst, id.c_str(), &active) == MXL_STATUS_OK && active;
}

std::string storage_name(bool cuda) { return cuda ? "cuda-linear" : "host"; }
}  // namespace

// Vienas prisijungimo bandymas: prašymas (slot'o būsenos nuotrauka) → rezultatas. try_attach neliečia slot'o, todėl gali
// vykti pagalbinėje gijoje; rezultatą slot'ui pritaiko apply (kviečianti gija).
struct InputSlot::Attempt {
    enum class Kind { None, NotYet, Rejected, Ok };
    // prašymas
    mxlInstance inst = nullptr;
    std::string id;
    bool have_want = false, check_storage = false, want_cuda = false, ever = false, detached = false, waiting = false;
    vramflow::VideoFormat want{};
    std::uint64_t detach_head = 0, detach_tai = 0, waiting_head = 0, now = 0;
    // rezultatas
    Kind kind = Kind::None;
    bool clear_reason = false, set_waiting_head = false;
    std::string reason;  // Rejected: formato / saugyklos atmetimas; NotYet: reader'io klaida (laukimo eilutei) arba tuščia
    func::VideoIn in;    // Ok
};

// --- Pagalbinė gija (poll_async) ------------------------------------------------------------------------------------
struct InputSlot::Helper {
    std::mutex m;
    std::condition_variable cv;
    bool pending = false, done = false, quit = false;
    Attempt a;
    std::vector<std::pair<mxlInstance, func::VideoIn>> release;  // atjungti reader'iai (atlaisvinami čia, ne tick'e)
    std::thread th;

    void run() {
        std::unique_lock lk(m);
        for (;;) {
            cv.wait(lk, [&] { return quit || pending || !release.empty(); });
            while (!release.empty()) {
                auto r = std::move(release.back());
                release.pop_back();
                lk.unlock();
                r.second.release(r.first);
                lk.lock();
            }
            if (pending) {
                Attempt x = std::move(a);
                lk.unlock();
                InputSlot::try_attach(x);
                lk.lock();
                a = std::move(x);
                x.in.r = nullptr;
                pending = false;
                done = true;
            }
            if (quit && release.empty() && !pending) {
                if (done && a.kind == Attempt::Kind::Ok && a.in.r) a.in.release(a.inst);  // paruoštas, bet neperimtas
                return;
            }
        }
    }
};

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
    stop_helper();
    release();
    helper_ = std::move(o.helper_);
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

InputSlot::~InputSlot() {
    stop_helper();
    release();
}

void InputSlot::release() {
    if (in_.r && inst_) in_.release(inst_);
    in_.r = nullptr;
}

void InputSlot::detach() {
    // Re-attach sąlygai: head ir momentas atjungimo metu (flow perkurtas — head neskaitomas: 0, tada lemia lastWriteTime)
    const std::uint64_t h = in_.r ? in_.head() : MXL_UNDEFINED_INDEX;
    detach_head_ = h == MXL_UNDEFINED_INDEX ? 0 : h;
    detach_tai_ = mxlGetTime();
    if (helper_ && in_.r && inst_) {  // poll_async režimas: reader'io (cuda-linear atvaizdavimo) atlaisvinimas — ne tick'e
        {
            std::lock_guard g(helper_->m);
            helper_->release.emplace_back(inst_, std::move(in_));
        }
        in_.r = nullptr;
        helper_->cv.notify_one();
    } else {
        release();
    }
    early_since_ = 0;
    state_ = State::Detached;
    next_try_ = 0;  // kitas poll bando iškart (gamintojas gali jau būti grįžęs)
}

// Brangioji dalis: gyvumas, reader'is (cuda-linear — payload'o atvaizdavimas), formatas, head ir „rašomas“ sąlygos.
void InputSlot::try_attach(Attempt& a) {
    a.kind = Attempt::Kind::NotYet;
    // Tik gyvas writer'is: nužudyto gamintojo pasenęs flow (failai liko) neprijungiamas.
    if (!flow_active(a.inst, a.id)) {
        a.clear_reason = true;  // flow nėra (arba be writer'io) — ankstesnė atmetimo priežastis nebegalioja
        return;
    }
    func::VideoIn fresh;
    std::string err;
    static const std::atomic<bool> no_stop{false};
    if (!fresh.open(a.inst, a.id, OpenPolicy{0, 0}, no_stop, err, static_cast<const vramflow::VideoFormat*>(nullptr))) {
        fresh.release(a.inst);
        a.reason = "flow " + a.id + " yra, bet reader'is neatsidaro: " + err;  // matoma laukimo eilutėje ir ribos klaidoje
        return;
    }
    // Formatas / saugykla: expect (arba pirmas prijungtas formatas) ir pirmo prijungimo saugykla.
    std::string bad;
    if (a.have_want && !(fresh.fmt.raster == a.want.raster && vramflow::same_rate(fresh.fmt.grain_rate(), a.want.grain_rate())))
        bad = "flow " + a.id + ": formatas " + vramflow::format_name(fresh.fmt) + " != laukiamas " + vramflow::format_name(a.want);
    else if (a.check_storage && fresh.cuda != a.want_cuda)
        bad = "flow " + a.id + ": saugykla " + storage_name(fresh.cuda) + " != " + (a.ever ? "buvusi " : "laukiama ") +
              storage_name(a.want_cuda);
    if (!bad.empty()) {
        fresh.release(a.inst);
        a.kind = Attempt::Kind::Rejected;
        a.reason = bad;
        return;
    }
    // Gamintojas jau gamina (head žinomas) — kitaip pirmi tick'ai būtų vėlyvi; bandoma vėl po kRetryNs.
    const std::uint64_t h = fresh.head();
    if (h == MXL_UNDEFINED_INDEX || h == 0) {
        fresh.release(a.inst);
        return;
    }
    // Po atjungimo — tik jei flow'as nuo tada rašomas (head pajudėjo arba lastWriteTime šviežias): kitaip „gyvas“ buvo tik
    // kito skaitytojo flock (mxlIsFlowActive), o flow'as — pasenęs.
    if (a.detached) {
        mxlFlowRuntimeInfo rt{};
        const bool info = mxlFlowReaderGetRuntimeInfo(fresh.r, &rt) == MXL_STATUS_OK;
        if (!(h > a.detach_head || (info && rt.lastWriteTime > a.detach_tai))) {
            fresh.release(a.inst);
            return;
        }
    }
    // Pirmas prisijungimas (peržiūra D6): ta pati taisyklė — tik flow'as, kuris RAŠOMAS (lastWriteTime ne senesnis kaip
    // kStaleNs arba head pajudėjo nuo ankstesnio bandymo). Kitaip nužudyto gamintojo pasenęs flow + kito skaitytojo flock
    // („gyvas“ mxlIsFlowActive) būtų prijungtas ir pirmi tick'ai skaitytų seną ring'ą.
    if (a.waiting) {
        mxlFlowRuntimeInfo rt{};
        const bool info = mxlFlowReaderGetRuntimeInfo(fresh.r, &rt) == MXL_STATUS_OK;
        const bool recent = info && rt.lastWriteTime + kStaleNs >= a.now;
        const bool moved = a.waiting_head != 0 && h > a.waiting_head;
        a.set_waiting_head = true;
        a.waiting_head = h;
        if (!(recent || moved)) {
            fresh.release(a.inst);
            return;
        }
    }
    a.in = std::move(fresh);
    fresh.r = nullptr;
    a.kind = Attempt::Kind::Ok;
}

InputSlot::Attempt InputSlot::request(mxlInstance inst, std::uint64_t now_tai_ns) const {
    Attempt a;
    a.inst = inst;
    a.id = id_;
    a.have_want = have_want_;
    a.want = want_;
    a.check_storage = ever_ || lock_storage_;
    a.want_cuda = want_cuda_;
    a.ever = ever_;
    a.detached = state_ == State::Detached;
    a.waiting = state_ == State::Waiting;
    a.detach_head = detach_head_;
    a.detach_tai = detach_tai_;
    a.waiting_head = waiting_head_;
    a.now = now_tai_ns;
    return a;
}

InputSlot::State InputSlot::apply(Attempt& a) {
    if (a.set_waiting_head) waiting_head_ = a.waiting_head;
    if (a.clear_reason) reason_.clear();
    switch (a.kind) {
        case Attempt::Kind::None: return state_;
        case Attempt::Kind::NotYet:
            if (!a.reason.empty()) reason_ = a.reason;
            return state_;
        case Attempt::Kind::Rejected:
            if (reason_ != a.reason)
                std::fprintf(stderr, "[vramflow] %s%s\n", a.reason.c_str(), on_change_ == FormatChange::Exit ? "" : " — laukiama");
            reason_ = a.reason;
            if (on_change_ == FormatChange::Exit) state_ = State::Mismatch;
            return state_;
        case Attempt::Kind::Ok: break;
    }
    in_ = std::move(a.in);
    a.in.r = nullptr;
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

InputSlot::State InputSlot::poll(mxlInstance inst, std::uint64_t now_tai_ns) {
    if (state_ == State::Attached || state_ == State::Mismatch) return state_;
    if (now_tai_ns < next_try_) return state_;
    next_try_ = now_tai_ns + kRetryNs;
    inst_ = inst;
    Attempt a = request(inst, now_tai_ns);
    try_attach(a);
    return apply(a);
}

InputSlot::State InputSlot::poll_async(mxlInstance inst, std::uint64_t now_tai_ns) {
    if (state_ == State::Attached || state_ == State::Mismatch) return state_;
    inst_ = inst;
    prepare_async();
    Helper& h = *helper_;
    std::unique_lock lk(h.m);
    if (h.done) {  // ankstesnio bandymo rezultatas — pritaikomas čia (perkėlimas, be sisteminių kvietimų)
        h.done = false;
        Attempt a = std::move(h.a);
        h.a.in.r = nullptr;
        lk.unlock();
        return apply(a);
    }
    if (h.pending || now_tai_ns < next_try_) return state_;
    next_try_ = now_tai_ns + kRetryNs;
    h.a = request(inst, now_tai_ns);
    h.pending = true;
    lk.unlock();
    h.cv.notify_one();
    return state_;
}

void InputSlot::prepare_async() {
    if (helper_) return;
    helper_ = std::make_shared<Helper>();
    helper_->th = std::thread([h = helper_.get()] { h->run(); });
}

void InputSlot::stop_helper() {
    if (!helper_) return;
    {
        std::lock_guard g(helper_->m);
        helper_->quit = true;
    }
    helper_->cv.notify_one();
    if (helper_->th.joinable()) helper_->th.join();
    helper_.reset();
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
