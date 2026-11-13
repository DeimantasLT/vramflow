// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/stats.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>

namespace mxlv::engine {

const char* to_string(Health h) {
    switch (h) {
        case Health::Healthy: return "Healthy";
        case Health::PartiallyHealthy: return "PartiallyHealthy";
        case Health::Unhealthy: return "Unhealthy";
        case Health::Inactive: return "Inactive";
    }
    return "?";
}

const char* to_string(EventKind k) {
    switch (k) {
        case EventKind::UnitFailed: return "unit_failed";
        case EventKind::UnitRetryFailed: return "unit_retry_failed";
        case EventKind::UnitRecovered: return "unit_recovered";
        case EventKind::InputAttached: return "input_attached";
        case EventKind::InputReattached: return "input_reattached";
        case EventKind::InputDetached: return "input_detached";
        case EventKind::InputStalled: return "input_stalled";
        case EventKind::InputResumed: return "input_resumed";
        case EventKind::OutputError: return "output_error";
        case EventKind::OutputRecovered: return "output_recovered";
        case EventKind::FormatChange: return "format_change";
        case EventKind::ClockStep: return "clock_step";
        case EventKind::Warning: return "warning";
        case EventKind::LayerAbsent: return "layer_absent";
        case EventKind::LayerPresent: return "layer_present";
    }
    return "?";
}

Health input_health(InState s, const InCounters& d) {
    if (s == InState::Disabled) return Health::Inactive;
    if (s != InState::Live || d.ok == 0) return Health::Unhealthy;
    if (d.late != 0 || d.lost != 0 || d.invalid != 0) return Health::PartiallyHealthy;
    return Health::Healthy;
}

Health output_health(const OutCounters& d, bool enabled) {
    if (!enabled) return Health::Inactive;
    if (d.ok + d.late == 0) return Health::Unhealthy;
    const std::uint64_t gaps = d.gap_invalid + d.gap_frozen + d.gap_black + d.gap_left + d.gap_startup + d.gap_upstream + d.gap_align + d.lapped +
                               d.late_skips + d.clock_skips + d.warmup_skips + d.commit_errors;
    return gaps != 0 ? Health::PartiallyHealthy : Health::Healthy;
}

Health process_health(std::span<const Health> in, std::span<const Health> out) {
    std::size_t active = 0, healthy = 0, out_active = 0, out_bad = 0, in_active = 0, in_bad = 0;
    for (Health h : in) {
        if (h == Health::Inactive) continue;
        ++active;
        ++in_active;
        healthy += h == Health::Healthy;
        in_bad += h == Health::Unhealthy;
    }
    for (Health h : out) {
        if (h == Health::Inactive) continue;
        ++active;
        ++out_active;
        healthy += h == Health::Healthy;
        out_bad += h == Health::Unhealthy;
    }
    if (active == 0) return Health::Inactive;
    if (healthy == active) return Health::Healthy;
    if (out_active != 0 ? out_bad == out_active : in_bad == in_active) return Health::Unhealthy;
    return Health::PartiallyHealthy;
}

Health unit_health(const UnitCounters& d, bool failed) {
    if (failed) return Health::Unhealthy;
    const std::uint64_t lost = d.gaps - d.gaps_clock - d.gaps_no_content - d.gaps_upstream - d.gaps_align + d.lapped + d.late_skips;
    const std::uint64_t any = d.ok + d.late + d.gaps + d.startup + d.lapped + d.late_skips + d.clock_skips + d.warmup_skips;
    if (any == 0) return Health::Inactive;
    if (d.ok + d.late == 0) return Health::Unhealthy;
    return lost != 0 ? Health::PartiallyHealthy : Health::Healthy;
}

Health units_health(std::span<const Health> units) {
    std::size_t active = 0, healthy = 0, bad = 0;
    for (Health h : units) {
        if (h == Health::Inactive) continue;
        ++active;
        healthy += h == Health::Healthy;
        bad += h == Health::Unhealthy;
    }
    if (active == 0) return Health::Inactive;
    if (healthy == active) return Health::Healthy;
    if (bad == active) return Health::Unhealthy;
    return Health::PartiallyHealthy;
}

// --- Hist64 ---------------------------------------------------------------------------------------------------------
int Hist64::bucket_of(std::uint64_t ns) {
    const std::uint64_t v = ns >> 10;  // ≈ µs
    if (v == 0) return 0;
    const int msb = 63 - std::countl_zero(v);
    const int sub = msb >= 2 ? static_cast<int>((v >> (msb - 2)) & 3) : static_cast<int>((v << (2 - msb)) & 3);
    return std::min(63, 1 + msb * 4 + sub);
}

std::uint64_t Hist64::value_of(int b) {
    if (b <= 0) return 0;
    const int msb = (b - 1) / 4;
    const int sub = (b - 1) % 4;
    const std::uint64_t v = msb >= 2 ? (std::uint64_t{4 + static_cast<unsigned>(sub)} << (msb - 2))
                                     : (std::uint64_t{1} << msb);
    return v << 10;
}

std::uint64_t Hist64::quantile(double q) const {
    if (n_ == 0) return 0;
    const auto target = static_cast<std::uint64_t>(q * static_cast<double>(n_ - 1)) + 1;
    std::uint64_t acc = 0;
    for (int b = 0; b < 64; ++b) {
        acc += c_[static_cast<std::size_t>(b)];
        if (acc >= target) return value_of(b);
    }
    return value_of(63);
}

// --- StatsHub -------------------------------------------------------------------------------------------------------
StatsHub::StatsHub(double interval_s, StatsSink* sink)
    : interval_ns_(static_cast<std::uint64_t>(interval_s * 1e9)), sink_(sink) {}

StatsHub::~StatsHub() { stop(); }

CounterId StatsHub::register_counter(std::string name) {
    counter_names_.push_back(std::move(name));
    return static_cast<CounterId>(counter_names_.size() - 1);
}

HistId StatsHub::register_hist(std::string name) {
    hist_names_.push_back(std::move(name));
    hist_.emplace_back();
    return static_cast<HistId>(hist_names_.size() - 1);
}

std::uint32_t StatsHub::add_unit(std::string name) {
    unit_names_.push_back(std::move(name));
    unit_.emplace_back();
    return static_cast<std::uint32_t>(unit_names_.size() - 1);
}

std::uint32_t StatsHub::add_input(std::string name) {
    in_names_.push_back(std::move(name));
    in_.emplace_back();
    in_owner_.push_back(unit_names_.empty() ? 0 : static_cast<std::uint32_t>(unit_names_.size() - 1));
    return static_cast<std::uint32_t>(in_.size() - 1);
}

std::uint32_t StatsHub::add_output(std::string name) {
    out_names_.push_back(std::move(name));
    out_.emplace_back();
    out_owner_.push_back(unit_names_.empty() ? 0 : static_cast<std::uint32_t>(unit_names_.size() - 1));
    return static_cast<std::uint32_t>(out_.size() - 1);
}

void StatsHub::prepare() {
    fn_.assign(unit_names_.size() * counter_names_.size(), 0);
    in_prev_ = in_;
    out_prev_ = out_;
    unit_prev_ = unit_;
    fn_prev_ = fn_;
    eng_prev_ = eng_;
    in_hist_.assign(in_.size(), Hist64{});
    out_hist_.assign(out_.size(), Hist64{});
    in_state_.assign(in_.size(), InState::Waiting);
    out_on_.assign(out_.size(), 1);
    unit_failed_.assign(unit_.size(), 0);
    in_h_.assign(in_.size(), Health::Unhealthy);
    out_h_.assign(out_.size(), Health::Unhealthy);
    unit_h_.assign(unit_.size(), Health::Inactive);
    for (Snapshot& s : buf_) {
        s.in.assign(in_.size(), InSnap{});
        s.out.assign(out_.size(), OutSnap{});
        s.unit.assign(unit_.size(), UnitSnap{});
        for (std::size_t k = 0; k < in_.size(); ++k) s.in[k].unit = in_owner_[k];
        for (std::size_t k = 0; k < out_.size(); ++k) s.out[k].unit = out_owner_[k];
        s.fn.assign(fn_.size(), 0);
        s.hist.assign(hist_.size(), HistSnap{});
        s.in_names = &in_names_;
        s.out_names = &out_names_;
        s.unit_names = &unit_names_;
        s.counter_names = &counter_names_;
        s.hist_names = &hist_names_;
    }
    if (sink_ != nullptr && !th_.joinable()) th_ = std::thread([this] { publisher(); });
}

namespace {
InCounters minus(const InCounters& a, const InCounters& b) {
    return {a.ok - b.ok,         a.late - b.late,         a.lost - b.lost,
            a.invalid - b.invalid, a.too_late - b.too_late, a.read_errors - b.read_errors,
            a.reattached - b.reattached, a.stalls - b.stalls, a.detaches - b.detaches};
}
OutCounters minus(const OutCounters& a, const OutCounters& b) {
    OutCounters r;
    r.ok = a.ok - b.ok;
    r.late = a.late - b.late;
    r.gap_invalid = a.gap_invalid - b.gap_invalid;
    r.gap_frozen = a.gap_frozen - b.gap_frozen;
    r.gap_black = a.gap_black - b.gap_black;
    r.gap_left = a.gap_left - b.gap_left;
    r.gap_startup = a.gap_startup - b.gap_startup;
    r.gap_upstream = a.gap_upstream - b.gap_upstream;
    r.gap_align = a.gap_align - b.gap_align;
    r.lapped = a.lapped - b.lapped;
    r.late_skips = a.late_skips - b.late_skips;
    r.clock_skips = a.clock_skips - b.clock_skips;
    r.warmup_skips = a.warmup_skips - b.warmup_skips;
    r.commit_errors = a.commit_errors - b.commit_errors;
    r.already_committed = a.already_committed - b.already_committed;
    return r;
}
UnitCounters minus(const UnitCounters& a, const UnitCounters& b) {
    UnitCounters r;
    r.ok = a.ok - b.ok;
    r.late = a.late - b.late;
    r.gaps = a.gaps - b.gaps;
    r.startup = a.startup - b.startup;
    r.lapped = a.lapped - b.lapped;
    r.late_skips = a.late_skips - b.late_skips;
    r.clock_skips = a.clock_skips - b.clock_skips;
    r.warmup_skips = a.warmup_skips - b.warmup_skips;
    r.gaps_clock = a.gaps_clock - b.gaps_clock;
    r.gaps_no_content = a.gaps_no_content - b.gaps_no_content;
    r.gaps_upstream = a.gaps_upstream - b.gaps_upstream;
    r.gaps_align = a.gaps_align - b.gaps_align;
    r.failures = a.failures - b.failures;
    r.retries = a.retries - b.retries;
    r.recoveries = a.recoveries - b.recoveries;
    return r;
}
}  // namespace

void StatsHub::end_tick(std::uint64_t now) {
    if (!started_) {  // pirmas kvietimas — intervalo pradžia
        started_ = true;
        t_begin_ = now;
        next_due_ = now + interval_ns_;
        return;
    }
    if (now < next_due_) return;
    next_due_ += interval_ns_;  // pastovus ritmas; atsilikus — nuo dabar
    if (next_due_ <= now) next_due_ = now + interval_ns_;
    if (full_.load(std::memory_order_acquire) != 0) {  // ankstesnė kopija nepaimta — skirtumai pereina į kitą
        ++dropped_;
        return;
    }
    snapshot(now);
}

void StatsHub::flush(std::uint64_t now) {
    if (!started_ || buf_[0].unit.size() != unit_.size() || buf_[0].in.size() != in_.size()) return;  // be prepare()
    const auto t0 = std::chrono::steady_clock::now();
    while (sink_ != nullptr && th_.joinable() && full_.load(std::memory_order_acquire) != 0 &&
           std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {
        cv_.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (full_.load(std::memory_order_acquire) != 0) {
        ++dropped_;
        return;
    }
    snapshot(now);
}

void StatsHub::snapshot(std::uint64_t now) {
    Snapshot& s = buf_[w_.load(std::memory_order_acquire)];
    s.seq = ++seq_;
    s.t_begin_tai = t_begin_;
    s.t_end_tai = now;
    s.dropped_lines = dropped_;
    for (std::size_t k = 0; k < in_.size(); ++k) {
        InSnap& x = s.in[k];
        x.d = minus(in_[k], in_prev_[k]);
        x.state = in_state_[k];
        x.health = in_h_[k] = input_health(x.state, x.d);
        x.arrival_p50_ns = in_hist_[k].quantile(0.5);
        x.arrival_p99_ns = in_hist_[k].quantile(0.99);
        x.arrival_max_ns = in_hist_[k].max();
        in_hist_[k].reset();
        in_prev_[k] = in_[k];
    }
    for (std::size_t k = 0; k < out_.size(); ++k) {
        OutSnap& x = s.out[k];
        x.d = minus(out_[k], out_prev_[k]);
        x.enabled = out_on_[k] != 0;
        x.health = out_h_[k] = output_health(x.d, x.enabled);
        x.commit_p50_ns = out_hist_[k].quantile(0.5);
        x.commit_p99_ns = out_hist_[k].quantile(0.99);
        x.commit_max_ns = out_hist_[k].max();
        out_hist_[k].reset();
        out_prev_[k] = out_[k];
    }
    for (std::size_t k = 0; k < unit_.size(); ++k) {
        UnitSnap& x = s.unit[k];
        x.d = minus(unit_[k], unit_prev_[k]);
        x.failed = unit_failed_[k] != 0;
        x.health = unit_h_[k] = unit_health(x.d, x.failed);
        unit_prev_[k] = unit_[k];
    }
    for (std::size_t k = 0; k < fn_.size(); ++k) {
        s.fn[k] = fn_[k] - fn_prev_[k];
        fn_prev_[k] = fn_[k];
    }
    for (std::size_t k = 0; k < hist_.size(); ++k) {
        s.hist[k] = HistSnap{hist_[k].count(), hist_[k].quantile(0.5), hist_[k].quantile(0.99)};
        hist_[k].reset();
    }
    EngineCounters& e = s.engine;
    e.ticks = eng_.ticks - eng_prev_.ticks;
    e.process_calls = eng_.process_calls - eng_prev_.process_calls;
    e.guard_overrun = eng_.guard_overrun - eng_prev_.guard_overrun;
    e.carry_overrun = eng_.carry_overrun - eng_prev_.carry_overrun;
    e.cut_batches = eng_.cut_batches - eng_prev_.cut_batches;
    e.clock_steps = eng_.clock_steps - eng_prev_.clock_steps;
    e.clock_backward = eng_.clock_backward - eng_prev_.clock_backward;
    e.step_gaps = eng_.step_gaps - eng_prev_.step_gaps;
    e.held_ns_max = eng_.held_ns_max;
    eng_prev_ = eng_;
    // Proceso būsena: iš vienetų (flow'ų); be vienetų — iš įėjimų / išėjimų (§6 D6 lentelė).
    s.health = health_ = unit_.empty() ? process_health(in_h_, out_h_) : units_health(unit_h_);
    t_begin_ = now;
    if (sink_ == nullptr) return;
    full_.store(1, std::memory_order_release);
    cv_.notify_one();  // neblokuoja; publikavimo gija dar tikrina kas 50 ms (pažadinimas be užrakto gali pasimesti)
}

void StatsHub::event(EventKind k, std::uint32_t id, std::uint64_t tai, std::uint64_t index, std::int64_t value, const char* text) {
    const std::uint32_t w = ev_w_.load(std::memory_order_relaxed);
    if (w - ev_r_.load(std::memory_order_acquire) >= kEvents) {
        ++ev_dropped_;
        return;
    }
    Event& e = ev_[w % kEvents];
    e.tai = tai;
    e.index = index;
    e.value = value;
    e.id = id;
    e.kind = k;
    e.text[0] = '\0';
    if (text != nullptr) {
        std::strncpy(e.text, text, sizeof e.text - 1);
        e.text[sizeof e.text - 1] = '\0';
    }
    ev_w_.store(w + 1, std::memory_order_release);
    cv_.notify_one();
}

void StatsHub::drain_events() {
    const StatsNames names{&unit_names_, &in_names_, &out_names_};
    for (;;) {
        const std::uint32_t r = ev_r_.load(std::memory_order_relaxed);
        if (r == ev_w_.load(std::memory_order_acquire)) return;
        const Event e = ev_[r % kEvents];
        ev_r_.store(r + 1, std::memory_order_release);
        sink_->event(e, names);
    }
}

void StatsHub::publisher() {
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait_for(lk, std::chrono::milliseconds(50), [&] {
                return full_.load(std::memory_order_acquire) != 0 || quit_.load(std::memory_order_acquire) ||
                       ev_r_.load(std::memory_order_relaxed) != ev_w_.load(std::memory_order_acquire);
            });
        }
        drain_events();  // įvykiai — prieš to intervalo eilutę
        if (full_.load(std::memory_order_acquire) != 0) {
            const int r = w_.load(std::memory_order_acquire);
            w_.store(1 - r, std::memory_order_release);
            full_.store(0, std::memory_order_release);
            sink_->publish(buf_[r]);
            continue;
        }
        if (quit_.load(std::memory_order_acquire)) {
            drain_events();
            return;
        }
    }
}

void StatsHub::stop() {
    if (!th_.joinable()) return;
    quit_.store(true, std::memory_order_release);
    cv_.notify_one();
    th_.join();
}

}  // namespace mxlv::engine
