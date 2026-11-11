// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/clock.hpp"

namespace mxlv::engine {

VirtualClock::VirtualClock(std::uint64_t start_tai_ns, std::int64_t start_mono_ns)
    : tai_(start_tai_ns), mono_(start_mono_ns), mono_start_(start_mono_ns) {}

void VirtualClock::advance(std::uint64_t ns) {
    const auto d = static_cast<std::int64_t>(ns);
    if (step_pending_ && step_at_mono_ <= mono_ + d) {
        const std::int64_t before = step_at_mono_ > mono_ ? step_at_mono_ - mono_ : 0;
        mono_ += before;
        tai_ += static_cast<std::uint64_t>(before);
        step_pending_ = false;
        step(step_delta_);
        mono_ += d - before;
        tai_ += static_cast<std::uint64_t>(d - before);
        return;
    }
    mono_ += d;
    tai_ += ns;
}

void VirtualClock::step(std::int64_t delta_ns) {
    tai_ = static_cast<std::uint64_t>(static_cast<std::int64_t>(tai_) + delta_ns);
}

void VirtualClock::schedule_step(std::int64_t at_mono_ns, std::int64_t delta_ns) {
    step_pending_ = true;
    step_at_mono_ = at_mono_ns;
    step_delta_ = delta_ns;
}

void VirtualClock::sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) {
    ++sleeps;
    if (on_sleep) on_sleep(tai_ns);
    // Kaip tikras laikrodis gabalais: po suplanuoto žingsnio tikslas perskaičiuojamas naujoje laiko juostoje.
    while (!stop.load(std::memory_order_relaxed) && tai_ < tai_ns) {
        const std::uint64_t d = tai_ns - tai_;
        if (step_pending_ && step_at_mono_ < mono_ + static_cast<std::int64_t>(d)) {
            const std::int64_t before = step_at_mono_ > mono_ ? step_at_mono_ - mono_ : 0;
            advance(static_cast<std::uint64_t>(before));
            if (step_pending_) {  // at_mono_ == mono_: advance(0) jo nepritaikė
                step_pending_ = false;
                step(step_delta_);
            }
            continue;
        }
        advance(d);
    }
    if (oversleep_ != 0) {
        const std::uint64_t o = oversleep_;
        oversleep_ = 0;
        advance(o);
    }
}

std::int64_t StepDetector::check(const Clock& c) {
    const std::int64_t off = static_cast<std::int64_t>(c.now_tai()) - c.now_mono_raw();
    if (!init_) {
        init_ = true;
        last_ = off;
        return 0;
    }
    const std::int64_t d = off - last_;
    last_ = off;
    if (d > kStepNs || d < -kStepNs) {
        ++steps;
        if (d < 0) back_ns += static_cast<std::uint64_t>(-d);
        return d;
    }
    return 0;
}

}  // namespace mxlv::engine
