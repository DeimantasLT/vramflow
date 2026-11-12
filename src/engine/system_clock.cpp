// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/system_clock.hpp"

#include <mxl/time.h>

#include <algorithm>
#include <ctime>

namespace mxlv::engine {

std::uint64_t SystemClock::now_tai() const { return mxlGetTime(); }

std::int64_t SystemClock::now_mono_raw() const {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

void SystemClock::sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) {
    // Absoliutus miegas (M14) ne ilgesniais kaip 50 ms gabalais: sustabdymas ir laikrodžio žingsnis pastebimi greitai.
    for (;;) {
        const std::uint64_t n = mxlGetTime();
        if (stop.load(std::memory_order_relaxed) || n >= tai_ns) return;
        mxlSleepUntil(std::min<std::uint64_t>(tai_ns, n + 50'000'000));
    }
}

}  // namespace mxlv::engine
