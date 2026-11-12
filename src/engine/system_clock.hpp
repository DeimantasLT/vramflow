// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Tikras variklio laikrodis (planas §6 D5, §7): mxlGetTime (TAI), CLOCK_MONOTONIC_RAW, mxlSleepUntil ≤ 50 ms gabalais
// (A27, M14). Atskirai nuo mxl_ports.hpp, kad jį galėtų linkuoti libvramflow-gfx (ritmo vykdytojas, V14) be
// runtime / InputSlot priklausomybių.
#include "engine/clock.hpp"

namespace mxlv::engine {

class SystemClock final : public Clock {
public:
    std::uint64_t now_tai() const override;
    std::int64_t now_mono_raw() const override;
    void sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) override;
};

}  // namespace mxlv::engine
