// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variklio laikrodis (planas docs/planai/2026-11-12-tick-variklis.md §6 D5, §7). Be MXL: SystemClock (mxlGetTime,
// CLOCK_MONOTONIC_RAW, mxlSleepUntil) gyvena mxlv_engine_mxl; čia — sąsaja, virtualus laikrodis testams ir žingsnių
// detektorius (buvęs func::ClockWatch).
#include <atomic>
#include <cstdint>
#include <functional>

namespace mxlv::engine {

using Index = std::uint64_t;
inline constexpr Index kNoIndex = ~Index{0};

class Clock {
public:
    virtual ~Clock() = default;
    virtual std::uint64_t now_tai() const = 0;      // TAI ns
    virtual std::int64_t now_mono_raw() const = 0;  // CLOCK_MONOTONIC_RAW ns (žingsnių detektoriui)
    // Absoliučiai iki TAI laiko; grįžta anksčiau, jei nustatytas stop. Tikras laikrodis miega ≤ 50 ms gabalais.
    virtual void sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) = 0;
};

// Testų laikrodis: laikas juda tik advance / sleep_until / step. Viena gija (variklio); kitos gijos jo neliečia.
class VirtualClock final : public Clock {
public:
    explicit VirtualClock(std::uint64_t start_tai_ns, std::int64_t start_mono_ns = 1'000'000'000);

    std::uint64_t now_tai() const override { return tai_; }
    std::int64_t now_mono_raw() const override { return mono_; }
    void sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) override;

    void advance(std::uint64_t ns);    // abu laikrodžiai (suplanuoti žingsniai pritaikomi pakeliui)
    void step(std::int64_t delta_ns);  // tik TAI (laikrodžio žingsnis)

    // --- Testų scenarijams (ne §7 sąsajos dalis) ---
    std::int64_t mono_start() const { return mono_start_; }
    // TAI žingsnis delta_ns, kai monotoninis laikas pasieks at_mono_ns (vienas suplanuotas vienu metu).
    void schedule_step(std::int64_t at_mono_ns, std::int64_t delta_ns);
    // Kitas sleep_until prabus ns vėliau už tikslą (variklio gija „pavėlavo“).
    void oversleep_once(std::uint64_t ns) { oversleep_ = ns; }
    // Kviečiamas kiekvieno sleep_until pradžioje su tikslu (TAI). Testas neturi jame alokuoti, jei tikrina alokacijas.
    std::function<void(std::uint64_t target_tai)> on_sleep;
    std::uint64_t sleeps = 0;

private:
    std::uint64_t tai_;
    std::int64_t mono_;
    std::int64_t mono_start_;
    bool step_pending_ = false;
    std::int64_t step_at_mono_ = 0, step_delta_ = 0;
    std::uint64_t oversleep_ = 0;
};

// (TAI − MONOTONIC_RAW) pokytis > 5 ms tarp patikrų — žingsnis (slew'as nematomas). Kaip func::ClockWatch.
struct StepDetector {
    static constexpr std::int64_t kStepNs = 5'000'000;
    std::int64_t check(const Clock&);  // 0 — nebuvo; kitaip Δ ns (su ženklu)
    std::uint64_t steps = 0, back_ns = 0;

private:
    bool init_ = false;
    std::int64_t last_ = 0;
};

}  // namespace mxlv::engine
