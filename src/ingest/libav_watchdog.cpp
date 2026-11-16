// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/libav_watchdog.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace mxlv::ingest {

LibavWatchdog::LibavWatchdog(std::uint64_t margin_ns, int exit_code, Clock now, OnStuck on_stuck, int poll_ms)
    : margin_(margin_ns), exit_code_(exit_code), now_(std::move(now)), on_stuck_(std::move(on_stuck)), poll_ms_(poll_ms) {
    if (!on_stuck_) {
        const int code = exit_code_;
        const std::uint64_t margin = margin_;
        on_stuck_ = [code, margin](const std::string& name, const char* what, Stuck why, std::uint64_t elapsed_ns) {
            // tik stderr (be žurnalo užraktų — juos gali laikyti kitos gijos) ir iškart baigti: užstrigusios gijos atšaukti negalima
            std::fprintf(stderr,
                         "[ingest] %s: libav kvietimas „%s“ neišėjo per %.1f s (%s, atsarga %.0f s) — gija libav viduje neatšaukiama (libav ≤ 8.0 "
                         "http nukreipimų kešo kilpa?), procesas baigiamas exit %d; prižiūrėtojas turi paleisti iš naujo\n",
                         name.c_str(), what, static_cast<double>(elapsed_ns) / 1e9,
                         why == Stuck::Deadline ? "viršyta pertraukimo riba" : "pertraukimo kabliukas nekviečiamas", static_cast<double>(margin) / 1e9,
                         code);
            std::fflush(stderr);
            std::_Exit(code);
        };
    }
    th_ = std::thread([this] { run(); });
}

LibavWatchdog::~LibavWatchdog() {
    {
        std::lock_guard g(m_);
        stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

CallWatch* LibavWatchdog::add(const std::string& name) {
    std::lock_guard g(m_);
    CallWatch& w = w_.emplace_back();
    w.name = name;
    w.margin_ns = margin_;
    return &w;
}

void LibavWatchdog::run() {
    std::unique_lock lk(m_);
    while (!stop_) {
        cv_.wait_for(lk, std::chrono::milliseconds(poll_ms_), [this] { return stop_; });
        if (stop_) break;
        const std::uint64_t now = now_();
        for (CallWatch& w : w_) {
            const char* what = w.what.load(std::memory_order_acquire);
            const std::uint64_t since = w.since.load(std::memory_order_acquire);
            const Stuck s = call_stuck(what, since, w.kill_at.load(std::memory_order_acquire), w.last_poll.load(std::memory_order_relaxed), w.margin_ns, now);
            if (s == Stuck::No) continue;
            // dar kartą: kvietimas galėjo ką tik baigtis ir prasidėti kitas (what / since perrašyti) — tikrinamas tas pats kvietimas
            if (w.what.load(std::memory_order_acquire) != what || w.since.load(std::memory_order_acquire) != since) continue;
            on_stuck_(w.name, what, s, now - since);
        }
    }
}

}  // namespace mxlv::ingest
