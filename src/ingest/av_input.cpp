// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/av_input.hpp"

extern "C" {
#include <libavutil/error.h>
}
#include <mxl/time.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

namespace mxlv::ingest {

std::string averr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(e, b, sizeof b);
    return b;
}

std::uint64_t tai_now() { return mxlGetTime(); }

int interrupt_cb(void* p) {
    auto* x = static_cast<Interrupt*>(p);
    if (x->stop != nullptr && x->stop->load(std::memory_order_relaxed)) return 1;
    const std::uint64_t now = tai_now();
    const std::uint64_t last = x->last_pkt.load(std::memory_order_relaxed);
    if (last != 0 && x->silence_ns != 0 && now - last > x->silence_ns) {
        x->silent = true;
        return 1;
    }
    if (last == 0 && x->deadline_tai != 0 && now > x->deadline_tai) {
        x->timed_out = true;
        return 1;
    }
    return 0;
}

std::string protocol_error(const std::string& url) {
    const auto colon = url.find("://");
    if (colon == std::string::npos) return {};  // failas — visada
    const std::string scheme = url.substr(0, colon);
    void* it = nullptr;
    const char* name = nullptr;
    while ((name = avio_enum_protocols(&it, 0)) != nullptr)
        if (scheme == name) return {};
    return scheme + " protokolas nepalaikomas šio FFmpeg (libavformat " + LIBAVFORMAT_IDENT + "): " + url;
}

void reconnect_loop(const std::string& name, const std::atomic<bool>& stop, const std::function<bool(std::string& why)>& session,
                    std::uint64_t& reconnects, std::string& error, const std::function<void(const std::string&)>& log) {
    double backoff = 1;
    while (!stop.load()) {
        const std::uint64_t t0 = tai_now();
        std::string why;
        const bool fatal = session(why);
        if (stop.load()) break;
        if (fatal) {
            error = why;
            log(name + ": " + why + " — srautas sustabdytas");
            break;
        }
        ++reconnects;
        if (tai_now() - t0 >= 60'000'000'000ull) backoff = 1;
        char b[64];
        std::snprintf(b, sizeof b, "%.0f", backoff);
        log(name + ": " + why + " — jungiamasi iš naujo po " + b + " s (" + std::to_string(reconnects) + ")");
        const std::uint64_t until = tai_now() + static_cast<std::uint64_t>(backoff * 1e9);
        while (!stop.load() && tai_now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        backoff = std::min(30.0, 2 * backoff);
    }
}

}  // namespace mxlv::ingest
