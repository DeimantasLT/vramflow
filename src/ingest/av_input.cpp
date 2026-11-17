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

#include "common/textf.hpp"
#include "ingest/url_policy.hpp"

namespace mxlv::ingest {

std::string averr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(e, b, sizeof b);
    return b;
}

std::uint64_t tai_now() { return mxlGetTime(); }

int interrupt_cb(void* p) {
    auto* x = static_cast<Interrupt*>(p);
    const std::uint64_t now = tai_now();
    // R13-2: libav kabliuką kviečia kiekviename laukime (retry_transfer_wrapper — kiekvienam skaitymui, avio.c n6.1:357-381; HLS
    // perkrovimo laukimas — kas 100 ms, hls.c:1538-1541) — sargui tai pažanga
    if (x->watch != nullptr) x->watch->last_poll.store(now, std::memory_order_relaxed);
    if (x->stop != nullptr && x->stop->load(std::memory_order_relaxed)) return 1;
    if (x->abort.load(std::memory_order_relaxed)) return 1;
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

LibavCall::LibavCall(Interrupt& in, const char* what) : w_(in.watch) {
    if (w_ == nullptr) return;
    const std::uint64_t now = tai_now();
    const std::uint64_t last = in.last_pkt.load(std::memory_order_relaxed);
    // ta pati riba kaip interrupt_cb: tyla — nuo paskutinio paketo, atidarymo riba — kol paketo nėra
    const std::uint64_t bound = last != 0 && in.silence_ns != 0 ? last + in.silence_ns : last == 0 && in.deadline_tai != 0 ? in.deadline_tai : 0;
    prev_what_ = w_->what.load(std::memory_order_relaxed);
    prev_since_ = w_->since.load(std::memory_order_relaxed);
    prev_kill_ = w_->kill_at.load(std::memory_order_relaxed);
    w_->what.store(nullptr, std::memory_order_release);  // kol perrašoma — ne „viduje“ (sargas nemato pusiau atnaujintos būsenos)
    w_->since.store(now, std::memory_order_release);
    w_->kill_at.store(bound != 0 ? std::max(bound, now) + w_->margin_ns : 0, std::memory_order_release);
    w_->what.store(what, std::memory_order_release);
}

LibavCall::~LibavCall() {
    if (w_ == nullptr) return;
    w_->what.store(nullptr, std::memory_order_release);
    if (prev_what_ == nullptr) return;
    w_->since.store(prev_since_, std::memory_order_release);
    w_->kill_at.store(prev_kill_, std::memory_order_release);
    w_->what.store(prev_what_, std::memory_order_release);
}

std::string protocol_error(const std::string& url) {
    const std::string scheme = url_scheme(url);
    if (scheme.empty()) return {};  // failas — visada
    void* it = nullptr;
    const char* name = nullptr;
    while ((name = avio_enum_protocols(&it, 0)) != nullptr)
        if (scheme == name) return {};
    return scheme + " protokolas nepalaikomas šio FFmpeg (libavformat " + LIBAVFORMAT_IDENT + "): " + redact_url(url);
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
            log(strf("%s: %s — srautas sustabdytas", name.c_str(), why.c_str()));
            break;
        }
        ++reconnects;
        if (tai_now() - t0 >= 60'000'000'000ull) backoff = 1;
        log(strf("%s: %s — jungiamasi iš naujo po %.0f s (%llu)", name.c_str(), why.c_str(), backoff,
                 static_cast<unsigned long long>(reconnects)));
        const std::uint64_t until = tai_now() + static_cast<std::uint64_t>(backoff * 1e9);
        while (!stop.load() && tai_now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        backoff = std::min(30.0, 2 * backoff);
    }
}

}  // namespace mxlv::ingest
