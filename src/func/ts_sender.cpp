// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/ts_sender.hpp"

extern "C" {
#include <libavutil/error.h>
}

#include <algorithm>
#include <cstring>
#include <utility>

#include "func/stats_lines.hpp"

namespace mxlv::func {

namespace {
constexpr std::chrono::milliseconds kBackoffFirst{500};
constexpr std::chrono::milliseconds kBackoffMax{5000};

std::string averr_text(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(e, b, sizeof b);
    return b;
}
}  // namespace

TsSender::TsSender(std::string url_redacted, OpenFn open, std::size_t max_queue_bytes, CloseFn close)
    : url_(std::move(url_redacted)), open_(std::move(open)), close_(std::move(close)) {
    ring_.resize(std::max(kChunk, max_queue_bytes / kTsPacket * kTsPacket));
    int_cb_.callback = &TsSender::interrupt_cb;
    int_cb_.opaque = this;
    th_ = std::thread([this] { run(); });
}

TsSender::~TsSender() { stop(std::chrono::milliseconds{0}); }

int TsSender::interrupt_cb(void* opaque) {
    const auto* s = static_cast<const TsSender*>(opaque);
    // open (pvz. SRT listener laukia imtuvo) nutraukiamas iškart sustabdant; rašymas — tik pasibaigus drain laikui
    return s->abort_.load() || (s->opening_.load() && s->stopping_.load()) ? 1 : 0;
}

void TsSender::push(const std::uint8_t* p, std::size_t n) {
    if (n == 0) return;
    std::uint64_t dropped = 0;
    {
        std::lock_guard g(m_);
        const std::size_t cap = ring_.size();
        if (stop_req_) {
            dropped = n / kTsPacket;  // gija sustabdyta — siųsti nebėra kam
        } else {
            if (n > cap) {  // didesnis už visą eilę: lieka tik galas (riba — 188 kartotinis, nes n ir cap tokie)
                dropped += (n - cap) / kTsPacket;
                p += n - cap;
                n = cap;
            }
            if (size_ + n > cap) {  // seniausi duomenys — sveikais TS paketais nuo galvos
                std::size_t d = (size_ + n - cap + kTsPacket - 1) / kTsPacket * kTsPacket;
                d = std::min(d, size_);
                head_ = (head_ + d) % cap;
                size_ -= d;
                dropped += d / kTsPacket;
            }
            const std::size_t tail = (head_ + size_) % cap;
            const std::size_t first = std::min(n, cap - tail);
            std::memcpy(ring_.data() + tail, p, first);
            std::memcpy(ring_.data(), p + first, n - first);
            size_ += n;
        }
    }
    if (dropped) packets_dropped_.fetch_add(dropped, std::memory_order_relaxed);
    cv_.notify_one();
}

SenderStats TsSender::stats() const {
    SenderStats s;
    s.bytes_sent = bytes_sent_.load(std::memory_order_relaxed);
    s.packets_dropped = packets_dropped_.load(std::memory_order_relaxed);
    s.reconnects = reconnects_.load(std::memory_order_relaxed);
    s.connected = connected_.load(std::memory_order_relaxed);
    return s;
}

std::size_t TsSender::pop(std::uint8_t* dst, std::size_t max) {
    const std::size_t cap = ring_.size();
    const std::size_t k = std::min(max, size_);
    const std::size_t first = std::min(k, cap - head_);
    std::memcpy(dst, ring_.data() + head_, first);
    std::memcpy(dst + first, ring_.data(), k - first);
    head_ = (head_ + k) % cap;
    size_ -= k;
    return k;
}

void TsSender::drop_queue() {
    if (size_) packets_dropped_.fetch_add(size_ / kTsPacket, std::memory_order_relaxed);
    head_ = size_ = 0;
}

bool TsSender::wait_backoff(std::chrono::milliseconds d) {
    std::unique_lock lk(m_);
    return !cv_.wait_for(lk, d, [&] { return stop_req_; });
}

void TsSender::run() {
    std::chrono::milliseconds backoff = kBackoffFirst;
    bool first = true, failing = false;
    std::vector<std::uint8_t> chunk(kChunk);
    auto do_close = [&](AVIOContext** pb) {
        if (!*pb) return;
        if (close_) close_(pb);
        else avio_closep(pb);
    };
    while (!stopping_.load()) {
        if (!first) reconnects_.fetch_add(1, std::memory_order_relaxed);
        first = false;
        AVIOContext* pb = nullptr;
        opening_.store(true);
        int e = open_(&pb, int_cb_);
        opening_.store(false);
        if (e < 0 || pb == nullptr) {
            do_close(&pb);
            if (stopping_.load()) break;
            if (!failing) log_line("enc", "išvestis " + url_ + ": nepavyko atidaryti (" + averr_text(e) + "), bandoma iš naujo");
            failing = true;
            if (!wait_backoff(backoff)) break;
            backoff = std::min(backoff * 2, kBackoffMax);
            continue;
        }
        failing = false;
        backoff = kBackoffFirst;
        connected_.store(true);
        log_line("enc", "išvestis " + url_ + ": prisijungta");
        bool broken = false;
        for (;;) {
            std::size_t n = 0;
            {
                std::unique_lock lk(m_);
                cv_.wait(lk, [&] { return size_ > 0 || stop_req_; });
                if (size_ == 0 || abort_.load()) break;  // sustabdyta: eilė išsiųsta arba drain laikas baigėsi
                n = pop(chunk.data(), kChunk);
            }
            avio_write(pb, chunk.data(), static_cast<int>(n));
            avio_flush(pb);
            if (pb->error < 0) {  // ryšys nutrūko (arba rašymas nutrauktas sustabdant): šis gabalas neišsiųstas
                e = pb->error;
                packets_dropped_.fetch_add(n / kTsPacket, std::memory_order_relaxed);
                broken = true;
                break;
            }
            bytes_sent_.fetch_add(n, std::memory_order_relaxed);
        }
        // sustabdant connected lieka sustabdymo metu buvusi būsena (santraukos out_connected); nutrūkus — false iki kito open
        if (broken && !stopping_.load()) connected_.store(false);
        do_close(&pb);
        if (!broken || stopping_.load()) break;
        log_line("enc", "išvestis " + url_ + ": ryšys nutrūko (" + averr_text(e) + "), jungiamasi iš naujo");
        if (!wait_backoff(backoff)) break;
    }
    {
        std::lock_guard g(m_);
        drop_queue();  // neišsiųsti likučiai suskaičiuojami (niekas nedingsta tyliai)
        exited_ = true;
    }
    exited_cv_.notify_all();
}

void TsSender::stop(std::chrono::milliseconds drain) {
    {
        std::lock_guard g(m_);
        if (stopped_) return;
        stopped_ = stop_req_ = true;
        stopping_.store(true);
    }
    cv_.notify_all();
    {
        std::unique_lock lk(m_);
        exited_cv_.wait_for(lk, drain, [&] { return exited_; });
    }
    abort_.store(true);  // drain baigėsi: užblokuotas rašymas nutraukiamas pertraukimo callback'u
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

}  // namespace mxlv::func
