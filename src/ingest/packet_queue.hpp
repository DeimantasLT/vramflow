// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Šaltinio paketų eilė demux → dekodavimo gija (ingest) — gryna logika be libav (tests/unit/test_packet_queue.cpp).
//
// Ribos (M1, round 10): elementų skaičius IR baitai (Σ paketų dydžių). Viršijus bet kurią, demux pusė išmeta seniausią medijos
// paketą (skaičiuojama dropped(), episodai — overflow_episodes(): žurnalui), kad atmintis būtų ribota net tada, kai serveris
// tinklo greičiu stumia ilgą didelės spartos segmentą. Dekodavimo gija paketus ima jų DTS laiku, todėl eilėje telpa
// latencija × srauto sparta; riba baitais — latencija × dosni sparta (ingest --max-queue-mb).
//
// Traits: static std::size_t bytes(const T&) (0 — žymė be paketo), static void drop(T&) (atlaisvinimas).
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>

namespace mxlv::ingest {

template <class T, class Traits>
class PacketQueue {
public:
    PacketQueue(std::size_t cap_items, std::uint64_t cap_bytes) : cap_(cap_items), cap_bytes_(cap_bytes) {}
    ~PacketQueue() { clear(); }
    PacketQueue(const PacketQueue&) = delete;
    PacketQueue& operator=(const PacketQueue&) = delete;

    void set_cap_bytes(std::uint64_t b) {
        std::lock_guard g(m_);
        cap_bytes_ = b;
    }
    void push(T it) {
        {
            std::lock_guard g(m_);
            const std::uint64_t nb = Traits::bytes(it);
            if (nb > 0) {
                bool over = false;
                while (!q_.empty() && (q_.size() >= cap_ || bytes_ + nb > cap_bytes_)) {
                    bytes_ -= Traits::bytes(q_.front());
                    Traits::drop(q_.front());
                    q_.pop_front();
                    ++dropped_;
                    over = true;
                }
                if (over && !in_overflow_) ++episodes_;
                in_overflow_ = over;
            }
            bytes_ += nb;
            q_.push_back(it);
        }
        cv_.notify_one();
    }
    // Laukia ≤ wait; false — tuščia.
    bool pop(T& out, std::chrono::milliseconds wait) {
        std::unique_lock lk(m_);
        if (!cv_.wait_for(lk, wait, [&] { return !q_.empty(); })) return false;
        out = q_.front();
        q_.pop_front();
        bytes_ -= Traits::bytes(out);
        return true;
    }
    void clear() {
        std::lock_guard g(m_);
        for (T& it : q_) Traits::drop(it);
        q_.clear();
        bytes_ = 0;
    }
    // B4: išmeta tik paketus (sesijos ir kitos žymės lieka); grąžina išmestų skaičių
    std::uint64_t drop_packets() {
        std::lock_guard g(m_);
        std::uint64_t n = 0;
        std::deque<T> keep;
        for (T& it : q_) {
            if (Traits::bytes(it) > 0) {
                Traits::drop(it);
                ++n;
            } else {
                keep.push_back(it);
            }
        }
        q_.swap(keep);
        bytes_ = 0;
        return n;
    }
    std::uint64_t dropped() const {
        std::lock_guard g(m_);
        return dropped_;
    }
    std::uint64_t overflow_episodes() const {
        std::lock_guard g(m_);
        return episodes_;
    }
    std::uint64_t bytes() const {
        std::lock_guard g(m_);
        return bytes_;
    }
    std::size_t size() const {
        std::lock_guard g(m_);
        return q_.size();
    }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<T> q_;
    std::size_t cap_;
    std::uint64_t cap_bytes_;
    std::uint64_t bytes_ = 0;
    std::uint64_t dropped_ = 0, episodes_ = 0;
    bool in_overflow_ = false;
};

}  // namespace mxlv::ingest
