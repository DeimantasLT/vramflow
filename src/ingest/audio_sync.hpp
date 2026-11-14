// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Garso sekimas mėginio tikslumu (K6) — gryna logika be libav (tests/unit/test_audio_sync.cpp).
//
// AudioSync: kiekvienam dekoduotam garso kadrui kviečiantysis apskaičiuoja, kuriame absoliučiame 48 kHz mėginyje TURĖTŲ prasidėti
// kita resamplerio išvestis (expected = round((PTS_TAI − swr delsa) · 48 000 / 10⁹), PTS_TAI — su SourceClock tolydžiu garso
// poslinkiu off_a), o AudioSync žino, kur ji iš tikrųjų prasidės (next — iki šiol išvestų mėginių galas). Paklaida err = expected − next:
//   |err| ≤ deadband        — nieko nekeisti;
//   |err| ≤ hard_samples     — swr_set_compensation(err, distance): err mėginių pridedama / atimama per distance išvesties mėginių,
//                              distance ≥ |err| · 10⁶ / max_ppm — tempo pokytis ≤ max_ppm (K6: ±0,1 %, negirdimas);
//   |err| > hard_samples     — perinkaravimas: next = expected (pradžia, sesijos pradžia, šuolis; skaičiuojama resyncs).
// Taip garsas be trūkių seka šaltinio laikrodį, o A/V poslinkis lieka ≤ 1 lauko (video slysta sveiku kadru — SourceClock).
//
// AudioRing: išvesties mėginiai absoliučiomis 48 kHz pozicijomis (MXL tęstinio flow'o indeksai); rašo garso gija, skaito tick'as
// (grain'o intervalas [sample_begin(i), sample_begin(i + 1))). Neparašytos pozicijos skaitomos tyla ir skaičiuojamos.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace mxlv::ingest {

struct AudioSyncConfig {
    double max_ppm = 1000.0;           // K6: ±0,1 %
    std::int64_t hard_samples = 960;   // 20 ms (1 laukas 50i) — daugiau — perinkaravimas
    std::int64_t deadband = 1;         // ≤ 1 mėginio — nekompensuoti
    std::int64_t min_distance = 1024;  // kompensacija ne trumpiau kaip per 1024 išvesties mėginius
};

struct AudioSyncStep {
    bool resync = false;
    std::int64_t err = 0;        // expected − next prieš sprendimą
    std::int64_t comp = 0;       // swr_set_compensation sample_delta (0 — be kompensacijos)
    std::int64_t distance = 0;   // swr_set_compensation compensation_distance
};

class AudioSync {
public:
    explicit AudioSync(const AudioSyncConfig& c = {}) : c_(c) {}
    AudioSyncStep on_frame(std::int64_t expected_next) {
        AudioSyncStep s;
        if (!started_) {
            started_ = true;
            next_ = expected_next;
            s.resync = true;
            return s;
        }
        s.err = expected_next - next_;
        const std::int64_t a = s.err < 0 ? -s.err : s.err;
        if (a > c_.hard_samples) {
            next_ = expected_next;
            s.resync = true;
            ++resyncs_;
            return s;
        }
        if (a <= c_.deadband) return s;
        s.comp = s.err;
        s.distance = std::max<std::int64_t>(c_.min_distance, static_cast<std::int64_t>(static_cast<double>(a) * 1e6 / c_.max_ppm + 0.5));
        return s;
    }
    void produced(std::int64_t n) { next_ += n; }
    std::int64_t next() const { return next_; }
    bool started() const { return started_; }
    void reset() { started_ = false; }
    std::uint64_t resyncs() const { return resyncs_; }

private:
    AudioSyncConfig c_;
    bool started_ = false;
    std::int64_t next_ = 0;
    std::uint64_t resyncs_ = 0;
};

class AudioRing {
public:
    // capacity — mėginiai kanalui (apvalinama aukštyn iki 2 laipsnio)
    void init(int channels, std::size_t capacity) {
        std::size_t c = 1;
        while (c < capacity) c <<= 1;
        cap_ = c;
        buf_.assign(static_cast<std::size_t>(channels), std::vector<float>(c, 0.0f));
        any_ = false;
    }
    int channels() const { return static_cast<int>(buf_.size()); }

    // [pos, pos + n): planes[ch][k]. pos > head — tarpas užpildomas tyla (pozicijos tampa galiojančiomis); pos < head — perrašoma.
    void write(std::int64_t pos, const float* const* planes, std::uint32_t n) {
        std::lock_guard g(m_);
        if (!any_ || pos > head_ + static_cast<std::int64_t>(cap_) || pos + static_cast<std::int64_t>(cap_) < head_) {
            any_ = true;
            lo_ = head_ = pos;
        }
        if (pos > head_) {
            put_silence(head_, pos - head_);
            silence_written_ += static_cast<std::uint64_t>(pos - head_);
        }
        for (std::size_t ch = 0; ch < buf_.size(); ++ch)
            for (std::uint32_t k = 0; k < n; ++k) buf_[ch][static_cast<std::size_t>(pos + k) & (cap_ - 1)] = planes[ch][k];
        lo_ = std::min(lo_, pos);
        head_ = std::max(head_, pos + static_cast<std::int64_t>(n));
        lo_ = std::max(lo_, head_ - static_cast<std::int64_t>(cap_));
    }
    // [pos, pos + n) → out[ch] (n float). Grąžina galiojančių mėginių skaičių; kiti — 0,0 (tyla). late — kiek pozicijų dar neparašyta
    // (pos ≥ head: garsas nespėjo); early — jau perrašytos arba prieš pradžią.
    std::uint32_t read(std::int64_t pos, std::uint32_t n, float* const* out, std::uint32_t& late) const {
        std::lock_guard g(m_);
        std::uint32_t valid = 0;
        late = 0;
        for (std::uint32_t k = 0; k < n; ++k) {
            const std::int64_t p = pos + k;
            const bool ok = any_ && p >= lo_ && p < head_;
            if (!ok && any_ && p >= head_) ++late;
            for (std::size_t ch = 0; ch < buf_.size(); ++ch) out[ch][k] = ok ? buf_[ch][static_cast<std::size_t>(p) & (cap_ - 1)] : 0.0f;
            valid += ok ? 1 : 0;
        }
        return valid;
    }
    std::int64_t head() const {
        std::lock_guard g(m_);
        return any_ ? head_ : 0;
    }
    std::uint64_t silence_written() const {
        std::lock_guard g(m_);
        return silence_written_;
    }

private:
    void put_silence(std::int64_t from, std::int64_t n) {
        n = std::min<std::int64_t>(n, static_cast<std::int64_t>(cap_));
        for (auto& b : buf_)
            for (std::int64_t k = 0; k < n; ++k) b[static_cast<std::size_t>(from + k) & (cap_ - 1)] = 0.0f;
    }
    mutable std::mutex m_;
    std::vector<std::vector<float>> buf_;
    std::size_t cap_ = 0;
    bool any_ = false;
    std::int64_t lo_ = 0, head_ = 0;
    std::uint64_t silence_written_ = 0;
};

}  // namespace mxlv::ingest
