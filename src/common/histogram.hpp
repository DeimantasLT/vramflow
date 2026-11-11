// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Log-linijinė histograma (32 sub-kibirai oktavai, ≤ ~3,1 % paklaida), reikšmės ns. Be alokacijų karštame kelyje.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace mxlv {

class Histogram {
public:
    static constexpr int kSub = 32;
    static constexpr int kSubBits = 5;
    static constexpr int kBuckets = kSub + (64 - kSubBits) * kSub;

    static int bucket_of(std::uint64_t v) {
        if (v < kSub) return static_cast<int>(v);
        const int msb = 63 - std::countl_zero(v);
        const int sub = static_cast<int>((v >> (msb - kSubBits)) & (kSub - 1));
        return kSub + (msb - kSubBits) * kSub + sub;
    }
    // Kibiro vidurio reikšmė.
    static std::uint64_t value_of(int b) {
        if (b < kSub) return static_cast<std::uint64_t>(b);
        const int oct = (b - kSub) / kSub;
        const int sub = (b - kSub) % kSub;
        const int msb = oct + kSubBits;
        const std::uint64_t lo = (1ull << msb) | (static_cast<std::uint64_t>(sub) << (msb - kSubBits));
        const std::uint64_t width = 1ull << (msb - kSubBits);
        return lo + width / 2;
    }

    void add(std::uint64_t v) {
        ++counts_[bucket_of(v)];
        ++n_;
        max_ = std::max(max_, v);
    }
    void merge(const Histogram& o) {
        for (int i = 0; i < kBuckets; ++i) counts_[i] += o.counts_[i];
        n_ += o.n_;
        max_ = std::max(max_, o.max_);
    }
    void reset() { *this = Histogram{}; }

    std::uint64_t count() const { return n_; }
    std::uint64_t max() const { return max_; }
    // q ∈ [0,1]. Tuščia → 0.
    std::uint64_t quantile(double q) const {
        if (n_ == 0) return 0;
        const auto target = static_cast<std::uint64_t>(q * static_cast<double>(n_ - 1)) + 1;
        std::uint64_t acc = 0;
        for (int i = 0; i < kBuckets; ++i) {
            acc += counts_[i];
            if (acc >= target) return std::min(value_of(i), max_);
        }
        return max_;
    }

private:
    std::array<std::uint64_t, kBuckets> counts_{};
    std::uint64_t n_ = 0;
    std::uint64_t max_ = 0;
};

}  // namespace mxlv
