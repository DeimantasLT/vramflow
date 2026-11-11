// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Šaltinio laikrodis — plano §P1 algoritmas (žingsnių numeriai komentaruose sutampa su planu).
#include "timing/source_clock.hpp"

#include <algorithm>
#include <stdexcept>

namespace mxlv::timing {

namespace {

constexpr std::int64_t kNsPerSec = 1'000'000'000;

// floor(n / d), d > 0
constexpr i128 floor_div(i128 n, i128 d) {
    i128 q = n / d;
    if (n % d != 0 && n < 0) --q;
    return q;
}

// ns(x) = floor(x · tb.num · 10⁹ / tb.den)
inline std::int64_t to_ns(std::int64_t x, Rational tb) {
    return static_cast<std::int64_t>(floor_div(i128{x} * tb.num * kNsPerSec, tb.den));
}

// round(a / b) · b, pusė — nuo nulio (b > 0)
constexpr std::int64_t round_to_multiple(std::int64_t a, std::int64_t b) {
    return a >= 0 ? (a + b / 2) / b * b : -((-a + b / 2) / b * b);
}

constexpr std::int64_t abs64(std::int64_t v) { return v < 0 ? -v : v; }

constexpr bool same_tb(Rational a, Rational b) { return a.num == b.num && a.den == b.den; }

}  // namespace

SourceClock::SourceClock(const SourceClockConfig& cfg) : cfg_(cfg) {
    if (cfg_.wrap_bits < 0 || cfg_.wrap_bits > 62) throw std::invalid_argument("SourceClock: wrap_bits turi būti 0..62");
    if (cfg_.align_rate.num < 0 || cfg_.align_rate.den <= 0) throw std::invalid_argument("SourceClock: blogas align_rate");
    if (cfg_.drift_window_ns <= 0 || cfg_.drift_windows < 1) throw std::invalid_argument("SourceClock: blogas dreifo langas");
    wrap_mod_ = cfg_.wrap_bits == 0 ? 0 : std::uint64_t{1} << cfg_.wrap_bits;
    if (cfg_.align_rate.num > 0) tf_ = time_of(1, cfg_.align_rate);
}

void SourceClock::add_stream(int stream, Rational time_base, bool is_video) {
    if (stream < 0) throw std::invalid_argument("SourceClock: neigiamas srauto numeris");
    if (time_base.num <= 0 || time_base.den <= 0) throw std::invalid_argument("SourceClock: bloga laiko bazė");
    if (static_cast<std::size_t>(stream) >= streams_.size()) streams_.resize(static_cast<std::size_t>(stream) + 1);
    Stream& s = streams_[static_cast<std::size_t>(stream)];
    s = Stream{};
    s.tb = time_base;
    s.added = true;
    s.is_video = is_video;
    has_video_ = has_video_ || is_video;
}

std::int64_t SourceClock::wrap_delta(std::int64_t a, std::int64_t b) const {
    // Skirtumas be persipildymo UB (C++20: modulinis konvertavimas apibrėžtas)
    const std::uint64_t diff = static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b);
    if (wrap_mod_ == 0) return static_cast<std::int64_t>(diff);
    const std::uint64_t r = diff & (wrap_mod_ - 1);
    return r >= wrap_mod_ / 2 ? static_cast<std::int64_t>(r) - static_cast<std::int64_t>(wrap_mod_) : static_cast<std::int64_t>(r);
}

std::int64_t SourceClock::quantise(std::int64_t off, std::int64_t pts_ns) const {
    // 5. Kvantavimas: fiksavimo paketo PTS ant artimiausio kadrų tinklelio taško (poslinkis pakinta ≤ Tf/2)
    if (cfg_.align_rate.num <= 0) return off;
    const std::int64_t t = pts_ns + off;
    return off + (time_of(index_of(t, cfg_.align_rate), cfg_.align_rate) - t);
}

void SourceClock::reset_drift() {
    win_open_ = false;
    low_ = high_ = 0;
    slip_accum_ = 0;
    pts_n_ = pts_head_ = 0;
}

void SourceClock::close_window(std::int64_t arrival_ns, Event& ev) {
    // Dreifo įvertis: mažiausių kvadratų nuolydis m_min(a), slydimai išlyginti (slip_accum_), paskutiniai kDriftPoints langų.
    pts_[pts_head_] = {arrival_ns, win_min_ - slip_accum_};
    pts_head_ = (pts_head_ + 1) % kDriftPoints;
    pts_n_ = std::min(pts_n_ + 1, kDriftPoints);
    if (pts_n_ >= 3) {
        const std::size_t first = (pts_head_ + kDriftPoints - pts_n_) % kDriftPoints;
        const std::int64_t x0 = pts_[first].arrival_ns, y0 = pts_[first].margin_ns;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (std::size_t i = 0; i < pts_n_; ++i) {
            const DriftPoint& p = pts_[(first + i) % kDriftPoints];
            const double x = static_cast<double>(p.arrival_ns - x0), y = static_cast<double>(p.margin_ns - y0);
            sx += x, sy += y, sxx += x * x, sxy += x * y;
        }
        const double n = static_cast<double>(pts_n_), den = n * sxx - sx * sx;
        if (den > 0) drift_ppm_.store((n * sxy - sx * sy) / den * 1e6, std::memory_order_relaxed);
    }

    // 9. Slydimas: drift_windows langų iš eilės už L ± drift_limit → ±1 kadras (po vieną per langą)
    if (tf_ == 0) return;
    if (win_min_ < cfg_.latency_ns - cfg_.drift_limit_ns) {
        ++low_;
        high_ = 0;
    } else if (win_min_ > cfg_.latency_ns + cfg_.drift_limit_ns) {
        ++high_;
        low_ = 0;
    } else {
        low_ = high_ = 0;
    }
    std::int64_t step = 0;
    if (low_ >= cfg_.drift_windows) step = tf_;          // per maža atsarga: kadras pakartojamas
    else if (high_ >= cfg_.drift_windows) step = -tf_;   // per didelė: kadras išmetamas
    if (step == 0) return;
    off_.store(off_.load(std::memory_order_relaxed) + step, std::memory_order_relaxed);
    slip_accum_ += step;
    drift_slips_.fetch_add(1, std::memory_order_relaxed);
    ev = Event::Slipped;
}

Stamp SourceClock::observe(int stream, std::int64_t pts_raw, std::int64_t dts_raw, std::int64_t dur_raw, std::int64_t arrival_tai_ns) {
    constexpr Stamp kDrop{0, 0, Event::Dropped};
    if (stream < 0 || static_cast<std::size_t>(stream) >= streams_.size() || !streams_[static_cast<std::size_t>(stream)].added) return kDrop;
    if (dts_raw == kNoTimestamp) dts_raw = pts_raw;
    if (pts_raw == kNoTimestamp) pts_raw = dts_raw;
    if (dts_raw == kNoTimestamp) return kDrop;

    Stream& s = streams_[static_cast<std::size_t>(stream)];
    const bool lock_kind = s.is_video || !has_video_;  // fiksavimo / dreifo srautas: video, o be video — bet kuris

    // 1. Išvyniojimas. Srautas, kurio paskutinis paketas iš senos epochos (ar pirmas jo paketas), pririšamas prie
    // paskutinio šaltinio paketo — taip visi srautai išvynioti vienodai (ir per wrap, ir po nutrūkimo).
    std::int64_t dts_ext;
    bool jump = false;
    if (s.have_last && s.epoch == epoch_) {
        const std::int64_t d = wrap_delta(dts_raw, s.last_raw);
        if (wrap_mod_ != 0 && d != static_cast<std::int64_t>(static_cast<std::uint64_t>(dts_raw) - static_cast<std::uint64_t>(s.last_raw)))
            wraps_.fetch_add(1, std::memory_order_relaxed);
        dts_ext = s.last_ext + d;
        // 2. Nutrūkimas
        const std::int64_t dn = to_ns(d, s.tb);
        jump = dn > cfg_.disc_fwd_ns || dn < -cfg_.disc_back_ns;
    } else if (have_ref_ && same_tb(ref_tb_, s.tb)) {
        dts_ext = ref_ext_ + wrap_delta(dts_raw, ref_raw_);
    } else if (s.have_last) {
        dts_ext = s.last_ext + wrap_delta(dts_raw, s.last_raw);
    } else {
        dts_ext = dts_raw;
    }
    // PTS — DTS atžvilgiu (PTS ir DTS gali būti skirtingose wrap pusėse)
    const std::int64_t pts_ext = dts_ext + wrap_delta(pts_raw, dts_raw);

    s.have_last = true;
    s.last_raw = dts_raw;
    s.last_ext = dts_ext;
    if (jump) {
        ++epoch_;  // kiti srautai savo kitą paketą pririš prie šio
        if (state_.load(std::memory_order_relaxed) == State::Locked) {
            state_.store(State::AwaitRelock, std::memory_order_relaxed);
            discontinuities_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    s.epoch = epoch_;
    have_ref_ = true;
    ref_tb_ = s.tb;
    ref_raw_ = dts_raw;
    ref_ext_ = dts_ext;

    const std::int64_t dts_ns = to_ns(dts_ext, s.tb), pts_ns = to_ns(pts_ext, s.tb);
    Event ev = Event::None;
    const State st = state_.load(std::memory_order_relaxed);
    if (st != State::Locked) {
        if (!lock_kind) {
            prelock_dropped_.fetch_add(1, std::memory_order_relaxed);
            return kDrop;
        }
        // 3. Fiksavimas / 4. perfiksavimas po nutrūkimo
        const std::int64_t fresh = arrival_tai_ns + cfg_.latency_ns - dts_ns;
        std::int64_t off = fresh;
        if (st == State::AwaitRelock) {
            const std::int64_t cont = end_tai_ - dts_ns;
            if (have_end_ && abs64(cont - fresh) <= cfg_.relock_slack_ns) {
                off = cont;
                relock_cont_.fetch_add(1, std::memory_order_relaxed);
            } else {
                relock_fresh_.fetch_add(1, std::memory_order_relaxed);
            }
            ev = Event::Relocked;
        } else {
            ev = Event::Locked;
        }
        off_.store(quantise(off, pts_ns), std::memory_order_relaxed);
        state_.store(State::Locked, std::memory_order_relaxed);
        reset_drift();
    } else if (lock_kind && win_open_ && arrival_tai_ns - win_start_ >= cfg_.drift_window_ns) {
        close_window(arrival_tai_ns, ev);  // gali pakeisti poslinkį — šis paketas jau su nauju
        win_open_ = false;
    }

    // 6. Žyma
    const std::int64_t off = off_.load(std::memory_order_relaxed);
    const Stamp out{pts_ns + off, dts_ns + off, ev};
    if (lock_kind) {
        end_tai_ = out.dts_tai_ns + (dur_raw > 0 ? to_ns(dur_raw, s.tb) : tf_);
        have_end_ = true;
        // 9. Atsarga m = dts_tai − a; lango minimumas
        const std::int64_t m = out.dts_tai_ns - arrival_tai_ns;
        if (!win_open_) {
            win_open_ = true;
            win_start_ = arrival_tai_ns;
            win_min_ = m;
        } else {
            win_min_ = std::min(win_min_, m);
        }
    }
    return out;
}

void SourceClock::on_clock_step(std::int64_t step_ns) {
    // 10. Laikrodžio šuolis: poslinkis sveiku kadrų skaičiumi (be tinklelio — tiksliai Δ). Dreifo langai iš naujo, nes
    // atvykimo laikas pašoko kartu.
    clock_steps_.fetch_add(1, std::memory_order_relaxed);
    if (state_.load(std::memory_order_relaxed) == State::Unlocked) return;
    const std::int64_t k = tf_ > 0 ? round_to_multiple(step_ns, tf_) : step_ns;
    off_.store(off_.load(std::memory_order_relaxed) + k, std::memory_order_relaxed);
    end_tai_ += k;
    reset_drift();
}

bool SourceClock::locked() const { return state_.load(std::memory_order_relaxed) == State::Locked; }

std::int64_t SourceClock::offset_ns() const { return off_.load(std::memory_order_relaxed); }

ClockStats SourceClock::stats() const {
    constexpr auto r = std::memory_order_relaxed;
    return {wraps_.load(r),       discontinuities_.load(r), relock_cont_.load(r),     relock_fresh_.load(r),
            drift_slips_.load(r), clock_steps_.load(r),     prelock_dropped_.load(r), drift_ppm_.load(r)};
}

std::uint64_t AudioCursor::place(std::int64_t pts_tai_ns, std::uint32_t n, std::uint32_t sample_rate, std::uint32_t tol, std::int64_t& gap) {
    // 8. Garso tolydumas: s = round(pts · sr / 10⁹); |s − next| ≤ tol → tęsiama nuo next (be apvalinimo plyšių)
    const std::uint64_t s = index_of(pts_tai_ns, Rational{sample_rate, 1});
    if (!started) {
        started = true;
        gap = 0;
        next = s + n;
        return s;
    }
    const std::int64_t diff = static_cast<std::int64_t>(s - next);
    if (abs64(diff) <= static_cast<std::int64_t>(tol)) {
        gap = 0;
        const std::uint64_t start = next;
        next += n;
        return start;
    }
    gap = diff;  // > 0 — spraga (nerašoma), < 0 — persidengimas (pirmi −gap mėginių metami)
    next = std::max(next, s + n);
    return s;
}

}  // namespace mxlv::timing
