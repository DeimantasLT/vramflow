// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Šaltinio laikrodis — plano §P1 algoritmas (žingsnių numeriai komentaruose sutampa su planu) su K6 dviem poslinkiais
// ir epochomis nutrūkimams (žr. source_clock.hpp).
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
    cur_.id = next_epoch_++;
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

bool SourceClock::is_jump(std::int64_t d, Rational tb) const {
    // 2. Nutrūkimas: DTS šuolis pirmyn > disc_fwd arba atgal > disc_back
    const std::int64_t dn = to_ns(d, tb);
    return dn > cfg_.disc_fwd_ns || dn < -cfg_.disc_back_ns;
}

std::int64_t SourceClock::quantise(std::int64_t off, std::int64_t pts_ns) const {
    // 5. Kvantavimas: video paketo PTS ant artimiausio kadrų tinklelio taško (poslinkis pakinta ≤ Tf/2)
    if (tf_ == 0) return off;
    const std::int64_t t = pts_ns + off;
    return off + (time_of(index_of(t, cfg_.align_rate), cfg_.align_rate) - t);
}

void SourceClock::start_epoch(std::int64_t arrival_ns) {
    if (cur_.ready) {  // užfiksuota epocha lieka gyva vėluojantiems kitų srautų paketams
        old_ = cur_;
        old_alive_ = true;
        old_deadline_ = arrival_ns + cfg_.epoch_grace_ns;
        discontinuities_.fetch_add(1, std::memory_order_relaxed);
    }
    cur_ = Epoch{};
    cur_.id = next_epoch_++;
}

void SourceClock::reset_drift() {
    win_open_ = false;
    low_ = high_ = 0;
    pts_n_ = pts_head_ = 0;
    have_last_media_ = false;
    ramp_acc_ = 0;
    ppm_ = 0;
    drift_ppm_.store(0.0, std::memory_order_relaxed);
}

void SourceClock::publish() {
    constexpr auto r = std::memory_order_relaxed;
    locked_pub_.store(cur_.ready, r);
    off_v_pub_.store(cur_.v_ready ? cur_.off_v : cur_.off_a, r);
    off_a_pub_.store(cur_.off_a, r);
}

bool SourceClock::close_window(std::int64_t arrival_ns) {
    // 9. Dreifo įvertis: mažiausių kvadratų nuolydis žalios atsargos m_min(a) per paskutinius kDriftPoints langų.
    pts_[pts_head_] = {arrival_ns, win_min_};
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
        if (den > 0) {
            ppm_ = std::clamp((n * sxy - sx * sy) / den * 1e6, -kMaxCompPpm, kMaxCompPpm);
            drift_ppm_.store(ppm_, std::memory_order_relaxed);
        }
    }
    // Apsauga: garso atsarga (su off_a) drift_windows langų iš eilės už L ± drift_limit — latencija pasikeitė
    // daugiau, nei sugeria rampa → naujas fiksavimas.
    const std::int64_t m_min = win_min_ + cur_.off_a;
    if (m_min < cfg_.latency_ns - cfg_.drift_limit_ns) {
        ++low_;
        high_ = 0;
    } else if (m_min > cfg_.latency_ns + cfg_.drift_limit_ns) {
        ++high_;
        low_ = 0;
    } else {
        low_ = high_ = 0;
    }
    return low_ >= cfg_.drift_windows || high_ >= cfg_.drift_windows;
}

Stamp SourceClock::observe(int stream, std::int64_t pts_raw, std::int64_t dts_raw, std::int64_t dur_raw, std::int64_t arrival_tai_ns) {
    constexpr Stamp kDrop{0, 0, Event::Dropped};
    if (stream < 0 || static_cast<std::size_t>(stream) >= streams_.size() || !streams_[static_cast<std::size_t>(stream)].added) return kDrop;
    if (dts_raw == kNoTimestamp) dts_raw = pts_raw;
    if (pts_raw == kNoTimestamp) pts_raw = dts_raw;
    if (dts_raw == kNoTimestamp) return kDrop;

    Stream& s = streams_[static_cast<std::size_t>(stream)];
    const bool lock_kind = s.is_video || !has_video_;  // fiksavimo / dreifo srautas: video, o be video — bet kuris
    if (old_alive_ && arrival_tai_ns > old_deadline_) old_alive_ = false;

    // 1. Išvyniojimas ir epochos parinkimas. Srautas dabartinėje epochoje — savo ankstesnio paketo atžvilgiu (šuolis =
    // nutrūkimas). Srautas, kurio paskutinis paketas senoje (dar gyvoje) epochoje: jei arti savo ankstesnio — tai
    // senos epochos paketas; kitaip jis jau naujoje ir pririšamas prie paskutinio dabartinės epochos paketo (taip pat
    // pirmas srauto paketas) — visi srautai išvynioti vienodai ir per wrap, ir po nutrūkimo.
    std::int64_t dts_ext;
    bool in_old = false, jump = false;
    if (s.have_last && s.epoch == cur_.id) {
        const std::int64_t d = wrap_delta(dts_raw, s.last_raw);
        dts_ext = s.last_ext + d;
        jump = is_jump(d, s.tb);
    } else if (s.have_last && old_alive_ && s.epoch == old_.id && !is_jump(wrap_delta(dts_raw, s.last_raw), s.tb)) {
        in_old = true;
        dts_ext = s.last_ext + wrap_delta(dts_raw, s.last_raw);
    } else if (cur_.have_ref && same_tb(cur_.ref_tb, s.tb)) {
        dts_ext = cur_.ref_ext + wrap_delta(dts_raw, cur_.ref_raw);
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
    if (jump) start_epoch(arrival_tai_ns);

    const std::int64_t dts_ns = to_ns(dts_ext, s.tb), pts_ns = to_ns(pts_ext, s.tb);
    const std::int64_t dur_ns = dur_raw > 0 ? to_ns(dur_raw, s.tb) : (s.is_video ? tf_ : 0);

    if (in_old) {  // senos epochos paketas — jos poslinkiais (sustingusiais nutrūkimo metu)
        const std::int64_t off = s.is_video && old_.v_ready ? old_.off_v : old_.off_a;
        const Stamp out{pts_ns + off, dts_ns + off, Event::None};
        s.end_tai = out.dts_tai_ns + dur_ns;
        s.have_end = true;
        return out;
    }

    s.epoch = cur_.id;
    cur_.have_ref = true;
    cur_.ref_tb = s.tb;
    cur_.ref_raw = dts_raw;
    cur_.ref_ext = dts_ext;
    if (wrap_mod_ != 0) {  // wrap — šaltinio laiko juostos perėjimas per M kartotinį (ne kiekvieno srauto)
        const auto hi = static_cast<std::int64_t>(floor_div(dts_ext, static_cast<i128>(wrap_mod_)));
        if (!have_wrap_hi_) {
            have_wrap_hi_ = true;
            wrap_hi_ = hi;
        } else if (hi > wrap_hi_) {
            wraps_.fetch_add(static_cast<std::uint64_t>(hi - wrap_hi_), std::memory_order_relaxed);
            wrap_hi_ = hi;
        }
    }
    if (old_alive_)
        old_alive_ = std::any_of(streams_.begin(), streams_.end(), [&](const Stream& t) { return t.have_last && t.epoch == old_.id; });

    Event ev = Event::None;
    const bool grid_video = s.is_video && tf_ > 0;
    if (!cur_.ready) {
        if (!ever_locked_) {
            // 3. Pirmas fiksavimas — ties pirmu video paketu (be video — pirmu paketu); ankstesni metami
            if (!lock_kind) {
                prelock_dropped_.fetch_add(1, std::memory_order_relaxed);
                return kDrop;
            }
            cur_.off_a = arrival_tai_ns + cfg_.latency_ns - dts_ns;
            if (grid_video) {
                cur_.off_v = quantise(cur_.off_a, pts_ns);
                cur_.off_a = cur_.off_v;  // pradžioje A ir V sutampa
                cur_.v_ready = true;
            }
            ever_locked_ = true;
            ev = Event::Locked;
        } else {
            // 4. Perfiksavimas — pirmas naujos epochos paketas (bet kurio srauto): tęsinys iš šio srauto pabaigos E
            const std::int64_t fresh = arrival_tai_ns + cfg_.latency_ns - dts_ns;
            const std::int64_t cont = s.end_tai - dts_ns;
            const bool use_cont = s.have_end && abs64(cont - fresh) <= cfg_.relock_slack_ns;
            (use_cont ? relock_cont_ : relock_fresh_).fetch_add(1, std::memory_order_relaxed);
            const std::int64_t off = use_cont ? cont : fresh;
            if (grid_video) {
                cur_.off_v = quantise(off, pts_ns);
                cur_.v_ready = true;
                // tęsinyje garsas pasislenka tiek pat, kiek video — A/V santykis išlieka
                cur_.off_a = use_cont && old_.v_ready ? old_.off_a + (cur_.off_v - old_.off_v) : cur_.off_v;
            } else {
                cur_.off_a = off;
            }
            ev = Event::Relocked;
        }
        cur_.ready = true;
        reset_drift();
    } else if (grid_video && !cur_.v_ready) {
        // Epochą pradėjo garsas: video tinklelis — artimiausias tolydžiam garso poslinkiui (≤ Tf/2)
        cur_.off_v = quantise(cur_.off_a, pts_ns);
        cur_.v_ready = true;
        ev = Event::Relocked;
    }

    if (lock_kind) {
        if (ev == Event::None && win_open_ && arrival_tai_ns - win_start_ >= cfg_.drift_window_ns) {
            win_open_ = false;
            if (close_window(arrival_tai_ns)) {
                cur_.off_a = arrival_tai_ns + cfg_.latency_ns - dts_ns;
                if (grid_video) cur_.off_a = cur_.off_v = quantise(cur_.off_a, pts_ns);
                relock_fresh_.fetch_add(1, std::memory_order_relaxed);
                ev = Event::Relocked;
                reset_drift();
            }
        }
        // Garso rampa: off_a seka šaltinio laikrodį (d off_a = −ppm · d medija), be laiptų
        if (have_last_media_) {
            ramp_acc_ -= ppm_ * 1e-6 * static_cast<double>(dts_ns - last_media_);
            const auto whole = static_cast<std::int64_t>(ramp_acc_);
            cur_.off_a += whole;
            ramp_acc_ -= static_cast<double>(whole);
        }
        have_last_media_ = true;
        last_media_ = dts_ns;
        // Video slydimas sveiku kadru, kad |off_v − off_a| ≤ Tf/2 (K6)
        if (tf_ > 0 && cur_.v_ready) {
            bool slipped = false;
            while (cur_.off_v - cur_.off_a > tf_ / 2) cur_.off_v -= tf_, slipped = true;
            while (cur_.off_a - cur_.off_v > tf_ / 2) cur_.off_v += tf_, slipped = true;
            if (slipped) {
                drift_slips_.fetch_add(1, std::memory_order_relaxed);
                if (ev == Event::None) ev = Event::Slipped;
            }
        }
    }

    // 6. Žyma
    const std::int64_t off = s.is_video && cur_.v_ready ? cur_.off_v : cur_.off_a;
    const Stamp out{pts_ns + off, dts_ns + off, ev};
    s.end_tai = out.dts_tai_ns + dur_ns;
    s.have_end = true;
    if (lock_kind) {  // žalia atsarga ns(dts) − a; lango minimumas
        const std::int64_t r = dts_ns - arrival_tai_ns;
        if (!win_open_) {
            win_open_ = true;
            win_start_ = arrival_tai_ns;
            win_min_ = r;
        } else {
            win_min_ = std::min(win_min_, r);
        }
    }
    publish();
    return out;
}

void SourceClock::on_clock_step(std::int64_t step_ns) {
    // 10. Laikrodžio šuolis: abu poslinkiai sveiku kadrų skaičiumi (be tinklelio — tiksliai Δ); dreifo langai iš
    // naujo, nes atvykimo laikas pašoko kartu.
    clock_steps_.fetch_add(1, std::memory_order_relaxed);
    if (!cur_.ready) return;
    const std::int64_t k = tf_ > 0 ? round_to_multiple(step_ns, tf_) : step_ns;
    cur_.off_v += k;
    cur_.off_a += k;
    old_.off_v += k;
    old_.off_a += k;
    for (Stream& s : streams_) s.end_tai += k;
    reset_drift();
    publish();
}

bool SourceClock::locked() const { return locked_pub_.load(std::memory_order_relaxed); }

std::int64_t SourceClock::offset_ns() const { return off_v_pub_.load(std::memory_order_relaxed); }

std::int64_t SourceClock::audio_offset_ns() const { return off_a_pub_.load(std::memory_order_relaxed); }

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
