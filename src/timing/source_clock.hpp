// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Šaltinio laikrodis: paketo PTS/DTS (srauto laiko bazė, galimas wrap) → TAI ns su VIENU poslinkiu šaltiniui, bendru
// visiems jo flow'ams (video, garsas, ...). Be MXL, be libav, be GPU: atvykimo laikas ir laikrodžio šuoliai paduodami
// parametrais, todėl visa logika testuojama vienetų testais. Algoritmas — docs/planai/2026-11-11-production-grade.md
// §P1 (10 žingsnių); dreifo politika — §6 K6 (čia tik įvertis ppm ir kadro slydimo įvykiai, kompensaciją taiko ingest).
//
// Gijos: observe() ir on_clock_step() kviečia viena (demux) gija; stats(), locked(), offset_ns() — bet kuri.
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#include "timing/index.hpp"

namespace mxlv::timing {

// Laiko žyma nežinoma (== AV_NOPTS_VALUE). Trūkstamas DTS imamas iš PTS ir atvirkščiai.
inline constexpr std::int64_t kNoTimestamp = INT64_MIN;

struct SourceClockConfig {
    int wrap_bits = 33;                            // 0 — be wrap (pvz. MP4); MPEG-TS — 33
    std::int64_t latency_ns = 1'000'000'000;      // L: TAI = atvykimas + L (K7)
    std::int64_t disc_fwd_ns = 1'000'000'000;     // DTS šuolis pirmyn > šio — nutrūkimas
    std::int64_t disc_back_ns = 0;                // DTS šuolis atgal > šio — nutrūkimas
    std::int64_t relock_slack_ns = 500'000'000;   // |tęsinys − naujas| ≤ šio → tęsiama be šuolio
    std::int64_t drift_limit_ns = 200'000'000;    // atsargos m_min nuokrypis nuo L, nuo kurio slystama kadru
    Rational align_rate{0, 1};                    // video kadrų dažnis; {0,1} — nekvantuoti, neslysti
    std::int64_t drift_window_ns = 10'000'000'000;  // m_min lango ilgis (atvykimo laiku)
    int drift_windows = 3;                        // tiek langų iš eilės už ribos → slydimas vienu kadru
};

// Slipped — šio paketo žyma jau su pakeistu poslinkiu (±1 kadras dėl dreifo, K6); žymos galioja.
enum class Event : std::uint8_t { None, Locked, Relocked, Dropped, Slipped };

struct Stamp {
    std::int64_t pts_tai_ns, dts_tai_ns;
    Event ev;  // Dropped → laikai negalioja
};

struct ClockStats {
    std::uint64_t wraps, discontinuities, relock_cont, relock_fresh, drift_slips, clock_steps, prelock_dropped;
    double drift_ppm;  // (šaltinio laikas / atvykimo laikas − 1) · 10⁶; + šaltinis greitesnis. 0 — dar nežinoma
};

class SourceClock {
public:
    explicit SourceClock(const SourceClockConfig& cfg);
    // Srauto registracija prieš pirmą jo paketą. time_base — libav AVStream::time_base.
    void add_stream(int stream, Rational time_base, bool is_video);
    Stamp observe(int stream, std::int64_t pts_raw, std::int64_t dts_raw, std::int64_t dur_raw, std::int64_t arrival_tai_ns);
    // TAI laikrodis pašoko Δ (ClockWatch): poslinkis keičiamas sveiku kadrų skaičiumi round(Δ / Tf) · Tf.
    void on_clock_step(std::int64_t step_ns);
    bool locked() const;
    std::int64_t offset_ns() const;
    ClockStats stats() const;

private:
    enum class State : std::uint8_t { Unlocked, Locked, AwaitRelock };
    struct Stream {
        Rational tb{0, 1};
        bool added = false, is_video = false, have_last = false;
        std::uint64_t epoch = 0;  // šaltinio epocha, kurioje matytas paskutinis paketas
        std::int64_t last_raw = 0, last_ext = 0;
    };
    struct DriftPoint {
        std::int64_t arrival_ns, margin_ns;
    };
    static constexpr std::size_t kDriftPoints = 30;  // regresija per 30 langų (5 min. numatytai)

    std::int64_t wrap_delta(std::int64_t a, std::int64_t b) const;  // (a − b) mod M → [−M/2, M/2)
    std::int64_t quantise(std::int64_t off, std::int64_t pts_ns) const;
    void reset_drift();
    void close_window(std::int64_t arrival_ns, Event& ev);

    SourceClockConfig cfg_;
    std::uint64_t wrap_mod_ = 0;  // M = 2^W; 0 — be wrap
    std::int64_t tf_ = 0;         // kadro trukmė ns; 0 — be tinklelio
    std::vector<Stream> streams_;
    bool has_video_ = false;

    std::atomic<State> state_{State::Unlocked};
    std::atomic<std::int64_t> off_{0};
    std::uint64_t epoch_ = 1;

    // Paskutinis šaltinio paketas (bet kurio srauto) — naujos epochos srautų išvyniojimo atrama.
    bool have_ref_ = false;
    Rational ref_tb_{0, 1};
    std::int64_t ref_raw_ = 0, ref_ext_ = 0;

    // E: paskutinio fiksavimo rūšies paketo dts_tai + trukmė (tęsiniui po nutrūkimo).
    bool have_end_ = false;
    std::int64_t end_tai_ = 0;

    // Dreifas (žingsnis 9)
    bool win_open_ = false;
    std::int64_t win_start_ = 0, win_min_ = 0;
    int low_ = 0, high_ = 0;
    std::int64_t slip_accum_ = 0;  // dreifo slydimų suma nuo fiksavimo (regresijai išlyginti)
    std::array<DriftPoint, kDriftPoints> pts_{};
    std::size_t pts_n_ = 0, pts_head_ = 0;

    std::atomic<std::uint64_t> wraps_{0}, discontinuities_{0}, relock_cont_{0}, relock_fresh_{0}, drift_slips_{0},
        clock_steps_{0}, prelock_dropped_{0};
    std::atomic<double> drift_ppm_{0.0};
};

// Garso rašymo žymeklis (žingsnis 8): sulygina paketus į ištisinę mėginių seką, kai apvalinimo paklaida ≤ tol.
struct AudioCursor {
    bool started = false;
    std::uint64_t next = 0;  // kito laukiamo mėginio indeksas
    // Grąžina pradžios mėginio indeksą; gap > 0 spraga, < 0 persidengimas (mėginiais).
    // Persidengimo atveju rašoma tik dalis nuo seno `next` (pirmi −gap paketo mėginiai metami); `next` atgal nejuda.
    std::uint64_t place(std::int64_t pts_tai_ns, std::uint32_t n, std::uint32_t sample_rate, std::uint32_t tol, std::int64_t& gap);
    void reset() {
        started = false;
        next = 0;
    }
};

}  // namespace mxlv::timing
