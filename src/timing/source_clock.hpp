// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Šaltinio laikrodis: paketo PTS/DTS (srauto laiko bazė, galimas wrap) → TAI ns, bendras visiems šaltinio flow'ams
// (video, garsas, ...). Be MXL, be libav, be GPU: atvykimo laikas ir laikrodžio šuoliai paduodami parametrais, todėl
// visa logika testuojama vienetų testais. Algoritmas — docs/planai/2026-11-11-production-grade.md §P1; dreifo
// politika — §6 K6.
//
// Du poslinkiai (K6): TAI = ns(medijos laikas) + off.
//   off_a (garsas ir kiti ne video srautai) — tolydus: seka šaltinio laikrodį be laiptų (rampa pagal drift_ppm);
//         garso mėginių skaičių prie jos pritaiko ingest per swr_set_compensation (P3), todėl garse nėra trūkių.
//   off_v (video) — kadrų tinklelyje (kiekvienas kadras tiksliai ant grain'o laiko); slysta sveiku kadru (±Tf),
//         kai |off_v − off_a| viršytų Tf/2. Todėl A/V poslinkis visada ≤ Tf/2 (≤ 1 laukas 50i) ir to paties PTS
//         video bei garsas gauna indeksus, sutampančius iki Tf/2.
// Be video tinklelio (align_rate {0,1}) abu poslinkiai sutampa.
//
// Nutrūkimas pradeda naują epochą; senoji lieka gyva epoch_grace_ns arba kol kiekvienas srautas atsiųs naujos
// epochos paketą — taip TS sujungime vėluojantis ar lenkiantis garsas žymimas savo epochos poslinkiu.
//
// Gijos: observe() ir on_clock_step() kviečia viena (demux) gija; stats(), locked(), offset_ns(),
// audio_offset_ns() — bet kuri.
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
    std::int64_t drift_limit_ns = 200'000'000;    // atsarga m_min už L ± šio drift_windows langų → naujas fiksavimas
    Rational align_rate{0, 1};                    // video kadrų dažnis; {0,1} — nekvantuoti, neslysti
    std::int64_t drift_window_ns = 10'000'000'000;  // m_min lango ilgis (atvykimo laiku)
    int drift_windows = 3;                        // langų iš eilės už ribos iki naujo fiksavimo
    std::int64_t epoch_grace_ns = 2'000'000'000;  // kiek ilgiausiai po nutrūkimo priimami senos epochos paketai
};

// Locked — pirmas fiksavimas. Relocked — naujos epochos poslinkis nustatytas (nutrūkimas arba latencijos pokytis);
// taip pat žymimas pirmas naujos epochos video paketas, jei epochą pradėjo garsas. Slipped — šio video paketo
// žyma jau su ±1 kadro poslinkiu (dreifas, K6). Dropped — laikai negalioja.
enum class Event : std::uint8_t { None, Locked, Relocked, Dropped, Slipped };

struct Stamp {
    std::int64_t pts_tai_ns, dts_tai_ns;
    Event ev;
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
    // TAI laikrodis pašoko Δ (ClockWatch): abu poslinkiai keičiami round(Δ / Tf) · Tf (be tinklelio — Δ).
    void on_clock_step(std::int64_t step_ns);
    bool locked() const;
    std::int64_t offset_ns() const;        // video poslinkis (be video tinklelio — == audio_offset_ns)
    std::int64_t audio_offset_ns() const;  // tolydus garso poslinkis
    ClockStats stats() const;

    static constexpr double kMaxCompPpm = 1000.0;  // rampa ribojama swr kompensacijos biudžetu (±0,1 %, K6)

private:
    struct Stream {
        Rational tb{0, 1};
        bool added = false, is_video = false, have_last = false, have_end = false;
        std::uint64_t epoch = 0;  // epocha, kuriai priklausė paskutinis paketas
        std::int64_t last_raw = 0, last_ext = 0;
        std::int64_t end_tai = 0;  // paskutinio paketo dts_tai + trukmė (tęsiniui po nutrūkimo)
    };
    struct Epoch {
        std::uint64_t id = 0;
        bool ready = false;    // off_a nustatytas
        bool v_ready = false;  // off_v nustatytas (pirmas šios epochos video paketas)
        std::int64_t off_v = 0, off_a = 0;
        bool have_ref = false;  // paskutinis šios epochos paketas — kitų srautų išvyniojimo atrama
        Rational ref_tb{0, 1};
        std::int64_t ref_raw = 0, ref_ext = 0;
    };
    struct DriftPoint {
        std::int64_t arrival_ns, margin_ns;
    };
    static constexpr std::size_t kDriftPoints = 30;  // regresija per 30 langų (5 min. numatytai)

    std::int64_t wrap_delta(std::int64_t a, std::int64_t b) const;  // (a − b) mod M → [−M/2, M/2)
    bool is_jump(std::int64_t d, Rational tb) const;
    std::int64_t quantise(std::int64_t off, std::int64_t pts_ns) const;
    void start_epoch(std::int64_t arrival_ns);
    void reset_drift();
    void publish();
    // Grąžina true, jei atsarga per ilgai už ribos (reikia naujo fiksavimo).
    bool close_window(std::int64_t arrival_ns);

    SourceClockConfig cfg_;
    std::uint64_t wrap_mod_ = 0;  // M = 2^W; 0 — be wrap
    std::int64_t tf_ = 0;         // kadro trukmė ns; 0 — be tinklelio
    std::vector<Stream> streams_;
    bool has_video_ = false;
    bool ever_locked_ = false;

    Epoch cur_, old_;
    bool old_alive_ = false;
    std::int64_t old_deadline_ = 0;
    std::uint64_t next_epoch_ = 1;

    bool have_wrap_hi_ = false;
    std::int64_t wrap_hi_ = 0;  // didžiausias floor(ext / M) — wrap skaičiuojamas šaltiniui, ne srautui

    // Dreifas: rampa, m_min langai, regresija
    double ramp_acc_ = 0;  // off_a rampos trupmena (ns)
    bool have_last_media_ = false;
    std::int64_t last_media_ = 0;
    bool win_open_ = false;
    std::int64_t win_start_ = 0, win_min_ = 0;  // win_min_ — žalia atsarga ns(dts) − a (be poslinkio)
    int low_ = 0, high_ = 0;
    std::array<DriftPoint, kDriftPoints> pts_{};
    std::size_t pts_n_ = 0, pts_head_ = 0;
    double ppm_ = 0;

    std::atomic<bool> locked_pub_{false};
    std::atomic<std::int64_t> off_v_pub_{0}, off_a_pub_{0};
    std::atomic<std::uint64_t> wraps_{0}, discontinuities_{0}, relock_cont_{0}, relock_fresh_{0}, drift_slips_{0},
        clock_steps_{0}, prelock_dropped_{0};
    std::atomic<double> drift_ppm_{0.0};
};

// Garso rašymo žymeklis (žingsnis 8): sulygina paketus į ištisinę mėginių seką, kai apvalinimo paklaida ≤ tol.
// n — tiek mėginių, kiek bus įrašyta (po swr kompensacijos), todėl dreifas nesikaupia į spragas.
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
