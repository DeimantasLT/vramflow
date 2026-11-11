// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Koderio garso laiko juosta (kodo auditas 2026-11-11, radinys 2): AAC kadrų PTS visada išvedamas iš kadro / grain'o
// indekso (sample_begin), ne iš nuo pirmo kadro skaičiuojamo skaitiklio. Gryna logika — be libav, MXL ir GPU.
//
// Modelis: koderio FIFO laiko mėginius [head, head + queued); kiekvienas kadro segmentas [begin, begin + count) turi
// atsidurti FIFO tiksliai savo vietoje. Kai FIFO pabaiga nesutampa su segmento pradžia:
//   spraga ≤ max_silence — užpildoma tyla (AAC srautas lieka tolydus, PTS teisingas, girdimo „šiukšlių“ nėra);
//   spraga > max_silence — perinkaravimas: FIFO likutis papildomas tyla iki pilno AAC kadro ir išstumiamas su senuoju
//                          PTS, toliau head = begin (TS'e lieka garso spraga, sutampanti su video spraga; ilgos tylos
//                          sekundžių sekundėmis nekoduojame vienu kartu — darbininkas nestringa);
//   persidengimas (begin < FIFO pabaiga) — jau praeityje esantys segmento mėginiai išmetami (PTS niekada neina atgal).
// Invariantas po kiekvieno submit(): head + queued == begin + count (arba nepasikeitęs, jei visas segmentas praeityje).
#include <algorithm>
#include <cstdint>

namespace mxlv::timing {

// Ką darbininkas turi padaryti su FIFO prieš rašydamas segmento mėginius (tvarka — kaip laukai).
struct AudioPlan {
    std::int64_t flush_pad = 0;     // 1) tylos mėginiai FIFO likučiui iki pilno kadro (tik perinkaruojant)
    std::int64_t flush_frames = 0;  // 2) kiek pilnų kadrų išstumti iš FIFO ...
    std::int64_t flush_pts = 0;     //    ... pirmojo PTS (mėginiais; kiti — + k · frame_size)
    std::int64_t silence = 0;       // 3) tylos mėginiai prieš segmentą (spragai užpildyti)
    std::int64_t skip = 0;          // 4) segmento pradžios mėginiai, kurių nerašyti (jau praeityje)
};

class AudioTimeline {
public:
    // frame_size — AAC kadro mėginiai (1024); max_silence — ilgiausia tyla, kuria dar užpildoma spraga (>= frame_size).
    explicit AudioTimeline(std::int64_t frame_size = 1024, std::int64_t max_silence = 48000)
        : frame_(std::max<std::int64_t>(frame_size, 1)), max_silence_(std::max(max_silence, frame_)) {}

    // Segmentas [begin, begin + count) (mėginių indeksai nuo epochos, t. y. sample_begin(kadrų dažnis, F)).
    AudioPlan submit(std::int64_t begin, std::int64_t count) {
        AudioPlan p;
        if (head_ < 0) {  // pirmas segmentas — inkaras
            head_ = begin;
            queued_ = count;
            return p;
        }
        const std::int64_t end = head_ + queued_;
        if (begin > end) {
            const std::int64_t gap = begin - end;
            if (gap <= max_silence_) {
                p.silence = gap;
                queued_ += gap + count;
                silence_samples_ += gap;
                return p;
            }
            if (queued_ > 0) {
                const std::int64_t rem = queued_ % frame_;
                p.flush_pad = rem ? frame_ - rem : 0;
                p.flush_frames = (queued_ + p.flush_pad) / frame_;
                p.flush_pts = head_;
                silence_samples_ += p.flush_pad;
            }
            head_ = begin;
            queued_ = count;
            ++reanchors_;
            return p;
        }
        p.skip = std::min(count, end - begin);
        queued_ += count - p.skip;
        skipped_samples_ += p.skip;
        return p;
    }

    bool ready() const { return head_ >= 0 && queued_ >= frame_; }
    // Kito pilno kadro PTS (mėginiais) — kviesti tik kai ready(); FIFO nuskaitomas tiek pat.
    std::int64_t take() {
        const std::int64_t pts = head_;
        head_ += frame_;
        queued_ -= frame_;
        return pts;
    }

    std::int64_t head() const { return head_; }
    std::int64_t queued() const { return queued_; }
    std::int64_t frame_size() const { return frame_; }
    std::uint64_t silence_samples() const { return static_cast<std::uint64_t>(silence_samples_); }
    std::uint64_t skipped_samples() const { return static_cast<std::uint64_t>(skipped_samples_); }
    std::uint64_t reanchors() const { return static_cast<std::uint64_t>(reanchors_); }

private:
    std::int64_t frame_, max_silence_;
    std::int64_t head_ = -1;  // FIFO pirmo mėginio indeksas (< 0 — dar neinkaruota)
    std::int64_t queued_ = 0;
    std::int64_t silence_samples_ = 0, skipped_samples_ = 0, reanchors_ = 0;
};

}  // namespace mxlv::timing
