// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// TsSender — vieno enc srauto tinklo išvestis (OK-1): muxer'is → ribota baitų eilė → siuntimo gija → avio (udp / srt).
// Kodavimo kelias tinklo nelaukia: push niekada neblokuoja (tik trumpas užraktas ir memcpy). Gija atidaro išvestį, rašo
// 1316 B (7 × 188) gabalais, nutrūkus ryšiui uždaro ir jungiasi iš naujo su atidėjimu 0,5 → 1 → 2 → 4 → 5 s (po sėkmingo
// atidarymo — vėl nuo 0,5 s). SRT listener „atidarymas“ trunka, kol prisijungia imtuvas (connected — false).
// Pilna eilė → išmetami SENIAUSI duomenys sveikais 188 B TS paketais (packets_dropped), todėl imtuvas po bet kokio
// prisijungimo gauna duomenis nuo TS paketo ribos (PAT / PMT muxer'is kartoja pats, kas 0,1 s).
// Prielaida: push gauna sveikus TS paketus (n — 188 kartotinis; mpegts muxer'is rašo po 188 B, AVIOContext buferis 1316 B).
#pragma once

extern "C" {
#include <libavformat/avio.h>
}

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mxlv::func {

struct SenderStats {
    std::uint64_t bytes_sent = 0;       // sėkmingai išrašyti baitai (188 kartotinis)
    std::uint64_t packets_dropped = 0;  // 188 B TS paketai, kurie neišsiųsti: pilna eilė, nepavykęs rašymas, likę sustabdant
    std::uint64_t reconnects = 0;       // atidarymo bandymai po pirmojo (nepavykę open ir prisijungimai po atsijungimo)
    bool connected = false;             // išvestis atidaryta (UDP — iškart; SRT listener — kai prisijungė imtuvas); po stop() —
                                        // būsena sustabdymo metu
};

class TsSender {
public:
    static constexpr std::size_t kTsPacket = 188;
    static constexpr std::size_t kChunk = 7 * kTsPacket;  // 1316 B — vienas UDP datagramas / SRT live paketas
    static constexpr std::chrono::milliseconds kDrainDefault{500};

    // Išvesties atidarymas (produkcijoje — avio_open2 su šiuo pertraukimo callback'u: jis grąžina 1, kai stop() nori
    // nutraukti užblokuotą open / rašymą). Testuose — imitacija. Grąžina < 0 — klaida.
    using OpenFn = std::function<int(AVIOContext**, const AVIOInterruptCB&)>;
    // Uždarymas (tuščias — avio_closep, t. y. kontekstas iš avio_open2).
    using CloseFn = std::function<void(AVIOContext**)>;

    // url_redacted — tik žurnalui (be paslapčių). max_queue_bytes apvalinamas žemyn iki 188 kartotinio (bent kChunk).
    TsSender(std::string url_redacted, OpenFn open, std::size_t max_queue_bytes, CloseFn close = {});
    ~TsSender();
    TsSender(const TsSender&) = delete;
    TsSender& operator=(const TsSender&) = delete;

    // Muxer'io gija. Niekada neblokuoja (išskyrus trumpą užraktą): pilna eilė → seniausi 188 B kartotiniai išmetami.
    void push(const std::uint8_t* p, std::size_t n);
    // Bet kuri gija (statistikos gija, santrauka).
    SenderStats stats() const;
    // Sustabdo giją: prisijungus eilė išsiunčiama ne ilgiau kaip drain, tada užblokuotas rašymas / open nutraukiamas
    // (pertraukimo callback'as). Visa trukmė ≤ drain + ~0,2 s. Kartotinis kvietimas — nieko nedaro.
    void stop(std::chrono::milliseconds drain = kDrainDefault);

private:
    void run();
    bool wait_backoff(std::chrono::milliseconds d);  // false — sustabdyta
    std::size_t pop(std::uint8_t* dst, std::size_t max);  // m_ laikomas
    void drop_queue();  // m_ laikomas: likusi eilė → packets_dropped
    static int interrupt_cb(void* opaque);

    std::string url_;
    OpenFn open_;
    CloseFn close_;
    AVIOInterruptCB int_cb_{};

    mutable std::mutex m_;
    std::condition_variable cv_;       // duomenys eilėje / sustabdymas / atidėjimo pabaiga
    std::condition_variable exited_cv_;
    std::vector<std::uint8_t> ring_;   // žiedas; head_ visada TS paketo riboje
    std::size_t head_ = 0, size_ = 0;
    bool stop_req_ = false, exited_ = false, stopped_ = false;  // m_

    std::atomic<bool> stopping_{false};  // = stop_req_ (pertraukimo callback'ui be užrakto)
    std::atomic<bool> abort_{false};     // drain baigėsi — nutraukti ir rašymą
    std::atomic<bool> opening_{false};   // gija open viduje (pertraukimas — iškart, kai stopping_)
    std::atomic<std::uint64_t> bytes_sent_{0}, packets_dropped_{0}, reconnects_{0};
    std::atomic<bool> connected_{false};
    std::thread th_;  // paskutinis: gija paleidžiama, kai visi laukai paruošti
};

}  // namespace mxlv::func
