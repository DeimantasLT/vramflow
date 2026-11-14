// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bendra libavformat įėjimo logika `dec` ir `ingest` funkcijoms (anksčiau — tik dec.cpp, B3): pertraukiamas I/O (sustabdymas,
// tyla ilgiau nei --input-silence-ms, atidarymo riba) ir sesijų ciklas su atidėjimu (jungtis iš naujo 1, 2, 4 … ≤ 30 s).
extern "C" {
#include <libavformat/avformat.h>
}

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace mxlv::ingest {

std::string averr(int e);
std::uint64_t tai_now();  // TAI ns (mxlGetTime)

// libavformat pertraukimas: kiekviename laukime (atidarymas, zondavimas, av_read_frame) — stop (sustabdymas, iškart), tyla
// (paskutinis paketas senesnis nei silence_ns; tik gavus bent vieną paketą sesijoje) ir atidarymo riba (deadline_tai; 0 — be).
// Kabliukas grąžina 1 — nutraukti (AVERROR_EXIT).
struct Interrupt {
    const std::atomic<bool>* stop = nullptr;
    std::atomic<std::uint64_t> last_pkt{0};  // TAI; 0 — sesijoje dar nė vieno paketo
    std::uint64_t silence_ns = 0;
    std::uint64_t deadline_tai = 0;
    bool silent = false, timed_out = false;
    std::atomic<bool> abort{false};  // sesija nutraukiama iš vidaus (A1: HLS grojaraščių kilpa vidury sesijos)
    void packet() { last_pkt.store(tai_now(), std::memory_order_relaxed); }
    void reset_session() {
        last_pkt.store(0, std::memory_order_relaxed);
        silent = timed_out = false;
        abort.store(false, std::memory_order_relaxed);
    }
};
int interrupt_cb(void* p);
inline void attach_interrupt(AVFormatContext* ic, Interrupt* in) {
    ic->interrupt_callback.callback = interrupt_cb;
    ic->interrupt_callback.opaque = in;
}

// URL schema palaikoma sisteminio FFmpeg? (avio_enum_protocols). Tuščia — taip; kitaip — aiški klaida
// („srt protokolas nepalaikomas šio FFmpeg (libavformat …)“).
std::string protocol_error(const std::string& url);

// Sesijų ciklas (B3): session(why) — viena įėjimo sesija, true — nepataisoma klaida (jungtis iš naujo beprasmiška). Pasibaigusi /
// tyli sesija — log("<name>: <why> — jungiamasi iš naujo po N s (k)"), reconnects++, laukimas su atidėjimu 1, 2, 4 … ≤ 30 s
// (nutraukiamas stop per ≤ 20 ms); sesija, trukusi ≥ 60 s, atidėjimą grąžina į 1 s. Nepataisoma — error užpildomas,
// log("<name>: <why> — srautas sustabdytas"), ciklas baigiasi.
void reconnect_loop(const std::string& name, const std::atomic<bool>& stop, const std::function<bool(std::string& why)>& session,
                    std::uint64_t& reconnects, std::string& error, const std::function<void(const std::string&)>& log);

}  // namespace mxlv::ingest
