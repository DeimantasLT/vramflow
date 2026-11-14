// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Garso dekoderis ingest'ui (planas §P3, K6, K10): libavcodec → libswresample (48 kHz float32 planar, --audio-channels) su
// mėginio tikslumo sekimu (AudioSync: swr_set_compensation ≤ max_ppm, perinkaravimas virš hard_samples). Paketų pts — TAI ns
// (SourceClock tolydus garso poslinkis off_a), todėl išvesties mėginių vieta — absoliutus 48 kHz MXL indeksas.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libswresample/swresample.h>
}

#include <cstdint>
#include <string>
#include <vector>

#include "ingest/audio_sync.hpp"

namespace mxlv::ingest {

struct AudioChunk {
    std::int64_t start = 0;          // pirmo mėginio absoliuti 48 kHz pozicija
    std::uint32_t n = 0;
    const float* const* planes = nullptr;  // channels() plokštumų, galioja iki kito receive
    bool resync = false;
};

class AudioDecoder {
public:
    AudioDecoder() = default;
    ~AudioDecoder() { close(); }
    AudioDecoder(const AudioDecoder&) = delete;
    AudioDecoder& operator=(const AudioDecoder&) = delete;

    bool open(const AVCodecParameters* par, int out_channels, const AudioSyncConfig& sync, std::string& err);
    void close();
    bool is_open() const { return c_ != nullptr; }
    int send(AVPacket* pkt);  // 0, AVERROR(EAGAIN) — pirma receive, kita — klaida (skaičiuojama); neatidarytas — AVERROR(EINVAL)
    // Kitas iškoduotas ir resamplintas gabalas absoliučioje 48 kHz ašyje; false — šiuo metu nėra.
    bool receive(AudioChunk& out);
    // Nauja įėjimo sesija: dekoderis ir resampleris išvalomi, sekimas iš naujo (pirmas kadras — perinkaravimas).
    void reset();
    int channels() const { return out_ch_; }
    std::uint64_t errors() const { return errors_; }
    std::uint64_t resyncs() const { return sync_.resyncs(); }
    std::int64_t comp_abs_samples() const { return comp_abs_; }  // Σ |kompensacija| (mėginiai) — informacinis

private:
    bool ensure_swr(const AVFrame* f);
    AVCodecContext* c_ = nullptr;
    SwrContext* swr_ = nullptr;
    AVFrame* fr_ = nullptr;
    int out_ch_ = 2;
    int in_rate_ = 0, in_fmt_ = -1;
    AVChannelLayout in_layout_{};
    AudioSync sync_;
    std::vector<std::vector<float>> buf_;
    std::vector<float*> ptr_;
    std::uint64_t errors_ = 0;
    std::int64_t comp_abs_ = 0;
};

}  // namespace mxlv::ingest
