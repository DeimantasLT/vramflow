// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Vienas demuxeris šaltiniui (brief'o pamoka 2): video ir garsas iš to paties libavformat konteksto, todėl abu žymimi tuo pačiu
// SourceClock (pamoka 1: MXL indeksas = PTS + VIENAS šaltinio poslinkis). Transportai — libavformat protokolai (K10):
// udp, rtp, srt, http(s) / hls, failas. Pertraukiamas I/O ir tyla — ingest/av_input.hpp (bendra su dec).
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <atomic>
#include <cstdint>
#include <string>

#include "common/format.hpp"
#include "ingest/av_input.hpp"
#include "timing/source_clock.hpp"

namespace mxlv::ingest {

struct SourceSpec {
    std::string url;
    std::string format = "auto";  // "auto" | "1080i50" …
    bool audio = true;
    int audio_channels = 2;
    std::uint32_t silence_ms = 2000;  // --input-silence-ms
};

enum class Kind : std::uint8_t { Video, Audio };
constexpr int kVideoStream = 0, kAudioStream = 1;  // SourceClock srautų numeriai (ne libav indeksai)

struct Packet {
    AVPacket* pkt = nullptr;  // pts / dts — TAI ns (laiko bazė 1/10⁹), kai pažymėtas
    std::int64_t pts_tai_ns = 0, dts_tai_ns = 0;
    bool key = false;
    Kind kind = Kind::Video;
    timing::Event ev = timing::Event::None;
};

// Srauto aprašas iš zondavimo (formato sprendimui ir dekoderiams).
struct StreamInfo {
    bool video = false, audio = false;
    AVRational video_tb{0, 1}, audio_tb{0, 1};
    const AVCodecParameters* vpar = nullptr;  // galioja, kol Demux atidarytas
    const AVCodecParameters* apar = nullptr;
};

class Demux {
public:
    Demux() = default;
    ~Demux() { close(); }
    Demux(const Demux&) = delete;
    Demux& operator=(const Demux&) = delete;

    // Atidaro ir zonduoja (blokuoja iki duomenų / open_timeout_ns / stop). err — URL ir libav priežastis.
    bool open(const SourceSpec& s, const std::atomic<bool>& stop, std::int64_t open_timeout_ns, std::string& err);
    void close();
    bool is_open() const { return ic_ != nullptr; }
    const StreamInfo& info() const { return info_; }
    // Video formatas iš srauto parametrų (--format auto); false — nepalaikomas, why įvardija formatą (kodekas, chroma, bitai,
    // laukų tvarka, rastras, dažnis). HEVC 8 bitų 4:2:0 palaikomas (tas pats NVDEC / programinis kelias).
    bool video_format(vramflow::VideoFormat& out, std::string& why) const;
    // Kitas paketas (video arba garso; kiti srautai praleidžiami). < 0 — pabaiga / klaida / tyla (why).
    int read(AVPacket* pkt, Kind& kind, std::string& why);
    // HLS: #EXT-X-TARGETDURATION ns (0 — ne HLS / nežinoma); naudojama latencijai ≥ 3 segmentai (K7).
    std::int64_t hls_target_ns() const { return hls_target_ns_; }
    Interrupt& interrupt() { return intr_; }

private:
    AVFormatContext* ic_ = nullptr;
    Interrupt intr_;
    int vs_ = -1, as_ = -1;
    StreamInfo info_;
    std::uint32_t silence_ms_ = 0;
    std::int64_t hls_target_ns_ = 0;
    int field_order_ = AV_FIELD_UNKNOWN;  // laukų tvarka iš rakto kadro (key_field_order)
    int key_field_order();
};

// HLS grojaraščio #EXT-X-TARGETDURATION (sekundės → ns) per avio; 0 — nerasta. Master grojaraštyje — pirmo varianto.
std::int64_t hls_target_duration_ns(const std::string& url, AVIOInterruptCB* cb);
bool is_hls_url(const std::string& url);

}  // namespace mxlv::ingest
