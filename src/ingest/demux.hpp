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
#include "ingest/h264_scan.hpp"
#include "ingest/url_policy.hpp"
#include "timing/source_clock.hpp"

namespace mxlv::ingest {

struct SourceSpec {
    std::string url;
    std::string format = "auto";  // "auto" | "1080i50" …
    bool audio = true;
    int audio_channels = 2;
    std::uint32_t silence_ms = 2000;  // --input-silence-ms
    double hls_max_target_s = 30;     // --hls-max-target-s (A1)
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
    // B7: kuo prasidėjo zondavimo skenavimas (H.264) ir kiek paketų tam išmesta
    Entry entry() const { return entry_; }
    std::uint64_t entry_discarded() const { return entry_discarded_; }
    Interrupt& interrupt() { return intr_; }

private:
    AVFormatContext* ic_ = nullptr;
    Interrupt intr_;
    int vs_ = -1, as_ = -1;
    StreamInfo info_;
    std::uint32_t silence_ms_ = 0;
    std::int64_t hls_target_ns_ = 0;
    int field_order_ = AV_FIELD_UNKNOWN;  // laukų tvarka iš rakto kadro (key_field_order)
    Entry entry_ = Entry::None;
    std::uint64_t entry_discarded_ = 0;
    int key_field_order();

public:
    // libav io_open sargas (A1; viešas tik dėl C kabliuko): HLS — kiekvienas libav atidarymas (grojaraščiai, segmentai, raktai)
    // tik http / https; atidarymų daugiau nei kMaxOpensPerSec per sekundę — grojaraščių kilpa (libav HLS demuxeris perkraunamą
    // media grojaraštį, tapusį master, kaupia be galo ir galiausiai griūva) → sesija nutraukiama, šaltinis jungiasi iš naujo.
    int guarded_open(AVFormatContext* s, AVIOContext** pb, const char* url, int flags, AVDictionary** opts);
    static constexpr int kMaxOpensPerSec = 50;

private:
    int (*orig_io_open_)(AVFormatContext*, AVIOContext**, const char*, int, AVDictionary**) = nullptr;
    bool guard_http_ = false;
    std::uint64_t open_win0_ = 0;
    int opens_in_win_ = 0;
    std::uint64_t opens_total_ = 0;  // sesijos libav atidarymai (pirmas — viršutinis URL)
    static void ff_close(AVFormatContext* s, AVIOContext** pb);
    std::uint64_t refused_opens_ = 0;
    std::string shown_;  // šaltinio URL be paslapčių
};

// HTTP(S) šaltinio patikra prieš libav (A1, B8): ar turinys — M3U8 (hls), ir jei taip — master → media (ne giliau, be kilpų),
// visos nuorodos tik http / https, #EXT-X-TARGETDURATION ≤ lim.max_target_s (target_ns — didžiausias iš variantų; 0 — nėra).
// false — šaltinis netinkamas (err — aiški priežastis be paslapčių).
bool hls_precheck(const std::string& url, AVIOInterruptCB* cb, const HlsLimits& lim, bool& hls, std::int64_t& target_ns, std::string& err);

}  // namespace mxlv::ingest
