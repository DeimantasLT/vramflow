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
#include <vector>

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
    std::size_t hls_max_playlist = 1u << 20;  // --hls-max-playlist, baitai (H2)
    bool hls_same_host = false;               // --hls-same-host: segmentai / raktai / variantai tik iš grojaraščio serverio
};

// Atsisiųstas grojaraštis: tekstas ir HTTP atsakymo savybės (nukreipimo URL, mime tipas, slapukai) — libav gauna tuos pačius baitus.
struct FetchedPlaylist {
    std::string text, location, mime, cookies;
};
// H2: atminties AVIOContext su grojaraščio tekstu ir HTTP savybėmis (av_opt_get location / mime_type / cookies / rw_timeout veikia);
// atlaisvinamas TIK free_mem_playlist_io (ne avio_close).
AVIOContext* mem_playlist_io(const FetchedPlaylist& f);
bool is_mem_playlist_io(const AVIOContext* io);
void free_mem_playlist_io(AVIOContext** io);
// A (round 11): kontekstas, kuris pirmiausia atiduoda sargo peržiūrėtus baitus (head, n ≤ 4096) iš atminties, tada skaito toliau iš
// inner (tinklo konteksto niekada negrąžina atgal — jokios antros užklausos); av_opt_get(…, AV_OPT_SEARCH_CHILDREN) — per inner.
// Atlaisvinamas TIK free_peek_io (grąžina inner — jį uždaro kvietėjas tuo pačiu būdu, kaip atidarė).
AVIOContext* peek_io(AVIOContext* inner, const std::uint8_t* head, int n);
bool is_peek_io(const AVIOContext* io);
AVIOContext* free_peek_io(AVIOContext** io);

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
    // libav io_open sargas (A1, H1; viešas tik dėl C kabliuko), kiekvienam HTTP(S) šaltiniui: kiekvienas libav atidarymas (grojaraščiai,
    // segmentai, raktai) tik http / https; ne HLS šaltinis — tik vienas atidarymas sesijai, o jo atsakymas M3U8 — atmetamas; atidarymų daugiau nei kMaxOpensPerSec per sekundę — grojaraščių kilpa (libav HLS demuxeris perkraunamą
    // media grojaraštį, tapusį master, kaupia be galo ir galiausiai griūva) → sesija nutraukiama, šaltinis jungiasi iš naujo.
    int guarded_open(AVFormatContext* s, AVIOContext** pb, const char* url, int flags, AVDictionary** opts);
    int guarded_close(AVFormatContext* s, AVIOContext* pb);
    static constexpr int kMaxOpensPerSec = 50;

private:
    int pre_open(const std::string& u);                                // schema, ribos — prieš atidarymą
    int post_open(AVFormatContext* s, AVIOContext** pb, const std::string& u);  // peržiūra, M3U8 patikra, peek_io
    int open_top(const std::string& url);  // A: ne HLS HTTP viršutinis kontekstas (mūsų, AVFMT_FLAG_CUSTOM_IO) → top_pb_

private:
    int (*orig_io_open_)(AVFormatContext*, AVIOContext**, const char*, int, AVDictionary**) = nullptr;
    int (*orig_io_close2_)(AVFormatContext*, AVIOContext*) = nullptr;
    HlsLimits lim_;
    FetchedPlaylist top_;            // H2: viršutinis grojaraštis — tie patys baitai patikrai ir libav (AVFMT_FLAG_CUSTOM_IO)
    // viršutinis kontekstas (mūsų, AVFMT_FLAG_CUSTOM_IO; libav jo neuždaro): HLS — grojaraščio atminties kontekstas, ne HLS HTTP —
    // peek_io virš avio_open2 konteksto (A: avformat_close_input jį uždarytų avio_close, kuris peek_io nepalaiko)
    AVIOContext* top_pb_ = nullptr;
    void free_top_pb();
    bool guard_http_ = false;  // HTTP(S) šaltinis (H1: sargas visiems, ne tik HLS)
    bool hls_ = false;         // patikra rado M3U8
    std::uint64_t open_win0_ = 0;
    int opens_in_win_ = 0;
    std::uint64_t opens_total_ = 0;  // sesijos libav atidarymai (pirmas — viršutinis URL)
    static void ff_close(AVFormatContext* s, AVIOContext** pb);
    std::uint64_t refused_opens_ = 0;
    std::string shown_;  // šaltinio URL be paslapčių
    std::size_t init_bytes_ = 0;  // F: libav EXT-X-MAP init sekcijų sąmata šiai sesijai (init_section_bytes)
    // M2: libav segmentų sąmata kiekvienam sesijos grojaraščiui (URL, kurį libav atidaro → paskutinio analizavimo segment_bytes)
    std::vector<std::pair<std::string, std::size_t>> seg_bytes_;
    std::vector<std::string> playlist_urls_;  // E: URL, kurių M3U8 atsakymas leidžiamas (viršutinis, location, variantai)
};

// HTTP(S) šaltinio patikra prieš libav (A1, B8): ar turinys — M3U8 (hls), ir jei taip — master → media (ne giliau, be kilpų),
// visos nuorodos tik http / https (check_playlist), #EXT-X-TARGETDURATION ≤ lim.max_target_s (target_ns — didžiausias iš variantų; 0 — nėra).
// false — šaltinis netinkamas (err — aiški priežastis be paslapčių).
// top — viršutinis atsakymas (HLS atveju libav gauna būtent jį, ne antrą užklausą — H1 / H2).
bool hls_precheck(const std::string& url, AVIOInterruptCB* cb, const HlsLimits& lim, bool same_host, bool& hls, std::int64_t& target_ns,
                  FetchedPlaylist& top, std::string& err);

}  // namespace mxlv::ingest
