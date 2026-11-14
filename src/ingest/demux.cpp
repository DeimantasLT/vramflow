// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/demux.hpp"

#include "ingest/h264_scan.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mxlv::ingest {

bool is_hls_url(const std::string& url) {
    const auto q = url.find('?');
    const std::string p = url.substr(0, q);
    return p.size() >= 5 && p.compare(p.size() - 5, 5, ".m3u8") == 0;
}

std::int64_t hls_target_duration_ns(const std::string& url, AVIOInterruptCB* cb) {
    AVIOContext* io = nullptr;
    if (avio_open2(&io, url.c_str(), AVIO_FLAG_READ, cb, nullptr) < 0) return 0;
    std::string text;
    char buf[4096];
    for (int n; (n = avio_read(io, reinterpret_cast<unsigned char*>(buf), sizeof buf)) > 0 && text.size() < (1u << 20);)
        text.append(buf, static_cast<std::size_t>(n));
    avio_closep(&io);
    const auto t = text.find("#EXT-X-TARGETDURATION:");
    if (t != std::string::npos) return static_cast<std::int64_t>(std::atof(text.c_str() + t + 22) * 1e9);
    // master grojaraštis: pirmas variantas (santykinis URL — šalia master)
    const auto s = text.find("#EXT-X-STREAM-INF");
    if (s == std::string::npos) return 0;
    const auto l0 = text.find('\n', s);
    if (l0 == std::string::npos) return 0;
    std::string v = text.substr(l0 + 1, text.find('\n', l0 + 1) - l0 - 1);
    while (!v.empty() && (v.back() == '\r' || v.back() == ' ')) v.pop_back();
    if (v.empty()) return 0;
    if (v.find("://") == std::string::npos) v = url.substr(0, url.rfind('/') + 1) + v;
    return v == url ? 0 : hls_target_duration_ns(v, cb);
}

bool Demux::open(const SourceSpec& s, const std::atomic<bool>& stop, std::int64_t open_timeout_ns, std::string& err) {
    close();
    silence_ms_ = s.silence_ms;
    intr_.reset_session();
    intr_.stop = &stop;
    intr_.silence_ns = std::uint64_t{s.silence_ms} * 1'000'000;
    intr_.deadline_tai = open_timeout_ns > 0 ? tai_now() + static_cast<std::uint64_t>(open_timeout_ns) : 0;
    if (const std::string pe = protocol_error(s.url); !pe.empty()) {
        err = pe;
        return false;
    }
    std::string url = s.url;
    // UDP (kaip dec): didelis FIFO ir socket buferis, perpildymas ne fatalus; tylą ir sustabdymą sprendžia kabliukas
    if (url.rfind("udp://", 0) == 0 && url.find("fifo_size") == std::string::npos)
        url += std::string(url.find('?') == std::string::npos ? "?" : "&") + "fifo_size=131072&overrun_nonfatal=1&buffer_size=8388608";
    ic_ = avformat_alloc_context();
    attach_interrupt(ic_, &intr_);
    // Zondavimo metu perskaityti paketai NEgrąžinami vėliau: jie jau sekundės senumo, o atvykimo laikas žymimas skaitant —
    // SourceClock fiksavimas (atvykimas + latencija − PTS) būtų pasislinkęs tiek, kiek truko zondavimas
    ic_->flags |= AVFMT_FLAG_NOBUFFER;
    AVDictionary* o = nullptr;
    // Trumpas zondavimas: TS H.264 / AAC parametrai žinomi per < 0,5 s (numatytasis 5 MB būtų ~4 s ir pliūpsnis po to)
    av_dict_set(&o, "probesize", "2000000", 0);
    av_dict_set(&o, "analyzeduration", "1000000", 0);
    const bool hls = is_hls_url(s.url);
    if (hls) {
        av_dict_set(&o, "live_start_index", "-1", 0);  // nuo paskutinio segmento: pradžios pliūpsnis — vienas segmentas
        av_dict_set(&o, "http_persistent", "0", 0);
    }
    int e = avformat_open_input(&ic_, url.c_str(), nullptr, &o);  // klaidoje ic_ atlaisvinamas
    av_dict_free(&o);
    auto fail = [&](const std::string& what, int code) {
        const std::string why = intr_.timed_out ? "nėra duomenų per --input-wait-s" : (intr_.silent ? "įėjimas tylus" : averr(code));
        err = s.url + ": " + what + ": " + why;
        close();
        return false;
    };
    if (e < 0) {
        ic_ = nullptr;
        return fail("atidaryti nepavyko", e);
    }
    if ((e = avformat_find_stream_info(ic_, nullptr)) < 0) return fail("zondavimas nepavyko", e);
    vs_ = av_find_best_stream(ic_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    as_ = s.audio ? av_find_best_stream(ic_, AVMEDIA_TYPE_AUDIO, -1, vs_, nullptr, 0) : -1;
    if (vs_ < 0) vs_ = -1;
    if (as_ < 0) as_ = -1;
    if (vs_ < 0 && as_ < 0) {
        err = s.url + ": nėra nei video, nei garso srauto";
        close();
        return false;
    }
    info_ = StreamInfo{};
    info_.video = vs_ >= 0;
    info_.audio = as_ >= 0;
    if (info_.video) {
        info_.video_tb = ic_->streams[vs_]->time_base;
        info_.vpar = ic_->streams[vs_]->codecpar;
    }
    if (info_.audio) {
        info_.audio_tb = ic_->streams[as_]->time_base;
        info_.apar = ic_->streams[as_]->codecpar;
    }
    for (unsigned k = 0; k < ic_->nb_streams; ++k)
        if (static_cast<int>(k) != vs_ && static_cast<int>(k) != as_) ic_->streams[k]->discard = AVDISCARD_ALL;
    field_order_ = info_.video ? key_field_order() : AV_FIELD_UNKNOWN;
    hls_target_ns_ = hls ? hls_target_duration_ns(s.url, &ic_->interrupt_callback) : 0;
    // HLS: paketai ateina segmentais, tarp jų — tyla iki segmento trukmės ir grojaraščio atnaujinimo; tylos riba — ne mažiau
    // kaip 3 segmentai (kitaip kiekvienas tarpas tarp segmentų atrodytų kaip nutrūkęs šaltinis)
    if (hls_target_ns_ > 0)
        intr_.silence_ns = std::max<std::uint64_t>(intr_.silence_ns, 3 * static_cast<std::uint64_t>(hls_target_ns_));
    silence_ms_ = static_cast<std::uint32_t>(intr_.silence_ns / 1'000'000);
    intr_.deadline_tai = 0;  // atidaryta — toliau tylą sprendžia silence_ns
    return true;
}

// Skenavimas ir laukų tvarka iš RAKTO kadro (ingest/h264_scan.hpp): libav zondavimo field_order nustatomas pagal pirmą
// pasitaikiusį kadrą, todėl prisijungus srauto viduryje gali skirtis nuo tikrosios (ir HLS atveju jo gali nebūti). Skaitoma iki IDR
// (ne daugiau kaip 2000 paketų; jie metami — dekoderis vis tiek pradėtų nuo rakto kadro). HEVC — libav reikšmė (interlaced HEVC
// praktiškai nenaudojamas).
int Demux::key_field_order() {
    const AVCodecParameters* par = info_.vpar;
    if (par->codec_id != AV_CODEC_ID_H264) return par->field_order;
    H264Scan scan;
    if (par->extradata && par->extradata_size > 0) scan.feed(par->extradata, static_cast<std::size_t>(par->extradata_size));
    AVPacket* pkt = av_packet_alloc();
    int read = 0;
    for (int n = 0; n < 2000 && !scan.done(); ++n) {
        const int e = av_read_frame(ic_, pkt);
        ++read;
        if (e == AVERROR(EAGAIN)) continue;
        if (e < 0) break;
        intr_.packet();
        if (pkt->stream_index == vs_) scan.feed(pkt->data, static_cast<std::size_t>(pkt->size));
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    static const char* const kName[] = {"nežinoma", "progresyvus", "TFF", "BFF"};
    std::fprintf(stderr, "[ingest] skenavimas iš IDR: %s (pic_struct %d, paketų %d)\n", kName[static_cast<int>(scan.scan())], scan.pic_struct(), read);
    switch (scan.scan()) {
        case Scan::Progressive: return AV_FIELD_PROGRESSIVE;
        case Scan::Tff: return AV_FIELD_TT;
        case Scan::Bff: return AV_FIELD_BB;
        case Scan::Unknown: break;
    }
    // nežinoma (POC tipas 1 / 2, nėra IDR): interlaced — transliacijos numatytoji TFF; progresyvus — kaip praneša libav
    return par->field_order == AV_FIELD_PROGRESSIVE || par->field_order == AV_FIELD_UNKNOWN ? AV_FIELD_PROGRESSIVE : AV_FIELD_TT;
}

void Demux::close() {
    if (ic_) avformat_close_input(&ic_);
    ic_ = nullptr;
    vs_ = as_ = -1;
    info_ = StreamInfo{};
}

bool Demux::video_format(vramflow::VideoFormat& out, std::string& why) const {
    if (!info_.video) {
        why = "nėra video srauto";
        return false;
    }
    const AVCodecParameters* p = info_.vpar;
    const AVStream* st = ic_->streams[vs_];
    if (p->codec_id != AV_CODEC_ID_H264 && p->codec_id != AV_CODEC_ID_HEVC) {
        why = std::string("video kodekas ") + avcodec_get_name(p->codec_id) + " nepalaikomas (H.264, HEVC)";
        return false;
    }
    const auto pf = static_cast<AVPixelFormat>(p->format);
    if (pf != AV_PIX_FMT_YUV420P && pf != AV_PIX_FMT_YUVJ420P) {
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(pf);
        const char* name = d ? d->name : "nežinomas";
        std::string what = name;
        if (d && (d->log2_chroma_w != 1 || d->log2_chroma_h != 1)) what += d->log2_chroma_h == 0 && d->log2_chroma_w == 1 ? " (4:2:2)" : " (ne 4:2:0)";
        if (d && d->comp[0].depth != 8) what += " (" + std::to_string(d->comp[0].depth) + " bitų)";
        why = std::string(avcodec_get_name(p->codec_id)) + " pikselių formatas " + what + " nepalaikomas (tik 8 bitų 4:2:0)";
        return false;
    }
    vramflow::VideoFormat f{};
    f.raster.w = p->width;
    f.raster.h = p->height;
    f.raster.alpha = false;
    switch (field_order_) {
        case AV_FIELD_TT: f.raster.interlaced = true; break;
        case AV_FIELD_BB:
        case AV_FIELD_BT:
        case AV_FIELD_TB:
            why = std::to_string(p->width) + "x" + std::to_string(p->height) + " interlaced su apatiniu lauku pirmu (BFF) nepalaikomas (tik TFF)";
            return false;
        default: f.raster.interlaced = false; break;
    }
    if (p->chroma_location == AVCHROMA_LOC_TOPLEFT) f.siting = vramflow::ChromaSiting420::Type2TopLeft;
    else if (p->chroma_location == AVCHROMA_LOC_CENTER) f.siting = vramflow::ChromaSiting420::Type1Center;
    AVRational r = st->avg_frame_rate.num > 0 ? st->avg_frame_rate : st->r_frame_rate;
    if (r.num <= 0 || r.den <= 0) {
        why = "nežinomas kadrų dažnis";
        return false;
    }
    av_reduce(&r.num, &r.den, r.num, r.den, 1 << 30);
    // interlaced: libav kartais praneša laukų dažnį (50) — kadrų dažnis 25
    if (f.raster.interlaced && (r.num == 50 * r.den || r.num * 1001 == 60000 * r.den)) r.den *= 2;
    av_reduce(&r.num, &r.den, r.num, r.den, 1 << 30);
    f.frame_rate = {r.num, r.den};
    if (!vramflow::format_supported(f, why)) {
        why = vramflow::format_name(f) + ": " + why;
        return false;
    }
    out = f;
    return true;
}

int Demux::read(AVPacket* pkt, Kind& kind, std::string& why) {
    for (;;) {
        const int e = av_read_frame(ic_, pkt);
        if (e == AVERROR(EAGAIN)) continue;
        if (e < 0) {
            why = intr_.silent ? "įėjimas tylus > " + std::to_string(silence_ms_) + " ms" : "read_frame: " + averr(e);
            return e;
        }
        intr_.packet();
        if (pkt->stream_index == vs_) kind = Kind::Video;
        else if (pkt->stream_index == as_) kind = Kind::Audio;
        else {
            av_packet_unref(pkt);
            continue;
        }
        return 0;
    }
}

}  // namespace mxlv::ingest
