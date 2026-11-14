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
#include <vector>

namespace mxlv::ingest {

namespace {

// Grojaraščio tekstas per avio — TIK http / https (protocol_whitelist; įdėti tcp / tls — HTTP transportas), skaitymo riba
// rw_timeout 10 s (prisijungus iš naujo atidarymo ribos nėra — sustojęs serveris kitaip laikytų šaltinį amžinai).
// sniff — pirmiausia tik ≤ 4 KiB: ne M3U8 (pvz. tiesioginis MPEG-TS per HTTP) — toliau neskaitoma. false — klaida (err).
bool fetch_playlist(const std::string& url, AVIOInterruptCB* cb, std::size_t max_bytes, bool sniff, std::string& text, std::string& err) {
    AVDictionary* d = nullptr;
    av_dict_set(&d, "protocol_whitelist", "http,https,tcp,tls", 0);
    av_dict_set(&d, "rw_timeout", "10000000", 0);
    AVIOContext* io = nullptr;
    const int e = avio_open2(&io, url.c_str(), AVIO_FLAG_READ, cb, &d);
    av_dict_free(&d);
    if (e < 0) {
        err = "grojaraščio atidaryti nepavyko: " + averr(e) + ": " + redact_url(url);
        return false;
    }
    text.clear();
    char buf[4096];
    bool checked = !sniff;
    for (int n; (n = avio_read(io, reinterpret_cast<unsigned char*>(buf), sizeof buf)) > 0;) {
        text.append(buf, static_cast<std::size_t>(n));
        if (!checked) {
            checked = true;
            if (!parse_playlist(text.substr(0, 4096)).m3u) break;  // ne grojaraštis — tiek pakanka
        }
        if (text.size() > max_bytes) {
            avio_closep(&io);
            err = "grojaraštis didesnis nei " + std::to_string(max_bytes >> 10) + " KiB: " + redact_url(url);
            return false;
        }
    }
    avio_closep(&io);
    return true;
}

bool check_refs(const std::string& base, const std::vector<std::string>& refs, std::string& err) {
    std::string r;
    for (const std::string& x : refs)
        if (!resolve_playlist_url(base, x, r, err)) return false;
    return true;
}

}  // namespace

bool hls_precheck(const std::string& url, AVIOInterruptCB* cb, const HlsLimits& lim, bool& hls, std::int64_t& target_ns, std::string& err) {
    hls = false;
    target_ns = 0;
    std::string text;
    if (!fetch_playlist(url, cb, lim.max_bytes, true, text, err)) return false;
    const PlaylistInfo top = parse_playlist(text);
    if (!top.m3u) return true;  // ne HLS (turinys nuspręstas, ne URL galūnė — B8)
    hls = true;
    double target = top.target_s;
    if (top.master) {
        // Gylis: master → media, ne giliau. Variantas, kuris pats yra master (A→B→A, A→A) — klaida (kitaip libav HLS demuxeris
        // kilpoje kraunasi grojaraščius be galo, o ankstesnė rekursija pripildydavo steką ir nužudydavo visą procesą)
        if (top.variants.size() + 1 > lim.max_playlists) {
            err = "master grojaraštyje " + std::to_string(top.variants.size()) + " variantų > " + std::to_string(lim.max_playlists - 1);
            return false;
        }
        std::vector<std::string> seen{url};
        for (const std::string& v : top.variants) {
            std::string vu;
            if (!resolve_playlist_url(url, v, vu, err)) return false;
            if (vu == url) {
                err = "įdėtas master grojaraštis (variantas rodo į save): " + redact_url(vu);
                return false;
            }
            if (std::find(seen.begin(), seen.end(), vu) != seen.end()) continue;
            seen.push_back(vu);
            std::string vt;
            if (!fetch_playlist(vu, cb, lim.max_bytes, false, vt, err)) return false;
            const PlaylistInfo pv = parse_playlist(vt);
            if (!pv.m3u) {
                err = "variantas ne M3U8 grojaraštis: " + redact_url(vu);
                return false;
            }
            if (pv.master) {
                err = "įdėtas master grojaraštis (gylis > 1: master → media): " + redact_url(vu);
                return false;
            }
            if (!check_refs(vu, pv.refs, err)) return false;
            target = std::max(target, pv.target_s);
        }
    }
    if (!check_refs(url, top.refs, err)) return false;  // media segmentai arba master EXT-X-SESSION-KEY
    if (target > lim.max_target_s) {
        char b[160];
        std::snprintf(b, sizeof b, "#EXT-X-TARGETDURATION %.0f s > riba %.0f s (--hls-max-target-s)", target, lim.max_target_s);
        err = b;
        return false;
    }
    target_ns = target > 0 ? static_cast<std::int64_t>(target * 1e9) : 0;
    return true;
}

bool Demux::open(const SourceSpec& s, const std::atomic<bool>& stop, std::int64_t open_timeout_ns, std::string& err) {
    close();
    silence_ms_ = s.silence_ms;
    intr_.reset_session();
    intr_.stop = &stop;
    intr_.silence_ns = std::uint64_t{s.silence_ms} * 1'000'000;
    intr_.deadline_tai = open_timeout_ns > 0 ? tai_now() + static_cast<std::uint64_t>(open_timeout_ns) : 0;
    const std::string shown = redact_url(s.url);  // žurnalams ir klaidoms — be paslapčių (A1)
    if (!source_url_allowed(s.url, err)) return false;
    if (const std::string pe = protocol_error(s.url); !pe.empty()) {
        err = pe;
        return false;
    }
    const std::string scheme = url_scheme(s.url);
    // HLS atpažįstamas pagal turinį (#EXTM3U), ne pagal .m3u8 galūnę (B8); grojaraštis patikrinamas prieš libav (A1)
    bool hls = false;
    hls_target_ns_ = 0;
    if (scheme == "http" || scheme == "https") {
        AVIOInterruptCB cb{interrupt_cb, &intr_};
        HlsLimits lim;
        lim.max_target_s = s.hls_max_target_s;
        std::string why;
        if (!hls_precheck(s.url, &cb, lim, hls, hls_target_ns_, why)) {
            err = shown + ": HLS: " + (intr_.timed_out ? "nėra duomenų per --input-wait-s" : intr_.silent ? "įėjimas tylus" : why);
            return false;
        }
    }
    std::string url = s.url;
    // UDP (kaip dec): didelis FIFO ir socket buferis, perpildymas ne fatalus; tylą ir sustabdymą sprendžia kabliukas
    if (url.rfind("udp://", 0) == 0 && url.find("fifo_size") == std::string::npos)
        url += std::string(url.find('?') == std::string::npos ? "?" : "&") + "fifo_size=131072&overrun_nonfatal=1&buffer_size=8388608";
    ic_ = avformat_alloc_context();
    attach_interrupt(ic_, &intr_);
    shown_ = shown;
    guard_http_ = hls;
    opens_in_win_ = 0;
    open_win0_ = 0;
    opens_total_ = 0;
    orig_io_open_ = ic_->io_open;
    ic_->opaque = this;
    ic_->io_open = [](AVFormatContext* c, AVIOContext** pb, const char* u, int fl, AVDictionary** op) {
        return static_cast<Demux*>(c->opaque)->guarded_open(c, pb, u, fl, op);
    };
    // Zondavimo metu perskaityti paketai NEgrąžinami vėliau: jie jau sekundės senumo, o atvykimo laikas žymimas skaitant —
    // SourceClock fiksavimas (atvykimas + latencija − PTS) būtų pasislinkęs tiek, kiek truko zondavimas
    ic_->flags |= AVFMT_FLAG_NOBUFFER;
    AVDictionary* o = nullptr;
    // Trumpas zondavimas: TS H.264 / AAC parametrai žinomi per < 0,5 s (numatytasis 5 MB būtų ~4 s ir pliūpsnis po to)
    av_dict_set(&o, "probesize", "2000000", 0);
    av_dict_set(&o, "analyzeduration", "1000000", 0);
    // Baltieji sąrašai (A1): paveldi visi įdėti atidarymai (HLS segmentai ir jų demuxeriai, RTP → UDP, HTTP → TCP / TLS)
    av_dict_set(&o, "protocol_whitelist", protocol_whitelist_for(scheme).c_str(), 0);
    av_dict_set(&o, "format_whitelist", format_whitelist_for(scheme).c_str(), 0);
    if (hls) {
        av_dict_set(&o, "live_start_index", "-1", 0);  // nuo paskutinio segmento: pradžios pliūpsnis — vienas segmentas
        av_dict_set(&o, "http_persistent", "0", 0);
    }
    int e = avformat_open_input(&ic_, url.c_str(), nullptr, &o);  // klaidoje ic_ atlaisvinamas
    av_dict_free(&o);
    auto fail = [&](const std::string& what, int code) {
        const std::string why = intr_.timed_out ? "nėra duomenų per --input-wait-s" : (intr_.silent ? "įėjimas tylus" : averr(code));
        err = shown + ": " + what + ": " + why;
        close();
        return false;
    };
    if (e < 0) {
        ic_ = nullptr;
        return fail("atidaryti nepavyko", e);
    }
    if ((e = avformat_find_stream_info(ic_, nullptr)) < 0) return fail("zondavimas nepavyko", e);
    // Apkrautoje mašinoje (daug šaltinių, siuntėjų) per analyzeduration video parametrų kartais dar nėra („unspecified size“) —
    // zonduojama dar ne daugiau kaip 3 kartus, kol žinomas dydis ir pikselių formatas (išmatuota: 8 šaltiniai — 8-asis atmestas)
    for (int k = 0; k < 3; ++k) {
        bool unknown = false;
        for (unsigned i = 0; i < ic_->nb_streams; ++i) {
            const AVCodecParameters* p = ic_->streams[i]->codecpar;
            if (p->codec_type == AVMEDIA_TYPE_VIDEO && (p->width <= 0 || p->format < 0)) unknown = true;
        }
        if (!unknown) break;
        if ((e = avformat_find_stream_info(ic_, nullptr)) < 0) return fail("zondavimas nepavyko", e);
    }
    vs_ = av_find_best_stream(ic_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    as_ = s.audio ? av_find_best_stream(ic_, AVMEDIA_TYPE_AUDIO, -1, vs_, nullptr, 0) : -1;
    if (vs_ < 0) vs_ = -1;
    if (as_ < 0) as_ = -1;
    if (vs_ < 0 && as_ < 0) {
        err = shown + ": nėra nei video, nei garso srauto";
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

int Demux::guarded_open(AVFormatContext* s, AVIOContext** pb, const char* url, int flags, AVDictionary** opts) {
    const std::string u = url ? url : "";
    if (guard_http_) {
        std::string inner = u;
        if (url_scheme(inner) == "crypto+http" || url_scheme(inner) == "crypto+https") inner = inner.substr(7);  // AES-128 segmentas
        if (const std::string sc = url_scheme(inner); sc != "http" && sc != "https") {
            ++refused_opens_;
            std::fprintf(stderr, "[ingest] %s: HLS: atidarymas atmestas — schema „%s“ neleidžiama (tik http / https): %s\n", shown_.c_str(),
                         sc.empty() ? "vietinis kelias" : sc.c_str(), redact_url(u).c_str());
            return AVERROR(EPERM);
        }
        const std::uint64_t now = tai_now();
        if (now - open_win0_ >= 1'000'000'000ull) {
            open_win0_ = now;
            opens_in_win_ = 0;
        }
        if (++opens_in_win_ > kMaxOpensPerSec) {
            if (!intr_.abort.exchange(true))
                std::fprintf(stderr, "[ingest] %s: HLS: > %d grojaraščių / segmentų atidarymų per sekundę — kilpa (įdėtas master grojaraštis?), sesija "
                                     "nutraukiama\n", shown_.c_str(), kMaxOpensPerSec);
            return AVERROR_EXIT;
        }
    }
    int r = orig_io_open_(s, pb, url, flags, opts);
    if (r < 0 || !guard_http_) return r;
    // Turinio peržiūra (pirmi ≤ 4 KiB, grąžinama atgal buferio ribose): master grojaraštis leidžiamas tik pirmam sesijos atidarymui
    // (viršutinis URL). Perkraunamas media grojaraštis, tapęs master, libav HLS demuxeriui duoda „tuščius“ grojaraščius, ir jis
    // griūva (išmatuota: SIGSEGV, visas procesas) — todėl atmetama ANKSČIAU, nei libav jį perskaito, ir sesija nutraukiama.
    const bool first_open = opens_total_++ == 0;
    unsigned char head[4096];
    const int n = avio_read(*pb, head, sizeof head);
    if (n > 0 && !first_open) {
        const PlaylistInfo pi = parse_playlist(std::string(reinterpret_cast<const char*>(head), static_cast<std::size_t>(n)));
        if (pi.m3u && pi.master) {
            ff_close(s, pb);
            if (!intr_.abort.exchange(true))
                std::fprintf(stderr, "[ingest] %s: HLS: įdėtas master grojaraštis vidury sesijos (%s) — atmesta, sesija nutraukiama\n", shown_.c_str(),
                             redact_url(u).c_str());
            return AVERROR(EPERM);
        }
    }
    if (n >= 0 && avio_seek(*pb, 0, SEEK_SET) == 0) return 0;
    if (n < 0 && n != AVERROR_EOF) return 0;  // klaidą pamatys skaitytojas
    // peržiūra netilpo buferyje — atidaroma iš naujo
    ff_close(s, pb);
    return orig_io_open_(s, pb, url, flags, opts);
}

void Demux::ff_close(AVFormatContext* s, AVIOContext** pb) {
    if (s->io_close2) s->io_close2(s, *pb);
    else avio_close(*pb);
    *pb = nullptr;
}

int Demux::read(AVPacket* pkt, Kind& kind, std::string& why) {
    for (;;) {
        const int e = av_read_frame(ic_, pkt);
        if (e == AVERROR(EAGAIN)) continue;
        if (e < 0) {
            why = intr_.abort.load() ? "HLS grojaraščių kilpa — sesija nutraukta"
                  : intr_.silent     ? "įėjimas tylus > " + std::to_string(silence_ms_) + " ms"
                                     : "read_frame: " + averr(e);
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
