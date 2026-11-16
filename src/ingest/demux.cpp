// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/demux.hpp"

#include "ingest/h264_scan.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mxlv::ingest {

namespace {

// H2 / H3: libav gauna TIKSLIAI tuos grojaraščio baitus, kuriuos patikrino sargas (atminties AVIOContext, ne antra HTTP užklausa).
// AVIOContext::av_class — mūsų klasė be parinkčių, kurios vaikas (opaque) — MemIo su location / mime_type / cookies / rw_timeout:
// av_opt_get(..., AV_OPT_SEARCH_CHILDREN) juos randa kaip tikro HTTP konteksto (libav HLS: nukreipimo URL — santykinėms nuorodoms,
// mime — zondavimui, rw_timeout — įdėtiems atidarymams). Uždaro tik mūsų io_close2 kabliukas (avio_close tokio konteksto nepalaiko).
struct MemIo {
    const AVClass* cls;
    char* location;
    char* mime_type;
    char* cookies;
    std::int64_t rw_timeout;
    std::uint8_t* data;
    std::int64_t size, pos;
};

constexpr std::int64_t kRwTimeoutUs = 10'000'000;  // HTTP skaitymo riba (rw_timeout, µs): sustojęs serveris — klaida, ne kabojimas

const AVClass* mem_io_class() {
    static const AVOption opts[] = {
        {"location", "", offsetof(MemIo, location), AV_OPT_TYPE_STRING, {0}, 0, 0, AV_OPT_FLAG_DECODING_PARAM, nullptr},
        {"mime_type", "", offsetof(MemIo, mime_type), AV_OPT_TYPE_STRING, {0}, 0, 0, AV_OPT_FLAG_DECODING_PARAM, nullptr},
        {"cookies", "", offsetof(MemIo, cookies), AV_OPT_TYPE_STRING, {0}, 0, 0, AV_OPT_FLAG_DECODING_PARAM, nullptr},
        {"rw_timeout", "", offsetof(MemIo, rw_timeout), AV_OPT_TYPE_INT64, {kRwTimeoutUs}, -1, INT64_MAX, AV_OPT_FLAG_DECODING_PARAM, nullptr},
        {},
    };
    static const AVClass c = [] {
        AVClass k{};
        k.class_name = "vramflow-playlist";
        k.item_name = av_default_item_name;
        k.option = opts;
        k.version = LIBAVUTIL_VERSION_INT;
        return k;
    }();
    return &c;
}

const AVClass* mem_pb_class() {
    static const AVClass c = [] {
        AVClass k{};
        k.class_name = "vramflow-playlist-io";
        k.item_name = av_default_item_name;
        k.version = LIBAVUTIL_VERSION_INT;
        k.child_next = [](void* obj, void* prev) -> void* { return prev ? nullptr : static_cast<AVIOContext*>(obj)->opaque; };
        return k;
    }();
    return &c;
}

int mem_read(void* op, std::uint8_t* buf, int n) {
    auto* m = static_cast<MemIo*>(op);
    const std::int64_t left = m->size - m->pos;
    if (left <= 0) return AVERROR_EOF;
    const int k = static_cast<int>(std::min<std::int64_t>(n, left));
    std::memcpy(buf, m->data + m->pos, static_cast<std::size_t>(k));
    m->pos += k;
    return k;
}

std::int64_t mem_seek(void* op, std::int64_t off, int whence) {
    auto* m = static_cast<MemIo*>(op);
    if (whence & AVSEEK_SIZE) return m->size;
    whence &= ~AVSEEK_FORCE;
    const std::int64_t p = whence == SEEK_SET ? off : whence == SEEK_CUR ? m->pos + off : whence == SEEK_END ? m->size + off : -1;
    if (p < 0 || p > m->size) return AVERROR(EINVAL);
    m->pos = p;
    return p;
}

void free_mem(MemIo* m) {
    if (!m) return;
    av_free(m->location);
    av_free(m->mime_type);
    av_free(m->cookies);
    av_free(m->data);
    av_free(m);
}

}  // namespace

AVIOContext* mem_playlist_io(const FetchedPlaylist& f) {
    auto* m = static_cast<MemIo*>(av_mallocz(sizeof(MemIo)));
    auto* buf = static_cast<std::uint8_t*>(av_malloc(32768));
    if (!m || !buf) {
        av_free(m);
        av_free(buf);
        return nullptr;
    }
    m->cls = mem_io_class();
    m->location = av_strdup(f.location.c_str());  // tuščios eilutės, ne nullptr: libav kai kur skaito buf[0] be patikros
    m->mime_type = av_strdup(f.mime.c_str());
    m->cookies = av_strdup(f.cookies.c_str());
    m->rw_timeout = kRwTimeoutUs;
    m->size = static_cast<std::int64_t>(f.text.size());
    m->data = static_cast<std::uint8_t*>(av_malloc(f.text.size() + 1));
    if (!m->location || !m->mime_type || !m->cookies || !m->data) {
        free_mem(m);
        av_free(buf);
        return nullptr;
    }
    std::memcpy(m->data, f.text.data(), f.text.size());
    AVIOContext* io = avio_alloc_context(buf, 32768, 0, m, mem_read, nullptr, mem_seek);
    if (!io) {
        free_mem(m);
        av_free(buf);
        return nullptr;
    }
    io->av_class = mem_pb_class();
    return io;
}

bool is_mem_playlist_io(const AVIOContext* io) { return io != nullptr && io->read_packet == mem_read; }

void free_mem_playlist_io(AVIOContext** io) {
    if (!io || !*io) return;
    free_mem(static_cast<MemIo*>((*io)->opaque));
    av_freep(&(*io)->buffer);
    avio_context_free(io);
}

namespace {

// A (round 11): sargo peržiūra be grįžimo tinkle. avio_seek(…, 0) po 4 KiB skaitymo tinklo kontekste ne visada lieka buferyje:
// ieškomo HTTP (Accept-Ranges) AVIO buferis 32 KiB (aviobuf.c n6.1:985-989 dvigubina tik neieškomam), antras pildymas rašo nuo
// buferio pradžios (aviobuf.c:563-564), tada avio_seek kviečia http_seek (aviobuf.c:346-353) — NAUJA užklausa, kurios atsakymo sargas
// nemato. Todėl peržiūrėti baitai atiduodami iš atminties, o toliau skaitoma iš vidinio konteksto (avio_read_partial — kiek yra, ne
// laukiant pilno buferio). Atgal — tik peržiūros ribose, kol vidinis dar neskaitytas; kitaip ENOSYS (tinkle niekada).
struct PeekIo {
    AVIOContext* inner;
    std::uint8_t head[4096];
    int n;
    std::int64_t pos;  // loginė pozicija (0 — atsakymo pradžia)
};

int peek_read(void* op, std::uint8_t* buf, int size) {
    auto* p = static_cast<PeekIo*>(op);
    if (p->pos < p->n) {
        const int k = std::min(size, p->n - static_cast<int>(p->pos));
        std::memcpy(buf, p->head + p->pos, static_cast<std::size_t>(k));
        p->pos += k;
        return k;
    }
    const int r = avio_read_partial(p->inner, buf, size);
    if (r > 0) p->pos += r;
    return r == 0 ? AVERROR_EOF : r;
}

std::int64_t peek_seek(void* op, std::int64_t off, int whence) {
    auto* p = static_cast<PeekIo*>(op);
    // M1: dydis — TIK tiesiai (http_seek_internal AVSEEK_SIZE grąžina Content-Length be užklausos, http.c n6.1:1908-1909). Ne avio_size:
    // jo atsarginis kelias (dydis < 0, pvz. Content-Length ≥ 2^63) ieško SEEK_END ir atgal (aviobuf.c:383-389) — ieškomame http tai
    // naujos užklausos (http.c:1938-1960), kurių atsakymų sargas nemato. < 0 — ENOSYS, be jokio atsarginio kelio.
    if (whence & AVSEEK_SIZE) {
        const std::int64_t sz = p->inner->seek ? p->inner->seek(p->inner->opaque, 0, AVSEEK_SIZE) : -1;
        return sz < 0 ? AVERROR(ENOSYS) : sz;
    }
    whence &= ~AVSEEK_FORCE;
    const std::int64_t t = whence == SEEK_SET ? off : whence == SEEK_CUR ? p->pos + off : -1;
    if (t == p->pos) return t;
    if (t < 0 || p->pos > p->n || t > p->n) return AVERROR(ENOSYS);
    p->pos = t;
    return t;
}

const AVClass* peek_pb_class() {
    static const AVClass c = [] {
        AVClass k{};
        k.class_name = "vramflow-peek-io";
        k.item_name = av_default_item_name;
        k.version = LIBAVUTIL_VERSION_INT;
        // location / mime_type / cookies — iš vidinio konteksto (jo klasė ff_avio_class → URLContext → http)
        k.child_next = [](void* obj, void* prev) -> void* { return prev ? nullptr : static_cast<PeekIo*>(static_cast<AVIOContext*>(obj)->opaque)->inner; };
        return k;
    }();
    return &c;
}

}  // namespace

AVIOContext* peek_io(AVIOContext* inner, const std::uint8_t* head, int n) {
    auto* p = static_cast<PeekIo*>(av_mallocz(sizeof(PeekIo)));
    auto* buf = static_cast<std::uint8_t*>(av_malloc(32768));
    AVIOContext* io = p && buf ? avio_alloc_context(buf, 32768, 0, p, peek_read, nullptr, peek_seek) : nullptr;
    if (!io) {
        av_free(p);
        av_free(buf);
        return nullptr;
    }
    p->inner = inner;
    p->n = std::clamp(n, 0, static_cast<int>(sizeof p->head));
    std::memcpy(p->head, head, static_cast<std::size_t>(p->n));
    io->seekable = 0;
    io->av_class = peek_pb_class();
    return io;
}

bool is_peek_io(const AVIOContext* io) { return io != nullptr && io->read_packet == peek_read; }

AVIOContext* free_peek_io(AVIOContext** io) {
    if (!io || !*io) return nullptr;
    auto* p = static_cast<PeekIo*>((*io)->opaque);
    AVIOContext* inner = p->inner;
    av_free(p);
    av_freep(&(*io)->buffer);
    avio_context_free(io);
    return inner;
}

namespace {

// HTTP atsakymo savybė (location, mime_type, cookies) iš libav konteksto; "" — nėra
std::string io_prop(AVIOContext* io, const char* name) {
    std::uint8_t* v = nullptr;
    if (av_opt_get(io, name, AV_OPT_SEARCH_CHILDREN, &v) < 0 || v == nullptr) return {};
    std::string r(reinterpret_cast<const char*>(v));
    av_free(v);
    return r;
}

// Likęs grojaraščio tekstas iš io (jau perskaityta — text) iki max_bytes. false — per didelis (err) arba skaitymo klaida.
bool read_rest(AVIOContext* io, std::size_t max_bytes, const std::string& shown, std::string& text, std::string& err) {
    char buf[4096];
    for (int n; (n = avio_read(io, reinterpret_cast<unsigned char*>(buf), sizeof buf)) != 0;) {
        if (n < 0) {
            if (n == AVERROR_EOF) break;
            err = "grojaraščio skaityti nepavyko: " + averr(n) + ": " + shown;
            return false;
        }
        text.append(buf, static_cast<std::size_t>(n));
        if (text.size() > max_bytes) {
            err = "grojaraštis didesnis nei " + std::to_string(max_bytes >> 10) + " KiB (--hls-max-playlist): " + shown;
            return false;
        }
    }
    return true;
}

// Grojaraštis per avio — TIK http / https (protocol_whitelist; įdėti tcp / tls — HTTP transportas), skaitymo riba rw_timeout 10 s.
// sniff — pirmiausia tik ≤ 4 KiB: ne M3U8 (pvz. tiesioginis MPEG-TS per HTTP) — toliau neskaitoma. M3U8 — visas (≤ max_bytes).
bool fetch_playlist(const std::string& url, AVIOInterruptCB* cb, std::size_t max_bytes, bool sniff, FetchedPlaylist& f, std::string& err) {
    AVDictionary* d = nullptr;
    av_dict_set(&d, "protocol_whitelist", "http,https,tcp,tls", 0);
    av_dict_set(&d, "rw_timeout", std::to_string(kRwTimeoutUs).c_str(), 0);
    AVIOContext* io = nullptr;
    const int e = avio_open2(&io, url.c_str(), AVIO_FLAG_READ, cb, &d);
    av_dict_free(&d);
    if (e < 0) {
        err = "grojaraščio atidaryti nepavyko: " + averr(e) + ": " + redact_url(url);
        return false;
    }
    f = FetchedPlaylist{};
    char buf[4096];
    const int n = avio_read(io, reinterpret_cast<unsigned char*>(buf), sizeof buf);
    if (n > 0) f.text.assign(buf, static_cast<std::size_t>(n));
    bool ok = true;
    if (n > 0 && (!sniff || parse_playlist(f.text).m3u)) ok = read_rest(io, max_bytes, redact_url(url), f.text, err);
    f.location = io_prop(io, "location");
    if (f.location.empty()) f.location = url;
    f.mime = io_prop(io, "mime_type");
    f.cookies = io_prop(io, "cookies");
    avio_closep(&io);
    return ok;
}

}  // namespace

bool hls_precheck(const std::string& url, AVIOInterruptCB* cb, const HlsLimits& lim_in, bool same_host, bool& hls, std::int64_t& target_ns,
                  FetchedPlaylist& top, std::string& err) {
    hls = false;
    target_ns = 0;
    if (!fetch_playlist(url, cb, lim_in.max_bytes, true, top, err)) return false;
    // R13-1: galutinis URL po libav nukreipimų — tik tiksliai „http://“ / „https://“. libav http seka Location į bet kokią schemą ir vis
    // tiek jungiasi tcp į jos host:port (http.c n6.1:212-264, 434, 815-821), o tokio URL kanoninis autoritetas tuščias — anksčiau tai
    // išjungdavo --hls-same-host; santykinės nuorodos be to būtų sprendžiamos pagal ne http bazę
    if (!exact_http_url(top.location)) {
        err = "galutinis URL po nukreipimų ne http:// / https:// (libav http vis tiek jungtųsi į jo host:port): " + redact_url(top.location);
        return false;
    }
    const PlaylistInfo pt = parse_playlist(top.text);
    if (!pt.m3u) return true;  // ne HLS (turinys nuspręstas, ne URL galūnė — B8)
    hls = true;
    const std::string& base = top.location;  // nukreipus — santykinės nuorodos pagal galutinį URL (kaip libav)
    HlsLimits lim = lim_in;
    if (same_host) {
        lim.only_authority = canonical_authority(base);  // R13-1: nustatytas ir tuščias — niekas, ne „bet kuris“
        if (lim.only_authority->empty()) {
            err = "--hls-same-host: viršutinio grojaraščio serverio nustatyti nepavyko (host'as tuščias arba ilgesnis nei 1023 baitai): " +
                  redact_url(base);
            return false;
        }
    }
    // Gylis: master → media, ne giliau. Variantas, kuris pats yra master (A→B→A, A→A) — klaida (kitaip libav HLS demuxeris
    // kilpoje kraunasi grojaraščius be galo, o ankstesnė rekursija pripildydavo steką ir nužudydavo visą procesą)
    if (!check_playlist(base, pt, true, lim, err)) return false;
    double target = pt.target_s;
    std::size_t seg_bytes = segment_bytes(base, pt);  // M2: visų sesijos grojaraščių libav segmentų sąmata
    if (pt.master) {
        std::vector<std::string> seen{url, base};
        for (const std::string& v : pt.variants) {
            std::string vu;
            if (!resolve_playlist_url(base, v, vu, err)) return false;
            if (vu == url || vu == base) {
                err = "įdėtas master grojaraštis (variantas rodo į save): " + redact_url(vu);
                return false;
            }
            if (std::find(seen.begin(), seen.end(), vu) != seen.end()) continue;
            seen.push_back(vu);
            FetchedPlaylist fv;
            if (!fetch_playlist(vu, cb, lim.max_bytes, false, fv, err)) return false;
            // D: varianto atsakymas po nukreipimų — irgi tik iš leistino serverio (nukreipimo užklausos jau padarytos — README)
            if (!same_host_allows(lim, fv.location)) {
                err = "variantas nukreiptas į kitą serverį (--hls-same-host: tik " + same_host_name(lim) + "): " + redact_url(fv.location);
                return false;
            }
            const PlaylistInfo pv = parse_playlist(fv.text);
            if (!pv.m3u) {
                err = "variantas ne M3U8 grojaraštis: " + redact_url(vu);
                return false;
            }
            if (!check_playlist(fv.location, pv, false, lim, err)) return false;
            if ((seg_bytes += segment_bytes(fv.location, pv)) > lim.max_session_bytes) {
                err = "libav segmentų sąmata visiems grojaraščiams " + std::to_string(seg_bytes >> 10) + " KiB > riba " +
                      std::to_string(lim.max_session_bytes >> 10) + " KiB";
                return false;
            }
            target = std::max(target, pv.target_s);
        }
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
    entry_ = Entry::None;
    entry_discarded_ = 0;
    lim_ = HlsLimits{};
    lim_.max_target_s = s.hls_max_target_s;
    lim_.max_bytes = s.hls_max_playlist;
    top_ = FetchedPlaylist{};
    const bool http = scheme == "http" || scheme == "https";
    if (http) {
        AVIOInterruptCB cb{interrupt_cb, &intr_};
        std::string why;
        bool ok = false;
        {
            LibavCall g(intr_, "avio_open2 / avio_read (HTTP patikra)");  // R13-2: sargui — kiekvienas libav I/O
            ok = hls_precheck(s.url, &cb, lim_, s.hls_same_host, hls, hls_target_ns_, top_, why);
        }
        if (!ok) {
            err = shown + ": HLS: " + (intr_.timed_out ? "nėra duomenų per --input-wait-s" : intr_.silent ? "įėjimas tylus" : why);
            return false;
        }
    }
    std::string url = s.url;
    // UDP (kaip dec): didelis FIFO ir socket buferis, perpildymas ne fatalus; tylą ir sustabdymą sprendžia kabliukas
    if (url.rfind("udp://", 0) == 0 && url.find("fifo_size") == std::string::npos)
        url += std::string(url.find('?') == std::string::npos ? "?" : "&") + "fifo_size=131072&overrun_nonfatal=1&buffer_size=8388608";
    shown_ = shown;
    // H1: sargas — kiekvienam HTTP(S) šaltiniui (ne tik HLS): patikra ir libav atidarymas — dvi atskiros užklausos, serveris gali
    // atsakyti skirtingai (pirma TS, antra M3U8)
    guard_http_ = http;
    hls_ = hls;
    if (hls && s.hls_same_host) lim_.only_authority = canonical_authority(top_.location);  // sargui — tas pats serveris kaip patikroje
    // E: URL, kuriuose libav gali gauti M3U8: viršutinis ir jo location (media grojaraščio perkrovimai — location, hls.c n6.1:782-783,
    // 613-621), master variantai ir EXT-X-MEDIA grojaraščiai, išspręsti kaip libav (new_playlist → ff_make_absolute_url, hls.c:324)
    playlist_urls_.clear();
    init_bytes_ = 0;
    seg_bytes_.clear();
    if (hls) {
        // L1: viršutinio media grojaraščio perkrovimus libav atidaro normalizuotu URL (new_playlist → ff_make_absolute_url(NULL, location),
        // hls.c n6.1:324, 613-621) — jis irgi žinomas grojaraštis; jo segmentų sąmata — tuo pačiu raktu, kurį sargas mato perkraunant
        const std::string top_url = libav_playlist_url(top_.location);
        playlist_urls_ = {s.url, top_.location, top_url};
        init_bytes_ = init_section_bytes(top_.location, parse_playlist(top_.text));  // F: viršutinis — libav jį irgi analizuoja
        seg_bytes_.emplace_back(top_url, segment_bytes(top_.location, parse_playlist(top_.text)));  // M2
        std::string r, why;
        for (const std::string& v : parse_playlist(top_.text).variants)
            if (resolve_playlist_url(top_.location, v, r, why)) playlist_urls_.push_back(r);
    }
    auto fail = [&](const std::string& what, int code) {
        const std::string why = intr_.timed_out ? "nėra duomenų per --input-wait-s" : (intr_.silent ? "įėjimas tylus" : averr(code));
        err = shown + ": " + what + ": " + why;
        close();
        return false;
    };
    // Apkrautoje mašinoje (daug šaltinių, siuntėjų) ar sraute, kurio SPS retas (intra atnaujinimas, prisijungimas vidury), per
    // analyzeduration video parametrų kartais dar nėra („unspecified size“) — zonduojama iš naujo ne daugiau kaip 3 kartus, kaskart
    // NAUJAI atidarant įėjimą su 2× ilgesniu langu (išmatuota: 8 šaltiniai — 8-asis atmestas). Anksčiau avformat_find_stream_info
    // buvo kviečiamas pakartotinai tame pačiame kontekste — libavformat 60 su AVFMT_FLAG_NOBUFFER tada griūdavo (SIGSEGV, visas procesas).
    for (int k = 0; k < 4; ++k) {
        ic_ = avformat_alloc_context();
        attach_interrupt(ic_, &intr_);
        opens_in_win_ = 0;
        open_win0_ = 0;
        opens_total_ = 0;
        orig_io_open_ = ic_->io_open;
        orig_io_close2_ = ic_->io_close2;
        ic_->opaque = this;
        ic_->io_open = [](AVFormatContext* c, AVIOContext** pb, const char* u, int fl, AVDictionary** op) {
            return static_cast<Demux*>(c->opaque)->guarded_open(c, pb, u, fl, op);
        };
        ic_->io_close2 = [](AVFormatContext* c, AVIOContext* pb) { return static_cast<Demux*>(c->opaque)->guarded_close(c, pb); };
        // H2: HLS viršutinis grojaraštis — TIE PATYS patikrinti baitai (atminties kontekstas, AVFMT_FLAG_CUSTOM_IO), ne antra HTTP
        // užklausa, kurios atsakymas galėtų skirtis (H1)
        if (hls) {
            free_top_pb();
            top_pb_ = mem_playlist_io(top_);
            if (!top_pb_) {
                avformat_free_context(ic_);
                ic_ = nullptr;
                return fail("atidaryti nepavyko", AVERROR(ENOMEM));
            }
            ic_->pb = top_pb_;
        } else if (http) {
            // A: tiesioginis MPEG-TS per HTTP — viršutinis kontekstas mūsų (peek_io): libav jo neatidaro ir neuždaro (avio_close
            // peek_io nepalaiko), sargo peržiūra be antros užklausos
            if (const int e = open_top(url); e < 0) {
                avformat_free_context(ic_);
                ic_ = nullptr;
                return fail("atidaryti nepavyko", e);
            }
        }
        // Zondavimo metu perskaityti paketai NEgrąžinami vėliau: jie jau sekundės senumo, o atvykimo laikas žymimas skaitant —
        // SourceClock fiksavimas (atvykimas + latencija − PTS) būtų pasislinkęs tiek, kiek truko zondavimas
        ic_->flags |= AVFMT_FLAG_NOBUFFER;
        AVDictionary* o = nullptr;
        // Trumpas zondavimas: TS H.264 / AAC parametrai žinomi per < 0,5 s (numatytasis 5 MB būtų ~4 s ir pliūpsnis po to)
        av_dict_set(&o, "probesize", std::to_string(2'000'000 << k).c_str(), 0);
        av_dict_set(&o, "analyzeduration", std::to_string(1'000'000 << k).c_str(), 0);
        // Baltieji sąrašai (A1): paveldi visi įdėti atidarymai (HLS segmentai ir jų demuxeriai, RTP → UDP, HTTP → TCP / TLS)
        av_dict_set(&o, "protocol_whitelist", protocol_whitelist_for(scheme, hls).c_str(), 0);
        av_dict_set(&o, "format_whitelist", format_whitelist_for(scheme, hls).c_str(), 0);
        if (hls) {
            av_dict_set(&o, "live_start_index", "-1", 0);  // nuo paskutinio segmento: pradžios pliūpsnis — vienas segmentas
            av_dict_set(&o, "http_persistent", "0", 0);
            // E: segmentų demuxeris be hls — įdėtas kontekstas gauna šį format_whitelist (avformat_open_input taiko AVFormatContext
            // parinktis iš žodyno po ff_copy_whiteblacklists, demux.c n6.1:251, hls.c:2141-2146; DICT tipas „k=v:k=v“, opt.c:459-464)
            av_dict_set(&o, "seg_format_options", ("format_whitelist=" + segment_format_whitelist()).c_str(), 0);
        }
        // HTTP(S): sustojęs serveris (prisijungus, bet neatsakant) — skaitymo klaida po 10 s, ne amžinas laukimas (ir prisijungiant iš naujo)
        if (http) av_dict_set(&o, "rw_timeout", std::to_string(kRwTimeoutUs).c_str(), 0);
        int e = 0;
        {
            LibavCall g(intr_, "avformat_open_input");
            e = avformat_open_input(&ic_, url.c_str(), nullptr, &o);  // klaidoje ic_ atlaisvinamas
        }
        av_dict_free(&o);
        if (e < 0) {
            ic_ = nullptr;
            return fail("atidaryti nepavyko", e);
        }
        {
            LibavCall g(intr_, "avformat_find_stream_info");
            e = avformat_find_stream_info(ic_, nullptr);
        }
        if (e < 0) return fail("zondavimas nepavyko", e);
        bool unknown = false;
        for (unsigned i = 0; i < ic_->nb_streams; ++i) {
            const AVCodecParameters* p = ic_->streams[i]->codecpar;
            if (p->codec_type == AVMEDIA_TYPE_VIDEO && (p->width <= 0 || p->format < 0)) unknown = true;
        }
        if (!unknown || k == 3) break;
        std::fprintf(stderr, "[ingest] %s: video parametrų dar nėra (%d) — zonduojama iš naujo, langas %d ms\n", shown.c_str(), k + 1, 1000 << (k + 1));
        LibavCall g(intr_, "avformat_close_input");
        avformat_close_input(&ic_);
        free_top_pb();
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
        int e = 0;
        {
            LibavCall g(intr_, "av_read_frame (įėjimo taško paieška)");
            e = av_read_frame(ic_, pkt);
        }
        ++read;
        if (e == AVERROR(EAGAIN)) continue;
        if (e < 0) break;
        intr_.packet();
        if (pkt->stream_index == vs_) scan.feed(pkt->data, static_cast<std::size_t>(pkt->size));
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    static const char* const kName[] = {"nežinoma", "progresyvus", "TFF", "BFF"};
    static const char* const kEntry[] = {"nėra", "IDR", "recovery point SEI", "I pjūvis"};
    entry_ = scan.entry();
    entry_discarded_ = static_cast<std::uint64_t>(read);
    // B7: įėjimo taškas — IDR, recovery point arba I pjūvis (atviras GOP, intra atnaujinimas); išmesti paketai skaičiuojami ir rodomi
    std::fprintf(stderr, "[ingest] %s: skenavimas iš įėjimo taško (%s): %s (pic_struct %d), išmesta paketų %d%s\n", shown_.c_str(),
                 kEntry[static_cast<int>(entry_)], kName[static_cast<int>(scan.scan())], scan.pic_struct(), read,
                 scan.done() ? "" : " — NĖRA dekoduojamo įėjimo taško");
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
    LibavCall g(intr_, "avformat_close_input");
    if (ic_) avformat_close_input(&ic_);
    ic_ = nullptr;
    free_top_pb();
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
    if (guard_http_)
        if (const int e = pre_open(u); e < 0) return e;
    const int r = orig_io_open_(s, pb, url, flags, opts);
    if (r < 0 || !guard_http_) return r;
    return post_open(s, pb, u);
}

int Demux::pre_open(const std::string& u) {
    std::string inner = u;
    if (url_scheme(inner) == "crypto+http" || url_scheme(inner) == "crypto+https") inner = inner.substr(7);  // AES-128 segmentas
    if (const std::string sc = url_scheme(inner); sc != "http" && sc != "https") {
        ++refused_opens_;
        std::fprintf(stderr, "[ingest] %s: HLS: atidarymas atmestas — schema „%s“ neleidžiama (tik http / https): %s\n", shown_.c_str(),
                     sc.empty() ? "vietinis kelias" : sc.c_str(), redact_url(u).c_str());
        return AVERROR(EPERM);
    }
    // H1: „http:host/…“ (be „//“) — libav host'ą ima iš kelio (av_url_split, utils.c n6.1:377-383); niekada neatidaroma
    if (http_scheme_without_slashes(u)) {
        ++refused_opens_;
        std::fprintf(stderr, "[ingest] %s: HLS: atidarymas atmestas — URL be „//“ po schemos (av_url_split host'ą imtų iš kelio): %s\n",
                     shown_.c_str(), redact_url(u).c_str());
        return AVERROR(EPERM);
    }
    // D: --hls-same-host — ir pats atidaromas URL (ne tik grojaraščio tekstas), prieš libav atidarymą
    if (!same_host_allows(lim_, u)) {
        ++refused_opens_;
        if (!intr_.abort.exchange(true))
            std::fprintf(stderr, "[ingest] %s: HLS: atidarymas į kitą serverį (--hls-same-host: tik %s): %s — atmesta, sesija nutraukiama\n",
                         shown_.c_str(), same_host_name(lim_).c_str(), redact_url(u).c_str());
        return AVERROR(EPERM);
    }
    if (!hls_ && opens_total_ > 0) {  // H1: tiesioginis MPEG-TS per HTTP — vienas atidarymas sesijai; daugiau — ne mūsų srautas
        ++refused_opens_;
        std::fprintf(stderr, "[ingest] %s: HTTP (ne HLS): papildomas libav atidarymas atmestas: %s\n", shown_.c_str(), redact_url(u).c_str());
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
    return 0;
}

int Demux::open_top(const std::string& url) {
    if (const int e = pre_open(url); e < 0) return e;
    LibavCall g(intr_, "avio_open2 (HTTP)");  // R13-2: ir post_open peržiūros skaitymas
    AVDictionary* d = nullptr;
    av_dict_set(&d, "protocol_whitelist", protocol_whitelist_for(url_scheme(url), false).c_str(), 0);
    av_dict_set(&d, "rw_timeout", std::to_string(kRwTimeoutUs).c_str(), 0);
    AVIOContext* pb = nullptr;
    const int e = avio_open2(&pb, url.c_str(), AVIO_FLAG_READ, &ic_->interrupt_callback, &d);
    av_dict_free(&d);
    if (e < 0) return e;
    // ff_close (refuse) uždaro per ic_->io_close2 → orig (io_close2_default = avio_close) — tas pats, kas avio_open2 kontekstui
    if (const int r = post_open(ic_, &pb, url); r < 0) return r;
    free_top_pb();
    top_pb_ = pb;
    ic_->pb = pb;
    return 0;
}

int Demux::post_open(AVFormatContext* s, AVIOContext** pb, const std::string& u) {
    ++opens_total_;
    // Sesija nutraukiama su priežastimi žurnale (vieną kartą); šaltinis jungiasi iš naujo
    auto refuse = [&](const std::string& why) {
        if (*pb) ff_close(s, pb);
        if (!intr_.abort.exchange(true)) std::fprintf(stderr, "[ingest] %s: HLS: %s — atmesta, sesija nutraukiama\n", shown_.c_str(), why.c_str());
        return AVERROR(EPERM);
    };
    // D: --hls-same-host — galutinis atsakymo URL po libav http nukreipimų (iki 8, http.c n6.1:54, 419-438; max_redirects parinkties
    // 6.1 neturi), PRIEŠ skaitant turinį. Nukreipimų užklausos jau padarytos (README). crypto+ konteksto location nepasiekiamas
    // (crypto klasė be child_next) — tada patikrinta tik prieš atidarymą.
    if (lim_.only_authority)
        if (const std::string loc = io_prop(*pb, "location"); !loc.empty() && !same_host_allows(lim_, loc))
            return refuse("atsakymas nukreiptas į kitą serverį (--hls-same-host: tik " + same_host_name(lim_) + "): " + redact_url(u) + " → " +
                          redact_url(loc));
    // Turinio peržiūra (pirmi ≤ 4 KiB): M3U8 atpažįstamas iš pradžios (libav reikalauja #EXTM3U pirmoje eilutėje); ne M3U8 (segmentai,
    // raktai, tiesioginis TS) — peržiūrėti baitai atiduodami iš atminties (peek_io), tinklo kontekstas atgal negrąžinamas (A)
    unsigned char head[4096];
    const int n = avio_read(*pb, head, sizeof head);
    const bool m3u = n > 0 && parse_playlist(std::string(reinterpret_cast<const char*>(head), static_cast<std::size_t>(n))).m3u;
    if (!m3u) {
        // klaida / EOF be duomenų — tuščia peržiūra; vidinio konteksto klaida lieka (avio_read_partial ją grąžins skaitytojui)
        AVIOContext* w = peek_io(*pb, head, n < 0 ? 0 : n);
        if (!w) return refuse("atminties nepakanka");
        *pb = w;
        return 0;
    }
    // H1: patikra matė ne M3U8, o libav gavo M3U8 — atmetama (format_whitelist vis tiek tik mpegts)
    if (!hls_) return refuse("HTTP atsakymas — M3U8 grojaraštis, nors patikra rado ne HLS");
    // E: M3U8 priimamas tik iš žinomų grojaraščių URL; segmentas, raktas ar init sekcija, atsakę M3U8, — atmetami (ir libav segmentų
    // demuxeriui hls neleidžiamas — seg_format_options)
    if (std::find(playlist_urls_.begin(), playlist_urls_.end(), u) == playlist_urls_.end())
        return refuse("M3U8 atsakymas ne grojaraščio URL (segmentas, raktas ar init sekcija): " + redact_url(u));
    // H2 / H3: M3U8 perskaitomas VISAS (≤ --hls-max-playlist), patikrinamas visas (master tik viršutinis — perkraunamas media
    // grojaraštis, tapęs master, libav HLS demuxeriui duoda „tuščius“ grojaraščius ir jis griūva; ribos; nuorodų schemos), o libav
    // gauna TUOS PAČIUS baitus per atminties kontekstą — jis niekada neanalizuoja to, ko sargas nematė.
    FetchedPlaylist f;
    f.text.assign(reinterpret_cast<const char*>(head), static_cast<std::size_t>(n));
    std::string why;
    if (!read_rest(*pb, lim_.max_bytes, redact_url(u), f.text, why)) return refuse(why);
    f.location = io_prop(*pb, "location");
    if (f.location.empty()) f.location = u;
    f.mime = io_prop(*pb, "mime_type");
    f.cookies = io_prop(*pb, "cookies");
    ff_close(s, pb);
    const PlaylistInfo pi = parse_playlist(f.text);
    if (!check_playlist(f.location, pi, false, lim_, why)) return refuse(why);
    // F: libav kiekvieno perkrovimo EXT-X-MAP init sekcijas kaupia iki sesijos pabaigos (hls.c n6.1:456, 251-269) — sesijos sąmata
    if ((init_bytes_ += init_section_bytes(f.location, pi)) > lim_.max_init_bytes)
        return refuse("EXT-X-MAP init sekcijų sąmata per sesiją " + std::to_string(init_bytes_ >> 10) + " KiB > riba " +
                      std::to_string(lim_.max_init_bytes >> 10) + " KiB (libav jų neatlaisvina iki sesijos pabaigos)");
    // M2: libav segmentų sąrašas — kiekvieno grojaraščio paskutinio analizavimo (ankstesnis atlaisvinamas, hls.c n6.1:1010-1027);
    // visų sesijos grojaraščių suma ≤ max_session_bytes
    const std::size_t sb = segment_bytes(f.location, pi);
    auto it = std::find_if(seg_bytes_.begin(), seg_bytes_.end(), [&](const auto& e) { return e.first == u; });
    if (it == seg_bytes_.end()) seg_bytes_.emplace_back(u, sb);
    else it->second = sb;
    std::size_t total = 0;
    for (const auto& e : seg_bytes_) total += e.second;
    if (total > lim_.max_session_bytes)
        return refuse("libav segmentų sąmata visiems grojaraščiams " + std::to_string(total >> 10) + " KiB > riba " +
                      std::to_string(lim_.max_session_bytes >> 10) + " KiB");
    *pb = mem_playlist_io(f);
    return *pb ? 0 : AVERROR(ENOMEM);
}

int Demux::guarded_close(AVFormatContext* s, AVIOContext* pb) {
    if (pb != nullptr && pb == top_pb_) return 0;  // viršutinį atlaisvina Demux (AVFMT_FLAG_CUSTOM_IO)
    if (is_mem_playlist_io(pb)) {
        free_mem_playlist_io(&pb);
        return 0;
    }
    if (is_peek_io(pb)) pb = free_peek_io(&pb);  // vidinis — libav atidarytas, uždaromas jo būdu
    if (orig_io_close2_) return orig_io_close2_(s, pb);
    avio_close(pb);
    return 0;
}

void Demux::free_top_pb() {
    if (is_peek_io(top_pb_)) {
        AVIOContext* inner = free_peek_io(&top_pb_);
        avio_close(inner);  // open_top: avio_open2
    }
    free_mem_playlist_io(&top_pb_);
}

void Demux::ff_close(AVFormatContext* s, AVIOContext** pb) {
    if (s->io_close2) s->io_close2(s, *pb);
    else avio_close(*pb);
    *pb = nullptr;
}

int Demux::read(AVPacket* pkt, Kind& kind, std::string& why) {
    for (;;) {
        int e = 0;
        {
            LibavCall g(intr_, "av_read_frame");
            e = av_read_frame(ic_, pkt);
        }
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
