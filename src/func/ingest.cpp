// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// ingest — tikras šaltinis su garsu (planas docs/planai/2026-11-11-production-grade.md §P3, user story P3): MPEG-TS per UDP /
// RTP / SRT arba HLS (libavformat, K10) → VIENAS demuxeris šaltiniui → video (NVDEC arba programinis) ir garso (libswresample)
// dekodavimas → video (v210, cuda-linear arba host) ir garso (48 kHz float32) flow'ai, pririšti prie šaltinio PTS.
//
// Laiko modelis (brief'o pamokos 1–4, K6, K7):
//   * kiekvienas paketas žymimas demux metu: TAI = PTS + šaltinio poslinkis (timing::SourceClock, bendras visiems šaltinio
//     flow'ams); pirmas video paketas fiksuoja poslinkį = atvykimas + latencija (--latency-ms; HLS — ≥ 3 segmentai) − PTS;
//   * video kadras tampa grain'u, kurio indeksas = index_of(PTS_TAI) — ne pagal atvykimo tvarką; tick'as indeksui i ima
//     tiksliai tą kadrą (nėra — spraga pagal --on-loss; vienas kadras pakartojamas, jei kitas jau laukia — frame sync);
//   * paketas į dekoderį leidžiamas ne anksčiau kaip --decode-lead-ms prieš jo DTS_TAI (ingest/feed_gate.hpp); kadras, kuris
//     nebespės, nededamas — praleidžiama iki kito rakto kadro (perkrova nesugriūna);
//   * garsas sekamas mėginio tikslumu: tolydus garso poslinkis + swr_set_compensation (≤ --max-comp-ppm, K6 ±0,1 %), video
//     slysta sveiku kadru (interlaced — lauko pora), todėl A/V ≤ 1 laukas; dreifas matomas statistikoje (drift_ppm, drift_slips).
// Gijos šaltiniui: demux (skaito, žymi, eilės), video dekodavimas (atsarga, kadrų žemėlapis), garso dekodavimas (žiedas). MXL
// rašo tik tick gija (bendras engine::TickEngine, šaltiniai — vienetai kaip dec; 25p šaltinis 50 Hz variklyje — kas antrą tick'ą).
// Formatas pasikeitė vidury srauto (K17) — tas šaltinis sustabdomas su aiškia klaida, kiti tęsia, exit 6.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/log.h>
}
#include <cstdarg>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "func/engine_glue.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "func/stats_lines.hpp"
#include "ingest/audio_decode.hpp"
#include "ingest/av_input.hpp"
#include "ingest/demux.hpp"
#include "ingest/feed_gate.hpp"
#include "ingest/packet_queue.hpp"
#include "ingest/url_policy.hpp"
#include "ingest/video_decode.hpp"
#include "media/cpu/cpu_media.hpp"
#include "media/media_kernels.hpp"
#include "timing/index.hpp"
#include "timing/source_clock.hpp"

namespace mxlv::func {
namespace {

namespace ig = mxlv::ingest;

// Eilės elementas: paketas (TAI žymos) arba naujos sesijos žymė su tos sesijos kodeko parametrais.
struct QItem {
    ig::Packet p;
    AVCodecParameters* par = nullptr;  // != nullptr — nauja sesija (dekoderis atidaromas iš naujo); savininkas — eilė / gija
    bool no_stream = false;            // nauja sesija be šio srauto (A2: garso nebėra po prisijungimo iš naujo)
    bool flush = false;                // B4: perfiksavimas — dekoderis išvalomas, laukę kadrai išmetami, pradedama nuo įėjimo taško
    std::uint64_t cut_frame = 0;       // M2 (flush): išmetami kadrai, kurių indeksas ≥ šio (tas pats pjūvio momentas kaip garso)
    bool resync = false;               // M2: perfiksavimas — garsas pririšamas prie naujos ašies (AudioSync::realign)
    std::int64_t cut_sample = INT64_MIN;  // M2 (resync): garso žiedas nupjaunamas čia (INT64_MIN — nepjauti)
};

// Eilės elementų savybės (ingest/packet_queue.hpp): baitai — paketo dydis (žymės — 0), atlaisvinimas.
struct QItemTraits {
    static std::size_t bytes(const QItem& it) { return it.p.pkt ? static_cast<std::size_t>(std::max(it.p.pkt->size, 1)) : 0; }
    static void drop(QItem& it) {
        if (it.p.pkt) av_packet_free(&it.p.pkt);
        if (it.par) avcodec_parameters_free(&it.par);
    }
};

// M1: riba elementais (apsauga nuo milijonų mažų paketų) ir baitais (--max-queue-mb; numatyta — latencija × dosni sparta)
class PacketQueue : public ig::PacketQueue<QItem, QItemTraits> {
public:
    using ig::PacketQueue<QItem, QItemTraits>::PacketQueue;
    static void drop(QItem& it) { QItemTraits::drop(it); }
};

using Counter = std::atomic<std::uint64_t>;
constexpr double kQueueMbitS = 80;  // M1: numatytosios eilės ribos sparta (Mbit/s) — dosniai virš transliacijos įnašo srautų

struct Source {
    int idx = 0;
    std::string name;
    ig::SourceSpec spec;
    ig::Demux demux;
    bool video = false, audio = false;
    vramflow::VideoFormat fmt{};
    std::int64_t latency_ns = 0;
    std::unique_ptr<timing::SourceClock> clock;
    std::atomic<std::int64_t> pending_step{0};  // tick'o pastebėtas laikrodžio šuolis → demux gijai (SourceClock — viena gija)
    AVCodecParameters* vpar0 = nullptr;         // pirmos sesijos parametrai (perduodami dekoderio gijoms)
    AVCodecParameters* apar0 = nullptr;
    PacketQueue vq{40000, std::uint64_t{64} << 20}, aq{40000, std::uint64_t{64} << 20};  // baitų riba — nustatoma zondavus (M1)
    std::uint64_t queue_cap_bytes = 0;
    std::uint64_t overflow_logged = 0;  // demux gija: paskutinis žurnale parodytas perpildymo episodas
    // Dekoduoti kadrai pagal kadro indeksą (flow kadrų dažniu); tick'as ima tiksliai savo
    std::mutex fm;
    std::map<std::uint64_t, AVFrame*> frames;
    std::atomic<std::uint64_t> taken_upto{0};  // didžiausias kadro indeksas, kurį tick'as jau bandė paimti (+1; 0 — dar nė vieno)
    ig::AudioRing ring;
    std::atomic<bool> fmt_stopped{false}, fmt_reported{false};
    std::mutex why_m;
    std::string fmt_why;
    std::thread th_demux, th_video, th_audio;
    std::uint64_t reconnects = 0;
    std::string error;
    // statistika (gijos)
    Counter pkts_v{0}, pkts_a{0}, frames_decoded{0}, late_frames{0}, slip_drop_frames{0}, store_drops{0}, vdec_errors{0},
        adec_errors{0}, audio_resyncs{0}, skip_runs{0}, skip_dropped{0}, audio_chunks{0}, stamp_dropped{0}, flag_mismatch{0},
        prestart_dropped{0}, audio_absent{0}, audio_absent_pkts{0}, decoder_fallbacks{0}, reconnects_live{0}, entry_dropped{0},
        no_entry_point{0}, probe_scan_dropped{0}, relock_events{0}, relock_flushed_pkts{0}, relock_flushed_frames{0}, audio_fades{0}, audio_relock_resyncs{0};
    std::uint64_t relock_seen = 0;  // demux gija: paskutinis apdorotas RelockInfo::seq
    std::atomic<std::int64_t> comp_abs{0};
    // tick gija
    AVFrame* cur = nullptr;   // dabartinis kadras (interlaced — abu laukai)
    AVFrame* prev = nullptr;  // paskutinis geras (pakartojimui)
    std::uint64_t cur_frame = ~0ull;
    bool repeating = false;
    bool ever = false;  // bent vienas kadras jau rodytas (iki tol — paleidimas: latencija dar nepraėjo, ne praradimas)
    std::uint64_t repeat_frames = 0, missing_frames = 0, startup_frames = 0, late_taken = 0;
    std::uint64_t audio_blocks = 0, audio_late_samples = 0, audio_silent_samples = 0, audio_write_errors = 0;

    void stop_format(const std::string& why) {
        {
            std::lock_guard g(why_m);
            if (fmt_stopped.load()) return;
            fmt_why = why;
        }
        fmt_stopped = true;
    }
    std::string format_why() {
        std::lock_guard g(why_m);
        return fmt_why;
    }
    ~Source() {
        // gijos baigiasi pačios, kai nustatytas bendras stop (kviečiantysis jį nustato prieš naikindamas šaltinius)
        for (std::thread* t : {&th_demux, &th_video, &th_audio})
            if (t->joinable()) t->join();
        for (auto& [k, f] : frames) av_frame_free(&f);
        av_frame_free(&cur);
        av_frame_free(&prev);
        avcodec_parameters_free(&vpar0);
        avcodec_parameters_free(&apar0);
    }
};

struct Shared {
    const Opts& o;
    ig::Decoder decoder = ig::Decoder::Nvdec;
    bool to_device = true;  // kadrai CUDA (G); false — host yuv420p (C)
    AVBufferRef* hw = nullptr;
    int dev = -1;
    ig::FeedGateConfig gate;
    ig::AudioSyncConfig async;
    const std::atomic<bool>& stop;
    std::int64_t open_ns = 0;  // --input-wait-s, ns (0 — neribotai)
};

// libav žurnalas be paslapčių (A1): HLS / HTTP / SRT moduliai į žurnalą rašo URL (segmentų žetonai, passphrase) — kiekviena eilutė
// suformatuojama ir išvaloma redact_urls_in. Lygis — kaip numatytojo (av_log_get_level).
void redacting_av_log(void* avcl, int level, const char* fmt, va_list vl) {
    if (level > av_log_get_level()) return;
    static thread_local int prefix = 1;
    char line[2048];
    av_log_format_line2(avcl, level, fmt, vl, line, sizeof line, &prefix);
    std::fputs(ig::redact_urls_in(line).c_str(), stderr);
}

void install_redacting_av_log() { av_log_set_callback(redacting_av_log); }

AVCodecParameters* par_copy(const AVCodecParameters* p) {
    if (!p) return nullptr;
    AVCodecParameters* c = avcodec_parameters_alloc();
    avcodec_parameters_copy(c, p);
    return c;
}

std::string frame_desc(const AVFrame* f, bool inter) {
    return std::to_string(f->width) + "x" + std::to_string(f->height) + (inter ? " interlaced" : " progresyvus");
}

// --- demux gija ---------------------------------------------------------------------------------------------------
// B4: perfiksavimas — aiškus įvykis: žurnalo eilutė su priežastimi ir dydžiu, skaitiklis (santrauka, gyva statistika). Jei poslinkis
// pasikeitė ≥ pusę kadro, eilėje laukiantys video paketai (pažymėti SENU poslinkiu — kitaip jie liktų su juo, kai šuolis didesnis už
// dekodavimo atsargą) išmetami ir video gija pradeda nuo įėjimo taško nauju poslinkiu; garsas perjungiamas su perėjimu (AudioRing).
constexpr std::int64_t kRelockAbsorbNs = 20'000'000;  // M2: tiek sandūros šuolio sugeriama be pjūvio (AudioSync hard_samples 960 / 48 kHz)

void on_relock(Source& s) {
    const timing::RelockInfo ri = s.clock->last_relock();
    if (ri.seq == s.relock_seen) return;  // epochą pradėjęs garsas — tik video tinklelio nustatymas, ne perfiksavimas
    s.relock_seen = ri.seq;
    ++s.relock_events;
    char b[96];
    std::snprintf(b, sizeof b, "%+.1f ms", static_cast<double>(ri.step_ns) / 1e6);
    std::string what = s.name + ": perfiksavimas — " + timing::relock_reason_name(ri.reason) + ", poslinkis " + b;
    // M2: sandūros trūkis / persidengimas ≤ 20 ms (= AudioSync hard_samples, vienas 50 Hz garso blokas) sugeriamas vietoje: video —
    // ≤ 1 kadras (frame sync: pakartojimas / slip_drop), garsas — pririšamas prie naujos ašies (žymė resync, 5 ms perėjimas).
    // Anksčiau nuo Tf/2 viskas buvo išmetama ir likdavo ~latencijos tyla, o garsas dar ir nebūdavo perinkaruojamas (≤ 20 ms —
    // tik lėta kompensacija). Didesnis šuolis: kas senoje ašyje dar nepateikta, atmetama ABIEM srautams nuo TO PATIES momento
    // (dabar + 40 ms, video — kadro riboje; L2): eilės paketai, dekoduoti kadrai nuo pjūvio (žymė flush), garso žiedas nuo pjūvio (su
    // užgesinimu). Naujas turinys — nauju poslinkiu; garsas visada perinkaruojamas tame pačiame taške kaip video.
    const bool cut = std::llabs(ri.step_ns) > kRelockAbsorbNs;
    std::int64_t cut_sample = INT64_MIN;
    if (cut) {
        auto cut_tai = static_cast<std::int64_t>(ig::tai_now()) + 40'000'000;
        std::uint64_t cut_frame = 0;
        if (s.video) {
            cut_frame = timing::index_of(cut_tai, s.fmt.frame_rate) + 1;
            cut_tai = timing::time_of(cut_frame, s.fmt.frame_rate);
        }
        cut_sample = static_cast<std::int64_t>(static_cast<long double>(cut_tai) * 48000.0L / 1e9L);
        const std::uint64_t nv = s.video ? s.vq.drop_packets() : 0, na = s.audio ? s.aq.drop_packets() : 0;
        s.relock_flushed_pkts += nv + na;
        if (s.video) {
            QItem fl;
            fl.flush = true;
            fl.cut_frame = cut_frame;
            s.vq.push(fl);
        }
        const std::int64_t gone = s.audio ? s.ring.truncate(cut_sample, 240) : 0;
        what += "; nepateikta sena ašis atmesta nuo to paties momento: video paketų " + std::to_string(nv) + ", garso paketų " +
                std::to_string(na) + ", garso " + std::to_string(gone * 1000 / 48000) + " ms (su užgesinimu)";
    } else {
        what += "; ≤ " + std::to_string(kRelockAbsorbNs / 1'000'000) + " ms — sugerta vietoje (video ≤ 1 kadras, garsas perinkaruojamas)";
    }
    if (s.audio) {
        QItem rs;
        rs.resync = true;
        rs.cut_sample = cut_sample;
        s.aq.push(rs);
    }
    log_line("ingest", what);
}

void demux_thread(Source& s, Shared& c) {
    bool first = true;
    AVPacket* pkt = av_packet_alloc();
    ig::reconnect_loop(
        s.name, c.stop,
        [&](std::string& why) -> bool {
            if (!first) {
                s.reconnects_live = s.reconnects;  // ta pati (demux) gija didina reconnects — gyvai statistikai (B8)
                std::string err;
                // H3: HTTP(S) — ir prisijungiant iš naujo atidarymas ribotas (--input-wait-s, numatyta 30 s): serveris, kuris priima
                // ryšį, bet neatsako, kitaip laikytų šaltinį amžinai. UDP / RTP / SRT laukia paketų be ribos (šaltinis grįš pats).
                const std::string sc = ig::url_scheme(s.spec.url);
                const std::int64_t reopen_ns = sc == "http" || sc == "https" ? (c.open_ns > 0 ? c.open_ns : 30'000'000'000LL) : 0;
                if (!s.demux.open(s.spec, c.stop, reopen_ns, err)) {
                    why = err;
                    return false;
                }
                const ig::StreamInfo& in = s.demux.info();
                s.probe_scan_dropped += s.demux.entry_discarded();
                vramflow::VideoFormat f{};
                std::string fw;
                // B6: nepavykęs / neišsamus zondavimas (nėra video, nežinomas dydis ar pikselių formatas) — NE formato pasikeitimas:
                // sesija baigiama, jungiamasi iš naujo su atidėjimu. Formatą apibrėžia rastras ir skenavimas; kadrų dažnis zonduojant
                // gali būti įvertintas (avg_frame_rate) — jo neatitikimas tik žurnale (kadrų indeksus lemia dekoduotų kadrų PTS).
                if (s.video && (!in.video || in.vpar->width <= 0 || in.vpar->format < 0)) {
                    why = std::string("zondavimas po prisijungimo iš naujo neišsamus (") + (in.video ? "nežinomas dydis / formatas" : "nėra video") +
                          ") — bandoma iš naujo";
                    s.demux.close();
                    return false;
                }
                if (s.video && !s.demux.video_format(f, fw) && fw.find("dažnis") != std::string::npos) {
                    f = s.fmt;  // tik dažnis nežinomas — rastras / skenavimas tikrinami žemiau iš parametrų
                    f.raster.w = in.vpar->width;
                    f.raster.h = in.vpar->height;
                    fw.clear();
                }
                const bool same_raster = f.raster.w == s.fmt.raster.w && f.raster.h == s.fmt.raster.h && f.raster.interlaced == s.fmt.raster.interlaced;
                if (s.video && (!fw.empty() || !same_raster)) {
                    why = "formatas pasikeitė po prisijungimo iš naujo: " + (fw.empty() ? vramflow::format_name(f) : fw) + " (flow " +
                          vramflow::format_name(s.fmt) + ")";
                    s.stop_format(why);
                    s.demux.close();
                    return true;
                }
                if (s.video && (f.frame_rate.num != s.fmt.frame_rate.num || f.frame_rate.den != s.fmt.frame_rate.den))
                    log_line("ingest", s.name + ": po prisijungimo iš naujo zonduotas kadrų dažnis " + vramflow::format_name(f) + " (flow " +
                                           vramflow::format_name(s.fmt) + ") — laikomas įverčiu, šaltinis tęsiamas");
                // nauja sesija: dekoderiai atidaromi iš naujo. Laikrodis — tas pats: siuntėjas pradėjo iš naujo (PTS šuolis) —
                // nauja epocha ir fiksavimas; gyvo šaltinio PTS tęsiasi — tas pats PTS → TAI atvaizdis (indeksai nepasislenka)
                if (s.video) s.vq.push(QItem{{}, par_copy(in.vpar)});
                if (s.audio) s.aq.push(in.audio ? QItem{{}, par_copy(in.apar)} : QItem{{}, nullptr, true});
            }
            first = false;
            while (!c.stop.load() && !s.fmt_stopped.load()) {
                ig::Kind k{};
                if (s.demux.read(pkt, k, why) < 0) break;
                if ((k == ig::Kind::Video && !s.video) || (k == ig::Kind::Audio && !s.audio)) {
                    av_packet_unref(pkt);
                    continue;
                }
                if (const std::int64_t d = s.pending_step.exchange(0); d != 0) s.clock->on_clock_step(d);
                const auto arrival = static_cast<std::int64_t>(ig::tai_now());
                const int sid = k == ig::Kind::Video ? ig::kVideoStream : ig::kAudioStream;
                const timing::Stamp st = s.clock->observe(sid, pkt->pts, pkt->dts, pkt->duration, arrival);
                if (st.ev == timing::Event::Dropped) {
                    ++s.stamp_dropped;
                    av_packet_unref(pkt);
                    continue;
                }
                if (st.ev == timing::Event::Relocked) on_relock(s);
                QItem it;
                it.p.pkt = av_packet_alloc();
                av_packet_move_ref(it.p.pkt, pkt);
                it.p.pkt->pts = st.pts_tai_ns;
                it.p.pkt->dts = st.dts_tai_ns;
                it.p.pkt->time_base = AVRational{1, 1'000'000'000};
                // B1: paketas be PTS (PAFF antras laukas) — pkt->pts lieka AV_NOPTS_VALUE (kadro laikas — iš pirmo lauko), vartams — DTS
                it.p.pts_tai_ns = st.pts_tai_ns == timing::kNoTimestamp ? st.dts_tai_ns : st.pts_tai_ns;
                it.p.dts_tai_ns = st.dts_tai_ns;
                it.p.key = (it.p.pkt->flags & AV_PKT_FLAG_KEY) != 0;
                it.p.kind = k;
                it.p.ev = st.ev;
                PacketQueue& q = k == ig::Kind::Video ? s.vq : s.aq;
                ++(k == ig::Kind::Video ? s.pkts_v : s.pkts_a);
                q.push(it);
                // M1: eilė pasiekė baitų / elementų ribą — seniausi paketai išmetami (queue_drops); žurnale — viena eilutė episodui
                if (const std::uint64_t ep = s.vq.overflow_episodes() + s.aq.overflow_episodes(); ep != s.overflow_logged) {
                    s.overflow_logged = ep;
                    log_line("ingest", s.name + ": eilės riba " + std::to_string(s.queue_cap_bytes >> 20) +
                                           " MB pasiekta — seniausi paketai išmetami (dekodavimas nespėja arba serveris stumia daugiau nei latencija; " +
                                           "išmesta iš viso " + std::to_string(s.vq.dropped() + s.aq.dropped()) + ", --max-queue-mb)");
                }
            }
            s.demux.close();
            if (why.empty()) why = "sesija baigėsi";
            return s.fmt_stopped.load();
        },
        s.reconnects, s.error, [](const std::string& t) { log_line("ingest", t); });
    av_packet_free(&pkt);
}

// --- video dekodavimo gija ----------------------------------------------------------------------------------------
constexpr std::uint64_t kNoEntryPkts = 250;       // ~10 s 25p be įėjimo taško → „nėra dekoduojamo įėjimo taško“ (B7)
constexpr std::uint64_t kDecodeFailStreak = 100;  // ~4 s 25p: paketų iš eilės su klaida be nė vieno kadro → dekoderis laikomas sugedusiu
void store_frame(Source& s, ig::DecodedFrame& df) {
    AVFrame* f = df.f;
    const bool want_dev = f->format == AV_PIX_FMT_CUDA;
    const bool ok_fmt = want_dev || f->format == AV_PIX_FMT_YUV420P || f->format == AV_PIX_FMT_YUVJ420P;
    const bool inter = s.fmt.raster.interlaced;
    if (!ok_fmt || f->width != s.fmt.raster.w || f->height != s.fmt.raster.h) {  // raiška pasikeitė vidury srauto (K17)
        s.stop_format("formatas pasikeitė vidury srauto: " + frame_desc(f, df.interlaced) + " (flow " + vramflow::format_name(s.fmt) + ")");
        av_frame_free(&f);
        return;
    }
    // Kadro žymė neatitinka flow'o skenavimo (informacinis, kaip dec flag_mismatch): laukų tvarka nustatyta iš srauto (pirmas IDR);
    // kadro žymes nustato koderis (x264 — pagal savo įėjimo kadrų žymes), todėl neatitikimas — ne formato pasikeitimas.
    if (inter ? (!df.interlaced || !df.tff) : df.interlaced) ++s.flag_mismatch;
    ++s.frames_decoded;
    const std::uint64_t F = timing::index_of(df.pts_tai_ns, s.fmt.frame_rate);
    if (F < s.taken_upto.load(std::memory_order_acquire)) {  // jo laikas praėjo — tick'as jau bandė jį paimti
        ++s.late_frames;
        av_frame_free(&f);
        return;
    }
    // Kadro laikas praėjo, o tick'as šio šaltinio dar nerodo (paleidimas: variklis startuoja tik sukūrus visų šaltinių flow'us) —
    // kadras niekam nereikalingas. Kitaip žemėlapis augtų iki ribos, o CUDA kadrų baseinas — kartu (baseinas nesitraukia:
    // išmatuota ~340 MB VRAM šaltiniui vietoje ~100 MB).
    if (df.pts_tai_ns + timing::time_of(1, s.fmt.frame_rate) < static_cast<std::int64_t>(ig::tai_now())) {
        ++s.prestart_dropped;
        av_frame_free(&f);
        return;
    }
    std::lock_guard g(s.fm);
    // Kadrai, kurių laikas (visas kadro periodas) jau praėjo, žemėlapyje nereikalingi — jie lieka tik tada, kai tick'as šio šaltinio
    // dar neima (paleidimas: variklis startuoja sukūrus VISŲ šaltinių flow'us; šaltinis 0 laukia, kol zonduojami kiti). Be šio
    // valymo ankstyvo šaltinio žemėlapis (ir CUDA kadrų baseinas — jis nesitraukia) augdavo iki ribos: VRAM augo kvadratiškai
    // šaltinių skaičiui (išmatuota: 4 šaltiniai — 1,3 GB vietoje ~0,4 GB).
    const auto now = static_cast<std::int64_t>(ig::tai_now());
    const std::uint64_t f_past = timing::index_of(now - timing::time_of(1, s.fmt.frame_rate), s.fmt.frame_rate);
    while (!s.frames.empty() && s.frames.begin()->first < f_past) {
        av_frame_free(&s.frames.begin()->second);
        s.frames.erase(s.frames.begin());
        ++s.prestart_dropped;
    }
    auto [it, fresh] = s.frames.emplace(F, f);
    if (!fresh) {  // tas pats indeksas antrą kartą: video slydimas −1 kadras (šaltinis greitesnis, K6) — senesnis išmetamas
        av_frame_free(&it->second);
        it->second = f;
        ++s.slip_drop_frames;
    }
    while (s.frames.size() > 256) {  // apsauga (atsarga ribota vartais; čia — tik nuo nenumatytų PTS)
        av_frame_free(&s.frames.begin()->second);
        s.frames.erase(s.frames.begin());
        ++s.store_drops;
    }
}

void video_thread(Source& s, Shared& c) {
    ig::VideoDecoder dec;
    ig::SkipToKey skip;
    std::string err;
    // NVDEC paviršių atsarga virš DPB: kadrai iškart nukopijuojami į savo baseiną (VideoDecoder::compact), todėl dekoderiui
    // pakanka vieno papildomo (numatyta 1; anksčiau atsarga · kadrų dažnis + 8 — kiekvienas laikomas kadras kainavo du buferius)
    const int extra = c.o.surfaces > 0 ? c.o.surfaces : 1;
    // B8: dekoderis šaltiniui — NVDEC / cuvid, nepavykus (atidaryti arba ilgai nedekoduoja) — programinis (tas pats H.264 / HEVC
    // 8 bitų 4:2:0 kelias, G variante kadrai įkeliami į GPU), su žurnalo eilute ir skaitikliu decoder_fallbacks. Programinis irgi
    // nepavyko — šaltinis sustabdomas matomai (vieneto gedimas, kiti tęsia).
    ig::Decoder cur_dec = c.decoder;
    // sesijos parametrų KOPIJA (eilės žymės parametrai atlaisvinami iškart po atidarymo — rodyklė į juos būtų kabanti)
    std::unique_ptr<AVCodecParameters, void (*)(AVCodecParameters*)> cur_par(nullptr, [](AVCodecParameters* p) { avcodec_parameters_free(&p); });
    auto open = [&](const AVCodecParameters* par) {
        cur_par.reset(par_copy(par));
        std::string e1;
        const bool hw_test_fail = c.o.test_hw_decoder_fail && cur_dec != ig::Decoder::Sw;
        if (!hw_test_fail && dec.open(par, cur_dec, c.hw, c.to_device, c.o.threads, extra, c.dev, e1)) return true;
        if (hw_test_fail) e1 = std::string(ig::decoder_name(cur_dec)) + " atidaryti nepavyko (--test-hw-decoder-fail)";
        if (cur_dec != ig::Decoder::Sw) {
            log_line("ingest", s.name + ": " + e1 + " — perjungiama į programinį dekoderį");
            ++s.decoder_fallbacks;
            cur_dec = ig::Decoder::Sw;
            if (dec.open(par, cur_dec, c.hw, c.to_device, c.o.threads, extra, c.dev, err)) return true;
            e1 += "; programinis: " + err;
        }
        s.stop_format(s.name + ": dekoderis: " + e1);
        return false;
    };
    if (!open(s.vpar0)) return;
    std::uint64_t streak = 0;  // paketų iš eilės su dekoderio klaida ir be naujo kadro
    const bool is_h264 = s.vpar0->codec_id == AV_CODEC_ID_H264;
    ig::H264Scan entry_scan;
    std::uint64_t waited = 0;
    auto reset_scan = [&] {
        entry_scan = ig::H264Scan{};
        if (cur_par && cur_par->extradata && cur_par->extradata_size > 0)
            entry_scan.feed(cur_par->extradata, static_cast<std::size_t>(cur_par->extradata_size));
    };
    reset_scan();
    bool need_key = true;  // po (re)atidarymo — nuo rakto kadro (be atramos kadrų dekoderis išvestų iškraipytą vaizdą)
    ig::DecodedFrame df;
    auto drain = [&] {
        while (dec.receive(df, err)) store_frame(s, df);
        if (!err.empty()) {
            s.stop_format(s.name + ": " + err);
            err.clear();
        }
    };
    QItem it;
    while (!c.stop.load() && !s.fmt_stopped.load()) {
        if (!s.vq.pop(it, std::chrono::milliseconds(20))) continue;
        if (it.flush) {  // B4: perfiksavimas — senos ašies dekoduoti kadrai išmetami, pradedama nuo įėjimo taško
            dec.flush();
            skip.reset();
            need_key = true;
            waited = 0;
            reset_scan();
            // L2: tik nuo pjūvio kadro (tas pats momentas kaip garso žiedo pjūvis); ankstesni dar rodomi
            std::lock_guard g(s.fm);
            for (auto f = s.frames.lower_bound(it.cut_frame); f != s.frames.end();) {
                av_frame_free(&f->second);
                f = s.frames.erase(f);
                ++s.relock_flushed_frames;
            }
            continue;
        }
        if (it.par) {
            const bool ok = open(it.par);
            avcodec_parameters_free(&it.par);
            skip.reset();
            need_key = true;
            if (!ok) break;
            continue;
        }
        if (need_key) {
            // B7: įėjimo taškas — rakto paketas ARBA (H.264) recovery point SEI / I pjūvis (atviras GOP, intra atnaujinimas: IDR
            // gali niekada neateiti). Išmesti paketai skaičiuojami; per ilgai be įėjimo taško — matoma būsena (žurnalas, skaitiklis).
            bool entry = it.p.key;
            if (!entry && is_h264 && it.p.pkt) entry = entry_scan.feed(it.p.pkt->data, static_cast<std::size_t>(it.p.pkt->size));
            if (!entry) {
                ++s.entry_dropped;
                if (++waited == kNoEntryPkts) {
                    ++s.no_entry_point;
                    log_line("ingest", s.name + ": nėra dekoduojamo įėjimo taško (" + std::to_string(waited) +
                                           " paketų be IDR / recovery point / I pjūvio) — laukiama toliau");
                }
                PacketQueue::drop(it);
                continue;
            }
            static const char* const kEntry[] = {"rakto paketas", "IDR", "recovery point SEI", "I pjūvis"};
            if (waited > 0)
                log_line("ingest", s.name + ": dekodavimas pradėtas nuo įėjimo taško (" +
                                       kEntry[it.p.key ? 0 : static_cast<int>(entry_scan.entry())] + ") po " + std::to_string(waited) + " išmestų paketų");
            waited = 0;
            reset_scan();
        }
        need_key = false;
        // Atsarga (pamoka 3): į dekoderį ne anksčiau kaip lead prieš DTS_TAI
        while (!c.stop.load()) {
            const auto now = static_cast<std::int64_t>(ig::tai_now());
            if (ig::feed_decision(it.p.dts_tai_ns, now, c.gate) == ig::Feed::Send) break;
            const std::int64_t wait = std::min<std::int64_t>(it.p.dts_tai_ns - c.gate.lead_ns - now, 10'000'000);
            std::this_thread::sleep_for(std::chrono::nanoseconds(std::max<std::int64_t>(wait, 100'000)));
        }
        bool flush = false;
        if (!skip.admit(it.p.pts_tai_ns, it.p.key, static_cast<std::int64_t>(ig::tai_now()), c.gate, flush)) {
            PacketQueue::drop(it);  // nebespės — metama iškart (dekoderio laiko nekainuoja)
            continue;
        }
        if (flush) dec.flush();
        // testinis dekoderio sulėtinimas — tik siunčiamiems paketams (kaip tikra dekodavimo kaina)
        if (c.o.test_decode_delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(c.o.test_decode_delay_ms));
        const std::uint64_t errs0 = dec.errors(), frames0 = s.frames_decoded.load();
        int e = dec.send(it.p.pkt);
        while (e == AVERROR(EAGAIN) && !c.stop.load()) {
            drain();
            e = dec.send(it.p.pkt);
        }
        PacketQueue::drop(it);
        drain();
        // gyva statistika (B8): skaitikliai — iškart, ne gijos pabaigoje
        s.skip_runs = skip.skip_runs();
        s.skip_dropped = skip.dropped();
        s.vdec_errors = dec.errors();  // kaupiamas per atidarymus
        streak = dec.errors() > errs0 && s.frames_decoded.load() == frames0 ? streak + 1 : 0;
        if (streak >= kDecodeFailStreak) {
            streak = 0;
            if (cur_dec == ig::Decoder::Sw) {
                s.stop_format(s.name + ": dekoderis nedekoduoja (" + std::to_string(kDecodeFailStreak) + " paketų iš eilės su klaida)");
                break;
            }
            log_line("ingest", s.name + ": " + ig::decoder_name(cur_dec) + " nedekoduoja (" + std::to_string(kDecodeFailStreak) +
                                   " paketų iš eilės su klaida) — perjungiama į programinį dekoderį");
            ++s.decoder_fallbacks;
            cur_dec = ig::Decoder::Sw;
            if (!dec.open(cur_par.get(), cur_dec, c.hw, c.to_device, c.o.threads, extra, c.dev, err)) {
                s.stop_format(s.name + ": programinis dekoderis: " + err);
                break;
            }
            need_key = true;
            skip.reset();
        }
    }
}

// --- garso dekodavimo gija ----------------------------------------------------------------------------------------
// A2: garso kelias būna „nėra“, kai sesijoje nėra garso srauto arba dekoderio atidaryti nepavyko: paketai metami (skaičiuojami),
// žiedas nerašomas — tick'as flow'ui rašo tylą (kaip spragai), žurnale — viena eilutė; kitoje sesijoje bandoma iš naujo.
// (Anksčiau nepavykęs atidarymas palikdavo uždarytą dekoderį ir kitas paketas kviesdavo avcodec_send_packet(nullptr) — procesas krisdavo.)
void audio_thread(Source& s, Shared& c) {
    ig::AudioDecoder dec;
    int sessions = 0;
    bool absent = false;
    std::uint64_t resync_base = 0, realign_base = 0;
    auto open = [&](const AVCodecParameters* par) {
        resync_base += dec.resyncs();
        realign_base += dec.realigns();
        std::string why;
        bool ok = false;
        if (par == nullptr) why = "sesijoje nėra garso srauto";
        else if (c.o.test_audio_reopen_fail && sessions > 0) why = "dekoderio atidaryti nepavyko (--test-audio-reopen-fail)";
        else ok = dec.open(par, c.o.audio_channels, c.async, why);
        ++sessions;
        if (ok) {
            if (absent) log_line("ingest", s.name + ": garsas vėl yra (" + std::string(avcodec_get_name(par->codec_id)) + ")");
            absent = false;
            return;
        }
        dec.close();
        absent = true;
        ++s.audio_absent;
        log_line("ingest", s.name + ": garso nėra: " + why + " — flow'e tyla, bandoma kitoje sesijoje");
    };
    open(s.apar0);
    ig::AudioChunk ch;
    QItem it;
    while (!c.stop.load() && !s.fmt_stopped.load()) {
        if (!s.aq.pop(it, std::chrono::milliseconds(20))) continue;
        if (it.par || it.no_stream) {
            open(it.par);
            avcodec_parameters_free(&it.par);
            continue;
        }
        if (it.resync) {  // M2: perfiksavimas — pjūvis (ir vėl: per tą laiką gal įrašyta senos ašies) ir pririšimas prie naujos ašies
            if (it.cut_sample != INT64_MIN) s.ring.truncate(it.cut_sample, 240);
            dec.realign();
            continue;
        }
        if (absent) {
            ++s.audio_absent_pkts;
            PacketQueue::drop(it);
            continue;
        }
        // B4: perinkaravimas (sesijos pradžia, laikrodžio šuolis) — su 5 ms perėjimu (seno galo užgesinimas, naujo įžiebimas), ne kietu kirpimu
        auto put = [&] {
            if (ch.resync && s.ring.write_resync(ch.start, ch.planes, ch.n, 240)) ++s.audio_fades;
            else if (!ch.resync) s.ring.write(ch.start, ch.planes, ch.n);
            ++s.audio_chunks;
        };
        int e = dec.send(it.p.pkt);
        while (e == AVERROR(EAGAIN)) {
            while (dec.receive(ch)) put();
            e = dec.send(it.p.pkt);
        }
        PacketQueue::drop(it);
        while (dec.receive(ch)) put();
        s.audio_resyncs = resync_base + dec.resyncs();
        s.audio_relock_resyncs = realign_base + dec.realigns();
        s.comp_abs = dec.comp_abs_samples();
        s.adec_errors = dec.errors();
    }
}

// --- tick'as ------------------------------------------------------------------------------------------------------
// 25p šaltinis 50 Hz variklyje: vienetas dalyvauja kas k-tą indeksą (stride), flow'o indeksas — i / k.
class StridePort final : public engine::OutputPort {
public:
    StridePort(engine::OutputPort& in, std::uint32_t k) : in_(in), k_(k) {}
    engine::OpenResult open(engine::Index i, void*& slot) override { return in_.open(i / k_, slot); }
    bool commit() override { return in_.commit(); }
    void cancel() override { in_.cancel(); }
    bool fill_gap(engine::Index i, engine::GapAction a, engine::Index from) override {
        return in_.fill_gap(i / k_, a, from == engine::kNoIndex ? from : from / k_);
    }
    bool flush_gaps() override { return in_.flush_gaps(); }
    bool interlaced() const override { return in_.interlaced(); }
    std::uint32_t slot_count() const override { return in_.slot_count(); }
    void release() override { in_.release(); }
    bool reopen() override { return in_.reopen(); }

private:
    engine::OutputPort& in_;
    std::uint32_t k_;
};

struct IngestFn final : engine::TickFunction {
    const Opts& o;
    Stats& st;
    std::vector<std::unique_ptr<Source>>& src;
    const std::vector<std::uint32_t>& stride;
    std::vector<AudioOut>& aout;
    mxlInstance inst;
    bool cpu;
    cudaStream_t stream;
    ThreadPool& pool;
    double warm;
    vramflow::Rational erate;
    std::uint64_t t_wake = 0;
    std::vector<std::vector<float>> abuf;
    std::vector<float*> aptr;
    struct Job {
        media::SplitJob j;
        std::uint32_t unit;
        media::Dim dim;
        bool inter;
        int field, pitch_y, pitch_uv;
        media::ChromaMode mode;
        const AVFrame* f;
        bool done;
    };
    std::vector<Job> jobs;
    std::vector<media::SplitJob> part;

    IngestFn(const Opts& o_, Stats& st_, std::vector<std::unique_ptr<Source>>& src_, const std::vector<std::uint32_t>& stride_,
             std::vector<AudioOut>& aout_, mxlInstance inst_, bool cpu_, cudaStream_t stream_, ThreadPool& pool_, double warm_,
             vramflow::Rational erate_)
        : o(o_), st(st_), src(src_), stride(stride_), aout(aout_), inst(inst_), cpu(cpu_), stream(stream_), pool(pool_), warm(warm_),
          erate(erate_) {
        // didžiausias grain'as mėginiais (25p — 1920) — buferiai iš anksto (tick'e nealokuojama)
        abuf.assign(static_cast<std::size_t>(o.audio_channels), std::vector<float>(4096));
        for (auto& b : abuf) aptr.push_back(b.data());
        jobs.reserve(src.size());
        part.reserve(src.size());
    }

    vramflow::Rational unit_rate(std::uint32_t n) const { return {erate.num, erate.den * stride[n]}; }

    // B8: gyva šaltinio statistika --stats-file eilutėse (flows[n].fn): skirtumai stumiami kiekvieną tick'ą (Σ eilučių == santrauka)
    static constexpr const char* kLive[] = {"pkts_video",     "pkts_audio",        "frames_decoded",       "late_frames",
                                            "missing_frames", "repeat_frames",     "decode_errors",        "queue_drops",
                                            "skip_dropped_pkts", "audio_silent_samples", "audio_absent_pkts", "input_reconnects",
                                            "drift_slips",    "relocks",           "decoder_fallbacks"};
    static constexpr std::size_t kNLive = sizeof kLive / sizeof kLive[0];
    engine::StatsHub* hub = nullptr;
    std::vector<engine::CounterId> live_id;
    std::vector<std::uint64_t> live_last;
    void register_live(engine::StatsHub& h) {
        hub = &h;
        for (const char* k : kLive) live_id.push_back(h.register_counter(k));
        live_last.assign(src.size() * kNLive, 0);
    }
    static std::uint64_t live_value(const Source& s, std::size_t c) {
        const timing::ClockStats cs = s.clock->stats();
        switch (c) {
            case 0: return s.pkts_v;
            case 1: return s.pkts_a;
            case 2: return s.frames_decoded;
            case 3: return s.late_frames;
            case 4: return s.missing_frames;
            case 5: return s.repeat_frames;
            case 6: return s.vdec_errors + s.adec_errors;
            case 7: return s.vq.dropped() + s.aq.dropped();
            case 8: return s.skip_dropped;
            case 9: return s.audio_silent_samples;
            case 10: return s.audio_absent_pkts;
            case 11: return s.reconnects_live;
            case 12: return cs.drift_slips;
            case 13: return cs.relock_fresh + cs.relock_cont;
            case 14: return s.decoder_fallbacks;
            default: return 0;
        }
    }
    void push_live() {
        if (!hub) return;
        for (std::size_t n = 0; n < src.size(); ++n)
            for (std::size_t c = 0; c < kNLive; ++c) {
                const std::uint64_t v = live_value(*src[n], c);
                std::uint64_t& last = live_last[n * kNLive + c];
                if (v > last) hub->count(static_cast<std::uint32_t>(n), live_id[c], v - last);
                last = std::max(last, v);
            }
    }

    // Kadro F paėmimas iš žemėlapio (tiksliai F). Nėra, bet kitas jau laukia ir paskutinis buvo tikras — vienas pakartojimas
    // (frame sync: video slydimas +1 kadras arba vienas prarastas kadras).
    void take(Source& s, std::uint64_t F) {
        if (s.cur && !s.repeating) {
            av_frame_free(&s.prev);
            s.prev = s.cur;
        } else {
            av_frame_free(&s.cur);
        }
        s.cur = nullptr;
        const bool was_repeat = s.repeating;
        s.repeating = false;
        s.cur_frame = F;
        s.taken_upto.store(F + 1, std::memory_order_release);
        std::lock_guard g(s.fm);
        while (!s.frames.empty() && s.frames.begin()->first < F) {  // jų indeksai praleisti (variklio praleidimas)
            av_frame_free(&s.frames.begin()->second);
            s.frames.erase(s.frames.begin());
            ++s.late_taken;
        }
        if (!s.frames.empty() && s.frames.begin()->first == F) {
            s.cur = s.frames.begin()->second;
            s.frames.erase(s.frames.begin());
            s.ever = true;
            return;
        }
        if (!s.ever) {
            ++s.startup_frames;
            return;
        }
        if (s.prev && !was_repeat && !s.frames.empty()) {
            s.cur = av_frame_clone(s.prev);
            s.repeating = true;
            ++s.repeat_frames;
            return;
        }
        ++s.missing_frames;
    }

    void stop_unit(std::uint32_t n, engine::TickContext& ctx) {
        Source& s = *src[n];
        if (s.fmt_reported.exchange(true)) return;
        const std::string why = s.format_why();
        log_line("ingest", s.name + ": " + why + " — šaltinis sustabdytas (K17), kiti tęsia");
        st.flows[n].inc("format_changed");
        if (n < aout.size()) aout[n].release(inst);  // garso flow'as nebepildomas
        ctx.fail_unit(n, why);
    }

    void begin_tick(engine::Index i, engine::TickContext& ctx) override {
        t_wake = now_tai();
        for (std::uint32_t n = 0; n < src.size(); ++n) {
            Source& s = *src[n];
            if (s.fmt_stopped.load()) {
                stop_unit(n, ctx);
                continue;
            }
            if (!s.video || i % stride[n] != 0) continue;
            const std::uint64_t g = i / stride[n];
            const bool inter = s.fmt.raster.interlaced;
            const std::uint64_t F = inter ? g / 2 : g;
            if (!inter || (g & 1) == 0 || s.cur_frame != F) take(s, F);
            if (!s.cur) ctx.no_content(n);
        }
    }

    bool process(const engine::Batch& b, engine::TickContext& ctx) override {
        jobs.clear();
        for (const engine::UnitTick& ut : b.units) {
            if (ut.out.empty()) continue;  // tik garsas
            const std::uint32_t n = ut.unit;
            Source& s = *src[n];
            const AVFrame* f = s.cur;
            const bool inter = s.fmt.raster.interlaced;
            const std::uint64_t g = ut.index / stride[n];
            Job j{};
            j.j = media::SplitJob{f->data[0], f->data[1], static_cast<std::uint32_t*>(ut.out[0])};
            j.unit = n;
            j.dim = media::Dim{s.fmt.raster.w, s.fmt.raster.h};
            j.inter = inter;
            j.field = inter ? static_cast<int>(g & 1) : -1;
            j.pitch_y = f->linesize[0];
            j.pitch_uv = f->linesize[1];
            // interlaced flow: chroma pagal kadro žymę (kaip dec auto): interlaced kadras — lauko, progresyvus (PsF) — kadro
            j.mode = !inter || !(f->flags & AV_FRAME_FLAG_INTERLACED) ? media::ChromaMode::FrameSited : media::ChromaMode::FieldSited;
            j.f = f;
            jobs.push_back(j);
        }
        cudaError_t ce = cudaSuccess;
        if (cpu) {
            pool.run(static_cast<int>(jobs.size()), [&](int k) {
                const Job& j = jobs[static_cast<std::size_t>(k)];
                media::cpu::split_yuv420p(j.f->data[0], j.f->data[1], j.f->data[2], j.f->linesize[0], j.f->linesize[1], j.dim, j.field,
                                          j.mode, j.j.dst);
            });
        } else {
            // vienas paleidimas kiekvienai (rastras, laukas, chroma, pitch) grupei — dažniausiai visi šaltiniai vienoje
            for (std::size_t a = 0; a < jobs.size() && !ce; ++a) {
                if (jobs[a].done) continue;
                part.clear();
                const Job y = jobs[a];
                for (std::size_t k = a; k < jobs.size(); ++k) {
                    Job& x = jobs[k];
                    if (x.done || x.dim.w != y.dim.w || x.dim.h != y.dim.h || x.inter != y.inter || x.field != y.field || x.mode != y.mode ||
                        x.pitch_y != y.pitch_y || x.pitch_uv != y.pitch_uv)
                        continue;
                    part.push_back(x.j);
                    x.done = true;
                }
                ce = y.inter ? media::split_nv12(part.data(), static_cast<int>(part.size()), y.dim, y.field, y.mode, y.pitch_y, y.pitch_uv, stream)
                             : media::split_nv12_progressive(part.data(), static_cast<int>(part.size()), y.dim, y.pitch_y, y.pitch_uv, stream);
            }
            if (!ce && !jobs.empty()) ce = cudaStreamSynchronize(stream);
        }
        if (ce) {
            st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce);
            ctx.fatal(2, st.fatal);
            return false;
        }
        return true;
    }

    // Grain'o garsas: [sample_begin(g), sample_begin(g + 1)) iš žiedo (neparašyta — tyla, skaičiuojama). Rašoma KIEKVIENAM
    // indeksui (turinys ar spraga) — tęstinis flow'as be skylių.
    void write_audio(std::uint32_t n, engine::Index i) {
        if (n >= aout.size() || aout[n].w == nullptr || i % stride[n] != 0) return;
        Source& s = *src[n];
        const vramflow::Rational r = unit_rate(n);
        const std::uint64_t g = i / stride[n];
        const std::uint64_t s0 = vramflow::sample_begin(r, g), s1 = vramflow::sample_begin(r, g + 1);
        const auto count = static_cast<std::uint32_t>(s1 - s0);
        if (count > abuf[0].size()) return;
        std::uint32_t late = 0;
        const std::uint32_t valid = s.ring.read(static_cast<std::int64_t>(s0), count, aptr.data(), late);
        const mxlStatus as = aout[n].write(s1, count, [&](int ch, std::uint64_t k) { return abuf[static_cast<std::size_t>(ch)][k - s0]; });
        if (as != MXL_STATUS_OK) ++s.audio_write_errors;
        ++s.audio_blocks;
        s.audio_late_samples += late;
        s.audio_silent_samples += count - valid;
    }

    void end_unit(const engine::UnitTick& ut, engine::Outcome oc, engine::TickContext&) override {
        const std::uint32_t n = ut.unit;
        FlowStats& f = st.flows[n];
        switch (oc) {
            case engine::Outcome::Ready:
                if (!ut.out.empty()) f.inc("fields_ok");
                break;
            case engine::Outcome::Skipped: st.proc["clock_backward"]++; return;  // indeksas (ir garsas) jau parašytas
            case engine::Outcome::OutputError: f.inc("output_errors"); break;
            case engine::Outcome::GapNoContent: f.inc("gap_fields"); break;  // kadro nėra (dar nedekoduotas / prarastas)
            default: break;
        }
        write_audio(n, ut.index);
    }

    void on_skip(std::uint32_t unit, engine::Index from, engine::Index to, engine::TickContext&) override {
        for (engine::Index i = from; i < to; ++i) write_audio(unit, i);  // garso blokas — iš žiedo (video praleistas)
    }

    void end_tick(engine::Index i, engine::TickContext&) override {
        const mxlRational r = mxl_rate(erate);
        push_live();
        const std::uint64_t t_done = now_tai();
        if (static_cast<double>(t_done) >= warm) {
            st.add_hist("split_ms", t_done - t_wake);
            st.add_hist("out_lateness", t_done - mxlIndexToTimestamp(&r, i));
        }
    }

    void on_clock_step(std::int64_t d, engine::TickContext&) override {
        for (auto& s : src) s->pending_step += d;  // SourceClock keičia tik demux gija
    }
};

}  // namespace

int run_ingest(const Opts& o_in) {
    Opts o = o_in;
    Stats st;
    const std::size_t N = o.inputs.size();
    o.flows = static_cast<std::uint32_t>(N);
    st.flows.resize(N);
    st.set_seconds(o.seconds_kept);
    auto& stop = stop_flag();
    auto finish = [&](int rc) {
        st.fatal = ig::redact_urls_in(st.fatal);  // santraukoje — be paslapčių (A1; antra apsauga, pranešimai jau su išvalytais URL)
        st.exit_code = rc;
        write_summary(o, "ingest", st);
        std::fprintf(stderr, "[ingest] baigta, exit %d %s\n", rc, st.fatal.c_str());
        return rc;
    };
    if (N == 0) {
        st.fatal = "nenurodytas nė vienas --input";
        return finish(1);
    }
    // A1: operatoriaus URL schema — tik dokumentuotos (naudojimo klaida, prieš bet kokį tinklo veiksmą)
    for (const std::string& u : o.inputs)
        if (std::string why; !ig::source_url_allowed(u, why)) {
            st.fatal = "--input: " + why;
            return finish(1);
        }
    install_redacting_av_log();
    ig::Decoder decoder{};
    if (!ig::parse_decoder(o.decoder, decoder)) {
        st.fatal = "--decoder: " + o.decoder;
        return finish(1);
    }
    const bool cpu = o.impl == "cpu";
    if (cpu) decoder = ig::Decoder::Sw;  // C variantas: be CUDA — tik programinis dekoderis
    if (!cpu && o.storage == "host") {
        st.fatal = "ingest: --impl gpu --storage host (H) nepalaikomas — G (cuda-linear) arba C (--impl cpu)";
        return finish(1);
    }
    std::fprintf(stderr, "[ingest] %zu šaltiniai, dekoderis %s, %s\n", N, ig::decoder_name(decoder), cpu ? "C (host flow'ai)" : "G (cuda-linear)");
    AVBufferRef* hw = nullptr;
    int dev = -1;
    if (!cpu) {
        dev = cuda_device(o.device_uuid);  // FFmpeg pirmas sukuria pirminį kontekstą (kaip dec / enc), CUDA runtime — po to
        if (dev < 0) {
            st.fatal = "CUDA įrenginys";
            return finish(2);
        }
        char ord[8];
        std::snprintf(ord, sizeof ord, "%d", dev);
        if (const int e = av_hwdevice_ctx_create(&hw, AV_HWDEVICE_TYPE_CUDA, ord, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT); e < 0) {
            st.fatal = "hwdevice: " + ig::averr(e);
            return finish(2);
        }
        if (cudaSetDevice(dev) != cudaSuccess) {
            st.fatal = "cudaSetDevice";
            av_buffer_unref(&hw);
            return finish(2);
        }
        cudaFree(nullptr);
    }
    Shared sh{o, decoder, !cpu, hw, dev, {}, {}, stop, 0};
    sh.gate.lead_ns = static_cast<std::int64_t>(o.decode_lead_ms) * 1'000'000;
    sh.gate.min_ns = 0;
    sh.async.max_ppm = o.max_comp_ppm;

    // 1. Šaltiniai: atidaryti ir zonduoti (iki --input-wait-s) LYGIAGREČIAI (zondavimas + IDR — ~2–3 s šaltiniui; nuosekliai 8
    //    šaltiniai užtrukdavo ~24 s); formatas nustatomas PRIEŠ kuriant flow'us (pusiau sukurtų nelieka). Pirma klaida — bendras
    //    stop: kiti zondavimai nutraukiami, procesas baigiasi su ta klaida.
    std::vector<std::unique_ptr<Source>> src(N);
    std::vector<std::string> perr(N);
    std::vector<int> prc(N, 0);
    std::atomic<bool> failed_probe{false};
    const std::int64_t open_ns = o.input_wait_s > 0 ? static_cast<std::int64_t>(o.input_wait_s * 1e9) : 0;
    sh.open_ns = open_ns;
    auto probe = [&](std::size_t k) {
        auto& s = src[k];
        s = std::make_unique<Source>();
        s->idx = static_cast<int>(k);
        s->name = "šaltinis " + std::to_string(k) + " (" + ig::redact_url(o.inputs[k]) + ")";  // vardas eina į visus žurnalus
        s->spec.url = o.inputs[k];
        s->spec.hls_max_target_s = o.hls_max_target_s;
        s->spec.hls_max_playlist = o.hls_max_playlist;
        s->spec.format = o.format;
        s->spec.audio = o.ingest_audio;
        s->spec.audio_channels = o.audio_channels;
        s->spec.silence_ms = o.input_silence_ms;
        auto fail = [&](int code, std::string why) {
            prc[k] = code;
            perr[k] = std::move(why);
            if (!failed_probe.exchange(true)) stop = true;  // kiti zondavimai nutraukiami
        };
        std::string err;
        if (!s->demux.open(s->spec, stop, open_ns, err)) {
            if (failed_probe.load()) return;  // nutraukta dėl kito šaltinio klaidos
            return stop.load() ? fail(kStoppedWaiting, "sustabdyta laukiant " + ig::redact_url(o.inputs[k])) : fail(2, err);
        }
        const ig::StreamInfo& in = s->demux.info();
        s->probe_scan_dropped += s->demux.entry_discarded();
        s->video = in.video;
        s->audio = in.audio && o.ingest_audio;
        if (s->video) {
            std::string why;
            if (!s->demux.video_format(s->fmt, why)) return fail(2, s->name + ": nepalaikomas formatas — " + why);
            if (o.format != "auto") {
                vramflow::VideoFormat want{};
                if (!vramflow::parse_format(o.format, want, err) || !(want == s->fmt))
                    return fail(2, s->name + ": srauto formatas " + vramflow::format_name(s->fmt) + " != --format " + o.format);
            }
        }
        if (!s->video && !s->audio) return fail(2, s->name + ": nėra nei video, nei garso srauto");
        // K7: latencija; HLS — ne mažiau kaip 3 segmentai (pliūpsniai = segmentai), dreifo riba — segmentas
        s->latency_ns = static_cast<std::int64_t>(o.latency_ms) * 1'000'000;
        const std::int64_t seg = s->demux.hls_target_ns();
        timing::SourceClockConfig cc;
        if (seg > 0) {
            if (s->latency_ns < 3 * seg)
                std::fprintf(stderr, "[ingest] %s: HLS segmentas %.1f s — latencija %.1f s (≥ 3 segmentai, K7)\n", s->name.c_str(),
                             static_cast<double>(seg) / 1e9, 3.0 * static_cast<double>(seg) / 1e9);
            s->latency_ns = std::max(s->latency_ns, 3 * seg);
            // Segmentų atvykimo fazė lėtai slenka (grojaraščio perskaitymo ritmas) — atrodo kaip dreifas, nors laikrodis tas pats;
            // sekant nuolydį, garsas nuslinkdavo iki 20 ms ir video praslysdavo (išmatuota). B4: nuolydis nesekamas, bet atsargos
            // grįžtamasis ryšys už neutralios zonos (= segmentas) seka tikrą dreifą — be jo 100 ppm per ~5,5 val. → perfiksavimas.
            cc = timing::hls_clock_config(cc, seg);
        }
        cc.latency_ns = s->latency_ns;
        cc.align_rate = s->video ? timing::Rational{s->fmt.frame_rate.num, s->fmt.frame_rate.den} : timing::Rational{0, 1};
        cc.max_comp_ppm = o.max_comp_ppm;
        s->clock = std::make_unique<timing::SourceClock>(cc);
        if (s->video) s->clock->add_stream(ig::kVideoStream, {in.video_tb.num, in.video_tb.den}, true);
        if (s->audio) s->clock->add_stream(ig::kAudioStream, {in.audio_tb.num, in.audio_tb.den}, false);
        s->vpar0 = par_copy(in.vpar);
        s->apar0 = par_copy(in.apar);
        const double ring_s = static_cast<double>(s->latency_ns + sh.gate.lead_ns + 3'000'000'000LL) / 1e9;
        const auto ring_n = static_cast<std::size_t>(ring_s * 48000.0);
        // A1: žiedo dydis nuo latencijos (HLS — 3 segmentai) — riba baitais, kad grojaraštis negalėtų išpūsti atminties
        if (const std::uint64_t rb = ig::audio_ring_bytes(o.audio_channels, ring_n); rb > std::uint64_t{o.max_audio_ring_mb} << 20)
            return fail(2, s->name + ": garso žiedas " + std::to_string(rb >> 20) + " MB > riba " + std::to_string(o.max_audio_ring_mb) +
                               " MB (--max-audio-ring-mb; latencija " + std::to_string(s->latency_ns / 1'000'000) + " ms, kanalai " +
                               std::to_string(o.audio_channels) + ")");
        s->ring.init(o.audio_channels, ring_n);
        // M1: eilių riba baitais — (latencija + atsarga + 2 s) × 80 Mbit/s (dosni transliacijos sparta), arba --max-queue-mb
        s->queue_cap_bytes = o.max_queue_mb > 0 ? std::uint64_t{o.max_queue_mb} << 20
                                                : static_cast<std::uint64_t>(static_cast<double>(s->latency_ns + sh.gate.lead_ns + 2'000'000'000LL) / 1e9 *
                                                                             (kQueueMbitS * 1e6 / 8));
        s->vq.set_cap_bytes(s->queue_cap_bytes);
        s->aq.set_cap_bytes(s->queue_cap_bytes);
        std::fprintf(stderr, "[ingest] %s: %s%s, latencija %.0f ms\n", s->name.c_str(),
                     s->video ? vramflow::format_name(s->fmt).c_str() : "be video", s->audio ? " + garsas" : ", be garso",
                     static_cast<double>(s->latency_ns) / 1e6);
        // Šaltinio gijos — iškart po zondavimo. Demux: paketai žymimi atvykimo metu (kitaip, kol zonduojami kiti šaltiniai ir kuriami
        // flow'ai, jie kauptųsi UDP buferyje, o fiksavimo paketas gautų pavėluotą atvykimą — išmatuota: A/V per 60 s nuslinko −1,4 ms).
        // Dekodavimas: kadrai dekoduojami laiku (kitaip pirmi kadrai po paleidimo vėluodavo — late_frames).
        Source* sp = s.get();
        sp->th_demux = std::thread([sp, &sh] { demux_thread(*sp, sh); });
        if (sp->video) sp->th_video = std::thread([sp, &sh] { video_thread(*sp, sh); });
        if (sp->audio) sp->th_audio = std::thread([sp, &sh] { audio_thread(*sp, sh); });
    };
    {
        std::vector<std::thread> pt;
        for (std::size_t k = 0; k < N; ++k) pt.emplace_back(probe, k);
        for (auto& t : pt) t.join();
    }
    int rc = 0;
    for (std::size_t k = 0; k < N && !rc; ++k)
        if (prc[k]) {
            rc = prc[k];
            st.fatal = perr[k];
        }
    // Klaida — šaltinių gijos sustabdomos (bendras stop) ir sujungiamos (~Source)
    auto abort_sources = [&] {
        stop = true;
        src.clear();
        av_buffer_unref(&hw);
    };
    if (rc) {
        abort_sources();
        return finish(exit_code(rc));
    }

    // 2. Variklio dažnis — didžiausias grain'ų dažnis; kiti šaltiniai — kas k-tą indeksą (k sveikas)
    vramflow::VideoFormat efmt = vramflow::kFmt1080i50;  // tik garsas — 50 Hz blokai (960 mėginių)
    bool have_video = false, any_inter = false;
    for (auto& s : src)
        if (s->video && (!have_video || s->fmt.grain_rate().num * efmt.grain_rate().den > efmt.grain_rate().num * s->fmt.grain_rate().den)) {
            efmt = s->fmt;
            have_video = true;
        }
    const vramflow::Rational erate = efmt.grain_rate();
    std::vector<std::uint32_t> stride(src.size(), 1);
    for (std::size_t k = 0; k < src.size(); ++k) {
        if (!src[k]->video) continue;
        any_inter = any_inter || src[k]->fmt.raster.interlaced;
        const vramflow::Rational g = src[k]->fmt.grain_rate();
        const std::int64_t num = erate.num * g.den, den = erate.den * g.num;
        if (num % den != 0) {
            st.fatal = src[k]->name + ": grain'ų dažnis " + std::to_string(g.num) + "/" + std::to_string(g.den) + " nėra " +
                       std::to_string(erate.num) + "/" + std::to_string(erate.den) +
                       " dalmuo — skirtingų dažnių šeimų šaltiniai (pvz. 50 ir 59,94) — atskiruose procesuose";
            abort_sources();
            return finish(2);
        }
        stride[k] = static_cast<std::uint32_t>(num / den);
    }
    o.fmt = efmt;  // engine_config / deadline_ns / santraukos formatas — variklio dažnis
    engine::EngineConfig ecfg = engine_config(o, EngineTiming::Legacy, 0);
    if (any_inter) ecfg.skip_align = 2;  // po praleidimo tęsti nuo viršutinio lauko (25p — kas antrą — taip pat lyginis)
    EdgePlan plan;
    std::string err;
    if (!plan_edges(o, ecfg, {}, true, "ingest", st, plan, err)) {
        st.fatal = err;
        abort_sources();
        return finish(2);
    }

    // 3. Flow'ai: 07 video, 08 garsas (numeris = šaltinio eilės nr. + --flow-base); briaunos žyma: commit'o riba kaip šaltinio,
    //    turinio vėlinimas = latencija (kadras rodomas tiek po atvykimo)
    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) {
        st.fatal = "mxlCreateInstance";
        abort_sources();
        return finish(2);
    }
    mxlGarbageCollectFlows(inst);
    const std::string opts = cpu ? std::string{} : cuda_options(dev);
    std::vector<VideoOut> vout(src.size());
    std::vector<AudioOut> aout(src.size());
    std::vector<std::string> tags(src.size());
    for (std::size_t k = 0; k < src.size() && !rc; ++k) {
        Source& s = *src[k];
        const auto n = static_cast<std::uint32_t>(o.flow_base + k);
        tags[k] = engine::timing_tags_json(engine::OutputTiming{plan.out.commit_bound_ns, s.latency_ns});
        if (s.video &&
            !create_output(inst, vout[k], flow_id(kRoleIngest, n), video_def(kRoleIngest, n, o.run_seed, s.fmt, tags[k]), opts, s.fmt, o, err)) {
            st.fatal = s.name + ": video flow: " + err;
            rc = 2;
        }
        const vramflow::Rational ar = s.video ? s.fmt.grain_rate() : erate;
        if (!rc && s.audio &&
            !create_output(inst, aout[k], flow_id(kRoleIngestAudio, n), ingest_audio_def(n, o.run_seed, o.audio_channels, tags[k]), ar, o, err,
                           o.audio_channels)) {
            st.fatal = s.name + ": garso flow: " + err;
            rc = 2;
        }
    }
    auto release_all = [&] {
        for (auto& v : vout) v.release(inst);
        for (auto& a : aout) a.release(inst);
        mxlDestroyInstance(inst);
        av_buffer_unref(&hw);
    };
    if (rc) {
        stop = true;
        src.clear();
        release_all();
        return finish(rc);
    }
    st.proc["latency_ms"] = static_cast<std::uint64_t>(src[0]->latency_ns / 1'000'000);
    st.proc["decode_lead_ms"] = o.decode_lead_ms;

    // 4. Prievadai, variklis
    cudaStream_t stream{};
    // BLOKUOJANTIS stream'as (kaip dec): h264_cuvid kopija ir programinio kelio įkėlimas vyksta numatytajame (legacy) stream'e
    if (!cpu) cudaStreamCreate(&stream);
    GapFill gap_fill(gap_policy(o.on_loss), !cpu);
    for (std::size_t k = 0; k < src.size() && !rc; ++k)
        if (src[k]->video && !gap_fill.prepare(vout[k].raster, st.fatal)) rc = 2;
    std::vector<std::unique_ptr<engine::MxlVideoOutput>> ports(src.size());
    std::vector<std::unique_ptr<StridePort>> sports(src.size());
    for (std::size_t k = 0; k < src.size(); ++k) {
        if (!src[k]->video) continue;
        ports[k] = std::make_unique<engine::MxlVideoOutput>(vout[k], gap_fill.copy(), gap_fill.flush(), gap_fill.black(vout[k].raster));
        ports[k]->release_fn = [&vout, k, inst] { vout[k].release(inst); };
        // Atkūrimas (D8) — po išėjimo klaidų; formatas pasikeitęs (K17) — šaltinis neatkuriamas
        ports[k]->reopen_fn = [&, k] {
            if (src[k]->fmt_stopped.load()) return false;
            if (!cpu) cudaSetDevice(dev);
            std::string e;
            const auto n = static_cast<std::uint32_t>(o.flow_base + k);
            return create_output(inst, vout[k], flow_id(kRoleIngest, n), video_def(kRoleIngest, n, o.run_seed, src[k]->fmt, tags[k]), opts,
                                 src[k]->fmt, o, e);
        };
        if (stride[k] > 1) sports[k] = std::make_unique<StridePort>(*ports[k], stride[k]);
    }
    ThreadPool pool(cpu ? (o.threads > 0 ? o.threads : static_cast<int>(std::min<std::size_t>(src.size(), 4))) : 1);
    engine::SystemClock clk;
    EngineStats es(o, "ingest");
    engine::StatsHub& hub = es.hub;
    engine::TickEngine eng(ecfg, clk, hub, stop);
    for (std::size_t k = 0; k < src.size() && !rc; ++k) {
        engine::UnitSpec u;
        u.name = flow_id(src[k]->video ? kRoleIngest : kRoleIngestAudio, static_cast<std::uint32_t>(o.flow_base + k));
        u.stride = stride[k];
        u.residue = 0;
        if (src[k]->video) u.out.push_back(stride[k] > 1 ? static_cast<engine::OutputPort*>(sports[k].get()) : ports[k].get());
        std::uint32_t id = 0;
        if (!eng.add_unit(u, id, err)) {
            st.fatal = "add_unit: " + err;
            rc = 2;
        }
    }
    // 5. Šaltinių gijos (demux, video, garsas) jau veikia nuo zondavimo
    IngestFn fn(o, st, src, stride, aout, inst, cpu, stream, pool, warm_tai_after(o.warmup_s), erate);
    fn.register_live(hub);
    bool engine_loss = false;
    std::uint32_t failed = 0;
    engine::RunResult res;
    if (!rc) {
        res = eng.run(fn);
        es.finish();
        engine_loss = res.any_loss;
        failed = res.failed_units;
        rc = engine_rc(res, st);
        to_summary(hub, eng, res, static_cast<std::uint32_t>(src.size()), SkipScope::PerFlow, st);
    }
    stop = true;  // gijos baigiasi per ≤ ~100 ms (pertraukimo kabliukas, eilių laukimas 20 ms)
    for (auto& s : src) {
        for (std::thread* t : {&s->th_demux, &s->th_video, &s->th_audio})
            if (t->joinable()) t->join();
        s->vq.clear();
        s->aq.clear();
    }
    bool fmt_change = false;
    for (std::size_t k = 0; k < src.size(); ++k) {
        Source& s = *src[k];
        FlowStats& f = st.flows[k];
        const timing::ClockStats cs = s.clock->stats();
        f.inc("pkts_video", s.pkts_v);
        f.inc("pkts_audio", s.pkts_a);
        f.inc("frames_decoded", s.frames_decoded);
        f.inc("late_frames", s.late_frames);            // dekoduotas po savo laiko (pavėlavęs) — išmestas
        f.inc("missing_frames", s.missing_frames);      // tick'as kadro neturėjo (spraga pagal --on-loss)
        f.inc("startup_frames", s.startup_frames);      // iki pirmo kadro (latencija) — spragos, ne praradimas
        f.inc("repeat_frames", s.repeat_frames);        // frame sync: vienas kadras pakartotas
        f.inc("slip_drop_frames", s.slip_drop_frames);  // video slydimas −1 kadras: senesnis išmestas
        f.inc("skipped_frames", s.late_taken);          // jų indeksai praleisti varikliu
        f.inc("store_drops", s.store_drops);
        f.inc("prestart_dropped", s.prestart_dropped);  // dekoduotas po savo laiko, kol variklis dar nerodė (paleidimas)
        f.inc("flag_mismatch", s.flag_mismatch);
        f.inc("skip_to_key", s.skip_runs);
        f.inc("skip_dropped_pkts", s.skip_dropped);
        f.inc("decode_errors", s.vdec_errors + s.adec_errors);
        f.inc("queue_drops", s.vq.dropped() + s.aq.dropped());
        f.inc("queue_overflows", s.vq.overflow_episodes() + s.aq.overflow_episodes());  // M1: perpildymo episodai
        f.c["queue_cap_mb"] = s.queue_cap_bytes >> 20;
        f.inc("stamp_dropped", s.stamp_dropped);
        f.inc("audio_blocks", s.audio_blocks);
        f.inc("audio_chunks", s.audio_chunks);
        f.inc("audio_late_samples", s.audio_late_samples);
        f.inc("audio_silent_samples", s.audio_silent_samples);
        f.inc("audio_write_errors", s.audio_write_errors);
        f.inc("audio_resyncs", s.audio_resyncs);
        f.inc("audio_absent", s.audio_absent);            // sesijos be garso (nėra srauto / dekoderis neatsidarė) — flow'e tyla (A2)
        f.inc("audio_absent_pkts", s.audio_absent_pkts);  // tų sesijų garso paketai — išmesti
        f.inc("decoder_fallbacks", s.decoder_fallbacks);  // NVDEC / cuvid → programinis (B8)
        f.inc("entry_dropped_pkts", s.entry_dropped);    // B7: paketai prieš įėjimo tašką (po (re)atidarymo)
        f.inc("no_entry_point", s.no_entry_point);       // B7: per kNoEntryPkts be įėjimo taško
        f.inc("probe_scan_dropped_pkts", s.probe_scan_dropped);  // B7: zondavimo skenavimo išmesti paketai (iki įėjimo taško)
        f.inc("relock_events", s.relock_events);                 // B4: perfiksavimai (žurnale — priežastis ir dydis)
        f.inc("relock_flushed_pkts", s.relock_flushed_pkts);     // B4: senos ašies paketai, atmesti perfiksuojant
        f.inc("relock_flushed_frames", s.relock_flushed_frames);
        f.inc("audio_fades", s.audio_fades);                     // B4: garso perjungimai su perėjimu
        f.inc("audio_relock_resyncs", s.audio_relock_resyncs);   // M2: garsas pririštas prie naujos ašies po perfiksavimo
        f.c["comp_ppm_abs_x1000"] = static_cast<std::uint64_t>(std::llround(std::abs(cs.comp_ppm) * 1000));  // B3: taikoma rampa
        f.c["margin_err_ms_abs"] = static_cast<std::uint64_t>(std::llabs(cs.margin_err_ns) / 1'000'000);     // B3: atsargos paklaida
        f.inc("audio_comp_samples", static_cast<std::uint64_t>(s.comp_abs.load()));
        f.inc("wraps", cs.wraps);
        f.inc("discontinuities", cs.discontinuities);
        f.inc("relock_cont", cs.relock_cont);
        f.inc("relock_fresh", cs.relock_fresh);
        f.inc("drift_slips", cs.drift_slips);
        f.inc("prelock_dropped", cs.prelock_dropped);
        f.c["drift_ppm_abs_x1000"] = static_cast<std::uint64_t>(std::llround(std::abs(cs.drift_ppm) * 1000));
        f.c["drift_ppm_negative"] = cs.drift_ppm < 0 ? 1 : 0;
        if (s.reconnects) f.inc("input_reconnects", s.reconnects);
        if (s.fmt_stopped.load()) {
            fmt_change = true;
            f.c["format_changed"] = 1;
            if (st.fatal.find("CUDA") == std::string::npos) st.fatal = s.name + ": " + s.format_why();
        }
    }
    if (!cpu) cudaStreamDestroy(stream);
    src.clear();  // kadrai ir dekoderiai — prieš CUDA konteksto atlaisvinimą
    release_all();
    // K17: formatas pasikeitė — exit 6 (ir kai sustojo vienintelis šaltinis: variklis tai mato kaip „visi vienetai sugedo“)
    if (fmt_change && st.fatal.find("CUDA") == std::string::npos) {
        rc = kExitFormatChange;
        failed = 0;
    }
    bool loss = engine_loss;
    for (auto& f : st.flows)
        for (const char* k : {"late_frames", "missing_frames", "decode_errors", "output_errors", "audio_write_errors"})
            if (f.c.count(k) && f.c[k]) loss = true;
    return finish(exit_code_for(o, rc, loss, failed, st));
}

}  // namespace mxlv::func
