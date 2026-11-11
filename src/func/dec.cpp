// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// dec — F-DEC (architektūra GPU funkcijų §5.4): N × H.264 MPEG-TS per UDP → h264_cuvid (NVDEC, deint=weave)
// → NV12 CUDA kadrai → jitter eilė → batch kernel'is → N × MXL (cuda-linear), formatas --format (numatytasis 1080i50):
//  - interlaced flow: kadras → du v210 laukai; chroma pagal KADRO žymę AV_FRAME_FLAG_INTERLACED: pažymėtas interlaced —
//    lauko chroma (FieldSited), progresyvus kadras (PsF) — kadro chroma (FrameSited);
//  - progresyvus flow: kadras → vienas kadro grain'as (split_nv12_progressive, kadro chroma).
// Taip pat: `clip` (bandomojo klipo NV12 kadrai stdout'e iš CPU etalono) ir `ref-crc` (programinio dekoderio NV12 →
// laukų CRC32C etalonas bitų tikslumo patikrai).
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/crc32c.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/cpu/cpu_media.hpp"
#include "media/media_kernels.hpp"
#include "media/media_ref.hpp"

namespace mxlv::func {
namespace {

std::string averr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(e, b, sizeof b);
    return b;
}

enum AVPixelFormat pick_cuda(AVCodecContext*, const enum AVPixelFormat* f) {
    for (; *f != AV_PIX_FMT_NONE; ++f)
        if (*f == AV_PIX_FMT_CUDA) return *f;
    return AV_PIX_FMT_NONE;
}

struct DecFrame {
    std::int64_t n;  // kadro numeris nuo pirmo gauto
    AVFrame* f;
};

// Paskutinio laikrodžio šuolio monotoninis laikas. UDP įėjimas ateina realiu (monotoniniu) laiku, o tick'as — pagal
// TAI: šuolis atgal priverčia tick'ą laukti, ir eilė persipildo. Tokie perpildymai per 3 s po šuolio — aplinkos įvykis.
std::atomic<std::int64_t> g_clock_event_mono{INT64_MIN / 2};
std::int64_t mono_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
bool after_clock_event() { return mono_ns() - g_clock_event_mono.load() < 3'000'000'000ll; }

struct DecStream {
    int idx = 0;
    std::thread th;
    std::mutex m;
    std::deque<DecFrame> q;
    std::string error;
    bool opened = false;
    std::int64_t pts0 = AV_NOPTS_VALUE, prev_n = -1;
    // statistika (darbininkas)
    std::uint64_t frames_decoded = 0, pkts = 0, pkts_corrupt = 0, decode_errors = 0, flag_mismatch = 0, pts_gap_frames = 0,
                  overflow_drops = 0, send_eagain = 0, clock_overflow_drops = 0, startup_overflow_drops = 0;
    std::atomic<bool> playing{false};  // tick'as pradėjo atkurti šį srautą (iki tol perpildymas — paleidimo pliūpsnis)
};

void dec_worker(const Opts& o, AVBufferRef* hw, DecStream& s, const std::atomic<bool>& stop) {
    const bool inter = o.fmt.raster.interlaced;
    const double fps = static_cast<double>(o.fmt.frame_rate.num) / static_cast<double>(o.fmt.frame_rate.den);
    AVFormatContext* ic = nullptr;
    AVCodecContext* c = nullptr;
    AVPacket* pkt = av_packet_alloc();
    AVFrame* fr = av_frame_alloc();
    char url[256];
    std::snprintf(url, sizeof url,
                  "udp://127.0.0.1:%d?fifo_size=131072&overrun_nonfatal=1&buffer_size=8388608&timeout=20000000",
                  o.port_base + s.idx);
    // Trumpas zondavimas: numatytasis probesize 5 MB prie 8,5 Mbit/s = ~4,7 s, o sukaupti paketai po to
    // dekoduojami pliūpsniu. TS H.264 parametrai žinomi iš SPS per < 0,5 s.
    AVDictionary* io = nullptr;
    av_dict_set(&io, "probesize", "1000000", 0);
    av_dict_set(&io, "analyzeduration", "500000", 0);
    int e = avformat_open_input(&ic, url, av_find_input_format("mpegts"), &io);
    av_dict_free(&io);
    int vs = -1;
    const AVCodec* codec = nullptr;
    AVDictionary* d = nullptr;
    if (e < 0) { s.error = "open_input: " + averr(e); goto out; }
    if ((e = avformat_find_stream_info(ic, nullptr)) < 0) { s.error = "find_stream_info: " + averr(e); goto out; }
    vs = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vs < 0) { s.error = "nėra video srauto"; goto out; }
    if (o.impl == "cpu") {
        // C: FFmpeg programinis H.264 dekoderis (normatyviai bitų tikslus), kadrų + slice'ų gijos
        codec = avcodec_find_decoder_by_name("h264");
        c = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(c, ic->streams[vs]->codecpar);
        c->thread_count = o.threads > 0 ? o.threads : 2;
        c->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    } else {
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        c = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(c, ic->streams[vs]->codecpar);
        c->hw_device_ctx = av_buffer_ref(hw);
        c->get_format = pick_cuda;
        av_dict_set(&d, "deint", "weave", 0);
        av_dict_set(&d, "drop_second_field", "0", 0);
    }
    c->pkt_timebase = ic->streams[vs]->time_base;
    e = avcodec_open2(c, codec, &d);
    av_dict_free(&d);
    if (e < 0) { s.error = "cuvid open: " + averr(e); goto out; }
    s.opened = true;
    while (!stop.load()) {
        e = av_read_frame(ic, pkt);
        if (e == AVERROR(EAGAIN)) continue;
        if (e < 0) { if (!stop.load()) s.error = "read_frame: " + averr(e); break; }
        if (pkt->stream_index != vs) { av_packet_unref(pkt); continue; }
        ++s.pkts;
        if (pkt->flags & AV_PKT_FLAG_CORRUPT) ++s.pkts_corrupt;
        // EAGAIN = dekoderio išvesties eilė pilna: pirma nuskaityti kadrus, tada TĄ PATĮ paketą siųsti iš naujo.
        // (Ankstesnė klaida: paketas buvo išmetamas → trūkstami duomenys kadro viduryje → klaidų maskavimas.)
        bool pkt_pending = true;
        for (;;) {
            if (pkt_pending) {
                e = avcodec_send_packet(c, pkt);
                if (e == 0) pkt_pending = false;
                else if (e != AVERROR(EAGAIN)) { ++s.decode_errors; pkt_pending = false; }
                else ++s.send_eagain;
            }
            const int r = avcodec_receive_frame(c, fr);
            if (r != 0) {
                if (!pkt_pending) break;   // paketas priimtas ir kadrų daugiau nėra
                if (r != AVERROR(EAGAIN)) { ++s.decode_errors; pkt_pending = false; break; }
                continue;                  // dar laukiam vietos dekoderyje
            }
            const AVPixelFormat want_fmt = o.impl == "cpu" ? AV_PIX_FMT_YUV420P : AV_PIX_FMT_CUDA;
            if (fr->format != want_fmt || fr->width != o.fmt.raster.w || fr->height != o.fmt.raster.h) {
                s.error = "netikėtas kadro formatas " + std::to_string(fr->width) + "x" + std::to_string(fr->height) + " (laukta " +
                          vramflow::format_name(o.fmt) + ")";
                av_frame_unref(fr);
                goto out;
            }
            // žymė neatitinka flow'o skenavimo: interlaced flow'ui — ne TFF interlaced kadras (progresyvus kadras vis tiek
            // teisingas — kadro chroma), progresyviam — interlaced kadras (informacinis skaitiklis)
            if (inter ? (!(fr->flags & AV_FRAME_FLAG_INTERLACED) || !(fr->flags & AV_FRAME_FLAG_TOP_FIELD_FIRST))
                      : (fr->flags & AV_FRAME_FLAG_INTERLACED) != 0)
                ++s.flag_mismatch;
            if (fr->decode_error_flags) ++s.decode_errors;
            const std::int64_t pts = fr->best_effort_timestamp != AV_NOPTS_VALUE ? fr->best_effort_timestamp : fr->pts;
            const AVRational tb = ic->streams[vs]->time_base;
            if (s.pts0 == AV_NOPTS_VALUE) s.pts0 = pts;
            // kadro numeris: (pts − pts0) / kadro trukmė (1 / kadrų dažnis)
            const std::int64_t n = static_cast<std::int64_t>(std::llround(static_cast<double>(pts - s.pts0) * tb.num * fps / tb.den));
            if (s.prev_n >= 0 && n > s.prev_n + 1) s.pts_gap_frames += n - s.prev_n - 1;
            s.prev_n = std::max(s.prev_n, n);
            ++s.frames_decoded;
            if (s.idx == 0 && std::getenv("MXLV_DEC_TRACE")) {  // diagnostika: kadrų atvykimo laikai
                static FILE* tf = std::fopen((o.out + "/dec_trace0.csv").c_str(), "w");
                if (tf) std::fprintf(tf, "%llu,%lld,%zu\n", static_cast<unsigned long long>(now_tai()), static_cast<long long>(n), s.q.size());
            }
            AVFrame* keep = av_frame_alloc();
            av_frame_move_ref(keep, fr);
            std::lock_guard g(s.m);
            s.q.push_back({n, keep});
            while (s.q.size() > 60) {  // jitter eilės riba (2,4 s) — tik apsauga nuo atminties augimo
                av_frame_free(&s.q.front().f);
                s.q.pop_front();
                if (!s.playing.load()) ++s.startup_overflow_drops;          // zondavimo pliūpsnis, srautas dar negroja
                else if (after_clock_event()) ++s.clock_overflow_drops;    // tick'as laukė dėl laikrodžio šuolio atgal
                else ++s.overflow_drops;
            }
        }
        av_packet_unref(pkt);
    }
out:
    av_frame_free(&fr);
    av_packet_free(&pkt);
    avcodec_free_context(&c);
    if (ic) avformat_close_input(&ic);
    if (!s.error.empty()) std::fprintf(stderr, "[dec %d] %s\n", s.idx, s.error.c_str());
}

}  // namespace

int run_dec(const Opts& o) {
    Stats st;
    st.flows.resize(o.flows);
    auto& stop = stop_flag();
    std::vector<std::unique_ptr<DecStream>> ds;
    AVBufferRef* hw = nullptr;
    auto finish = [&](int rc) {
        st.exit_code = rc;
        write_summary(o, "dec", st);
        std::fprintf(stderr, "[dec] baigta, exit %d %s\n", rc, st.fatal.c_str());
        return rc;
    };
    // Variantas C (--impl cpu): programinis dekoderis → CPU split → host flow; CUDA nekuriamas.
    const bool cpu = o.impl == "cpu";
    ThreadPool pool(cpu ? (o.threads > 0 ? o.threads : static_cast<int>(std::min<unsigned>(o.flows, 4))) : 1);
    int dev = -1;
    if (!cpu) {
        // FFmpeg pirmas sukuria primary kontekstą (žr. enc.cpp), CUDA runtime prisijungia po to.
        dev = cuda_device(o.device_uuid);
        if (dev < 0) { st.fatal = "CUDA įrenginys"; return finish(2); }
        char ord[8];
        std::snprintf(ord, sizeof ord, "%d", dev);
        if (int e = av_hwdevice_ctx_create(&hw, AV_HWDEVICE_TYPE_CUDA, ord, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT); e < 0) {
            st.fatal = "hwdevice: " + averr(e);
            return finish(2);
        }
        if (cudaSetDevice(dev) != cudaSuccess) { st.fatal = "cudaSetDevice"; return finish(2); }
        cudaFree(nullptr);
    }
    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; return finish(2); }
    mxlGarbageCollectFlows(inst);
    std::vector<VideoOut> out(o.flows);
    std::string err;
    int rc = 0;
    const std::string opts = cpu ? std::string{} : cuda_options(dev);
    const bool inter = o.fmt.raster.interlaced;
    const media::Dim dim{o.fmt.raster.w, o.fmt.raster.h};
    const mxlRational rate = mxl_rate(o.fmt.grain_rate());
    const std::uint64_t gpf = inter ? 2 : 1;  // grain'ų kadre
    const int fps_i = static_cast<int>((o.fmt.frame_rate.num + o.fmt.frame_rate.den / 2) / o.fmt.frame_rate.den);  // ~kadrų per 1 s
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n)
        if (!out[n].create(inst, video_def(kRoleDec, o.flow_base + n, o.run_seed, o.fmt), opts, err, o.fmt)) { st.fatal = err; rc = 2; }
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n) {
        auto s = std::make_unique<DecStream>();
        s->idx = static_cast<int>(n);
        ds.push_back(std::move(s));
    }
    for (auto& s : ds) s->th = std::thread([&o, hw, sp = s.get(), &stop] { dec_worker(o, hw, *sp, stop); });

    cudaStream_t stream{};
    // BLOKUOJANTIS stream'as: h264_cuvid dekoduotą kadrą į išvesties buferį kopijuoja asinchroniškai numatytajame
    // (legacy) stream'e; non-blocking stream'as su juo nesisinchronizuotų ir split galėtų skaityti nebaigtą kopiją.
    if (!cpu) cudaStreamCreate(&stream);
    std::vector<AVFrame*> job_src;  // C: planar kadrai CPU split'ui
    std::vector<media::SplitJob> jobs;
    std::vector<std::uint32_t> job_flow;
    std::vector<media::ChromaMode> job_mode;  // interlaced flow: pagal kadro žymę
    std::vector<std::int64_t> anchor(o.flows, INT64_MIN);
    std::vector<AVFrame*> cur(o.flows, nullptr);
    std::vector<std::int64_t> cur_n(o.flows, -1);
    std::vector<int> underrun_run(o.flows, 0);
    std::vector<std::uint64_t> first_play(o.flows, 0);
    const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    const double warm = warm_tai_after(o.warmup_s);
    std::uint64_t i = mxlGetCurrentIndex(&rate) + 1;
    if (inter && (i & 1)) ++i;  // pradėti nuo viršutinio lauko
    ClockWatch cw;
    cw.stepped();

    while (!rc && !stop.load() && now_tai() < end_tai) {
        // Jei teks laukti ilgiau nei 2 laukus — laikrodis šoko atgal: žymim PRIEŠ laukimą, kad darbininkų eilių
        // perpildymas laukimo metu būtų priskirtas šuoliui
        if (mxlIndexToTimestamp(&rate, i) > now_tai() + 2 * (mxlIndexToTimestamp(&rate, 1) - mxlIndexToTimestamp(&rate, 0)))
            g_clock_event_mono = mono_ns();
        if (wait_until(rate, i, o.read_offset_us * 1000, stop)) {
            st.proc["clock_backward"]++;
            g_clock_event_mono = mono_ns();
        }
        if (stop.load()) break;
        const int p = inter ? static_cast<int>(i & 1) : 0;  // progresyviam kiekvienas grain'as — naujas kadras
        const std::uint64_t t_wake = now_tai();
        jobs.clear();
        job_flow.clear();
        job_src.clear();
        job_mode.clear();
        int pitch_y = -1, pitch_uv = -1;
        for (std::uint32_t n = 0; n < o.flows; ++n) {
            DecStream& s = *ds[n];
            if (p == 0) {  // naujas kadras
                if (cur[n]) { av_frame_free(&cur[n]); }
                cur_n[n] = -1;
                // Atkūrimas ATVYKIMO tvarka (ne pagal PTS): po vieną kadrą kiekvienam kadro tick'ui, buferio gylis
                // playout_frames. Tvirta prieš PTS nutrūkimus (kilpos sandūros, realūs srautai). anchor[n] — būsena:
                // INT64_MIN = dar pildomas buferis, 0 = groja.
                std::lock_guard g(s.m);
                // paleidimo pereinamasis laikotarpis (skaičiuojamas atskirai): proceso warmup IR 2 s nuo šio srauto
                // pirmo atkurto kadro (srautų zondavimas/startas nesutampa su proceso startu)
                const std::uint64_t now = now_tai();
                const bool in_warm = now < warm || first_play[n] == 0 || now < first_play[n] + 2'000'000'000ull;
                if (anchor[n] == INT64_MIN) {
                    if (static_cast<int>(s.q.size()) < o.playout_frames) { st.flows[n].inc("waiting_buffer"); continue; }
                    if (first_play[n] == 0) first_play[n] = now;
                    s.playing = true;
                    // paleidimo pliūpsnis (find_stream_info sukaupti paketai): paliekam tik naujausius playout_frames
                    while (static_cast<int>(s.q.size()) > o.playout_frames) {
                        av_frame_free(&s.q.front().f);
                        s.q.pop_front();
                        st.flows[n].inc("startup_discard");
                    }
                    anchor[n] = 0;
                }
                // per daug sukaupta (siuntėjas greitesnis / pliūpsnis) → seniausi išmetami ir užskaitomi
                // dekoderis kadrus išduoda pliūpsniais (B perrikiavimas, UDP gija) — riba +25 kadrų (1 s) sugeria drebėjimą;
                // viršijus — ilgalaikis greičių neatitikimas (tikras perpildymas)
                while (static_cast<int>(s.q.size()) > o.playout_frames + fps_i) {
                    av_frame_free(&s.q.front().f);
                    s.q.pop_front();
                    st.flows[n].inc(in_warm ? "warmup_overrun_frames" : (after_clock_event() ? "clock_overrun_frames" : "overrun_frames"));
                }
                if (!s.q.empty()) {
                    cur[n] = s.q.front().f;
                    cur_n[n] = s.q.front().n;
                    s.q.pop_front();
                    underrun_run[n] = 0;
                } else {
                    st.flows[n].inc(in_warm ? "warmup_underrun_frames" : "underrun_frames");
                    if (++underrun_run[n] >= fps_i) {  // 1 s be kadrų → iš naujo pildyti buferį
                        anchor[n] = INT64_MIN;
                        underrun_run[n] = 0;
                        st.flows[n].inc("rebuffer");
                    }
                }
            }
            if (!cur[n]) {
                const std::uint64_t now = now_tai();
                const bool w = now < warm || first_play[n] == 0 || now < first_play[n] + 2'000'000'000ull;
                st.flows[n].inc(w ? "warmup_underrun_fields" : "underrun_fields");
                continue;
            }
            if (!cpu) {  // GPU batch'as reikalauja vienodo pitch; CPU kiekvienam kadrui naudoja savo linesize
                if (pitch_y < 0) { pitch_y = cur[n]->linesize[0]; pitch_uv = cur[n]->linesize[1]; }
                if (cur[n]->linesize[0] != pitch_y || cur[n]->linesize[1] != pitch_uv) { st.flows[n].inc("pitch_mismatch"); continue; }
            }
            mxlStatus ms{};
            std::uint32_t* dst = out[n].open(i, ms);
            if (!dst) {
                if (ms == MXL_ERR_INVALID_ARG && i <= out[n].last_committed) { st.proc["clock_backward"]++; continue; }
                st.fatal = "OpenGrainSlot " + std::to_string(ms);
                rc = 2;
                break;
            }
            jobs.push_back({cur[n]->data[0], cur[n]->data[1], dst});
            job_flow.push_back(n);
            job_src.push_back(cur[n]);
            job_mode.push_back(inter && !(cur[n]->flags & AV_FRAME_FLAG_INTERLACED) ? media::ChromaMode::FrameSited : media::ChromaMode::FieldSited);
        }
        if (rc) break;
        cudaError_t ce = cudaSuccess;
        const int field = inter ? p : -1;
        if (cpu) {
            pool.run(static_cast<int>(jobs.size()), [&](int k) {
                const AVFrame* f = job_src[k];
                media::cpu::split_yuv420p(f->data[0], f->data[1], f->data[2], f->linesize[0], f->linesize[1], dim, field,
                                          inter ? job_mode[k] : media::ChromaMode::FrameSited, jobs[k].dst);
            });
        } else if (!jobs.empty()) {
            if (!inter) {
                ce = media::split_nv12_progressive(jobs.data(), static_cast<int>(jobs.size()), dim, pitch_y, pitch_uv, stream);
            } else {
                // vienas paleidimas kiekvienam chromos režimui (dažniausiai visi kadrai to paties tipo — vienas paleidimas)
                for (const media::ChromaMode m : {media::ChromaMode::FieldSited, media::ChromaMode::FrameSited}) {
                    std::vector<media::SplitJob> part;
                    for (std::size_t k = 0; k < jobs.size(); ++k)
                        if (job_mode[k] == m) part.push_back(jobs[k]);
                    if (!part.empty() && !ce) ce = media::split_nv12(part.data(), static_cast<int>(part.size()), dim, p, m, pitch_y, pitch_uv, stream);
                }
            }
        }
        if (!cpu && !ce) ce = cudaStreamSynchronize(stream);
        if (ce) { st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce); rc = 2; break; }
        for (std::uint32_t n : job_flow) {
            if (out[n].commit() != MXL_STATUS_OK) { st.fatal = "CommitGrain"; rc = 2; break; }
            st.flows[n].inc("fields_ok");
        }
        const std::uint64_t t_done = now_tai();
        if (t_done >= warm) {
            st.add_hist("split_ms", t_done - t_wake);
            st.add_hist("out_lateness", t_done - mxlIndexToTimestamp(&rate, i));
        }
        if (st.first_index == ~0ull) st.first_index = i;
        st.last_index = i;
        ++i;
        const bool clk = cw.stepped();
        if (clk) g_clock_event_mono = mono_ns();
        const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
        if (now_idx > i) {
            for (auto& f : st.flows) f.inc(clk ? "clock_skips" : (t_done < warm ? "warmup_skips" : "late_skips"), now_idx - i);
            i = inter ? now_idx + (now_idx & 1) : now_idx;  // interlaced: tęsti nuo viršutinio lauko
        }
    }
    stop = true;  // darbininkai baigia (UDP read su timeout)
    for (auto& s : ds)
        if (s->th.joinable()) s->th.join();
    st.proc["clock_steps"] = cw.steps;
    st.proc["clock_back_ms"] = cw.back_ns / 1'000'000ull;
    for (std::uint32_t n = 0; n < ds.size(); ++n) {
        DecStream& s = *ds[n];
        auto& f = st.flows[n];
        f.inc("frames_decoded", s.frames_decoded);
        f.inc("pkts", s.pkts);
        f.inc("pkts_corrupt", s.pkts_corrupt);
        f.inc("decode_errors", s.decode_errors);
        f.inc("flag_mismatch", s.flag_mismatch);
        f.inc("pts_gap_frames", s.pts_gap_frames);
        f.inc("overflow_drops", s.overflow_drops);
        f.inc("send_eagain", s.send_eagain);
        f.inc("clock_overflow_drops", s.clock_overflow_drops);
        f.inc("startup_overflow_drops", s.startup_overflow_drops);
        // Laikrodžio domenų neatitikimas: siuntėjai tempą skaičiuoja monotoniniu laikrodžiu, atkūrimas — TAI.
        // Kiek TAI šoko atgal (WSL2 koreguoja ~1,3 s kas ~31 s), tiek kadrų atkūrimas privalo išmesti — tokį
        // overrun kiekį (+1 kadras per šuolį) priskiriam aplinkai; perteklius lieka pajėgumo praradimu.
        const std::uint64_t frame_ns = static_cast<std::uint64_t>(mxlIndexToTimestamp(&rate, gpf) - mxlIndexToTimestamp(&rate, 0));
        std::uint64_t allow = cw.back_ns / frame_ns + cw.steps;
        for (const char* k : {"overrun_frames", "overflow_drops"}) {  // ta pati tolerancija abiem eilėms kartu
            const std::uint64_t ov = f.c[k];
            const std::uint64_t mv = std::min(ov, allow);
            allow -= mv;
            f.c[k] = ov - mv;
            if (!f.c[k]) f.c.erase(k);
            f.inc("clock_drift_dropped_frames", mv);
        }
        if (!s.opened && !rc) { st.fatal = "srautas " + std::to_string(n) + ": " + s.error; rc = 2; }
        for (auto& q : s.q) av_frame_free(&q.f);
        if (cur[n]) av_frame_free(&cur[n]);
    }
    if (!cpu) cudaStreamDestroy(stream);
    for (auto& v : out) v.release(inst);
    mxlDestroyInstance(inst);
    av_buffer_unref(&hw);
    if (!rc)
        for (auto& f : st.flows)
            // pts_gap_frames — informacinis (siuntėjo kilpos sandūros); praradimai — tik tai, kas nepasiekė išėjimo
            for (const char* k : {"overrun_frames", "underrun_fields", "late_skips", "pkts_corrupt", "decode_errors",
                                  "overflow_drops", "pitch_mismatch"})
                if (f.c.count(k) && f.c[k]) rc = 4;
    return finish(rc);
}

// clip: `frames` kadrų iš CPU etalono į stdout raw NV12 (ffmpeg kodavimui). Interlaced: gen_field laukai → weave;
// progresyvus: gen_field kadras (indeksas = first_field + k) → progresyvus weave.
int run_clip(const Opts& o, std::uint32_t frames, std::uint64_t first_field) {
    const media::Dim d{o.fmt.raster.w, o.fmt.raster.h};
    const bool inter = o.fmt.raster.interlaced;
    const std::size_t ysz = static_cast<std::size_t>(d.w) * d.h;
    std::vector<std::uint32_t> top(media::grain_words(d, inter)), bot(top.size());
    std::vector<std::uint8_t> nv12(ysz * 3 / 2);
    const media::GenParams g{o.run_seed, o.flow_base, o.complexity};
    for (std::uint32_t k = 0; k < frames; ++k) {
        const std::uint64_t i = first_field + (inter ? 2ull : 1ull) * k;
        media::ref::gen_field(g, i, d, inter, top.data());
        if (inter) media::ref::gen_field(g, i + 1, d, inter, bot.data());
        media::ref::weave_nv12(top.data(), bot.data(), nv12.data(), nv12.data() + ysz, d.w, d.w, d, inter);
        if (std::fwrite(nv12.data(), 1, nv12.size(), stdout) != nv12.size()) return 2;
    }
    return 0;
}

// ref-crc: raw NV12 (programinio dekoderio išvestis) → kiekvieno grain'o v210 CRC32C + barkodas (JSON į stdout).
// Interlaced: du laukai kadrui (lauko chroma); progresyvus: vienas kadro grain'as (kadro chroma).
int run_ref_crc(const Opts& o, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 1;
    const media::Dim d{o.fmt.raster.w, o.fmt.raster.h};
    const bool inter = o.fmt.raster.interlaced;
    const std::size_t ysz = static_cast<std::size_t>(d.w) * d.h;
    const int words = media::v210_words_per_line(d.w);
    std::vector<std::uint8_t> nv12(ysz * 3 / 2);
    std::vector<std::uint32_t> field(media::grain_words(d, inter));
    std::printf("{\"fields\":[");
    bool first = true;
    while (std::fread(nv12.data(), 1, nv12.size(), f) == nv12.size()) {
        for (int p = 0; p < (inter ? 2 : 1); ++p) {
            if (inter) media::ref::split_nv12(nv12.data(), nv12.data() + ysz, d.w, d.w, d, p, media::ChromaMode::FieldSited, field.data());
            else media::ref::split_nv12_progressive(nv12.data(), nv12.data() + ysz, d.w, d.w, d, field.data());
            const std::uint32_t crc = crc32c(reinterpret_cast<const std::uint8_t*>(field.data()), field.size() * 4);
            const std::uint32_t mk = media::ref::decode_marker(field.data(), words);
            std::printf("%s[%u,%u]", first ? "" : ",", mk, crc);
            first = false;
        }
    }
    std::printf("]}\n");
    std::fclose(f);
    return 0;
}

// split-raw: raw NV12 kadrai → v210 kadrai (eilutės iš eilės) nepriklausomu etalonu, chromos PSNR įrankiui
// (tests/perf/chroma_psnr.py). frame_chroma — kadro chroma (progresyvus kelias); kitaip kadras skaidomas į du laukus su
// LAUKO chroma ir laukai vėl supinami eilutėmis (taip progresyvų turinį apdorotų senas interlaced kelias).
int run_split_raw(const Opts& o, const std::string& in, const std::string& out, bool frame_chroma) {
    const media::Dim d{o.fmt.raster.w, o.fmt.raster.h};
    if (!media::dim_ok(d, !frame_chroma)) { std::fprintf(stderr, "split-raw: netinkami matmenys\n"); return 1; }
    FILE* fi = std::fopen(in.c_str(), "rb");
    FILE* fo = std::fopen(out.c_str(), "wb");
    if (!fi || !fo) {
        if (fi) std::fclose(fi);
        if (fo) std::fclose(fo);
        std::fprintf(stderr, "split-raw: nepavyko atidaryti failų\n");
        return 1;
    }
    const std::size_t ysz = static_cast<std::size_t>(d.w) * d.h;
    const int words = media::v210_words_per_line(d.w);
    std::vector<std::uint8_t> nv12(ysz * 3 / 2);
    std::vector<std::uint32_t> frame(media::grain_words(d, false)), field(media::grain_words(d, true));
    int rc = 0;
    while (std::fread(nv12.data(), 1, nv12.size(), fi) == nv12.size()) {
        if (frame_chroma) {
            media::ref::split_nv12_progressive(nv12.data(), nv12.data() + ysz, d.w, d.w, d, frame.data());
        } else {
            for (int p = 0; p < 2; ++p) {
                media::ref::split_nv12(nv12.data(), nv12.data() + ysz, d.w, d.w, d, p, media::ChromaMode::FieldSited, field.data());
                for (int j = 0; j < d.h / 2; ++j)
                    std::memcpy(frame.data() + static_cast<std::size_t>(2 * j + p) * words, field.data() + static_cast<std::size_t>(j) * words,
                                static_cast<std::size_t>(words) * 4);
            }
        }
        if (std::fwrite(frame.data(), 4, frame.size(), fo) != frame.size()) { rc = 2; break; }
    }
    std::fclose(fi);
    if (std::fclose(fo) != 0) rc = 2;
    return rc;
}

}  // namespace mxlv::func
