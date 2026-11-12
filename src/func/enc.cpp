// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// enc — F-ENC (architektūra GPU funkcijų §5.3): N × MXL (+ audio) → H.264 7,5 Mbit/s + AAC 192k → MPEG-TS → UDP.
// --format: interlaced (1080i50 / 59,94) — du laukai suaudžiami į kadrą; progresyvus — vienas kadro grain'as = kadras
// (progresyvus weave, chroma (c[2k] + c[2k+1] + 4) >> 3), koderis be interlaced vėliavų.
//  (a) --encoder nvenc: du laukai suaudžiami GPU tiesiai į CUDA AVFrame (be host kopijos) → h264_nvenc.
//      NVENC šioje kortoje neturi lauko kodavimo (field_encoding=0, išmatuota) → srautas formaliai progresyvus 1080p25
//      su interlaced turiniu. Tinka talpos matavimui ir vidinei kontribucijai, NE 1080i50 emisijai.
//  (b) --encoder x264: tas pats weave → device staging → vienas D2H į pinned buferį → libx264 tikras interlaced (MBAFF, tff).
// Gijos: 1 tick gija (MXL + batch kernel'is) + po vieną darbininką srautui (kodavimas, AAC, mux, UDP). Darbininkai MXL
// nekviečia. Eilė ribota: pilna → kadras atmetamas ir SKAIČIUOJAMAS (niekas nedingsta tyliai).
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}
#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/index_math.hpp"
#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "func/engine_glue.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/cpu/cpu_media.hpp"
#include "media/media_kernels.hpp"
#include "runtime/input_slot.hpp"
#include "timing/audio_timeline.hpp"

namespace mxlv::func {
namespace {

constexpr int kQueueMax = 4;
// Vien garso elementų (kai kadro video atmestas, o jo garsas vis tiek koduojamas) atsarga virš kQueueMax: jie maži (~15 KB),
// bet eilė vis tiek ribota — užstrigus darbininkui garsas praleidžiamas, o spraga užpildoma tyla (AudioTimeline).
constexpr int kQueueAudioOnlyMax = 2 * kQueueMax;
// Ilgiausia spraga, užpildoma tyla (mėginiais, 1 s): apima atmestus kadrus ir trumpus šuolius; ilgesnė — perinkaravimas.
constexpr std::int64_t kMaxSilenceSamples = 48000;

std::string averr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(e, b, sizeof b);
    return b;
}

struct Item {
    AVFrame* video = nullptr;  // CUDA (nvenc) arba NV12 sistemos atmintyje (x264); pts = F; nullptr — tik garsas
    std::vector<float> l, r;   // kadro mėginiai [sample_begin(kadrų dažnis, F), sample_begin(…, F + 1))
    std::int64_t frame = 0;
    std::uint64_t t_ready = 0;  // TAI, kada kadras paruoštas (latencijai)
    bool audio_ok = false;
};

// Vieno srauto koderis + mux. Valdo darbininko gija.
struct Stream {
    int n = 0;
    std::mutex m;
    std::condition_variable cv;
    std::deque<Item> q;
    bool done = false;
    std::thread th;
    std::string error;
    // statistika (rašo darbininkas; skaitoma po join)
    std::uint64_t frames_encoded = 0, packets_video = 0, packets_audio = 0, bytes_video = 0, mux_errors = 0, encode_errors = 0;
    Histogram encode_ms;
    // Garso laiko juosta (PTS iš kadro indekso): srauto būsena, ne funkcijos static — keli srautai vienoje gijoje nesusimaišo.
    mxlv::timing::AudioTimeline audio_tl;
    // x264 staging
    std::uint8_t* d_stage = nullptr;  // device NV12 pitch = plotis
    std::vector<std::uint8_t*> pinned;  // laisvi host buferiai
    std::mutex pool_m;
    bool pinned_fallback = false;
};

struct Ctx {
    const Opts* o = nullptr;
    AVBufferRef* hw = nullptr;
    std::vector<AVBufferRef*> frames;  // per srautą (nvenc)
    int w = 1920, h = 1080;
    std::size_t nv12_bytes() const { return static_cast<std::size_t>(w) * h * 3 / 2; }
};

// H.264 lygis pagal makrobloko greitį: ≤ 1080p30 — 4.0, ≤ 1080p60 — 4.2, UHD — 5.1 (≤ 2160p30) / 5.2.
const char* h264_level(const vramflow::VideoFormat& f) {
    const double fps = static_cast<double>(f.frame_rate.num) / static_cast<double>(f.frame_rate.den);
    if (f.raster.h > 1088) return fps > 31 ? "5.2" : "5.1";
    return fps > 31 ? "4.2" : "4.0";
}

void return_pinned(void* opaque, std::uint8_t* data) {
    auto* s = static_cast<Stream*>(opaque);
    std::lock_guard g(s->pool_m);
    s->pinned.push_back(data);
}

AVFrame* wrap_pinned(const Ctx& cx, Stream& s, std::uint8_t* buf) {
    AVFrame* f = av_frame_alloc();
    f->format = AV_PIX_FMT_NV12;
    f->width = cx.w;
    f->height = cx.h;
    f->buf[0] = av_buffer_create(buf, cx.nv12_bytes(), return_pinned, &s, 0);
    f->data[0] = buf;
    f->data[1] = buf + static_cast<std::size_t>(cx.w) * cx.h;
    f->linesize[0] = cx.w;
    f->linesize[1] = cx.w;
    return f;
}

// Vienas AAC kadras iš FIFO su nurodytu PTS (mėginiais, 1/48000) → koderis → mux.
void encode_audio(AVCodecContext* ac, AVAudioFifo* fifo, AVFormatContext* oc, AVPacket* pkt, Stream& s, std::int64_t pts) {
    AVFrame* af = av_frame_alloc();
    af->nb_samples = ac->frame_size;
    af->format = AV_SAMPLE_FMT_FLTP;
    av_channel_layout_copy(&af->ch_layout, &ac->ch_layout);
    af->sample_rate = 48000;
    av_frame_get_buffer(af, 0);
    av_audio_fifo_read(fifo, reinterpret_cast<void**>(af->data), ac->frame_size);
    af->pts = pts;
    if (avcodec_send_frame(ac, af) >= 0)
        while (avcodec_receive_packet(ac, pkt) == 0) {
            ++s.packets_audio;
            pkt->stream_index = 1;
            av_packet_rescale_ts(pkt, ac->time_base, oc->streams[1]->time_base);
            if (av_interleaved_write_frame(oc, pkt) < 0) ++s.mux_errors;
        }
    av_frame_free(&af);
}

// n tylos mėginių į FIFO (abu kanalai).
void write_silence(AVAudioFifo* fifo, std::int64_t n) {
    static const std::vector<float> zeros(4800, 0.f);  // tik skaitoma
    while (n > 0) {
        const int k = static_cast<int>(std::min<std::int64_t>(n, static_cast<std::int64_t>(zeros.size())));
        float* z = const_cast<float*>(zeros.data());  // av_audio_fifo_write neketina rašyti į šaltinį
        float* planes[2] = {z, z};
        av_audio_fifo_write(fifo, reinterpret_cast<void**>(planes), k);
        n -= k;
    }
}

void worker(Ctx& cx, Stream& s) {
    const Opts& o = *cx.o;
    const bool nvenc = o.encoder == "nvenc";
    const vramflow::VideoFormat& fmt = o.fmt;
    const bool inter = fmt.raster.interlaced;
    AVCodecContext* vc = nullptr;
    AVCodecContext* ac = nullptr;
    AVFormatContext* oc = nullptr;
    AVAudioFifo* fifo = nullptr;
    AVPacket* pkt = av_packet_alloc();
    auto fail = [&](const std::string& e) {
        s.error = e;
        std::fprintf(stderr, "[enc %d] %s\n", s.n, e.c_str());
    };
    int e = 0;
    // --- video koderis
    const AVCodec* vcodec = avcodec_find_encoder_by_name(nvenc ? "h264_nvenc" : "libx264");
    if (!vcodec) { fail("koderis nerastas"); goto out; }
    vc = avcodec_alloc_context3(vcodec);
    vc->width = cx.w;
    vc->height = cx.h;
    vc->time_base = {static_cast<int>(fmt.frame_rate.den), static_cast<int>(fmt.frame_rate.num)};  // pts = kadro numeris
    vc->framerate = {static_cast<int>(fmt.frame_rate.num), static_cast<int>(fmt.frame_rate.den)};
    vc->sample_aspect_ratio = {1, 1};
    vc->bit_rate = o.vbitrate;
    vc->rc_max_rate = o.vbitrate;
    vc->rc_buffer_size = static_cast<int>(o.vbitrate);
    vc->gop_size = static_cast<int>((fmt.frame_rate.num + fmt.frame_rate.den / 2) / fmt.frame_rate.den);  // ~1 s
    vc->color_primaries = AVCOL_PRI_BT709;
    vc->color_trc = AVCOL_TRC_BT709;
    vc->colorspace = AVCOL_SPC_BT709;
    vc->color_range = AVCOL_RANGE_MPEG;
    vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER * 0;  // TS: antraštės sraute
    {
        AVDictionary* d = nullptr;
        if (nvenc) {
            vc->pix_fmt = AV_PIX_FMT_CUDA;
            vc->hw_frames_ctx = av_buffer_ref(cx.frames[s.n]);
            const bool ll = o.tune == "ll";
            vc->max_b_frames = ll ? 0 : 2;
            av_dict_set(&d, "preset", o.preset.c_str(), 0);
            av_dict_set(&d, "tune", o.tune.c_str(), 0);
            av_dict_set(&d, "rc", "cbr", 0);
            av_dict_set(&d, "profile", "high", 0);
            av_dict_set(&d, "level", h264_level(fmt), 0);
            av_dict_set(&d, "aud", "1", 0);
            av_dict_set(&d, "strict_gop", "1", 0);
            av_dict_set(&d, "forced-idr", "1", 0);
            av_dict_set(&d, "rc-lookahead", "0", 0);
            av_dict_set(&d, "surfaces", "8", 0);
            if (ll) av_dict_set(&d, "zerolatency", "1", 0);
            else av_dict_set(&d, "b_ref_mode", "middle", 0);
        } else {
            vc->pix_fmt = AV_PIX_FMT_NV12;
            if (inter) vc->flags |= AV_CODEC_FLAG_INTERLACED_DCT | AV_CODEC_FLAG_INTERLACED_ME;
            vc->max_b_frames = 2;
            vc->thread_count = o.x264_threads;
            vc->thread_type = FF_THREAD_FRAME;
            av_dict_set(&d, "preset", o.x264_preset.c_str(), 0);
            av_dict_set(&d, "profile", "high", 0);
            av_dict_set(&d, "level", h264_level(fmt), 0);
            av_dict_set(&d, "x264-params", inter ? "tff=1:nal-hrd=cbr:force-cfr=1:scenecut=0:aud=1" : "nal-hrd=cbr:force-cfr=1:scenecut=0:aud=1", 0);
        }
        e = avcodec_open2(vc, vcodec, &d);
        av_dict_free(&d);
        if (e < 0) { fail("video avcodec_open2: " + averr(e)); goto out; }
    }
    // --- audio koderis
    if (o.audio) {
        const AVCodec* acodec = avcodec_find_encoder_by_name("aac");
        ac = avcodec_alloc_context3(acodec);
        ac->sample_fmt = AV_SAMPLE_FMT_FLTP;
        ac->sample_rate = 48000;
        av_channel_layout_default(&ac->ch_layout, 2);
        ac->bit_rate = o.abitrate;
        ac->time_base = {1, 48000};
        if ((e = avcodec_open2(ac, acodec, nullptr)) < 0) { fail("aac avcodec_open2: " + averr(e)); goto out; }
        fifo = av_audio_fifo_alloc(AV_SAMPLE_FMT_FLTP, 2, 48000);
        s.audio_tl = mxlv::timing::AudioTimeline(ac->frame_size, kMaxSilenceSamples);
    }
    // --- mux
    {
        char url[256];
        std::snprintf(url, sizeof url, "udp://127.0.0.1:%d?pkt_size=1316&buffer_size=4194304", o.port_base + s.n);
        if ((e = avformat_alloc_output_context2(&oc, nullptr, "mpegts", url)) < 0) { fail("mux alloc: " + averr(e)); goto out; }
        AVStream* vs = avformat_new_stream(oc, nullptr);
        avcodec_parameters_from_context(vs->codecpar, vc);
        vs->time_base = {1, 90000};
        if (ac) {
            AVStream* as = avformat_new_stream(oc, nullptr);
            avcodec_parameters_from_context(as->codecpar, ac);
            as->time_base = {1, 90000};
        }
        if ((e = avio_open2(&oc->pb, url, AVIO_FLAG_WRITE, nullptr, nullptr)) < 0) { fail("avio_open2: " + averr(e)); goto out; }
        AVDictionary* md = nullptr;
        av_dict_set_int(&md, "muxrate", o.muxrate, 0);
        av_dict_set(&md, "pcr_period", "20", 0);
        av_dict_set(&md, "mpegts_flags", "+system_b", 0);
        // PTS = TAI·90 kHz (iš MXL indekso). Be copyts muxer'is prideda max_delay prie PTS/DTS.
        av_dict_set(&md, "mpegts_copyts", "1", 0);
        oc->max_delay = 500000;
        if ((e = avformat_write_header(oc, &md)) < 0) { fail("write_header: " + averr(e)); av_dict_free(&md); goto out; }
        av_dict_free(&md);
    }
    // --- ciklas
    for (;;) {
        Item it;
        {
            std::unique_lock lk(s.m);
            s.cv.wait(lk, [&] { return !s.q.empty() || s.done; });
            if (s.q.empty()) break;
            it = std::move(s.q.front());
            s.q.pop_front();
        }
        if (it.video) {
            it.video->pts = it.frame;
            const std::uint64_t t0 = now_tai();
            e = avcodec_send_frame(vc, it.video);
            av_frame_free(&it.video);
            if (e < 0) ++s.encode_errors;
            else {
                while ((e = avcodec_receive_packet(vc, pkt)) == 0) {
                    ++s.packets_video;
                    s.bytes_video += pkt->size;
                    pkt->stream_index = 0;
                    av_packet_rescale_ts(pkt, vc->time_base, oc->streams[0]->time_base);
                    if (av_interleaved_write_frame(oc, pkt) < 0) ++s.mux_errors;
                }
                s.encode_ms.add(now_tai() - t0);
                ++s.frames_encoded;
            }
        }
        if (ac && !it.l.empty()) {
            // PTS iš kadro indekso (ne skaitiklis): kadro F mėginiai — [sample_begin(F), sample_begin(F + 1)). Spraga po
            // atmestų / praleistų kadrų — tyla arba perinkaravimas (timing/audio_timeline.hpp), todėl A/V poslinkis nesikaupia.
            const auto begin = static_cast<std::int64_t>(vramflow::sample_begin(o.fmt.frame_rate, static_cast<std::uint64_t>(it.frame)));
            const mxlv::timing::AudioPlan p = s.audio_tl.submit(begin, static_cast<std::int64_t>(it.l.size()));
            if (p.flush_frames) {
                write_silence(fifo, p.flush_pad);
                for (std::int64_t k = 0; k < p.flush_frames; ++k) encode_audio(ac, fifo, oc, pkt, s, p.flush_pts + k * ac->frame_size);
            }
            write_silence(fifo, p.silence);
            if (p.skip < static_cast<std::int64_t>(it.l.size())) {
                float* planes[2] = {it.l.data() + p.skip, it.r.data() + p.skip};
                av_audio_fifo_write(fifo, reinterpret_cast<void**>(planes), static_cast<int>(static_cast<std::int64_t>(it.l.size()) - p.skip));
            }
            while (s.audio_tl.ready()) encode_audio(ac, fifo, oc, pkt, s, s.audio_tl.take());
        }
    }
    avcodec_send_frame(vc, nullptr);
    while (avcodec_receive_packet(vc, pkt) == 0) {
        pkt->stream_index = 0;
        av_packet_rescale_ts(pkt, vc->time_base, oc->streams[0]->time_base);
        av_interleaved_write_frame(oc, pkt);
    }
    if (ac) {  // AAC flush
        avcodec_send_frame(ac, nullptr);
        while (avcodec_receive_packet(ac, pkt) == 0) {
            pkt->stream_index = 1;
            av_packet_rescale_ts(pkt, ac->time_base, oc->streams[1]->time_base);
            av_interleaved_write_frame(oc, pkt);
        }
    }
    av_write_trailer(oc);
out:
    {  // likusius kadrus atlaisvinti
        std::lock_guard lk(s.m);
        for (auto& it : s.q) av_frame_free(&it.video);
        s.q.clear();
    }
    if (oc) {
        if (oc->pb) avio_closep(&oc->pb);
        avformat_free_context(oc);
    }
    if (fifo) av_audio_fifo_free(fifo);
    avcodec_free_context(&vc);
    avcodec_free_context(&ac);
    av_packet_free(&pkt);
}

}  // namespace

int run_enc(const Opts& o) {
    Stats st;
    st.flows.resize(o.flows);
    auto& stop = stop_flag();
    const bool nvenc = o.encoder == "nvenc";
    Ctx cx;
    cx.o = &o;
    cx.w = o.fmt.raster.w;
    cx.h = o.fmt.raster.h;
    const bool inter = o.fmt.raster.interlaced;
    const media::Dim dim{cx.w, cx.h};
    const mxlRational rate = mxl_rate(o.fmt.grain_rate());
    const std::uint64_t gpf = inter ? 2 : 1;  // grain'ų kadre
    const std::size_t nv12 = cx.nv12_bytes(), ysz = static_cast<std::size_t>(cx.w) * cx.h;
    std::vector<std::unique_ptr<Stream>> streams;
    auto finish = [&](int rc) {
        st.exit_code = rc;
        std::string extra = "\"encoder\":\"" + o.encoder + "\",\"preset\":\"" + (nvenc ? o.preset + "/" + o.tune : o.x264_preset) + "\",\"streams\":[";
        for (std::size_t k = 0; k < streams.size(); ++k) {
            const Stream& s = *streams[k];
            char b[512];
            std::snprintf(b, sizeof b,
                          "%s{\"frames_encoded\":%llu,\"packets_video\":%llu,\"packets_audio\":%llu,\"bytes_video\":%llu,"
                          "\"mux_errors\":%llu,\"encode_errors\":%llu,\"encode_p50_ms\":%.3f,\"encode_p99_ms\":%.3f,\"error\":\"%s\"}",
                          k ? "," : "", (unsigned long long)s.frames_encoded, (unsigned long long)s.packets_video,
                          (unsigned long long)s.packets_audio, (unsigned long long)s.bytes_video, (unsigned long long)s.mux_errors,
                          (unsigned long long)s.encode_errors, s.encode_ms.quantile(0.5) / 1e6, s.encode_ms.quantile(0.99) / 1e6,
                          s.error.c_str());
            extra += b;
        }
        extra += "]";
        write_summary(o, "enc", st, extra);
        std::fprintf(stderr, "[enc] baigta, exit %d %s\n", rc, st.fatal.c_str());
        return rc;
    };
    // Variantas C (--impl cpu): host flow → CPU weave → libx264; CUDA ir FFmpeg hw įrenginys nekuriami.
    const bool cpu = o.impl == "cpu";
    if (cpu && nvenc) { st.fatal = "--impl cpu reikalauja --encoder x264"; return finish(1); }
    ThreadPool pool(cpu ? (o.threads > 0 ? o.threads : static_cast<int>(std::min<unsigned>(o.flows, 4))) : 1);
    // Tvarka svarbi: FFmpeg pirmas sukuria primary kontekstą (CU_CTX_SCHED_BLOCKING_SYNC), CUDA runtime ir MXL
    // plugin'as po to prisijungia prie jau aktyvaus. Atvirkščiai FFmpeg atsisako („incompatible flags").
    if (!cpu) {
        const int dev = cuda_device(o.device_uuid);
        if (dev < 0) { st.fatal = "CUDA įrenginys"; return finish(2); }
        char ord[8];
        std::snprintf(ord, sizeof ord, "%d", dev);
        if (int e = av_hwdevice_ctx_create(&cx.hw, AV_HWDEVICE_TYPE_CUDA, ord, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT); e < 0) {
            st.fatal = "av_hwdevice_ctx_create: " + averr(e);
            return finish(2);
        }
        if (cudaSetDevice(dev) != cudaSuccess) { st.fatal = "cudaSetDevice"; return finish(2); }
        cudaFree(nullptr);
    }
    for (std::uint32_t n = 0; n < o.flows && nvenc; ++n) {
        AVBufferRef* fr = av_hwframe_ctx_alloc(cx.hw);
        auto* fc = reinterpret_cast<AVHWFramesContext*>(fr->data);
        fc->format = AV_PIX_FMT_CUDA;
        fc->sw_format = AV_PIX_FMT_NV12;
        fc->width = cx.w;
        fc->height = cx.h;
        fc->initial_pool_size = 16;  // iš anksto: pirmas cudaMalloc WSL'e trunka dešimtis ms ir vėlintų tick'ą
        if (int e = av_hwframe_ctx_init(fr); e < 0) { st.fatal = "hwframe_ctx_init: " + averr(e); return finish(2); }
        cx.frames.push_back(fr);
    }

    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; return finish(2); }
    // Įėjimai (planas §P6): video per InputSlot (laukimas pagal --input-wait-ms, re-attach); garsas atidaromas iš naujo,
    // kai jo video slot'as prisijungia iš naujo (tas pats gamintojas) arba skaitymas grąžina FLOW_INVALID.
    std::vector<runtime::InputSlot> in;
    in.reserve(o.flows);
    for (std::uint32_t n = 0; n < o.flows; ++n) in.push_back(make_input(o, flow_id(o.in_role, o.flow_base + n), &o.fmt));
    const std::vector<runtime::InputSlot*> in_ptr = slot_ptrs(in);
    std::vector<AudioIn> ain(o.audio ? o.flows : 0);
    std::vector<std::uint64_t> seen_reattach(o.flows, 0);
    std::string err;
    int rc = attach_inputs(in_ptr, inst, o, "enc", st.fatal);  // 4 — neatsirado per ribą; kStoppedWaiting — sustabdyta (exit 0)
    for (std::uint32_t n = 0; n < ain.size() && !rc; ++n)
        if (!ain[n].open(inst, flow_id(kRoleAudio, o.flow_base + n), open_policy(o), stop, err)) {
            if (stop.load()) rc = kStoppedWaiting;
            else { st.fatal = err; rc = kExitInputs; }
        }
    // Garsas — variklio Wanted įėjimas (MxlAudioInput): laukiamas tik iki a(i), vieneto klasės nelemia, variklio gijos
    // niekada nemigdo (anksčiau read_audio(wait) laukdavo iki T + --deadline-ms — vienas srautas be garso vėlindavo visus).
    // Pradinis laukimas — kaip anksčiau (AudioIn::open pagal --input-wait-ms); reader'is perduodamas prievadui.
    const engine::AudioFrames afr{o.fmt.frame_rate, static_cast<std::uint32_t>(gpf)};
    std::vector<engine::MxlAudioInput> ports_audio;
    ports_audio.reserve(ain.size());
    for (std::uint32_t n = 0; n < ain.size(); ++n) {
        ports_audio.emplace_back(inst, flow_id(kRoleAudio, o.flow_base + n), afr, ain[n].r);
        ain[n].r = nullptr;
    }

    cudaStream_t stream{};
    if (!cpu) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n) {
        auto s = std::make_unique<Stream>();
        s->n = static_cast<int>(n);
        if (cpu) {  // C: tik sistemos atmintis
            for (int k = 0; k < kQueueMax + 3; ++k) s->pinned.push_back(static_cast<std::uint8_t*>(std::malloc(nv12)));
            s->pinned_fallback = true;
        } else if (!nvenc) {
            if (cudaMalloc(reinterpret_cast<void**>(&s->d_stage), nv12) != cudaSuccess) { st.fatal = "staging"; rc = 2; break; }
            for (int k = 0; k < kQueueMax + 3; ++k) {
                std::uint8_t* p = nullptr;
                if (cudaHostAlloc(reinterpret_cast<void**>(&p), nv12, cudaHostAllocDefault) != cudaSuccess) {
                    (void)cudaGetLastError();
                    p = static_cast<std::uint8_t*>(std::malloc(nv12));
                    s->pinned_fallback = true;
                }
                s->pinned.push_back(p);
            }
        }
        streams.push_back(std::move(s));
    }
    for (auto& s : streams) s->th = std::thread([&cx, sp = s.get()] { worker(cx, *sp); });
    // palaukti, kol koderiai atsidarys (13-a NVENC sesija turi aiškiai nepavykti)
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    for (auto& s : streams)
        if (!s->error.empty() && !rc) { st.fatal = "srautas " + std::to_string(s->n) + ": " + s->error; rc = 2; }

    std::vector<std::vector<float>> al(o.flows), ar(o.flows);
    const double warm = warm_tai_after(o.warmup_s);
    const std::uint64_t test_drop_from = now_tai() + static_cast<std::uint64_t>(o.test_drop_after_s * 1e9);
    std::uint64_t F = mxlGetCurrentIndex(&rate) / gpf + 1;
    // Kadro (vieneto indeksas i_bot) garsas į item.l / item.r (true — yra). Niekada nelaukia. engine — variklio Wanted
    // įėjimo rezultatas šiam indeksui (nullptr — nesulaukta iki a(i) ir po apdorojimo); be jo (kadras atmestas) — vienas
    // neblokuojantis bandymas. Kadro mėginiai sveikaisiais skaičiais: 1080i50 / 25p — 1920, 50p — 960, 29,97 — 1601 / 1602.
    auto take_audio = [&](std::uint32_t n, engine::Index i_bot, Item& item, const engine::GrainRef* engine_ref) {
        engine::MxlAudioInput& p = ports_audio[n];
        bool have = false;
        if (engine_ref != nullptr) {
            have = engine_ref->ptr != nullptr && p.last() == i_bot;
        } else {
            engine::GrainRef g;
            have = p.probe(i_bot, g) == engine::Probe::Ready;
        }
        const std::uint32_t a_count = afr.count(i_bot);
        item.l = std::move(al[n]);
        item.r = std::move(ar[n]);
        item.l.resize(a_count);
        item.r.resize(a_count);
        if (have) {
            std::copy(p.left(), p.left() + a_count, item.l.begin());
            std::copy(p.right(), p.right() + a_count, item.r.begin());
        }
        al[n].clear();
        ar[n].clear();
        return have;
    };
    // Kadro video atmestas — jo garsas vis tiek koduojamas (radinys 2: kitaip garso laiko juosta trumpėja ir lūpų
    // sinchronas slenka). Garso neperskaičius ar eilei perpildžius — praleidžiama: darbininkas spragą užpildys tyla.
    auto push_audio_only = [&](std::uint32_t n, Item&& item) {
        av_frame_free(&item.video);
        if (item.l.empty()) return;
        Stream& s = *streams[n];
        {
            std::lock_guard g(s.m);
            if (s.q.size() >= kQueueAudioOnlyMax) return;
            s.q.push_back(std::move(item));
        }
        s.cv.notify_one();
        st.flows[n].inc("audio_only_frames");
    };
    auto audio_of_dropped = [&](std::uint32_t n, engine::Index i_bot) {
        if (ports_audio.empty()) return;
        Item item;
        item.frame = static_cast<std::int64_t>(i_bot / gpf);
        if (take_audio(n, i_bot, item, nullptr)) push_audio_only(n, std::move(item));
    };

    // Bendras tick variklis (planas V12): N sink vienetų — video įėjimas (grains = laukų kadre, stride = gpf, tick'as
    // ties apatiniu lauku), išėjimų nėra; process — weave visiems paruoštiems vienu batch'u, end_unit — Item į darbininko
    // eilę (arba tik garsas, kai kadras atmestas). Garsas — Wanted įėjimas (ports_audio), UnitTick.in[gpf].
    struct EncFn final : engine::TickFunction {
        const Opts& o;
        Stats& st;
        const engine::StatsHub& hub;
        const engine::TickEngine* eng = nullptr;
        std::vector<runtime::InputSlot>& in;
        std::vector<engine::MxlAudioInput>& ports_audio;
        std::vector<std::uint64_t>& seen_reattach;
        std::vector<std::unique_ptr<Stream>>& streams;
        Ctx& cx;
        mxlInstance inst;
        bool cpu, nvenc, inter;
        media::Dim dim;
        std::size_t nv12, ysz;
        std::uint64_t gpf;
        ThreadPool& pool;
        cudaStream_t stream;
        double warm;
        std::uint64_t test_drop_from;
        std::uint64_t& F;
        const std::atomic<bool>& stop;
        std::function<bool(std::uint32_t, engine::Index, Item&, const engine::GrainRef*)> take_audio;
        std::function<void(std::uint32_t, Item&&)> push_audio_only;
        std::function<void(std::uint32_t, engine::Index)> audio_of_dropped;
        std::vector<media::WeaveJob> jobs;
        std::vector<std::uint32_t> job_flow;
        std::vector<AVFrame*> frame_of;           // vieneto kadras (nvenc) po process
        std::vector<std::uint8_t*> buf_of;        // vieneto host buferis (x264) po process
        std::vector<char> alloc_failed;
        std::vector<engine::InCounters> seen;
        std::uint64_t t_done = 0;
        TickTimes times;  // weave_ms — tik indeksams, kurių kadrai apdoroti (atskirai kiekvienam indeksui)

        EncFn(const Opts& o_, Stats& st_, const engine::StatsHub& hub_, std::vector<runtime::InputSlot>& in_,
              std::vector<engine::MxlAudioInput>& pa_, std::vector<std::uint64_t>& sr_, std::vector<std::unique_ptr<Stream>>& streams_, Ctx& cx_,
              mxlInstance inst_, bool cpu_, bool nvenc_, bool inter_, media::Dim dim_, std::size_t nv12_, std::size_t ysz_, std::uint64_t gpf_,
              ThreadPool& pool_, cudaStream_t stream_, double warm_, std::uint64_t tdf_, std::uint64_t& F_, const std::atomic<bool>& stop_)
            : o(o_), st(st_), hub(hub_), in(in_), ports_audio(pa_), seen_reattach(sr_), streams(streams_), cx(cx_), inst(inst_),
              cpu(cpu_), nvenc(nvenc_), inter(inter_), dim(dim_), nv12(nv12_), ysz(ysz_), gpf(gpf_), pool(pool_), stream(stream_), warm(warm_),
              test_drop_from(tdf_), F(F_), stop(stop_), frame_of(o_.flows, nullptr), buf_of(o_.flows, nullptr), alloc_failed(o_.flows, 0),
              seen(o_.flows) {
            jobs.reserve(o.flows);
            job_flow.reserve(o.flows);
        }

        void begin_tick(engine::Index i, engine::TickContext&) override {
            if (i % gpf != gpf - 1) return;
            times.wake(i, now_tai());
            F = i / gpf;
            // [§P6] po video re-attach (tas pats gamintojas paleistas iš naujo) — ir garso reader'is: prievadas atjungiamas,
            // variklis jį mato kaip Gone ir prijungia prie naujo flow'o (attach_poll, kas 100 ms)
            for (std::uint32_t n = 0; n < ports_audio.size(); ++n)
                if (in[n].reattached() != seen_reattach[n]) {
                    seen_reattach[n] = in[n].reattached();
                    ports_audio[n].detach();
                }
        }

        bool process(const engine::Batch& b, engine::TickContext& ctx) override {
            F = b.index / gpf;
            times.phase_begin(b.index, now_tai());
            jobs.clear();
            job_flow.clear();
            for (const engine::UnitTick& ut : b.units) {
                const std::uint32_t n = ut.unit;
                frame_of[n] = nullptr;
                buf_of[n] = nullptr;
                alloc_failed[n] = 0;
                const auto* top = static_cast<const std::uint32_t*>(ut.in[0].ptr);
                const auto* bot = static_cast<const std::uint32_t*>(ut.in[gpf - 1].ptr);
                media::WeaveJob j{top, bot, nullptr, nullptr};
                if (nvenc) {
                    AVFrame* f = av_frame_alloc();
                    if (av_hwframe_get_buffer(cx.frames[n], f, 0) < 0) {
                        av_frame_free(&f);
                        alloc_failed[n] = 1;
                        continue;
                    }
                    j.dst_y = f->data[0];
                    j.dst_uv = f->data[1];
                    if (!job_flow.empty() && f->linesize[0] != frame_of[job_flow[0]]->linesize[0]) {
                        av_frame_free(&f);
                        st.fatal = "skirtingi pitch";
                        ctx.fatal(2, st.fatal);
                        return false;
                    }
                    frame_of[n] = f;
                } else if (!cpu) {
                    j.dst_y = streams[n]->d_stage;
                    j.dst_uv = streams[n]->d_stage + ysz;
                }
                jobs.push_back(j);
                job_flow.push_back(n);
            }
            cudaError_t ce = cudaSuccess;
            const int py = nvenc && !job_flow.empty() ? frame_of[job_flow[0]]->linesize[0] : cx.w;
            const int puv = nvenc && !job_flow.empty() ? frame_of[job_flow[0]]->linesize[1] : cx.w;
            if (cpu) {
                // C: buferis iš pool'o, CPU weave tiesiai iš host flow slot'ų (gijų baseine)
                for (std::size_t k = 0; k < jobs.size(); ++k) {
                    Stream& s = *streams[job_flow[k]];
                    std::lock_guard g(s.pool_m);
                    if (s.pinned.empty()) continue;
                    buf_of[job_flow[k]] = s.pinned.back();
                    s.pinned.pop_back();
                }
                pool.run(static_cast<int>(jobs.size()), [&](int k) {
                    std::uint8_t* hb = buf_of[job_flow[k]];
                    if (hb) media::cpu::weave_nv12(jobs[k].top, jobs[k].bottom, hb, hb + ysz, cx.w, cx.w, dim, inter);
                });
            } else if (!jobs.empty()) {
                ce = media::weave_nv12(jobs.data(), static_cast<int>(jobs.size()), dim, inter, py, puv, stream);
            }
            if (!cpu && !ce && !nvenc) {
                for (std::size_t k = 0; k < jobs.size(); ++k) {
                    Stream& s = *streams[job_flow[k]];
                    std::lock_guard g(s.pool_m);
                    if (s.pinned.empty()) continue;  // pool tuščias → kadras bus atmestas (eilė pilna)
                    buf_of[job_flow[k]] = s.pinned.back();
                    s.pinned.pop_back();
                    ce = cudaMemcpyAsync(buf_of[job_flow[k]], s.d_stage, nv12, cudaMemcpyDeviceToHost, stream);
                    if (ce) break;
                }
            }
            if (!cpu && !ce) ce = cudaStreamSynchronize(stream);
            if (ce) {
                st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce);
                ctx.fatal(2, st.fatal);
                return false;
            }
            return true;
        }

        void release_unit(std::uint32_t n) {
            if (frame_of[n]) av_frame_free(&frame_of[n]);
            if (buf_of[n]) {
                std::lock_guard g(streams[n]->pool_m);
                streams[n]->pinned.push_back(buf_of[n]);
                buf_of[n] = nullptr;
            }
        }

        void end_unit(const engine::UnitTick& ut, engine::Outcome oc, engine::TickContext&) override {
            const std::uint32_t n = ut.unit;
            F = ut.index / gpf;
            FlowStats& fs = st.flows[n];
            const engine::InCounters& c = hub.in(eng->in_id(n, 0));
            engine::InCounters& p = seen[n];
            const bool inv = c.invalid > p.invalid, tl = c.too_late > p.too_late, det = c.detaches > p.detaches;
            p = c;
            if (c.stalls && c.stalls > fs.c["stalls"]) fs.c["stalls"] = c.stalls;
            switch (oc) {
                case engine::Outcome::Ready: break;
                case engine::Outcome::Lapped:
                    fs.inc("lapped");
                    release_unit(n);
                    audio_of_dropped(n, ut.index);
                    return;
                case engine::Outcome::GapBlocked: fs.inc("missing_frames"); audio_of_dropped(n, ut.index); return;
                case engine::Outcome::GapClock: fs.inc("clock_skips_frames"); audio_of_dropped(n, ut.index); return;
                case engine::Outcome::Skipped: st.proc["clock_backward"]++; audio_of_dropped(n, ut.index); return;
                case engine::Outcome::OutputError: fs.inc("output_errors"); audio_of_dropped(n, ut.index); return;  // sink: neturėtų būti
                case engine::Outcome::GapNoContent: audio_of_dropped(n, ut.index); return;
                case engine::Outcome::GapLate:
                    fs.inc(det ? "missing_frames" : tl ? "too_late" : inv ? "in_invalid" : "late_frames");
                    audio_of_dropped(n, ut.index);
                    return;
            }
            if (alloc_failed[n]) {
                fs.inc("frame_alloc_errors");
                audio_of_dropped(n, ut.index);
                return;
            }
            AVFrame* f = frame_of[n];
            frame_of[n] = nullptr;
            if (!nvenc) {
                if (!buf_of[n]) {
                    fs.inc("frames_dropped_queue");
                    audio_of_dropped(n, ut.index);
                    return;
                }
                f = wrap_pinned(cx, *streams[n], buf_of[n]);
                buf_of[n] = nullptr;
                if (inter) f->flags |= AV_FRAME_FLAG_INTERLACED | AV_FRAME_FLAG_TOP_FIELD_FIRST;
            }
            Item item;
            item.video = f;
            item.frame = static_cast<std::int64_t>(F);
            item.t_ready = times.first(ut.index);
            if (!ports_audio.empty() && !take_audio(n, ut.index, item, ut.in.size() > gpf ? &ut.in[gpf] : nullptr)) {
                fs.inc("audio_underrun_frames");
                std::fill(item.l.begin(), item.l.end(), 0.f);
                std::fill(item.r.begin(), item.r.end(), 0.f);
            }
            Stream& s = *streams[n];
            {
                std::unique_lock g(s.m);
                const bool test_drop = o.test_drop_every && item.t_ready >= test_drop_from && F % o.test_drop_every == 0;
                if (s.q.size() >= kQueueMax || test_drop) {
                    g.unlock();
                    fs.inc("frames_dropped_queue");
                    push_audio_only(n, std::move(item));  // video atmestas, garsas — ne
                    return;
                }
                s.q.push_back(std::move(item));
            }
            s.cv.notify_one();
            fs.inc("frames_queued");
            t_done = now_tai();
            times.unit_done(ut.index, t_done, true);
            const mxlRational r = mxl_rate(o.fmt.grain_rate());
            fs.note_lateness(t_done, mxlIndexToTimestamp(&r, ut.index), warm);
        }

        void end_tick(engine::Index i, engine::TickContext&) override {
            if (i % gpf != gpf - 1) return;
            const std::uint64_t td = now_tai();
            const TickTimes::Result t = times.end(i, td);
            if (static_cast<double>(td) >= warm) {
                const mxlRational r = mxl_rate(o.fmt.grain_rate());
                if (t.processed) st.add_hist("weave_ms", t.proc);  // be apdorotų kadrų (visi spragos) — nėra ką matuoti
                st.add_hist("frame_lateness", td - mxlIndexToTimestamp(&r, i));
            }
        }
    };
    std::vector<engine::MxlVideoInput> ports_in;
    ports_in.reserve(o.flows);
    for (auto& slot : in) ports_in.emplace_back(slot, inst);
    engine::SystemClock clk;
    engine::StatsHub hub;
    engine::TickEngine eng(engine_config(o, EngineTiming::Isolation, o.read_offset_us * 1000), clk, hub, stop);
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n) {
        engine::UnitSpec u;
        u.name = flow_id(o.in_role, o.flow_base + n);
        u.in.push_back(engine::InputSpec{&ports_in[n], {}, static_cast<std::uint8_t>(gpf), engine::Need::Required, engine::InMode::Exact});
        if (n < ports_audio.size())  // garsas: Wanted — vieneto klasės nelemia, laukiamas tik iki a(i)
            u.in.push_back(engine::InputSpec{&ports_audio[n], {}, 1, engine::Need::Wanted, engine::InMode::Exact});
        u.stride = static_cast<std::uint32_t>(gpf);
        u.residue = static_cast<std::uint32_t>(gpf - 1);
        std::uint32_t id = 0;
        if (!eng.add_unit(u, id, err)) { st.fatal = "add_unit: " + err; rc = 2; }
    }
    EncFn fn(o, st, hub, in, ports_audio, seen_reattach, streams, cx, inst, cpu, nvenc, inter, dim, nv12, ysz, gpf, pool, stream,
             warm, test_drop_from, F, stop);
    fn.eng = &eng;
    fn.take_audio = take_audio;
    fn.push_audio_only = push_audio_only;
    fn.audio_of_dropped = audio_of_dropped;
    bool engine_loss = false;
    if (!rc) {
        const engine::RunResult res = eng.run(fn);
        engine_loss = res.any_loss;
        if (res.reason == engine::EndReason::Fatal) {
            if (st.fatal.empty()) st.fatal = res.fatal;
            rc = res.exit_code ? res.exit_code : 2;
        } else if (res.reason == engine::EndReason::FormatChange) {
            st.fatal = res.fatal;
            rc = kExitFormatChange;
        }
        to_summary(hub, eng, res, o.flows, SkipScope::PerFlowFrames, st);
        if (res.first_index != engine::kNoIndex) {
            st.first_index = res.first_index - (res.first_index % gpf);
            st.last_index = res.last_index;
        }
    }
    for (auto& s : streams) {
        {
            std::lock_guard g(s->m);
            s->done = true;
        }
        s->cv.notify_one();
    }
    for (auto& s : streams)
        if (s->th.joinable()) s->th.join();
    for (std::uint32_t n = 0; n < streams.size(); ++n) {
        Stream& s = *streams[n];
        st.flows[n].inc("frames_encoded", s.frames_encoded);
        st.flows[n].inc("packets_audio", s.packets_audio);
        if (o.audio) {  // garso laiko juosta: tyla spragoms, perinkaravimai, išmesti persidengę mėginiai (ne praradimai)
            st.flows[n].inc("audio_silence_samples", s.audio_tl.silence_samples());
            st.flows[n].inc("audio_reanchors", s.audio_tl.reanchors());
            st.flows[n].inc("audio_skipped_samples", s.audio_tl.skipped_samples());
        }
        st.flows[n].inc("encode_errors", s.encode_errors);
        st.flows[n].inc("mux_errors", s.mux_errors);
        st.hist["encode_ms"].merge(s.encode_ms);
        if (!s.error.empty() && !rc) { st.fatal = "srautas " + std::to_string(n) + ": " + s.error; rc = 2; }
        if (s.d_stage) cudaFree(s.d_stage);
        for (std::uint8_t* p : s.pinned) {
            if (s.pinned_fallback) std::free(p);
            else cudaFreeHost(p);
        }
    }
    if (!cpu) cudaStreamDestroy(stream);
    for (std::uint32_t n = 0; n < o.flows; ++n) {
        st.flows[n].c["reattached"] = in[n].reattached();
        in[n].release();
    }
    for (std::uint32_t n = 0; n < ports_audio.size(); ++n) {
        if (ports_audio[n].attaches() > 0) st.flows[n].inc("audio_reattached", ports_audio[n].attaches());  // prisijungimai po starto
        ports_audio[n].detach();
    }
    for (auto& a : ain) a.release(inst);
    mxlDestroyInstance(inst);
    for (auto* fr : cx.frames) av_buffer_unref(&fr);
    av_buffer_unref(&cx.hw);
    if (!rc)
        for (auto& f : st.flows)
            for (const char* k : {"late_frames", "missing_frames", "lapped", "too_late", "in_invalid", "frames_dropped_queue", "late_skips_frames",
                                  "audio_underrun_frames", "encode_errors", "mux_errors", "frame_alloc_errors"})
                if (f.c.count(k) && f.c[k]) rc = 4;
    // variklio praradimai (RunResult::any_loss, taip pat išėjimo klaidos) — exit 4, kaip praradimų raktai prieš migraciją
    if (!rc && engine_loss) rc = 4;
    return finish(exit_code(rc));  // kStoppedWaiting — sustabdyta belaukiant įėjimų (tvarkinga pabaiga, exit 0)
}

}  // namespace mxlv::func
