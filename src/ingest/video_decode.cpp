// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/video_decode.hpp"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
}
#include <cuda_runtime_api.h>

#include <cstring>

#include "ingest/av_input.hpp"

namespace mxlv::ingest {

namespace {
enum AVPixelFormat pick_cuda(AVCodecContext*, const enum AVPixelFormat* f) {
    for (; *f != AV_PIX_FMT_NONE; ++f)
        if (*f == AV_PIX_FMT_CUDA) return *f;
    return AV_PIX_FMT_NONE;  // be CUDA paviršių — klaida (programinio atsarginio kelio čia nėra: --decoder sw)
}
}  // namespace

bool parse_decoder(const std::string& s, Decoder& out) {
    if (s == "nvdec") out = Decoder::Nvdec;
    else if (s == "cuvid") out = Decoder::Cuvid;
    else if (s == "sw") out = Decoder::Sw;
    else return false;
    return true;
}

const char* decoder_name(Decoder d) {
    switch (d) {
        case Decoder::Nvdec: return "nvdec";
        case Decoder::Cuvid: return "cuvid";
        case Decoder::Sw: return "sw";
    }
    return "?";
}

bool VideoDecoder::open(const AVCodecParameters* par, Decoder d, AVBufferRef* hw, bool to_device, int threads, int extra_frames,
                        int device, std::string& err) {
    close();
    d_ = d;
    device_ = device;
    to_device_ = d == Decoder::Sw && to_device;
    const AVCodec* codec = nullptr;
    if (d == Decoder::Cuvid) codec = avcodec_find_decoder_by_name(par->codec_id == AV_CODEC_ID_HEVC ? "hevc_cuvid" : "h264_cuvid");
    else codec = avcodec_find_decoder(par->codec_id);
    if (!codec) {
        err = std::string("dekoderis nerastas: ") + decoder_name(d) + " " + avcodec_get_name(par->codec_id);
        return false;
    }
    c_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(c_, par);
    c_->pkt_timebase = AVRational{1, 1'000'000'000};
    AVDictionary* o = nullptr;
    if (d == Decoder::Nvdec) {
        hw_ = av_buffer_ref(hw);
        c_->hw_device_ctx = av_buffer_ref(hw);
        c_->get_format = pick_cuda;
        c_->extra_hw_frames = extra_frames;
        c_->thread_count = 1;
    } else if (d == Decoder::Cuvid) {
        hw_ = av_buffer_ref(hw);
        c_->hw_device_ctx = av_buffer_ref(hw);
        c_->get_format = pick_cuda;
        av_dict_set(&o, "deint", "weave", 0);
        av_dict_set(&o, "drop_second_field", "0", 0);
    } else {
        c_->thread_count = threads > 0 ? threads : 2;
        c_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
        if (to_device_) hw_ = av_buffer_ref(hw);
    }
    const int e = avcodec_open2(c_, codec, &o);
    av_dict_free(&o);
    if (e < 0) {
        err = std::string(decoder_name(d)) + " atidaryti nepavyko: " + averr(e);
        close();
        return false;
    }
    tmp_ = av_frame_alloc();
    return true;
}

void VideoDecoder::close() {
    if (stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
    avcodec_free_context(&c_);
    av_buffer_unref(&frames_);
    av_buffer_unref(&hw_);
    av_frame_free(&tmp_);
    av_frame_free(&nv12_);
}

int VideoDecoder::send(AVPacket* pkt) {
    const int e = avcodec_send_packet(c_, pkt);
    if (e != 0 && e != AVERROR(EAGAIN)) ++errors_;
    return e;
}

bool VideoDecoder::ensure_pool(int w, int h, std::string& err) {
    if (frames_) {
        const auto* fc = reinterpret_cast<const AVHWFramesContext*>(frames_->data);
        if (fc->width == w && fc->height == h) return true;
        av_buffer_unref(&frames_);  // raiška pasikeitė (kviečiantysis šaltinį stabdo; baseinas — naujam dydžiui)
    }
    frames_ = av_hwframe_ctx_alloc(hw_);
    auto* fc = reinterpret_cast<AVHWFramesContext*>(frames_->data);
    fc->format = AV_PIX_FMT_CUDA;
    fc->sw_format = AV_PIX_FMT_NV12;
    fc->width = w;
    fc->height = h;
    if (const int e = av_hwframe_ctx_init(frames_); e < 0) {  // baseinas dinaminis: kadrų tiek, kiek jų laikoma
        err = "CUDA kadrų baseinas: " + averr(e);
        av_buffer_unref(&frames_);
        return false;
    }
    return true;
}

bool VideoDecoder::compact(AVFrame* src, AVFrame*& dst, std::string& err) {
    if (!ensure_pool(src->width, src->height, err)) return false;
    if (!stream_) {
        cudaStream_t s{};
        // BLOKUOJANTIS stream'as: libav nvdec / cuvid kadrą į išvesties buferį kopijuoja asinchroniškai numatytajame (legacy)
        // stream'e — non-blocking stream'as su juo nesisinchronizuotų ir kopijuotų dar nebaigtą (išmatuota: A/V nuslinko kadru)
        if (cudaSetDevice(device_) != cudaSuccess || cudaStreamCreate(&s) != cudaSuccess) {
            err = "CUDA stream'as kopijai";
            return false;
        }
        stream_ = s;
    }
    dst = av_frame_alloc();
    if (const int e = av_hwframe_get_buffer(frames_, dst, 0); e < 0) {
        err = "CUDA kadras: " + averr(e);
        av_frame_free(&dst);
        return false;
    }
    auto* s = static_cast<cudaStream_t>(stream_);
    const auto w = static_cast<std::size_t>(src->width);
    // NV12: Y (h eilučių) ir supinta UV (h / 2 eilučių); kopija baigiama prieš atlaisvinant libav kadrą (dekoderis paviršių
    // gali tuoj pat perrašyti)
    cudaError_t ce = cudaMemcpy2DAsync(dst->data[0], static_cast<std::size_t>(dst->linesize[0]), src->data[0],
                                       static_cast<std::size_t>(src->linesize[0]), w, static_cast<std::size_t>(src->height),
                                       cudaMemcpyDeviceToDevice, s);
    if (ce == cudaSuccess)
        ce = cudaMemcpy2DAsync(dst->data[1], static_cast<std::size_t>(dst->linesize[1]), src->data[1], static_cast<std::size_t>(src->linesize[1]),
                               w, static_cast<std::size_t>(src->height / 2), cudaMemcpyDeviceToDevice, s);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(s);
    if (ce != cudaSuccess) {
        err = std::string("CUDA kopija: ") + cudaGetErrorString(ce);
        av_frame_free(&dst);
        return false;
    }
    av_frame_copy_props(dst, src);
    return true;
}

bool VideoDecoder::to_device(AVFrame* src, AVFrame*& dst, std::string& err) {
    if (!ensure_pool(src->width, src->height, err)) return false;
    if (!nv12_ || nv12_->width != src->width || nv12_->height != src->height) {
        av_frame_free(&nv12_);
        nv12_ = av_frame_alloc();
        nv12_->format = AV_PIX_FMT_NV12;
        nv12_->width = src->width;
        nv12_->height = src->height;
        if (av_frame_get_buffer(nv12_, 64) < 0) {
            err = "NV12 buferis";
            return false;
        }
    }
    // yuv420p → NV12 (host): Y eilutės, U / V supinami
    for (int y = 0; y < src->height; ++y)
        std::memcpy(nv12_->data[0] + static_cast<std::ptrdiff_t>(y) * nv12_->linesize[0], src->data[0] + static_cast<std::ptrdiff_t>(y) * src->linesize[0],
                    static_cast<std::size_t>(src->width));
    for (int y = 0; y < src->height / 2; ++y) {
        const std::uint8_t* u = src->data[1] + static_cast<std::ptrdiff_t>(y) * src->linesize[1];
        const std::uint8_t* v = src->data[2] + static_cast<std::ptrdiff_t>(y) * src->linesize[2];
        std::uint8_t* d = nv12_->data[1] + static_cast<std::ptrdiff_t>(y) * nv12_->linesize[1];
        for (int x = 0; x < src->width / 2; ++x) {
            d[2 * x] = u[x];
            d[2 * x + 1] = v[x];
        }
    }
    dst = av_frame_alloc();
    if (const int e = av_hwframe_get_buffer(frames_, dst, 0); e < 0) {
        err = "CUDA kadras: " + averr(e);
        av_frame_free(&dst);
        return false;
    }
    // Įkėlimas numatytajame (legacy) CUDA stream'e; ingest split'as — BLOKUOJANČIAME stream'e, todėl tvarka garantuota
    if (const int e = av_hwframe_transfer_data(dst, nv12_, 0); e < 0) {
        err = "įkėlimas į GPU: " + averr(e);
        av_frame_free(&dst);
        return false;
    }
    av_frame_copy_props(dst, src);
    return true;
}

bool VideoDecoder::receive(DecodedFrame& out, std::string& err) {
    const int r = avcodec_receive_frame(c_, tmp_);
    if (r != 0) {
        if (r != AVERROR(EAGAIN) && r != AVERROR_EOF) ++errors_;
        return false;
    }
    if (tmp_->decode_error_flags) ++errors_;
    out.pts_tai_ns = tmp_->best_effort_timestamp != AV_NOPTS_VALUE ? tmp_->best_effort_timestamp : tmp_->pts;
    out.interlaced = (tmp_->flags & AV_FRAME_FLAG_INTERLACED) != 0;
    out.tff = (tmp_->flags & AV_FRAME_FLAG_TOP_FIELD_FIRST) != 0;
    if (to_device_ || tmp_->format == AV_PIX_FMT_CUDA) {
        AVFrame* dev = nullptr;
        const bool ok = to_device_ ? to_device(tmp_, dev, err) : compact(tmp_, dev, err);
        av_frame_unref(tmp_);
        if (!ok) return false;
        out.f = dev;
        return true;
    }
    out.f = av_frame_alloc();
    av_frame_move_ref(out.f, tmp_);
    return true;
}

void VideoDecoder::flush() {
    if (c_) avcodec_flush_buffers(c_);
}

}  // namespace mxlv::ingest
