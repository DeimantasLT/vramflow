// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Video dekoderis ingest'ui (planas §P3): NVDEC per libavcodec arba programinis. Paketų pts / dts — TAI ns (laiko bazė 1/10⁹),
// todėl kadro best_effort_timestamp jau yra jo pateikimo laikas TAI (B kadrų pertvarka — dekoderio).
//   nvdec — h264 / hevc + hwaccel cuda (kadrai — CUDA NV12; numatytasis: mažesnė išvesties delsa nei cuvid, žr. rezultatus);
//   cuvid — h264_cuvid / hevc_cuvid (deint=weave; kadras išeina 4–8 kadrais vėliau nei paketas, brief'o pamoka 3);
//   sw    — FFmpeg programinis (kadrų + slice'ų gijos); to_device: kadras perkeliamas į CUDA NV12 (U / V supinami CPU, tada
//           av_hwframe_transfer_data) — tas pats GPU kelias kaip NVDEC; kitaip — host yuv420p (C variantas, CPU split).
// VRAM (US 12): NVDEC kadras laiko ir dekoderio paviršių, ir išvesties kopiją (libav nvdec / cuvid) — atsargoje laikomas kadras
// kainuotų du NV12 buferius. Todėl nvdec / cuvid kadras iškart nukopijuojamas (D2D, savo stream'e, sync) į savo baseino kadrą,
// o libav kadras atlaisvinamas: dekoderiui pakanka DPB + kelių paviršių, laikomas kadras — vienas buferis.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include <cstdint>
#include <string>

namespace mxlv::ingest {

enum class Decoder : std::uint8_t { Nvdec, Cuvid, Sw };
bool parse_decoder(const std::string& s, Decoder& out);
const char* decoder_name(Decoder d);

struct DecodedFrame {
    AVFrame* f = nullptr;          // CUDA NV12 (nvdec / cuvid / sw + to_device) arba host yuv420p (sw)
    std::int64_t pts_tai_ns = 0;
    bool interlaced = false, tff = false;
};

class VideoDecoder {
public:
    VideoDecoder() = default;
    ~VideoDecoder() { close(); }
    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;

    // hw — CUDA įrenginio kontekstas (nvdec / cuvid / sw + to_device). threads — programiniam (0 — 2). extra_frames — kiek
    // dekoduotų kadrų laikoma ne dekoderyje (atsarga + tick'as): nvdec paviršių baseinui.
    // device — CUDA įrenginys (kopijai D2D dekodavimo gijoje).
    bool open(const AVCodecParameters* par, Decoder d, AVBufferRef* hw, bool to_device, int threads, int extra_frames, int device,
              std::string& err);
    void close();
    // Paketas (pts / dts — TAI ns). 0 — priimtas; AVERROR(EAGAIN) — išvesties eilė pilna: pirma receive, tada tą patį paketą
    // siųsti iš naujo; kita — dekoderio klaida (skaičiuojama, tęsiama).
    int send(AVPacket* pkt);
    // Kitas kadras; false — šiuo metu nėra. Kadras perduodamas kviečiančiajam (av_frame_free).
    bool receive(DecodedFrame& out, std::string& err);
    void flush();
    bool is_open() const { return c_ != nullptr; }
    std::uint64_t errors() const { return errors_; }

private:
    bool to_device(AVFrame* src, AVFrame*& dst, std::string& err);
    bool compact(AVFrame* src, AVFrame*& dst, std::string& err);  // CUDA → savo baseino kadras (D2D)
    bool ensure_pool(int w, int h, std::string& err);
    AVCodecContext* c_ = nullptr;
    Decoder d_ = Decoder::Nvdec;
    bool to_device_ = false;
    AVBufferRef* hw_ = nullptr;      // ref (nvdec / cuvid / to_device)
    AVBufferRef* frames_ = nullptr;  // sw → CUDA NV12 kadrų baseinas
    AVFrame* tmp_ = nullptr;         // gautas kadras
    AVFrame* nv12_ = nullptr;        // host NV12 (sw → CUDA)
    std::uint64_t errors_ = 0;
    int device_ = -1;
    void* stream_ = nullptr;  // cudaStream_t (antraštė be CUDA)
};

}  // namespace mxlv::ingest
