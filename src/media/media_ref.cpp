// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// NEPRIKLAUSOMI skaliariniai CPU etalonai (architektūra GPU funkcijų §6). Rašyta iš dokumento formulių:
// ciklai per pikselius ir chromos poras, o ne per v210 grupes kaip kernel'iuose. v210 eilutė čia traktuojama kaip
// mėginių seka Cb0 Y0 Cr0 Y1 Cb1 Y2 Cr1 Y3 … (po 3 mėginius 32 bitų žodyje, bitai 0–9, 10–19, 20–29) — tai ta pati
// §6.1 lentelė, bet išreikšta kitaip nei v210.hpp pack_group/unpack_group.
#include "media/media_ref.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

namespace mxlv::media::ref {
namespace {

constexpr int kChromaPairs = kWidth / 2;    // 960 chromos porų eilutėje
constexpr int kChromaRowsField = kFieldHeight / 2;  // 270 lauko chromos eilių NV12 kadre

// Mėginio eilės numeris v210 eilutėje (§6.1): pora i (pikseliai 2i, 2i+1) užima mėginius 4i..4i+3 = Cb_i, Y_2i, Cr_i, Y_2i+1.
int serial_y(int x) { return 4 * (x / 2) + 1 + 2 * (x % 2); }
int serial_cb(int i) { return 4 * i; }
int serial_cr(int i) { return 4 * i + 2; }

int get10(const std::uint32_t* line, int s) { return static_cast<int>((line[s / 3] >> (10 * (s % 3))) & 0x3FFu); }

// Įrašo 10 bitų mėginį, nekeisdamas kitų žodžio bitų (įskaitant 30–31).
void put10(std::uint32_t* line, int s, int v) {
    std::uint32_t& w = line[s / 3];
    const int sh = 10 * (s % 3);
    w = (w & ~(0x3FFu << sh)) | ((static_cast<std::uint32_t>(v) & 0x3FFu) << sh);
}

const std::uint32_t* field_line(const std::uint32_t* f, int j) { return f + static_cast<std::size_t>(j) * kWordsPerLine; }
std::uint32_t* field_line(std::uint32_t* f, int j) { return f + static_cast<std::size_t>(j) * kWordsPerLine; }

int round_half_up(double v) { return static_cast<int>(std::floor(v + 0.5)); }

std::string av_err(int e) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(e, buf, sizeof buf);
    return buf;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Spalvos ir blend (§6.3)
// ---------------------------------------------------------------------------------------------------------------------

void rgb_to_ycbcr10(int r, int g, int b, int& y, int& cb, int& cr) {
    const double R = r / 255.0, G = g / 255.0, B = b / 255.0;
    const double Yp = 0.2126 * R + 0.7152 * G + 0.0722 * B;
    y = std::clamp(round_half_up(64.0 + 876.0 * Yp), 64, 940);
    cb = std::clamp(round_half_up(512.0 + 896.0 * (B - Yp) / 1.8556), 64, 960);
    cr = std::clamp(round_half_up(512.0 + 896.0 * (R - Yp) / 1.5748), 64, 960);
}

int blend(int s, int l, int a) { return (s * (255 - a) + l * a + 127) / 255; }

// ---------------------------------------------------------------------------------------------------------------------
// Logotipas (§6.3)
// ---------------------------------------------------------------------------------------------------------------------

bool logo_from_rgba(const std::uint8_t* rgba, int w, int h, Logo& out) {
    if (w <= 0 || h <= 0 || (w % 2) != 0 || rgba == nullptr) return false;
    const int cw = w / 2;
    out = Logo{};
    out.w = w;
    out.h = h;
    out.y10.resize(static_cast<std::size_t>(w) * h);
    out.a8.resize(static_cast<std::size_t>(w) * h);
    out.cb10.resize(static_cast<std::size_t>(cw) * h);
    out.cr10.resize(static_cast<std::size_t>(cw) * h);
    out.ac8.resize(static_cast<std::size_t>(cw) * h);
    std::vector<int> cbp(static_cast<std::size_t>(w)), crp(static_cast<std::size_t>(w));
    for (int row = 0; row < h; ++row) {
        // pirmiausia — kiekvieno pikselio Y/Cb/Cr ir alfa
        for (int x = 0; x < w; ++x) {
            const std::uint8_t* p = rgba + (static_cast<std::size_t>(row) * w + x) * 4;
            int y = 0;
            rgb_to_ycbcr10(p[0], p[1], p[2], y, cbp[x], crp[x]);
            out.y10[static_cast<std::size_t>(row) * w + x] = static_cast<std::uint16_t>(y);
            out.a8[static_cast<std::size_t>(row) * w + x] = p[3];
        }
        // chromos poros: alfa svertinis vidurkis, kad permatomi pikseliai nedažytų
        for (int i = 0; i < cw; ++i) {
            const int a0 = rgba[(static_cast<std::size_t>(row) * w + 2 * i) * 4 + 3];
            const int a1 = rgba[(static_cast<std::size_t>(row) * w + 2 * i + 1) * 4 + 3];
            const std::size_t o = static_cast<std::size_t>(row) * cw + i;
            out.ac8[o] = static_cast<std::uint8_t>((a0 + a1 + 1) >> 1);
            if (a0 + a1 == 0) {
                out.cb10[o] = 512;
                out.cr10[o] = 512;
            } else {
                const double sum = a0 + a1;
                out.cb10[o] = static_cast<std::uint16_t>(round_half_up((a0 * double(cbp[2 * i]) + a1 * double(cbp[2 * i + 1])) / sum));
                out.cr10[o] = static_cast<std::uint16_t>(round_half_up((a0 * double(crp[2 * i]) + a1 * double(crp[2 * i + 1])) / sum));
            }
        }
    }
    return true;
}

Logo builtin_logo() {
    // 384×216: 8 px visiškai permatomas rėmelis; viduje alfa gradientas iš kairės į dešinę 0..255 (visos reikšmės),
    // viršuje 48 eilučių balta juosta, žemiau — raudona / žalia / mėlyna vertikalios juostos (trečdaliai).
    constexpr int W = 384, H = 216, B = 8, kBand = 48;
    constexpr int iw = W - 2 * B;  // 368
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(W) * H * 4, 0);
    for (int y = B; y < H - B; ++y)
        for (int x = B; x < W - B; ++x) {
            std::uint8_t* p = rgba.data() + (static_cast<std::size_t>(y) * W + x) * 4;
            const int ix = x - B;
            if (y < B + kBand) {
                p[0] = p[1] = p[2] = 255;
            } else {
                const int stripe = ix * 3 / iw;
                p[0] = stripe == 0 ? 255 : 0;
                p[1] = stripe == 1 ? 255 : 0;
                p[2] = stripe == 2 ? 255 : 0;
            }
            p[3] = static_cast<std::uint8_t>(ix * 255 / (iw - 1));
        }
    Logo l;
    logo_from_rgba(rgba.data(), W, H, l);
    return l;
}

std::string load_png_rgba(const std::string& path, std::vector<std::uint8_t>& rgba, int& w, int& h) {
    struct FmtDel {
        void operator()(AVFormatContext* p) const { avformat_close_input(&p); }
    };
    struct CodecDel {
        void operator()(AVCodecContext* p) const { avcodec_free_context(&p); }
    };
    struct PktDel {
        void operator()(AVPacket* p) const { av_packet_free(&p); }
    };
    struct FrameDel {
        void operator()(AVFrame* p) const { av_frame_free(&p); }
    };

    const AVInputFormat* ifmt = av_find_input_format("image2");
    if (!ifmt) return "libavformat be image2 demux'erio";
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "pattern_type", "none", 0);  // kelias — ne šablonas (pvz. su '%')
    AVFormatContext* raw_ic = nullptr;
    int e = avformat_open_input(&raw_ic, path.c_str(), ifmt, &opts);
    av_dict_free(&opts);
    if (e < 0) return "avformat_open_input(" + path + "): " + av_err(e);
    std::unique_ptr<AVFormatContext, FmtDel> ic(raw_ic);
    if ((e = avformat_find_stream_info(ic.get(), nullptr)) < 0) return "avformat_find_stream_info: " + av_err(e);
    const int si = av_find_best_stream(ic.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (si < 0) return "nėra vaizdo srauto: " + path;
    const AVCodecParameters* par = ic->streams[si]->codecpar;
    if (par->codec_id != AV_CODEC_ID_PNG) return std::string("ne PNG: ") + avcodec_get_name(par->codec_id);
    const AVCodec* dec = avcodec_find_decoder(par->codec_id);
    if (!dec) return "nėra PNG dekoderio";
    std::unique_ptr<AVCodecContext, CodecDel> cc(avcodec_alloc_context3(dec));
    if (!cc) return "avcodec_alloc_context3";
    if ((e = avcodec_parameters_to_context(cc.get(), par)) < 0) return "avcodec_parameters_to_context: " + av_err(e);
    if ((e = avcodec_open2(cc.get(), dec, nullptr)) < 0) return "avcodec_open2: " + av_err(e);
    std::unique_ptr<AVPacket, PktDel> pkt(av_packet_alloc());
    std::unique_ptr<AVFrame, FrameDel> fr(av_frame_alloc());
    if (!pkt || !fr) return "av_packet_alloc/av_frame_alloc";

    bool got = false;
    while (!got && av_read_frame(ic.get(), pkt.get()) >= 0) {
        if (pkt->stream_index == si) {
            if ((e = avcodec_send_packet(cc.get(), pkt.get())) < 0) {
                av_packet_unref(pkt.get());
                return "avcodec_send_packet: " + av_err(e);
            }
            got = avcodec_receive_frame(cc.get(), fr.get()) == 0;
        }
        av_packet_unref(pkt.get());
    }
    if (!got) {  // dekoderio ištuštinimas
        avcodec_send_packet(cc.get(), nullptr);
        got = avcodec_receive_frame(cc.get(), fr.get()) == 0;
    }
    if (!got) return "PNG nepavyko dekoduoti: " + path;

    const auto fmt = static_cast<AVPixelFormat>(fr->format);
    if (fmt != AV_PIX_FMT_RGBA && fmt != AV_PIX_FMT_RGB24) {
        const char* n = av_get_pix_fmt_name(fmt);
        return std::string("nepalaikomas PNG pikselių formatas: ") + (n ? n : "?") + " (reikia RGBA arba RGB24)";
    }
    w = fr->width;
    h = fr->height;
    rgba.assign(static_cast<std::size_t>(w) * h * 4, 0);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* src = fr->data[0] + static_cast<std::ptrdiff_t>(y) * fr->linesize[0];
        std::uint8_t* dst = rgba.data() + static_cast<std::size_t>(y) * w * 4;
        for (int x = 0; x < w; ++x) {
            if (fmt == AV_PIX_FMT_RGBA) {
                for (int c = 0; c < 4; ++c) dst[x * 4 + c] = src[x * 4 + c];
            } else {
                for (int c = 0; c < 3; ++c) dst[x * 4 + c] = src[x * 3 + c];
                dst[x * 4 + 3] = 255;
            }
        }
    }
    return "";
}

void logo_field(const std::uint32_t* in, std::uint32_t* out, const Logo& logo, int X, int Y, int parity) {
    std::copy(in, in + kFieldWords, out);  // viskas už stačiakampio — bitas į bitą
    const int cw = logo.w / 2;
    for (int j = 0; j < kFieldHeight; ++j) {
        const int fy = 2 * j + parity;
        const int ly = fy - Y;
        if (ly < 0 || ly >= logo.h) continue;
        std::uint32_t* line = field_line(out, j);
        for (int x = 0; x < kWidth; ++x) {
            const int lx = x - X;
            if (lx < 0 || lx >= logo.w) continue;
            const std::size_t o = static_cast<std::size_t>(ly) * logo.w + lx;
            const int s = serial_y(x);
            put10(line, s, blend(get10(line, s), logo.y10[o], logo.a8[o]));
        }
        for (int i = 0; i < kChromaPairs; ++i) {
            const int lx = 2 * i - X;  // poros lyginis pikselis logotipo koordinatėmis (X lyginis)
            if (lx < 0 || lx >= logo.w) continue;
            const std::size_t o = static_cast<std::size_t>(ly) * cw + lx / 2;
            const int a = logo.ac8[o];
            put10(line, serial_cb(i), blend(get10(line, serial_cb(i)), logo.cb10[o], a));
            put10(line, serial_cr(i), blend(get10(line, serial_cr(i)), logo.cr10[o], a));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// NV12 ⇄ laukai (§6.2, §6.4)
// ---------------------------------------------------------------------------------------------------------------------

void weave_nv12(const std::uint32_t* top, const std::uint32_t* bottom, std::uint8_t* y, std::uint8_t* uv, int pitch_y,
                int pitch_uv) {
    const std::uint32_t* fields[2] = {top, bottom};
    // luma: kadro eilutė r = 2L + p ← lauko p eilutė L
    for (int r = 0; r < kFrameHeight; ++r) {
        const std::uint32_t* line = field_line(fields[r % 2], r / 2);
        for (int x = 0; x < kWidth; ++x)
            y[static_cast<std::size_t>(r) * pitch_y + x] = static_cast<std::uint8_t>(std::min(255, (get10(line, serial_y(x)) + 2) >> 2));
    }
    // chroma: UV eilė u = 2k + p ← lauko p chromos eilutės 2k ir 2k+1 (viršutinis ¼, apatinis ¾)
    for (int u = 0; u < kFrameHeight / 2; ++u) {
        const int p = u % 2, k = u / 2;
        const std::uint32_t* l0 = field_line(fields[p], 2 * k);
        const std::uint32_t* l1 = field_line(fields[p], 2 * k + 1);
        const int w0 = p == 0 ? 3 : 1, w1 = p == 0 ? 1 : 3;
        for (int i = 0; i < kChromaPairs; ++i) {
            const int cb = (w0 * get10(l0, serial_cb(i)) + w1 * get10(l1, serial_cb(i)) + 2) >> 2;
            const int cr = (w0 * get10(l0, serial_cr(i)) + w1 * get10(l1, serial_cr(i)) + 2) >> 2;
            uv[static_cast<std::size_t>(u) * pitch_uv + 2 * i] = static_cast<std::uint8_t>(std::min(255, (cb + 2) >> 2));
            uv[static_cast<std::size_t>(u) * pitch_uv + 2 * i + 1] = static_cast<std::uint8_t>(std::min(255, (cr + 2) >> 2));
        }
    }
}

void split_nv12(const std::uint8_t* y, const std::uint8_t* uv, int pitch_y, int pitch_uv, int parity, std::uint32_t* out) {
    std::fill(out, out + kFieldWords, 0u);
    // lauko `parity` chromos eilė c[k] = UV eilė 2k + parity
    auto c = [&](int k, int i, int comp) {
        k = std::clamp(k, 0, kChromaRowsField - 1);
        return static_cast<int>(uv[static_cast<std::size_t>(2 * k + parity) * pitch_uv + 2 * i + comp]);
    };
    for (int j = 0; j < kFieldHeight; ++j) {
        std::uint32_t* line = field_line(out, j);
        const int row = 2 * j + parity;
        for (int x = 0; x < kWidth; ++x) put10(line, serial_y(x), y[static_cast<std::size_t>(row) * pitch_y + x] << 2);
        const int k = j / 2;
        for (int i = 0; i < kChromaPairs; ++i)
            for (int comp = 0; comp < 2; ++comp) {
                int acc = 0;  // aštuntosiomis, §6.4 lentelė
                if (parity == 0 && j % 2 == 0) acc = 1 * c(k - 1, i, comp) + 7 * c(k, i, comp);
                if (parity == 0 && j % 2 == 1) acc = 5 * c(k, i, comp) + 3 * c(k + 1, i, comp);
                if (parity == 1 && j % 2 == 0) acc = 3 * c(k - 1, i, comp) + 5 * c(k, i, comp);
                if (parity == 1 && j % 2 == 1) acc = 7 * c(k, i, comp) + 1 * c(k + 1, i, comp);
                put10(line, comp == 0 ? serial_cb(i) : serial_cr(i), (acc + 1) >> 1);
            }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Generatorius ir barkodas (§6.5, §6.1)
// ---------------------------------------------------------------------------------------------------------------------

void gen_field(const GenParams& g, std::uint64_t index, std::uint32_t* out) {
    std::fill(out, out + kFieldWords, 0u);  // bitai 30–31 = 0
    const int p = static_cast<int>(index & 1u);
    for (int j = 0; j < kFieldHeight; ++j) {
        std::uint32_t* line = field_line(out, j);
        const int fy = 2 * j + p;
        for (int x = 0; x < kWidth; ++x) put10(line, serial_y(x), gen_y(g, index, x, fy));
        for (int i = 0; i < kChromaPairs; ++i) {
            std::uint16_t cb = 0, cr = 0;
            gen_c(g, index, 2 * i, fy, cb, cr);
            put10(line, serial_cb(i), cb);
            put10(line, serial_cr(i), cr);
        }
    }
}

std::uint32_t decode_marker(const std::uint32_t* field) {
    std::uint32_t word = 0;
    for (int k = 0; k < 32; ++k) {
        int sum = 0;  // bloko centro 8×8: lauko eilutės 4..11, x 32k+12..32k+19
        for (int j = 4; j < 12; ++j)
            for (int x = 32 * k + 12; x < 32 * k + 20; ++x) sum += get10(field_line(field, j), serial_y(x));
        if (sum > 502 * 64) word |= 1u << k;  // vidurkis > 502
    }
    return word;
}

std::uint32_t decode_marker_nv12(const std::uint8_t* y, int pitch_y, int parity) {
    std::uint32_t word = 0;
    for (int k = 0; k < 32; ++k) {
        int sum = 0;
        for (int j = 4; j < 12; ++j)
            for (int x = 32 * k + 12; x < 32 * k + 20; ++x) sum += y[static_cast<std::size_t>(2 * j + parity) * pitch_y + x];
        if (sum > 125 * 64) word |= 1u << k;  // vidurkis > 125 (8 bit)
    }
    return word;
}

}  // namespace mxlv::media::ref
