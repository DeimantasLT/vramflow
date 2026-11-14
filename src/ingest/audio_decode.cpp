// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/audio_decode.hpp"

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
}

#include <cmath>
#include <cstdlib>

#include "ingest/av_input.hpp"

namespace mxlv::ingest {

bool AudioDecoder::open(const AVCodecParameters* par, int out_channels, const AudioSyncConfig& sync, std::string& err) {
    close();
    const AVCodec* codec = avcodec_find_decoder(par->codec_id);
    if (!codec) {
        err = std::string("garso dekoderis nerastas: ") + avcodec_get_name(par->codec_id);
        return false;
    }
    c_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(c_, par);
    c_->pkt_timebase = AVRational{1, 1'000'000'000};
    if (const int e = avcodec_open2(c_, codec, nullptr); e < 0) {
        err = std::string("garso dekoderis ") + codec->name + ": " + averr(e);
        close();
        return false;
    }
    out_ch_ = out_channels;
    sync_ = AudioSync(sync);
    fr_ = av_frame_alloc();
    buf_.assign(static_cast<std::size_t>(out_ch_), {});
    ptr_.assign(static_cast<std::size_t>(out_ch_), nullptr);
    return true;
}

void AudioDecoder::close() {
    avcodec_free_context(&c_);
    swr_free(&swr_);
    av_frame_free(&fr_);
    av_channel_layout_uninit(&in_layout_);
    in_rate_ = 0;
    in_fmt_ = -1;
}

int AudioDecoder::send(AVPacket* pkt) {
    if (!c_) return AVERROR(EINVAL);  // neatidarytas (A2: kviečiantysis garsą laiko „nėra“; čia — tik apsauga nuo nullptr)
    const int e = avcodec_send_packet(c_, pkt);
    if (e != 0 && e != AVERROR(EAGAIN)) ++errors_;
    return e;
}

bool AudioDecoder::ensure_swr(const AVFrame* f) {
    if (swr_ && f->sample_rate == in_rate_ && f->format == in_fmt_ && av_channel_layout_compare(&f->ch_layout, &in_layout_) == 0) return true;
    swr_free(&swr_);
    AVChannelLayout out{};
    av_channel_layout_default(&out, out_ch_);
    if (swr_alloc_set_opts2(&swr_, &out, AV_SAMPLE_FMT_FLTP, 48000, &f->ch_layout, static_cast<AVSampleFormat>(f->format), f->sample_rate, 0,
                            nullptr) < 0) {
        av_channel_layout_uninit(&out);
        return false;
    }
    av_channel_layout_uninit(&out);
    // Resampleris įjungtas visada (ir 48 → 48 kHz): kompensacija (swr_set_compensation) jo vėliau neperjungia ir delsa nekinta
    av_opt_set_int(swr_, "flags", SWR_FLAG_RESAMPLE, 0);
    if (swr_init(swr_) < 0) {
        swr_free(&swr_);
        return false;
    }
    in_rate_ = f->sample_rate;
    in_fmt_ = f->format;
    av_channel_layout_uninit(&in_layout_);
    av_channel_layout_copy(&in_layout_, &f->ch_layout);
    sync_.reset();  // nauji parametrai — pradžia iš naujo
    return true;
}

bool AudioDecoder::receive(AudioChunk& out) {
    if (!c_) return false;
    for (;;) {
        const int r = avcodec_receive_frame(c_, fr_);
        if (r != 0) {
            if (r != AVERROR(EAGAIN) && r != AVERROR_EOF) ++errors_;
            return false;
        }
        const std::int64_t pts = fr_->best_effort_timestamp != AV_NOPTS_VALUE ? fr_->best_effort_timestamp : fr_->pts;
        if (pts == AV_NOPTS_VALUE || !ensure_swr(fr_)) {
            ++errors_;
            av_frame_unref(fr_);
            continue;
        }
        // Kur turi prasidėti ši išvestis: įėjimo laikas, nuo kurio resampleris dar laiko mėginius (delsa — prieš konvertuojant)
        const std::int64_t delay_ns = swr_get_delay(swr_, 1'000'000'000);
        const auto expected = static_cast<std::int64_t>(std::llround(static_cast<long double>(pts - delay_ns) * 48000.0L / 1e9L));
        const AudioSyncStep st = sync_.on_frame(expected);
        if (st.comp != 0) comp_abs_ += std::llabs(st.comp);
        swr_set_compensation(swr_, static_cast<int>(st.comp), static_cast<int>(st.comp != 0 ? st.distance : 0));
        const int cap = swr_get_out_samples(swr_, fr_->nb_samples) + 64;
        for (std::size_t ch = 0; ch < buf_.size(); ++ch) {
            if (buf_[ch].size() < static_cast<std::size_t>(cap)) buf_[ch].resize(static_cast<std::size_t>(cap));
            ptr_[ch] = buf_[ch].data();
        }
        const int n = swr_convert(swr_, reinterpret_cast<std::uint8_t**>(ptr_.data()), cap,
                                  const_cast<const std::uint8_t**>(fr_->extended_data), fr_->nb_samples);
        av_frame_unref(fr_);
        if (n < 0) {
            ++errors_;
            continue;
        }
        out.start = sync_.next();
        out.n = static_cast<std::uint32_t>(n);
        out.planes = ptr_.data();
        out.resync = st.resync;
        sync_.produced(n);
        if (n == 0) continue;
        return true;
    }
}

void AudioDecoder::reset() {
    if (c_) avcodec_flush_buffers(c_);
    swr_free(&swr_);
    in_rate_ = 0;
    in_fmt_ = -1;
    sync_.reset();
}

}  // namespace mxlv::ingest
