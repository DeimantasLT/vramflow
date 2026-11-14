// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// H.264 skenavimo ir laukų tvarkos nustatymas iš bitų srauto (Annex B), be libav. Kodėl ne libav: zondavimas, prasidėjęs srauto
// viduryje, praneša field_order pagal pirmą pasitaikiusį kadrą (ne pagal rakto kadrą), todėl tas pats srautas nuo pradžios ir nuo
// vidurio gali gauti skirtingą tvarką (išmatuota 2026-11-14). Sprendžiama ties pirmu IDR, signalų pirmumo tvarka:
//   SPS frame_mbs_only_flag == 1                         — progresyvus;
//   pic_timing SEI pic_struct 3 / 5 (top-bottom[-top])    — TFF; 4 / 6 (bottom-top[-bottom]) — BFF (VUI pic_struct_present_flag);
//   laukų poros (PAFF)                                    — pirmas koduotas IDR laukas: viršutinis — TFF, apatinis — BFF;
//   kadras, POC tipas 0 ir delta_pic_order_cnt_bottom     — > 0 TFF (viršutinio lauko POC mažesnis), < 0 BFF;
//   kitaip                                                — TFF (transliacijos numatytoji).
// Pastaba: x264 pic_struct ima iš įėjimo kadro TOP_FIELD_FIRST žymės — be jos (rawvideo įėjimas) TFF turinys pažymimas BFF; todėl
// 62-make-av-clip.sh įėjimą žymi setfield=tff (bff klipui — setfield=bff).
// Gryna logika — tests/unit/test_h264_scan.cpp.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mxlv::ingest {

enum class Scan : std::uint8_t { Unknown, Progressive, Tff, Bff };

namespace h264 {

class Bits {
public:
    Bits(const std::uint8_t* p, std::size_t n) : p_(p), n_(n) {}
    std::uint32_t u(int k) {
        std::uint32_t v = 0;
        for (int i = 0; i < k; ++i) v = (v << 1) | bit();
        return v;
    }
    std::uint32_t ue() {
        int z = 0;
        while (!bit() && z < 32 && ok()) ++z;
        return z == 0 ? 0 : ((1u << z) - 1) + u(z);
    }
    std::int32_t se() {
        const std::uint32_t k = ue();
        return (k & 1) ? static_cast<std::int32_t>((k + 1) / 2) : -static_cast<std::int32_t>(k / 2);
    }
    bool ok() const { return pos_ <= n_ * 8; }
    bool more() const { return pos_ < n_ * 8; }

private:
    std::uint32_t bit() {
        if (pos_ >= n_ * 8) {
            ++pos_;
            return 0;
        }
        const std::uint32_t b = (p_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1u;
        ++pos_;
        return b;
    }
    const std::uint8_t* p_;
    std::size_t n_, pos_ = 0;
};

// NAL turinys be emuliacijos apsaugos baitų (00 00 03 → 00 00), be antraštės baito.
inline std::vector<std::uint8_t> rbsp(const std::uint8_t* p, std::size_t n) {
    std::vector<std::uint8_t> o;
    o.reserve(n);
    int zeros = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (zeros >= 2 && p[i] == 3) {
            zeros = 0;
            continue;
        }
        zeros = p[i] == 0 ? zeros + 1 : 0;
        o.push_back(p[i]);
    }
    return o;
}

struct Sps {
    bool valid = false, frame_mbs_only = true, pic_struct_present = false, hrd = false;
    std::uint32_t log2_max_frame_num = 4, poc_type = 0, log2_max_poc_lsb = 4, cpb_len = 24, dpb_len = 24;
};
struct Pps {
    bool valid = false, bottom_poc_present = false;
    std::uint32_t sps_id = 0;
};

inline void skip_scaling_list(Bits& b, int size) {
    int last = 8, next = 8;
    for (int j = 0; j < size; ++j) {
        if (next != 0) next = (last + b.se() + 256) % 256;
        last = next == 0 ? last : next;
    }
}

inline void hrd_parameters(Bits& b, Sps& s) {
    const std::uint32_t cnt = b.ue() + 1;
    b.u(4);
    b.u(4);
    for (std::uint32_t i = 0; i < cnt && i < 32; ++i) {
        b.ue();
        b.ue();
        b.u(1);
    }
    b.u(5);  // initial_cpb_removal_delay_length
    s.cpb_len = b.u(5) + 1;
    s.dpb_len = b.u(5) + 1;
    b.u(5);  // time_offset_length
    s.hrd = true;
}

inline void vui(Bits& b, Sps& s) {
    if (b.u(1) && b.u(8) == 255) b.u(32);  // aspect_ratio_idc, Extended_SAR
    if (b.u(1)) b.u(1);                    // overscan
    if (b.u(1)) {                          // video_signal_type
        b.u(4);
        if (b.u(1)) b.u(24);
    }
    if (b.u(1)) {  // chroma_loc
        b.ue();
        b.ue();
    }
    if (b.u(1)) {  // timing_info
        b.u(32);
        b.u(32);
        b.u(1);
    }
    const bool nal = b.u(1) != 0;
    if (nal) hrd_parameters(b, s);
    const bool vcl = b.u(1) != 0;
    if (vcl) hrd_parameters(b, s);
    if (nal || vcl) b.u(1);  // low_delay_hrd
    s.pic_struct_present = b.u(1) != 0;
}

inline Sps parse_sps(const std::vector<std::uint8_t>& r, std::uint32_t& id) {
    Sps s;
    Bits b(r.data(), r.size());
    const std::uint32_t profile = b.u(8);
    b.u(16);  // constraint flags + level
    id = b.ue();
    if (profile == 100 || profile == 110 || profile == 122 || profile == 244 || profile == 44 || profile == 83 || profile == 86 ||
        profile == 118 || profile == 128 || profile == 138 || profile == 139 || profile == 134 || profile == 135) {
        const std::uint32_t chroma = b.ue();
        if (chroma == 3) b.u(1);
        b.ue();  // bit_depth_luma
        b.ue();  // bit_depth_chroma
        b.u(1);  // qpprime
        if (b.u(1))  // seq_scaling_matrix_present
            for (int i = 0; i < (chroma != 3 ? 8 : 12); ++i)
                if (b.u(1)) skip_scaling_list(b, i < 6 ? 16 : 64);
    }
    s.log2_max_frame_num = b.ue() + 4;
    s.poc_type = b.ue();
    if (s.poc_type == 0) {
        s.log2_max_poc_lsb = b.ue() + 4;
    } else if (s.poc_type == 1) {
        b.u(1);
        b.se();
        b.se();
        const std::uint32_t n = b.ue();
        for (std::uint32_t i = 0; i < n && i < 256; ++i) b.se();
    }
    b.ue();  // max_num_ref_frames
    b.u(1);  // gaps_in_frame_num_allowed
    b.ue();  // pic_width_in_mbs_minus1
    b.ue();  // pic_height_in_map_units_minus1
    s.frame_mbs_only = b.u(1) != 0;
    if (!s.frame_mbs_only) b.u(1);  // mb_adaptive_frame_field
    b.u(1);                          // direct_8x8_inference
    if (b.u(1)) {                    // frame_cropping
        b.ue();
        b.ue();
        b.ue();
        b.ue();
    }
    if (b.u(1)) vui(b, s);
    s.valid = b.ok();
    return s;
}

inline Pps parse_pps(const std::vector<std::uint8_t>& r, std::uint32_t& id) {
    Pps p;
    Bits b(r.data(), r.size());
    id = b.ue();
    p.sps_id = b.ue();
    b.u(1);  // entropy_coding_mode
    p.bottom_poc_present = b.u(1) != 0;
    p.valid = b.ok();
    return p;
}

// Annex B NAL vienetai buferyje: kiekvienam — (pradžia po starto kodo, ilgis).
template <class F>
void for_each_nal(const std::uint8_t* p, std::size_t n, F&& f) {
    std::size_t i = 0, start = SIZE_MAX;
    while (i + 3 <= n) {
        if (p[i] == 0 && p[i + 1] == 0 && p[i + 2] == 1) {
            if (start != SIZE_MAX) {
                std::size_t end = i;
                while (end > start && p[end - 1] == 0) --end;  // 4 baitų starto kodo nulis
                f(p + start, end - start);
            }
            i += 3;
            start = i;
            continue;
        }
        ++i;
    }
    if (start != SIZE_MAX && start < n) f(p + start, n - start);
}

}  // namespace h264

// SPS / PPS / SEI kaupiami iš eilės paduodamų buferių (extradata, paketai); pirmas IDR pjūvis nulemia rezultatą.
class H264Scan {
public:
    // Grąžina true, kai rezultatas žinomas (scan()).
    bool feed(const std::uint8_t* p, std::size_t n) {
        h264::for_each_nal(p, n, [&](const std::uint8_t* nal, std::size_t len) {
            if (done_ || len < 2) return;
            const int type = nal[0] & 0x1F;
            if (type == 7) {
                std::uint32_t id = 0;
                const h264::Sps s = h264::parse_sps(h264::rbsp(nal, len), id);
                if (s.valid && id < 32) {
                    sps_[id] = s;
                    last_sps_ = static_cast<int>(id);
                }
            } else if (type == 8) {
                std::uint32_t id = 0;
                const h264::Pps q = h264::parse_pps(h264::rbsp(nal, len), id);
                if (q.valid && id < 256) pps_[id] = q;
            } else if (type == 6) {
                sei(h264::rbsp(nal, len));
            } else if (type == 9 || type == 1) {
                pic_struct_ = -1;  // nauja prieigos dalis — senas pic_struct nebegalioja
            } else if (type == 5) {
                slice(h264::rbsp(nal, len));
            }
        });
        return done_;
    }
    Scan scan() const { return scan_; }
    bool done() const { return done_; }
    int pic_struct() const { return decided_ps_; }  // IDR prieigos dalies pic_struct (< 0 — nebuvo)

private:
    void sei(const std::vector<std::uint8_t>& r) {
        if (last_sps_ < 0) return;
        const h264::Sps& s = sps_[last_sps_];
        std::size_t i = 0;
        while (i + 2 <= r.size()) {
            std::uint32_t type = 0, size = 0;
            while (i < r.size() && r[i] == 0xFF) type += 255, ++i;
            if (i >= r.size()) return;
            type += r[i++];
            while (i < r.size() && r[i] == 0xFF) size += 255, ++i;
            if (i >= r.size()) return;
            size += r[i++];
            if (i + size > r.size()) return;
            if (type == 1 && s.pic_struct_present) {  // pic_timing
                h264::Bits b(r.data() + i, size);
                if (s.hrd) {
                    b.u(static_cast<int>(s.cpb_len));
                    b.u(static_cast<int>(s.dpb_len));
                }
                pic_struct_ = static_cast<int>(b.u(4));
            }
            i += size;
            if (i < r.size() && r[i] == 0x80) return;  // rbsp_trailing_bits
        }
    }
    void slice(const std::vector<std::uint8_t>& r) {
        h264::Bits b(r.data(), r.size());
        b.ue();  // first_mb_in_slice
        b.ue();  // slice_type
        const std::uint32_t pps_id = b.ue();
        if (pps_id >= 256 || !pps_[pps_id].valid) return;
        const h264::Pps& p = pps_[pps_id];
        if (p.sps_id >= 32 || !sps_[p.sps_id].valid) return;
        const h264::Sps& s = sps_[p.sps_id];
        done_ = true;
        decided_ps_ = pic_struct_;
        if (s.frame_mbs_only) {
            scan_ = Scan::Progressive;
            return;
        }
        if (pic_struct_ == 3 || pic_struct_ == 5) {
            scan_ = Scan::Tff;
            return;
        }
        if (pic_struct_ == 4 || pic_struct_ == 6) {
            scan_ = Scan::Bff;
            return;
        }
        b.u(static_cast<int>(s.log2_max_frame_num));  // frame_num
        if (b.u(1)) {                                  // field_pic_flag: pirmas koduotas IDR laukas
            scan_ = b.u(1) ? Scan::Bff : Scan::Tff;
            return;
        }
        b.ue();  // idr_pic_id
        if (s.poc_type == 0 && p.bottom_poc_present) {
            b.u(static_cast<int>(s.log2_max_poc_lsb));  // pic_order_cnt_lsb
            const std::int32_t d = b.se();               // apatinio lauko POC − viršutinio
            if (d != 0) {
                scan_ = d > 0 ? Scan::Tff : Scan::Bff;
                return;
            }
        }
        scan_ = Scan::Tff;  // interlaced kadras be kitų signalų — numatytoji TFF
    }
    h264::Sps sps_[32];
    h264::Pps pps_[256];
    int last_sps_ = -1, pic_struct_ = -1, decided_ps_ = -1;
    bool done_ = false;
    Scan scan_ = Scan::Unknown;
};

}  // namespace mxlv::ingest
