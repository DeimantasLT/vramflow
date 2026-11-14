// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// H.264 skenavimas ir laukų tvarka iš IDR prieigos dalies (ingest/h264_scan.hpp): sintetinis bitų srautas (SPS su VUI / HRD,
// PPS, pic_timing SEI, IDR pjūvio antraštė).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "ingest/h264_scan.hpp"

using namespace mxlv::ingest;

namespace {
struct BW {
    std::vector<std::uint8_t> b;
    int bits = 0;
    void put(std::uint32_t v, int k) {
        for (int i = k - 1; i >= 0; --i) {
            if (bits % 8 == 0) b.push_back(0);
            if ((v >> i) & 1u) b.back() |= static_cast<std::uint8_t>(1u << (7 - bits % 8));
            ++bits;
        }
    }
    void ue(std::uint32_t v) {
        const std::uint32_t x = v + 1;
        int len = 0;
        while ((x >> len) > 1) ++len;
        put(0, len);
        put(x, len + 1);
    }
    void se(std::int32_t v) { ue(v > 0 ? static_cast<std::uint32_t>(2 * v - 1) : static_cast<std::uint32_t>(-2 * v)); }
    void trail() {
        put(1, 1);
        while (bits % 8) put(0, 1);
    }
};

// NAL su starto kodu; emuliacijos apsauga (00 00 0x ≤ 3 → 00 00 03 0x)
void nal(std::vector<std::uint8_t>& out, int type, int ref, const std::vector<std::uint8_t>& payload) {
    out.insert(out.end(), {0, 0, 0, 1});
    out.push_back(static_cast<std::uint8_t>((ref << 5) | type));
    int zeros = 0;
    for (std::uint8_t c : payload) {
        if (zeros >= 2 && c <= 3) {
            out.push_back(3);
            zeros = 0;
        }
        out.push_back(c);
        zeros = c == 0 ? zeros + 1 : 0;
    }
}

struct Opt {
    bool frame_mbs_only = false, field_pic = false, bottom_first = false, high = true, vui = true;
    int pic_struct = -1;  // < 0 — be pic_timing SEI (ir be pic_struct_present_flag)
    int delta = -1;       // delta_pic_order_cnt_bottom (kadrui)
};

std::vector<std::uint8_t> stream(const Opt& o) {
    std::vector<std::uint8_t> s;
    nal(s, 9, 0, {0xF0});  // AUD
    BW sps;
    sps.put(o.high ? 100 : 77, 8);
    sps.put(0, 8);
    sps.put(40, 8);
    sps.ue(0);  // sps_id
    if (o.high) {
        sps.ue(1);  // chroma 4:2:0
        sps.ue(0);
        sps.ue(0);
        sps.put(0, 1);
        sps.put(1, 1);  // seq_scaling_matrix_present — vienas sąrašas (patikrina praleidimą)
        sps.put(1, 1);
        for (int j = 0; j < 16; ++j) sps.se(j == 0 ? 8 : 0);
        for (int i = 1; i < 8; ++i) sps.put(0, 1);
    }
    sps.ue(0);  // log2_max_frame_num − 4
    sps.ue(0);  // poc type 0
    sps.ue(2);  // log2_max_poc_lsb − 4
    sps.ue(4);  // max refs
    sps.put(0, 1);
    sps.ue(119);
    sps.ue(o.frame_mbs_only ? 67 : 33);
    sps.put(o.frame_mbs_only ? 1 : 0, 1);
    if (!o.frame_mbs_only) sps.put(1, 1);  // mb_adaptive_frame_field
    sps.put(1, 1);                          // direct_8x8
    sps.put(1, 1);                          // frame_cropping
    sps.ue(0);
    sps.ue(0);
    sps.ue(0);
    sps.ue(4);
    const bool ps = o.pic_struct >= 0;
    sps.put(o.vui ? 1 : 0, 1);
    if (o.vui) {
        sps.put(1, 1);  // aspect_ratio_info
        sps.put(1, 8);  // 1:1
        sps.put(0, 1);  // overscan
        sps.put(1, 1);  // video_signal_type
        sps.put(5, 3);
        sps.put(0, 1);
        sps.put(1, 1);
        sps.put(1, 8);
        sps.put(1, 8);
        sps.put(1, 8);
        sps.put(0, 1);   // chroma_loc
        sps.put(1, 1);   // timing_info
        sps.put(1, 32);
        sps.put(50, 32);
        sps.put(1, 1);
        sps.put(1, 1);   // nal_hrd
        sps.ue(0);       // cpb_cnt − 1
        sps.put(4, 4);
        sps.put(6, 4);
        sps.ue(1000);
        sps.ue(2000);
        sps.put(1, 1);
        sps.put(23, 5);  // initial_cpb_removal_delay_length − 1
        sps.put(23, 5);  // cpb_removal_delay_length − 1 → 24
        sps.put(23, 5);  // dpb_output_delay_length − 1 → 24
        sps.put(24, 5);
        sps.put(0, 1);   // vcl_hrd
        sps.put(0, 1);   // low_delay
        sps.put(ps ? 1 : 0, 1);
        sps.put(0, 1);   // bitstream_restriction
    }
    sps.trail();
    nal(s, 7, 3, sps.b);
    BW pps;
    pps.ue(0);
    pps.ue(0);
    pps.put(1, 1);
    pps.put(1, 1);
    pps.trail();
    nal(s, 8, 3, pps.b);
    if (ps && o.vui) {
        BW pt;  // pic_timing: cpb_removal_delay (24), dpb_output_delay (24), pic_struct (4), clock_timestamp_flag × n = 0
        pt.put(0, 24);
        pt.put(2, 24);
        pt.put(static_cast<std::uint32_t>(o.pic_struct), 4);
        pt.put(0, o.pic_struct >= 5 ? 3 : (o.pic_struct >= 3 ? 2 : 1));
        while (pt.bits % 8) pt.put(0, 1);
        BW sei;
        sei.put(1, 8);
        sei.put(static_cast<std::uint32_t>(pt.b.size()), 8);
        for (std::uint8_t c : pt.b) sei.put(c, 8);
        sei.trail();
        nal(s, 6, 0, sei.b);
    }
    BW sl;
    sl.ue(0);
    sl.ue(7);
    sl.ue(0);
    sl.put(0, 4);  // frame_num
    if (!o.frame_mbs_only) {
        sl.put(o.field_pic ? 1 : 0, 1);
        if (o.field_pic) sl.put(o.bottom_first ? 1 : 0, 1);
    }
    sl.ue(0);      // idr_pic_id
    sl.put(1, 6);  // poc lsb
    if (!o.field_pic) sl.se(o.delta);
    sl.trail();
    nal(s, 5, 3, sl.b);
    return s;
}
}  // namespace

TEST_CASE("h264_scan: laukų tvarka iš IDR prieigos dalies", "[ingest][h264_scan]") {
    struct Case {
        Opt o;
        Scan want;
    };
    Opt prog;
    prog.frame_mbs_only = true;
    Opt tff_ps, bff_ps, tff_ps3, bff_ps3, poc_bff, poc_tff, poc_none, field_top, field_bottom, novui;
    tff_ps.pic_struct = 3;  // pic_struct viršesnis už POC (delta −1)
    bff_ps.pic_struct = 4;
    bff_ps.delta = 1;
    tff_ps3.pic_struct = 5;
    bff_ps3.pic_struct = 6;
    poc_tff.delta = 1;  // be pic_struct — POC
    poc_none.delta = 0;
    field_top.field_pic = true;
    field_bottom.field_pic = field_bottom.bottom_first = true;
    novui.vui = false;
    novui.delta = 1;
    const Case cases[] = {{prog, Scan::Progressive}, {tff_ps, Scan::Tff},       {bff_ps, Scan::Bff},       {tff_ps3, Scan::Tff},
                          {bff_ps3, Scan::Bff},      {poc_bff, Scan::Bff},      {poc_tff, Scan::Tff},      {poc_none, Scan::Tff},
                          {field_top, Scan::Tff},    {field_bottom, Scan::Bff}, {novui, Scan::Tff}};
    int k = 0;
    for (const Case& c : cases) {
        for (const bool high : {true, false}) {
            INFO("atvejis " << k << " high " << high);
            Opt o = c.o;
            o.high = high;
            H264Scan s;
            const auto bs = stream(o);
            CHECK(s.feed(bs.data(), bs.size()));
            CHECK(s.scan() == c.want);
        }
        ++k;
    }
}

TEST_CASE("h264_scan: be SPS / PPS — nežinoma, kol jų nėra", "[ingest][h264_scan]") {
    Opt o;
    o.pic_struct = 4;
    const auto bs = stream(o);
    std::size_t idr = 0;
    for (std::size_t i = bs.size() - 5; i > 0; --i)
        if (bs[i] == 0 && bs[i + 1] == 0 && bs[i + 2] == 0 && bs[i + 3] == 1) {
            idr = i;
            break;
        }
    H264Scan s;
    CHECK_FALSE(s.feed(bs.data() + idr, bs.size() - idr));  // tik IDR
    CHECK(s.scan() == Scan::Unknown);
    CHECK(s.feed(bs.data(), bs.size()));  // dabar su SPS / PPS / SEI
    CHECK(s.scan() == Scan::Bff);
}
