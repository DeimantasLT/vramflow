// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// common/format.hpp (planas docs/planai/2026-11-11-production-grade.md §P2): formato aprašas, grain'o dažnis, vardų
// skaitymas ir atmetimas, grain'o dydžiai, garso mėginių ribos grain'ui [PF-5], flow aprašo pirmyn-atgal.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

#include "common/format.hpp"

using namespace vramflow;

namespace {
VideoFormat parsed(const char* s) {
    VideoFormat f{};
    std::string err;
    INFO(s);
    REQUIRE(parse_format(s, f, err));
    CHECK(err.empty());
    return f;
}
bool same(Rational a, Rational b) { return a.num == b.num && a.den == b.den; }
}  // namespace

TEST_CASE("format: grain_rate", "[format]") {
    CHECK(same(parsed("1080i50").grain_rate(), {50, 1}));
    CHECK(same(parsed("1080p25").grain_rate(), {25, 1}));
    CHECK(same(parsed("1080p50").grain_rate(), {50, 1}));
    CHECK(same(parsed("1080p59.94").grain_rate(), {60000, 1001}));
    CHECK(same(parsed("1080i59.94").grain_rate(), {60000, 1001}));
    CHECK(same(parsed("1080p29.97").grain_rate(), {30000, 1001}));
    CHECK(same(kFmt1080i50.grain_rate(), {50, 1}));
    CHECK(same(kFmt1080i50.frame_rate, {25, 1}));
    // interlaced: kadrų dažnis = pusė laukų dažnio
    CHECK(same(parsed("1080i59.94").frame_rate, {30000, 1001}));
    CHECK(parsed("1080i50") == kFmt1080i50);
}

TEST_CASE("format: parse / atmetimas", "[format]") {
    const VideoFormat p25 = parsed("1080p25");
    CHECK(p25.raster.w == 1920);
    CHECK(p25.raster.h == 1080);
    CHECK_FALSE(p25.raster.interlaced);
    CHECK_FALSE(p25.raster.alpha);
    const VideoFormat p720 = parsed("720p50");
    CHECK(p720.raster.w == 1280);
    CHECK(p720.raster.h == 720);
    const VideoFormat uhd = parsed("2160p50");
    CHECK(uhd.raster.w == 3840);
    CHECK(uhd.raster.h == 2160);
    for (const char* bad : {"1080i25", "1081p50", "1080p24", "720i50", "2160i50", "", "1080", "p50", "1080x50", "1080p50x",
                            "1080i60", "-1080p50"}) {
        VideoFormat f{};
        std::string err;
        INFO(bad);
        CHECK_FALSE(parse_format(bad, f, err));
        CHECK_FALSE(err.empty());
    }
    // klaidos tekstas pasako, kas palaikoma
    VideoFormat f{};
    std::string err;
    CHECK_FALSE(parse_format("1080i25", f, err));
    CHECK(err.find("1080i25") != std::string::npos);
    // vardas pirmyn-atgal
    for (const char* s : {"720p25", "720p50", "720p29.97", "720p59.94", "1080i50", "1080i59.94", "1080p25", "1080p50",
                          "1080p29.97", "1080p59.94", "2160p25", "2160p50", "2160p29.97", "2160p59.94"})
        CHECK(format_name(parsed(s)) == s);
}

TEST_CASE("format: format_supported", "[format]") {
    std::string why;
    CHECK(format_supported(kFmt1080i50, why));
    VideoFormat bff = kFmt1080i50;
    bff.raster.h = 1081;
    CHECK_FALSE(format_supported(bff, why));
    CHECK_FALSE(why.empty());
    VideoFormat r24 = parsed("1080p25");
    r24.frame_rate = {24, 1};
    CHECK_FALSE(format_supported(r24, why));
    VideoFormat i720 = parsed("720p50");
    i720.raster.interlaced = true;
    i720.frame_rate = {25, 1};
    CHECK_FALSE(format_supported(i720, why));
    // chromos padėtis: įgyvendintas tik 0 tipas (kiti — klaida, ne tylus neteisingas rezultatas)
    VideoFormat c1 = kFmt1080i50;
    c1.siting = ChromaSiting420::Type1Center;
    CHECK_FALSE(format_supported(c1, why));
    CHECK(why.find("chrom") != std::string::npos);
}

TEST_CASE("format: grain dydžiai", "[format]") {
    CHECK(gfx::grain_bytes(parsed("1080p25").raster) == 5'529'600);
    CHECK(gfx::grain_bytes(parsed("720p50").raster) == 2'488'320);
    CHECK(gfx::grain_bytes(parsed("2160p50").raster) == 22'118'400);
    CHECK(gfx::grain_bytes(kFmt1080i50.raster) == 2'764'800);  // laukas
}

TEST_CASE("format: sample_begin 60000/1001", "[format][PF-5]") {
    constexpr Rational r5994{60000, 1001};
    const std::uint64_t want[] = {0, 801, 1602, 2402, 3203, 4004};
    for (std::uint64_t i = 0; i < 6; ++i) CHECK(sample_begin(r5994, i) == want[i]);
    CHECK(sample_begin(r5994, 60'000) == 48'048'000);
    static_assert(sample_begin(Rational{60000, 1001}, 5) == 4004);
    std::uint64_t bad50 = 0, bad25 = 0;
    for (std::uint64_t i = 0; i < 1'000'000; ++i) {
        bad50 += sample_begin({50, 1}, i) != 960 * i;
        bad25 += sample_begin({25, 1}, i) != 1920 * i;
    }
    CHECK(bad50 == 0);
    CHECK(bad25 == 0);
    // 59,94: grain'ų ilgiai 800 arba 801, 5 grain'ų suma visada 4004 (48000·5·1001/60000)
    std::uint64_t bad_len = 0;
    for (std::uint64_t i = 0; i < 100'000; ++i) {
        const std::uint64_t n = sample_begin(r5994, i + 1) - sample_begin(r5994, i);
        bad_len += n != 800 && n != 801;
        bad_len += sample_begin(r5994, 5 * i + 5) - sample_begin(r5994, 5 * i) != 4004;
    }
    CHECK(bad_len == 0);
    CHECK(max_samples_per_grain({50, 1}) == 960);
    CHECK(max_samples_per_grain({25, 1}) == 1920);
    CHECK(max_samples_per_grain(r5994) == 801);
}

TEST_CASE("format: flow aprašas pirmyn-atgal", "[format]") {
    for (const char* s : {"720p25", "720p50", "720p29.97", "720p59.94", "1080i50", "1080i59.94", "1080p25", "1080p50",
                          "1080p29.97", "1080p59.94", "2160p25", "2160p50", "2160p29.97", "2160p59.94"}) {
        INFO(s);
        const VideoFormat f = parsed(s);
        const std::string def = video_flow_def("4d584c47-0001-4000-8000-000000000000", "t", f, "g:Video 0");
        VideoFormat back{};
        std::string err;
        REQUIRE(format_from_flow_def(def, back, err));
        CHECK(back == f);
        CHECK(format_supported(back, err));
    }
    // MXL konvencija: interlaced flow apraše grain_rate = kadrų dažnis (MXL grain'ų ritmas dvigubas)
    const std::string def = video_flow_def("4d584c47-0001-4000-8000-000000000000", "t", kFmt1080i50, "g");
    CHECK(def.find(R"("grain_rate":{"numerator":25,"denominator":1})") != std::string::npos);
    CHECK(def.find(R"("interlace_mode":"interlaced_tff")") != std::string::npos);
    VideoFormat back{};
    std::string err;
    CHECK_FALSE(format_from_flow_def(R"({"media_type":"video/v210","frame_width":1920,"frame_height":1080})", back, err));
    CHECK(err.find("grain_rate") != std::string::npos);
}
