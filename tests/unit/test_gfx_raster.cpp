// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Raster aprašas, v210a key pakavimas ir flow aprašo skaitymas (alfa kompozitoriaus 2 užduotis).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gfx/raster.hpp"

using namespace vramflow::gfx;

TEST_CASE("gfx: eilučių ilgiai", "[gfx]") {
    CHECK(fill_line_bytes(1920) == 5120);
    CHECK(fill_line_bytes(1280) == 3456);
    CHECK(fill_line_bytes(384) == 1024);
    CHECK(key_line_bytes(1920) == 2560);
    CHECK(key_line_bytes(1280) == 1708);
    CHECK(key_line_bytes(382) == 512);
}

TEST_CASE("gfx: 1080i50 v210a grain dydžiai", "[gfx]") {
    Raster r = kHd1080i;
    CHECK(grain_lines(r) == 540);
    CHECK(fill_bytes(r) == 2764800);
    r.alpha = true;
    CHECK(key_bytes(r) == 1382400);
    CHECK(grain_bytes(r) == 4147200);
    CHECK(grain_bytes(kHd1080i) == 2764800);  // be alfa — tik fill
    Raster p{1280, 720, false, true};
    CHECK(grain_lines(p) == 720);
}

TEST_CASE("gfx: key_set/key_get [PF-2]", "[gfx]") {
    for (int w : {1920, 1280, 382, 384}) {
        const std::size_t words = key_line_bytes(w) / 4;
        std::vector<std::uint32_t> line(words, 0);
        for (int x = 0; x < w; ++x) key_set(line.data(), x, static_cast<std::uint32_t>((x * 37 + 5) & 0x3FF));
        for (int x = 0; x < w; ++x) CHECK(key_get(line.data(), x) == static_cast<std::uint32_t>((x * 37 + 5) & 0x3FF));
        for (std::uint32_t v : line) CHECK((v >> 30) == 0);
        // nenaudoti paskutinio bloko mėginiai = 0
        const int used = w % 3;
        if (used != 0) {
            const std::uint32_t last = line.back();
            for (int s = used; s < 3; ++s) CHECK(((last >> (10 * s)) & 0x3FF) == 0);
        }
    }
    // w = 1280: 426 pilni blokai + 2 mėginiai → bitai 20–31 paskutiniame bloke = 0
    std::vector<std::uint32_t> line(key_line_bytes(1280) / 4, 0);
    for (int x = 0; x < 1280; ++x) key_set(line.data(), x, 0x3FF);
    CHECK((line.back() >> 20) == 0);
    // key_set perrašo, nesugadindamas kaimynų
    key_set(line.data(), 4, 0x155);
    CHECK(key_get(line.data(), 3) == 0x3FF);
    CHECK(key_get(line.data(), 4) == 0x155);
    CHECK(key_get(line.data(), 5) == 0x3FF);
}

TEST_CASE("gfx: raster_from_flow_def", "[gfx]") {
    std::ifstream f(std::string(MXLV_TEST_DATA_DIR) + "/v210a_flow.json");
    REQUIRE(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    Raster r{};
    std::string err;
    REQUIRE(raster_from_flow_def(ss.str(), r, err));
    CHECK(r.w == 1920);
    CHECK(r.h == 1080);
    CHECK_FALSE(r.interlaced);
    CHECK(r.alpha);

    const std::string i50 =
        R"({"media_type":"video/v210","frame_width":1920,"frame_height":1080,"interlace_mode":"interlaced_tff"})";
    REQUIRE(raster_from_flow_def(i50, r, err));
    CHECK((r.w == 1920 && r.h == 1080 && r.interlaced && !r.alpha));

    // interlace_mode: tik progressive / nenurodyta / interlaced_tff; BFF ir nežinomos reikšmės atmetamos
    const auto im = [](const char* m) {
        return std::string(R"({"media_type":"video/v210a","frame_width":8,"frame_height":4,"interlace_mode":")") + m + R"("})";
    };
    CHECK(raster_from_flow_def(im("progressive"), r, err));
    CHECK_FALSE(r.interlaced);
    CHECK(raster_from_flow_def(im("interlaced_tff"), r, err));
    CHECK(r.interlaced);
    err.clear();
    CHECK_FALSE(raster_from_flow_def(im("interlaced_bff"), r, err));
    CHECK(err.find("interlaced_bff") != std::string::npos);
    CHECK_FALSE(raster_from_flow_def(im("segmented"), r, err));
    CHECK_FALSE(raster_from_flow_def(im(""), r, err));
    CHECK(raster_from_flow_def(R"({"media_type":"video/v210a","frame_width":8,"frame_height":4})", r, err));  // nenurodyta = progresyvus
    CHECK_FALSE(r.interlaced);

    CHECK_FALSE(raster_from_flow_def(R"({"media_type":"video/h264","frame_width":1,"frame_height":2})", r, err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(raster_from_flow_def(R"({"media_type":"video/v210","frame_height":2})", r, err));
}

TEST_CASE("gfx: composite_job_valid — vienas patikrinimas visiems variantams (R13)", "[gfx]") {
    std::uint8_t buf[64] = {};
    const Raster ia{8, 8, true, true}, iv{8, 8, true, false}, pa{8, 8, false, true}, lr{4, 4, true, true};
    Layer L{buf, lr, 2, 4, 1.0f};
    std::string why;
    auto job = [&](const std::uint8_t* base, const Raster& br, const Layer* l, int k, const Raster& o, std::uint8_t* out) {
        return CompositeJob{base, br, l, k, o, out};
    };
    CHECK(composite_job_valid(job(buf, ia, &L, 1, ia, buf), 0));
    CHECK(composite_job_valid(job(nullptr, ia, &L, 1, ia, buf), 1));  // be bazės + v210a — gerai
    CHECK(composite_job_valid(job(buf, iv, nullptr, 0, iv, buf), 0));  // v210 su bazė — gerai
    CHECK(composite_job_valid(job(nullptr, ia, nullptr, 0, ia, buf), 7) == false);  // paritetas interlaced
    CHECK(composite_job_valid(job(nullptr, pa, nullptr, 0, pa, buf), 7));           // progresyviam ignoruojamas

    // R12: be bazės — v210 išėjimas netinkamas
    CHECK_FALSE(composite_job_valid(job(nullptr, iv, &L, 1, iv, buf), 0, &why));
    CHECK_FALSE(why.empty());
    CHECK_FALSE(composite_job_valid(job(nullptr, iv, nullptr, 0, iv, buf), 0));

    CHECK_FALSE(composite_job_valid(job(buf, ia, &L, 1, ia, nullptr), 0));          // out == nullptr
    CHECK_FALSE(composite_job_valid(job(buf, ia, &L, -1, ia, buf), 0));             // k < 0
    CHECK_FALSE(composite_job_valid(job(buf, ia, nullptr, 1, ia, buf), 0));         // layers == nullptr
    CHECK_FALSE(composite_job_valid(job(buf, Raster{6, 8, true, true}, &L, 1, ia, buf), 0));  // bazės rastras kitoks
    CHECK_FALSE(composite_job_valid(job(buf, pa, &L, 1, ia, buf), 0));              // bazės interlace kitoks
    CHECK_FALSE(composite_job_valid(job(buf, ia, &L, 1, Raster{7, 8, true, true}, buf), 0));  // nelyginis plotis
    Layer b = L;
    b.grain = nullptr;
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.x = 3;
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.x = -2;
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.y = -2;
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.y = 3;  // nelyginis y interlaced (R1)
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.opacity = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.opacity = std::numeric_limits<float>::infinity();
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.opacity = -std::numeric_limits<float>::infinity();
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    b = L; b.r = Raster{4, 4, false, true};  // sluoksnio interlace kitoks
    CHECK_FALSE(composite_job_valid(job(buf, ia, &b, 1, ia, buf), 0));
    // progresyvus: nelyginis y leidžiamas
    Layer p{buf, Raster{4, 4, false, true}, 2, 3, 1.0f};
    CHECK(composite_job_valid(job(buf, pa, &p, 1, pa, buf), 0));
}
