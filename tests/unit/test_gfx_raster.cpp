// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Raster aprašas, v210a key pakavimas ir flow aprašo skaitymas (alfa kompozitoriaus 2 užduotis).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
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

    CHECK_FALSE(raster_from_flow_def(R"({"media_type":"video/h264","frame_width":1,"frame_height":2})", r, err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(raster_from_flow_def(R"({"media_type":"video/v210","frame_height":2})", r, err));
}
