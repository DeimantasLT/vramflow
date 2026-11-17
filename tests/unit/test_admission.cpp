// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Priėmimo kontrolė (planas §P4): sprendimo ribos (NVDEC, NVENC sesijos, VRAM), talpos failas (schema 1, UUID, tvarkyklė,
// lower_bound) ir ingest srauto paklausa.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "runtime/admission.hpp"

using namespace mxlv::runtime;
using Catch::Approx;

namespace {
const std::string kUuid = "GPU-be7b6e9b-27ec-ac11-92d2-d91c03de8d4f";  // tests/unit/data/capacity-<uuid>.json (RTX 4070 Laptop, P8)
const std::string kData = MXLV_TEST_DATA_DIR;
constexpr double k1080 = 1920.0 * 1080 * 25 / 1e6;  // 51,84 Mpix/s

std::string fixture() {
    std::ifstream f(kData + "/capacity-" + kUuid + ".json");
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
std::string replace(std::string s, const std::string& a, const std::string& b) {
    const std::size_t p = s.find(a);
    REQUIRE(p != std::string::npos);
    return s.replace(p, a.size(), b);
}
}  // namespace

TEST_CASE("admission: ribos — 16-tas 1080p25 priimamas, 17-tas atmetamas dėl nvdec", "[admission]") {
    Capacity c;
    c.nvdec_mpix_s = 1000;
    c.headroom = 0.85;
    Demand one;
    one.nvdec_mpix_s = k1080;
    Demand used;
    for (int n = 1; n <= 16; ++n) {
        const Decision d = decide(used, one, c);
        INFO(n);
        REQUIRE(d.admit);
        used = used + one;
    }
    CHECK(used.nvdec_mpix_s == Approx(829.44));
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "nvdec");
    CHECK(d.limit == Approx(850));
    CHECK(d.used == Approx(829.44));
    CHECK(d.requested == Approx(51.84));
    const std::string t = describe(d);
    CHECK(t.find("nvdec") != std::string::npos);
    CHECK(t.find("829,44") != std::string::npos);
    CHECK(t.find("51,84") != std::string::npos);
    CHECK(t.find("850,00") != std::string::npos);
}

TEST_CASE("admission: VRAM riboja pirmiau — 6 GiB, 135 MiB srautui: 38-tas priimamas, 39-tas atmetamas", "[admission]") {
    Capacity c;
    c.nvdec_mpix_s = 1e6;  // NVDEC neriboja
    c.vram_bytes = std::uint64_t{6} << 30;
    Demand one;
    one.nvdec_mpix_s = k1080;
    one.vram_bytes = std::uint64_t{135} << 20;
    Demand used;
    for (int n = 1; n <= 38; ++n) {
        INFO(n);
        REQUIRE(decide(used, one, c).admit);
        used = used + one;
    }
    CHECK(used.vram_bytes == (std::uint64_t{5130} << 20));  // 5130 MB ≤ 5222 MB
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "vram");
    CHECK(d.limit == Approx(0.85 * static_cast<double>(std::uint64_t{6} << 30)));
    CHECK(describe(d).find("MB") != std::string::npos);
}

TEST_CASE("admission: NVENC sesijos — atskiras resursas, 13-ta atmetama, kai Mpix/s dar leistų", "[admission]") {
    Capacity c;
    c.nvenc_mpix_s = 100000;  // pralaidumas neriboja
    c.nvenc_sessions = 12;
    Demand one;
    one.nvenc_mpix_s = k1080;
    one.nvenc_sessions = 1;
    Demand used;
    for (int n = 1; n <= 12; ++n) {
        INFO(n);
        REQUIRE(decide(used, one, c).admit);  // headroom sesijoms netaikomas: 12, ne floor(0,85 · 12) = 10
        used = used + one;
    }
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "nvenc_sessions");
    CHECK(d.limit == Approx(12));
    CHECK(d.used == Approx(12));
    CHECK(d.requested == Approx(1));
}

TEST_CASE("admission: nvenc Mpix/s riboja prieš sesijas; nežinoma talpa (0) netikrinama", "[admission]") {
    Capacity c;
    c.nvenc_mpix_s = 200;  // riba 170: 3 srautai (155,52) telpa, 4-as — ne
    c.nvenc_sessions = 12;
    Demand one;
    one.nvenc_mpix_s = k1080;
    one.nvenc_sessions = 1;
    one.nvdec_mpix_s = 1e9;          // NVDEC talpa nežinoma — netikrinama
    one.vram_bytes = ~std::uint64_t{0} >> 1;  // VRAM talpa nežinoma — netikrinama
    Demand used;
    for (int n = 1; n <= 3; ++n) {
        REQUIRE(decide(used, one, c).admit);
        used = used + one;
    }
    const Decision d = decide(used, one, c);
    CHECK_FALSE(d.admit);
    CHECK(d.engine == "nvenc");
    CHECK(d.limit == Approx(170));
}

TEST_CASE("capacity: P8 failas (RTX 4070 Laptop) — varikliai, sesijos, funkcijų VRAM, bound", "[admission][capacity]") {
    Capacity c;
    std::string err;
    REQUIRE(load_capacity(kData + "/capacity-" + kUuid + ".json", kUuid, c, err));
    CHECK(err.empty());
    CHECK(c.nvdec_mpix_s == Approx(1140.48));
    CHECK_FALSE(c.nvdec_lower_bound);
    CHECK(c.nvenc_mpix_s == Approx(622.08));
    CHECK(c.nvenc_lower_bound);
    CHECK(c.nvenc_sessions == 12);
    CHECK(c.gpu_uuid == kUuid);
    CHECK(c.driver == "596.49");
    CHECK(c.format == "1080i50");
    CHECK(c.vram_bytes == 0);  // faile nėra — iš įrenginio
    CHECK(c.source.find("capacity-" + kUuid) != std::string::npos);
    REQUIRE(c.functions.count("dec"));
    CHECK(c.functions["dec"].vram_per_flow_bytes == 191029576u);
    CHECK(c.functions["dec"].vram_offset_bytes == 102760448u);
    CHECK(c.functions["dec"].knee == 22);
    REQUIRE(c.functions.count("enc-nvenc-ll"));
    CHECK(c.functions["enc-nvenc-ll"].vram_per_flow_bytes == 132340777u);
}

TEST_CASE("capacity: katalogas → capacity-<uuid>.json; UUID didžiosiomis tinka", "[admission][capacity]") {
    Capacity c;
    std::string err;
    std::string up = kUuid;
    for (std::size_t i = 4; i < up.size(); ++i) up[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(up[i])));
    REQUIRE(load_capacity(kData, kUuid, c, err));
    CHECK(c.nvenc_sessions == 12);
    CHECK(load_capacity(kData + "/capacity-" + kUuid + ".json", up, c, err));
}

TEST_CASE("capacity: kita GPU (UUID nesutampa) → failas nenaudojamas, klaidoje abu UUID", "[admission][capacity]") {
    Capacity c;
    std::string err;
    const std::string other = "GPU-00000000-1111-2222-3333-444444444444";
    CHECK_FALSE(load_capacity(kData + "/capacity-" + kUuid + ".json", other, c, err));
    CHECK(err.find(kUuid) != std::string::npos);
    CHECK(err.find(other) != std::string::npos);
}

TEST_CASE("capacity: nėra failo → „talpa neišmatuota“", "[admission][capacity]") {
    Capacity c;
    std::string err;
    CHECK_FALSE(load_capacity(kData, "GPU-00000000-1111-2222-3333-444444444444", c, err));
    CHECK(err.find("talpa neišmatuota") != std::string::npos);
    CHECK_FALSE(load_capacity("/nonexistent/capacity.json", "", c, err));
    CHECK(err.find("talpa neišmatuota") != std::string::npos);
}

TEST_CASE("capacity: schema, trūkstami laukai, netinkamas JSON — klaida", "[admission][capacity]") {
    const std::string f = fixture();
    Capacity c;
    std::string err;
    REQUIRE(parse_capacity(f, c, err));
    CHECK_FALSE(parse_capacity(replace(f, "\"schema\": 1", "\"schema\": 2"), c, err));
    CHECK(err.find("schema") != std::string::npos);
    CHECK_FALSE(parse_capacity(replace(f, "\"driver\": \"596.49\"", "\"drv\": \"596.49\""), c, err));
    CHECK(err.find("gpu.driver") != std::string::npos);
    CHECK_FALSE(parse_capacity(f.substr(0, f.size() / 2), c, err));
    CHECK_FALSE(parse_capacity("[1,2]", c, err));
    CHECK_FALSE(parse_capacity(replace(f, "\"mpix_s\": 1140.48", "\"mpix_s\": -5"), c, err));
    // vram_per_flow_bytes: null (VRAM neišmatuotas) — leidžiama, 0
    Capacity n;
    REQUIRE(parse_capacity(replace(f, "\"vram_per_flow_bytes\": 191029576", "\"vram_per_flow_bytes\": null"), n, err));
    CHECK(n.functions["dec"].vram_per_flow_bytes == 0);
    // gpu.vram_bytes (neprivalomas) — naudojamas
    Capacity v;
    REQUIRE(parse_capacity(replace(f, "\"driver\": \"596.49\"", "\"driver\": \"596.49\", \"vram_bytes\": 8585216000"), v, err));
    CHECK(v.vram_bytes == 8585216000u);
}

TEST_CASE("capacity: įspėjimai — kita tvarkyklė (abi versijos), lower_bound", "[admission][capacity]") {
    Capacity c;
    std::string err;
    REQUIRE(parse_capacity(fixture(), c, err));
    auto has = [](const std::vector<std::string>& v, const std::string& a, const std::string& b = {}) {
        for (const std::string& s : v)
            if (s.find(a) != std::string::npos && (b.empty() || s.find(b) != std::string::npos)) return true;
        return false;
    };
    const auto same = capacity_warnings(c, "596.49");
    CHECK_FALSE(has(same, "tvarkyklės versija"));
    CHECK(has(same, "nvenc", "lower_bound"));  // NVENC Mpix/s — apatinė riba (sesijų riba nulėmė kelią)
    CHECK_FALSE(has(same, "nvdec"));
    const auto other = capacity_warnings(c, "600.10");
    CHECK(has(other, "tvarkyklės versija", "596.49"));
    CHECK(has(other, "596.49", "600.10"));
    CHECK(has(capacity_warnings(c, ""), "tvarkyklės versija nežinoma"));  // dabartinė versija nežinoma — irgi įspėjimas
}

TEST_CASE("ingest_demand: Mpix/s pagal formatą; VRAM — NV12 paviršiai + v210 ring'as", "[admission]") {
    CHECK(mpix_s(vramflow::kFmt1080i50) == Approx(51.84));
    vramflow::VideoFormat p50{{1920, 1080, false, false}, {50, 1}};
    CHECK(mpix_s(p50) == Approx(103.68));
    const Demand d = ingest_demand(vramflow::kFmt1080i50, 10, 11, true);
    CHECK(d.nvdec_mpix_s == Approx(51.84));
    CHECK(d.nvenc_mpix_s == 0);
    CHECK(d.nvenc_sessions == 0);
    // 10 × 1920 × 1088 × 1,5 + 11 × (40 · 128 B × 540 eilučių lauke)
    CHECK(d.vram_bytes == 10ull * 1920 * 1088 * 3 / 2 + 11ull * 5120 * 540);
    const Demand sw = ingest_demand(p50, 4, 10, false);
    CHECK(sw.nvdec_mpix_s == 0);
    CHECK(sw.vram_bytes == 4ull * 1920 * 1088 * 3 / 2 + 10ull * 5120 * 1080);
}
