// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// common/parse_num.hpp: griežtas dešimtainių skaičių skaitymas (vietoj atoi / sscanf, kurie perpildymo nepraneša) ir
// ref-crc failo analizė. Planas docs/planai/2026-11-17-production-ready.md §2.4 (clang-tidy cert-err34-c).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include "common/parse_num.hpp"
#include "common/textf.hpp"

using mxlv::parse_ref_crc;
using mxlv::read_file;
using mxlv::take_char;
using mxlv::take_num;

TEST_CASE("take_num: skaičius teksto pradžioje — reikšmė ir likutis", "[parse_num]") {
    std::string_view s = "123,45";
    int v = -1;
    REQUIRE(take_num(s, v));
    CHECK(v == 123);
    CHECK(s == ",45");
    s = "-7x";
    REQUIRE(take_num(s, v));
    CHECK(v == -7);
    CHECK(s == "x");
    unsigned long long big = 0;
    s = "18446744073709551615";
    REQUIRE(take_num(s, big));
    CHECK(big == 18446744073709551615ull);
    CHECK(s.empty());
}

TEST_CASE("take_num: ne skaičius, „+“, tarpas, perpildymas — false, tekstas ir reikšmė nepakinta", "[parse_num]") {
    for (const char* bad : {"", "abc", "+5", " 5", "99999999999", "-", "x1"}) {
        std::string_view s = bad;
        int v = 42;
        INFO("tekstas '" << bad << "'");
        CHECK_FALSE(take_num(s, v));
        CHECK(v == 42);
        CHECK(s == bad);
    }
    std::string_view s = "-1";
    unsigned u = 7;
    CHECK_FALSE(take_num(s, u));  // beženkliam tipui minusas netinka
    CHECK(u == 7);
    s = "4294967296";  // 2^32
    std::uint32_t u32 = 9;
    CHECK_FALSE(take_num(s, u32));
    CHECK(u32 == 9);
    s = "4294967295";
    CHECK(take_num(s, u32));
    CHECK(u32 == 4294967295u);
}

TEST_CASE("take_char: tikslus simbolis pradžioje", "[parse_num]") {
    std::string_view s = ":x";
    CHECK(take_char(s, ':'));
    CHECK(s == "x");
    CHECK_FALSE(take_char(s, ':'));
    CHECK(s == "x");
    s = "";
    CHECK_FALSE(take_char(s, ':'));
}

TEST_CASE("parse_ref_crc: ref-crc išvestis → barkodas : CRC", "[parse_num]") {
    const auto m = parse_ref_crc(R"({"fields":[[16,123],[17,4294967295],[18,0]]})" "\n");
    REQUIRE(m.size() == 3);
    CHECK(m.at(16) == 123u);
    CHECK(m.at(17) == 4294967295u);
    CHECK(m.at(18) == 0u);
}

TEST_CASE("parse_ref_crc: tarpai leidžiami; netinkamos ir netelpančios poros praleidžiamos", "[parse_num]") {
    const auto m = parse_ref_crc("{\"fields\":[[ 1 , 10 ], [2,], [3,30, [x,5], [4,4294967296], [5,50]]}");
    CHECK(m.size() == 2);
    CHECK(m.at(1) == 10u);
    CHECK(m.at(5) == 50u);
    CHECK(parse_ref_crc("").empty());
    CHECK(parse_ref_crc("{\"fields\":[]}").empty());
}

TEST_CASE("read_file: visas failo turinys; nesamo failo — false", "[parse_num]") {
    std::string text = "senas";
    CHECK_FALSE(read_file("/nonexistent/vramflow-test-file", text));
    std::FILE* f = std::tmpfile();
    REQUIRE(f != nullptr);
    const std::string big(200000, 'k');  // ilgesnis už skaitymo bloką
    REQUIRE(std::fwrite(big.data(), 1, big.size(), f) == big.size());
    REQUIRE(std::fflush(f) == 0);
    const std::string path = "/proc/self/fd/" + std::to_string(fileno(f));
    REQUIRE(read_file(path, text));
    CHECK(text == big);
    CHECK(mxlv::close_checked(f));
    REQUIRE(read_file("/proc/self/statm", text));  // procfs: dydis iš anksto nežinomas (st_size == 0)
    CHECK_FALSE(text.empty());
}
