// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// common/textf.hpp: formatavimas be nupjovimo (strf), failo uždarymas su rašymo klaidos patikra (close_checked) ir
// errno tekstas, saugus gijoms (errno_text). Planas docs/planai/2026-11-17-production-ready.md §2.4.
#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <string>

#include "common/textf.hpp"

using mxlv::close_checked;
using mxlv::errno_text;
using mxlv::strf;

TEST_CASE("strf: trumpas tekstas — kaip snprintf", "[textf]") {
    CHECK(strf("%d flow'ai, %.3f ms", 16, 1.5) == "16 flow'ai, 1.500 ms");
    CHECK(strf("%s", "").empty());
    CHECK(strf("be argumentų") == "be argumentų");
}

TEST_CASE("strf: ilgas tekstas nenupjaunamas (ilgesnis už vidinį steko buferį)", "[textf]") {
    const std::string big(5000, 'x');
    const std::string s = strf("{\"error\":\"%s\"}", big.c_str());
    REQUIRE(s.size() == big.size() + 12);
    CHECK(s.substr(0, 10) == "{\"error\":\"");
    CHECK(s.substr(s.size() - 2) == "\"}");
    CHECK(s.find_first_not_of('x', 10) == s.size() - 2);
}

TEST_CASE("strf: ilgiai ties vidinio buferio riba (255, 256, 257 baitai)", "[textf]") {
    for (const std::size_t n : {std::size_t{254}, std::size_t{255}, std::size_t{256}, std::size_t{257}, std::size_t{1024}}) {
        const std::string in(n, 'a');
        const std::string out = strf("%s", in.c_str());
        INFO("ilgis " << n);
        CHECK(out == in);
    }
}

TEST_CASE("close_checked: sėkmingas rašymas — true; nullptr — true", "[textf]") {
    CHECK(close_checked(nullptr));
    std::FILE* f = std::tmpfile();
    REQUIRE(f != nullptr);
    REQUIRE(std::fputs("eilutė\n", f) >= 0);
    CHECK(close_checked(f));
}

TEST_CASE("close_checked: rašymo klaida, paaiškėjanti tik uždarant (/dev/full) — false ir errno ENOSPC", "[textf]") {
    std::FILE* f = std::fopen("/dev/full", "w");
    if (f == nullptr) SKIP("nėra /dev/full");
    // Tekstas lieka stdio buferyje: fputs pavyksta, klaida atsiranda tik išrašant
    REQUIRE(std::fputs("santrauka, kuri niekada nepasieks disko\n", f) >= 0);
    errno = 0;
    CHECK_FALSE(close_checked(f));
    CHECK(errno == ENOSPC);
}

TEST_CASE("close_checked: rašymo klaida, įvykusi anksčiau (srauto klaidos žymė) — false", "[textf]") {
    std::FILE* f = std::fopen("/dev/full", "w");
    if (f == nullptr) SKIP("nėra /dev/full");
    REQUIRE(std::fputs("x", f) >= 0);
    REQUIRE(std::fflush(f) != 0);  // klaida jau įvyko; fclose po jos gali grąžinti 0, nes buferis tuščias
    CHECK_FALSE(close_checked(f));
}

TEST_CASE("errno_text: tas pats tekstas kaip strerror, be bendro statinio buferio", "[textf]") {
    CHECK(errno_text(ENOENT) == "No such file or directory");
    CHECK(errno_text(EACCES) == "Permission denied");
    CHECK(errno_text(ENOSPC) == "No space left on device");
}

TEST_CASE("write_file: įrašo visus baitus; nesamas katalogas ir pilnas įrenginys — false", "[textf]") {
    const std::string path = "/dev/shm/vf-textf-" + std::to_string(::getpid()) + ".bin";
    const std::string data(100000, 'q');
    REQUIRE(mxlv::write_file(path, data.data(), data.size()));
    std::string back;
    REQUIRE(mxlv::read_file(path, back));
    CHECK(back == data);
    CHECK(std::remove(path.c_str()) == 0);
    CHECK_FALSE(mxlv::write_file("/nonexistent-dir/x.bin", data.data(), data.size()));
    if (std::FILE* probe = std::fopen("/dev/full", "w")) {
        CHECK(close_checked(probe));
        CHECK_FALSE(mxlv::write_file("/dev/full", data.data(), data.size()));
    }
}
