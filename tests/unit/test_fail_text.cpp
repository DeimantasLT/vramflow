// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Klaidų tekstai santraukos „fatal“ laukui (planas docs/planai/2026-11-17-production-ready.md §2.5, kriterijai K4, S9):
// tekste turi būti priežastis, o ne tik MXL statusas ar libav klaidos kodas. Čia — gryna teksto logika; pačius gedimus
// sukelia tests/functional/faults.py.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

#include "runtime/fail_text.hpp"

using namespace mxlv::runtime;

namespace {
constexpr std::uint64_t kMiB = 1ull << 20;
}

TEST_CASE("fail_text: nerašomas domenas — EACCES, kiti faktai neskelbiami", "[fail_text]") {
    StorageFacts f;
    f.writable = false;
    f.need = 28 * kMiB;
    f.free = 4 * kMiB;
    f.total = 16 * kMiB;
    CHECK(storage_fail_text(f) == "; domeno katalogas nerašomas (EACCES, permission denied)");
    f.cuda = true;
    CHECK(storage_fail_text(f) == "; domeno katalogas nerašomas (EACCES, permission denied)");
}

TEST_CASE("fail_text: host saugykla — vieta domene; išvada tik kai laisvos mažiau nei reikia žiedui", "[fail_text]") {
    StorageFacts f;
    f.need = 28 * kMiB;
    f.free = 15 * kMiB;
    f.total = 16 * kMiB;
    CHECK(storage_fail_text(f) ==
          "; saugykla host: domene laisva 15 MiB iš 16 MiB, žiedui reikia 28 MiB — domene nėra vietos (ENOSPC, no space left on device)");
    f.free = 28 * kMiB;  // lygiai tiek, kiek reikia — išvados nėra (priežastis kita)
    f.total = 64 * kMiB;
    CHECK(storage_fail_text(f) == "; saugykla host: domene laisva 28 MiB iš 64 MiB, žiedui reikia 28 MiB");
    f.need = 0;  // žiedo dydis nežinomas — tik išmatuota vieta
    CHECK(storage_fail_text(f) == "; saugykla host: domene laisva 28 MiB iš 64 MiB");
}

TEST_CASE("fail_text: cuda-linear — VRAM skaičiai; išvada tik kai laisvos mažiau nei reikia žiedui", "[fail_text]") {
    StorageFacts f;
    f.cuda = true;
    f.need = 28 * kMiB;
    f.free = 12 * kMiB;
    f.total = 8188 * kMiB;
    CHECK(storage_fail_text(f) == "; saugykla cuda-linear: VRAM laisva 12 MiB iš 8188 MiB, žiedui reikia 28 MiB — VRAM neužtenka");
    f.free = 7000 * kMiB;
    CHECK(storage_fail_text(f) == "; saugykla cuda-linear: VRAM laisva 7000 MiB iš 8188 MiB, žiedui reikia 28 MiB");
}

TEST_CASE("fail_text: neišmatuota (total 0) — jokių skaičių ir jokios išvados", "[fail_text]") {
    StorageFacts f;
    f.need = 28 * kMiB;
    CHECK(storage_fail_text(f).empty());
    f.cuda = true;
    CHECK(storage_fail_text(f).empty());
}

TEST_CASE("fail_text: MiB apvalinimas — laisva žemyn, reikia aukštyn (išvada nepagražinama)", "[fail_text]") {
    StorageFacts f;
    f.need = 27 * kMiB + 1;       // 27,000001 MiB → 28
    f.free = 28 * kMiB - 1;       // 27,999999 MiB → 27
    f.total = 64 * kMiB + kMiB / 2;  // 64,5 → 64
    CHECK(storage_fail_text(f) == "; saugykla host: domene laisva 27 MiB iš 64 MiB, žiedui reikia 28 MiB");
    f.free = 27 * kMiB;  // baitais mažiau nei reikia — išvada yra, nors suapvalinti skaičiai atrodo artimi
    CHECK(storage_fail_text(f) ==
          "; saugykla host: domene laisva 27 MiB iš 64 MiB, žiedui reikia 28 MiB — domene nėra vietos (ENOSPC, no space left on device)");
}

TEST_CASE("fail_text: CUDA atminties skyrimo klaida — kas, CUDA klaidos tekstas ir VRAM", "[fail_text]") {
    CHECK(cuda_alloc_fail_text("scratch", "out of memory", 12 * kMiB, 8188 * kMiB) ==
          "scratch: CUDA out of memory; VRAM laisva 12 MiB iš 8188 MiB");
    CHECK(cuda_alloc_fail_text("staging", "unknown error", 0, 0) == "staging: CUDA unknown error");
}

TEST_CASE("fail_text: NVENC kodas 21 — sesijų riba; žurnalo eilutės rodyklė pašalinama", "[fail_text]") {
    const std::string line = "[h264_nvenc @ 0x7025640009c0] OpenEncodeSessionEx failed: incompatible client key (21): (no details)\n";
    CHECK(nvenc_open_fail_text("Invalid argument", line) ==
          "video avcodec_open2: Invalid argument (OpenEncodeSessionEx failed: incompatible client key (21): (no details)) — NVENC sesijos "
          "negauta: tvarkyklės kodas 21 reiškia, kad viršyta vienalaikių NVENC sesijų riba (šios sistemos licencijos tipas daugiau neleidžia)");
}

TEST_CASE("fail_text: NVENC kodas 10 — atmintis; kitas kodas — be spėjimo apie priežastį", "[fail_text]") {
    CHECK(nvenc_open_fail_text("Cannot allocate memory", "[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10): (no details)") ==
          "video avcodec_open2: Cannot allocate memory (OpenEncodeSessionEx failed: out of memory (10): (no details)) — NVENC sesijos "
          "negauta: tvarkyklei neužteko atminties sesijai (kodas 10)");
    CHECK(nvenc_open_fail_text("Generic error in an external library", "[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: generic error (20)") ==
          "video avcodec_open2: Generic error in an external library (OpenEncodeSessionEx failed: generic error (20)) — NVENC sesijos negauta");
}

TEST_CASE("fail_text: koderio atidarymo klaida be NVENC sesijos eilutės — sesijų riba neminima", "[fail_text]") {
    CHECK(nvenc_open_fail_text("Invalid argument", "") == "video avcodec_open2: Invalid argument");
    CHECK(nvenc_open_fail_text("Invalid argument", "[h264_nvenc @ 0x1] Level 4.0 is not supported\n") ==
          "video avcodec_open2: Invalid argument (Level 4.0 is not supported)");
    // eilutė be „[…] “ priešdėlio paliekama kaip yra
    CHECK(nvenc_open_fail_text("Invalid argument", "No capable devices found") ==
          "video avcodec_open2: Invalid argument (No capable devices found)");
}
