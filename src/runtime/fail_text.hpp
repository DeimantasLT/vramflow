// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Klaidų tekstai santraukos „fatal“ laukui (planas docs/planai/2026-11-17-production-ready.md §2.5, kriterijai K4, S9).
//
// MXL ir libav apie nepavykusį paleidimą praneša tik kodu: mxlCreateFlowWriter — statusu (dažniausiai 1, UNKNOWN),
// avcodec_open2 — „Invalid argument“. Operatoriui to neužtenka, todėl priežastis nustatoma išmatuojant pačiam (ar domenas
// rašomas, kiek jame vietos, kiek laisvos VRAM, kiek reikia žiedui) ir čia paverčiama tekstu. Taisyklė: skelbiami tik
// išmatuoti faktai, o išvada („neužtenka“, „nėra vietos“) — tik kai ji iš tų faktų seka. Spėjimų nėra.
//
// Gryna logika be CUDA, MXL ir libav — matavimus atlieka kvietėjas (func_common.cpp, enc.cpp); testai — tests/unit/test_fail_text.cpp.
#include <cstdint>
#include <string>

namespace mxlv::runtime {

// Saugyklos, kurioje nepavyko sukurti išėjimo flow žiedo, išmatuoti faktai.
struct StorageFacts {
    bool cuda = false;        // cuda-linear (VRAM); kitaip — host (domeno failų sistema)
    bool writable = true;     // ar domeno katalogas šiam procesui rašomas
    std::uint64_t need = 0;   // žiedo dydis baitais (grain'o dydis × grain'ų skaičius); 0 — nežinomas
    std::uint64_t free = 0;   // laisva baitais: VRAM arba domeno failų sistemoje
    std::uint64_t total = 0;  // iš viso baitais; 0 — išmatuoti nepavyko (tada skaičiai neskelbiami)
};

// Tekstas, pridedamas prie „mxlCreateFlowWriter status N“. Prasideda „; “; tuščias — nėra ko pasakyti.
// MiB: laisva ir visa apvalinama žemyn, reikiama — aukštyn; išvada daroma iš baitų, ne iš suapvalintų skaičių.
std::string storage_fail_text(const StorageFacts& f);

// CUDA atminties skyrimo klaida: „<kas>: CUDA <klaidos tekstas>; VRAM laisva X MiB iš Y MiB“ (total 0 — be VRAM dalies).
std::string cuda_alloc_fail_text(const std::string& what, const std::string& cuda_error, std::uint64_t free, std::uint64_t total);

// Video koderio atidarymo klaida. av_error — av_strerror tekstas; log_line — paskutinė libav klaidos eilutė šioje gijoje
// (gali būti tuščia). NVENC sesijos eilutė („OpenEncodeSessionEx failed … (kodas)“) paaiškinama: 21 — vienalaikių sesijų
// riba (nvEncodeAPI.h: „feature … not available for the license type“; išmatuota su tvarkykle 596.49 — 13-a sesija),
// 10 — tvarkyklei neužteko atminties. Kitos eilutės tik pacituojamos.
std::string nvenc_open_fail_text(const std::string& av_error, const std::string& log_line);

}  // namespace mxlv::runtime
