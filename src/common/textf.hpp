// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Tekstas ir failai be tyliai prarandamų klaidų (planas docs/planai/2026-11-17-production-ready.md §2.4; clang-tidy
// cert-err33-c, concurrency-mt-unsafe). Testai — tests/unit/test_textf.cpp.
#include <cstdarg>
#include <cstdio>
#include <string>
#include <system_error>

namespace mxlv {

// printf formatas → std::string, be nupjovimo. Fiksuotas buferis su snprintf ilgesnį tekstą nupjauna tyliai (taip buvo
// galima sugadinti santraukos JSON); čia trumpas tekstas telpa į steko buferį, ilgesniam skiriama tiek, kiek reikia.
// Formato klaida (vsnprintf < 0) — tuščias tekstas.
[[gnu::format(printf, 1, 2)]] inline std::string strf(const char* fmt, ...) {
    char b[256];
    std::va_list ap, again;
    va_start(ap, fmt);
    va_copy(again, ap);
    const int n = std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    std::string s;
    if (n >= 0 && static_cast<std::size_t>(n) < sizeof b) {
        s.assign(b, static_cast<std::size_t>(n));
    } else if (n >= 0) {
        s.resize(static_cast<std::size_t>(n));
        if (std::vsnprintf(s.data(), s.size() + 1, fmt, again) != n) s.clear();
    }
    va_end(again);
    return s;
}

// Failo, į kurį rašyta, uždarymas. fprintf / fputs klaida lieka srauto būsenoje, o dalis duomenų išrašoma tik uždarant,
// todėl rašymo sėkmė tikrinama čia: false — bent dalis duomenų neįrašyta (priežastis — errno, kai klaida įvyko uždarant).
// nullptr — true.
inline bool close_checked(std::FILE* f) {
    if (f == nullptr) return true;
    const bool failed_before = std::ferror(f) != 0;  // fclose po ankstesnės klaidos gali grąžinti 0 (buferis jau tuščias)
    const bool closed = std::fclose(f) == 0;
    return closed && !failed_before;
}

// Skaityto failo uždarymas: rašymo klaidos būti negali, todėl rezultato nėra ko tikrinti. nullptr — nieko.
inline void close_read(std::FILE* f) {
    if (f != nullptr) (void)std::fclose(f);
}

// Visas failo turinys į out. false — failo atidaryti nepavyko arba skaitymo klaida (out tada — tai, kas perskaityta).
// Tinka ir procfs failams, kurių dydis iš anksto nežinomas.
inline bool read_file(const std::string& path, std::string& out) {
    out.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char b[4096];
    for (;;) {
        const std::size_t n = std::fread(b, 1, sizeof b, f);
        out.append(b, n);
        if (n < sizeof b) break;  // failo pabaiga arba klaida — toliau neskaitoma
    }
    const bool ok = std::ferror(f) == 0;
    close_read(f);
    return ok;
}

// Failas vienu rašymu (santraukos, diagnostikos išklotinės). false — neatidarytas, įrašyta ne viskas arba klaida uždarant
// (priežastis — errno); failas tada gali būti likęs nepilnas.
inline bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const bool wrote = std::fwrite(data, 1, bytes, f) == bytes;
    return close_checked(f) && wrote;
}

// errno reikšmės tekstas. std::strerror standarte nėra saugi gijoms (bendras statinis buferis); šis variantas — saugus.
inline std::string errno_text(int e) { return std::generic_category().message(e); }

}  // namespace mxlv
