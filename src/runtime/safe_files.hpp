// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Saugus nuomos ir priėmimo knygos failų kūrimas (planas docs/planai/2026-11-17-production-ready.md §1.2, kriterijus S4).
//
// Nuomos katalogas `<domenas>.vramflow/` yra šalia domeno, dažniausiai /dev/shm (teisės 1777): kitas vietinis vartotojas
// gali jį sukurti pirmas arba padėti jame simbolinę nuorodą ar FIFO. Todėl:
//   - katalogas atidaromas O_DIRECTORY | O_NOFOLLOW ir patikrinamas: savininkas ir teisės ne platesnės nei domeno;
//   - failai kuriami openat(O_CREAT | O_EXCL | O_NOFOLLOW) — esamo vardo (ir nuorodos) niekada neperrašo;
//   - svetimi failai atidaromi O_NOFOLLOW | O_NONBLOCK ir priimami tik paprasti (S_ISREG): FIFO atidarymo nepakabina.
// Teisės: katalogo — domeno katalogo teisės, failų — tos pačios be vykdymo bitų; nuo umask nepriklauso (fchmod).
// Atmesta: fiksuotos teisės 0775 / 0664 (platesnės nei uždaro domeno; siauresnės nei bendro grupei).
#include <string>

namespace mxlv::runtime {

// Kanoninis domeno kelias: simbolinės nuorodos išskleistos, `/./` ir galiniai `/` pašalinti (neegzistuojančiai kelio daliai —
// leksinis normalizavimas). Tas pats domenas kitu keliu duoda tą patį nuomos katalogą, taigi ir vieną priėmimo knygą.
std::string canonical_domain(const std::string& domain);

// Atidaro nuomos katalogą `<kanoninis domenas>.vramflow`; create — sukuria, jei jo nėra. Grąžina fd (O_CLOEXEC) arba -1:
// err netuščias — priežastis („nuomos katalogas …“); err tuščias — katalogo nėra (tik kai create == false).
// Atmetama: ne katalogas arba simbolinė nuoroda; savininkas, kuris nėra nei šis procesas, nei domeno katalogo savininkas
// (išskyrus domeną, rašomą visiems, ir grupei rašomą domeną su ta pačia grupe); svetimas katalogas su platesnėmis teisėmis
// nei domeno. Savas katalogas su platesnėmis teisėmis susiaurinamas iki domeno teisių.
int open_lease_dir(const std::string& domain, bool create, std::string& err);

// Naujas failas kataloge: O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC; teisės — katalogo be vykdymo bitų.
// Vardas jau užimtas (taip pat ir simbolinės nuorodos) → -1, errno == EEXIST.
int create_file_at(int dirfd, const std::string& name);

// Esamas PAPRASTAS failas kataloge: flags | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC ir fstat. Simbolinė nuoroda, FIFO, lizdas,
// katalogas → -1 (errno ELOOP arba EINVAL).
int open_regular_at(int dirfd, const std::string& name, int flags);

// Bendras užrakto failas: atidaromas arba sukuriamas (O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC); tik paprastas failas.
int open_lock_file_at(int dirfd, const std::string& name);

}  // namespace mxlv::runtime
