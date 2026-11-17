// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Saugus nuomos ir priėmimo knygos failų kūrimas (planas docs/planai/2026-11-17-production-ready.md §1.2, kriterijus S4).
//
// Nuomos katalogas `<domenas>.vramflow/` yra šalia domeno, dažniausiai /dev/shm (teisės 1777): kitas vietinis vartotojas
// gali jį sukurti pirmas arba padėti jame simbolinę nuorodą ar FIFO. Todėl:
//   - katalogas atidaromas O_DIRECTORY | O_NOFOLLOW ir patikrinamas: savininkas ir teisės ne platesnės nei turi būti;
//   - failai kuriami openat(O_CREAT | O_EXCL | O_NOFOLLOW) — esamo vardo (ir nuorodos) niekada neperrašo;
//   - svetimi failai atidaromi O_NOFOLLOW | O_NONBLOCK ir priimami tik paprasti (S_ISREG): FIFO atidarymo nepakabina;
//   - failai šalinami tik laikant katalogo užraktą (DirLock) ir tik jei vardas tebenurodo tą patį failą.
//
// Teisės: prieiga prie nuomos katalogo = RAŠYMO teisė į domeną. Savininkas — visada; grupė ir „kiti“ — tik jei domeno
// katalogas jiems rašomas (0755 domenas → 0700 katalogas ir 0600 failai; 0770 → 0770 ir 0660). Kas failo neatidaro, tas
// negali jo ir užrakinti: flock užtenka skaitymo aprašo, todėl skaitomas užrakto failas leistų pašaliečiui pakabinti visus
// domeno procesus arba laikyti mirusio proceso nuomą „gyvą“. Nuo umask teisės nepriklauso (fchmod).
// Atmesta: fiksuotos teisės 0775 / 0664; domeno teisių kopija (0755 domenui duotų visiems skaitomus užrakto failus).
#include <sys/types.h>

#include <string>

namespace mxlv::runtime {

// Kanoninis domeno kelias: simbolinės nuorodos išskleistos, `/./` ir galiniai `/` pašalinti (neegzistuojančiai kelio daliai —
// leksinis normalizavimas). Tas pats domenas kitu keliu duoda tą patį nuomos katalogą, taigi ir vieną priėmimo knygą.
std::string canonical_domain(const std::string& domain);

// Atidarytas ir patikrintas nuomos katalogas.
struct LeaseDir {
    int fd = -1;                           // katalogo aprašas (O_CLOEXEC); uždaro kvietėjas
    mode_t file_mode = 0600;               // naujų failų teisės
    gid_t group = static_cast<gid_t>(-1);  // naujų failų grupė (domeno grupė, kai domenas rašomas grupei); -1 — nekeisti
};

// Atidaro nuomos katalogą `<kanoninis domenas>.vramflow`; create — sukuria, jei jo nėra. true — out užpildytas. false:
// err netuščias — priežastis („nuomos katalogas …“); err tuščias — katalogo nėra (tik kai create == false).
// Atmetama: ne katalogas arba simbolinė nuoroda; savininkas, kuris nėra nei šis procesas, nei domeno katalogo savininkas
// (išskyrus domeną, rašomą visiems, ir grupei rašomą domeną su ta pačia grupe); svetimas katalogas su platesnėmis teisėmis,
// nei priklauso. Savas katalogas su platesnėmis teisėmis susiaurinamas. Kai domeno katalogo (dar) nėra, esamas katalogas
// nekeičiamas ir priimamas tik savas.
bool open_lease_dir(const std::string& domain, bool create, LeaseDir& out, std::string& err);

// Naujas failas kataloge: O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, teisės ir grupė — iš LeaseDir.
// Vardas jau užimtas (taip pat ir simbolinės nuorodos) → -1, errno == EEXIST.
int create_file_at(const LeaseDir& d, const std::string& name);

// Esamas PAPRASTAS failas kataloge: flags | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC ir fstat. Simbolinė nuoroda, FIFO, lizdas,
// katalogas → -1 (errno ELOOP arba EINVAL).
int open_regular_at(int dirfd, const std::string& name, int flags);

// Ar vardas kataloge tebenurodo tą patį failą kaip aprašas fd (tas pats įrenginys ir inode)? Prieš šalinant pagal vardą.
bool same_file_at(int dirfd, const std::string& name, int fd);

// Katalogo užraktas: flock(LOCK_EX) ant `<katalogas>/<name>` (failas sukuriamas, jei nėra) su RIBOTU laukimu — kito proceso
// laikomas užraktas nepakabina be galo. ok() == false: užrakto negauta per timeout_ms arba failo nepavyko atidaryti.
class DirLock {
public:
    DirLock(const LeaseDir& d, const char* name, int timeout_ms = 5000);
    ~DirLock();
    DirLock(const DirLock&) = delete;
    DirLock& operator=(const DirLock&) = delete;
    bool ok() const { return locked_; }

private:
    int fd_ = -1;
    bool locked_ = false;
};

}  // namespace mxlv::runtime
