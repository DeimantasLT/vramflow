# „Production ready“ 1 fazė — saugumas ir build (2026-11-17)

**Planas:** `docs/planai/2026-11-17-production-ready.md`, paketai 1.0–1.6. **Aplinka:** RTX 4070 Laptop 8 GB, tvarkyklė 596.49,
WSL2 Ubuntu 24.04, `clang++-19` (C++) ir `g++` (CUDA host dalis), glibc 2.39.

## 1.1 Build grūdinimas (kriterijus S6)

**Kas padaryta.** `cmake/hardening.cmake` — `vramflow_hardening(<tikslas>)`, kviečiama iš `vramflow_warnings()`, todėl vėliavas
gauna kiekvienas savas tikslas (C tikslas `vramflow-gfx-c-check` ją kviečia tiesiogiai):

| Kur | Vėliavos |
|---|---|
| C, C++ ir CUDA host dalis | `-fstack-protector-strong -fstack-clash-protection -fcf-protection=full`; C++ ir CUDA — `-D_GLIBCXX_ASSERTIONS` |
| Optimizuotas build be sanitizerių | `-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3` |
| Susiejimas (vykdomieji, `.so`) | `-Wl,-z,relro,-z,now -Wl,-z,noexecstack`; PIE — `CMAKE_POSITION_INDEPENDENT_CODE` |

Parinktis `VRAMFLOW_HARDENING` (numatyta `ON`). `compile_commands.json` dabar generuojamas visada.

**Testas** `ctest build_hardening` (`tests/arch/check_hardening.py`) tikrina ne fiksuotą sąrašą, o viską, ką randa: kiekvieną
`compile_commands.json` įrašą iš `src/` ir `tests/` ir kiekvieną `vramflow-*`, `mxlv-*`, `libvramflow*.so*` ELF failą build
kataloge (PIE, `GNU_RELRO`, `BIND_NOW`, nevykdomas stekas; pristatomiems binarams — `__stack_chk_fail`).

| Paleidimas | Rezultatas |
|---|---|
| Prieš (negrūdintas build) | FAIL: 104 vertimo vienetai be vėliavų, 22 ELF failai su klaidomis (`BIND_NOW` nėra) |
| Po | PASS: 104 vienetai ir 22 ELF failai tinka; švarus build — 0 įspėjimų; `ctest` 484 / 484 |

CET žyma (`IBT`, `SHSTK`) binare yra tik ten, kur nesusieta `CUDA::cudart_static` (pvz. `mxlv-admission-unit`); pristatomuose
binaruose jos nėra, nes NVIDIA statinė biblioteka be jos. Todėl testas CET rodo tik informacijai.

**Kaina.** Tas pats matavimas prieš ir po, 3 × 30 s (`tests/perf/capacity.py`):

| Scenarijus | Matas | Prieš | Po |
|---|---|---|---|
| logo, CPU variantas, N = 4 | CPU branduoliai: `gen` / `logo` / `verify` (vidurkis) | 0,6239 / 0,1608 / 0,0292 | 0,6267 / 0,1607 / 0,0297 |
| | visų trijų suma | 0,8139 | 0,8171 (+0,4 %) |
| | apdorojimas p50, ms | 0,942; 0,942; 0,958 | 0,942; 0,958; 0,958 |
| | apdorojimas p99, ms | 1,950; 1,950; 2,130 | 2,130; 2,081; 1,982 |
| logo, GPU, N = 64 | praėjo / praradimai | 3 iš 3 / 0 | 3 iš 3 / 0 |
| | apdorojimas p50, ms | 2,654; 2,916; 3,310 | 2,654; 3,375; 2,654 |
| grandinė, GPU, N = 8 | praėjo / CPU srautui / e2e p99, ms | 3 iš 3 / 0,126–0,142 / 287,1–288,0 | 3 iš 3 / 0,126–0,127 / 286,6–287,1 |
| grandinė ties keliu, N = 11, pakaitomis po 5 | praėjo | negrūdintas: 4 iš 5 | grūdintas: 4 iš 5 |
| | CPU srautui (praėjusiuose) | 0,139–0,168 | 0,135–0,137 |

Išvada: CPU kaina — +0,4 % (mažiau nei skirtumas tarp pakartojimų, ±0,7 %); vėlinimas ir kelias nepakito. Riba 2 % neviršyta.
Matavimo ribos: histogramos žingsnis ≈ 1,7 %, GPU varianto p50 tarp pakartojimų svyruoja ±30 %, todėl mažesnių nei ~2 %
skirtumų vėlinime šis matavimas neskiria.

**Šalutinis radinys (planui 3.2, G2).** Grandinė ties keliu N = 11 šiandien praėjo 10 iš 13 paleidimų (P8 kampanijoje — 3 iš
3); krenta tuo pačiu simptomu kaip prie N = 12 (`dec.late_skips`, `enc.late_frames`), vienodai su grūdinimu ir be jo.

## 1.2 Nuomos ir priėmimo knygos failai (kriterijus S4, peržiūros fokusas RF-2)

**Problema.** Nuomos katalogas `<domenas>.vramflow/` yra šalia domeno, dažniausiai `/dev/shm` (teisės 1777). Senas kodas jį
kūrė `mkdir 0775` ir, jei jau buvo, naudojo netikrindamas; laikiną nuomos failą atidarydavo `O_CREAT | O_TRUNC` nuspėjamu
vardu; svetimus `.lease` ir `.adm` failus — paprastu `O_RDONLY`; katalogo vardą darė iš kelio teksto.

**Kas padaryta.** `src/runtime/safe_files.{hpp,cpp}`; `DomainLease` ir `Ledger` dirba tik per patikrinto katalogo aprašą
(`openat`, `unlinkat`, `renameat2`):

| Grėsmė | Dabar |
|---|---|
| Vietoje katalogo — simbolinė nuoroda | atidaroma `O_DIRECTORY \| O_NOFOLLOW` → klaida „nuomos katalogas … (atmetama)“, exit 2 |
| Katalogą pirmas sukūrė kitas vartotojas | atmetama, jei savininkas nėra nei šis procesas, nei domeno katalogo savininkas (išimtys: domenas, rašomas visiems; grupei rašomas domenas su ta pačia grupe) |
| Svetimo katalogo teisės platesnės nei domeno | atmetama; savas katalogas susiaurinamas iki domeno teisių |
| Nuoroda vietoje laikino nuomos failo arba `admission.lock` | failai kuriami `O_EXCL \| O_NOFOLLOW`; taikinys neliečiamas |
| FIFO vietoje `.lease` ar `.adm` | atidaroma `O_NOFOLLOW \| O_NONBLOCK`, priimami tik paprasti failai — nepakimba |
| Tas pats domenas kitu keliu (nuoroda, `/./`) [RF-2] | katalogo vardas — iš kanoninio kelio: viena nuoma ir viena knyga |

Teisės: katalogo — kaip domeno katalogo, failų — tos pačios be vykdymo bitų; nuo `umask` nepriklauso (`fchmod`). Atmesta:
fiksuotos `0775` / `0664` (uždaram domenui per plačios, grupei bendram — per siauros). Harness'as (`tests/perf/domain.py`)
naudoja tą patį kanoninį kelią ir taip pat neseka nuorodų.

**Testai** `tests/unit/test_safe_files.cpp` (9 atvejai, `mxlv-admission-unit`), per viešą sąsają:

| Paleidimas | Rezultatas |
|---|---|
| Senas kodas (`dc59355`) + nauji testai | krenta visi 9 atvejai (32 iš 124 tvirtinimų): FIFO pakabina `live_owners` ir `Ledger::total` (vaikas nužudomas po 1 s), nuorodos sekamos, teisės nepaveldimos, per nuorodą — atskira knyga, svetimas savininkas priimamas |
| Naujas kodas | 9 iš 9 (124 tvirtinimai); visas `mxlv-admission-unit` — 32 atvejai, 405 tvirtinimai; nuomos testai `vramflow-runtime-tests "[lease]"` — 4 atvejai, 33 tvirtinimai |

Švarus build — 0 įspėjimų; `ctest` 492 / 493 (krito tik žinomas laiko testas `gap_policy … < 1 ms`).
Funkciniai testai per nuomą ir knygą tikrais binarais (`fstart_*`, `overload_*`, `fcomp1`): 26 iš 26 PASS; po jų `/dev/shm`
nelieka nė vieno nuomos katalogo.
