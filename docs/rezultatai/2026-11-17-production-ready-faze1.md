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
