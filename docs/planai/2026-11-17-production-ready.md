# VRAMFlow „production ready“ (O / G / K / S / U) — įgyvendinimo planas

> **Vykdytojams:** kiekviena fazė — nauja sesija; paketai vykdomi pagal `superpowers:executing-plans` (TDD). Po kiekvieno
> paketo atnaujinama būsenos lentelė (§8). Nestabilūs testai ir našumo mįslės — `superpowers:systematic-debugging`.

> **Būsena (2026-11-17):** planas. Kodas nekeistas. Bazė (§3) patvirtinta vykdymu ant `e570bae`; viskas kita paremta
> kodo skaitymu (failas:eilutė) — kur reikia vykdymo, pažymėta **[nepatikrinta]** ir surašyta §9.

**Užduotis:** autoriaus užduoties aprašas, 2026-11-17 (laikomas ne šiame repo); jo kriterijai perkelti į §2.
Ankstesnis planas ir sprendimai K1–K17: `docs/planai/2026-11-11-production-grade.md` — **priimti, neperklausiami**.

**Tikslas:** funkcinė dalis padaryta (P1–P8, P10, P4 priėmimo kontrolė). Šis planas prideda įrodymus, be kurių sprendimo
negalima statyti į gamybą: build grūdinimas, saugus failų kūrimas, paleidimas ne root, sanitizeriai ir fuzz testai,
klaidų kelių testai, nestabilių laiko testų priežastys, VRAM optimizavimas su „prieš / po“, 2 h soak, diegimas.

**Architektūros kryptis:**

1. **Naujų funkcijų nėra.** P9 multivieweris, P4 termino vartai (`--shed`), nauji formatai, HLS grūdinimas — ne šio plano.
2. **Kiekvienas kriterijus turi testą arba matavimą repo**, kurį galima pakartoti viena komanda (`ctest`, `functional.py
   --only …`, skriptas `scripts/linux/`). Rankiniai įrodymai netinka.
3. **Grūdinimas ir sanitizeriai — atskiri build katalogai**, ne atskiros kodo šakos: `~/build/vramflow/func` (gamybinis),
   `…/asan`, `…/tsan`, `…/fuzz` (su `clang++-18`, kuris turi vykdymo bibliotekas; `clang++-19` jų neturi).
4. **Našumo pakeitimas be „prieš / po“ lentelės nepriimamas**; regresijos vartai (`tests/perf/regress.py`) atsiranda 3 fazėje
   ir nuo tada saugo kiekvieną vėlesnį paketą.

**Technologijos:** C++20, CUDA 13.2, MXL PR #653 (`6dddc91d`), FFmpeg 6.1.1 (sisteminis), Catch2 3, CLI11, Python 3,
`clang++-19` (gamybinis build), `clang++-18` (sanitizeriai, fuzz), `clang-tidy-19`. Naujų priklausomybių nėra.

---

## 1. Bendri ribojimai (privalomi)

- Commit'ai: Conventional Commits anglų kalba, trailer'is `Assisted-by: Claude Code (<modelio ID>)`, tiesiai į `main`;
  po patikros — `git push origin main`. Jokių kitų šakų, jokio force-push. Niekada `git add -A` (apie 20 failų skiriasi tik
  eilučių pabaigomis): stage'inami tik savo failai, tikrinama `git show --stat`. Paketas ≤ ~400 pakeistų eilučių.
- **Pirma klausti:** nauja ar keičiama priklausomybė (ir `apt` paketai, Docker atvaizdai), įspėjimų ar testų slopinimas
  (`NOLINT`, sanitizerių slopinimo sąrašai), slenksčių keitimas, testų trynimas ar perrašymas, CI/CD, viešos sąsajos
  (`include/vramflow/gfx.h`) keitimas, bet kas destruktyvaus, kitų GPU darbų stabdymas.
- Testai niekada neatlaisvinami. TDD: pirma krentantis testas (parodyti išvestį), tada kodas.
- „Padaryta“ = ką tik paleistos komandos išvestis. 0 kompiliatoriaus įspėjimų. Kodo komentarai lietuviškai; README, ADR,
  `SECURITY.md`, `docs/operations.md` — angliškai.
- Našumo pakeitimui — „prieš / po“ iš ≥ 3 pakartojimų. Tik sintetinė medžiaga. MXL viduje — tik nesuspausta.
- Vieno paketo žingsniai (visiems vienodi, žemiau nekartojami): (1) krentantis testas; (2) paleisti — FAIL su laukiamu
  tekstu; (3) kodas; (4) testas PASS, švarus build 0 įspėjimų, `ctest`; (5) susiję funkciniai testai; (6) commit ir push.

## 2. Kriterijai (iš užduoties)

Būsenos: **PASS** (skaičius + komanda), **FAIL** (skaičius + kodėl), **NEPATIKRINAMA ČIA** (ko reikia lauko bandyme).

| ID | Kriterijus | Paketas |
|---|---|---|
| O1 | VRAM biudžeto lentelė kiekvienai funkcijai; nepaaiškinta dalis ≤ 10 % | 4.2 |
| O2 | `dec`, `ingest`, `enc`: ≥ 20 % mažiau VRAM srautui, kelias nesumažėja, 0 vėlavimų | 4.3 |
| O3 | 2 h soak: VRAM augimas ≤ 32 MiB, RSS ≤ 1 %, fd augimas 0 | 5.1 |
| O4 | Tick'o cikle nėra GPU atminties skyrimų — skaitiklis po apšilimo = 0 | 4.1 |
| O5 | Išmatuota pastovi proceso dalis; VRAM srautui prie istorijos 100 ir 200 ms | 4.2 |
| O6 | Variklio gijos CPU srautui; CUDA laukimo režimas parinktas pagal matavimą | 4.4 |
| G1 | Prie N = `floor(0,85 · kelias)`: 0 vėluojančių ir praleistų grain'ų per 3 × 60 s | 3.3 |
| G2 | Grandinės vėlavimų prie N = 12 šakninė priežastis įrodyta; pataisyta arba dokumentuota | 3.2 |
| G3 | Keturių nestabilių laiko testų šakninės priežastys iš ≥ 20 pakartojimų; taisoma priežastis | 3.1 |
| G4 | Latencijos lentelė šuoliui (p50 / p99); vartai: p99 neblogėja > 10 % | 3.3 |
| G5 | Išmatuota: laikas iki pirmo commit, atsigavimas po `kill -9` | 3.3 |
| K1 | 0 įspėjimų; `ctest` 100 %; visas funkcinis rinkinys 100 % tris kartus iš eilės | 6.3 |
| K2 | ASan + UBSan visam host kodui; TSan gijų kodui; `compute-sanitizer` visiems branduoliams | 2.1, 2.2, 2.3 |
| K3 | `clang-tidy`: 0 neišspręstų radinių sutartame tikrinimų rinkinyje | 2.4 |
| K4 | Klaidų keliai turi testus; baigiasi per ≤ 5 s, kiti procesai nepaveikti | 2.5 |
| K5 | 2 h soak (grandinė, `ingest`): 0 late, 0 missing, A/V ≤ 1 laukas | 5.1 |
| K6 | GPU branduoliai bitų tikslumu prieš CPU etaloną — po kiekvieno optimizavimo | 4.1–4.4 |
| K7 | Operatoriaus vadovas, lauko bandymo instrukcija, README „Limits“ | 6.1 |
| K8 | Nepriklausoma peržiūra kiekvienai fazei; radiniai uždaryti | 6.3 |
| S1 | Grėsmių modelis ir `SECURITY.md`; kiekviena priemonė turi testą arba „nepriimta, nes …“ | 1.6 |
| S2 | Fuzz testai parseriams, gaunantiems ne mūsų baitus; ≥ 10 min kiekvienam be radinių | 1.5 |
| S3 | Piktybiško rašytojo testai: klaida, ne skaitymas už ribų (host ir GPU) | 1.4, 2.3 |
| S4 | Nuomos ir knygos failai kuriami nesekant nuorodų; teisių sprendimas dokumentuotas | 1.2 |
| S5 | Visas funkcinis rinkinys praeina ne root vartotoju | 1.3 |
| S6 | Build grūdinimas su `readelf` testu; kaina ≤ 2 % arba sprendžia autorius | 1.1 |
| S7 | Tinklo įėjimų eilių ribos ir sargas turi testus; dokumentuotas saugus paleidimas | 1.6 |
| S8 | Prisegtos versijos viename faile; licencijų suvestinė; paslapčių paieška istorijoje | 6.2 |
| S9 | GPU klaida → pranešimas ir exit kodas per ≤ 5 s; vieno proceso gedimas nepaveikia kitų | 2.5 |
| U1 | Build ir testai su PR #653 viršūne; skirtumai ataskaitoje | 5.2 |
| U2 | `cmake --install`, `RUNPATH` per `$ORIGIN`, versija binare ir santraukoje | 5.3 |
| U3 | VRAM flow tarp dviejų konteinerių: įrodyta arba paneigta | 5.4 |

**Nukrypimai nuo užduoties teksto (su priežastimi):**

- S2: CLI specifikacijų parseriams (`composite.cpp:67`, `gfxgen.cpp:67`, `verify.cpp:152`) — neigiami CLI testai, ne fuzz:
  tai operatoriaus įvestis (patikima pagal grėsmių modelį) ir `static` funkcijos `.cpp` failuose.
- S6: CET žyma (`IBT`, `SHSTK`) binare priklauso nuo visų susietų objektų; `CUDA::cudart_static` jos neturi. Tikrinama
  vėliava mūsų vertimo vienetuose, binaro žyma — tik informacijai.
- K2: TSan — tik CPU variantui (`--impl cpu`): su NVIDIA tvarkykle TSan nepalaikomas.
- S9: tikro GPU dingimo čia saugiai sukelti negalima — injekcija, o tikras atvejis lieka NEPATIKRINAMA ČIA.

## 3. Bazė (išmatuota 2026-11-17, RTX 4070 Laptop 8 GB, WSL2, tvarkyklė 596.49)

| Kas | Reikšmė | Komanda |
|---|---|---|
| Švarus build ant `e570bae` | 0 įspėjimų | `ninja clean`; `30-build-verify.sh -DMXLV_BUILD_FUNC=ON` |
| `ctest` | 482 / 483; krito `gap_policy: žemupio blokuojantis skaitymas … < 1 ms` | `ctest` (120,9 s) |
| Tas pats testas 20 kartų, GPU tuščia | 14 PASS, 6 FAIL; kritusių reikšmės 1,01–1,28 ms (riba 1 ms) | `vramflow-glue-tests "gap_policy: žemupio…"` |
| Visas funkcinis rinkinys (ant `8bd366d`) | 352 testai, 2 h 25 min, krito `fiso_stall[G:first]` (25 034 µs, riba 25 000 µs) | `functional.py <out>` |
| Kelias, VRAM srautui (3 × 30 s) | logo ≥ 64, 56,0 MiB; composite K=2 ≥ 32, 119,1; dec 22, 182,2; NVENC 12 (sesijų riba), 126,2; chain 11, 284,3 | `docs/rezultatai/2026-11-17-p8-talpa.md` |
| Binarai | PIE — taip; RELRO — dalinis; `BIND_NOW`, CET — nėra; `RUNPATH` — absoliutūs build keliai | `readelf` |
| Kompiliatoriai | C++ `clang++-19`, CUDA host `g++`; `compile_commands.json` negeneruojamas | `CMakeCache.txt` |
| Ne root vartotojas | yra `mxltest` (uid 1000); `/dev/dxg` — 666, `/dev/shm` — 1777; CUDA jo vardu veikia | `getent passwd`, `stat`, `setpriv … vramflow-verify gpu-info` |
| Įrankiai | `clang-tidy-19`, `run-clang-tidy-19`, `clang++-18` (su sanitizerių ir fuzzer bibliotekomis), `compute-sanitizer` — yra | `command -v` |
| Docker | WSL integracija šiam distro neįjungta (`docker` komandos nėra) | `docker info` |
| MXL PR #653 | OPEN, viršūnė `998100c2` (2026-11-14); prisegta `6dddc91d` | `gh pr view 653 --repo dmf-mxl/mxl` |

VRAM biudžetas G1 lygiams (pastovi dalis + N · srautui, iš talpos failo): logo N=54 — 3,3 GiB; composite N=27 — 3,6 GiB;
dec N=18 — 3,3 GiB; NVENC N=10 — 1,5 GiB; chain N=9 — 2,9 GiB. Visi telpa (laisva ≈ 6,5 GiB).

## 4. Sprendimai (autorius, 2026-11-17: numatytosios reikšmės patvirtintos)

| # | Klausimas | Sprendimas |
|---|---|---|
| D1 | Tikslinė gamybos aplinka | Natyvus Linux su procesais. Konteineriai — tik įrodymas U3 |
| D2 | Kūrimo įrankiai | `.clang-tidy` ir `clang++-18` sanitizerių bei fuzz build'ams — taip (nieko nediegiama). CI — ne |
| D3 | PR #653 prisegimas | Lieka `6dddc91d`; skirtumai su viršūne tik išmatuojami (U1) |
| D4 | Priėmimo knyga visai GPU | Nekeičiama (K8): knyga lieka domenui, riba dokumentuota |

## 5. Peržiūros fokusas — 5 gedimai, kurių testai dar nedengia

| # | Gedimas | Kodėl tikėtinas | Testas (savininkas) |
|---|---|---|---|
| RF-1 | Naujas tikslas be grūdinimo vėliavų | Vėliavos pridedamos funkcija; pamiršus ją iškviesti niekas nepastebi | 1.1: testas tikrina **visus** `compile_commands.json` įrašus iš `src/` ir `tests/` ir visus `vramflow-*`, `mxlv-*`, `libvramflow*` ELF failus build kataloge, ne fiksuotą sąrašą |
| RF-2 | Tas pats domenas kitu keliu (simbolinė nuoroda, `/./`) → kita nuoma ir kita knyga | `lease_dir()` kelią ima tekstu (`domain_lease.cpp`), o MXL jį kanonizuoja; du procesai tada nemato vienas kito rezervacijų | 1.2: `ledger: tas pats domenas per simbolinę nuorodą — viena knyga` |
| RF-3 | Sanitizeriai žali tik todėl, kad scenarijai neina klaidų keliais | ASan randa tik tai, kas įvykdoma | 2.5 → 2.1: `faults.py` testai įtraukiami į ASan scenarijų sąrašą |
| RF-4 | Parseris priima reikšmę, kuri vėliau perpildo aritmetiką | „Nelūžta“ ≠ „priimta reikšmė saugi“ (pvz. `frame_width` 2³¹−1 → `grain_bytes`) | 1.5: fuzz invariantas — priimtam formatui `grain_bytes` ≤ 7680×4320 `v210a` dydis ir skaičiuojama 64 bitais |
| RF-5 | Nuotėkis ne pastovioje apkrovoje, o perjungimuose | Re-attach skiria scratch (`composite.cpp:510`); pastovus soak to nemato | 5.1: soak variantas su gamintojo perkrovimu kas 60 s, 30 min — VRAM ir fd lygūs |

## 6. Darbai

### Fazė 1 — saugumas ir build

Pirmiausia, nes pigu, o grūdinimo vėliavos keičia binarus — jos turi būti prieš visus našumo matavimus.

#### 1.0 Smulkus taisymas

`README.md` exit kodų lentelėje eilutė apie atmetimą mini tik `--admission enforce`; po K13 pataisos tas pats galioja ir
numatytajai `auto` su talpos failu. Pataisyti tekstą (ir `README.lt.md`, jei kartojasi). Commit: `docs: exit code 5 also
applies to the default auto mode`.

#### 1.1 Build grūdinimas (S6, RF-1)

- **Kurti:** `cmake/hardening.cmake`, `tests/arch/check_hardening.py`.
- **Keisti:** `CMakeLists.txt` (`include`, `CMAKE_EXPORT_COMPILE_COMMANDS ON`, `CMAKE_POSITION_INDEPENDENT_CODE ON`,
  `check_pie_supported()`), `cmake/warnings.cmake` (`vramflow_warnings()` pabaigoje kviečia `vramflow_hardening()` — taip
  dengiami visi savi tikslai).
- **Sąsaja:** `vramflow_hardening(<tikslas>)`; parinktis `VRAMFLOW_HARDENING` (numatyta `ON`).
  C++ vertimo vienetams: `-fstack-protector-strong -fstack-clash-protection -fcf-protection=full -D_GLIBCXX_ASSERTIONS`
  ir (tik optimizuotam build'ui be sanitizerių) `-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3`. CUDA host daliai — tos pačios per
  `-Xcompiler`. Susiejimui: `-Wl,-z,relro,-z,now -Wl,-z,noexecstack`.
- **Testas** `build_hardening` (`ctest`; `SKIP_RETURN_CODE 77`, kai nėra `readelf`):
  - kiekvienam ELF failui build kataloge pagal šabloną `vramflow-*`, `mxlv-*`, `libvramflow*.so*`: `Type: DYN`; yra
    `GNU_RELRO`; `readelf -d` rodo `BIND_NOW` arba `FLAGS_1 … NOW`; `GNU_STACK` be `E`; yra `__stack_chk_fail`;
  - kiekvienam `compile_commands.json` įrašui, kurio failas yra `src/` ar `tests/` ir baigiasi `.cpp`: komandoje yra visos
    penkios vėliavos.
  Prieš pakeitimą laukiama: `FAIL … BIND_NOW nėra`.
- **Kaina** (prieš / po, 3 × 30 s): `capacity.py --func logo --n 32,64`; `--func chain --n 8`; `--func logo --impl cpu
  --storage host --n 4` (CPU branduoliai — čia `_GLIBCXX_ASSERTIONS` gali kainuoti). Jei kelias ar p99 pablogėja > 2 % —
  lentelė autoriui, sprendžia jis.
- **Commit:** `build: hardening flags for every target with a readelf gate`.

#### 1.2 Nuomos ir knygos failai (S4, RF-2)

Šiandien: katalogas `<domenas>.vramflow/` kuriamas `mkdir 0775` ir, jei jau yra, naudojamas netikrinant
(`domain_lease.cpp:132`, `admission.cpp:476`); laikinas nuomos failas — `open(O_CREAT|O_TRUNC)` nuspėjamu vardu (`:137`);
svetimi `.lease` / `.adm` failai atidaromi `O_RDONLY` (FIFO tokį atidarymą pakabintų). `/dev/shm` — 1777, todėl kitas
vietinis vartotojas katalogą gali sukurti pirmas.

- **Kurti:** `src/runtime/safe_files.hpp`, `src/runtime/safe_files.cpp`, `tests/unit/test_safe_files.cpp`.
- **Keisti:** `src/runtime/domain_lease.cpp` (`lease_dir`, `acquire`, `probe`, `live_owners`, `flow_owner`),
  `src/runtime/admission.cpp` (`DirLock`, `Ledger::open`, `Ledger::others`), `cmake/runtime.cmake`,
  `tests/perf/domain.py` (tas pats katalogo kelias kaip `lease_dir`).
- **Sąsaja:**

```cpp
namespace mxlv::runtime {
// Atidaro (jei reikia — sukuria) `<kanoninis domenas>.vramflow/`; grąžina katalogo fd (O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC).
// Klaida: ne katalogas; savininkas ≠ domeno katalogo savininkas ir ≠ geteuid(); teisės platesnės nei domeno katalogo.
int open_lease_dir(const std::string& domain, std::string& err);
// openat(dirfd, name, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode). Jau yra (ir nuoroda) → -1, errno = EEXIST.
int create_file_at(int dirfd, const std::string& name, mode_t mode);
// openat(dirfd, name, flags | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC) ir fstat: tik paprastas failas; kitaip -1, errno = EINVAL.
int open_regular_at(int dirfd, const std::string& name, int flags);
}
```

- **Teisių taisyklė (README „Domain ownership“; po peržiūros — rezultatų §1.7):** prieiga prie nuomos katalogo = RAŠYMO
  teisė į domeną (`0755` domenas → `0700` katalogas ir `0600` failai; `0770` → `0770` ir `0660` su domeno grupe), nes `flock`
  užtenka skaitomo failo. Užraktų laukiama ribotai (5 s); nuomos skelbiamos ir negyvos šalinamos tik po `lease.lock`.
  `lease_dir()` domeno kelią kanonizuoja.
- **Testai** (`mxlv-admission-unit`):
  - `safe_files: simbolinė nuoroda vietoje nuomos katalogo atmetama` — `acquire` → `false`, tekste „ne katalogas“; nuorodos
    taikinys lieka tuščias;
  - `safe_files: create_file_at neseka nuorodos` — `x` → nuoroda į failą su turiniu `A`; grąžina -1 (`EEXIST`), turinys `A`;
  - `safe_files: FIFO vietoje .lease nepakabina live_owners` — grįžta per < 1 s, sąrašas tuščias;
  - `safe_files: FIFO vietoje .adm nepakabina Ledger::total` — grįžta per < 1 s, suma 0;
  - `safe_files: platesnės nei domeno teisės atmetamos` — domenas `0750`, nuomos katalogas `0777` → klaida „teisės“;
  - `safe_files: teisės paveldimos` — domenas `0750` → katalogas `0750`, failas `0640`; `0770` → `0770` ir `0660`;
  - **[RF-2]** `ledger: tas pats domenas per simbolinę nuorodą — viena knyga` — A rezervuoja per tikrą kelią, B per
    nuorodą mato `total` = A rezervacija;
  - `safe_files: svetimas savininkas atmetamas` — tik root (`chown 65534`); ne root — `SKIP` su priežastimi.
  Esami nuomos ir knygos testai lieka žali; `kill -9` testas — 30 kartų iš eilės.
- **Commit:** `fix(runtime): lease and ledger files never follow links; one ledger per canonical domain`.

#### 1.3 Paleidimas ne root (S5)

- Vartotojas `mxltest` jau yra — nieko kurti nereikia. Patikrinta 2026-11-17: jis skaito build ir MXL katalogus, rašo į
  `/dev/shm`, o `vramflow-verify gpu-info` jo vardu grąžina GPU sąrašą (CUDA veikia). Visas rinkinys jo vardu dar neleistas.
- **Kurti:** `scripts/linux/45-nonroot.sh <vartotojas> [funkcinio rinkinio parinktys]`: kopijuoja build išvestį ir MXL
  `install/` (13 MiB) į `/home/<vartotojas>/vramflow-nonroot/`, per `setpriv --reuid --regid --init-groups` paleidžia `ctest`
  (iš kopijos) ir `functional.py`; pabaigoje palygina rezultatų rinkinį su root paleidimu.
- **Keisti (jei kris):** `tests/functional/functional.py:30` (`MXL_INFO` numatytasis kelias `/root/…` → pagal `HOME` arba
  `MXL_SRC`), testai, kuriems būtinas root, — `SKIP` su priežastimi (ne `PASS`).
- **Dokumentai:** README skyrius „Running as a service user“: prieiga prie GPU įrenginių (`/dev/nvidia*`, WSL — `/dev/dxg`),
  `/dev/shm`, domeno katalogo teisės, `CAP_SYS_NICE` tik `--rt-priority`.
- **Priėmimas:** funkcinis rinkinys ne root — tas pats PASS rinkinys kaip root (nestabilūs laiko testai vertinami 3.1).

#### 1.4 Piktybiškas rašytojas (S3)

Šiandien `VideoIn::get` tikrina tik slot ribą ir `info.index` (`flows.cpp:379`); `grainSize` tikrinamas tik rašymo kelyje
(`:196`). Layout prieš rastrą tikrinamas atidarant (`check_layout`).

- **Kurti:** `tests/gpu/test_hostile_writer.cpp` (į `vramflow-flows-tests`).
- **Testai** (rašytojas — tiesiai per MXL API; skaitytojas — `VideoIn`, abu saugojimo tipai):
  - `hostile: grainSize ≠ rastro dydžiui` (0, tikras + 1, `UINT32_MAX`) → `get` grąžina ne `MXL_STATUS_OK`, rodyklė `nullptr`;
  - `hostile: validSlices > totalSlices` ir `totalSlices == 0` → tas pats;
  - `hostile: flow aprašas didesnis už layout` (flow sukurtas 1280×720, `flow_def.json` perrašytas į 1920×1080) →
    `VideoIn::open` → `false`, tekste „layout“;
  - `hostile: neįmanomi matmenys apraše` (`frame_width` 0, −1, 2³¹−1; `frame_height` 0) → `format_from_flow_def` → `false`.
- **Keisti (laukiama, kad pirmi du kris):** `src/func/flows.cpp` (`VideoIn::get`), jei reikia — `src/engine/mxl_ports.cpp`.
- **Patikra:** tas pats testų failas po ASan (2.1) ir `compute-sanitizer memcheck` (2.3) — 0 ataskaitų.
- **Commit:** `fix(flows): reader validates the grain header before use; hostile writer tests`.

#### 1.5 Fuzz testai (S2, RF-4)

- **Kurti:** `cmake/fuzz.cmake` (parinktis `MXLV_BUILD_FUZZ`, numatyta `OFF`), `tests/fuzz/fuzz_flow_def.cpp`,
  `fuzz_capacity.cpp`, `fuzz_playlist.cpp`, `fuzz_url.cpp`, `tests/fuzz/replay_main.cpp`, `tests/fuzz/corpus/<taikinys>/`,
  `scripts/linux/56-fuzz.sh`.
- **Taikiniai** (visi — gryna logika, be MXL, CUDA ir libav):

| Taikinys | Funkcijos | Invariantas, kai priimta |
|---|---|---|
| `fuzz_flow_def` | `vramflow::format_from_flow_def` (`src/common/format.hpp:180`) | **[RF-4]** 1 ≤ w, h ≤ 65536 (parseris palaikomo dydžio netikrina — tai daro `check_layout`); `grain_bytes` (64 bitų) ≤ 65536×65536 `v210a` dydis |
| `fuzz_capacity` | `mxlv::runtime::parse_capacity` (`admission.hpp:64`) | visi skaičiai baigtiniai ir ≥ 0; `nvenc_sessions` ≥ 0 |
| `fuzz_playlist` | `parse_playlist`, `check_playlist`, `resolve_playlist_url` (`src/ingest/url_policy.hpp:75,134,35`) | priimto grojaraščio nuorodų ≤ `HlsLimits::max_segments` |
| `fuzz_url` | `source_url_allowed`, `canonical_authority`, `redact_url` (`url_policy.hpp:24,120,43`) | priimto URL schema — iš leidžiamų sąrašo; `redact_url` išvestyje nėra slaptažodžio |

- `56-fuzz.sh`: build su `CC_BIN=clang-18 CXX_BIN=clang++-18 -DMXLV_BUILD_FUZZ=ON` į `~/build/vramflow/fuzz`
  (`-fsanitize=fuzzer,address,undefined`); kiekvienam taikiniui `-max_total_time=${FUZZ_SECONDS:-600} -timeout=5
  -rss_limit_mb=2048`; rezultatas `fuzz.json` (taikinys, vykdymų skaičius, radiniai).
- **Regresija be fuzzer bibliotekos:** `replay_main.cpp` — `main`, kuris per tą patį `LLVMFuzzerTestOneInput` paleidžia
  visus korpuso failus; `ctest` testai `fuzz_replay_<taikinys>` gamybiniame build'e. Radinys → failas į korpusą → lieka
  testu visam laikui.
- **CLI specifikacijos** (nukrypimas, §2): neigiami atvejai `tests/functional` (`fcli1` šalia): `--layer-xy "1,"`, `"a,b"`,
  `"1,2,3"`, `"99999999999,0"` → exit 1 su tekstu.
- **Commit'ai:** po vieną taikiniui.

#### 1.6 Grėsmių modelis, `SECURITY.md`, saugus paleidimas (S1, S7)

- **Kurti:** `docs/architektura/2026-11-NN-gresmiu-modelis.md` (LT), `SECURITY.md` (EN).
- **Grėsmių modelio skyriai (fiksuoti):** turtas; pasitikėjimo ribos (domeno katalogas = riba; tinklo įėjimai nepatikimi;
  kiti domeno procesai pusiau patikimi; operatoriaus konfigūracija patikima); grėsmių lentelė `T-n | šaltinis | poveikis |
  priemonė | testas arba „nepriimta, nes …“ | likutinė rizika`. Kiekviena 1.1–1.5 ir 2.5 priemonė — eilutė su testo vardu.
- **`SECURITY.md`:** palaikomos versijos; kaip pranešti (GitHub privatus pranešimas — **autorius turi jį įjungti repo
  nustatymuose; asmeninio el. pašto nerašyti**); pasitikėjimo modelio santrauka; saugus paleidimas: ne root, atskiras
  vartotojas, systemd (`NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp`, `DeviceAllow` GPU įrenginiams),
  konteineris be `--privileged`.
- **S7:** eilių ribų ir sargo testai jau yra (`tests/unit/test_packet_queue.cpp`, `test_libav_watchdog.cpp`) — patikrinti,
  kad dengia baitų ir kiekio ribas bei exit 7; ko trūksta — pridėti. Tinklo įėjimų kodas kitaip neliečiamas.

### Fazė 2 — tikrinimo gylis

Prieš greičio darbus: sanitizeriai gali parodyti nestabilių testų priežastis.

#### 2.1 ASan + UBSan visam host kodui (K2, RF-3)

- **Keisti:** `scripts/linux/50-sanitizers.sh`: build `CC_BIN=clang-18 CXX_BIN=clang++-18 -DMXLV_ENABLE_SANITIZERS=ON
  -DMXLV_BUILD_FUNC=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo` į `~/build/vramflow/asan`. **[nepatikrinta]**, ar `clang++-18`
  kodas susiejamas su `clang++-19` kompiliuota MXL (C sąsaja — turėtų).
- **Paleidžiama:** visi ne GPU vieneto binarai (`mxlv-unit`, `mxlv-engine-unit`, `mxlv-gfx-unit`, `mxlv-ingest-unit`,
  `mxlv-media-cpu-unit`, `mxlv-media-unit`, `mxlv-timing-unit`, `mxlv-admission-unit`); GPU testai su
  `ASAN_OPTIONS=protect_shadow_gap=0` (`vramflow-flows-tests`, `vramflow-runtime-tests`, `vramflow-glue-tests "~[realtime]"`,
  `vramflow-gfx-writer-tests "~[realtime]"`); funkciniai scenarijai su `FUNC_BIN` iš ASan build'o:
  `fcomp1,ffmt_logo,ffmt_dec,ffmt_enc,overload_partial` ir **[RF-3]** visi `faults.py` testai (2.5).
- **Vertinimas:** tik „sanitizerio ataskaitų mūsų kode = 0 ir nėra avarijos (exit 134, 139)“. Laiko tvirtinimai po ASan
  nevertinami (lėtėja ~2×) — tai įrašoma į `sanitizers.json` kaip `INFO`.
- Trečiųjų šalių nuotėkiai (tvarkyklė, libav) — slopinimo sąrašas tik autoriui patvirtinus.

#### 2.2 TSan (K2)

- **Keisti:** `CMakeLists.txt` — parinktis `MXLV_ENABLE_TSAN` (nesuderinama su `MXLV_ENABLE_SANITIZERS` — konfigūravimo
  klaida); `50-sanitizers.sh` — build į `~/build/vramflow/tsan`.
- **Paleidžiama:** `mxlv-engine-unit`, `mxlv-ingest-unit`, `mxlv-timing-unit`, `mxlv-admission-unit`,
  `vramflow-glue-tests "~[realtime]"`; `vramflow-func` be GPU: `gen → logo → verify` su `--impl cpu --storage host` 10 s ir
  `ingest --decoder sw` 20 s.
- **Priėmimas:** 0 duomenų lenktynių mūsų kode.

#### 2.3 `compute-sanitizer` visiems branduoliams (K2, S3)

- **Keisti:** `50-sanitizers.sh` — `memcheck` ir šiems binarams: `mxlv-media-gpu-tests`, `vramflow-gfx-gpu-tests`,
  `vramflow-v210a-gpu-tests`, `vramflow-flows-tests` (su 1.4 testais).
- **Pridėti kraštinius rastrus**, jei GPU testuose jų nėra: plotis ne 48 kartotinis (1280), 382×40, 3840×2160.
- **Priėmimas:** kiekvienam `ERROR SUMMARY: 0 errors`.

#### 2.4 `clang-tidy` (K3)

- **Kurti:** `.clang-tidy` (`bugprone-*`, `cert-*`, `concurrency-*`, `performance-*`, `clang-analyzer-*`;
  `HeaderFilterRegex: 'src/.*'`), `scripts/linux/55-clang-tidy.sh` (`run-clang-tidy-19 -p ~/build/vramflow/func`, be `.cu`;
  išvestis — radiniai ir skaičiai pagal tikrinimą).
- **Eiga:** (1) bazė be taisymų → lentelė autoriui: kiek kurio tikrinimo radinių ir kuriuos siūlau išjungti konfigūracijoje
  kaip triukšmą (tai tikrinimų parinkimas, ne radinio slopinimas); (2) taisymai po vieną tikrinimų grupę, elgsena
  nekeičiama, testai žali; (3) etapas `run-all.sh`.
- **Priėmimas:** 0 radinių sutartame rinkinyje; `NOLINT` — tik su priežastimi ir autoriaus žinia.

#### 2.5 Klaidų kelių testai (K4, S9)

- **Kurti:** `tests/functional/faults.py` (registruoti `functional.py`).
- **Keisti:** `src/func/main_func.cpp`, `func_common.hpp` — paslėpta parinktis `--test-cuda-fail-after S` (`group("")`, kaip
  esamos `--test-*`); `logo.cpp`, `gen.cpp` — injekcija ten, kur tikrinamas `cudaStreamSynchronize` rezultatas.
- **Testai** (kiekvienas matuoja laiką nuo gedimo iki proceso pabaigos; riba 5 s):

| Testas | Kaip sukeliama | Laukiama |
|---|---|---|
| `fault_vram_exhausted[gen\|logo\|dec]` | `vramflow-verify hog --leave-free-mb` palieka mažiau, nei reikia | exit 2, pranešime VRAM / `cudaMalloc`; domene nėra šio paleidimo flow'ų; šalia veikianti `gen → verify` pora — `late == 0` |
| `fault_nvenc_sessions` | veikia `enc --flows 12`, paleidžiamas antras `enc --flows 1 --admission off` | antras — exit 2 su tvarkyklės ribos tekstu; pirmas — 0 late |
| `fault_cuda_midrun[logo\|gen]` | `--test-cuda-fail-after 5` | exit 2 per ≤ 5 s, CUDA klaidos tekstas, išėjimo flow'ai atlaisvinti, `summary.exit_code == 2` |
| `fault_shm_full` | domenas 16 MiB `tmpfs` (tik root; kitaip `SKIP`) | exit 2 su aiškiu tekstu; dalinių flow katalogų nelieka |
| `fault_domain_readonly` | domeno katalogas `0500` (tik ne root; kitaip `SKIP`) | exit 2 su `EACCES` tekstu |

- Tikras GPU dingimas — NEPATIKRINAMA ČIA (§2).

### Fazė 3 — greitis ir stabilumas

#### 3.1 Nestabilių laiko testų šakninės priežastys (G3)

Testai: `ctest gap_policy … < 1 ms` (bazė: 6 iš 20 krenta, 1,01–1,28 ms), `fiso_stall` (25 034 µs ir 34 227 µs prieš
25 000 µs), `fsplit[composite-G]` (`mismatch_fields` 20), `interop_foreign_writer`.

- **Kurti:** `tests/perf/flake.py <testas> --runs 20` — kartoja testą, renka išmatuotas reikšmes iš santraukų ir aplinkos
  pėdsaką (GPU dažnis ir galios būsena 10 Hz, ar veikia `keepwarm`), rašo skirstinį.
- **Hipotezės, kiekviena su skiriamuoju matavimu:**
  1. variklio gijos pabudimo vėlavimas (planuoklė) — ar kritimai dingsta su `--rt-priority 50 --cpu-affinity`;
  2. WSL2 GPU galios būsenų perjungimai — ar kritimai sutampa su dažnio šuoliais;
  3. riba išvesta natyviam Linux (`config/thresholds.json`: „~10× virš native“) — ar skirstinys čia iš principo netelpa;
  4. tikras kodo kelio vėlavimas — pėdsakas `VRAMFLOW_TRACE_*`.
- **Taisyklė:** 4 — taisomas kodas; 1–2 — testas pats susikuria aplinką, kurios jam reikia (prioritetas, branduolys), riba
  nekeičiama; 3 — duomenys autoriui, ribą keičia tik jis.
- **Priėmimas:** kiekvienas testas 20 iš 20 dviejose atskirose sesijose arba autoriaus patvirtinta platformos riba.

#### 3.2 Grandinės vėlavimai prie N = 12 (G2)

- **Atkūrimas:** `capacity.py --func chain --n 11,12 --repeats 5 --seconds 60`; renkama: `dec.late_skips`,
  `enc.late_frames`, `proc_ms` / `wait_ms` / `held_ms` histogramos, GPU variklių apkrova 10 Hz, CPU pagal giją.
- **Hipotezės:** (1) GPU laiko dalijimas tarp trijų procesų kontekstų — `logo` vienas iki 64 veikia, grandinėje ne;
  (2) CPU: 12 srautų × (demux + AAC + mux gijos) nešiojamame procesoriuje; (3) `dec` atkūrimo eilė apkrovoje
  (`late_skips` — pirmas simptomas); (4) NVENC ties sesijų riba elgiasi kitaip (vienas `enc` prie 12 praeina).
- **Rezultatas:** priežastis su eksperimentų lentele; jei kodas — pataisa ir kelias „prieš / po“ (3 × 60 s); jei platforma —
  įrašas `docs/rezultatai/` ir README §8.

#### 3.3 Regresijos vartai ir matavimai (G1, G4, G5)

- **Kurti:** `tests/perf/regress.py`, `tests/perf/test_regress.py` (`ctest perf_regress`), `tests/perf/baseline/`.
- **Sąsaja:** `regress.py <kampanija> --baseline <failas> [--update]` → exit 0 / 1; FAIL, kai kelias mažesnis, p99 > 1,10 ×
  bazės arba VRAM srautui > 1,05 × bazės.
- **Vieneto testai:** vienodi → PASS; kelias −1 → FAIL su funkcijos vardu; p99 +11 % → FAIL; funkcijos nėra kampanijoje →
  FAIL „nematuota“; p99 +9 % → PASS.
- **G1:** `capacity.py --func <f> --n <floor(0,85·kelias)> --repeats 3 --seconds 60` — logo 54, composite 27, dec 18,
  NVENC 10, chain 9; `ingest` — pagal `ingest_capacity.py` kelią.
- **G4:** lentelė p50 / p99 iš santraukų histogramų prie N = 1 ir N = 0,85 · kelias. **G5:** laikas nuo proceso paleidimo iki
  pirmos `--stats-file` eilutės su `ok > 0`; atsigavimas po `kill -9` — iš `startstop` testų arba naujas matavimas.

### Fazė 4 — VRAM

Tik ant stabilios bazės; kiekvienas paketas baigiasi `regress.py` ir bitų tikslumo testais (K6).

#### 4.1 Skyrimų skaitiklis (O4)

- **Kurti:** `src/common/cuda_alloc.hpp`, `tests/arch/check_cuda_alloc.sh` (`ctest arch_cuda_alloc`).
- **Sąsaja:**

```cpp
namespace mxlv {
cudaError_t cuda_malloc(void** p, std::size_t bytes);
cudaError_t cuda_free(void* p);
cudaError_t cuda_host_alloc(void** p, std::size_t bytes, unsigned flags);
cudaError_t cuda_host_free(void* p);
void cuda_warm_context();                 // vietoje cudaFree(nullptr)
void cuda_alloc_mark_steady();            // kviečiama pasibaigus apšilimui
std::uint64_t cuda_allocs_after_steady(); // skyrimų ir atlaisvinimų skaičius po žymos
}
```

- **Keisti:** visi tiesioginiai `cudaMalloc`, `cudaFree`, `cudaHostAlloc`, `cudaMallocHost`, `cudaFreeHost` kvietimai `src/`
  (≈ 60 vietų) — mechaniškai; santraukoje naujas laukas `gpu_allocs_after_warmup`; auksiniai raktų failai
  `tests/functional/data/summary_keys_*.json` gauna neprivalomą raktą.
- **Testai:** `arch_cuda_alloc` — `src/` nėra tiesioginių kvietimų už `cuda_alloc.hpp` ribų; funkciniuose `fcomp1`,
  `ffmt_logo`, `ffmt_dec`, `ffmt_enc` ir `ingest` testuose — `gpu_allocs_after_warmup == 0`.
- Riba: libav vidiniai skyrimai (NVDEC, NVENC paviršiai) čia nematomi — juos dengia O3.

#### 4.2 VRAM biudžetas (O1, O5)

- **Kurti:** `tests/perf/vram_budget.py`, `tests/perf/test_vram_budget.py` (`ctest perf_vram_budget`).
- **Vieneto testai:** 1080i50 `v210` laukas 2 764 800 B × 10 = 26,4 MiB; 1080p50 — 5 529 600 B × 10; NV12 1920×1088 —
  3 133 440 B.
- **Matavimai:** istorija 100 ms prieš 200 ms (logo, dec); 1 procesas × 8 srautai prieš 8 procesus × 1 srautą (logo).
- **Rezultatas:** lentelė kiekvienai funkcijai `eilutė | teoriškai | išmatuota | skirtumas`; rekomendacijos README.

#### 4.3 Dekoderio ir koduotuvo paviršiai (O2)

Įtarimai **[nepatikrinta matavimu]**: `dec` (`h264_cuvid`) nenurodo `surfaces` (`dec.cpp:122-127`) — FFmpeg numatytoji
tikėtina 25; `enc` — `initial_pool_size = 16` ir `surfaces=8` (`enc.cpp:204,416`).

- **Eksperimentai** (kiekvienam: VRAM srautui iš N = 1, 4, 8; kelias; late; pirmo tick'o vėlavimas):
  `dec` `surfaces` ∈ {numatyta, 20, 16, 12, 10, 8}; `ingest` `extra_hw_frames` esama prieš mažiausią veikiančią;
  `enc` `initial_pool_size` ∈ {16, 8, 4, 2} × `surfaces` ∈ {8, 4}.
- **Sprendimas po matavimo:** skaičius išvedamas iš srauto (`max_dec_frame_buffering` + atkūrimo kadrai + 2); CLI parinktis
  pridedama tik jei formulė netinka visiems testiniams srautams.
- **Testai:** formulės vieneto testas; `ffmt_dec`, `ffmt_dec_chroma`, `ffmt_enc`, `ingest` rinkinys žali; bitų tikslumas
  nepakitęs.
- **Priėmimas:** ≥ 20 % mažiau VRAM srautui `dec` ir `enc`, kelias ne mažesnis, 0 late; kitaip — skaičiai, kodėl ne.

#### 4.4 CPU ir CUDA laukimo režimas (O6)

- **Matavimas:** variklio gijos CPU srautui ir commit p99 prie 0,85 · kelio (logo, dec, enc) su numatytuoju, `blocking`,
  `yield`, `spin` režimu (`cudaSetDeviceFlags` prieš konteksto sukūrimą; `enc` ir `dec` jau naudoja FFmpeg pirminį
  kontekstą su `BLOCKING_SYNC` — `enc.cpp:395`).
- **Sprendimas:** pagal lentelę; parinktis `--cuda-sched` tik jei vienas režimas netinka visoms funkcijoms.

### Fazė 5 — soak, upstream, diegimas

#### 5.1 Soak (K5, O3, RF-5)

- **Kurti:** `tests/perf/soak.py`, `tests/perf/drift.py` (slinkio skaičiavimas, iškeltas iš `scale.py`),
  `tests/perf/test_drift.py`.
- **Ką daro:** paleidžia scenarijų, kas 10 s kiekvienam procesui renka `VmRSS`, fd ir gijų skaičių, įrenginio VRAM; skaito
  `--stats-file` eilutes (late, missing); pabaigoje taiko `config/thresholds.json` → `soak` ir A/V patikrą.
- **Paleidimai:** grandinė N = 9, 2 h; `ingest` prie 0,85 · kelio, 2 h; **[RF-5]** grandinė su gamintojo perkrovimu kas 60 s,
  30 min. Tuo metu GPU nieko kito neleidžiama.
- **Priėmimas:** 0 late, 0 missing, A/V ≤ 1 laukas, VRAM augimas ≤ 32 MiB, RSS ≤ 1 %, fd augimas 0.

#### 5.2 PR #653 viršūnė (U1)

- **Kurti:** `scripts/linux/12-build-mxl-head.sh` — atskiras klonas `~/src/mxl-pr653-head` ties PR viršūne (SHA įrašomas),
  VRAMFlow build `~/build/vramflow/func-head` su `MXL_SRC` perrašymu.
- **Paleidžiama:** `ctest`, `functional.py --only fcomp1,ffmt_logo,ffmt_dec,ffmt_enc,interop…`, MXL testai su `upstream/`
  pataisomis U1–U3 (ar dar taikosi).
- **Rezultatas:** skirtumų lentelė `docs/rezultatai/`; prisegimo nekeičiame (D3).

#### 5.3 Diegimas ir versija (U2)

- **Kurti:** `cmake/version.cmake` (`git describe --tags --always --dirty` → `VRAMFLOW_VERSION`), `cmake/install.cmake`,
  `tests/arch/check_install.sh` (`ctest install_tree`).
- **Keisti:** `src/func/main_func.cpp`, `src/main.cpp` (`--version`), `versions_json` (laukas `vramflow`).
- **Testas `install_tree`:** `cmake --install <build> --prefix <tmp>`; `<tmp>/bin/vramflow-func --version` veikia;
  `readelf -d` `RUNPATH` turi `$ORIGIN` ir neturi build katalogo kelio; `libvramflow-gfx.so` ir `gfx.h` įdiegti.

#### 5.4 VRAM flow tarp konteinerių (U3)

- **Pirma klausti:** Docker Desktop WSL integracija šiam distro išjungta — ją įjungti turi autorius; bazinio atvaizdo
  siuntimas (keli GiB) — taip pat tik jam leidus.
- **Žinoma iš anksto (1 fazės peržiūra):** nuomos katalogas `<domenas>.vramflow` yra šalia domeno, todėl montuoti reikia
  tėvinį katalogą — sumontavus tik domeną kiekvienas konteineris turėtų savo nuomas ir savo priėmimo knygą.
- **Eksperimentas:** atvaizdas su 5.3 diegimo medžiu; du konteineriai su ta pačia GPU ir bendru domenu (tėvinis katalogas
  su `<d>` ir `<d>.vramflow` kaip bind mount): A — `gen --storage cuda-linear`, B — `verify`. Matrica: numatytosios vardų erdvės, `--ipc=host`,
  `--pid=host`. `cuda-linear` skaitytojas fd gauna per Unix lizdą flow kataloge (`payload.sock`).
- **Rezultatas:** `docs/rezultatai/…-konteineriai.md` — veikia arba ne ir kodėl; jei veikia — minimalus pavyzdys.

### Fazė 6 — dokumentai ir priėmimas

#### 6.1 Operatoriaus vadovas ir lauko bandymas (K7)

- **Kurti:** `docs/operations.md` (EN): paleidimo tvarka, exit kodai, gyva statistika, talpos matavimas ir priėmimo
  kontrolė, atnaujinimas, žinomos ribos; `docs/lauko-bandymas.md` (LT) ir `scripts/linux/90-field-test.sh` — viena komanda
  natyviame Linux: versijos, build, `ctest`, `PROFILE=knee` kampanija, `regress.py`, pasirinktinai 2 h soak; išvestis —
  vienas archyvas.
- **Keisti:** README §11 „Limits“.

#### 6.2 Versijos, licencijos, paslaptys (S8)

- **Kurti:** `config/versions.json` (MXL SHA, CUDA, FFmpeg, mažiausia tvarkyklė, kompiliatoriai) ir
  `scripts/linux/05-check-versions.sh` (lygina su aplinka, įspėja).
- **Licencijos** (`NOTICE`, README §12): kodas Apache-2.0; FFmpeg — sisteminis paketas, susiejamas dinamiškai, čia
  sukompiliuotas su `--enable-gpl --enable-libx264`; ką tai reiškia platinant binarus ar atvaizdus — inžinerinė suvestinė,
  teisinį vertinimą daro autorius.
- **Paslaptys:** raktų šablonų paieška visoje istorijoje (`git log -p`); rezultatas — skaičius ir vietos.

#### 6.3 Galutinė patikra (K1, K8)

- Švarus build, `ctest`, `50-sanitizers.sh`, `56-fuzz.sh`, `55-clang-tidy.sh`, `regress.py`; visas funkcinis rinkinys tris
  kartus iš eilės (≈ 7,5 h, fone).
- `/code-review high` kiekvienos fazės suminiam diff'ui; radiniai uždaromi arba pagrįstai atmetami.
- Galutinė kriterijų lentelė ir likusios rizikos.

## 7. Priklausomybės

```
1.1 (grūdinimas) ──► visi našumo matavimai (3.x, 4.x)
1.2, 1.4, 1.5 ──► 1.6 (grėsmių modelis remiasi jų testais)
1.3 (ne root) ──► 2.5 fault_domain_readonly
2.5 (klaidų keliai) ──► 2.1 (ASan scenarijai)
2.1–2.3 ──► 3.1 (sanitizeriai gali paaiškinti nestabilumą)
3.1, 3.2 ──► 3.3 (vartai tik ant stabilios bazės) ──► 4.x
4.x ──► 5.1 (soak ant galutinio kodo)
5.3 (diegimas) ──► 5.4 (konteineriai), 6.1 (lauko bandymas)
```

## 8. Būsenos lentelė

Atnaujinama po kiekvieno paketo: būsena (`nepradėta` / `vyksta` / `PASS` / `FAIL` / `NEPATIKRINAMA ČIA`), skaičius, įrodymo
failas arba commit.

| ID | Būsena | Skaičius / pastaba | Įrodymas |
|---|---|---|---|
| O1 | nepradėta | | |
| O2 | nepradėta | bazė: dec 182,2 MiB, NVENC 126,2 MiB srautui | `docs/rezultatai/2026-11-17-p8-talpa.md` |
| O3 | nepradėta | | |
| O4 | nepradėta | | |
| O5 | nepradėta | | |
| O6 | nepradėta | | |
| G1 | nepradėta | | |
| G2 | nepradėta | bazė: chain kelias 11 | `docs/rezultatai/2026-11-17-p8-talpa.md` |
| G3 | nepradėta | bazė: `gap_policy < 1 ms` — 6 iš 20 krenta | §3 |
| G4 | nepradėta | | |
| G5 | nepradėta | | |
| K1 | nepradėta | bazė: `ctest` 482 / 483; funkcinis 351 / 352. Po 1 fazės (`dad306b`): `ctest` 510 / 510 ir 509 / 510 (krito tik `gap_policy … < 1 ms`); visas funkcinis rinkinys kaip root — 353 iš 353, vienas paleidimas (kriterijui reikia trijų iš eilės, 6.3) | §3; `docs/rezultatai/2026-11-17-production-ready-faze1.md` |
| K2 | vyksta | `compute-sanitizer`: `memcheck` 5 GPU testų binarams ir rašytojo / skaitytojo porai, `racecheck`, `synccheck` — 0 klaidų, 3 paleidimai iš eilės (9 eilutės `PASS`). Rado MXL `cuda-linear` įskiepio lenktynes (asinchroninis `cudaMemset`): pataisyta pas mus, juodraštis `upstream/U7`. ASan + UBSan ir TSan — dar ne | `scripts/linux/50-sanitizers.sh cs`; `docs/rezultatai/2026-11-17-production-ready-faze2.md` §2.3 |
| K3 | PASS | bazė 480 radinių 24 tikrinimuose; 5 tikrinimai išjungti su priežastimi `.clang-tidy` faile (205 radiniai), likę ištaisyti kode; galutinis — 0 radinių, 62 vertimo vienetai, `NOLINT` nėra. Po taisymų `ctest` 534 / 534, funkciniai 89 / 89 | `scripts/linux/55-clang-tidy.sh`; `docs/rezultatai/2026-11-17-production-ready-faze2.md` §2.4 |
| K4 | PASS | 7 klaidų kelių testai (10 atvejų) praeina kaip root ir ne root; nuo gedimo iki pabaigos 0,06–1,92 s (riba 5 s); domenas po gedimo švarus; kaimyno pora — 0 praradimų ir 0 vėlavimų po gedimo. VRAM išsekimas hog'u — `SKIP` ×3: WSL2 jo nesukelia (išmatuota), tą patį kelią dengia `fault_alloc_fail`; tikras išsekimas — NEPATIKRINAMA ČIA. Kaimyno vėlavimai startuojant kitam GPU procesui (6 per 34 poras) — ne gedimo pasekmė, perkelta į G3 | `tests/functional/faults.py`; `docs/rezultatai/2026-11-17-production-ready-faze2.md` §2.5 |
| K5 | nepradėta | | |
| K6 | nepradėta | esami etalono testai žali | `ctest` |
| K7 | nepradėta | | |
| K8 | vyksta | 1 fazė: nepriklausoma peržiūra — 9 radinių grupės, pataisyta arba dokumentuota | `docs/rezultatai/2026-11-17-production-ready-faze1.md` §1.7 |
| S1 | PASS | grėsmių modelis: 17 grėsmių, kiekviena su testu arba „nepriimta, nes …“; `SECURITY.md` | `docs/architektura/2026-11-17-gresmiu-modelis.md`; `SECURITY.md` |
| S2 | PASS | 4 taikiniai po 600 s be radinių (203,6 mln. / 23,7 mln. / 2,5 mln. / 7,0 mln. vykdymų); vienas radinys trumpame paleidime pataisytas ir įdėtas į korpusą | `scripts/linux/56-fuzz.sh`; `ctest fuzz_replay_*`; `docs/rezultatai/2026-11-17-production-ready-faze1.md` §1.5 |
| S3 | PASS | skaitymo kelias (host ir `cuda-linear`): 4 piktybiško rašytojo atvejai, prieš pataisą krito 3; GPU pusė: `compute-sanitizer memcheck` visiems branduolių testams ir flow'ų skaitymo testams (tarp jų piktybiško rašytojo) — 0 klaidų | `tests/gpu/test_hostile_writer.cpp`; `docs/rezultatai/2026-11-17-production-ready-faze1.md` §1.4; `docs/rezultatai/2026-11-17-production-ready-faze2.md` §2.3 |
| S4 | PASS | `[safe_files]` 16 atvejų, 305 tvirtinimai (po peržiūros: prieiga pagal rašymo teisę į domeną, riboti užraktai, šalinimas po užraktu); `fstart_*`, `overload_*` 28 iš 28 kaip root ir ne root | `tests/unit/test_safe_files.cpp`; `docs/rezultatai/2026-11-17-production-ready-faze1.md` §1.2 |
| S5 | PASS | `mxltest`: build 0 įspėjimų, `ctest` 503 / 503, funkciniai 353 — 351 PASS / INFO; 2 FAIL (`fgold_dec`) buvo skripto klaida, po pataisos praeina (28 iš 28 pakartotų). Po peržiūros pataisų visas rinkinys ne root nekartotas | `scripts/linux/45-nonroot.sh`; `docs/rezultatai/2026-11-17-production-ready-faze1.md` §1.3 |
| S6 | PASS | 104 vertimo vienetai ir 22 ELF failai tinka; kaina: CPU +0,4 %, kelias ir vėlinimas nepakito | `ctest build_hardening`; `docs/rezultatai/2026-11-17-production-ready-faze1.md` §1.1 |
| S7 | PASS | eilių ribų ir sargo testai jau buvo (4 + 2 vieneto atvejai, funkciniai `--max-queue-mb` ir `ingest_libav_watchdog`); saugus paleidimas — `SECURITY.md` | `docs/rezultatai/2026-11-17-production-ready-faze1.md` §1.6 |
| S8 | nepradėta | | |
| S9 | PASS | injektuota CUDA klaida tick'e (`gen`, `logo`): „CUDA: unknown error“, exit 2 per 0,07–0,11 s, išėjimai atlaisvinti; NVENC sesijos atsisakymas: exit 2 per 1,8–1,9 s su tvarkyklės eilute ir paaiškinimu, veikiantis `enc` — be praradimų. Tikras GPU dingimas — NEPATIKRINAMA ČIA | `fault_cuda_midrun`, `fault_nvenc_sessions`; `docs/rezultatai/2026-11-17-production-ready-faze2.md` §2.5 |
| U1 | nepradėta | | |
| U2 | nepradėta | | |
| U3 | nepradėta | Docker WSL integracija išjungta | §3 |

## 9. Kas šiame plane nepatikrinta vykdymu

- Ar `clang++-18` kompiliuotas kodas susiejamas su `clang++-19` kompiliuota MXL ir ar ASan build'as su `MXLV_BUILD_FUNC=ON`
  iš viso susirenka (2.1).
- FFmpeg `h264_cuvid` numatytasis paviršių skaičius ir ar jo mažinimas tikrai mažina VRAM (4.3).
- Ar U1–U3 pataisos taikosi PR #653 viršūnei ir ar VRAMFlow su ja kompiliuojasi (5.2).
- Ar `cuda-linear` fd perdavimas veikia tarp konteinerių (5.4).
- Kiek užtrunka `clang-tidy` visam `src/` ir kiek radinių duoda pasirinktas rinkinys (2.4).
- Visi natyvaus Linux skaičiai: šis planas vykdomas WSL2; natyvus matavimas — lauko bandymas (6.1).
- Tikras VRAM išsekimas (2.5): WSL2 kitam procesui duoda daugiau VRAM, nei `cudaMemGetInfo` rodo laisvos, todėl
  `fault_vram_exhausted[gen|logo|dec]` čia visada `SKIP` ir dar nė karto nebuvo žali; „scratch“ / „staging“ skyrimo
  klaidų tekstai įvykdyti tik vienetų testu. Tikrinti natyviame Linux (6.1).

Patikrinta 1 fazėje ir iš sąrašo išbraukta (`docs/rezultatai/2026-11-17-production-ready-faze1.md`):

- grūdinimo vėliavos su `-Werror`: švarus build — 0 įspėjimų C, C++ (`clang++-19`) ir CUDA host daliai
  (`nvcc -Xcompiler`); `ctest build_hardening` — 104 vertimo vienetai ir 22 ELF failai (1.1);
- visas funkcinis rinkinys vartotojo `mxltest` vardu: 353 rezultatai, root neprireikė nė vienam testui (1.3, ant
  `b8168d6`). Po peržiūros pataisų ne root pakartoti tik 28 testai, einantys per pakeistą kodą; kaip root visas
  rinkinys pakartotas ant `dad306b`: 353 rezultatai, 0 FAIL.
