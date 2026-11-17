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

## 1.4 Piktybiškas rašytojas (kriterijus S3, host ir `cuda-linear` skaitymo kelias)

**Problema.** MXL priima bet kokią grain'o antraštę: rašytojas gali commit'inti `grainSize` 0, 4 294 967 295 ar
`validSlices` > `totalSlices` — commit grąžina OK. `VideoIn::get` tikrino tik slot ribą ir indeksą, todėl už tokią antraštę
grąžindavo rodyklę kaip už sveiką grain'ą. Flow aprašo sveikieji buvo skaitomi atlaidžiai: `"frame_width":1e9` virsdavo 1.

**Kas padaryta.**

- `VideoIn::get` (`src/func/flows.cpp`) — vienintelė vieta, per kurią funkcijos skaito grain'us: galiojančio grain'o
  `grainSize` turi būti šio rastro, `totalSlices` ≠ 0 ir `validSlices == totalSlices`; kitaip `MXL_ERR_INTERNAL` ir jokios
  rodyklės. INVALID grain'o (spragos) antraštė nevertinama — jo turinio niekas neskaito.
- `json_int` (`src/gfx/raster.hpp`): tik skaitmenys, po jų — JSON skyriklis; `1e9`, `1920.5`, `"1920"`, `-1`, 65537 atmetami.
  Klaidos tekstas dabar sako „nėra tinkamo frame_width (sveikasis 1…65536)“.

**Testai** `tests/gpu/test_hostile_writer.cpp` (4 atvejai, `vramflow-flows-tests`; rašytojas — tiesiai per MXL API):

| Atvejis | Prieš pataisą | Po |
|---|---|---|
| Netikra antraštė, host: `grainSize` ∈ {0, tikras ± 1, 2³²−1}; pjūviai {0/0, 0/7, 540/541, 540/65535} | FAIL: `get` grąžina OK ir rodyklę | PASS |
| Tas pats, `cuda-linear` | FAIL | PASS |
| Flow aprašas (1920×1080) didesnis už tikrą saugyklą (1280×720) | PASS — `check_layout` jau saugojo | PASS |
| Neįmanomi matmenys apraše (0, −1, 65537, 2³¹−1, 10²⁰, `1e9`, eilutė, tuščia) | FAIL (`1e9` priimta kaip 1) | PASS |

Iš viso prieš pataisą krito 3 atvejai iš 4 (33 tvirtinimai iš 107); po — visi 107 tvirtinimai praeina. Po sveikos antraštės
skaitytojas tebeskaito (testas tai tikrina po kiekvienos netikrų antraščių serijos).

Švarus build — 0 įspėjimų; funkciniai testai, einantys per skaitymo kelią (`f7_teeth`, `fgfx1`, `fcomp1–3`, `ffmt_logo`,
`ffmt_logo720`, `ffmt_comp`, `ffmt_enc`, `ffmt_dec`, `fstart_restart`, `fstart_attach`): 49 iš 49 PASS.

Kas lieka: GPU branduolių ribų patikra `compute-sanitizer` (planas 2.3) — S3 GPU pusė iki tol remiasi šiais testais.

### 1.2 papildymas: grupei bendras domenas

Peržiūrint savo 1.2 kodą rasta spraga: kai domeną dalijasi skirtingi vartotojai per vieną grupę (pvz. konteineriai su bendru
`fsGroup`), naujas nuomos katalogas gaudavo kūrėjo pirminę grupę, todėl kitas vartotojas jo nepriimdavo. Dabar naujas
katalogas gauna domeno grupę ir `setgid` bitą (failai jame paveldi grupę). Testas `safe_files: grupei bendras domenas …`
(tik root: vaiko procesas kito vartotojo ir grupės teisėmis, abi tvarkos — katalogą pirmas sukuria root arba kitas
vartotojas): be pataisos krenta (katalogo grupė 0 arba 54323 vietoje 54321, antras vartotojas nuomos negauna), su pataisa —
`[safe_files]` 10 atvejų, 149 tvirtinimai.

## 1.5 Fuzz testai parseriams (kriterijus S2, peržiūros fokusas RF-4)

**Kas padaryta.** Keturi taikiniai `tests/fuzz/fuzz_<taikinys>.cpp` (gryna logika, be MXL, CUDA ir libav), kiekvienas su
invariantais, ne tik „nelūžta“:

| Taikinys | Ką fuzzina | Invariantai, kai įėjimas priimtas |
|---|---|---|
| `flow_def` | `format_from_flow_def` | matmenys 1…65536; interlaced aukštis lyginis; `grain_bytes` ne didesnis už 65536 × 65536 `v210a` (64 bitai) |
| `capacity` | `parse_capacity` (ranka rašytas JSON skaitytuvas) | visi skaičiai baigtiniai ir neneigiami; atmetimas — su priežastimi |
| `playlist` | `parse_playlist`, `check_playlist`, `resolve_playlist_url`, libav atminties sąmatos | ribos neviršytos; išspręsta nuoroda — tik `http` / `https`; klaidos tekste nėra bazinio URL paslapčių |
| `url` | `source_url_allowed`, `redact_url`, `canonical_authority` ir kt. | leista schema — iš dokumentuoto sąrašo; išvalytame URL nėra paslapties žymės; libav grojaraščio URL ≤ 4095 B |

Iš to paties šaltinio daromi du vykdomieji: `mxlv-fuzz-<taikinys>` (libFuzzer + ASan + UBSan, `MXLV_BUILD_FUZZ=ON`,
`clang++-18`; `scripts/linux/56-fuzz.sh`) ir `mxlv-fuzz-replay-<taikinys>` gamybiniame build'e — `ctest
fuzz_replay_<taikinys>` paleidžia korpusą `tests/fuzz/corpus/<taikinys>/`. Radinys įdedamas į korpusą ir lieka testu.

**Radinys.** Trumpame (15 s) paleidime `capacity` taikinys rado: `"knee": 2232340777` netelpa į `int` ir tyliai virsta
neigiamu skaičiumi. Ta pati klaida galiojo ir kitiems laukams: `"sessions": 3000000000` duotų neigiamą NVENC sesijų ribą, o
neigiama riba reiškia „be ribos“ — sugadintas talpos failas būtų tyliai išjungęs sesijų apsaugą. Pataisa
(`src/runtime/admission.cpp`): sveikieji laukai turi būti sveikieji leistinose ribose (sesijos ir kelias ≤ 10⁶, baitai ≤ 2⁵³),
visi skaičiai — baigtiniai; kitaip klaida su lauko vardu. Testas `capacity: skaičius už lauko ribų atmetamas, ne apkarpomas`
(prieš pataisą krito 11 iš 27 tvirtinimų) ir fuzzerio įėjimas korpuse (`crash-knee-overflow.json`; prieš pataisą
`fuzz_replay_capacity` baigdavosi `abort`).

**Paleidimas po pataisos** (`FUZZ_SECONDS=600`, po vieną branduolį):

| Taikinys | Vykdymų per 600 s | Radinių |
|---|---:|---:|
| `flow_def` | 203 584 315 | 0 |
| `capacity` | 23 692 361 | 0 |
| `playlist` | 2 543 923 | 0 |
| `url` | 6 999 219 | 0 |

**Nukrypimas nuo užduoties.** CLI specifikacijų parseriams fuzz taikinių nėra (planas §2): tai operatoriaus įvestis ir
`static` funkcijos; neigiami CLI testai jiems — paketas 2.5 kartu su klaidų keliais. Flow aprašo invariantas — 65536, ne
7680 × 4320: parseris palaikomo dydžio netikrina (grafikos sluoksniai gali būti bet kokio dydžio), tai daro
`check_layout` ir `format_supported`.

Švarus build — 0 įspėjimų; `ctest` 502 / 503 (krito tik žinomas laiko testas `gap_policy … < 1 ms`).

## 1.3 Paleidimas ne root (kriterijus S5)

**Kas padaryta.** `scripts/linux/45-nonroot.sh <vartotojas> [build | ctest | functional | all]` — tas pats build, `ctest` ir
funkcinis rinkinys kito vartotojo vardu (`setpriv`). Vartotojo kataloge lieka tik jo paties build ir testų išvestis;
žurnalus root rašo į savo katalogą. Testų harness'e nebeliko įrašytų `/root/…` kelių (`functional.py`, `chain.py`,
`interop.py`: MXL įrankiai imami šalia MXL bibliotekos).

**Rezultatai** (vartotojas `mxltest`, uid 1000; WSL2, GPU per `/dev/dxg`):

| Paleidimas | Build | `ctest` | Funkciniai |
|---|---|---|---|
| Visas rinkinys ant `b8168d6` | 0 įspėjimų | 503 / 503 | 353 rezultatai: 319 PASS, 34 INFO, 2 FAIL, 0 SKIP |
| Po skripto ir peržiūros pataisų (§1.7) | 0 įspėjimų | 508 / 509 (tik žinomas `gap_policy … < 1 ms`) | 28 iš 28 (`fstart_*`, `overload_*`, `fcomp1`, `fgold_dec`) |

Abu FAIL pirmame paleidime — `fgold_dec[C]` ir `fgold_dec[G]`: ne teisių, o mano skripto klaida (neperdavė klipo kelio
`FGOLD_CLIP` / `FGOLD_REF`, testas klipo ieškojo vartotojo namų kataloge). Su pataisytu skriptu abu praeina. Visi kiti testai
— tie patys PASS kaip root; nė vienam funkciniam testui root neprireikė. Vieneto testai, kuriems reikia kito vartotojo
imitacijos (`safe_files: …` su `chown` / `setuid`), ne root vartotojui praleidžiami su priežastimi (SKIP).

**Kas nepatikrinta.** Po peržiūros pataisų visas funkcinis rinkinys ne root vartotoju pakartotas nebuvo — tik 28 testai,
einantys per pakeistą kodą. Natyviame Linux (ne WSL2) ne root paleidimas nebandytas.

## 1.6 Grėsmių modelis ir `SECURITY.md` (kriterijai S1, S7)

`docs/architektura/2026-11-17-gresmiu-modelis.md`: turtas (4), pasitikėjimo ribos (5), 17 grėsmių su priemone, testu ir
likutine rizika, skyrius „nepriimta, ir kodėl“ (sandbox aplink libav, TLS tikrinimas, turinio parašai, knyga visai GPU).
`SECURITY.md` (angliškai): kaip pranešti, pasitikėjimo modelis, saugus paleidimas (ne root, domeno ir jo tėvinio katalogo
teisės, systemd, konteineriai), kas tikrinama automatiškai.

S7 (tinklo įėjimų ribos ir sargas): naujo kodo neprireikė, testai jau yra — eilių ribos kiekiu ir baitais
(`tests/unit/test_packet_queue.cpp`, 4 atvejai; funkcinis testas su `--max-queue-mb` tikrina `queue_drops` ir RSS), libav sargas
(`tests/unit/test_libav_watchdog.cpp`, 2 atvejai; `ingest_libav_watchdog[open | sigterm | segment]` — exit 7). Visi jie praėjo ir
ne root paleidime.

**Reikia autoriaus veiksmo.** `SECURITY.md` nurodo pranešti per GitHub privatų pranešimą („Report a vulnerability“) — jį
reikia įjungti repo nustatymuose.

## 1.7 Nepriklausoma peržiūra ir pataisos (kriterijus K8)

Fazės saugumo kodą parašiau pats, todėl paprašiau atskiro agento jį peržiūrėti priešiškai (tik skaitymas). Kiekvieną radinį
patikrinau kode prieš taisydamas.

| # | Radinys | Būsena |
|---|---|---|
| 1 | `flock` užtenka skaitymo aprašo: `0755` domenui nuomos failai buvo skaitomi visiems, todėl pašalietis galėjo pakabinti visus domeno procesus (užraktas ant `admission.lock`) arba laikyti mirusio proceso nuomą „gyvą“ | Pataisyta: prieiga prie nuomos katalogo — tik tiems, kam domenas rašomas (`0755` → `0700` / `0600`); užrakto laukiama ne ilgiau kaip 5 s, negavus — rezervacija atmetama |
| 2 | Konteineriuose nuomos katalogas (šalia domeno) lieka už sumontuoto domeno ribų — kiekvienas konteineris turėtų savo knygą; `SECURITY.md` patarimas buvo klaidingas | Dokumentuota: montuoti tėvinį katalogą; patikra — planas 5.4 |
| 3 | Negyvos nuomos šalinimas pagal vardą galėjo pašalinti ką tik tuo pačiu `run_id` paskelbtą gyvą nuomą | Pataisyta: skelbimas ir šalinimas — tik po katalogo užraktu `lease.lock`, šalinama tik jei vardas tebenurodo tą patį failą |
| 4 | Dingus domenui (pvz. perkuriant) esamas grupei bendras katalogas buvo susiaurinamas iki `0700`; siaurinant dingdavo `setgid`; nepavykus `fchown` grupės teisės atitekdavo ne tai grupei | Pataisyta visi trys |
| 5 | Kūrimo langas: tarp `mkdir` ir `fchmod` kitas procesas failų teises imdavo iš laikinų katalogo teisių; `reserve` / `release` / `total` netikrino, ar užraktas gautas | Pataisyta: failų teisės ir grupė — iš domeno, ne iš katalogo būsenos; be užrakto rezervacija atmetama |
| 6 | `45-nonroot.sh`: root rašė žurnalus į vartotojo katalogą (nuoroda ten leistų perrašyti bet kurį failą); su vienu argumentu skriptas neveikė | Pataisyta abu |
| 7 | `grain_rate` skaitomas atlaidžiai (`25e3`, milžiniškos reikšmės perpildytų sandaugas); kito proceso knygos failo neigiama eilutė mažino „užimta“ visiems | Pataisyta: griežtas sveikasis 1…1 000 000; neigiamos ir sugadintos eilutės neskaičiuojamos. Nepakeista: `totalSlices` nelyginamas su eilučių skaičiumi (atminties saugumui neturi įtakos) |
| 8 | Fuzz build'e UBSan pranešimas nenutraukdavo paleidimo; grūdinimo testas nematė vėlesnės panaikinančios vėliavos; `_FORTIFY_SOURCE` buvo pridedamas ir neoptimizuotam build'ui | Pataisyta visi trys. Ankstesnių fuzz paleidimų žurnaluose UBSan pranešimų nebuvo (patikrinta), tad jų rezultatai galioja |
| 9 | Testai: trys tvirtinimai negalėjo kristi; grupės testas netikrino dviejų neprivilegijuotų vartotojų; piktybiško rašytojo testas priimdavo bet kokią klaidą (ir laukimo ribą) | Pataisyta: testai sugriežtinti, pridėti trūkstami atvejai |

Likutinė rizika, kurios kodu nepašalinsi: kas pirmas sukuria `<domenas>.vramflow` kataloge, rašomame visiems (pats
`/dev/shm`), tas neleidžia startuoti (exit 2). Priemonė — domeną laikyti tėviniame kataloge, kurį rašyti gali tik jo
vartotojai (`SECURITY.md`).

**Testai po pataisų.** `[safe_files]` — 16 atvejų, 305 tvirtinimai (prieš seną kodą `b8168d6` krito 6 iš 15, o užimto užrakto
testas pakibo); visas `mxlv-admission-unit` — 40 atvejų, 613 tvirtinimų; `[hostile]` — 4 atvejai, 111 tvirtinimų, dabar
reikalaujantys būtent `MXL_ERR_INTERNAL`. Švarus build — 0 įspėjimų; `ctest` 509 / 509; funkciniai 28 iš 28 kaip root ir
28 iš 28 kaip `mxltest`.

Fuzzeriai pakartoti su nutraukiančiu UBSan (`-fno-sanitize-recover=undefined`): `flow_def` — 224 285 703 vykdymai per 600 s;
`capacity` — 4 321 715, `playlist` — 1 050 200, `url` — 3 055 160 per 120 s; radinių nėra. Dažnio skaitymo testas `format:
flow aprašo grain_rate …` prieš seną kodą krenta (`25e3`, `25.5`, `+25`, 1 000 001 ir didesni buvo priimami).

## Fazės pabaigos būsena

Paskutinė patikra ant `31997b4`: švarus build — 0 įspėjimų; `ctest` 510 / 510. Kriterijai: S1, S2, S4, S5, S6, S7 — PASS;
S3 — skaitymo kelias PASS, GPU branduolių `compute-sanitizer` lieka paketui 2.3; K8 — šios fazės peržiūra atlikta.

Po dokumentų commit'o `646dbd5` (kodas tas pats: `ninja: no work to do`) `ctest` pakartotas: 509 / 510, 136 s. Krito tik
žinomas nestabilus laiko testas `gap_policy … < 1 ms`. Jo 20 pakartojimų ant to paties build: 14 praėjo, 6 krito — tiek pat,
kiek bazėje (6 iš 20, plano §3). Kritusių reikšmės: 1,251; 1,252; 1,252; 1,253; 1,258; 1,296 ms prieš 1 ms ribą. Penkios iš
šešių telpa į 7 µs — tai stebėjimas, ne paaiškinimas; priežasties paieška — 3 fazės darbas (kriterijus G3).

Visas funkcinis rinkinys kaip root ant `dad306b` (kodas tas pats kaip `31997b4`: `ninja: no work to do`), 16:50–19:15,
2 h 25 min: 353 rezultatai — 321 PASS, 32 INFO, 0 FAIL, 0 SKIP (`functional.py <out>`, exit 0). Po jo neliko nė vieno
`vramflow` proceso ir nė vieno nuomos katalogo. Į jį įeina 25 `interop_*` rezultatai (visi PASS, tarp jų
`interop_foreign_writer`), kurių 28 testų pakartojime po peržiūros pataisų nebuvo: griežtesnis `grain_rate` skaitymas
svetimų rašytojų flow aprašų neatmeta. Bazėje (plano §3) tas pats rinkinys turėjo vieną kritimą — `fiso_stall[G:first]`;
šįkart jis praėjo, bet tai vienas paleidimas, ne įrodymas, kad nestabilumas dingo (3 fazė, G3).

Žurnale liko 4 `ConnectionResetError` pėdsakai iš testų HTTP serverio (`tests/functional/ingest.py:552`): klientas nutraukė
jungtį, kol serveris rašė atsakymą. Rezultatų tai nekeičia, bet testų išvestis nėra švari — neištaisyta, įrašyta kaip
smulkmena.

Kas šioje fazėje nepatikrinta:

- visas funkcinis rinkinys ne root vartotoju po peržiūros pataisų: jo vardu pakartoti tik 28 testai, einantys per pakeistą
  kodą; paskutinis pilnas ne root paleidimas — ant `b8168d6`;
- natyvus Linux (viskas matuota WSL2) ir konteineriai;
- `clang++-18` sanitizerių build'as visam projektui (kol kas tik fuzz taikiniams) — paketas 2.1;
- grupei bendro domeno scenarijus tikrais vartotojais ir tikra grupe (testuose — išgalvoti uid / gid per `setuid`).
