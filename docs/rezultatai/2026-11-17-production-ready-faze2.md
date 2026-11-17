# „Production ready“ — 2 fazė: tikrinimo gylis

Planas: `docs/planai/2026-11-17-production-ready.md`, paketai 2.1–2.5. Aplinka: RTX 4070 Laptop 8 GB, WSL2 (Ubuntu 24.04),
tvarkyklė 596.49, CUDA 13.2, MXL PR #653 ties `6dddc91d`. Bazė — `a3220eb` (1 fazės pabaiga).

Paketų tvarka pakeista: 2.5 → 2.4 → 2.1 / 2.2 / 2.3. Planas sanitizerius rašė pirmus, bet po kodo keitimų juos tektų
kartoti, o 2.1 scenarijų sąrašui reikia 2.5 testų (plano RF-3).

## 2.5 Klaidų kelių testai (K4, S9)

Testai — `tests/functional/faults.py` (registruoti `functional.py`). Kiekvienas sukelia vieną gedimą ir tikrina keturis
dalykus: sugedęs procesas baigiasi exit 2 ir santraukos `fatal` lauke parašyta priežastis; jis baigiasi per 5 s; domeno
turinys po jo toks pat kaip prieš jį; tame pačiame domene veikianti pora `gen → verify --mode gen` nepraranda nė vieno
grain'o.

### Pirmas paleidimas — prieš kodo pakeitimus

| Testas | Rezultatas | Ką parodė |
|---|---|---|
| `fault_cuda_midrun[gen]`, `[logo]` | FAIL, exit 109 | parinkties `--test-cuda-fail-after` nėra — laukta |
| `fault_shm_full` | FAIL | exit 2 per < 1 s ir švarus domenas jau buvo; `fatal` — tik „video 0: mxlCreateFlowWriter status 1“ |
| `fault_nvenc_sessions` | FAIL | exit 2; `fatal` — „srautas 0: video avcodec_open2: Invalid argument“; tikroji priežastis tik libav žurnale: „OpenEncodeSessionEx failed: incompatible client key (21)“ |
| `fault_vram_exhausted[gen]`, `[logo]` | FAIL, exit 0 | procesas **nesugedo**: gavo visus žiedus, nors laisvos VRAM tiek nebuvo |
| `fault_vram_exhausted[dec]` | FAIL | testo klaida: 16 išėjimų ir vienas siuntėjas — exit 2 po 20 s „srautas 1: įėjimas neatsirado“ |

### VRAM išsekimo WSL2 sukelti nepavyksta

Plano prielaida „`hog` palieka mažiau VRAM, nei reikia → kūrimas nepavyksta“ šioje platformoje neteisinga. Matavimas:
`vramflow-verify hog --leave-free-mb 300` užėmė 6 720 MiB ir pranešė, kad laisva 334 MiB. Po to `gen --flows 16` gavo visus
16 žiedų ir 14 s dirbo be praradimų (700 laukų kiekvienam, `any_loss` false). „Windows“ skaitikliai tuo metu: dedicated
6 862 → 7 440 MiB, shared 84 → 158 MiB. T. y. kitam procesui WDDM duoda daugiau, nei `cudaMemGetInfo` rodo laisvos, o
perteklių perkelia į bendrą sistemos atmintį.

Ką tai keičia:

- `fault_vram_exhausted[gen|logo|dec]` paliktas pagal planą, bet kai sugesti turėjęs procesas sukuria visus savo flow'us,
  rezultatas — `SKIP` su skaičiais. Čia taip nutinka visada; tikras variantas lieka natyviam Linux (lauko bandymas).
  **Šie trys testai dar nė karto nebuvo žali — jų teisingumas ant platformos, kur gedimas įvyksta, nepatikrintas.**
- Pridėtas plane nebuvęs `fault_alloc_fail[gen|logo|dec]`: paslėpta parinktis `--test-create-fail 2` „sugadina“ trečio iš
  keturių išėjimų saugyklos skyrimą. Gedimas dirbtinis, bet funkcijos reakcija — jau sukurtų išėjimų atlaisvinimas, exit
  kodas ir pranešimas — tikra. Dalinio kūrimo valymo iki šiol nedengė nė vienas testas.
- Tikro CUDA „out of memory“ per vieną milžinišką užklausą (pvz. 64 GiB žiedas) nebandžiau: host'e užimta 62 iš 128 GiB
  commit ribos, WDDM elgsena su tokia užklausa nežinoma, o tuo pačiu kompiuteriu dirbama nuotoliniu būdu.

### Kodo pakeitimai

- `src/runtime/fail_text.{hpp,cpp}` — gryna teksto logika (be CUDA, MXL, libav). Taisyklė: skelbiami tik išmatuoti
  faktai, išvada — tik kai ji iš jų seka. Vienetų testai `tests/unit/test_fail_text.cpp`: ant tuščios realizacijos krito
  8 atvejai iš 9, po realizacijos — 19 tvirtinimų 9 atvejuose.
- `create_output` (`func_common.cpp`): MXL klaidai (ne „flow jau yra“) priežastis išmatuojama — ar domeno katalogas
  rašomas (`access`), kiek laisvos VRAM (`cudaMemGetInfo`) arba vietos domene (`statvfs`), kiek reikia žiedui (MXL
  formulė: istorija × grain'ų dažnis).
- `enc.cpp`: paskutinė libav klaidos eilutė gijoje įsimenama ir pacituojama pranešime; NVENC kodas 21 paaiškinamas kaip
  vienalaikių sesijų riba (`nvEncodeAPI.h`: „feature … not available for the license type“; čia — 13-a sesija).
- Radinys pakeliui: `enc` santraukos `streams[].error` buvo rašomas `%s` į 512 B buferį be JSON ekranavimo — ilgesnis ar
  kabutes turintis tekstas būtų sugadinęs santrauką. Dabar — `json_escape_into`; testas tikrina ir šį lauką.
- `gen`, `logo`, `enc`: „scratch“ / „staging“ klaidos dabar turi CUDA klaidos tekstą ir laisvą VRAM. **Šių vietų čia
  įvykdyti negalima** (tikro išsekimo nėra) — patikrinta tik teksto funkcija vienetų testu.
- Paslėptos parinktys: `--test-cuda-fail-after S` (`gen`, `logo`), `--test-create-fail K` (visos funkcijos). `--help` jų
  nerodo.

### Rezultatas po pakeitimų

| Testas | root | ne root (`mxltest`) | Laikas iki pabaigos | `fatal` |
|---|---|---|---|---|
| `fault_vram_exhausted[gen\|logo\|dec]` | SKIP ×3 | SKIP ×3 | — | 24 flow'ai (≈ 648 MiB) sukurti per 0,4 s, kai laisva 334 MiB |
| `fault_alloc_fail[gen]` | PASS | PASS | 0,27–0,32 s | „video 2: mxlCreateFlowWriter status 1 (--test-create-fail: dirbtinė klaida); saugykla cuda-linear: VRAM laisva 6998 MiB iš 8187 MiB, žiedui reikia 27 MiB“ |
| `fault_alloc_fail[logo]`, `[dec]` | PASS | PASS | 0,27–0,32 s | „out 2: …“ — tas pats |
| `fault_nvenc_sessions` | PASS | PASS | 1,82–1,92 s | „srautas 0: video avcodec_open2: Invalid argument (OpenEncodeSessionEx failed: incompatible client key (21): (no details)) — NVENC sesijos negauta: tvarkyklės kodas 21 reiškia, kad viršyta vienalaikių NVENC sesijų riba …“ |
| `fault_cuda_midrun[gen]`, `[logo]` | PASS | PASS | 0,07–0,11 s nuo injekcijos | „CUDA: unknown error“ |
| `fault_shm_full` | PASS | SKIP (reikia `mount`) | 0,06–0,12 s | „video 0: mxlCreateFlowWriter status 1; saugykla host: domene laisva 15 MiB iš 16 MiB, žiedui reikia 27 MiB — domene nėra vietos (ENOSPC, no space left on device)“ |
| `fault_domain_readonly` | SKIP (root apeina teises) | PASS | 0,32 s | „video 0: mxlCreateFlowWriter status 10; domeno katalogas nerašomas (EACCES, permission denied)“ |

`fault_nvenc_sessions`: pirmas `enc --flows 12` tuo metu dirbo be praradimų (exit 0, `any_loss` false, 0 kodavimo klaidų).
`fault_domain_readonly` prieš pakeitimą atskirai nepaleistas (pirmas paleidimas buvo tik kaip root, o jam šis testas —
`SKIP`); kad be naujo teksto jis kristų, matyti iš to paties tvirtinimo `fault_shm_full` teste.

### Kaimyno vėlavimai — ne gedimo, o procesų starto pasekmė

Planas kaimynui reikalavo `late == 0`. Du paleidimai iš trijų dėl to krito, bet ne dėl gedimo:

- kaimyno `gen` pavėlavo 23,1 ms (riba 8 ms) praėjus 0,26 s nuo jo paties starto — kai šalia startavo jo `verify` —
  ir 7,05 s **prieš** injekciją;
- kitą kartą — 22,1 ms, praėjus 178 ms nuo dar sveikai startuojančio `gen`, 5 s prieš injekciją.

Atskiras matavimas: 20 `gen` startų (CUDA kontekstas ir vienas flow) šalia vieno veikiančio `gen` — 0 vėlavimų, didžiausias
commit'o vėlinimas 7,39 ms prie 8 ms ribos. Testuose, kur šalia veikia keli GPU procesai (`gen`, `verify`, `keepwarm` ir
tikrinamas procesas), per visus penkis paleidimus: 6 vėlūs commit'ai (17,7–29,3 ms po T(i)) per 34 kaimyno poras; 5 iš 6 —
`fault_cuda_midrun` porose, kur papildomas procesas šalia veikia ilgiausiai.

Todėl testas vėlavimus skirsto pagal laiką:

- gedimas veikimo metu (`fault_cuda_midrun`): vėlavimas nuo 0,1 s prieš injekciją iki pabaigos — `FAIL`;
- gedimas paleidžiant (visi kiti): vėlavimai užrašomi rezultate, bet nevertinami — proceso startas ir gedimas sutampa, o
  sveikas startas daro tą patį.

Praradimai (praleisti indeksai, patikros klaidos, exit kodai) vertinami visada ir per visą kaimyno gyvavimą.

**Tai neišspręsta, tik atskirta nuo gedimų.** ~20 ms kaimyno sustojimas startuojant kitam GPU procesui yra tikras
poveikis kitiems procesams; jo priežastis ir dažnis — 3 fazės darbas (G3).

## 2.3 `compute-sanitizer` GPU branduoliams (K2, S3)

Etapas `cs` skripte `scripts/linux/50-sanitizers.sh`: `memcheck` penkiems GPU testų binarams, `racecheck` ir `synccheck`
atstovaujamiems testams, `memcheck` rašytojo ir skaitytojo porai atskiruose procesuose.

### Pirmas paleidimas

| Eilutė | Rezultatas |
|---|---|
| `memcheck`: `mxlv-gpu-tests`, `mxlv-media-gpu-tests`, `vramflow-gfx-gpu-tests`, `vramflow-v210a-gpu-tests` | 0 klaidų |
| `memcheck`: `vramflow-flows-tests` | **10 klaidų**, exit 86; vienas testas krito |
| `racecheck`, `synccheck`: `mxlv-gpu-tests` | 0 |
| `memcheck`: rašytojas | exit 143, suvestinės nėra |
| `memcheck`: skaitytojas | exit 3 „netikėtas layout“, 0 klaidų |

Trys skirtingos priežastys.

**1. Tikras radinys — MXL `cuda-linear` įskiepis.** Visos 10 klaidų — „Internal Sanitizer Error: The Sanitizer failed to
handle a hardware exception“ ir po jos einančios CUDA API klaidos. Įskiepis naują žiedą nulina `cudaMemset`
(`CudaLinearBackend.cpp:1277`) ir grįžta nelaukdamas; įrenginio atminčiai `cudaMemset` yra asinchroninis. Testas flow
sukuria ir po kelių milisekundžių atlaisvina — GPU darbas lieka atminčiai, kuri jau atjungiama.

| Matavimas | Prieš | Po |
|---|---|---|
| Vienas testas po `memcheck`, paleidimų su klaida | 3 iš 5 | 0 iš 6 |
| `cudaStreamQuery(0)` iškart po `mxlCreateFlowWriter` (atskira programa be mūsų kodo) | „dar vyksta“ 20 iš 20 (53 MiB žiedas) ir 20 iš 20 (527 MiB) | — |
| Naujas testas `flows: cuda-linear VideoOut::create grįžta tik baigus žiedo nulinimą` | krito 6 iš 6 | praėjo 6 iš 6 |
| Visas `vramflow-flows-tests` po `memcheck` | 10 klaidų | 0 klaidų, 3 iš 3 |

Pataisa — pas mus: `VideoOut::create` (`src/func/flows.cpp`) `cuda-linear` flow'ui laukia numatytojo srauto
(`cudaStreamSynchronize(nullptr)`). Kaina — kelios ms kuriant flow. Tas pats laukimas pridėtas testų pagalbininkui, kuris
MXL kviečia tiesiogiai (`tests/gpu/test_v210a_cuda_linear.cpp`): be jo antrame etapo paleidime klaida pasirodė ir ten.
Radinys aprašytas MXL projektui kaip juodraštis `upstream/U7-cuda-linear-async-memset/` (faktai, minimalus pavyzdys,
siūlomas pakeitimas). Niekur nesiųsta.

Ko **nepavyko** parodyti: kad dėl to prarandami duomenys. Testas iškart po sukūrimo trijose 527 MiB žiedo vietose įrašo po
64 KiB kitame (non-blocking) sraute ir perskaito — 8 paleidimuose neprarasta nė baito. Šis pavojus lieka teorinis.

**2. Skripto klaida — rašytojas.** Po skaitytojo skriptas siųsdavo `SIGTERM` pačiam `compute-sanitizer`, ir šis
baigdavosi be suvestinės. Dabar rašytojas baigia pats (`--duration`).

**3. Skripto klaida — skaitytojas.** Pora leidžiama 5 kadrų per sekundę greičiu (sanitizeris lėtina 10–100 kartų). Su
numatytąja 200 ms istorija toks flow turi vieno grain'o žiedą, o skaitytojas tokio nepriima. Dabar domenui nustatoma 2 s
istorija. Taip buvo ir senojoje skripto versijoje — ši eilutė, tikėtina, niekada nebuvo žalia.

### Rezultatas

Trys paleidimai iš eilės, kiekviename 9 eilutės `PASS`, 0 `FAIL`: `memcheck` — 5 binarai, rašytojas ir skaitytojas po
0 klaidų; `racecheck` — 0 pavojų; `synccheck` — 0 klaidų. `initcheck` nevykdomas: jis nepalaiko IPC / VMM alokacijų
(NVIDIA „Known Issues“).

Kraštiniai rastrai GPU testuose jau buvo, naujų nepridėta: `media` — 1280×720 (plotis ne 48 kartotinis) ir 3840×2160;
`gfx` — pločiai 1920, 1280, 382 ir 94, rastrai 382×32 ir 382×64; `verify` branduoliai — ilgis 1 000 003 (nesidalija iš
256). Tikslaus 382×40 rastro nėra — yra to paties pločio 382×32 ir 382×64.

Vienas dalykas paliktas: `vramflow-verify writer` flow kuria per MXL tiesiogiai ir nulinimo nelaukia. Jo eilutė po
`memcheck` švari (3 iš 3), todėl nekeista — įrašyta kaip smulkmena.

## 2.4 `clang-tidy` (K3)

Konfigūracija — `.clang-tidy`, paleidimas — `scripts/linux/55-clang-tidy.sh` (visi `src/` `.cpp` ir `.c` vertimo vienetai;
`.cu` failų `clang-tidy` netikrina, nes juos kompiliuoja `nvcc`). Rinkinys: `bugprone-*`, `cert-*`, `concurrency-*`,
`performance-*`, `clang-analyzer-*`.

### Bazė — prieš bet kokius taisymus

62 vertimo vienetai, 122 s, **480 unikalių radinių** 24 tikrinimuose, kompiliavimo klaidų 0.

| Tikrinimas | Radinių | Sprendimas |
|---|---|---|
| `cert-err33-c` (nepatikrinta grąžinama reikšmė) | 152 | sukonfigūruota ir taisoma — žr. žemiau |
| `bugprone-easily-swappable-parameters` | 97 | **išjungta**: žymi kiekvieną (w, h), (x, y), (num, den) porą; ištaisoma tik stipriais tipais visose sąsajose |
| `bugprone-implicit-widening-of-multiplication-result` | 84 | **išjungta**: visi 84 — indeksai eilutės viduje (`16 * g`, `3 * i`); rastras ribotas formato patikra (≤ 7680 × 4320), eilutė × pitch visur jau `size_t` |
| `concurrency-mt-unsafe` | 26 | funkcijų rinkinys `glibc` (tikslinė platforma): lieka 19 `strerror` — taisoma; `getenv` (5) ir `readdir` (3) glibc dokumentuoja kaip saugius mūsų naudojimui |
| `bugprone-narrowing-conversions` | 23 | taisoma (aiškūs `static_cast<double>`) |
| `performance-enum-size` | 20 | **išjungta**: viešos C antraštės enum'ų keisti negalima (ABI), kitur naudos nėra |
| `bugprone-assignment-in-if-condition` | 18 | taisoma (`if (e = f(); e < 0)`) |
| `cert-err34-c` (`atoi`, `sscanf` be klaidų) | 9 | taisoma |
| `performance-inefficient-vector-operation` | 8 | taisoma (`reserve`) |
| `clang-analyzer-unix.Stream` | 7 | taisoma |
| `bugprone-multi-level-implicit-pointer-conversion` | 6 | taisoma (aiškūs `static_cast<void*>`) |
| `bugprone-misplaced-widening-cast` | 5 | taisoma |
| `bugprone-branch-clone` | 4 | taisoma (vienodos šakos sujungtos, komentarai palikti) |
| `bugprone-incorrect-roundings` | 3 | taisoma (`llround`) |
| `bugprone-empty-catch` | 3 | palikta su komentaru „C ABI:“ — tuščias `catch` leidžiamas tik C sąsajos riboje |
| `clang-analyzer-optin.performance.Padding` | 3 | **išjungta**: laukų pertvarkymas dėl užpildo; objektų po vieną procese |
| kiti aštuoni tikrinimai (po 1–2) | 12 | taisoma |

Išjungti keturi tikrinimai — 204 radiniai iš 480. Kiekvieno priežastis parašyta `.clang-tidy` faile šalia.

`cert-err33-c`: iš 152 radinių 69 — `fprintf` ir `fputs` į stderr, t. y. diagnostika, kurios rašymo klaidos nėra kam pranešti.
Todėl iš CERT funkcijų sąrašo pašalintos penkios išvedimo funkcijos (`fprintf`, `fputc`, `fputs`, `putc`, `vfprintf`): jų
klaida lieka srauto būsenoje, o rašant į failą ji patikrinama uždarant — `fclose` ir `fflush` sąraše lieka.
Tas pats principas `fflush`: periodinis CSV išrašymas ir `--stats-file` (skaitytuvas gali būti uždarytas) rezultato
netikrina sąmoningai — tai pažymėta `(void)` ir komentaru.

Su taisymais atsirado dar vienas radinys ir penktas išjungtas tikrinimas: `cert-dcl50-cpp` žymi naują `mxlv::strf` kaip
C stiliaus kintamo argumentų skaičiaus funkciją. Ji tokia tyčia: su `[[gnu::format(printf, …)]]` kompiliatorius patikrina
kiekvieno kvietimo formatą ir argumentus, o parametrų paketo variantas tos patikros netektų.

### Taisymai

Du nauji failai su vienetų testais (pirma krentantys, tada žali; kartu 91 tvirtinimas 14 atvejų):

- `src/common/textf.hpp` — `strf` (printf formatas į `std::string` be nupjovimo), `close_checked` ir `write_file` (rašymo
  klaida pastebima ir tada, kai ji paaiškėja tik uždarant), `read_file`, `close_read`, `errno_text` (gijoms saugus
  `strerror` pakaitalas);
- `src/common/parse_num.hpp` — `take_num`, `take_char` (griežtas skaičių skaitymas: perpildymas — klaida) ir
  `parse_ref_crc`.

| Grupė | Vietų | Kas padaryta |
|---|---|---|
| `snprintf` į fiksuotą buferį | 41 | → `strf`. Tikra rizika buvo trijose vietose: flow aprašas (2 048 B buferis su `%s` žymoms), variklio įvykio tekstas (160 B su klaidos tekstu — nupjovimas galėjo perkirsti UTF-8 simbolį), `vramflow-verify` flow santrauka (4 096 B) |
| `fclose`, `fwrite`, `fflush` | 26 | santraukos ir diagnostikos failai rašomi per `write_file`, CSV uždaromi per `close_checked`; nesėkmė — eilutė į stderr (anksčiau — tyla) |
| `strerror` | 19 | → `errno_text` |
| priskyrimas `if` sąlygoje | 18 | `if ((e = f()) < 0)` → `if (e = f(); e < 0)` |
| sveikasis → `double` | 23 | aiškūs `static_cast<double>` |
| `atoi`, `sscanf`, `fscanf` | 9 | → `take_num` / `parse_ref_crc` |
| `fread` ciklai | 7 | skaitymas sustoja po pirmo nepilno bloko (`read_file`); ciklų sąlygoje — pastovus dydis |
| kita | 42 | `reserve` (8), aiškūs rodyklių keitimai (6), daugyba prieš platinimą (5), vienodos `switch` šakos (4), negyvi priskyrimai (4), `clock_gettime` / `signal` / `gmtime_r` rezultatai (4), apvalinimas `llround` (3), kita (8) |

Sąmoningi elgsenos pokyčiai (visi — griežtesnė patikra arba matomesnė klaida):

- `--layer-geom WxH` nebepriima teksto po skaičių (anksčiau „1920x1080abc“ tiko);
- `--layer-xy`, `--layer-absent`, `--layer-either`: „+“ ar tarpas prieš skaičių ir netelpantis skaičius — klaida;
- nepavykęs santraukos, CSV ar diagnostikos failo rašymas — eilutė į stderr;
- išimtis, pasiekusi `main`, — exit 2 su pranešimu (anksčiau — `SIGABRT` be pranešimo);
- `options.json` skaitymo klaida — numatytoji 200 ms istorija (anksčiau buvo naudojama perskaityta dalis).

### Rezultatas

| Paleidimas | Radinių |
|---|---|
| bazė | 480 |
| po konfigūracijos ir pirmo taisymų paketo | 152 |
| po visų taisymų | 5 |
| galutinis | **0** (62 vertimo vienetai, 96 s, exit 0) |

Build — 0 įspėjimų; po taisymų `ctest` 534 iš 534 (137 s), 89 funkciniai testai per pakeistą kodą — visi praėjo. `NOLINT` kode nėra nė vieno. Etapas pridėtas į `scripts/linux/run-all.sh` (`55-clang-tidy`).
