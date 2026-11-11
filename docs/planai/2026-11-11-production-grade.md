# VRAMFlow „production-grade“ programa (P1–P10) — įgyvendinimo planas

> **Būsena:** planas, **laukia autoriaus patvirtinimo**. Kodas nekeistas, niekas nekompiliuota ir nepaleista —
> visi teiginiai apie kodą paremti skaitymu (failas:eilutė), ne vykdymu. Kur reikia vykdymo — pažymėta **[nepatikrinta čia]**.
> **Vykdytojams:** superpowers:executing-plans arba subagent-driven-development; kiekvienas P — atskira šaka ir
> **nauja sesija**. P3, P4, P9 nepradedami, kol autorius nepatvirtino user stories (`docs/user-stories/2026-11-11-P*.md`).

**Brief'as:** `docs/reikalavimai/2026-11-11-production-grade.md` (toliau **[B]**; pamokos **L1–L19**, darbai **P1–P10**).

**Tikslas:** VRAMFlow iš „1080i50 sintetinių testų įrankio“ tampa biblioteka ir funkcijomis, kurios atlaiko tikrus
šaltinius: laikas pagal medijos PTS, bet koks rastras / kadrų dažnis, ingest su garsu, perkrova be griūties,
sutartis su MXL etaloniniais įrankiais, paleidimas bet kokia tvarka, gyva statistika, talpa pagal variklį,
GPU multivieweris. Viskas nepriklauso nuo konkrečios mašinos ir patikrinama ant bet kurios NVIDIA GPU.

**Architektūros kryptis:**

1. **Trys nauji bendri sluoksniai, be naujų priklausomybių:** `mxlv_timing` (PTS → TAI → indeksas; be MXL ir be libav),
   `VideoFormat` (rastras + kadrų dažnis; apibendrina esamą `gfx::Raster`), `mxlv_runtime` (įėjimų laukimas /
   re-attach, domeno nuoma, periodinė statistika, priėmimo knyga). Funkcijos (`gen`, `logo`, `composite`, `enc`,
   `dec`, `verify`, nauji `ingest`, `mv`) naudoja tik juos.
2. **Viena gija valdo MXL handle'us** (writer'iai nėra thread-safe) — esamas „tick“ variklis lieka; demux / dekodavimo
   gijos su MXL nesiliečia, jos tik deda kadrus su jau apskaičiuotu TAI laiku.
3. **Matematika vieną kartą etalone**, CPU ir CUDA prieš jį bitų tikslumu (kaip `gfx_ref` / `media_ref`).
4. **MXL viduje viskas nesuspausta** (v210 / v210a 10 bitų, float32 garsas); suspausta tik įėjimuose / išėjimuose.
5. **Matavimai „prieš / po“ yra plano dalis**, ne priedas: pirmas darbas po build pataisų — natyvus bazinis
   permatavimas (L19).

**Technologijos:** C++20, CUDA 13.2, MXL PR #653 (`6dddc91d92f5`), FFmpeg libav* (jau naudojama), Catch2 3, CLI11,
Python 3 (harness). Naujos priklausomybės — tik per „Klausimai autoriui“.

---

## 1. Bendri ribojimai (privalomi; iš [B] §3–4 ir projekto taisyklių)

- Commit'ai: Conventional Commits anglų kalba, maži; trailer'is — žr. **K1** (brief'as: `Assisted-by: <įrankis> (<modelis>)`).
  Draft PR ≤ ~400 pakeistų eilučių. **Jokio `git push` savo iniciatyva.**
- **Pirma klausti:** naujos / keičiamos priklausomybės (tikslus pin, ≥ 30 d. senumo), įspėjimų ar testų slopinimas,
  CI/CD, testų trynimas / perrašymas, bet kas destruktyvaus, kitų GPU darbų stabdymas mašinoje.
- Naujas naudotojui matomas funkcionalumas (P3, P4, P9) — pirma patvirtinta user story (LT, Given/When/Then,
  ≥ 1 teigiamas + ≥ 1 neigiamas kriterijus).
- Tik sintetinė arba vieša testinė medžiaga; jokių gamybinių sistemų ir asmens duomenų.
- MXL viduje — tik nesuspausta. Kodo komentarai lietuviškai; README / ADR angliškai. **0 kompiliatoriaus įspėjimų.**
- „Padaryta“ = 0 įspėjimų, visi testai praeina (nestabilus `382x40` — raportuojamas, kol P10 jo neištaisė; **jo
  negalima praleisti ar atlaisvinti neklausus**), nauji testai kiekvienam elgsenos pokyčiui, „prieš / po“ lentelė
  kiekvienam našumo pokyčiui, atnaujinti dokumentai, PR tekstas su Done / Verified (skaičiai) / Caveats / Next.
- MXL faktai, kuriais remiamasi: indeksas = round(t · rate) nuo TAI; interlaced grain = laukas; v210 eilutė
  `ceil(w/48)·128` B; sąveikos formatas — host atminties flow'ai; praleistas grain'as skaitomas kaip OK su
  `MXL_GRAIN_FLAG_INVALID`; reader'io timeout = `TOO_EARLY`; writer'iai ne thread-safe.
- **VRAM biudžetas (8 GB plokštė, ~6,5 GB laisva po WSL / darbalaukio):**

  | Flow | Grain'as | Ring'as (10 grain'ų) |
  |---|---|---|
  | 1080i50 v210 (laukas) | 2 764 800 B | 26,4 MiB |
  | 720p50 v210 (`27·128·720`) | 2 488 320 B | 23,7 MiB |
  | 1080p25 / p50 v210 | 5 529 600 B | 52,7 MiB |
  | UHD p50 v210 (`80·128·2160`) | 22 118 400 B | 211 MiB |
  | Canvas 3440×1440 v210 (`72·128·1440`) | 13 271 040 B | 126,6 MiB |

  Ingest srautas (1080): dekoderio paviršiai S × 3,13 MB (S = 8…25) + video ring'as + laikomi kadrai ≈ 80–135 MB
  ⇒ čia telpa ≤ ~48 srautų pagal VRAM; riba vis tiek bus NVDEC. Ring'o grain'ų skaičius priklauso nuo dažnio —
  visada imamas iš `layout.slotCount`, ne iš prielaidos.

---

## 2. Brief'as prieš kodą

Vertinimas: **taip** / **iš dalies** / **ne** / **nepatikrinama čia**.

| L | Teiginys | Kur kode | Verdiktas | Pastaba |
|---|---|---|---|---|
| 1 | `dec` atkuria atvykimo tvarka per plaukiojančią eilę | `src/func/dec.cpp:270-272` (komentaras „ATVYKIMO tvarka (ne pagal PTS)“), eilė `:165`, `:279-297`; PTS naudojamas tik spragų skaičiui `:149-155` | taip | Garso `dec` iš viso neturi; garsą rašo tik `gen` (`gen.cpp:117`). |
| 2 | Vienas demuxeris šaltiniui | `dec.cpp:122` išmeta visus ne video paketus | taip | A/V iš vieno demuxerio nėra iš principo. |
| 3 | Dekoderiui reikia laiko atsargos | `dec.cpp:118-135` — paketai į dekoderį iškart, kai atvyko; jokio DTS vartų | taip | Atsargą šiandien duoda tik `--playout-frames 6` (`func_common.hpp:72`). |
| 4 | Perkrova griūva; pavėluoti kadrai vis tiek dekoduojami | `dec.cpp:165-171`, `:293-297` — metama tik iš **dekoduotų** kadrų eilės | taip | Jokio „praleisti iki rakto kadro“. Pačios griūties čia nematavau. |
| 5 | 1080i50 įrašyta visur | `src/gfx/raster.hpp:21` (`kHd1080i`); `src/func/flows.hpp:31-32` (`kFieldRate`, `kAudioPerField`); `flows.cpp:108-111` (`check_rate` 50/1), kviečiama `:149`, `:213`; `src/media/v210.hpp:13-19`; `media_kernels.cu:257`; `dec.cpp:142`; `enc.cpp:39,80`; `logo.cpp:61,103`; `verify.cpp:90,314-316`; `gfx_writer.cpp:224` | iš dalies | `gfxgen` / `composite` / `libvramflow-gfx` jau priima bet kokį `Raster` (w, h, interlaced, alpha), bet **tik 50/1** — progresyvus = tik 50p. `gen`, `logo`, `enc`, `dec`, `verify` ir visi `media` kernel'iai — tik 1920×1080i50. |
| 5 | `split_nv12` ima lauko chromą iš kas antros 4:2:0 eilutės | `src/media/media_kernels.cu:163-178` (UV eilė `2k + parity`), etalonas `media_ref.cpp:272` | taip | Interlaced šaltiniui tai teisinga, progresyviam — klaidinga. PSNR skaičiai (50/54 → 62/66 dB) **nepatikrinami čia**. |
| 5 | Progresyvus kadras per interlaced kelią | `dec.cpp:147` — tik skaitiklis `flag_mismatch`, kadras vis tiek skaidomas kaip laukai | taip | |
| 6 | Chroma padėtis / skenavimas turi būti aiškūs per flow | Nėra jokio lauko; `Raster` neturi nei dažnio, nei chroma padėties | taip | |
| 7 | Host writer'iai nenustato `validSlices` | Nustato: `flows.cpp:171-172` (`VideoOut::commit`), `verify_writer.cpp:196,227`; `libvramflow-gfx` eina per `VideoOut` (`gfx_writer.cpp:196`) | **ne (kode)** / taip (README) | **README pavyzdys tikrai praleidžia:** `README.md:188-192` commit'ina be `info.validSlices = info.totalSlices`. Taisoma P5 PR-1. |
| 8 | Trūkstamas indeksas | Funkcijos tyčia necommit'ina: `logo.cpp:160`, `composite.cpp:274`, `dec.cpp:312-316`, `gfx_writer.cpp:257-258` (`no_frame`) | taip (elgsena) | Kad `mxl-gst-sink` dėl to stringa — **nepatikrinama čia**. Pastaba: tai kertasi su faktu „praleistas grain'as = OK + INVALID“; P5 pirmas žingsnis tai išmatuoja. |
| 9 | Niekada netestuota su MXL įrankiais kaip vartotojais | `tests/functional/functional.py:243,337` naudoja `mxl-info`; `mxl-gst-*` — niekur | iš dalies | `mxl-info` jau naudojamas. `mxl-gst-*` atmeta interlaced flow'us (`docs/architektura/2026-11-10-gpu-media-funkcijos.md:19`, G3) ⇒ sąveikos testams reikia P2. |
| 10 | Riba — fiksuotų funkcijų variklis | `tests/perf/capacity.py:51,63-70` jau matuoja `util_enc` / `util_dec` 1 Hz; README `:351` tą pačią išvadą daro apie NVENC | iš dalies (jau yra) | Trūksta: VRAM vienam flow, kopijavimo variklių, automatinio „kelio“. 3090 skaičiai **nepatikrinami čia**. |
| 11 | Nėra priėmimo kontrolės | `capacity.py:160-177` (`comp_guard`) — tik harness'o VRAM / `/dev/shm` patikra; funkcijose nieko | taip | |
| 12 | Fiksuotas 15 s laukimas → tylus išėjimas | `logo.cpp:103`, `enc.cpp:351,353`, `composite.cpp:135,164`; `verify.cpp:314,316` (20 s) | iš dalies | Laukimas tikrai įrašytas. Išėjimas **ne tylus**: `fatal` į `summary-*.json` ir stderr, exit 4. Re-attach yra tik `composite` (`composite.cpp:184-221,253-256`). |
| 13 | Vieno paleidimo valymas ištrina kito flow'us | C++ funkcijos domeno netrina (tik `mxlGarbageCollectFlows`: `gen.cpp:37`, `dec.cpp:216`, `gfxgen.cpp:126`, `composite.cpp:122`). Trina **harness'as** fiksuotais vardais: `capacity.py:266-267,387`, `functional.py:66-67`, `scale.py:71-72,91`, `examples/ograf/run-demo.sh:16,25` | iš dalies | Problema — harness'e ir fiksuotuose flow ID (`flows.cpp:16-20`; antras writer'is → `created=false` → klaida `flows.cpp:148`). |
| 14 | Statistika tik pabaigoje | `func_common.cpp:66` (`write_summary`), kviečiama tik iš `finish` (pvz. `logo.cpp:72-74`); ad-hoc `MXLV_DEC_TRACE` `dec.cpp:157-160` | taip | |
| 15 | Nėra zero-copy rodymo | Kode nėra GL / Vulkan / EGL (grep tuščias) | taip | |
| 16 | `CMAKE_CUDA_ARCHITECTURES` po `project()` | `CMakeLists.txt:4` (`project`), `:11-13` (`if(NOT DEFINED …) set(… 89)`) | taip (skaitant) | Po `project(… CUDA)` CMake kintamąjį jau būna inicializavęs kompiliatoriaus numatytuoju ⇒ `:11-13` — negyvas kodas. Tikra architektūra **[nepatikrinta čia]** — P10 pirmas žingsnis ją nuskaito iš `CMakeCache.txt`. Papildomai: įrašytas `89` neveiktų ant sm_86. |
| 17 | Kompiliatoriai įrašyti, skriptai `scripts/wsl/` | `scripts/wsl/10-build-mxl.sh:17`, `30-build-verify.sh:10-11`; WSL keliai: `env.sh:12`, `run-all.sh:34-35`, `capacity.py:44`, `scale.py:79`; `/root/…` `functional.py:22` | taip | `CC_BIN` / `CXX_BIN` čia **nėra** — tai prototipo šakos pataisa. |
| 18 | `gfx_writer: progresyvus 382x40 …` nestabilus | `tests/gpu/test_gfx_writer.cpp:260-277` | nepatikrinama čia | Testas egzistuoja; nestabilumo nemačiau (paleidimai uždrausti). Hipotezės — P10. |
| 19 | Visi skelbti skaičiai — iš nešiojamos GPU su sm_75 JIT | `README.md:340` (RTX 4070 Laptop, WSL2) | taip (jei L16 pasitvirtins) | |

**Išvada:** klaidingas šioje repo tik L7 „kodo“ dalis (pagalbininkai `validSlices` nustato; klysta README). L9, L10,
L12, L13 — iš dalies jau padaryta arba netiksliai suformuluota. Viskas, kas remiasi 3090 matavimais ir prototipu
(`~/evidence/vf-live/tools/`, šaka `feature/vramflow-readme-build-fix`), čia nepasiekiama — planas jų nenaudoja.

---

## 3. Priklausomybės ir eilė

```
P10a (build) ──► M0 (bazinis permatavimas) ──► visi „prieš“ skaičiai
P10b (382x40 priežastis) ─ lygiagrečiai
P5a (README + sutarties pagalbininkai)
P2 (formatai) ──► P3, P5b, P9
P1 (laikas)   ──► P3 ──► P4
P6 (start/stop, nuoma) ──► P3, P4 (knyga gyvena nuomos kataloge)
P7 (gyva statistika)   ──► P4, P8, P9 (aliarmai)
P8 (talpa v2) ──► P4 (išmatuota talpa priėmimui)
P5b (sąveikos testai) ← P2, P3
P10c (skriptų pervadinimas) — bet kada po P10a, atskiru PR
```

**Rekomenduojama eilė ir kodėl:**

1. **P10a** — CUDA architektūra ir kompiliatoriai. Kol tai netaisyta, kiekvienas matavimas abejotinas (L16, L19).
2. **M0** — bazinis natyvus permatavimas (logo, dec, enc, chain, composite) su esamu harness'u; tai „prieš“ stulpelis visiems.
3. **P10b** — `382x40` šakninė priežastis. Be jos „visi testai praeina“ vartai neturi prasmės.
4. **P5a** — README klaida ir `GapPolicy` pagalbininkas: maža, o visi vėlesni writer'iai jį naudoja.
5. **P2** — formatai. Didžiausias mechaninis pakeitimas; kuo vėliau, tuo daugiau kodo teks perrašyti. Reikalingas P3 (p25 šaltiniai), P5b (`mxl-gst-*` priima tik progresyvų), P9.
6. **P1** — laiko modelis. Nepriklauso nuo P2 (savi `Rational`), galima lygiagrečiai su P2.
7. **P6** — laukimas, re-attach, domeno nuoma. Reikalinga prieš ingest (tinklo šaltiniai startuoja lėtai).
8. **P7** — gyva statistika. Be jos P3 / P4 derinimas vėl reikalautų ad-hoc įrankių.
9. **P3** — ingest su garsu (reikia P1, P2, P6, P7).
10. **P8** — talpos harness v2; jo išvestis — talpos failas priėmimo kontrolei.
11. **P4** — termino vartai (galima iškart po P3) + priėmimo kontrolė (po P8).
12. **P5b** — sąveikos testai su `mxl-gst-sink`, `mxl-gst-cuda-sink`, `mxl-info`.
13. **P9** — multivieweris; paskutinis, nes naudoja viską.
14. **P10c** — skriptų pervadinimas (mechaninis; atskirai, kad nesimaišytų su elgsenos PR).

---

## 4. Peržiūros fokusas — 5 gedimai, kurių testai dar nedengia

| # | Gedimas | Kodėl tikėtinas | Testas (savininkas) |
|---|---|---|---|
| PF-1 | PTS persivertimas (2^33) tuo pačiu metu su B kadrų perrikiavimu: DTS jau persivertė, PTS dar ne (arba atvirkščiai) | Kiekvienas TS srautas tai padaro kas 26,5 h; vienas neteisingas išvyniojimas = 26,5 h šuolis | P1 `source_clock: PTS ir DTS skirtingose wrap pusėse` |
| PF-2 | Šaltinio laikrodis lėtesnis / greitesnis už TAI (±50 ppm = 4,3 s per parą): latencijos buferis išsenka per valandas | Tikri šaltiniai niekada neturi mūsų laikrodžio; 60 s testai to nemato | P1 `source_clock: dreifas +200 ppm → kadro slydimai, ne griūtis`; P3 pagreitintas funkcinis testas |
| PF-3 | Gamintojas perkraunamas su **kitu formatu** (1080i50 → 1080p25) po tuo pačiu flow ID | Šiandien `composite` tokį įėjimą uždaro visam laikui (`reopen_mismatch`, `composite.cpp:219`), kiti — neapibrėžta | P6 `input_slot: formatas pasikeitė po re-attach` |
| PF-4 | `kill -9`: lieka domeno nuoma ir priėmimo knygos rezervacija → talpa „užimta“ amžinai, domenas neišvalomas | Field teste procesai buvo žudomi; failai be `flock` semantikos lieka | P6 `domain_lease: SIGKILL atlaisvina nuomą`; P4 `ledger: mirusio proceso rezervacija nebeskaičiuojama` |
| PF-5 | 59,94 Hz: 800,8 garso mėginio grain'ui — fiksuotas 800 arba 801 kaupia A/V poslinkį (1 s per ~17 min) | `kAudioPerField = 960` įrašytas (`flows.hpp:32`); apibendrinant lengva palikti sveiką dalmenį | P2 `format: sample_begin 60000/1001` (5 grain'ai = 4004 mėginiai) |

Kiekvienas turi testą jį valdančiame punkte (pažymėta **[PF-n]**).

---

## 5. Darbai

### M0. Bazinis permatavimas (po P10a, prieš viską kitą)

- Paleisti esamą `scripts/…/70-capacity.sh` (PROFILE=full) ir `71-compare.sh` ant natyviai sukompiliuoto kodo.
- Įrašyti į rezultatus: `CMAKE_CUDA_ARCHITECTURES` iš `CMakeCache.txt`, GPU modelis, tvarkyklė, energijos profilis, laikrodžio zondas.
- Rezultatas: `docs/rezultatai/<data>-bazinis-natyvus.md` su lentele „sm_75 JIT (senas README) | natyvus“.
- **Ant bet kurios NVIDIA GPU:** tie patys skriptai; pakopų sąrašas N nurodomas argumentu, VRAM sargas jau yra harness'e.
- Vienas PR (tik dokumentas + rezultatai).

---

### P10. Kokybės skola

**Tikslas:** teisinga CUDA architektūra be rankinių vėliavėlių; kompiliatoriai keičiami; skriptai nepririšti prie WSL;
`382x40` testo šakninė priežastis.

**Sprendimai**

- Architektūra: prieš `project()` — `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS}) set(CMAKE_CUDA_ARCHITECTURES native)`.
  Atmesta: palikti `89` (neveikia sm_86); `all-major` (ilgas build, didelis binaras).
- Tikra architektūra patenka į binarą ir į `summary-*.json` (`versions.cuda_arch`) — kad matavimas pats pasakytų, kuo kompiliuotas.
- Kompiliatoriai: `CC_BIN` / `CXX_BIN` / `CUDA_HOST_CXX` aplinkos kintamieji skriptuose; numatytieji — `clang-19` / `clang++-19`, jei yra, kitaip `cc` / `c++`. Atmesta: CMake preset'ai (antras tiesos šaltinis šalia MXL preset'ų).
- Skriptai: `scripts/wsl/` → `scripts/linux/`; `scripts/wsl/*.sh` lieka vienos eilutės nukreipimais vienam leidimui (žr. **K2**). `nvidia-smi` — per `command -v`, WSL kelias tik kaip atsarginis.
- `382x40`: **pirma diagnozė, tada pataisa; testas neliečiamas be autoriaus leidimo.**

**Failai**

- Keisti: `CMakeLists.txt:1-13`; `src/mxlio/common_io.cpp` (`versions_json` + `cuda_arch`); `scripts/wsl/env.sh`, `10-build-mxl.sh:17`, `30-build-verify.sh:10-11`, `run-all.sh:34-35`; `tests/perf/capacity.py:44,67`, `tests/perf/scale.py:79`, `tests/functional/functional.py:22`; `README.md` §3, §7, §9.
- Sukurti: `scripts/linux/lib/tools.sh` (`find_nvidia_smi`, `pick_compiler`), `tests/perf/gpuinfo.py` (bendras `nvidia_smi()` kelias Python harness'ui).

**Sąsajos**

```cmake
# CMakeLists.txt: mxlv_common gauna apibrėžimą
target_compile_definitions(mxlv_common INTERFACE VRAMFLOW_CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES}")
```
```bash
# scripts/linux/lib/tools.sh
pick_compiler()   # nustato CC_BIN, CXX_BIN, CUDA_HOST_CXX; klaida, jei nėra nė vieno
find_nvidia_smi() # išveda kelią arba grąžina 1
```

**Testai**

- `ctest -R build_arch` (naujas CTest skriptas `tests/build/check_arch.cmake`): `CMakeCache.txt` reikšmė ≠ tuščia ir ≠ `75`, kai GPU nėra sm_75; `vramflow-func --version` išveda tą pačią reikšmę. Praleidžiamas be GPU.
- `tests/build/test_tools.sh`: `CC_BIN=gcc CXX_BIN=g++ pick_compiler` → `gcc` / `g++`; be kintamųjų ir be `clang-19` → `cc` / `c++`; `find_nvidia_smi` su tuščiu `PATH` → kodas 1.
- 382x40 diagnostikos protokolas (ne testas — matavimas, rezultatai į `docs/rezultatai/`):
  1. 50 paleidimų `vramflow-gfx-writer-tests "gfx_writer: progresyvus 382x40*"` → lentelė: kuri iš 4 `GENERATE` kombinacijų (`cuda_store × cuda_mem`) ir kuris tvirtinimas (`vfgfx_write` kodas ar `chk.matches`) krenta, su `vfgfx_last_error` ir MXL statusu.
  2. Hipotezės tikrinamos po vieną (diagnostinėje šakoje, ne `main`):
     - **H-A (laikas):** rastras mažas, abu `vfgfx_write` baigiasi per < 1 ms, o `base + 1` yra ateities indeksas. Jei skaitymas / rašymas priklauso nuo „dabar“, greita mašina krenta dažniau nei lėta. Patikra: užrašyti `mxlGetCurrentIndex` prie kiekvieno kvietimo; kritimų koreliacija su `now_index < base + 1`.
     - **H-B (< 64 KiB kopijos):** tai vienintelis failo testas, kuriame ir kadras (61 920 B), ir grain'as (61 440 B) mažesni už 64 KiB — CUDA mažas pageable kopijas grąžina prieš DMA pabaigą, o bibliotekos stream'ai `NonBlocking` (`gfx_writer.cpp:379-380`). Patikra: tas pats testas su 382×48 (74 304 B) — tik diagnostikai.
     - **H-C (neinicijuotas padding):** etalonas pildo 0xCD ir perrašo viską; ar CPU kelias (`gfx_cpu.cpp`) perrašo paskutinės grupės ir key eilutės likutį? Patikra: slot'ą prieš rašymą užpildyti 0xEE.
  3. Rasta priežastis → regresijos testas, kuris krenta 100 % be pataisos, + pataisa. Nerasta per 1 dieną → STOP, ataskaita autoriui.

**Matavimai:** `prieš / po`: CUDA architektūra binare; logo N=8 kernel'io laikas p50 (ms); pirmo paleidimo laikas (JIT) s.

**PR:** (1) CMake architektūra + `cuda_arch` į santraukas + testas (~80 eil.); (2) kompiliatorių pasirinkimas + `tools.sh` + testas (~120); (3) skriptų perkėlimas su nukreipimais + README (~250, daugiausia pervadinimai); (4) 382x40 diagnozės ataskaita; (5) 382x40 pataisa + regresijos testas.

**Rizikos:** `native` reikalauja GPU konfigūravimo metu (CI be GPU → `CUDAARCHS` privalomas; README tai pasako). Pervadinus skriptus lūžta išorinės nuorodos — todėl nukreipimai.

**Klausimai:** K2.

---

### P5. Writer / reader sutartis ir sąveika

**Tikslas:** joks writer'is nepalieka reader'io kabėti; elgsena su indeksų spragomis apibrėžta, dokumentuota ir patikrinta su MXL etaloniniais įrankiais.

**Sprendimai**

- `VideoOut::commit` jau nustato `validSlices` — paliekam; pridedam **`GapPolicy`** ir vienintelį kelią, kuriuo funkcijos „nepateikia“ indekso.
- Politikos: `Leave` (šiandien: necommit'inti), `Invalid` (commit'inti grain'ą su `MXL_GRAIN_FLAG_INVALID`, `validSlices = totalSlices`, turinys neliečiamas), `Repeat` (paskutinis geras turinys, be žymės). Numatytoji — **K3**; rekomendacija: `Invalid` naujam `ingest`, esamoms funkcijoms `Leave`, kol P5b neparodys, ko reikia etaloniniams reader'iams. Atmesta: visur `Repeat` (slepia gedimus; skaitytojas nebeatskiria sustingusio vaizdo nuo gedimo).
- README §5.1 pavyzdys taisomas iškart (L7), su taisyklėmis: `validSlices`, kiekvienas indeksas, reader'io elgsena su `INVALID` / `TOO_EARLY` / `TOO_LATE`.
- Sąveikos testai tik su tuo, ką MXL įrankiai priima: `mxl-gst-sink` — host, progresyvus; `mxl-gst-cuda-sink` — CUDA v210, reikia ekrano (`xvfb-run`, `docs/architektura/2026-11-10-vramflow-verify.md:30`); `mxl-info` — viskas.

**Failai**

- Keisti: `README.md:160-222`; `src/func/flows.hpp`, `flows.cpp` (`VideoOut`); `src/func/func_common.hpp` (`Opts::gap_policy`); `src/func/main_func.cpp` (`--gap-policy`); `logo.cpp:160`, `composite.cpp:274`, `dec.cpp:312-316` (per bendrą kelią).
- Sukurti: `tests/gpu/test_contract.cpp` (taikinys `vramflow-flows-tests`), `tests/functional/interop.py`, `docs/architektura/<data>-writer-reader-sutartis.md` (EN — ADR).

**Sąsajos**

```cpp
namespace mxlv::func {
enum class GapPolicy { Leave, Invalid, Repeat };
// Pateikia indeksą be naujo turinio pagal politiką. Grąžina MXL statusą; Leave → MXL_STATUS_OK ir nieko nedaro.
// Repeat be ankstesnio gero grain'o elgiasi kaip Invalid.
mxlStatus VideoOut::skip(std::uint64_t index, GapPolicy p);
bool parse_gap_policy(const std::string& s, GapPolicy& out);   // "leave" | "invalid" | "repeat"
}
```

**Testai** (`tests/gpu/test_contract.cpp`, host ir cuda-linear)

- `contract: commit visada pilni slice'ai` — po `open` + `commit` reader'is su `MXL_GRAIN_VALID_SLICES_ALL`, timeout 0 → `MXL_STATUS_OK`, `validSlices == totalSlices` (1080i: 540).
- `contract: skip(Invalid) skaitomas kaip OK + INVALID` — indeksai 100, 101 (skip), 102 → skaitant 101: `MXL_STATUS_OK`, `flags & MXL_GRAIN_FLAG_INVALID != 0`, `info.index == 101`.
- `contract: skip(Leave) — reader'is gauna tai, ką duoda MXL` — užrašo faktinį statusą indeksui 101 po commit'o 102 (tvirtina tik tai, kas išmatuota pirmame paleidime; reikšmė įrašoma į ADR).
- `contract: skip(Repeat) kartoja paskutinį gerą` — grain'as 101 baitas į baitą == 100, be INVALID.
- `contract: skip(Repeat) be ankstesnio → Invalid`.
- `contract: writer sustojo` — paskutinis indeksas 102, skaitymas 103 su 20 ms timeout → `MXL_ERR_OUT_OF_RANGE_TOO_EARLY` per ≤ 40 ms (ne kabėjimas).

**Funkciniai** (`tests/functional/interop.py`; praleidžiami su aiškia priežastimi, jei įrankio nėra)

- `interop_info`: `gen --format 1080p50` (po P2) → `mxl-info -f <id>`: exit 0, išvestyje `1920`, `1080`, `50/1`.
- `interop_gst_sink_ok`: host 1080p50 flow 20 s → `mxl-gst-sink` su `fakevideosink` (jei įrankis leidžia pasirinkti; kitaip `xvfb-run`): procesas gyvas po 20 s, GStreamer klaidų 0, gauta ≥ 990 kadrų iš 1000.
- `interop_gst_sink_gap` (neigiamas): tas pats, writer'is kas 50-tą indeksą daro `skip(policy)` kiekvienai politikai → lentelė „politika → sink'as tęsia / stringa“; `Invalid` ir `Repeat` privalo tęsti (≥ 970 kadrų); `Leave` rezultatas užrašomas.
- `interop_gst_cuda_sink`: cuda-linear 1080p50 → `mxl-gst-cuda-sink` po `xvfb-run` 20 s: gyvas, 0 klaidų.

**Matavimai:** `Invalid` / `Repeat` kaina: lentelė „politika | commit p50 / p99 ms | papildomas kopijavimas MB/s“ (logo N=8).

**PR:** (1) README pataisa + taisyklės (~60); (2) `GapPolicy` + `skip` + kontrakto testai (~300); (3) funkcijų perjungimas į `skip` + `--gap-policy` (~200); (4) `interop.py` + ADR (~350, po P2).

**Rizikos:** `mxl-gst-*` gali būti nesukompiliuoti šioje MXL instaliacijoje arba reikalauti GStreamer paketų (**K4**). `verify` skaitikliai `missing` / `invalid` pasikeis, kai politika ne `Leave` — testų lūkesčiai keičiami tik tame pačiame PR ir tik su autoriaus žinia.

**Klausimai:** K3, K4.

---

### P2. Formatui neutralūs flow'ai

**Tikslas:** kiekvienas flow turi aiškų formatą (rastras, skenavimas, kadrų dažnis); visos funkcijos priima bent 1080i50, 1080p25, 1080p50; progresyvus 4:2:0 → 4:2:2 teisingas.

**Sprendimai**

- `VideoFormat = gfx::Raster + frame_rate`. `Raster` nekeičiamas (jau naudojamas viešame gfx kelyje). Atmesta: naujas tipas vietoj `Raster` (perrašytų visą gfx medį be naudos).
- Grain'o dažnis: interlaced = 2 × kadrų dažnis (laukas = grain'as), progresyvus = kadrų dažnis. Flow apraše `grain_rate` lieka kadrų dažnis interlaced atveju, kaip dabar (`flows.cpp:28`, `flows.hpp:25`).
- Palaikoma: 1280×720, 1920×1080, 3840×2160; p25, p50, p29,97, p59,94; i50, i59,94 (tik 1080, TFF). BFF ir kiti — atmetami su aiškia klaida. Atmesta: bet koks dažnis (netestuojama).
- `kFieldRate` ir `kAudioPerField` šalinami; laukimas ir indeksai — per formatą. Garso ribos grain'ui — sveikųjų skaičių formule (PF-5).
- Chroma: lauko chromos režimas (esamas) ir **kadro chromos režimas** (naujas). Pasirenkama **kiekvienam kadrui** pagal dekoderio žymę `AV_FRAME_FLAG_INTERLACED`; flow skenavimas fiksuotas kuriant flow.
- Kadro chromos formulė (H.264 / MPEG-2 4:2:0, chromos eilė k tarp luma eilių 2k ir 2k+1), 8 → 10 bitų be apvalinimo:
  luma eilė `f`, `k = f >> 1`: lyginė `f` → `c10 = 3·c[k] + c[max(k−1, 0)]`; nelyginė → `c10 = 3·c[k] + c[min(k+1, K−1)]`
  (maks. 1020). Luma `y10 = y8 << 2` (kaip `media_kernels.cu:162`). Atvirkščiai (4:2:2 → 4:2:0, progresyviam `enc`):
  `c8 = (c10[2k] + c10[2k+1] + 4) >> 3`.
- Chromos padėtis (4:2:0 tipas) — metaduomuo `ChromaSiting420`; įgyvendinamas tik tipas 0 (kairė, tarp eilučių); kiti → klaida, ne tylus neteisingas rezultatas.

**Failai**

- Sukurti: `src/common/format.hpp` (header-only), `tests/unit/test_format.cpp`.
- Keisti: `src/func/flows.hpp`, `flows.cpp` (`video_def`, `raster_flow_def`, `check_rate`, `wait_until`, `VideoOut::create`, `VideoIn::open`, `AudioOut`); `src/media/v210.hpp`, `media_kernels.hpp/.cu`, `media_ref.hpp/.cpp`, `cpu/cpu_media.hpp/.cpp` (matmenys parametrais); `gen.cpp`, `logo.cpp`, `composite.cpp`, `enc.cpp`, `dec.cpp`, `verify.cpp`, `gfxgen.cpp`, `gfx_writer.cpp:224`; `main_func.cpp` (`--format`); testai `tests/unit/test_media*.cpp`, `tests/gpu/test_media_kernels.cpp`, `tests/gpu/test_flows.cpp`, `tests/functional/functional.py`; README §6, §11.

**Sąsajos**

```cpp
namespace vramflow {
struct Rational { std::int64_t num, den; };
enum class ChromaSiting420 { Type0Left, Type1Center, Type2TopLeft };
struct VideoFormat {
    gfx::Raster raster;                 // w, h, interlaced, alpha
    Rational frame_rate;                // 25/1, 50/1, 30000/1001, 60000/1001
    constexpr Rational grain_rate() const;   // interlaced ? {2·num, den} : frame_rate
};
constexpr VideoFormat kFmt1080i50{{1920, 1080, true, false}, {25, 1}};
bool parse_format(const std::string& s, VideoFormat& out, std::string& err);   // "1080i50", "1080p25", "720p50", "2160p50", "1080p59.94" …
bool format_supported(const VideoFormat& f, std::string& why);
bool format_from_flow_def(const std::string& json, VideoFormat& out, std::string& err);
std::string video_flow_def(const std::string& id, const std::string& label, const VideoFormat& f, std::uint64_t run_seed);
// Garso mėginio indeksas, kuriuo prasideda grain'as i: round(i · sr · den / num), sveikaisiais skaičiais.
constexpr std::uint64_t sample_begin(Rational grain_rate, std::uint64_t index, std::uint32_t sample_rate = 48000);
}
namespace mxlv::func {
bool wait_until(mxlRational rate, std::uint64_t index, std::int64_t offset_ns, const std::atomic<bool>& stop);
bool VideoOut::create(mxlInstance, const std::string& def, const std::string& options, std::string& err, const vramflow::VideoFormat& f);
bool VideoIn::open(mxlInstance, const std::string& flow, const OpenPolicy&, const std::atomic<bool>& stop, std::string& err,
                   const vramflow::VideoFormat* expect = nullptr);     // OpenPolicy — P6; iki tol wait_ms
}
namespace mxlv::media {
enum class ChromaMode { FieldSited, FrameSited };
struct Dim { int w, h; };                       // kadro matmenys
// field: −1 = visas kadras (progresyvus flow), 0 / 1 = lauko paritetas (interlaced flow).
cudaError_t split_nv12(const SplitJob*, int n, Dim d, int field, ChromaMode m, int pitch_y, int pitch_uv, cudaStream_t s);
cudaError_t split_nv12_progressive(const SplitJob*, int n, Dim d, int pitch_y, int pitch_uv, cudaStream_t s);  // = field −1, FrameSited
cudaError_t weave_nv12(const WeaveJob*, int n, Dim d, bool interlaced, int pitch_y, int pitch_uv, cudaStream_t s);
// tas pats rinkinys ref:: ir cpu:: vardų erdvėse
}
```

**Testai**

- `format: grain_rate` — 1080i50 → 50/1; 1080p25 → 25/1; 1080p59,94 → 60000/1001; 1080i59,94 → 60000/1001.
- `format: parse / atmetimas` — `"1080p25"` OK; `"1080i25"`, `"1081p50"`, `"1080p24"` → false su tekstu.
- `format: grain dydžiai` — 1080p: 5 529 600 B; 720p: 2 488 320 B; 2160p: 22 118 400 B; 1080i laukas: 2 764 800 B.
- **[PF-5]** `format: sample_begin 60000/1001` — indeksai 0…5 → 0, 801, 1602, 2402, 3203, 4004; indeksas 60 000 → 48 048 000; 50/1: `sample_begin(i) == 960·i` visiems i < 10⁶; 25/1: `1920·i`.
- `format: flow aprašas pirmyn-atgal` — `format_from_flow_def(video_flow_def(f)) == f` visiems palaikomiems.
- `flows: create/open tikrina grainRate prieš formatą` — 1080p25 flow: `cfg.common.grainRate == 25/1`; atidarant su `expect = 1080p50` → klaida su abiem dažniais tekste.
- `media ref: split kadro chroma — rankinės reikšmės` — chroma eilės {100, 200}: luma eilė 0 → 400, 1 → 500, 2 → 700, 3 → 800.
- `media cuda/cpu: split_nv12_progressive == ref` — 0 skirtingų baitų; 1920×1080, 1280×720, 3840×2160; pitch 1920 ir 2048.
- `media cuda/cpu: split FrameSited laukui == ref` — interlaced flow, progresyviai pažymėtas kadras, abu paritetai.
- `media: senas kelias nepakito` — `split_nv12(…, {1920,1080}, parity, FieldSited)` == dabartinis rezultatas (CRC iš `ref_crc.json`).
- Chroma PSNR prieš FFmpeg (`tests/perf/chroma_psnr.py`): sintetinis progresyvus NV12 (zonų plokštė + `clip` turinys) → mūsų 4:2:2 10 bitų prieš `ffmpeg … -pix_fmt yuv422p10le`: U ir V PSNR ≥ 60 dB (pikas 1023), Y skirtingų mėginių 0. Neigiamas: tas pats per `FieldSited` → PSNR < 56 dB (įrodo, kad testas skiria).
- Funkciniai: `gen | logo | verify`, `gen | composite | verify`, `gen | enc | ts_sink` kiekvienam iš 1080i50, 1080p25, 1080p50: 0 teisingumo klaidų, 0 prarastų per 20 s. Neigiamas: `logo` su įėjimu 720p ir logotipu už ribų → exit ≠ 0 su tekstu.

**Matavimai:** lentelė `formatas | funkcija | N* | laukas / kadras p50, p99 ms | VRAM vienam flow MB | SM %`; „prieš“ = M0 (tik 1080i50), „po“ = 1080i50 (turi nepablogėti > 3 %), 1080p25, 1080p50.

**PR:** (1) `format.hpp` + testai (~250); (2) `flows` perjungimas į `VideoFormat`, `wait_until(rate)`, `AudioOut` pagal `sample_begin` (~350); (3) media etalonas + CPU su `Dim` (~350); (4) media CUDA kernel'iai su `Dim` (~350); (5) kadro chroma (ref + cpu + cuda) + PSNR įrankis (~350); (6) `gen`, `logo`, `verify` (~300); (7) `enc`, `dec` (per kadro žymę) (~350); (8) `composite`, `gfxgen`, `gfx_writer` dažnis + funkciniai testai + README (~300).

**Rizikos:** NVENC interlaced / progresyvus skirtumai (`README.md:400`); UHD p50 ring'as 211 MiB — VRAM sargas privalomas; `libvramflow-gfx` ABI: `vfgfx_*` signatūros nekinta (dažnis skaitomas iš flow aprašo).

**Klausimai:** K5.

---

### P1. Laiko modelis bibliotekoje

**Tikslas:** viena „šaltinio laikrodžio“ klasė: PTS → TAI → MXL indeksas su vienu poslinkiu šaltiniui, bendru visiems jo flow'ams; naudojama kiekvienos ingest tipo funkcijos.

**Sprendimai**

- Biblioteka `mxlv_timing` be MXL ir be libav (laikas ir laikrodis paduodami parametrais) ⇒ visa logika testuojama be GPU ir be tinklo.
- **Laikas žymimas demux metu, ne po dekodavimo:** `observe()` kiekvienam paketui grąžina PTS ir DTS jau TAI nanosekundėmis; ingest perrašo `pkt->pts / dts` (laiko bazė 1/10⁹), dekoderis juos perneša į kadrą. Todėl nereikia epochų žymėjimo: vėlesnis poslinkio pakeitimas neliečia jau kelyje esančių kadrų. Atmesta: skaičiuoti iš kadro PTS po dekodavimo (kadrai iš senos epochos po persijungimo gautų naują poslinkį).
- Poslinkis fiksuojamas ties **pirmu video paketu** (šaltiniui be video — ties pirmu paketu); ankstesni paketai metami (`prelock_dropped`).
- Poslinkis kvantuojamas į kadrų tinklelį, kad kiekvienas kadras kristų tiksliai ant grain'o laiko (atsparu 90 kHz apvalinimui).
- Dreifas (**K6**, sprendimas §6): video — viso kadro pakartojimas / išmetimas (interlaced — laukų poromis), garsas
  sekamas sample-tiksliai per `libswresample` kompensaciją (≤ ±0,1 %), kad A/V poslinkis visada ≤ 1 laukas ir nebūtų
  girdimų trūkių. `mxlv_timing` grąžina dreifo įvertį (ppm) ir kadro slydimo įvykius; kompensaciją taiko ingest (P3).
  Testai: sintetinis šaltinis su +100 ppm ir −100 ppm dreifu 10 min — A/V poslinkis ≤ 1 laukas visą laiką, garse nėra
  nulinių / pasikartojančių blokų.

**Algoritmas (pinned; etalonas — vienintelis įgyvendinimas, testai tvirtina reikšmes)**

Žymėjimai: `M = 2^W` (W — wrap bitai; 0 = nėra), `tb` — srauto laiko bazė, `ns(x) = floor(x · tb.num · 10⁹ / tb.den)` (128 bitų
tarpiniai), `L` — latencija, `Tf` — kadro trukmė ns (iš `align_rate`), `off` — poslinkis (TAI = medijos ns + `off`).

1. **Išvyniojimas (kiekvienam srautui atskirai).** `d = ((dts_raw − last_dts_raw) mod M)`, perkeltas į `[−M/2, M/2)`;
   `dts_ext = last_dts_ext + d`. PTS išvyniojamas **DTS atžvilgiu**: `pts_ext = dts_ext + (((pts_raw − dts_raw) mod M)` perkeltas į `[−M/2, M/2))`. W = 0 → `ext = raw`. Pirmam paketui `dts_ext = dts_raw`.
2. **Nutrūkimas.** Jei `ns(d) > disc_fwd_ns` (numat. 1 s) arba `ns(d) < −disc_back_ns` (numat. 0) → nutrūkimas: visų šio šaltinio srautų būsena „laukia perfiksavimo“; paketai metami iki pirmo video paketo.
3. **Fiksavimas** (pirmas video paketas `p`, atvykimo laikas `a`): `off = a + L − ns(dts_ext(p))`.
4. **Perfiksavimas po nutrūkimo:** `cont = E − ns(dts_ext(p))`, kur `E` = paskutinio video paketo prieš nutrūkimą `dts_tai + trukmė`;
   `fresh = a + L − ns(dts_ext(p))`. Jei `|cont − fresh| ≤ relock_slack_ns` (numat. L/2) → `off = cont` (išėjimo laikas tęsiasi be šuolio), kitaip `off = fresh`.
5. **Kvantavimas** (po 3 ir 4, jei `align_rate.num > 0`): `t = ns(pts_ext(p)) + off`; `n = round(t / Tf)` (sveikaisiais: `(t·num + den·5·10⁸) / (den·10⁹)`); `off += time_of(n) − t`. Poslinkis pakinta ≤ Tf/2.
6. **Žyma:** `pts_tai = ns(pts_ext) + off`, `dts_tai = ns(dts_ext) + off`.
7. **Indeksai:** video `index = round(pts_tai · rate)`; interlaced flow — viršutinis laukas `2n` (lyginis, nes tinklelis kadrų), apatinis `2n + 1`. Garsas: `s = round(pts_tai · 48000 / 10⁹)`.
8. **Garso tolydumas (`AudioCursor`):** jei `|s − next| ≤ tol` (numat. 48 mėginiai = 1 ms) → rašoma nuo `next` (be 1 mėginio plyšių dėl apvalinimo); kitaip `gap = s − next`: teigiamas → spraga (nerašoma, `audio_gap_samples`), neigiamas → persidengianti dalis metama (`audio_overlap_samples`); `next = pradžia + n`.
9. **Dreifas:** atsarga `m = dts_tai − a`. Kas 10 s lange imamas `m_min`. Jei 3 langus iš eilės `m_min < L − drift_limit_ns` → `off += Tf`; jei `> L + drift_limit_ns` → `off −= Tf` (po vieną kadrą per langą; `drift_slips`).
10. **Laikrodžio šuolis** (`on_clock_step(Δ)`, iš esamo `ClockWatch`, `flows.hpp:40-46`): `off += round(Δ / Tf) · Tf`. Kadrai, kurių indeksas ≤ paskutinio commit'into, metami rašymo pusėje (`clock_back_dropped`).

**Failai**

- Sukurti: `src/timing/source_clock.hpp`, `source_clock.cpp`, `src/timing/index.hpp` (header-only), `cmake/timing.cmake`, `tests/unit/test_source_clock.cpp`, `tests/unit/test_timing_index.cpp`.
- Keisti: `CMakeLists.txt` (include), `tests/gpu/test_flows.cpp` (sutapimas su MXL), `docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md` (§ „PTS modelis“).

**Sąsajos**

```cpp
namespace mxlv::timing {
struct Rational { std::int64_t num, den; };
std::uint64_t index_of(std::int64_t tai_ns, Rational rate);        // round(t · rate)
std::int64_t  time_of(std::uint64_t index, Rational rate);         // == mxlIndexToTimestamp

struct SourceClockConfig {
    int wrap_bits = 33;
    std::int64_t latency_ns = 1'000'000'000;
    std::int64_t disc_fwd_ns = 1'000'000'000;
    std::int64_t disc_back_ns = 0;
    std::int64_t relock_slack_ns = 500'000'000;
    std::int64_t drift_limit_ns = 200'000'000;
    Rational align_rate{0, 1};          // video kadrų dažnis; {0,1} — nekvantuoti
};
enum class Event : std::uint8_t { None, Locked, Relocked, Dropped };
struct Stamp { std::int64_t pts_tai_ns, dts_tai_ns; Event ev; };    // Dropped → laikai negalioja
struct ClockStats { std::uint64_t wraps, discontinuities, relock_cont, relock_fresh, drift_slips, clock_steps, prelock_dropped; };

class SourceClock {
public:
    explicit SourceClock(const SourceClockConfig& cfg);
    void add_stream(int stream, Rational time_base, bool is_video);
    Stamp observe(int stream, std::int64_t pts_raw, std::int64_t dts_raw, std::int64_t dur_raw, std::int64_t arrival_tai_ns);
    void on_clock_step(std::int64_t step_ns);
    bool locked() const;
    std::int64_t offset_ns() const;
    ClockStats stats() const;
};
struct AudioCursor {
    // Grąžina pradžios mėginio indeksą; gap > 0 spraga, < 0 persidengimas (mėginiais).
    std::uint64_t place(std::int64_t pts_tai_ns, std::uint32_t n, std::uint32_t sample_rate, std::uint32_t tol, std::int64_t& gap);
};
}
```
Viena gija kviečia `observe` (demux gija); `stats()` — bet kuri (atominiai skaitikliai).

**Testai** (`tb = 1/90000`, kadras 3600 tikų = 40 ms, `L = 1 s`, `align_rate = 25/1`, jei nepasakyta kitaip)

- `index: sutampa su MXL` — 10⁵ atsitiktinių `t` × dažniai 25/1, 50/1, 60000/1001, 48000/1: `index_of == mxlTimestampToIndex`, `time_of == mxlIndexToTimestamp`.
- `source_clock: fiksavimas ir kvantavimas` — pirmas video paketas `dts = pts = 900 000`, atvyko `a = 1 700 000 000 123 456 789`: `Event::Locked`; `pts_tai % 40 000 000 == 0`; `|pts_tai − (a + L)| ≤ 20 000 000`.
- `source_clock: wrap` — `dts`: 8 589 930 992 (= 2³³ − 3600), 0, 3600: `dts_tai` skirtumai tiksliai 40 000 000 ir 40 000 000; `stats().wraps == 1`; įvykių nėra.
- **[PF-1]** `source_clock: PTS ir DTS skirtingose wrap pusėse` — `dts = 2³³ − 3600`, `pts = 3600`: `pts_tai − dts_tai == 80 000 000`; kitas paketas `dts = 0`, `pts = 2³³ − 3600` (B kadras): `pts_tai − dts_tai == −40 000 000`.
- `source_clock: 90 kHz drebėjimas` — PTS ±1 tikas nuo tinklelio 1000 kadrų: `index_of(pts_tai, 25/1)` iš eilės, be pasikartojimų ir spragų.
- `source_clock: nutrūkimas pirmyn / atgal, tęsinys` — po 100 kadrų DTS šoka +900 000 (10 s) arba −450 000 (kilpa), atvykimas tolydus: `Event::Relocked`; pirmo naujo video kadro `dts_tai` = ankstesnio + 40 000 000; `relock_cont == 1`.
- `source_clock: šaltinis dingo 10 s` — DTS +10 s ir atvykimas +10 s: `relock_fresh == 1`; `|dts_tai − (a + L)| ≤ 20 000 000`.
- `source_clock: du flow'ai lieka sulygiuoti` — video ir garso paketai su tuo pačiu PTS prieš ir po wrap + nutrūkimo: `|time_of(index_of(pts_tai_v, 25/1), 25/1) − s_a · 10⁹ / 48000| ≤ 10 417 ns` kiekvienai porai.
- `source_clock: garsas prieš fiksavimą metamas` — 5 garso paketai prieš pirmą video: visi `Event::Dropped`; `prelock_dropped == 5`.
- `source_clock: laikrodžio šuolis` — `on_clock_step(−1 290 000 000)` → `offset_ns` pakito tiksliai −1 280 000 000 (32 kadrai); `+15 000 000` → 0.
- **[PF-2]** `source_clock: dreifas +200 ppm` — imituojamas atvykimas 200 ppm lėtesnis nei DTS, 1 val. medijos laiko: `drift_slips` tarp 17 ir 19 (0,72 s / 40 ms); `|m_min − L| ≤ 240 000 000` visą laiką; nė vieno `Relocked`.
- `audio_cursor: tolydumas` — AAC kadrai po 1024 mėginius, PTS 90 kHz (21,33 ms → 1920 tikų): 1000 kadrų → pradžios tiksliai `s0 + 1024·k`, `gap == 0`.
- `audio_cursor: spraga ir persidengimas` — PTS +100 ms → `gap == 4800`; −50 ms → `gap == −2400`.
- `source_clock: tik garso šaltinis` — `align_rate {0,1}`: fiksuojama ties pirmu paketu, kvantavimo nėra (`pts_tai == a + L` tiksliai).

**Matavimai:** `observe()` kaina: ns vienam kvietimui (tikslas < 200 ns; `tests/perf/bench_timing.cpp`, be GPU). Našumo „prieš / po“ nėra (nauja biblioteka).

**PR:** (1) `index.hpp` + testai + MXL sutapimas (~150); (2) `SourceClock` išvyniojimas, fiksavimas, kvantavimas + testai (~350); (3) nutrūkimai, perfiksavimas + testai (~300); (4) dreifas, laikrodžio šuolis, `AudioCursor` + testai + dokumentas (~350).

**Rizikos:** `h264_cuvid` laiko žymes viduje verčia į 100 ns vienetus — paklaida ≤ 100 ns, kvantavimas ją sugeria (tikrinama P3 teste). HLS `EXT-X-DISCONTINUITY`: libavformat pats perskaičiuoja laiką — mūsų detektorius tada nieko nemato; tai gerai, bet turi būti patikrinta P3.

**Klausimai:** K6, K7.

---

### P6. Tvirtas paleidimas ir sustabdymas

**Tikslas:** paleidimo tvarka nesvarbi; flow'ai gali atsirasti, dingti ir grįžti; du paleidimai vienas kito netrina.

**Sprendimai**

- Įėjimų laukimas: `--input-wait-s S`; **numatyta −1 = neribotai** su eilute į stderr kas 5 s („laukiu flow X, N s“); `0` = klaida iškart. Atmesta: palikti 15 s numatytąjį (L12).
- Re-attach: `composite` logika (`composite.cpp:184-221,253-256`) iškeliama į bendrą `InputSlot` ir naudojama `logo`, `enc`, `verify`, `mv`.
- Formato pasikeitimas po re-attach (PF-3): `--on-format-change exit|wait`; numatyta `exit` (kodas 6, aiškus tekstas). Atmesta: tyliai prisitaikyti (išėjimo flow formatas fiksuotas).
- Domeno nuoma: failas su `flock` (branduolys atlaisvina ir po `kill -9`). Vieta — **K8**; rekomendacija: `<domenas>.vramflow/` šalia domeno (MXL domeno katalogas lieka švarus).
- Harness'as: unikalūs domenai (`mkdtemp`), trynimas tik per nuomos patikrą. Fiksuoti vardai (`/dev/shm/mxlf-<tag>-<n>`) šalinami.
- Išėjimo flow jau egzistuoja ir turi gyvą writer'į → klaida su savininko `run_id` tekste (ne tik „created=false“).

**Failai**

- Sukurti: `src/runtime/input_slot.hpp/.cpp`, `src/runtime/domain_lease.hpp/.cpp`, `cmake/runtime.cmake`, `tests/gpu/test_runtime.cpp`, `tests/perf/domain.py`, `tests/functional/startstop.py`.
- Keisti: `flows.hpp/.cpp` (`VideoIn::open`, `AudioIn::open` su `OpenPolicy`); `logo.cpp:103`, `enc.cpp:351-353`, `composite.cpp:135,164,184-221`, `verify.cpp:314-316`; `func_common.hpp` (`Opts`), `main_func.cpp`; `capacity.py:260-267,387`, `scale.py:69-91`, `functional.py:66-67`, `examples/ograf/run-demo.sh`.

**Sąsajos**

```cpp
namespace mxlv::runtime {
struct OpenPolicy { std::int64_t wait_ms = -1; std::uint32_t log_every_ms = 5000; };   // −1 neribotai, 0 iškart
enum class FormatChange { Exit, Wait };
class InputSlot {
public:
    enum class State { Waiting, Attached, Detached, Mismatch };
    InputSlot(std::string flow_id, const vramflow::VideoFormat* expect, FormatChange on_change);
    State poll(mxlInstance inst, std::uint64_t now_tai_ns);          // neblokuoja; bandymas ne dažniau kaip kas 100 ms
    mxlStatus get(std::uint64_t index, std::uint64_t timeout_ns, mxlGrainInfo& info, const std::uint32_t*& ptr);  // FLOW_INVALID → Detached
    State state() const;
    const vramflow::VideoFormat& format() const;
    std::uint64_t reattached() const;
};
class DomainLease {
public:
    static bool acquire(const std::string& domain, const std::string& run_id, DomainLease& out, std::string& err);
    static std::vector<std::string> live_owners(const std::string& domain);   // run_id sąrašas; negyvi ignoruojami
    ~DomainLease();
};
}
```
```python
# tests/perf/domain.py
def make_domain(tag: str) -> pathlib.Path            # /dev/shm/vf-<tag>-<8 hex>
def remove_domain(path) -> bool                       # False ir nieko netrina, jei yra gyvų savininkų
```

**Testai**

- `input_slot: vartotojas prieš gamintoją` — `poll` 3 s grąžina `Waiting`; writer'is sukuriamas → per ≤ 200 ms `Attached`.
- `input_slot: gamintojas nužudytas ir grįžo` — `Attached` → writer'is uždaromas → `Detached` per ≤ 2 grain'us; naujas writer'is tuo pačiu ID → `Attached`, `reattached() == 1`.
- **[PF-3]** `input_slot: formatas pasikeitė po re-attach` — 1080i50 → 1080p25: su `Exit` → `Mismatch`, tekste abu formatai; su `Wait` → lieka `Detached`, kol grįžta 1080i50, tada `Attached`.
- `open_policy: 0 → klaida iškart (< 100 ms)`; `−1 → vis dar laukia po 20 s` (testas su 20 s riba), sustabdymo signalas nutraukia per ≤ 100 ms.
- `domain_lease: du savininkai` — A ir B viename domene: `live_owners` = {A, B}; A baigia → {B}.
- **[PF-4]** `domain_lease: SIGKILL atlaisvina nuomą` — vaikas paima nuomą, `kill -9` → `live_owners` tuščias per ≤ 100 ms; `remove_domain` → True.
- `domain.py: neištrina svetimo` — gyvas `gen` domene; `remove_domain` → False, flow failai vietoje.
- Funkciniai (`startstop.py`): (a) `logo` paleistas 30 s prieš `gen` → po `gen` starto per 2 s `fields_ok` didėja, exit 0; (b) `gen` nužudomas ir po 5 s paleidžiamas vėl → `logo`: `reattached == 1`, po to 0 prarastų 20 s; (c) du `capacity.py` paleidimai lygiagrečiai tuo pačiu `--func logo --n 2` → abu PASS (šiandien vienas ištrina kito domeną); (d) neigiamas: `gen` antrą kartą tame pačiame domene ir tuo pačiu `--flow-base` → exit ≠ 0, tekste pirmojo `run_id`.

**Matavimai:** `re-attach laikas ms (p50 / maks. iš 20 bandymų)`; `poll` kaina tick'e µs (turi būti < 5 µs, kai `Attached`).

**PR:** (1) `OpenPolicy` + `--input-wait-s` visose funkcijose (~200); (2) `InputSlot` + testai (~350); (3) `composite` perkėlimas ant `InputSlot` (elgsena nekinta; esami PF-3 testai žali) (~250); (4) `logo`, `enc`, `verify` re-attach (~300); (5) `DomainLease` + testai (~250); (6) harness'o domenai + `startstop.py` (~300).

**Rizikos:** `flock` ant `tmpfs` (`/dev/shm`) veikia, ant tinklo FS — ne; nuomos katalogas visada šalia domeno (tas pats FS). `mxlGarbageCollectFlows` kito paleidimo metu — P6 testas (c) tai dengia.

**Klausimai:** K8.

---

### P7. Gyva stebėsena

**Tikslas:** veikiant matyti kiekvieno flow ok / late / lost, latenciją p50 / p99 ir kiekvieno variklio apkrovą; pabaigos santraukos lieka.

**Sprendimai**

- **JSON eilutės** (viena eilutė per intervalą) į failą arba stderr; Prometheus / HTTP — vėliau ir tik su autoriaus sprendimu (**K9**). Atmesta dabar: HTTP serveris (nauja priklausomybė arba savas kodas su saugumo paviršiumi).
- Skaitiklius rašo tik tick gija (`func_common.hpp:5`); kas intervalą ji **nukopijuoja** momentinę kopiją į publikuotoją; rašo atskira gija. Tick'as niekada neblokuojamas I/O.
- Varikliai per NVML (`nvml.h` yra CUDA įrankių rinkinyje; biblioteka `libnvidia-ml.so.1` — tvarkyklės dalis): NVDEC %, NVENC %, SM %, VRAM. Krovimas per `dlopen` — be NVML funkcija veikia, laukai `null`. Atmesta: `nvidia-smi` procesas iš C++.
- Histogramos: intervalo ir kaupiamoji (esama `Histogram`, `src/common/histogram.hpp`).

**Failai**

- Sukurti: `src/runtime/stats_publisher.hpp/.cpp`, `src/runtime/engine_probe.hpp/.cpp`, `tests/unit/test_stats_publisher.cpp`, `tests/gpu/test_engine_probe.cpp`, `tests/perf/stats_tail.py` (gyvas rodinys terminale), `docs/architektura/<data>-stebesena.md` (schema, EN).
- Keisti: `func_common.hpp/.cpp` (`Opts::stats_interval_s`, `stats_out`; `write_summary` nekinta), visos funkcijos (vienas `publisher.tick(st, now)` kvietimas cikle), `main_func.cpp`, `capacity.py` (skaito JSON eilutes vietoje atskiro `nvidia-smi` zondo — P8).

**Sąsajos**

```cpp
namespace mxlv::runtime {
struct EngineSample { double nvdec_pct, nvenc_pct, sm_pct; std::uint64_t vram_used_bytes, vram_total_bytes; bool valid; };
class EngineProbe {
public:
    bool open(const std::string& device_uuid, std::string& err);     // false → be NVML
    EngineSample sample();
};
class StatsPublisher {
public:
    StatsPublisher(const func::Opts& o, std::string role, std::string run_id);
    void tick(const func::Stats& s, std::uint64_t now_tai_ns);       // tick gija; kopijuoja tik kai praėjo intervalas
    void close();
};
}
```
Eilutės schema (v1): `{"v":1,"t":<TAI ns>,"role":"logo","run":"…","interval_s":1.0,"flows":[{"n":0,"id":"…","ok":50,"late":0,"lost":0,"invalid":0,"lat_p50_ms":1.2,"lat_p99_ms":2.9}],"engine":{"nvdec_pct":41.0,"nvenc_pct":0.0,"sm_pct":9.0,"vram_used_mb":2210},"clock":{"steps":0}}`.
Parinktys: `--stats-interval-s 1` (0 = išjungta), `--stats-out PATH|-`.

**Testai**

- `stats_publisher: viena eilutė per intervalą` — imituojamas laikas, intervalas 1 s, 10,5 s → tiksliai 10 eilučių; kiekviena galiojantis JSON su `v == 1`.
- `stats_publisher: skirtumai, ne sumos` — skaitiklis `fields_ok` 50, 100, 150 → eilutėse `ok` 50, 50, 50.
- `stats_publisher: tick neblokuojamas` — rašymo paskirtis užblokuota (FIFO be skaitytojo): 1000 `tick` kvietimų < 1 ms kiekvienas; `dropped_lines` skaitiklis > 0.
- `stats_publisher: išjungta` — intervalas 0 → failas nesukuriamas, `tick` kaina < 50 ns.
- `engine_probe: reikšmės ribose` (GPU) — `valid`, visi % ∈ [0, 100], `vram_used ≤ vram_total`; neigiamas: neegzistuojantis UUID → `open` false su tekstu.
- `engine_probe: be NVML` — `LD_LIBRARY_PATH` be bibliotekos → `open` false, funkcija tęsia, laukai `null`.
- Funkcinis: `logo` N=4 20 s su `--stats-out`: ≥ 18 eilučių; Σ `ok` per eilutes == `fields_ok` iš `summary-logo.json` (tikslus sutapimas).

**Matavimai:** `prieš / po` logo N=32: lauko p50 / p99 ms ir CPU branduolių vienam flow (stebėsena įjungta ir išjungta; leistinas skirtumas ≤ 1 %).

**PR:** (1) `StatsPublisher` + testai (~350); (2) `EngineProbe` + testai (~250); (3) įjungimas visose funkcijose + funkcinis testas (~250); (4) `stats_tail.py` + schema + README (~200).

**Rizikos:** WSL2 NVML neteikia atminties pagal procesą — VRAM vienam flow skaičiuojamas iš viso įrenginio skirtumo (P8); `utilization.decoder` atnaujinimo periodas ~1 s — trumpesnis intervalas beprasmis.

**Klausimai:** K9.

---

### P3. Tikra ingest funkcija su garsu

**User story:** `docs/user-stories/2026-11-11-P3-ingest.md` (laukia patvirtinimo).

**Tikslas:** `vramflow-func ingest`: TS per UDP / RTP / SRT ir HLS → vienas demuxeris šaltiniui → video (NVDEC arba programinis) ir garso dekodavimas → PTS pririšti video ir garso flow'ai. `dec` lieka testiniam klipui.

**Sprendimai**

- Transportas per **libavformat protokolus** (`udp`, `rtp`, `srt`, `hls` / `https`) — be naujų bibliotekų mūsų build'e; ar sisteminis FFmpeg turi `srt` — tikrinama paleidimo metu su aiškia klaida (**K10**). Atmesta: tiesioginis libsrt (nauja priklausomybė, dubliuoja libavformat).
- Gijos vienam šaltiniui: **demux** (skaito, žymi laiku per `SourceClock`, deda į paketų eilę), **video dekodavimas**, **garso dekodavimas**. MXL rašo tik tick gijos — po vieną kiekvienai grain'o dažnio grupei; kiekvienas writer'is priklauso tiksliai vienai gijai.
- Dvi atsargos: **latencija** `--latency-ms` (numat. 1000; suspausti paketai host RAM — pigu) ir **dekodavimo atsarga** `--decode-lead-ms` (numat. 400; kiek anksčiau prieš pateikimą paketas leidžiamas į dekoderį). 400 ms × 25 fps = ≤ 10 laikomų NV12 kadrų ≈ 31 MB srautui (vietoje 78 MB su 1 s). Atmesta: leisti paketus į dekoderį iškart (HLS pliūpsnis užpildo paviršius, VRAM nevaldoma).
- Kadras laukia savo indekso žemėlapyje `index → kadras`; tick'as indeksui `i` ima tiksliai `i`. Nėra → `VideoOut::skip(i, policy)` (P5). Pavėlavęs kadras (indeksas < einamojo) — metamas ir skaičiuojamas.
- Flow formatas nustatomas iš srauto parametrų prieš kuriant flow (`--format auto`); vidury srauto pasikeitus raiškai / skenavimui → šaltinis stabdomas su klaida (kiti tęsia); automatinis perkūrimas — vėlesnis etapas.
- Garsas: dekoduojamas, verčiamas į 48 kHz float32 (libswresample — FFmpeg dalis, bet nauja nuoroda build'e → **K10**), kanalai `--audio-channels` (numat. 2). Rašoma per `AudioCursor` mėginių indeksais; spragos nerašomos.
- Flow ID: naujos rolės `07` (ingest video) ir `08` (ingest garsas), numeris = šaltinio eilės numeris + `--flow-base`.

**Failai**

- Sukurti: `src/ingest/demux.hpp/.cpp`, `src/ingest/video_decode.hpp/.cpp`, `src/ingest/audio_decode.hpp/.cpp`, `src/ingest/feed_gate.hpp` (header-only, gryna logika; P4 ją plečia), `src/func/ingest.cpp`, `tests/unit/test_feed_gate.cpp`, `tests/functional/ingest.py`, `tests/perf/ts_burst.py`, `tests/perf/av_offset.py`, `scripts/linux/62-make-av-clip.sh`.
- Keisti: `flows.hpp/.cpp` (`AudioOut` kanalų skaičius, rolės), `main_func.cpp`, `cmake/func.cmake`, `func_common.hpp` (`Opts`), `src/media/gen_content.hpp` + `media_ref` (garso žymė klipui), README §6.

**Sąsajos**

```cpp
namespace mxlv::ingest {
struct SourceSpec { std::string url; std::string format; /* "auto" | "1080i50" … */ bool audio = true; int audio_channels = 2; };
struct Packet { AVPacket* pkt; std::int64_t pts_tai_ns, dts_tai_ns; bool key; int stream; };
class Demux {                                   // viena gija
public:
    bool open(const SourceSpec&, const timing::SourceClockConfig&, std::string& err);
    bool video_format(vramflow::VideoFormat& out) const;
    bool read(Packet& out);                     // false → pabaiga / klaida (error())
};
struct FeedGateConfig { std::int64_t lead_ns = 400'000'000; };
enum class Feed { Hold, Send };
Feed feed_decision(std::int64_t dts_tai_ns, std::int64_t now_tai_ns, const FeedGateConfig&);   // Send, kai dts_tai − now ≤ lead
struct DecodedFrame { AVFrame* f; std::int64_t pts_tai_ns; bool interlaced; };
class VideoDecoder { public: bool open(const AVCodecParameters*, bool nvdec, AVBufferRef* hw, int surfaces, std::string& err);
                     bool send(const Packet&); bool receive(DecodedFrame& out); void flush(); };
class AudioDecoder { public: bool open(const AVCodecParameters*, int out_channels, std::string& err);
                     bool send(const Packet&); bool receive(std::vector<float>& planar, std::uint32_t& n, std::int64_t& pts_tai_ns); };
}
namespace mxlv::func { int run_ingest(const Opts&); }
```
CLI: `vramflow-func ingest --domain D --input URL [--input URL …] [--flow-base B] [--format auto|…] [--decoder nvdec|sw] [--latency-ms 1000] [--decode-lead-ms 400] [--surfaces 0] [--audio 1] [--audio-channels 2] [--gap-policy invalid] [--input-wait-s -1]` + bendros parinktys.

**Testinė medžiaga (sintetinė):** `62-make-av-clip.sh` gamina klipus 1080i50 ir 1080p25: video — esamas `clip` su lauko / kadro žyme (`media_ref::decode_marker`); garsas — 1 mėginio impulsas (amplitudė 0,9) tiksliai kadro, kurio numeris ≡ 0 (mod 25), pradžioje, kitur tyla; H.264 + AAC, MPEG-TS. `av_offset.py` skaito abu flow'us ir skaičiuoja `impulso_mėginys / 48000 − grain'o_laikas`.

**Testai**

- `feed_gate: Hold / Send ribos` — `dts − now` = 400 000 001 ns → Hold; 400 000 000 → Send; neigiamas → Send.
- Funkciniai (`ingest.py`), kiekvienas 1080i50 ir 1080p25, NVDEC ir programinis:
  - `ingest_av_offset` — klipas per UDP realiu laiku 60 s: kiekvienam impulsui `|A/V poslinkis| ≤ 1 laukas` (20 ms; 1080p25 — ≤ 1 kadras = 40 ms; tikimasi ≤ 1 ms, užrašoma); video `fields_ok ≥ 99,8 %`; 0 `late`.
  - `ingest_bursty_no_drift` — `ts_burst.py` siunčia po 2 s medžiagos per 50 ms, tada tyla, 120 s, `--latency-ms 3000`: A/V poslinkio skirtumas tarp t = 10 s ir t = 110 s ≤ 1 ms; `drift_slips == 0`; `relock == 0`.
  - `ingest_marker_continuity` — video žymės išėjime iš eilės (kiekvienas indeksas = ankstesnis + 1 žymės vienetas), išskyrus kilpos sandūrą (`relock_cont` didėja 1 per kilpą, indeksai be spragos).
  - `ingest_pts_order_not_arrival` — siuntėjas kas 5 s vėlina paketus 300 ms (`ts_burst.py --stall`): išėjimo indeksų ir žymių atitikimas nepakinta (šiandieninis `dec` čia pasislinktų).
  - **[PF-2]** `ingest_drift_accelerated` — siuntėjas 0,5 % greitesnis (`--speed 1.005`), 120 s: `drift_slips ≥ 1`; A/V poslinkis visą laiką ≤ 1 laukas; nė vieno `underrun` > 1 kadro iš eilės.
  - `ingest_hls` — vietinis HLS (ffmpeg segmentuotojas + `python3 -m http.server` ant `127.0.0.1`) 60 s: tie patys A/V ir tolydumo kriterijai. `ingest_srt` — `srt://127.0.0.1:…`; praleidžiamas su priežastimi, jei FFmpeg be `srt`.
  - Neigiami: `ingest_bad_url` → exit ≠ 0 per ≤ `--input-wait-s`, tekste URL ir libav klaida; `ingest_no_video` (tik garso TS) → garso flow yra, video nėra, exit 0; `ingest_format_change` (klipas 1080i → 720p vidury) → šaltinis sustabdytas su tekstu, kiti šaltiniai tęsia, exit kodas 6; `ingest_unsupported` (4:2:2 arba 10 bitų H.264) → aiški klaida, ne iškraipytas vaizdas.
  - `ingest_progressive_chroma` — 1080p25 klipas: išėjimo chroma PSNR prieš FFmpeg ≥ 60 dB (per `chroma_psnr.py`).

**Matavimai:** `prieš` = `dec` (M0), `po` = `ingest`: `N* (srautų) | NVDEC % | SM % | VRAM srautui MB | CPU branduolių srautui | pateikimo vėlavimas p50 / p99 ms | A/V poslinkis vid. / maks. ms`; atskirai 1080i50 ir 1080p25; `--decode-lead-ms` 200 / 400 / 800 įtaka VRAM ir vėlavimams.

**PR:** (1) AV klipo generatorius + garso žymė + `av_offset.py` (~300); (2) `Demux` + `SourceClock` įjungimas (~350); (3) `feed_gate` + `VideoDecoder` (~350); (4) `run_ingest` tick variklis, tik video (~400); (5) `AudioDecoder` + garso flow (~350); (6) funkciniai testai UDP + pliūpsniai (~350); (7) HLS / SRT / RTP testai + README (~300).

**Rizikos:** kelių dažnių grupės viename procese (kelios tick gijos); HLS su atskirais garso grojaraščiais — libavformat sujungia į vieną kontekstą, bet atvykimas nesinchroninis: latencija turi dengti segmento trukmę (dokumentuojama, `--latency-ms` klaida, jei mažesnė už aptiktą segmentą); NVDEC seansų riba plokštėje.

**Klausimai:** K10, K11.

---

### P8. Talpos harness v2

**Tikslas:** talpa matuojama pagal variklį, automatiškai randamas „kelias“, VRAM vienam flow; README rezultatai permatuoti natyviai; išvestis — talpos failas P4.

**Sprendimai**

- Plėsti esamą `tests/perf/capacity.py` (ne antras harness'as). Variklių duomenys — iš P7 JSON eilučių (vienas tiesos šaltinis), ne iš atskiro `nvidia-smi`.
- **Kelio algoritmas:** N dvigubinamas nuo 1 iki pirmo FAIL arba VRAM sargo; tada dalijimas pusiau tarp paskutinio PASS ir pirmo FAIL; kandidatas PASS tik jei praeina R = 3 pakartojimus. Kelias = didžiausias toks N. PASS apibrėžimas — esamas (0 teisingumo klaidų, praradimai ≤ `--tol`).
- VRAM vienam flow = tiesės `mem_used(N)` nuolydis per PASS pakopas (mažiausi kvadratai; skelbiamas ir R²).
- Dekodavimo palyginimai: `h264_cuvid` prieš `h264` + `hwaccel cuda` (abu jau FFmpeg'e); tiesioginis Video Codec SDK — tik su autoriaus sprendimu (**K12**). Paviršių skaičius `--surfaces` 4 / 8 / 16 / 25.
- Talpos failas: `capacity-<gpu-uuid>.json` su išmatuotais `nvdec_mpix_s`, `nvenc_mpix_s`, `vram_per_flow_bytes` pagal formatą, kelio N ir matavimo metaduomenimis (architektūra, tvarkyklė, data).

**Failai**

- Keisti: `tests/perf/capacity.py` (`--knee auto`, `--repeats`, JSON eilučių skaitymas), `tests/perf/compare_report.py` (variklių ir VRAM stulpeliai), `scripts/linux/70-capacity.sh`, `README.md` §8, `docs/rezultatai/`.
- Sukurti: `tests/perf/knee.py` (gryna logika), `tests/perf/test_knee.py`, `tests/perf/capacity_file.py`, `src/func/dec.cpp` / `ingest` parinktis `--decoder-api cuvid|hwaccel`.

**Sąsajos**

```python
# tests/perf/knee.py
def find_knee(run_level, n_max: int, repeats: int = 3) -> KneeResult   # run_level(n) -> bool
@dataclass
class KneeResult: knee: int; tried: list[tuple[int, list[bool]]]; stopped_by: str   # "fail" | "n_max" | "vram_guard"
def vram_per_flow(levels: list[tuple[int, float]]) -> tuple[float, float, float]   # nuolydis MB, laisvasis narys MB, R²
```

**Testai** (`tests/perf/test_knee.py`, be GPU)

- `knee: tikslus slenkstis` — imituota `run_level(n) = n ≤ 13`, `n_max = 64` → `knee == 13`; bandytos pakopos 1, 2, 4, 8, 16, 12, 14, 13.
- `knee: nestabili pakopa` — N = 10 praeina 2 iš 3 → skaitoma FAIL; `knee == 9`.
- `knee: riba nepasiekta` — visada PASS, `n_max = 32` → `knee == 32`, `stopped_by == "n_max"`.
- `knee: N = 1 krenta` → `knee == 0` (ne išimtis).
- `vram_per_flow` — taškai (1, 310), (2, 420), (4, 640), (8, 1080) → nuolydis 110,0 ± 0,1 MB, R² ≥ 0,999.
- `capacity_file: schema` — sugeneruotas failas turi visus privalomus laukus; trūkstamas `nvdec_mpix_s` → įkėlimo klaida su lauko vardu.

**Matavimai (pagrindinė šio punkto išvestis)** — stulpeliai: `funkcija | formatas | variantas | kelio N | NVDEC % | NVENC % | SM % | VRAM srautui MB (R²) | CPU branduolių srautui | lauko p50 / p99 ms | GPU laikrodis MHz | pakartojimai`. „Prieš“ = README §8 (sm_75 JIT, WSL2), „po“ = natyviai sukompiliuota. Dekodavimo lentelė: `API | paviršiai | kelio N | NVDEC % | VRAM srautui`.
**Ant bet kurios NVIDIA GPU:** `bash scripts/linux/70-capacity.sh` be argumentų; N riba ir formatų sąrašas — aplinkos kintamaisiais; kiekvienas rezultatas turi GPU modelį ir `cuda_arch`.

**PR:** (1) `knee.py` + testai (~250); (2) `capacity.py` integracija + JSON eilutės (~350); (3) VRAM nuolydis + ataskaitos stulpeliai (~250); (4) `--decoder-api`, `--surfaces` + dekodavimo kampanija (~300); (5) talpos failas (~200); (6) permatuoti rezultatai + README (dokumentai).

**Rizikos:** nešiojamos GPU laikrodžio mažinimas prie mažos apkrovos (README `:354`) iškreipia kelią — harness'as užrašo laikrodį ir energijos profilį ir atmeta pakopas, kur laikrodis < 50 % nominalo (žymi `INVALID_ENV`, ne FAIL).

**Klausimai:** K12.

---

### P4. Termino vartai perkrovoje ir priėmimo kontrolė

**User story:** `docs/user-stories/2026-11-11-P4-overload-admission.md` (laukia patvirtinimo).

**Tikslas:** perkrova nebegriauna visų išėjimų; naudotojas negali tyliai paleisti daugiau, nei variklis aptarnauja.

**Sprendimai**

- **Termino vartai** (plečia P3 `feed_gate`): paketas, kuris nebespės (`pts_tai < now + min_decode_ns`), nebedekoduojamas; šaltinis pereina į `Skip` — meta paketus iki pirmo rakto kadro `K`, kuriam `pts_tai(K) ≥ now + min_decode_ns`, tada `flush()` ir tęsia. Praleisti indeksai — per `GapPolicy`. Atmesta: dekoduoti viską ir mesti kadrus (šiandien; L4).
- `min_decode_ns` — matuojamas kiekvienam šaltiniui (slenkantis p99 „paketas → kadras“ laikas), ne konstanta; pradinis 150 ms.
- Histerezė: ≥ 3 `Skip` įvykiai per 10 s → šaltinio būsena `Degraded` (į gyvą statistiką); `--shed newest|none` (numat. `none`): su `newest` vėliausiai priimtas šaltinis stabdomas, kad kiti atsigautų.
- **Priėmimas:** paklausa ir talpa — megapikseliais per sekundę vienam varikliui (1080p25 = 51,84 Mpix/s) ir VRAM baitais. Talpa — iš P8 talpos failo; be failo — `--admission warn` elgsena su aiškiu įspėjimu „talpa neišmatuota“. Atmesta: spėti talpą iš GPU modelio lentelės (sensta, neteisinga nešiojamoms).
- Režimai: `--admission off|warn|enforce` (numatytoji — **K13**; rekomendacija `warn`, kol nėra talpos failo, `enforce` — kai yra).
- **Knyga tarp procesų:** rezervacijos nuomos kataloge (P6), po vieną failą procesui su `flock`; mirusio proceso rezervacija nebeskaičiuojama (PF-4). Atmesta: demonas (naujas komponentas be reikalo).
- Atsarga `headroom` numat. 0,85 talpos.

**Failai**

- Sukurti: `src/runtime/admission.hpp/.cpp`, `tests/unit/test_admission.cpp`, `tests/unit/test_deadline_gate.cpp`, `tests/functional/overload.py`.
- Keisti: `src/ingest/feed_gate.hpp`, `src/func/ingest.cpp`, `src/func/enc.cpp` (NVENC rezervacija), `func_common.hpp`, `main_func.cpp`, `StatsPublisher` schema (`state`, `skip_events`), README.

**Sąsajos**

```cpp
namespace mxlv::ingest {
enum class GateState { Normal, Skip };
enum class GateAction { Hold, Send, Drop, SendAfterFlush };
struct DeadlineGate {
    GateState state = GateState::Normal;
    GateAction on_packet(std::int64_t pts_tai_ns, std::int64_t dts_tai_ns, bool key, std::int64_t now_tai_ns,
                         std::int64_t lead_ns, std::int64_t min_decode_ns);
    std::uint64_t skip_events = 0, dropped_packets = 0;
};
}
namespace mxlv::runtime {
struct Demand { double nvdec_mpix_s = 0, nvenc_mpix_s = 0; std::uint64_t vram_bytes = 0; };
struct Capacity { double nvdec_mpix_s, nvenc_mpix_s; std::uint64_t vram_bytes; double headroom = 0.85; std::string source; };
struct Decision { bool admit; std::string engine; double used, requested, limit; };
bool load_capacity(const std::string& path, const std::string& device_uuid, Capacity& out, std::string& err);
Demand ingest_demand(const vramflow::VideoFormat& f, int surfaces, std::uint32_t ring_grains, bool nvdec);
class Ledger {
public:
    static bool open(const std::string& domain, const std::string& device_uuid, Ledger& out, std::string& err);
    Decision reserve(const std::string& id, const Demand& d, const Capacity& c);   // atominė (flock)
    void release(const std::string& id);
    Demand total() const;                                                          // tik gyvų procesų
};
}
```
`DeadlineGate::on_packet` taisyklės (pinned): `Normal`: jei `pts_tai < now + min_decode` → `state = Skip`, `++skip_events`, `Drop`; kitaip jei `dts_tai − now > lead` → `Hold`; kitaip `Send`. `Skip`: jei `key` ir `pts_tai ≥ now + min_decode` → `state = Normal`, `SendAfterFlush`; kitaip `Drop`, `++dropped_packets`.

**Testai**

- `deadline_gate: normalus kelias` — `lead` 400 ms, `min_decode` 150 ms; paketas su `pts − now` = 500 ms, `dts − now` = 300 ms → `Send`; `dts − now` = 450 ms → `Hold`.
- `deadline_gate: pavėlavęs → Skip iki rakto` — seka: `pts − now` = 100 ms (ne raktas) → `Drop`, `skip_events == 1`; ne raktas 200 ms → `Drop`; raktas 120 ms → `Drop` (nespės); raktas 160 ms → `SendAfterFlush`, būsena `Normal`.
- `deadline_gate: be raktų` — 1000 ne raktinių paketų `Skip` būsenoje → visi `Drop`, `skip_events == 1` (ne 1000).
- `admission: ribos` — talpa NVDEC 1000 Mpix/s, `headroom` 0,85 (riba 850); 1080p25 srautai po 51,84: 16-tas (829,44) priimamas, 17-tas (881,28) atmetamas; `Decision.engine == "nvdec"`, `limit == 850`.
- `admission: VRAM riboja pirmiau` — talpa 6 GiB, srautas 135 MB → 38-tas priimamas (5130 MB ≤ 5222 MB), 39-tas (5265 MB) atmetamas, `engine == "vram"`.
- `ledger: du procesai` — A rezervuoja 10, B mato `total` = 10 ir gauna likutį; A `release` → B mato 0.
- **[PF-4]** `ledger: mirusio proceso rezervacija nebeskaičiuojama` — vaikas rezervuoja ir gauna `kill -9` → `total()` == 0 per ≤ 100 ms.
- `capacity: be failo` — `enforce` + nėra failo → exit ≠ 0 su tekstu „talpa neišmatuota, paleiskite 70-capacity“; `warn` → paleidžia su įspėjimu.
- Funkciniai (`overload.py`):
  - `overload_enforce_synthetic` (bet kuri GPU): talpos failas su riba 4 srautams, paleidžiami 8 → pirmi 4: 60 s, `late == 0`, `skip_events == 0`; 5–8: neįsileidžiami, pranešime variklis ir skaičiai, jų flow'ų domene nėra (`mxl-info -l` rodo 8 flow'us = 4 video + 4 garso); exit kodas 5.
  - `overload_enforce_measured` (ši GPU): `N_k` iš P8; paleidžiama `2·N_k` → priimta `floor(0,85·N_k)`, visiems priimtiems `late == 0` per 60 s.
  - `overload_no_collapse` (neigiamas, `--admission off`): `2·N_k` srautų 60 s → nė vienas išėjimas neturi > 2 s spragos iš eilės; kiekvieno `fields_ok ≥ 40 %` (siūloma riba, kalibruojama pirmame matavime); sustabdžius perteklinius `N_k` srautų, likę per ≤ 5 s grįžta į `late == 0` ir būseną `Normal`.
  - `overload_shed_newest`: `--shed newest`, `N_k + 4` srautų → sustabdomi tiksliai vėliausi, kol likę `Normal`; pirmi `floor(0,85·N_k)` — `late == 0` po 10 s.

**Matavimai:** `apkrova (× kelio N) | režimas | priimta | srautų su late = 0 | fields_ok % (min / vid.) | ilgiausia spraga s | atsigavimo laikas s | NVDEC %`; eilutės 1,0× / 1,5× / 2,0× kiekvienam iš `off` be vartų (= `dec` šiandien, „prieš“), `off` su vartais, `enforce`.

**PR:** (1) `DeadlineGate` + testai (~250); (2) įjungimas į ingest + `Degraded` + statistika (~300); (3) `Capacity`, `Demand`, sprendimo logika + testai (~300); (4) `Ledger` + testai (~300); (5) CLI, `enc` rezervacija, `--shed` (~300); (6) `overload.py` + rezultatai + README (~350).

**Rizikos:** GOP ilgis realiuose šaltiniuose 1–10 s — `Skip` gali trukti iki GOP; tai matoma ir dokumentuojama, ne slepiama. Talpos failas pasensta pakeitus tvarkyklę / GPU — faile yra tvarkyklės versija, neatitikimas → įspėjimas.

**Klausimai:** K13.

---

### P9. GPU stebėjimo / rodymo kelias

**User story:** `docs/user-stories/2026-11-11-P9-gpu-multiviewer.md` (laukia patvirtinimo).

**Tikslas:** multivieweris VRAM'e: sumažintų flow'ų tinklelis su aliarmais (juoda, sustingęs vaizdas, tyla, garso lygiai), canvas kaip MXL flow, pasirinktinai zero-copy rodymas.

**Sprendimai**

- **Du atskiri rezultatai:** (a) `vramflow-func mv` — canvas kaip `v210` cuda-linear MXL flow (patikrinama visur, įskaitant šią mašiną; rodyti gali bet kuris MXL vartotojas); (b) `vramflow-view` — pasirinktinis rodytuvas su CUDA–GL arba Vulkan interop. **(b) ant WSL2 nepatikrinama** (CUDA–OpenGL interop WSL nepalaikomas [nepatikrinta čia — patikrinti NVIDIA „CUDA on WSL“ apribojimus prieš pradedant]) ⇒ tikrinama tik natyviame Linux; sprendimas GL ar Vulkan — **K14**.
- Mastelio keitimas tiesiai v210 → v210 ploto vidurkiu (sveikieji skaičiai, etalonas + CPU + CUDA, 0 skirtingų baitų). Atmesta: per RGBA (du papildomi konvertavimai).
- Aliarmai skaičiuojami GPU redukcijos kernel'iu iš sumažintos plytelės; garsas — host'e iš garso flow. Būsenos į P7 JSON eilutes ir nupiešiamos canvas'e (rėmelis, lygių stulpeliai).
- Aliarmų apibrėžimai (numatytieji; slenksčiai parinktimis, žr. **K15**): **juoda** — plytelės Y vidurkis ≤ 80 ir maksimumas ≤ 128 (10 bitų) ≥ 2 s; **sustingęs** — to paties pariteto plytelių absoliučių skirtumų suma = 0 ≥ 2 s; **tyla** — visų kanalų RMS < −60 dBFS ≥ 5 s; **nėra signalo** — įėjimas `Waiting` / `Detached` (P6). Lygiai — pikas ir RMS dBFS per 100 ms.
- Įėjimai — per `InputSlot` (P6): multivieweris niekada nesustoja dėl dingusio flow.

**Failai**

- Sukurti: `src/mv/mv_ref.hpp/.cpp`, `src/mv/cpu/mv_cpu.cpp`, `src/mv/mv_kernels.hpp/.cu`, `src/mv/alarms.hpp` (header-only), `src/func/mv.cpp`, `cmake/mv.cmake`, `tests/unit/test_mv_ref.cpp`, `tests/unit/test_alarms.cpp`, `tests/gpu/test_mv_kernels.cpp`, `tests/functional/mv.py`; (b) `src/view/` + `cmake/view.cmake` (tik jei K14 patvirtinta).
- Keisti: `main_func.cpp`, `flows.hpp` (rolė `09`), `StatsPublisher` schema (`alarms`), README.

**Sąsajos**

```cpp
namespace vramflow::mv {
struct Tile { const std::uint8_t* src; gfx::Raster src_r; int x, y, w, h; };   // paskirtis canvas'e; x, w lyginiai
struct TileStats { std::uint32_t y_mean, y_max; std::uint64_t sad; };          // sad — prieš ankstesnę to paties pariteto plytelę
namespace ref  { void scale_tile(const Tile& t, int parity, const gfx::Raster& canvas, std::uint8_t* out, TileStats* st); }
namespace cuda { cudaError_t scale_tiles(const Tile* t, int n, int parity, const gfx::Raster& canvas, std::uint8_t* out,
                                         std::uint8_t* prev_small, TileStats* st_dev, cudaStream_t s); }
struct AlarmConfig { std::uint32_t black_mean = 80, black_max = 128; double black_s = 2, freeze_s = 2, silence_s = 5, silence_dbfs = -60; };
struct AlarmState { bool black, frozen, silent, no_signal; double peak_dbfs[2], rms_dbfs[2]; };
struct AlarmTracker { AlarmState update(const TileStats* st, bool attached, const float* l, const float* r, std::uint32_t n,
                                        std::int64_t now_tai_ns, const AlarmConfig& c); };
}
```
CLI: `vramflow-func mv --domain D --input UUID [--input …] --grid CxR | --layout file.json --canvas WxH --format 1080p50 [--audio UUID …] [--alarm-* …] [--out-flow 1]`.

**Testai**

- `mv ref: 2:1 mažinimas, rankinės reikšmės` — 4×4 Y blokas {64, 64, 940, 940 …} → vidurkis 502; chroma poros teisingai.
- `mv cuda/cpu == ref` — 0 skirtingų baitų: 1080i50 → 480×270, 1080p25 → 640×360, UHD → 960×540; 16 plytelių viename canvas'e.
- `mv: plytelė iš dalies už canvas` → klaida (ne kirpimas, ne rašymas už ribų).
- `alarms: juoda` — Y vidurkis 64 lygiai 2,0 s → aliarmas įsijungia ties 2,0 s (ne 1,98); vienas šviesus laukas → išsijungia.
- `alarms: sustingęs` — `sad == 0` 2 s → įsijungia; `sad == 1` → ne. Interlaced: du skirtingi, bet pastovūs laukai (stop-kadras) → **sustingęs** (lyginama pagal paritetą).
- `alarms: tyla ir lygiai` — sinusas −20 dBFS → RMS −23,0 ± 0,1, pikas −20,0 ± 0,1; nuliai 5 s → `silent`; −59 dBFS → ne.
- `alarms: nėra signalo` — `attached == false` → `no_signal`, kiti aliarmai nevertinami.
- Funkciniai (`mv.py`): 16 `gen` įėjimų → canvas 1920×1080p50 60 s: `fields_ok ≥ 99,8 %`, canvas == etalonui (kas 25-tas kadras); vienas `gen` nužudomas → per ≤ 1 s `no_signal` toje plytelėje, kiti nepaliesti, išėjimas be spragų; `gen --complexity 0` su juodu turiniu (nauja `gen` parinktis `--pattern black|freeze`) → `black` / `frozen` per 2 ± 0,1 s; neigiamas: 17 įėjimų į 4×4 tinklelį → exit ≠ 0 su tekstu.
- (b) `view`: tik natyviame Linux — 60 s rodymas, D2H baitų skaitiklis == 0 (matuojama per `cudaMemcpy` apvalkalą), pateikimo vėlavimas p99 ≤ 1 kadras.

**Matavimai:** `įėjimų N | canvas | SM % | VRAM MB | kadro p50 / p99 ms | D2H MB/s`; „prieš“ = rodymas per host kopiją (3440×1440: 13,3 MB per kadrą v210, ~20 MB RGBA), „po“ = canvas flow (0 D2H funkcijoje) ir `view` (0 D2H iš viso).

**PR:** (1) `mv_ref` + testai (~300); (2) CPU + CUDA kernel'iai + testai (~400); (3) `alarms.hpp` + testai (~300); (4) `run_mv` variklis + išėjimo flow (~400); (5) aliarmų piešimas + statistika (~300); (6) `mv.py` + `gen --pattern` (~300); (7–8) `vramflow-view` (po K14).

**Rizikos:** VRAM: 16 įėjimų × 26–53 MiB jau yra kituose procesuose; pats `mv` prideda canvas ring'ą (1080p50: 52,7 MiB; 3440×1440: 126,6 MiB) ir 16 mažų „ankstesnių“ plytelių (< 5 MB). Skirtingų dažnių įėjimai viename canvas'e: canvas dažnis fiksuotas, įėjimai imami „naujausias ne vėlesnis nei“ (kartojimas / praleidimas plytelėje — tik stebėjimui, dokumentuojama).

**Klausimai:** K14, K15.

---

## 6. Sprendimai

Autorius 2026-11-11 delegavo sprendimus su kriterijumi: praktiškas, universalus pasirinkimas transliacijai —
sinchronas ir lūpų sinchronas, signalų priėmimas ir siuntimas, kodavimas / dekodavimas, garsas, profesionalūs
standartai, nesuspaustas video su garsu ir grafika su alfa kanalu. Žemiau — priimti sprendimai (ne siūlymai).

| # | Klausimas | Sprendimas | Kodėl (transliacijos požiūriu) |
|---|---|---|---|
| K1 | Commit trailer'is | `Assisted-by: Claude Code (<modelis>)`; upstream pataisoms — jokių DI trailer'ių | Brief'as naujausias; DCO upstream'ui lieka autoriaus |
| K2 | Skriptų katalogas | `scripts/linux/` + `scripts/wsl/` nukreipimai vienam leidimui | Veikia natyviai ir WSL2, esamos instrukcijos nesulūžta |
| K3 | Spragų politika | Writer'is **commit'ina kiekvieną indeksą**. Kai turinio nėra — grain'as su `MXL_GRAIN_FLAG_INVALID`, turinys neišgalvojamas (kaip autorius nusprendė kompozitoriui). Parinktis `--on-loss invalid\|freeze\|black` (numatytoji `invalid`); `freeze` — pakartoti paskutinį gerą ne ilgiau nei `--hold-ms` (numatyta 1000), po to `black`. P5b išmatuoja, kaip `mxl-gst-sink` / `mxl-gst-cuda-sink` elgiasi su INVALID; jei jie vis tiek stringa, numatytoji tampa `freeze` ir tai įrašoma README | Indeksų tęstinumas = jokių užstrigusių skaitytojų; skaitytojas mato tikrą signalo praradimą (kaip SDI „no signal“ žymė), o eterio operatorius gali pasirinkti frame-sync elgseną |
| K4 | Sąveikos testai | Įdiegti `xvfb` (vienintelis trūkstamas; GStreamer ir `mxl-gst-*` jau yra). Testai be įrankių žymimi `SKIP` su priežastimi, ne `PASS` | Sąveika su MXL etaloniniais įrankiais — universalumo įrodymas |
| K5 | Formatai | 720p50, 1080i50, 1080p25, 1080p50, tada 59,94 šeima (1080i59.94, 720p59.94, 1080p29.97/59.94), UHD p50 / p59.94 — atskiri PR pabaigoje | EBU (50 Hz) ir SMPTE (59,94 Hz) šeimos — abi būtinos universaliam įrankiui |
| K6 | Dreifas ir lūpų sinchronas | Profesionalaus frame synchronizer'io modelis: video — viso **kadro** pakartojimas / išmetimas (interlaced — poromis, kad nesusimaišytų laukų tvarka), garsas sekamas **sample-tiksliai** per `libswresample` kompensaciją (`swr_set_compensation`, ≤ ±0,1 %), kad A/V poslinkis visada ≤ 1 laukas. Be girdimų trūkių | Lūpų sinchronas — privalomas; girdimas 40 ms trūkis eteryje nepriimtinas |
| K7 | Ingest latencija | 1000 ms TS / SRT / RTP; HLS — automatiškai ≥ 3 segmentai; parinktis `--latency-ms` | Atsparu tinklo drebėjimui, keičiama žemesnei latencijai |
| K8 | Nuomos / knygos failai | `<domenas>.vramflow/` šalia domeno | MXL domeno turinys lieka tik MXL; `mxl-info` ir GC jo neliečia |
| K9 | Gyva statistika | Dabar JSON eilutės (`--stats-file`, `--stats-interval`, numatyta 1 s); Prometheus / HTTP ir NMOS IS-12 / BCP-008 — vėlesni etapai, kai bus pasirinkta HTTP biblioteka (pin, ≥ 30 d.) | Jokios naujos priklausomybės dabar; JSON eilutes lengva paversti Prometheus / Grafana |
| K10 | SRT / HLS / RTP / garso konvertavimas | Per sisteminį FFmpeg (`libavformat` protokolai: `srt`, `udp`, `rtp`, `hls` — patikrinta FFmpeg 6.1.1 WSL'e) + `libswresample` (jau įdiegtas, ta pati FFmpeg versija). Tiesioginio `libsrt` nėra | Viena priklausomybė visiems įėjimams; FFmpeg — de facto standartas |
| K11 | Viešas LRT srautas rankiniam bandymui | Taip, tik rankiniams bandymams (viešai transliuojamas turinys); automatiniai testai — tik sintetika | Atitinka autoriaus turinio jautrumo taisyklę |
| K12 | Video Codec SDK palyginimui | Dabar tik `h264_cuvid` prieš `hwaccel cuda`; tiesioginis SDK — atskiras sprendimas vėliau | Be naujos licencijuotos priklausomybės |
| K13 | Priėmimo kontrolė | `enforce`, kai yra talpos failas (nauji srautai virš ribos atmetami su aiškia klaida; jau priimti niekada nesuprastinami); `warn`, kai talpos failo nėra; `off` — tik rankiniams bandymams | Eteryje esantys srautai saugomi pirmiausia |
| K14 | Zero-copy rodymas | Šiame etape — multiviewer'io drobė kaip **MXL flow'as VRAM'e** (bet kuris MXL vartotojas ją rodo / siunčia toliau, kaip SDI/IP multiviewer išėjimas). Tiesioginis ekranas vėliau per Vulkan išorinę atmintį (veikia ir be X serverio), natyviame Linux | Profesionalus multiviewer'is išveda signalą, o ne langą; WSL2 čia nepatikrinama |
| K15 | Aliarmų slenksčiai | Numatytieji iš §P9, visi keičiami parinktimis | — |
| K16 | Kelios GPU | Vienas procesas = viena GPU (`--device-uuid`), talpos knyga kiekvienai GPU atskira | Paprasta, nuspėjama, tinka orkestratoriams (K8s) |
| K17 | Formatas pasikeitė vidury srauto | Šaltinis sustabdomas su aiškia klaida ir statistikos įrašu; orkestratorius paleidžia iš naujo | Flow'o formatas MXL'e nekintamas — perkūrimas be žinios suklaidintų skaitytojus |
| R20 | Pageable įkėlimo apsauga `libvramflow-gfx` | Biblioteka **pati** užtikrina, kad CUDA kadras pilnai įrašytas prieš jį skaitydama (laukia visų konteksto darbų prieš konvertavimą); kaina pamatuojama ir dokumentuojama; ekspertams — vėliava `VFGFX_FLAG_CALLER_SYNCED` | Trečiųjų šalių programos neturi galėti sugadinti grafikos nepastebimai |

User stories P3, P4 ir P9 laikomos patvirtintomis su šiais sprendimais (autorius delegavo, 2026-11-11).

---

## 7. Kas šiame plane NEpatikrinta vykdymu

- Niekas nekompiliuota ir nepaleista (WSL galėjo būti užimtas) — visi verdiktai §2 paremti skaitymu.
- L16: tikroji `CMAKE_CUDA_ARCHITECTURES` reikšmė esamame build kataloge.
- L18: `382x40` nestabilumas ir visos trys hipotezės.
- L8: `mxl-gst-sink` elgsena su spragomis; ar `mxl-gst-*` yra šioje MXL instaliacijoje.
- Ar sisteminis FFmpeg turi `srt` protokolą ir kokia jo versija.
- CUDA–GL interop WSL2 apribojimas (P9 b).
- Visi 3090 skaičiai iš brief'o (800 fps, 125 MB srautui, 50/54 → 62/66 dB).
