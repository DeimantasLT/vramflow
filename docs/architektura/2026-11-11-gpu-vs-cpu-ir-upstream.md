# GPU prieš CPU: sąžiningas pajėgumo palyginimas, spragų analizė ir upstream pataisų planas

- **Data:** 2026-11-11
- **Tikslas (užsakovo žodžiais):** (1) sunkiausias apdorojimas turi persikelti iš RAM/CPU į GPU, ir tai reikia **išmatuoti** — kiek pilnų flow'ų telpa GPU kelyje, palyginti su CPU keliu; (2) paruošti MXL pataisas ir faktų paketą, kad GPU payload'us būtų galima siūlyti į standartą kaip patikimą, veikiantį sprendimą; (3) sinchronizacija ir tikslus laikas — privalomi ir visomis prasmėmis lygūs host MXL (laiko nustatymą aprašo atskiras dokumentas).
- **Skaitytojas:** įgyvendintojas. Remiasi `2026-11-10-vramflow-verify.md` (toliau **[V]**) ir `2026-11-10-gpu-media-funkcijos.md` (toliau **[F]**); jų turinys nekartojamas.
- **Šiame etape nieko nebuild'inta ir nepaleista** (vyksta talpos kampanija). Viskas žemiau — kodo ir upstream skaitymas + planas.

Žymėjimas:
**[patikrinta: šaltinis]** — perskaityta 2026-11-10/11. MXL keliai be prefikso = PR #653 head `thomastrue/mxl` SHA `6dddc91d92f56c3c9d90fa089c95d8f262bc2c92` (lokali kopija WSL `~/src/mxl-pr653`, šaka `pr-653`); „main" = `dmf-mxl/mxl` `3aaa8ed6` (2026-11-09).
**[išmatuota: šaltinis]** — mūsų matavimas šioje mašinoje (RTX 4070 Laptop 8 GB, Ryzen AI 9 HX 370, WSL2 mato 12 loginių CPU, 32 GB RAM).
**[prielaida]** — nepatikrinta; nurodyta, kuri užduotis (§6) tai patvirtina.
**[sprendimas]** — mūsų dizaino pasirinkimas; pažymėti **D-n** reikalauja Deimanto patvirtinimo (§7).

---

## 0. Svarbiausi radiniai (skaityti pirmiausia)

| # | Radinys | Šaltinis |
|---|---|---|
| K1 | **Dabartinis „host vs GPU" transporto matavimas nėra „CPU vs GPU".** `--storage host` kelyje turinį vis tiek generuoja ir tikrina GPU, tik per host flow su D2H/H2D kopijomis. Tai — **hibridas (H)**: „GPU funkcija ant šiandieninio MXL v1.1". Tikro CPU kelio (C) dar neturime nė vienai funkcijai. | [patikrinta: `src/mxlio/verify_writer.cpp:193,203,215` (D2H), `src/mxlio/verify_reader.cpp:304` (H2D)] |
| K2 | x264 kelias taip pat hibridinis: weave → NV12 daro GPU, tada vienas D2H į pinned buferį → libx264. Kad tai būtų CPU etalonas, reikia host flow + CPU konversijos. | [patikrinta: `src/func/enc.cpp:5,448`] |
| K3 | Reikia **trijų** variantų kiekvienai funkcijai: **C** (CPU + host flow), **H** (GPU + host flow), **G** (GPU + cuda-linear). `G/C` pagrindžia investiciją į GPU (transliuotojo nauda); `G/H` pagrindžia patį PR #653 (kam MXL reikia device storage) — upstream'ui svarbiausias būtent šis. | [sprendimas] |
| K4 | **Slice'ai / daliniai grain'ai device storage'e API lygiu JAU palaikomi**: `mxlFlowReaderGetGrainSlot(…, minValidSlices, timeoutNs, …)` ir `OpenGrainSlot` + `CommitGrain` su `validSlices < totalSlices`. UNSUPPORTED grąžina tik rodyklinis API (`OpenGrain`/`GetGrain*`). Trūksta **testų** (visi upstream testai naudoja tik `MXL_GRAIN_VALID_SLICES_ALL`) ir sutarties dokumentacijos. Kandidatas (c) = testai + doc, ne bibliotekos keitimas. | [patikrinta: `lib/include/mxl/flow.h` (PR diff: `mxlFlowReaderGetGrainSlot`, `mxlFlowWriterOpenGrainSlot`); `lib/internal/src/PosixDiscreteFlowReader.cpp:152-167,234`; `PosixDiscreteFlowWriter.cpp:104-118,170-175,208-238`; grep `validSlices` per `lib/tests/cuda/*`, `lib/tests/test_grain_storage.cpp` — tik `totalSlices`/`VALID_SLICES_ALL`] |
| K5 | **Sync grupės su device flow'ais pagal kodą turi veikti**: `waitForDataAt` kviečia `waitForGrain` → `getGrainImpl(…, nullptr, nullptr)` — payload'o neliečia, dirba tik su host shared memory antraštėmis ir futex'u. Testo nėra (vienintelis sync grupių testas — host flow). Kandidatas (b) = tik testas. | [patikrinta: `lib/internal/src/FlowSynchronizationGroup.cpp:77-99`; `PosixDiscreteFlowReader.cpp:90-108`; `lib/src/flow.cpp:1009-1040`; `lib/tests/test_flow_sync_groups.cpp:47` (vienas TEST_CASE, be `payload` opcijų)] / [prielaida: kad testas praeis be pataisų — U2] |
| K6 | **PR #653 pašalino iš `mxl-info` eilutes „Payload Location" / „Device Index" ir nieko nepridėjo** — po PR `mxl-info` nebeparodo, ar flow yra host, ar device. Kandidatas (d) galioja. | [patikrinta: `git diff 05b06b15 pr-653 -- tools/mxl-info/main.cpp lib/internal/src/FlowInfo.cpp` (−12/−12 eil.); grep `storage|payload|backend` per `tools/mxl-info/main.cpp` — 0 atitikmenų] |
| K7 | **Interlaced atmeta visi `mxl-gst` įrankiai, ne tik CUDA**: `sink.cpp:1419,1475`, `testsrc.cpp:867`, `cuda_testsrc.cpp:439-441`. Tai bendra upstream spraga, ne #653 trūkumas → kandidatas (e) kaip CUDA pataisa netinka; tai atskiras reikalavimas (REQ). Pati biblioteka interlaced palaiko. | [patikrinta: nurodytos eilutės; `lib/internal/src/FlowParser.cpp:288-307` per [F] G1] |
| K8 | `examples/cuda-cross-container-test.sh:26` — `sudo docker info` be `-n`: ne-root vartotojui be NOPASSWD testas kabo ties slaptažodžio užklausa. Kandidatas (a) galioja, 1 eilutė. | [patikrinta: failas, eil. 24-31] |
| K9 | **Upstream kelias — ne #653 „vienu gabalu", o serija**: #746 (bisect-pt, Pedro Ferreira) „Separate grain payload storage from the discrete flow ring" — pirmas iš serijos, konverguojančios su #653; toliau seks descriptor API, backend loader'is, `cuda-linear`. #653 dabar savyje laiko #746 kopiją „tik testavimui". Mūsų pataisos turi taikyti į tai, kas bus merge'inta. | [patikrinta: api.github.com `pulls/746` (open, mergeable clean, 18 failų, atnaujinta 2026-11-07); `issues/653/comments` (thomastrue 2026-10-30)] |
| K10 | Precedentas, kaip įnešti indėlį: danbouzaSWS testavo #653, komentare pateikė matavimus + pataisą su testu; autorius ją integravo per 3 dienas. Tai greičiausias ir pigiausias kelias mūsų (a), (b), (c) radiniams. | [patikrinta: `issues/653/comments` 2026-10-16, 10-31, 11-03] |
| K11 | Reikalavimas jau yra: `mxl-requirements#190` „[REQ] GPU memory support" (5 user stories, atviras). Architektūros užduotis — `mxl#594` (milestone v1.2, 40 komentarų). Naujo REQ GPU atminčiai kurti nereikia; mūsų įrodymai jungiami prie #190/#594. | [patikrinta: api.github.com `mxl-requirements/issues/190`, `mxl/issues/594`] |
| K12 | Griežtu kriterijumi (0 prarastų grain'ų) transporto N* WSL2 aplinkoje **abiem keliams yra 8**: cuda-linear N=16 gap=609, N=32 gap=1412; host N=16 gap=178, N=32 gap=94 400. GPU kelias degraduoja švelniai, host — griūva, bet „N_gpu/N_cpu" pagal 0 praradimų čia = 1. Riba tikėtina WSL2/WDDM jitter'io, ne MXL — autoritetingam skaičiui reikia natyvaus Linux. | [išmatuota: `evidence/20261110T194604Z/report.md:74-92`] / [prielaida: priežastis WSL — T14] |
| K13 | Device atmintį reader'is gauna **READWRITE** (host flow reader'is — read-only mmap). Klaidingas reader'is gali sugadinti writer'io payload'ą. Tai sąmoningai užfiksuota kaip atviras klausimas dma-buf/image backend'ams, bet galioja ir `cuda-linear`. | [patikrinta: `lib/payload-backends/cuda-linear/CudaLinearBackend.cpp:433-447` (`CU_MEM_ACCESS_FLAGS_PROT_READWRITE` vienintelis); `docs/PayloadStorage.md` „Open questions"] |
| K14 | `CudaLinearStorage.h` (storage tipo vardas, device rodyklės vieta `reserved` baituose) **neinstaliuojamas** — instaliuojama tik `.so`. Išorinė media funkcija turi kopijuoti vidinį header'į. Naujas mažas pataisos kandidatas (h). | [patikrinta: `lib/payload-backends/cuda-linear/CMakeLists.txt:58-61`; `#include "CudaLinearStorage.h"` tik repo viduje: `tools/mxl-gst/cuda_*.cpp`, `lib/tests/cuda/*`] |

---

## 1. Sąžiningas GPU-prieš-CPU palyginimas („kiek pilnų flow'ų")

### 1.1 Trys variantai ir ką kiekvienas įrodo

| Variantas | Flow storage | Apdorojimas | Atitinka | Šiandien turime |
|---|---|---|---|---|
| **C** | host (`mxlFlowWriterOpenGrain` / `GetGrain`, rodyklinis API) | tik CPU, GPU neliečiamas | tikra CPU MXL media funkcija (v1.1) | **nieko** [patikrinta: K1, K2] |
| **H** | host | GPU + H2D/D2H per PCIe | GPU funkcija ant MXL v1.1 (be #653) | transportas `--storage host`; x264 dalinai |
| **G** | `cuda-linear` | GPU, payload nepalieka VRAM | #653 | viskas (gen, logo, enc-nvenc, dec, verify) |

Taisyklės sąžiningumui [sprendimas]:
1. **Tas pats įėjimas, tas pats išėjimas, ta pati patikra.** C ir G variantų išėjimai tikrinami tuo pačiu `verify` ir tais pačiais nepriklausomais etalonais (`src/media/media_ref.cpp`). Logo ir dec: **0 skirtingų baitų** tarp C ir G išėjimų. Enc: kokybės suderinimas (§1.4).
2. **CPU variantas turi būti geras, ne „šiaudinis".** Rašomas taip, kaip rašytų kompetentingas CPU funkcijos autorius: SIMD (mašina turi AVX2 ir AVX-512 F/BW/VBMI [patikrinta: `/proc/cpuinfo` WSL]), tik būtinos kopijos, gijų skaičius parenkamas geriausias (sweep), `-O3 -march=x86-64-v3` + runtime dispatch. Ataskaitoje nurodomas kompiliatorius, vėliavėlės, gijos.
3. **Ta pati mašina, tas pats laikas, išskirtinis paleidimas.** Variantai leidžiami pakaitomis (C, G, H, C, G, H…), ≥3 pakartojimai, niekada lygiagrečiai su kitu benchmarku.
4. **Matuojama dvejopai:** (i) *funkcijos kaina* — tik funkcijos proceso branduoliai/srautui (perkeliama į kitą geležį); (ii) *sistemos N\** — visa grandinė (šaltinis + funkcija + patikra) toje pačioje pakopoje (tai, ką mašina realiai atlaiko).
5. **Skelbiami ir nepalankūs rezultatai.** Tikėtina, kad logo CPU variantas bus pigus (§1.3) — tai rašoma atvirai; nauda rodoma ten, kur ji yra (koderis, dekoderis, pilna grandinė, H→G kopijų pašalinimas).

### 1.2 PASS(N) ir N\* apibrėžimas (vienodas C, H, G)

`config/bench.json` (naujas; jokių hardcode) [sprendimas]:

| Kriterijus | Slenkstis | Šaltinis kode |
|---|---|---|
| Prarasti laukai (`gap`, `late_skips`, `frames_dropped_*`) | 0 | esami skaitikliai `FlowStats` [patikrinta: `src/func/func_common.hpp`] |
| Turinio klaidos (`mismatch_words`, `outside_diff_words`, CRC) | 0 | `verify` [patikrinta: [F] §7.1] |
| Indekso/laiko atitiktis: išėjimo grain indeksas = įėjimo indeksas + deklaruotas pastovus vėlinimas (laukais); barkodo indeksas = grain indeksas | 100 % imties | barkodas [patikrinta: `media_ref.hpp` `decode_marker`] |
| A/V sinchronizacija (enc/chain): audio mėginio indeksas ↔ video lauko indeksas, PTS skirtumas | \|Δ\| ≤ 1 audio kadras AAC (1024 mėg.) TS lygyje; 0 mėginių MXL lygyje | [prielaida: matuoklis `ts_sink.py` — T9] |
| Realaus laiko santykis | ≥ 0,999 | [patikrinta: `tests/perf/capacity.py` docstring] |
| e2e p99 (indekso laikas → išėjimas prieinamas) | ≤ 1 laukas (20 ms) logo/transportui; enc/dec — `deadline` iš konfigūracijos | [sprendimas] |
| Laikrodžio šuoliai lange | registruojami; pakopa su šuoliu kartojama (ne FAIL, ne PASS) | `ClockWatch` [patikrinta: `src/func/flows.hpp`] |
| Visi procesai | exit 0 | esama |

N\* = didžiausias N, kai PASS(N) ir visi mažesni PASS. Procedūra: 60 s atranka (10 s apšilimas) → **10 min patvirtinimas ties N\*** ir fiksuotas FAIL ties N\*+žingsnis → 2 h soak ties ⌊0,8·N\*⌋ (soak šiuo metu NOT_RUN [išmatuota: `evidence/…/report.md:94`]).

### 1.3 CPU atitikmenys pagal funkciją

**Bendras karkasas** [sprendimas]: `vramflow-func` gauna dvi naujas opcijas visoms subkomandoms — `--impl gpu|cpu` ir `--storage cuda-linear|host` (H = `--impl gpu --storage host`), plius `--threads K`. `--impl cpu` procese **neinicijuojamas CUDA kontekstas** (kitaip CUDA runtime gijos ir pinned atmintis iškreipia CPU/RAM skaičius); tai tikrinama: `nvidia-smi --query-compute-apps` nerodo proceso.

Reikalingi kodo pakeitimai:
- `src/func/func_common.hpp` `Opts`: `impl`, `storage`, `threads`; `src/func/main_func.cpp`: opcijos + validacija (`--impl cpu` ⇒ `--storage host`).
- `src/func/flows.{hpp,cpp}`: `VideoOut` šiuo metu grąžina device rodyklę (tik cuda-linear) — pridėti host šaką (kaip `VideoIn`, kuri jau skiria `cuda` [patikrinta: `src/func/flows.cpp:151`, `flows.hpp` `VideoIn::cuda`]). Abiem storage'ams naudoti **descriptor API** (`MapSlots`/`OpenGrainSlot`/`GetGrainSlot`) — jis veikia ir host flow [patikrinta: `lib/tests/test_grain_storage.cpp:174` „Mapped slots match the pointer based API"]; C variantui papildomai `--api pointer`, kad būtų matuojamas tikras v1.1 kelias [sprendimas].
- Tick variklis (`logo.cpp`, `enc.cpp`, `dec.cpp`) atskiriamas nuo skaičiavimo backend'o: sąsaja `Backend { prepare(); process(batch); finish(); }` su `Gpu*` ir `Cpu*` realizacijomis. Tick logika (laikas, deadline, skaitikliai) — **bendra**, kad skirtųsi tik tai, ką lyginame.

| Funkcija | C (CPU etalonas) | G | Pagrindimas / pastabos |
|---|---|---|---|
| **transport** | writer: `memcpy` iš iš anksto sugeneruoto rašto + antraštė; reader: kiekvieno žodžio palyginimas su formule (AVX2) arba CRC32C (SSE4.2 `crc32`). Naujas `src/mxlio/` `--impl cpu`. | esamas | „Paliesti kiekvieną baitą po kartą" — minimalus realus darbas. MXL kvietimų kaina (open/commit/get) abiem vienoda, nes antraštės visada host atmintyje [patikrinta: `docs/PayloadStorage.md:137`]. |
| **gen** (šaltinis) | `src/media/cpu/gen_cpu.cpp`: tas pats `f(run_seed, flow, index, x, fy)` SIMD. | esamas | Šaltinis nėra testuojama funkcija, bet C grandinėje jis ėda CPU. Todėl skelbiami abu skaičiai (§1.1 p.4). |
| **logo** | **Nuosavas C++ ant v210, ne FFmpeg** — `src/media/cpu/logo_cpu.cpp`: (1) `memcpy` įėjimo laukas → išėjimo grain'as (2 764 800 B); (2) blend tik logotipo stačiakampyje (384×216 kadre = 108 lauko eilutės), AVX2 unpack/blend/pack, ta pati sveikųjų formulė `(s·(255−a)+l·a+127)/255`. | `k_logo_batch` | FFmpeg `overlay` nedirba su v210: reikėtų v210→yuv422p10→overlay→v210, t. y. 3 pilni praėjimai per lauką — tai „tipinė", bet ne „gera" CPU funkcija. Pagrindinis etalonas — nuosavas (sąžiningas GPU atžvilgiu); FFmpeg filtrų grafas — **antrinis, informacinis** „tipinės integracijos" skaičius. Lūkestis: C ≈ 0,02–0,05 branduolio/srautui, t. y. panašiai kaip G (0,032 [išmatuota: užsakovo suvestinė, logo N=4]) [prielaida — T4]. Logo vienas pats GPU naudos **neparodys**; parodys grandinė ir H/G skirtumas. |
| **enc** | host flow → CPU weave v210→NV12/yuv420p (`src/media/cpu/convert_cpu.cpp`, ta pati lauką suvokianti formulė [F] §6.2) → libx264 tikras interlaced (MBAFF, tff) → AAC → TS. | weave GPU → `h264_nvenc` | x264 jau CPU (≈ 2 branduoliai/srautui [išmatuota: užsakovo suvestinė]); trūksta tik host įėjimo ir CPU konversijos (K2). **Išėjimai nėra ekvivalentūs**: NVENC šiame GPU neturi lauko kodavimo (`field_encoding=0`) [išmatuota: commit `92ed2cb`], todėl koduoja suaustą kadrą kaip progresyvų. Tai fiksuojama kaip apribojimas ir kompensuojama kokybės suderinimu (§1.4). |
| **dec** | libavcodec `h264` programinis (frame+slice threads; sweep `--threads 1,2,4,auto`) → CPU split NV12→v210 laukai ([F] §6.4 formulė) → host flow. | NVDEC → `k_split_nv12_batch` → cuda-linear | H.264 dekodavimas normatyviai bit-exact → **tas pats `ref_crc.json` tinka abiem** (NVDEC jau bit-exact [išmatuota: užsakovo suvestinė]). Stipriausia teisingumo savybė visame palyginime. |
| **chain** (= „pilnas flow") | TS/UDP → sw dec → host flow → CPU logo → host flow → x264+AAC → TS/UDP; audio flow pralaidžiamas (host abiem). | NVDEC → cuda → logo → cuda → NVENC | **Pagrindinis antraštinis skaičius.**264 ~7 Mbit/s išėjimas). |
| **chain-H** | — | NVDEC → D2H → host flow → H2D → logo → D2H → host flow → H2D → NVENC | Parodo, ką konkrečiai duoda #653: iš kiekvieno srauto pašalinamos 4 kopijos × 2,76 MB × 50 Hz = **553 MB/s** per PCIe ir host RAM [skaičiavimas iš [F] G1: 2 764 800 B/laukas]. |

Bit-exact reikalavimas C realizacijoms: `tests/unit/test_media_cpu.cpp` — `logo_cpu`, `convert_cpu`, `gen_cpu` prieš `media_ref` (0 skirtingų baitų, tie patys vektoriai kaip [F] §6.6). `media_ref` lieka nepriklausomu etalonu ir **nenaudojamas** kaip produkcinis C kelias (kitaip etalonas tikrintų pats save).

### 1.4 Koderių palyginimas vienodoje kokybėje

Problema: „N NVENC srautų prieš N x264 srautų" beprasmis, jei kokybė skiriasi. Planas [sprendimas]:

1. **Turinys:** ne sintetinis generatorius (jis klaidina koderius), o tikras 1080i50: (a) 60 s transliacijos ištrauka su judesiu, titrais, sportu; (b) atvira etaloninė seka. Apdorojama tik lokaliai, niekur nesiunčiama — hosted API čia nėra, todėl jautrumo klausimas nekyla; ar klipą galima dėti į viešą įrodymų paketą — D-6.
2. **R-D kreivės:** kiekvienai konfigūracijai 4 bitrate'ai {4; 6; 7,5; 10} Mbit/s CBR, ta pati GOP struktūra. Konfigūracijos: x264 {veryfast, medium} interlaced; NVENC {p4 ll, p4 hq, p7 hq}.
3. **Metrika laukais:** dekoduoti, `separatefields`, PSNR-Y ir SSIM kiekvienam laukui prieš šaltinio lauką (FFmpeg 6.1.1 yra WSL [patikrinta: `ffmpeg -version`]). VMAF interlaced turiniui nevaliduotas → tik informacinis ant bob-deinterlace [prielaida].
4. **BD-rate** (PSNR ir SSIM) NVENC prieš x264 → išvada forma: „NVENC p4-hq reikia +X % bitrate x264-veryfast kokybei".
5. **Suderintos poros talpai:** lyginamos konfigūracijos, kurių |BD-rate| ≤ 5 %; jei tokios nėra — lyginama **vienodame bitrate** ir šalia skelbiamas kokybės skirtumas dB. Talpa: N_nvenc (riba — 12 vienalaikių sesijų, 1 variklis [išmatuota: commit `92ed2cb`]) prieš N_x264.
6. Interlaced vėliavėlės: x264 `interlaced_frame=1,tff`; NVENC `interlaced_frame=0` — žinomas apribojimas [patikrinta: [F] §7.2], į PASS neįeina, bet ataskaitoje atskira eilutė „išėjimo ekvivalentiškumas: NE".

Nauji failai: `tests/perf/quality.py` (R-D taškai, BD-rate), `scripts/wsl/72-quality.sh`, `config/quality.json`.

### 1.5 Metrikos ir kaip jas gauti WSL2

| Metrika | Kaip | Būklė |
|---|---|---|
| CPU branduoliai/srautui (funkcija) | Σ(utime+stime)/trukmė pagal proceso kategoriją | yra [patikrinta: `tests/perf/capacity.py:31-37`] |
| CPU sistemos (Linux VM) | `/proc/stat` 1 Hz (pagauna kernel gijas, softirq UDP) | pridėti |
| CPU **Windows pusėje** | WSL2 GPU kelias dalį darbo atlieka host'e (dxgkrnl/WDDM), `/proc` to nemato → PowerShell `Get-Counter '\Processor Information(_Total)\% Processor Utility'` + `vmmem` procesas, 1 Hz, sinchronizuotas laiko žymomis. Be šito G kelio CPU būtų **nuvertintas**. | pridėti `scripts/win/host-sampler.ps1` [prielaida: host dalis reikšminga — T10 parodys] |
| RAM pralaidumas | `perf`, `pcm`, `turbostat` WSL nėra; PMU/uncore skaitikliai WSL2 neeksponuojami → **pakaitalai**: (1) *analitinė baitų apskaita* — kiekviena funkcija skaičiuoja į host atmintį parašytus/perskaitytus baitus (`host_bytes_rw` skaitiklis) → MB/s srautui, deterministiška; (2) *kanarėlė* — `tools/membw_canary.cpp`: viena prisegta gija sukasi `memcpy` 256 MB cikle ir kas 1 s rašo GB/s; kritimas nuo tuščios eigos = atminties magistralės konkurencija; (3) PSI `/proc/pressure/{cpu,memory,io}` (yra); (4) `/proc/meminfo` `Shmem` + `df /dev/shm` (host flow'ų tūris), `/proc/<pid>/smaps_rollup` Pss. | [patikrinta: `which perf pcm pcm-memory turbostat` — tuščia; `/proc/pressure/memory` yra] / pridėti |
| RAM pralaidumas (natyvus Linux) | `perf stat` uncore IMC / AMD, `pcm-memory`, resctrl MBM (`mbm_total_bytes` proceso grupei) — tikri skaičiai. | [prielaida: prieinamumas — T14] |
| GPU apkrova, VRAM, enc/dec | NVML WSL'e dalinis (utilization neveikia) [patikrinta: [V] §2, NVIDIA WSL gidas §4.1–4.2] → `nvidia-smi.exe dmon -s puvmet` **Windows pusėje** 1 Hz; VRAM — esamas `vram_mb()`. | pridėti į host sampler'į |
| Galia | RAPL zonų WSL nėra [patikrinta: `/sys/class/powercap` tuščias]; GPU `power.draw` per `nvidia-smi.exe`; CPU paketo galia — HWiNFO CSV žurnalas arba baterijos iškrovos greitis; **indikacinis**. Autoritetingai — sieninis matuoklis ant natyvaus serverio. | [prielaida] D-4 |
| Latencija | e2e, commit→visible p50/p99/p99.9/max, writer jitter | yra |

### 1.6 Kaip pateikti naudą

Lentelė kiekvienai funkcijai ir grandinei (generuoja `tests/perf/benefit.py` → `scripts/report.py` skyrius „Nauda"):

| Rodiklis | Formulė |
|---|---|
| Talpos santykis | `N*_G / N*_C` ir `N*_G / N*_H` (ta pati mašina) |
| Sutaupyti branduoliai | `(cores_C − cores_G) · N` ties N = N\*_C; taip pat „branduoliai/srautui" abiem |
| Sutaupytas host RAM srautas | `host_bytes_rw_C − host_bytes_rw_G` MB/s srautui (analitinis) + kanarėlės GB/s skirtumas |
| Sutaupytas RAM tūris | `Shmem` skirtumas: host flow 27,65 MB/flow tmpfs prieš 27,65 MB VRAM [patikrinta: [F] G2] |
| W/srautui | `(P_apkrova − P_tuščia) / N` — indikacinis |
| €/srautui | `config/cost.json`: serverio kaina, GPU kaina, €/kWh, amortizacija (visi įėjimai — **[prielaida]**, kol nepateiktos tikros kainos; D-4) |

**Perkeliamumas:** nešiojamo kompiuterio CPU/GPU santykis nėra serverio santykis. Todėl skelbiami *vienetiniai* dydžiai — branduoliai/srautui (C) ir srautai/GPU su ribojančiu veiksniu (G) — ir projekcija: `N_C = branduoliai / cores_per_stream`, `N_G = min(NVENC sesijos, VRAM, kernel'ių pralaidumas, likutinis CPU)`.

**VRAM biudžetas (8 GB)** grandinei G [skaičiavimas]: 2 video flow'ai/srautui × 27,65 MB = 55,3 MB [patikrinta: [F] G2]; NVDEC sesija ~0,1–0,2 GB, NVENC sesija ~0,1–0,25 GB [prielaida — matuojama ΔVRAM]. Blogiausiu atveju ~0,45 GB/srautui → N=12: ~5,4 GB + ~0,5 GB kontekstas ≈ **5,9 GB < 8 GB** (Windows darbalaukis papildomai ~0,5–1 GB). Laukiama riba — 12 NVENC sesijų, ne VRAM.

**Lūkestis, kurį matavimas turi patvirtinti arba paneigti** [prielaida — T11]: C grandinė ≈ 2 (x264) + ~0,4 (sw dec) + ~0,2 (konversijos, logo) ≈ 2,6 branduolio/srautui → N\*_C ≈ 3–4 ant 12 loginių CPU; N\*_G ≤ 12 → santykis ~3×.

---

## 2. Spragų analizė: cuda-linear prieš host MXL

Statusai: **VEIKIA** (patikrinta kodu ir mūsų testu), **KODAS OK / NETESTUOTA** (pagal kodą turi veikti, testo nėra), **TRŪKSTA**, **SKIRIASI** (elgsena kitokia nei host).

| # | Sritis | Statusas | Įrodymas | Kaip testuojame | Pataisa? |
|---|---|---|---|---|---|
| S1 | Diskretūs grain'ai, pilnas commit/skaitymas | VEIKIA | [patikrinta: `PosixDiscreteFlowWriter.cpp:104-118,208-238`; `lib/tests/cuda/test_cuda_payload.cpp:229,313`] [išmatuota: savitikra 16/16, funkciniai 7 PASS — commit `483a822`, `135ca92`] | esama | ne |
| S2 | Slice'ai: dalinis commit, `validSlices`, progresyvus skaitymas | KODAS OK / NETESTUOTA | K4. Commit su `validSlices < totalSlices` palieka grain'ą atidarytą ir žadina reader'ius (`PosixDiscreteFlowWriter.cpp:225-233`); pakartotinis `OpenGrainSlot` tam pačiam indeksui išsaugo antraštę (`:170-175`); reader'is lygina `validSlices >= min(minValidSlices, totalSlices)` (`PosixDiscreteFlowReader.cpp:234`). | naujas: writer GPU'je rašo po k eilučių, `cudaStreamSynchronize`, commit; reader `GetGrainSlot(minValidSlices=k·i)`; tikrinama, kad matomos eilutės teisingos, o latencija iki pirmo slice < pilno grain'o | **testai + doc** (U3). GPU specifika: prieš *kiekvieną* dalinį commit'ą įrenginio darbas tiems slice'ams turi būti baigtas — dokumentuota tik pilnam grain'ui [patikrinta: `docs/PluggablePayloads.md:94`]. Ateities spraga: vėliavėlė layout'e „palaiko dalinius grain'us" [patikrinta: `docs/PayloadStorage.md` „What the examples show" p.3]. |
| S3 | Dalinio skaitymo latencija | NEŽINOMA | WSL2 kiekviena GPU sinchronizacija ~1,5 ms [patikrinta: komentaras `src/media/media_kernels.hpp:2-3`; kilmė — mūsų matavimas] → slice'ai GPU'je WSL gali būti brangesni už pilną lauką | matuoti S2 teste: p50/p99 „pirmas slice matomas" host prieš cuda | ne (matavimas) |
| S4 | Blokuojantis / neblokuojantis skaitymas | VEIKIA | `GetGrainSlot(timeoutNs)`; `timeoutNs=0` — negrįžta laukęs; timeout → `MXL_ERR_OUT_OF_RANGE_TOO_EARLY`, ne `TIMEOUT` [patikrinta: `flow.h` `mxlFlowReaderGetGrainSlot` doc; `PosixDiscreteFlowReader.cpp:266-288`]. Atskiros `…SlotNonBlocking` funkcijos nėra — nereikia. | esama (`VideoIn::get(index, timeout_ns)`) | ne |
| S5 | Sync grupės (`mxlFlowSynchronizationGroup*`) su device flow'ais, mišrios su audio | KODAS OK / NETESTUOTA | K5 | (1) upstream stiliaus Catch2 testas per `test-file` backend'ą (be GPU, eina CI); (2) mūsų įrankyje: grupė {cuda video, host audio}, `WaitForDataAt` kiekvienam laukui, 0 neatitikimų per 10 min | **testas** (U2); pataisa tik jei kris |
| S6 | Indeksų/laiko matematika | VEIKIA (identiška host) | antraštės, head indeksas, futex — visada host shared memory [patikrinta: `docs/PayloadStorage.md:137`; `PosixDiscreteFlowWriter.cpp:217-233`] | esama; laiko šaltinio teisingumas — atskiras dokumentas; žr. `mxl#609` | ne |
| S7 | Flow gyvavimo ciklas: sukūrimas atomiškas | VEIKIA | backend'as ruošia saugyklą laikinajame kataloge prieš rename [patikrinta: `docs/PayloadStorage.md` „Publish complete flows"; `FlowManager.cpp:350`] | esama | ne |
| S8 | Vienas writer'is | SKIRIASI | backend flow — tiksliai vienas writer'is, antras → `MXL_ERR_CONFLICT`; host flow leidžia kelis [patikrinta: `FlowManager.cpp:431-435`; `flow.h` `mxlCreateFlowWriter` doc] | esama F6 | ne; dokumentuota |
| S9 | Writer'io avarija / paleidimas iš naujo | SKIRIASI | po `kill -9` naujas writer'is gauna CONFLICT, kol GC nepašalina katalogo [patikrinta: [V] R7]; jau prisijungę reader'iai atmintį išlaiko, **nauji reader'iai prisijungti negali** — `payload.sock` aptarnauja writer'io gija [patikrinta: `CudaLinearBackend.cpp:638,760-769`]. Host flow: naujas reader'is failus atsidaro ir po writer'io mirties. | F6 išplėsti: reader'is, startuojantis po writer'io mirties → laukiamas statusas užfiksuojamas | doc (U3); elgsena priimtina |
| S10 | GC | VEIKIA | `flock(LOCK_EX\|LOCK_NB)` ant data failo, `remove_all` [patikrinta: `Instance.cpp:404-411`]; VRAM atlaisvinama, kai visi importai uždaryti | esama; pridėti VRAM grįžimo patikrą po GC | ne |
| S11 | `mxl-info` / įrankių matomumas | TRŪKSTA | K6. Metaduomenys skaitomi be GPU (reader'io atidarymas nieko nemap'ina) [patikrinta: `flow.h` `mxlCreateFlowReader` doc; `test_grain_storage.cpp:494`], bet storage tipas nerodomas | `mxl-info` ant mūsų domeno prieš/po | **pataisa** (U5) |
| S12 | Interlaced | VEIKIA bibliotekoje; įrankiuose TRŪKSTA (bendrai) | K7; mūsų visi matavimai — 1080i50 ant cuda-linear | esama | ne CUDA pataisa → REQ (U8) |
| S13 | Audio / continuous flow'ai | TRŪKSTA (sąmoningai) | `Instance.cpp:324-327` „Continuous flows keep their payload in host memory"; `docs/PayloadStorage.md:115`; storage API continuous flow'ui → `INVALID_FLOW_READER` (`test_grain_storage.cpp:294`). #594: „later support for continuous audio" [patikrinta] | audio lieka host; A/V sync tikrinama S5 | ne dabar. 48 kHz stereo float32 = 0,38 MB/s — GPU naudos nėra |
| S14 | Data / ANC | TRŪKSTA (sąmoningai) | `CudaLinearBackend.cpp:1213-1216` (tik `MXL_DATA_FORMAT_VIDEO`); #594 non-goal [patikrinta] | — | ne |
| S15 | Event flow'ai | TRŪKSTA (sąmoningai) | `Instance.cpp:251-254` [patikrinta] | — | ne |
| S16 | Fabrics (tarp mazgų) | TRŪKSTA | `lib/fabrics/ofi/src/internal/Region.cpp:161` atmeta v2 flow'us su `UNSUPPORTED_OPERATION`; #654 (FI_HMEM) atviras, `dirty`, neatnaujintas nuo 2026-09-27, test planas nepažymėtas [patikrinta: api `pulls/654`]. Atviras dizaino klausimas: antraštė host'e + payload GPU = du nuotoliniai rašymai per slice [patikrinta: #594 jonasohland 2026-10-22] | negalime (nėra RDMA NIC) | ne mūsų apimtis |
| S17 | Keli GPU (peer access) | KODAS YRA / NETESTUOTA niekieno | `CudaLinearBackend.cpp:1142-1158`; upstream testas SKIP be 2 GPU (`test_cuda_payload.cpp:377-387`); PR test plane nepažymėta [patikrinta: api `pulls/653` body] | reikia mašinos su 2 NVIDIA GPU | ne; **mūsų testo rezultatas būtų vertingas indėlis** |
| S18 | Konteineriai | VEIKIA su išlyga | FD per `payload.sock` tmpfs domene, be bendro IPC/PID namespace [patikrinta: `examples/cuda-cross-container-test.sh:5-8`]; „šiam host'ui reikia `--privileged`" (`:13-14`); mūsų paleidimas kaip root — `evidence/…/20-upstream/ctest-container-as-root.log` [išmatuota] | esama + pakartoti be `--privileged` ant natyvaus Linux | (a) U1 |
| S19 | Saugumas: FD perdavimas | VEIKIA | `SO_PEERCRED`; FD gauna tik uid = savininkas, arba gid su `S_IWGRP`, arba `S_IWOTH`; socket'o režimas seka katalogo skaitymo teises [patikrinta: `CudaLinearBackend.cpp:719-749`] | naujas: kitas uid be teisių → atsisakoma; su grupės teise → pavyksta | testas (į U4 rinkinį) |
| S20 | Saugumas: reader'is gali rašyti | SKIRIASI | K13 | testas dokumentuoja elgseną | ne pataisa; **klausimas TSC** |
| S21 | Suderinamumas su v1.1 reader'iais | VEIKIA | flow data version 2 → seni SDK atsisako atidaryti [patikrinta: `Flow.hpp:21,27` per [V]; `docs/PayloadStorage.md` „It uses flow data version 2"] | — | ne |
| S22 | Programų prieiga prie storage aprašo | TRŪKSTA | K14 | — | **pataisa** (U6) |
| S23 | Ring'o istorijos trukmė | SKIRIASI kaina | 200 ms domeno lygio → 27,65 MB VRAM/flow; `mxl#582` siūlo peržiūrėti (GPU RAM brangesnė) [patikrinta: api `issues/582`] | mūsų ΔVRAM matavimai | ne; **duomenys į #582** |

**Išvada dėl „lygu host MXL visomis prasmėmis":** laiko, indeksų, sinchronizacijos ir ring'o protokolas yra *tas pats kodas* host ir device flow'ams — skiriasi tik tai, kur guli payload'as (S6). Elgsenos skirtumai, kuriuos reikia žinoti: S8, S9, S20. Nepadengta testais: S2, S5, S17. Sąmoningai už apimties: S13–S16.

---

## 3. Upstream pataisų planas

### 3.1 Proceso faktai

- C++20, be plėtinių; Catch2; `.clang-format`, `.clang-tidy`; **Doxygen kiekvienam naujam simboliui, įskaitant testų pagalbininkus**; `auto name = Type{…}` inicializacija; SPDX antraštės naujuose failuose; DCO `Signed-off-by` su organizacijos domeno el. paštu. [patikrinta: `CONTRIBUTING.md` (Coding standards, Variable initialization, Doxygen, Commit Sign-Off, Copyright Notices)]
- 2 Maintainer patvirtinimai; privalomi CI: DCO + Ubuntu 24.04 {arm64, x86_64} × {Clang, GCC} Release. **CI neturi GPU** → viskas, kas turi būti vykdoma CI, eina per `test-file` backend'ą. [patikrinta: `CONTRIBUTING.md` „Code Review and Required Approvals"; `.github/workflows/*` — nėra `cuda`; `docs/PayloadStorage.md:127`]
- „Significant" pakeitimams — pirma reikalavimo bilietas `dmf-mxl/mxl-requirements` ir aptarimas RC. [patikrinta: `CONTRIBUTING.md` „How to Contribute"; `AGENTS.md` „Significant changes"]
- DI pagalba leidžiama, bet **autorius privalo suprasti ir pagrįsti kiekvieną eilutę; commit'ą inicijuoja žmogus, ne agentas**. Dokumentacijos stilius: trumpi sakiniai, be reklaminių žodžių, be em-dash, be „trijų punktų iš įpročio". [patikrinta: `CONTRIBUTING.md`; `AGENTS.md` „Documentation", „Commit sign-off"] ⇒ įgyvendintojas paruošia pataisą ir testus mūsų repo `upstream/` kataloge; **Deimantas peržiūri, commit'ina su `--signoff` ir kelia pats**. Jokių DI `Co-Authored-By` upstream commit'uose [prielaida: šaka `845-…-not-ai-agents` egzistuoja, turinio neskaičiau].
- PR šablonas: „Details" + „Backport requirements". [patikrinta: `.github/pull_request_template.md`]

### 3.2 Kandidatų patikra ir tvarka

Tvarka — nuo pigiausio ir mažiausiai ginčytino. Kelias kiekvienam: **K** = komentaras #653 su pataisa ir įrodymu (K10 precedentas); **PR** = atskiras PR į `dmf-mxl/mxl` po to, kai atitinkama serijos dalis merge'inta; **REQ** = pirma reikalavimas.

| ID | Kandidatas | Galioja? | Failai | Testo dizainas (pirma testas) | Tikėtinos recenzentų pastabos | Įrodymų paketas | Kelias |
|---|---|---|---|---|---|---|---|
| **U1** | (a) `sudo -n docker info` | TAIP (K8) | `examples/cuda-cross-container-test.sh:26` | rankinis: ne-root be NOPASSWD, stdin = terminalas → prieš: kabo; po: exit 2 ir Catch2 `SKIP("Docker is not available.")` (`test_cuda_payload.cpp:297-310`) | jokių | komandų išvestis prieš/po | K |
| **U2** | (b) sync grupės su backend flow'ais | TAIP — trūksta testo (K5) | `lib/tests/test_flow_sync_groups.cpp` (+ `TEST_BACKEND_OPTIONS` iš `test_grain_storage.cpp`) | „Synchronization group : flows with a payload backend": grupė {backend video, host video, host audio}; prieš commit → `TOO_EARLY`; po commit visiems → `OK`; `AddPartialGrainReader(minValidSlices=k)` ant backend flow → `OK` po dalinio commit'o. Eina CI per `test-file`. CUDA variantas `lib/tests/cuda/` su `SKIP` be GPU. | „dubliuoja esamą testą" → parametrizuoti esamą `Repeated waits` per storage; laiko testų trapumas → jokių `sleep`, tik deterministiniai indeksai | mūsų 10 min paleidimas: {cuda video + host audio} grupė, 30 000 laukų, 0 neatitikimų (T7) | K, vėliau PR |
| **U3** | (c) daliniai grain'ai ant backend flow'ų | TAIP — API yra, trūksta testų ir sutarties (K4) | `lib/tests/test_grain_storage.cpp`; `docs/PluggablePayloads.md` (1 sakinys: dalinio commit'o sutartis); CUDA: `lib/tests/cuda/test_cuda_payload.cpp` | „Grain storage : Partial grains with a payload backend": `OpenGrainSlot` → rašyti k slice'ų → commit `validSlices=k` → `GetGrainSlot(minValidSlices=k, 0)` = OK, `(k+1, 0)` = `TOO_EARLY` → pakartotinai atidaryti tą patį indeksą (antraštė išsaugota) → commit pilną → OK. Veidrodis esamiems `Video Flow : Slices` (`test_flows.cpp:537`) ir `validSlices on grain open` (`:1299`). | „ar daliniai grain'ai turi būti garantuojami visiems storage tipams?" → ne; nurodyti `PayloadStorage.md` p.3 (vėliavėlė layout'e) kaip ateities darbą, nesiūlyti jos dabar | S3 latencijos matavimas host prieš cuda | K, vėliau PR |
| **U4** | (f) storage atitikties (conformance) rinkinys | TAIP — didžiausia vertė | naujas `lib/tests/StorageConformance.hpp` (šabloniniai atvejai virš `StorageFixture { options; write(slot, bytes); read(slot) }`), `lib/tests/test_storage_conformance.cpp` (host, test-file), `lib/tests/cuda/test_cuda_conformance.cpp` (cuda-linear, SKIP be GPU) | tie patys atvejai visiems storage per descriptor API: create/release; ring langas ir `TOO_LATE`; timeout → `TOO_EARLY`; šuolis pirmyn → `INVALID` vėliavėlė (`test_flows.cpp:1227` veidrodis); slice'ai (U3); futex žadinimas (`test_flows_timing.cpp:18` veidrodis); skaitymas atidaryto rašymo metu (`test_read_write_conflict.cpp:33`); writer'io atkūrimas → `FLOW_INVALID`; antras writer'is → host OK / backend `CONFLICT` (aiškiai užkoduotas skirtumas S8); reader'is be backend'o skaito metaduomenis; S19 teisės | apimtis ir CI laikas; „kodėl ne išplėsti esamus testus" → pasiūlyti kaip refaktoringą be dubliavimo; turi būti REQ arba #594 priėmimo kriterijus („tests for both allocation paths" jau yra #594 kriterijuose [patikrinta]) | lentelė „atvejis × storage" su PASS visur + mūsų selftest 16/16 su gedimų injekcijomis | **REQ/RC aptarimas → PR** (po serijos descriptor API dalies) |
| **U5** | (d) `mxl-info` rodo storage tipą | TAIP (K6) | `tools/mxl-info/main.cpp`; `docs/Tools.md` | eilutė „Payload storage: host-pointer \| com.nvidia.cuda-device-pointer \| <backend> (not available in this process)"; testas: `mxl-info` ant `test-file` flow → eilutė yra | iš kur imti vardą, kai backend'as neužsikrauna: `GetStorageLayout` tada grąžina `UNSUPPORTED` [patikrinta: `docs/PayloadStorage.md` „A reader opens such a flow…"], o `payload.json` — vidinis failas → gali reikėti mažos API papildos; sąsaja su `mxl-requirements#193` (mxl-info latencija, rochonma) — nekonfliktuoti | ekrano išvestis prieš/po ant mūsų domeno su 8 cuda flow'ais | REQ komentaras (#190 arba #193) → PR |
| **U6** | (h) instaliuoti `cuda-linear` storage aprašo header'į | TAIP (K14) | `lib/payload-backends/cuda-linear/CMakeLists.txt` (`install(FILES …)`), galimai perkelti į `lib/include/mxl/` | `test_pkgconfig_c_link.sh` stiliaus: išorinis C failas `#include` iš install prefikso ir kompiliuojasi | „backend sąsaja vidinė" — taip, bet **storage tipo aprašas** yra vieša sutartis pagal `PayloadStorage.md` („applications read them through definitions that come with the backend") [patikrinta]; pavadinimas ir vieta — maintainer'ių sprendimas | mūsų `src/mxlio/slots.hpp` kaip pavyzdys, kad be header'io tenka kopijuoti | K (klausimas autoriui) |
| **U7** | (g) laiko reikalavimų dokumentacija | DALINAI — `mxl#609` atviras, priskirtas KimonHoffmann [patikrinta] | `docs/Timing.md` skyrius „Host prerequisites" (TAI offset, patikros komanda); device flow'ams — sakinys, kad laiko elgsena identiška | doc-only | dubliavimas su #609 sprendimu → pirma komentaras #609 su mūsų WSL2 stebėjimais (laikrodžio šuoliai, TAI offset) | iš atskiro laiko dokumento | K į #609; PR tik jei paprašys |
| **U8** | (e) interlaced `mxl-gst` įrankiuose | NE kaip CUDA pataisa (K7) | — | — | apimtis didelė (visi įrankiai, GStreamer caps, laukų poravimas) | mūsų 1080i50 ant cuda-linear rezultatai rodo, kad biblioteka paruošta | **REQ** „interlaced support in tools" — tik jei LRT to reikia |
| **U9** | reader'io READWRITE (K13), nauji reader'iai po writer'io mirties (S9) | radiniai, ne pataisos | — | U4 testai dokumentuoja | — | — | klausimai #653 / #594 |

Papildomi indėliai be kodo: ΔVRAM ir ring'o kainos duomenys → `mxl#582`; 2-GPU peer testo rezultatas → #653.

### 3.3 Ko nesiūlome

Naujo viešo API (pvz., „supports partial grains" vėliavėlės), audio/ANC device storage, Fabrics pakeitimų — tai „significant", jau turi savininkus (#594, #654) ir mes neturime geležies tam patikrinti.

---

## 4. Įrodymų paketas MXL Requirements Council / TSC

Kas įtikina (pagal tai, ko TSC klausė #594 ir kas suveikė #653 komentaruose [patikrinta: K10, #594 komentarai]): atkuriamumas, neigiami rezultatai šalia teigiamų, palyginimas su host keliu tame pačiame teste.

**Struktūra** (anglų kalba; generuoja `scripts/report.py --lang en`):

1. **Summary** — 10 eilučių: kas testuota (SHA), kur, 5 skaičiai.
2. **Environment** — `manifest.json` (jau yra `evidence/…/manifest.json`): GPU, driver, CUDA, kernel, CPU, MXL SHA, mūsų SHA, kompiliatoriai, laiko šaltinis ir TAI offset. Aiškiai: „WSL2" arba „native Linux".
3. **Correctness (zero-error proofs)**
   - upstream testai: PASS/SKIP sąrašas (SKIP ≠ PASS) [yra: `20-upstream/`];
   - kiekvieno grain'o kiekvieno žodžio patikra: N grain'ų, 0 klaidų, su **savitikra** — kiekviena gedimo klasė injekuojama ir privalo būti aptikta (16/16) [išmatuota: commit `483a822`];
   - bit-exact GPU kernel'iai prieš nepriklausomus CPU etalonus; C ir G išėjimų tapatumas (logo, dec);
   - sanitizer'iai (memcheck/racecheck/synccheck) [yra: `50-sanitizers/`];
   - **soak ≥ 2 h** (dabar NOT_RUN — privaloma prieš teikiant);
   - atitikties lentelė „atvejis × {host, test-file, cuda-linear}" (U4).
4. **Timing and synchronisation** — indeksas = laikas (barkodas), sync grupės {video GPU + audio host}, A/V poslinkis, laikrodžio šuolių elgsena; nuoroda į laiko dokumentą.
5. **Capacity** — N\* lentelė C/H/G kiekvienai funkcijai ir grandinei; ribojantis veiksnys kiekvienam; `N*_G/N*_H` kaip #653 naudos matas.
6. **Latency** — commit→visible ir e2e p50/p99/p99.9; šalia danbouzaSWS natyvaus Linux skaičių (110–126 µs p50) kryžminei patikrai [patikrinta: #653 komentaras 2026-10-16].
7. **Resources** — branduoliai/srautui, host RAM srautas (analitinis + kanarėlė), VRAM/flow, galia (pažymėta „indicative").
8. **Findings for the PR** — U1–U9, kiekvienas su atkūrimo komanda.
9. **Limitations** — WSL2, vienas GPU, nėra RDMA, NVENC be lauko kodavimo, sintetinis/realus turinys.
10. **Reproduce** — viena komanda (`scripts/wsl/run-all.sh`), trukmė, reikalavimai; žali duomenys (`stages.jsonl`, per-proceso JSON santraukos, CSV) pridedami archyve.

**Kur teikti** [sprendimas]: (1) komentaras #653 — teisingumo + latencijos + U1–U3 radiniai; (2) komentaras `mxl-requirements#190` — talpos/naudos santrauka su nuoroda į paketą; (3) pristatymas savaitiniame RC/TSC susitikime (nemokami, registracija per EBU MXL grupę [patikrinta: `CONTRIBUTING.md` „How to engage"]).

**Būtina sąlyga patikimumui:** bent teisingumo ir latencijos dalys pakartotos ant **natyvaus Linux**. WSL2 rezultatai TSC akyse yra „veikia net ten", bet ne talpos įrodymas (K12).

---

## 5. Rizikos

| Rizika | Poveikis | Mažinimas |
|---|---|---|
| WSL2 jitter'is riboja N\* abiem keliams vienodai (K12) | santykis G/C iškreiptas | natyvus Linux (T14); WSL skaičiai žymimi „indicative" |
| Windows pusės CPU nematomas | G kelias atrodo pigesnis nei yra | host sampler'is (T10) |
| CPU etalonas per silpnas / per stiprus | ginčijamas rezultatas | SIMD + gijų sweep + viešas kodas; antrinis FFmpeg etalonas |
| #653 keičiasi (serija #746…) | pataisos nebetinka | U1–U3 teikti kaip komentarus dabar; PR — tik ant merge'intos bazės; prieš kiekvieną — `git fetch` ir pakartotinė patikra |
| Logo CPU pigesnis už GPU | „nauda" atrodo silpna | skelbti atvirai; antraštė — grandinė ir H→G |

---

## 6. Užduočių sąrašas (kiekviena su priėmimo patikra)

Tvarka privaloma. **T1 pradedama tik pasibaigus dabartinei talpos kampanijai.** Kiekviena užduotis baigiama paleistos komandos išvestimi, ne teiginiu.

| # | Užduotis | Failai | Priėmimo patikra |
|---|---|---|---|
| T1 | `Opts.impl/storage/threads`, CLI, `VideoOut` host šaka, backend sąsajos atskyrimas nuo tick variklio (be elgsenos pokyčio G keliui) | `src/func/{func_common.hpp,main_func.cpp,flows.*,logo.cpp,enc.cpp,dec.cpp,gen.cpp}` | esami `tests/functional` ir `tests/gpu` — tie patys PASS kaip prieš; `logo --impl gpu --storage cuda-linear` N=4 60 s: 0 klaidų, CPU/srautui ±10 % nuo ankstesnio |
| T2 | CPU SIMD branduoliai + unit testai prieš `media_ref` | `src/media/cpu/{logo_cpu,convert_cpu,gen_cpu}.{hpp,cpp}`, `tests/unit/test_media_cpu.cpp`, `cmake/media.cmake` | visi [F] §6.6 vektoriai; 0 skirtingų baitų 10⁴ atsitiktinių laukų; veikia su `-mno-avx2` (skaliarinis kelias) |
| T3 | `gen --impl cpu`, `verify` ant host flow'ų | `src/func/{gen,verify}.cpp` | `verify --mode gen` N=4 host: 0 `mismatch_words`; procesas nematomas `nvidia-smi --query-compute-apps` |
| T4 | `logo --impl cpu` | `src/func/logo.cpp` (`CpuLogo`) | `verify --mode logo` 0 klaidų; C ir G išėjimų CRC tam pačiam įėjimui sutampa; `--inject-every` aptinkamas |
| T5 | `dec --impl cpu` (libavcodec sw + CPU split) | `src/func/dec.cpp` | `verify --mode dec --ref ref_crc.json` 0 klaidų tuo pačiu etalonu kaip NVDEC |
| T6 | `enc --impl cpu` (host flow → CPU weave → x264) | `src/func/enc.cpp` | F-ENC ([F] §7.2): lauko tvarka, `interlaced_frame=1`; PSNR/SSIM ≥ slenksčių |
| T7 | Sync grupių testas mūsų įrankyje: {cuda video, host audio} + `AddPartialGrainReader` | `src/func/verify.cpp` (`--mode sync`), `tests/functional/functional.py` | 10 min, N=4: `WaitForDataAt` OK kiekvienam laukui, 0 neatitikimų; neigiamas testas (sustabdytas vienas flow) → `TOO_EARLY` |
| T8 | Slice'ų testas ir latencija host prieš cuda | `src/mxlio/verify_{writer,reader}.cpp` (`--slices k`) | 0 klaidų; lentelė „pirmas slice matomas" p50/p99 abiem storage |
| T9 | Transportas `--impl cpu`; A/V poslinkio matuoklis TS lygyje | `src/mxlio/*`, `tests/perf/ts_sink.py` | selftest 16/16 ir C variantui; A/V poslinkis raportuojamas mėginiais |
| T10 | Harness: `--impl/--storage`, sistemos CPU, kanarėlė, PSI, Shmem, Windows host sampler'is, `config/bench.json` | `tests/perf/capacity.py`, `tools/membw_canary.cpp`, `scripts/win/host-sampler.ps1`, `scripts/wsl/71-capacity-cpu.sh` | vienos pakopos JSON turi visus naujus laukus; kanarėlė tuščioje eigoje ±3 % per 3 paleidimus; host ir WSL laiko žymos sutampa ≤ 1 s |
| T11 | Kampanija C/H/G: transport, logo, enc, dec, chain; 3 pakartojimai, pakaitomis; 10 min patvirtinimas ties N\*; soak 2 h | `scripts/wsl/70-capacity.sh` išplėtimas | `capacity-*.json` visiems; N\* sklaida tarp pakartojimų ≤ 1 pakopa; soak PASS |
| T12 | Kokybės suderinimas (R-D, BD-rate) | `tests/perf/quality.py`, `scripts/wsl/72-quality.sh`, `config/quality.json` | BD-rate lentelė; žinomas testas: tas pats koderis prieš save → BD-rate 0 ± 0,5 % |
| T13 | Naudos skaičiavimas ir ataskaita (LT + EN) | `tests/perf/benefit.py`, `scripts/report.py`, `config/cost.json` | ataskaitos skyrius „Nauda" sugeneruotas iš žalių JSON; kiekvienas skaičius atsekamas iki failo |
| T14 | Pakartojimas ant natyvaus Linux: T11 dalis + `perf`/resctrl RAM pralaidumas | — | tas pats `run-all.sh` praeina; skirtumų lentelė WSL prieš natyvų |
| U1 | Pataisa + įrodymas | `upstream/U1-sudo-n/` (patch + prieš/po išvestis) | prieš: kabo; po: exit 2 per < 1 s |
| U2 | Catch2 testas (test-file) + CUDA variantas | `upstream/U2-sync-groups/` | `mxl-tests "[mxl sync groups]"` PASS su Clang ir GCC; `clang-format --dry-run -Werror` švaru; Doxygen be įspėjimų naujiems simboliams |
| U3 | Catch2 testai + doc sakinys | `upstream/U3-partial-grains/` | `mxl-tests "[grain storage]"` PASS; CUDA variantas PASS ant mūsų GPU |
| U4 | Atitikties rinkinys (po RC pritarimo) | `upstream/U4-conformance/` | lentelė „atvejis × storage" — visur PASS arba aiškiai užkoduotas skirtumas; CI laikas < 30 s |
| U5, U6 | Pataisos | `upstream/U5-mxl-info/`, `upstream/U6-install-header/` | U5: eilutė matoma host ir test-file flow'ams; U6: išorinis failas kompiliuojasi iš install prefikso |
| U-teik | Deimantas: peržiūra, `git commit --signoff`, komentarai/PR | — | nuorodos į komentarus/PR įrašytos į šį dokumentą |

Po T1–T6 ir po T11 — `/code-review` (rizikingi: tick variklio refaktoringas, CPU branduoliai). Po T13 — `code-simplifier`.

---

