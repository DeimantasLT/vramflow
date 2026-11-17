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
