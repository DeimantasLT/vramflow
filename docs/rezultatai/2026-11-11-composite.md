# composite — talpa N × K (C / H / G)

Data: 2026-11-11 (kampanija 13:15–14:33 vietos laiku). Užduotis 10, alfa kompozitorius. Žaliavos: `evidence/compare-composite/`
(git ignoruojama; lentelių eilutės nurodo `capacity-composite-k<K>-<V>.all.json`), aplinka — `evidence/compare-composite.env.txt`,
žurnalas — `evidence/compare-composite/compare.log`, praleistos pakopos — `skipped-after-fails.jsonl`.

## Išvados trumpai

- **Didžiausias N, kai visi mažesni PASS** (N* tol / N* griežtai): C — K=1: 16 / 2, K=2: 8 / 4, K=4: 4 / 4;
  H — K=1: 4 / 1, K=2: 4 / 0, K=4: 4 / 2; **G (po pataisos 39dcb80) — K=1: 4 / 4, K=2: 2 / 0, K=4: 0 / 0**.
- **Didžiausias praėjęs N (nebūtinai su visais mažesniais PASS), tol / griežtai:** C — K=1: 16 / 8, K=2: 8 / 4, K=4: 4 / 4;
  H — K=1: 4 / 4, K=2: 4 / 2, K=4: 4 / 2; **G — K=1: 32 / 32, K=2: 32 / nė vienas, K=4: 16 / 16** (K=4 N=32 nevykdyta:
  VRAM riba). G pakopų FAIL pasitaiko žemuose ir viduriniuose N (K=1 N=8, K=2 N=4, K=4 N=1), nors didesnės N praeina, todėl
  N* (visi mažesni PASS) čia pesimistiškas. **Hipotezė (neišbandyta):** FAIL/PASS kaitą lemia GPU dažnio šuoliai (SM dažnio
  vidurkis visose G pakopose 210–690 MHz) ir WSL2 drebėjimas; matavimai pavieniai (po vieną kiekvienai pakopai), be užfiksuotų
  dažnių ir be pakartojimų, todėl patikimai atskirti dažnio efektą nuo tikros talpos ribos šie duomenys neleidžia.
- **Pirmasis G matavimas buvo negaliojantis.** Pirmoje kampanijoje (iki `39dcb80`) `verify` G pakopose rado neatitikimų:
  K=1 N=2: 60, N=4: 120; K=2 N=2: 60; K=4 N=2: 60 (kitose G pakopose 0). Priežastis — `gfxgen` klaida: pageable `cudaMemcpy`
  H2D grįždavo prieš pasibaigiant DMA, o biblioteka konvertuodavo pusiau įkeltą kadrą (neatitiko tik flow n=1 logo
  stačiakampis, lyginiai laukai). C ir H (host kelias) neturėjo šios klaidos. Klaida ištaisyta `39dcb80`; G pakartota, visose
  pakartoto matavimo pakopose **0 teisingumo klaidų**. Seni G skaičiai pakeisti; C/H palikti kaip išmatuota.
- Žemuose N GPU dažnis dažnai likęs tuščios eigos (210 MHz, žr. aplinką); tai stebėjimas, o jo poveikis PASS/FAIL — hipotezė.
- composite proceso CPU branduoliai/srautui **PASS_tol pakopose** (K=1 / K=2 / K=4): C 0,086–0,149 / 0,249–0,348 / 0,561–0,671;
  H 0,064–0,305 / 0,081–0,337 / 0,144–0,379; G 0,007–0,060 / 0,011–0,082 / 0,020–0,102. Visose vykdytose pakopose (įskaitant
  FAIL): C 0,079–0,671, H 0,042–0,379, G 0,007–0,142.
- **Patikros aprėptis (R21):** lentelėse „patikra %“ = patikrinta mėginių / laukiama. Aprėptis < 50 % tik FAIL pakopose:
  K=2 C N=32 (34 %), K=2 H N=16 (43 %), K=4 C N=16 (30 %), K=4 H N=16 (7 %); nė viena PASS pakopa nepatikrinta mažiau nei per pusę,
  G pakopose aprėptis 71–101 % (mažiausia K=2 N=32: 71 %, K=1 N=32: 94 %, kitur ≥ 97 %). Aprėptis perskaičiuota iš `summary-verify-composite.json` (matavimo metu harness jos neįrašė).

## Metodas

- Harness: `tests/perf/capacity.py --func composite --layers K --n …` (kiekviena pakopa: gen + gfxgen `--mode pace` → `composite`
  → `verify --mode composite --every 25`; 60 s matavimas po 10 s apšilimo). Kampanija `scripts/wsl/71-compare.sh`, variantai
  pakaitomis C, H, G kiekvienai N; K išorinis ciklas. Komanda:
  `SECONDS_PER_LEVEL=60 WARMUP=10 LAYERS=1,2,4 STOP_FAILS=2 OUT_FORMAT=v210 SOURCE=fair bash scripts/wsl/71-compare.sh /root/evidence/compare-composite composite 1,2,4,8,16,32`.
- **Geometrija (K):** K=1 → `logo`; K=2 → `logo,full`; K=4 → `logo,full,full,full` (spec. §4.7). Fiksuota: logo (384×216 v210a)
  `--layer-xy 1488,64` `--opacity 1.0`; kiekvienas `full` (1920×1080 v210a) `0,0` `0.8`. Bazė `gen` (1080i50 v210), išėjimas
  `v210` (programos išėjimas; v210a kampanija nevykdyta).
- **PASS(N)** kaip logo: 0 praradimų ir 0 teisingumo klaidų, RT ≥ 0,999, visi procesai exit 0 (griežtai); PASS tol — 0 teisingumo
  klaidų ir praradimai ≤ 0,2 %. N* = didžiausias N, kai visi mažesni PASS.
- **Funkcijos CPU** = tik `composite` procesas; gen, gfxgen (grafikos šaltinis) ir `verify` — atskiri stulpeliai informacijai.
  C variantui `--source fair` (kaip logo): gen ir gfxgen veikia GPU→host (pigiai), kad būtų matuojama funkcija, o ne sintetinis
  CPU generatorius; todėl C eilučių „ΔVRAM“ ir „GPU %“ rodo **šaltinių** GPU naudojimą, ne composite.
- **Terminas (R18):** `--deadline-ms 30` — tas pats kaip ankstesnėse logo talpos kampanijose (`2026-11-11-gpu-vs-cpu.md`),
  palyginamumui; ne 15 ms (plano 15 ms buvo pavyzdys; R14: WSL2 gamintojų drebėjimas > 15 ms). Tikėtina, kad su 15 ms talpa nebūtų didesnė (nematuota).
- **Ankstyvas sustojimas:** po 2 iš eilės PASS_tol FAIL tam (K, variantui) didesnės N nevykdytos. Nevykdyta: H N=32 (K=1,2,4) ir
  C N=32 (K=4); G pakartotoje kampanijoje sustojimas nebuvo suveikęs (vykdytos visos N, išskyrus K=4 N=32 — VRAM riba). Tai ne
  VRAM sprendimai (išskyrus G K=4 N=32).
- **G pakartota po `39dcb80`** (atskiras katalogas `evidence/compare-composite-g/`, komanda ta pati su `VARIANTS=gpu-cuda`;
  pirmojo, negaliojančio G matavimo duomenys liko `evidence/compare-composite/capacity-composite-k*-G.all.json`).
- **C variantas:** `--impl cpu` priverstinai naudoja host storage (`src/func/main_func.cpp`: `if (o.impl == "cpu") o.storage = "host"`),
  CUDA kontekstas nekuriamas (`verify` C/H variantuose taip pat CPU).
- **Harness po matavimų pakeistas** (be perkartojimo): composite `--duration` sutrumpintas iki `total + 20` (anksčiau `+ 40`, tik
  tuščias laukimas po lango); jei tarpas tarp composite ir verify starto > 14 s, pakopa žymima „NEDERINA“ (ne PASS/FAIL). **Visa pateikta
  kampanija vyko su senąja `+ 40` uodega, todėl jos matavimo langas nebuvo paveiktas**, patikros aprėptis įrašoma į JSON, VRAM modelis pakeistas regresija (žr. žemiau). Matavimo langas ir
  PASS kriterijai nepakito.
- **10.1 dūmų bandymas** (K=2, N=1, 20 s + 5 s apšilimas, `--source fair`, `capacity.py` tiesiogiai): C — griežtai PASS, verif 0, 40 mėginių,
  CPU/srautui 0,354; G — pirmas paleidimas griežtai FAIL (1 `late_skips`, WSL2 drebėjimas), tol PASS, verif 0 (39 mėginiai); G pakartojimas
  — griežtai PASS, verif 0. Žaliavos: `evidence/composite-smoke/`.
- **Latencija:** `proc` — nuo visų įėjimų paruošimo iki išėjimo commit'o; `vėlavimas` — commit'as po lauko laiko (`out_lateness`);
  p50 / p99 ms, composite histogramos per visą jo veikimą.

## Aplinka

- GPU: NVIDIA GeForce RTX 4070 Laptop 8 GB, tvarkyklė 596.49, CUDA 13.2, maks. SM 3105 MHz; `power.limit` = N/A, maks. 105 W.
  WSL2 Ubuntu 24.04 ant Windows 11; 12 CPU gijų, 32 GB RAM, `/dev/shm` 16 GB.
- **Windows energijos profilis: „Whisper“** (tylusis; riboja GPU/CPU galią) — kaip dec/chain kampanijose. Profilis nekeistas.
  Skaičiai orientaciniai; autoritetingi — natyviame Linux su užfiksuotais GPU dažniais.
- **GPU dažniai:** kampanijos pradžioje GPU P8, 210 MHz. 1 Hz zondo SM dažnio vidurkiai: N ≤ 2 daugumoje variantų 210–520 MHz
  (tuščios eigos būsena), N=4 — nuo 210 iki 2595 MHz (G K=4: 212 MHz, H K=4: 2595 MHz), didesnėms N daugiausia 520–2595 MHz.
  **G pakartojime** SM dažnio vidurkis pakopose: K=1 — 210, 210, 210, 449, 211, 380 MHz (N=1…32); K=2 — 210, 210, 547, 212, 375,
  688; K=4 — 239, 512, 211, 352, 687 (kampanijoje niekur nepasiekta maks. 3105 MHz). Galima hipotezė, kad žemų N
  PASS/FAIL iš dalies priklauso nuo dažnio kilimo, o ne vien nuo apkrovos (panašų artefaktą aprašo
  `2026-11-11-gpu-vs-cpu.md`, „Grandinės G N\* = 1 yra artefaktas“); šioje kampanijoje tai netikrinta.
- **Laikrodžio zondas** (`80-clock-probe.py`, 30 s), C/H kampanija (ir pirmoji, negaliojanti G): prieš — realtime/raw 2,2 ppm, PHC 0,9 ppm, realtime−PHC 0,036 ms,
  žingsnių 0; po — −0,1 ppm, −0,2 ppm, −0,016 ms, žingsnių 0. G pakartojimas (15:04–15:39): prieš — 0,0 / −0,0 ppm, 0,002 ms; po —
  0,1 / −0,2 ppm, 0,012 ms, žingsnių 0 (`clock-before.json`, `clock-after.json` abiejuose kataloguose). Zondas aprėpia tik 30 s langą;
  **WSL2 gamintojų drebėjimas (pavieniai vėluojantys laukai) išlieka** ir lemia dalį pavienių „griežtai FAIL“ (R14).

## VRAM / atminties biudžetas (R19)

Ring'as = 10 laukų grain'ų flow'ui (anksčiau išmatuota: logo G ΔVRAM ≈ 27,6 MB/flow = 10 × 2 764 800 B). Baitai: v210 1920×1080i
2 764 800 B; v210a `full` fill + key 4 147 200 B; v210a `logo` 384×216: fill 1 024 × 108 = 110 592 B + key 512 × 108 = 55 296 B =
165 888 B; išėjimas v210 2 764 800 B.

Pirmas modelis (ring × 1,0 + 4 × 128 MiB kontekstų) nuvertino išmatuotą G ΔVRAM; vienataškė pataisa „× 1,25“ vis dar
nuvertino (~4 %). Dabar (`tests/perf/capacity.py`, `comp_vram_mb`) naudojama **tiesinė regresija pagal išmatuotą G ΔVRAM**
(K ∈ {1,2,4}, N ≤ 16, K=1,2 N=32): ΔVRAM ≈ 507 MiB (4 procesų CUDA kontekstai, tas pats visiems K) + N · (1,085 · ring(bazė + logo +
išėjimas) + 1,42 · ring(`full` sluoksniai)); išmatuotuose taškuose paklaida ≤ 1 %, sprendimui pridedama 5 % atsarga. Koeficientas
1,42 reiškia, kad kiekvienas `full` sluoksnis kainuoja ~56 MiB/N (ring'as 39,6 MiB + papildomi gfxgen/composite buferiai).

| K | N=4 (modelis / išmatuota ΔVRAM) | N=32 (modelis / išmatuota) |
|---|---|---|
| 1 | 743 / 743 MiB | 2 393 / 2 391 MiB |
| 2 | 967 / 967 MiB | 4 190 / 4 189 MiB |
| 4 | 1 417 / 1 417 MiB | 7 784 (ekstrapoliacija) / aborted: pakilo iki 7 223 MiB |

**G K=4 N=32 buvo paleista** (pirmas modelis rodė 6 047 MiB < 7 168): ΔVRAM pakilo iki 7 223 MiB (zondas 7 554 iš 8 188 MiB), GPU
100 %, `composite` nepradėjo išvesti ir buvo nutrauktas (killed). Tai **VRAM riba (išmatuota)**, o ne talpos rezultatas. Žaliavų
failo eilutė **ranka pakeista iš FAIL į „skipped“** (`manual_reclassification: true` JSON'e): tai **nėra harness išvestis**; originalus
harness įrašas — `evidence/compare-composite-g/capacity-composite-k4-G.attempt-n32.json` (lentelės eilutė nurodo būtent jį). Su
dabartiniu modeliu (7 784 · 1,05 = 8 174 MiB > 7 168) ši pakopa automatiškai praleidžiama. Pastebėta: `composite` prie VRAM
išsekimo nesibaigia aiškia klaida, o pakimba. C/H (host): ring'ai `/dev/shm`; didžiausia K=4 N=32 ~5,5 GiB < 75 % × 16 GiB, apsauga
netrigerinta.

## Rezultatai


Funkcijos CPU — tik `composite` procesas (šaltiniai gen, gfxgen ir patikra `verify` atskirai, informacijai). Latencija: `proc` — nuo visų įėjimų paruošimo iki išėjimo commit'o; `vėlavimas` — commit'as po lauko laiko (`out_lateness`), p50/p99 ms visame composite veikimo laike. GPU % / W — nvidia-smi 1 Hz matavimo lange. `patikra %` — `verify` aprėptis: patikrinta mėginių / laukiama (lauko indeksų intervalas × N / `--every`); `⚠` — aprėptis < 50% (PASS apibrėžimas nepakeistas, bet pakopa patikrinta tik iš dalies); `nuleista` — `check_dropped` (patikros, kurių verify nespėjo).

### K = 1 — C — CPU + host flow (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 101 | 0 | 0.149 | 0.010 | 0.179 | 0.74 / 1.46 | 4.78 / 6.62 | 258 | 11 | 4 | `compare-composite/capacity-composite-k1-C.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 100 | 0 | 0.128 | 0.007 | 0.138 | 1.01 / 1.82 | 7.01 / 9.04 | 262 | 20 | 4 | `compare-composite/capacity-composite-k1-C.all.json` |
| 4 | FAIL | PASS | 0.182 | 0.9991 | 0 | 480 | 100 | 0 | 0.110 | 0.005 | 0.115 | 1.33 / 2.19 | 10.88 / 13.76 | 270 | 37 | 4 | `compare-composite/capacity-composite-k1-C.all.json` |
| 8 | PASS | PASS | 0.000 | 1.0000 | 0 | 959 | 100 | 0 | 0.100 | 0.005 | 0.061 | 2.26 / 3.11 | 12.19 / 18.61 | 286 | 37 | 8 | `compare-composite/capacity-composite-k1-C.all.json` |
| 16 | FAIL | PASS | 0.109 | 0.9993 | 0 | 1914 | 100 | 0 | 0.086 | 0.004 | 0.024 | 3.70 / 4.78 | 10.62 / 24.38 | 318 | 19 | 16 | `compare-composite/capacity-composite-k1-C.all.json` |
| 32 | FAIL | FAIL | 5.692 | 0.9604 | 0 | 3464 | 90 | 125 | 0.079 | 0.003 | 0.022 | 7.27 / 9.83 | 20.71 / 32.77 | 384 | 43 | 26 | `compare-composite/capacity-composite-k1-C.all.json` |

N* griežtai = **2**, N* tol = **16**

### K = 1 — H — GPU + host flow (MXL v1.1) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 119 | 100 | 0 | 0.305 | 0.010 | 0.169 | 4.78 / 7.67 | 8.78 / 11.66 | 394 | 29 | 4 | `compare-composite/capacity-composite-k1-H.all.json` |
| 2 | FAIL | PASS | 0.018 | 0.9998 | 0 | 240 | 100 | 0 | 0.161 | 0.007 | 0.087 | 4.92 / 8.52 | 9.04 / 14.29 | 406 | 31 | 7 | `compare-composite/capacity-composite-k1-H.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 478 | 100 | 0 | 0.064 | 0.005 | 0.036 | 3.31 / 8.78 | 6.23 / 14.29 | 430 | 24 | 11 | `compare-composite/capacity-composite-k1-H.all.json` |
| 8 | FAIL | FAIL | 2.491 | 0.9843 | 0 | 945 | 98 | 0 | 0.047 | 0.005 | 0.026 | 5.96 / 22.28 | 10.09 / 31.72 | 478 | 31 | 15 | `compare-composite/capacity-composite-k1-H.all.json` |
| 16 | FAIL | FAIL | 30.042 | 0.7962 | 0 | 1487 | 78 | 0 | 0.042 | 0.004 | 0.028 | 11.93 / 25.43 | 19.14 / 37.22 | 576 | 61 | 28 | `compare-composite/capacity-composite-k1-H.all.json` |

N* griežtai = **1**, N* tol = **4**

### K = 1 — G — GPU + cuda-linear (PR #653) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 101 | 0 | 0.060 | 0.098 | 0.077 | 0.70 / 1.26 | 2.85 / 5.57 | 567 | 8 | 4 | `compare-composite-g/capacity-composite-k1-G.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 239 | 100 | 0 | 0.043 | 0.108 | 0.045 | 1.13 / 1.69 | 3.96 / 6.09 | 625 | 12 | 4 | `compare-composite-g/capacity-composite-k1-G.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 480 | 100 | 0 | 0.032 | 0.085 | 0.037 | 1.82 / 2.39 | 4.92 / 8.32 | 743 | 18 | 5 | `compare-composite-g/capacity-composite-k1-G.all.json` |
| 8 | FAIL | FAIL | 0.400 | 0.9980 | 0 | 955 | 100 | 0 | 0.025 | 0.101 | 0.023 | 2.65 / 6.09 | 6.75 / 14.55 | 977 | 27 | 8 | `compare-composite-g/capacity-composite-k1-G.all.json` |
| 16 | FAIL | PASS | 0.018 | 0.9998 | 0 | 1917 | 100 | 0 | 0.010 | 0.060 | 0.018 | 1.69 / 5.96 | 8.78 / 12.45 | 1449 | 32 | 12 | `compare-composite-g/capacity-composite-k1-G.all.json` |
| 32 | PASS | PASS | 0.000 | 1.0000 | 0 | 3624 | 94 | 213 | 0.007 | 0.039 | 0.010 | 1.82 / 2.19 | 9.31 / 11.66 | 2391 | 37 | 13 | `compare-composite-g/capacity-composite-k1-G.all.json` |

N* griežtai = **4**, N* tol = **4**

### K = 2 — C — CPU + host flow (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 101 | 0 | 0.348 | 0.031 | 0.172 | 1.26 / 1.98 | 5.18 / 6.88 | 258 | 11 | 4 | `compare-composite/capacity-composite-k2-C.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 239 | 100 | 0 | 0.317 | 0.031 | 0.136 | 1.95 / 2.85 | 7.80 / 9.83 | 262 | 20 | 4 | `compare-composite/capacity-composite-k2-C.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 480 | 100 | 0 | 0.283 | 0.044 | 0.117 | 3.05 / 3.77 | 12.98 / 15.34 | 270 | 37 | 4 | `compare-composite/capacity-composite-k2-C.all.json` |
| 8 | FAIL | PASS | 0.036 | 0.9998 | 0 | 957 | 100 | 0 | 0.249 | 0.069 | 0.063 | 5.18 / 6.49 | 15.86 / 21.76 | 286 | 38 | 8 | `compare-composite/capacity-composite-k2-C.all.json` |
| 16 | FAIL | FAIL | 6.345 | 0.9579 | 0 | 1813 | 94 | 0 | 0.224 | 0.077 | 0.029 | 9.57 / 14.81 | 19.14 / 34.08 | 320 | 24 | 16 | `compare-composite/capacity-composite-k2-C.all.json` |
| 32 | FAIL | FAIL | 89.492 | 0.3730 | 0 | 1292 | 34 ⚠ | 0 | 0.097 | 0.088 | 0.024 | 31.20 / 37.22 | 50.86 / 59.24 | 388 | 37 | 26 | `compare-composite/capacity-composite-k2-C.all.json` |

N* griežtai = **4**, N* tol = **8**

### K = 2 — H — GPU + host flow (MXL v1.1) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FAIL | PASS | 0.018 | 0.9998 | 0 | 120 | 101 | 0 | 0.337 | 0.029 | 0.139 | 6.88 / 8.78 | 10.09 / 12.98 | 398 | 32 | 6 | `compare-composite/capacity-composite-k2-H.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 100 | 0 | 0.198 | 0.031 | 0.082 | 7.14 / 8.78 | 10.88 / 13.50 | 414 | 40 | 9 | `compare-composite/capacity-composite-k2-H.all.json` |
| 4 | FAIL | PASS | 0.036 | 0.9998 | 0 | 480 | 100 | 0 | 0.081 | 0.044 | 0.035 | 4.78 / 16.12 | 7.93 / 21.76 | 446 | 26 | 16 | `compare-composite/capacity-composite-k2-H.all.json` |
| 8 | FAIL | FAIL | 0.273 | 0.9984 | 0 | 951 | 99 | 0 | 0.085 | 0.064 | 0.033 | 9.83 / 19.14 | 15.34 / 28.05 | 510 | 58 | 27 | `compare-composite/capacity-composite-k2-H.all.json` |
| 16 | FAIL | FAIL | 77.570 | 0.4511 | 0 | 827 | 43 ⚠ | 0 | 0.044 | 0.082 | 0.037 | 22.81 / 45.61 | 32.77 / 59.24 | 642 | 62 | 28 | `compare-composite/capacity-composite-k2-H.all.json` |

N* griežtai = **0**, N* tol = **4**

### K = 2 — G — GPU + cuda-linear (PR #653) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FAIL | PASS | 0.036 | 0.9998 | 0 | 120 | 101 | 0 | 0.082 | 0.158 | 0.111 | 1.10 / 1.72 | 3.96 / 6.49 | 623 | 11 | 5 | `compare-composite-g/capacity-composite-k2-G.all.json` |
| 2 | FAIL | PASS | 0.018 | 0.9998 | 0 | 238 | 100 | 0 | 0.062 | 0.451 | 0.070 | 1.46 / 2.13 | 7.14 / 9.57 | 737 | 27 | 5 | `compare-composite-g/capacity-composite-k2-G.all.json` |
| 4 | FAIL | FAIL | 2.382 | 0.9860 | 0 | 466 | 97 | 0 | 0.037 | 0.338 | 0.039 | 2.39 / 5.83 | 7.01 / 18.09 | 967 | 26 | 10 | `compare-composite-g/capacity-composite-k2-G.all.json` |
| 8 | FAIL | PASS | 0.073 | 0.9996 | 0 | 959 | 100 | 0 | 0.028 | 0.131 | 0.030 | 2.92 / 3.64 | 7.93 / 11.66 | 1427 | 33 | 13 | `compare-composite-g/capacity-composite-k2-G.all.json` |
| 16 | FAIL | PASS | 0.036 | 0.9998 | 0 | 1919 | 100 | 0 | 0.017 | 0.094 | 0.014 | 3.18 / 3.83 | 9.57 / 12.98 | 2349 | 37 | 14 | `compare-composite-g/capacity-composite-k2-G.all.json` |
| 32 | FAIL | PASS | 0.145 | 0.9991 | 0 | 2714 | 71 | 1119 | 0.011 | 0.056 | 0.007 | 3.31 / 4.10 | 10.88 / 15.86 | 4189 | 39 | 17 | `compare-composite-g/capacity-composite-k2-G.all.json` |

N* griežtai = **0**, N* tol = **2**

### K = 4 — C — CPU + host flow (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 101 | 0 | 0.671 | 0.100 | 0.179 | 2.08 / 2.85 | 6.23 / 8.06 | 258 | 11 | 4 | `compare-composite/capacity-composite-k4-C.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 100 | 0 | 0.631 | 0.171 | 0.143 | 3.51 / 4.16 | 9.57 / 11.40 | 262 | 20 | 4 | `compare-composite/capacity-composite-k4-C.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 480 | 100 | 0 | 0.561 | 0.233 | 0.123 | 5.83 / 7.41 | 16.38 / 18.61 | 270 | 37 | 4 | `compare-composite/capacity-composite-k4-C.all.json` |
| 8 | FAIL | FAIL | 11.638 | 0.9375 | 0 | 933 | 97 | 0 | 0.531 | 0.155 | 0.065 | 13.76 / 17.56 | 26.48 / 32.77 | 288 | 37 | 8 | `compare-composite/capacity-composite-k4-C.all.json` |
| 16 | FAIL | FAIL | 95.596 | 0.3195 | 0 | 571 | 30 ⚠ | 0 | 0.201 | 0.267 | 0.036 | 39.32 / 50.86 | 55.05 / 72.35 | 324 | 25 | 13 | `compare-composite/capacity-composite-k4-C.all.json` |

N* griežtai = **4**, N* tol = **4**

### K = 4 — H — GPU + host flow (MXL v1.1) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 101 | 0 | 0.379 | 0.099 | 0.121 | 6.75 / 12.19 | 9.83 / 15.86 | 406 | 35 | 8 | `compare-composite/capacity-composite-k4-H.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 100 | 0 | 0.150 | 0.162 | 0.056 | 4.16 / 13.50 | 6.88 / 17.56 | 430 | 31 | 14 | `compare-composite/capacity-composite-k4-H.all.json` |
| 4 | FAIL | PASS | 0.109 | 0.9995 | 0 | 480 | 100 | 0 | 0.144 | 0.230 | 0.050 | 8.52 / 14.03 | 13.24 / 19.66 | 478 | 47 | 27 | `compare-composite/capacity-composite-k4-H.all.json` |
| 8 | FAIL | FAIL | 39.521 | 0.7317 | 0 | 684 | 71 | 0 | 0.109 | 0.262 | 0.048 | 19.66 / 36.18 | 29.10 / 46.66 | 576 | 65 | 30 | `compare-composite/capacity-composite-k4-H.all.json` |
| 16 | FAIL | FAIL | 115.707 | 0.2153 | 0 | 135 | 7 ⚠ | 0 | 0.051 | 0.336 | 0.042 | 72.35 / 157.29 | 89.13 / 174.06 | 774 | 51 | 28 | `compare-composite/capacity-composite-k4-H.all.json` |

N* griežtai = **2**, N* tol = **4**

### K = 4 — G — GPU + cuda-linear (PR #653) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | patikra % | nuleista | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FAIL | FAIL | 0.509 | 0.9973 | 0 | 120 | 101 | 0 | 0.142 | 1.049 | 0.160 | 1.66 / 2.85 | 8.06 / 11.66 | 735 | 30 | 7 | `compare-composite-g/capacity-composite-k4-G.all.json` |
| 2 | FAIL | PASS | 0.055 | 0.9996 | 0 | 240 | 100 | 0 | 0.102 | 0.866 | 0.054 | 2.46 / 5.70 | 7.41 / 14.55 | 961 | 30 | 10 | `compare-composite-g/capacity-composite-k4-G.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 480 | 100 | 0 | 0.058 | 0.301 | 0.047 | 3.64 / 3.90 | 8.32 / 10.62 | 1417 | 35 | 12 | `compare-composite-g/capacity-composite-k4-G.all.json` |
| 8 | PASS | PASS | 0.000 | 1.0000 | 0 | 960 | 100 | 0 | 0.035 | 0.196 | 0.020 | 4.10 / 4.39 | 9.31 / 11.66 | 2327 | 37 | 14 | `compare-composite-g/capacity-composite-k4-G.all.json` |
| 16 | PASS | PASS | 0.000 | 1.0000 | 0 | 1905 | 99 | 13 | 0.020 | 0.106 | 0.007 | 4.10 / 4.39 | 10.36 / 14.03 | 4147 | 39 | 18 | `compare-composite-g/capacity-composite-k4-G.all.json` |
| 32 | praleista | praleista | | | | | | | | | | | | | | | VRAM riba (išmatuota): bandymas pakilo iki ΔVRAM 7223 MiB (zondas 7554 MiB iš 8188), composite nepradėjo išvesti ir buvo nutrauktas (killed); originalus įrašas capacity-composite-k4-G.attempt-n32.json. Pirmas modelis rodė 6047 MiB; regresijos modelis: 7784 MiB (× 1,05 atsarga = 8174 MiB) > 7168 MiB. (ne harness išvestis: rankinė perklasifikacija); `compare-composite-g/capacity-composite-k4-G.attempt-n32.json` |

N* griežtai = **0**, N* tol = **0** (N=32 praleista, žr. eilutės priežastį — riba nepasiekta tik todėl, kad pakopos nevykdytos)

### Suvestinė — didžiausias N (N*), kai visi mažesni PASS

Reikšmė „griežtai / tol“. Skliaustuose — praleistos didesnės pakopos (VRAM / rankinė perklasifikacija); tokiu atveju N* nėra talpos riba. Pakopos, kurių nevykdė stop-after-fails taisyklė, nėra N* dalis.

| K | C | H | G |
|---|---|---|---|
| 1 | 2 / 16 | 1 / 4 | 4 / 4 |
| 2 | 4 / 8 | 0 / 4 | 0 / 2 |
| 4 | 4 / 4 | 2 / 4 | 0 / 0 (N=32 praleista) |

Pakopos su patikros aprėptimi < 50% (R21): K=2 C N=32 (34 %), K=2 H N=16 (43 %), K=4 C N=16 (30 %), K=4 H N=16 (7 %).


## Pastabos ir apribojimai

- **G (po `39dcb80`):** 0 teisingumo klaidų visose vykdytose pakopose; patikros aprėptis 71–101 % (mažiausia K=2 N=32: 71 %). Dauguma G FAIL pakopų — pavieniai
  vėluojantys laukai (K=2: griežtai FAIL visur; tol PASS kai N=1, 2, 8, 16, 32, N=4: 2,4 % praradimų), `proc` p99 G pakopose ≈ 6 ms
  ar mažiau. **Hipotezė (neišbandyta):** žemos N pakopos nukenčia nuo žemo GPU dažnio, o K=4 N≈32 riboja VRAM; abu teiginiai
  remiasi pavieniais matavimais be užfiksuotų dažnių ir be pakartojimų. Tikslesniam G/C santykiui reikia užfiksuotų GPU dažnių ir
  pakartojimų.
- **H** K=1 jau N=8 FAIL (`proc` p99 22 ms, artėja prie 30 ms termino).
- **C** gerai skalėjasi mažiems K (K=1 PASS_tol iki N=16), bet K=4 jau N=8 FAIL (11,6 % praradimų, 0,53 branduolio/srautui).
- Nemonotoniškumas (K=1 C N=4 griežtai FAIL, N=8 PASS; K=1 H N=2 FAIL, N=4 PASS) — tikėtina, kad dėl WSL2 drebėjimo ir GPU dažnio perėjimų (netikrinta);
  pakartojimų nedaryta (viena pakopa vienam variantui). Pasikliauti reikia tendencija, ne pavieniu N.
- `verify` mėginių skaičius mažesnis didelės apkrovos FAIL pakopose (K=4 N=16: C 571, H 135 mėginiai; `check_dropped` ten = 0, todėl priežastis
  neišaiškinta — tikėtina, kad verify neturėjo ką tikrinti dėl praleistų composite laukų arba dėl apkrovos; nepatikrinta).
- Nevykdyta: v210a išėjimo kampanija, pakartojimai (išskyrus G po pataisos), G K=4 N=32 (VRAM).
- Logo kampanija vyko kitu energijos profiliu (`2026-11-11-gpu-vs-cpu.md`), todėl logo ir composite skaičiai tiesiogiai nepalyginami.
