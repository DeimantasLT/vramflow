# composite — talpa N × K (C / H / G)

Data: 2026-11-11 (kampanija 13:15–14:33 vietos laiku). Užduotis 10, alfa kompozitorius. Žaliavos: `evidence/compare-composite/`
(git ignoruojama; lentelių eilutės nurodo `capacity-composite-k<K>-<V>.all.json`), aplinka — `evidence/compare-composite.env.txt`,
žurnalas — `evidence/compare-composite/compare.log`, praleistos pakopos — `skipped-after-fails.jsonl`.

## Išvados trumpai

- **Didžiausias N, kai visi mažesni PASS** (N* tol / N* griežtai): C — K=1: 16 / 2, K=2: 8 / 4, K=4: 4 / 4;
  H — K=1: 4 / 1, K=2: 4 / 0, K=4: 4 / 2; **G — 1 / 1 visiems K** (G pakopose N ≥ 2 yra teisingumo klaidų, žr. toliau).
- **G (cuda-linear): `verify` randa neatitikimų** K=1 N=2: 60, N=4: 120; K=2 N=2: 60; K=4 N=2: 60 (kitose G pakopose 0).
  Tai teisingumo, ne talpos rezultatas. Pavyzdžiuose neatitinka tik antro flow'o (n=1) logo stačiakampis (lauko eilutės 32…139,
  x 1488…1871) ir tik lyginių indeksų (viršutinių) laukų. C ir H tomis pačiomis N — 0 klaidų. Pavyzdžiai:
  `evidence/compare-composite/composite-k1-G-n2/mismatch_n1_*.{actual,expected}.v210`. **Priežastis neištirta** (funkciniai
  testai tikrina `verify` tik su N=1; įtariama G kelio N ≥ 2 indeksavimas kompozitoriuje arba gfxgen pace režime).
  Todėl G talpos skaičiai **nėra patikimas „G pajėgumas“**.
- Žemuose N GPU dažnis dažnai likęs tuščios eigos (210 MHz, žr. aplinką), todėl žemų N rezultatai pesimistiški ir nemonotoniški.
- composite proceso CPU PASS pakopose: C ≈ 0,08–0,15 branduolio/srautui (K=1), 0,25–0,35 (K=2), 0,56–0,67 (K=4);
  G ≈ 0,03–0,13, H ≈ 0,04–0,38 (K=1…4).

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
  palyginamumui; ne 15 ms (plano 15 ms buvo pavyzdys; R14: WSL2 gamintojų drebėjimas > 15 ms). Su 15 ms talpa nebūtų didesnė.
- **Ankstyvas sustojimas:** po 2 iš eilės PASS_tol FAIL tam (K, variantui) didesnės N nevykdytos. Nevykdyta: K=1 G N=8,16,32 ir
  H N=32; K=2 G N=8,16,32 ir H N=32; K=4 G N=8,16,32, H N=32, C N=32. Tai ne VRAM sprendimai.
- **Latencija:** `proc` — nuo visų įėjimų paruošimo iki išėjimo commit'o; `vėlavimas` — commit'as po lauko laiko (`out_lateness`);
  p50 / p99 ms, composite histogramos per visą jo veikimą.

## Aplinka

- GPU: NVIDIA GeForce RTX 4070 Laptop 8 GB, tvarkyklė 596.49, CUDA 13.2, maks. SM 3105 MHz; `power.limit` = N/A, maks. 105 W.
  WSL2 Ubuntu 24.04 ant Windows 11; 12 CPU gijų, 32 GB RAM, `/dev/shm` 16 GB.
- **Windows energijos profilis: „Whisper“** (tylusis; riboja GPU/CPU galią) — kaip dec/chain kampanijose. Profilis nekeistas.
  Skaičiai orientaciniai; autoritetingi — natyviame Linux su užfiksuotais GPU dažniais.
- **GPU dažniai:** kampanijos pradžioje GPU P8, 210 MHz. 1 Hz zondo SM dažnio vidurkiai: N ≤ 2 daugumoje variantų 210–520 MHz
  (tuščios eigos būsena), N=4 — nuo 210 iki 2595 MHz (G K=4: 212 MHz, H K=4: 2595 MHz), didesnėms N daugiausia 520–2595 MHz.
  Todėl žemų N PASS/FAIL iš dalies priklauso nuo dažnio kilimo, ne vien nuo apkrovos (tą patį artefaktą aprašo
  `2026-11-11-gpu-vs-cpu.md`, „Grandinės G N\* = 1 yra artefaktas“).
- **Laikrodžio zondas** (`80-clock-probe.py`, 30 s): prieš — realtime/raw 2,2 ppm, PHC 0,9 ppm, realtime−PHC 0,036 ms, žingsnių 0;
  po — −0,1 ppm, −0,2 ppm, −0,016 ms, žingsnių 0 (`clock-before.json`, `clock-after.json`). Zondas aprėpia tik 30 s langą;
  **WSL2 gamintojų drebėjimas (pavieniai vėluojantys laukai) išlieka** ir lemia dalį pavienių „griežtai FAIL“ (R14).

## VRAM / atminties biudžetas (R19)

Ring'as = 10 laukų grain'ų flow'ui (anksčiau išmatuota: logo G ΔVRAM ≈ 27,6 MB/flow = 10 × 2 764 800 B). Baitai: v210 1920×1080i
2 764 800 B; v210a `full` fill + key 4 147 200 B; v210a `logo` 384×216: fill 1 024 × 108 = 110 592 B + key 512 × 108 = 55 296 B =
165 888 B; išėjimas v210 2 764 800 B. G: (bazė N + sluoksniai N·K + išėjimai N) × 10 + 4 × 128 MiB CUDA kontekstų (gen, gfxgen,
composite, verify); funkcija `comp_budget_mb` / `comp_guard` (`tests/perf/capacity.py`).

| K | N=4 (apskaič. / išmatuota ΔVRAM) | N=32 (apskaič.) |
|---|---|---|
| 1 | 729 / 743 MiB | 2 250 MiB |
| 2 | 887 / 967 MiB | 3 516 MiB |
| 4 | 1 204 / 1 417 MiB | 6 047 MiB |

K=4, N=32 pagal skaičiavimą ~6,0 GiB < 7 GiB riba, bet K=4 N=4 skaičiavimas nuvertino išmatuotą ΔVRAM ~18 %, t. y. N=32
realiai būtų ~7 GiB — riboje. **Dėl VRAM nepraleista nė viena pakopa**; G N=32 nevykdyta dėl ankstyvo sustojimo. C/H (host):
ring'ai `/dev/shm`; didžiausia K=4 N=32 ~5,5 GiB < 75 % × 16 GiB, apsauga netrigerinta.

## Rezultatai


Funkcijos CPU — tik `composite` procesas (šaltiniai gen, gfxgen ir patikra `verify` atskirai, informacijai). Latencija: `proc` — nuo visų įėjimų paruošimo iki išėjimo commit'o; `vėlavimas` — commit'as po lauko laiko (`out_lateness`), p50/p99 ms visame composite veikimo laike. GPU % / W — nvidia-smi 1 Hz matavimo lange.

### K = 1 — C — CPU + host flow (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 0.149 | 0.010 | 0.179 | 0.74 / 1.46 | 4.78 / 6.62 | 258 | 11 | 4 | `compare-composite/capacity-composite-k1-C.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 0.128 | 0.007 | 0.138 | 1.01 / 1.82 | 7.01 / 9.04 | 262 | 20 | 4 | `compare-composite/capacity-composite-k1-C.all.json` |
| 4 | FAIL | PASS | 0.182 | 0.9991 | 0 | 480 | 0.110 | 0.005 | 0.115 | 1.33 / 2.19 | 10.88 / 13.76 | 270 | 37 | 4 | `compare-composite/capacity-composite-k1-C.all.json` |
| 8 | PASS | PASS | 0.000 | 1.0000 | 0 | 959 | 0.100 | 0.005 | 0.061 | 2.26 / 3.11 | 12.19 / 18.61 | 286 | 37 | 8 | `compare-composite/capacity-composite-k1-C.all.json` |
| 16 | FAIL | PASS | 0.109 | 0.9993 | 0 | 1914 | 0.086 | 0.004 | 0.024 | 3.70 / 4.78 | 10.62 / 24.38 | 318 | 19 | 16 | `compare-composite/capacity-composite-k1-C.all.json` |
| 32 | FAIL | FAIL | 5.692 | 0.9604 | 0 | 3464 | 0.079 | 0.003 | 0.022 | 7.27 / 9.83 | 20.71 / 32.77 | 384 | 43 | 26 | `compare-composite/capacity-composite-k1-C.all.json` |

N* griežtai = **2**, N* tol = **16**

### K = 1 — H — GPU + host flow (MXL v1.1) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 119 | 0.305 | 0.010 | 0.169 | 4.78 / 7.67 | 8.78 / 11.66 | 394 | 29 | 4 | `compare-composite/capacity-composite-k1-H.all.json` |
| 2 | FAIL | PASS | 0.018 | 0.9998 | 0 | 240 | 0.161 | 0.007 | 0.087 | 4.92 / 8.52 | 9.04 / 14.29 | 406 | 31 | 7 | `compare-composite/capacity-composite-k1-H.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 478 | 0.064 | 0.005 | 0.036 | 3.31 / 8.78 | 6.23 / 14.29 | 430 | 24 | 11 | `compare-composite/capacity-composite-k1-H.all.json` |
| 8 | FAIL | FAIL | 2.491 | 0.9843 | 0 | 945 | 0.047 | 0.005 | 0.026 | 5.96 / 22.28 | 10.09 / 31.72 | 478 | 31 | 15 | `compare-composite/capacity-composite-k1-H.all.json` |
| 16 | FAIL | FAIL | 30.042 | 0.7962 | 0 | 1487 | 0.042 | 0.004 | 0.028 | 11.93 / 25.43 | 19.14 / 37.22 | 576 | 61 | 28 | `compare-composite/capacity-composite-k1-H.all.json` |

N* griežtai = **1**, N* tol = **4**

### K = 1 — G — GPU + cuda-linear (PR #653) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 0.071 | 0.105 | 0.084 | 0.86 / 1.85 | 3.57 / 6.09 | 567 | 8 | 4 | `compare-composite/capacity-composite-k1-G.all.json` |
| 2 | FAIL | FAIL | 0.000 | 1.0000 | 60 | 240 | 0.043 | 0.074 | 0.055 | 1.13 / 1.62 | 3.96 / 6.49 | 625 | 11 | 4 | `compare-composite/capacity-composite-k1-G.all.json` |
| 4 | FAIL | FAIL | 0.764 | 0.9945 | 120 | 480 | 0.032 | 0.084 | 0.037 | 1.75 / 2.39 | 4.92 / 8.32 | 743 | 18 | 5 | `compare-composite/capacity-composite-k1-G.all.json` |

N* griežtai = **1**, N* tol = **1**

### K = 2 — C — CPU + host flow (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 0.348 | 0.031 | 0.172 | 1.26 / 1.98 | 5.18 / 6.88 | 258 | 11 | 4 | `compare-composite/capacity-composite-k2-C.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 239 | 0.317 | 0.031 | 0.136 | 1.95 / 2.85 | 7.80 / 9.83 | 262 | 20 | 4 | `compare-composite/capacity-composite-k2-C.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 480 | 0.283 | 0.044 | 0.117 | 3.05 / 3.77 | 12.98 / 15.34 | 270 | 37 | 4 | `compare-composite/capacity-composite-k2-C.all.json` |
| 8 | FAIL | PASS | 0.036 | 0.9998 | 0 | 957 | 0.249 | 0.069 | 0.063 | 5.18 / 6.49 | 15.86 / 21.76 | 286 | 38 | 8 | `compare-composite/capacity-composite-k2-C.all.json` |
| 16 | FAIL | FAIL | 6.345 | 0.9579 | 0 | 1813 | 0.224 | 0.077 | 0.029 | 9.57 / 14.81 | 19.14 / 34.08 | 320 | 24 | 16 | `compare-composite/capacity-composite-k2-C.all.json` |
| 32 | FAIL | FAIL | 89.492 | 0.3730 | 0 | 1292 | 0.097 | 0.088 | 0.024 | 31.20 / 37.22 | 50.86 / 59.24 | 388 | 37 | 26 | `compare-composite/capacity-composite-k2-C.all.json` |

N* griežtai = **4**, N* tol = **8**

### K = 2 — H — GPU + host flow (MXL v1.1) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FAIL | PASS | 0.018 | 0.9998 | 0 | 120 | 0.337 | 0.029 | 0.139 | 6.88 / 8.78 | 10.09 / 12.98 | 398 | 32 | 6 | `compare-composite/capacity-composite-k2-H.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 0.198 | 0.031 | 0.082 | 7.14 / 8.78 | 10.88 / 13.50 | 414 | 40 | 9 | `compare-composite/capacity-composite-k2-H.all.json` |
| 4 | FAIL | PASS | 0.036 | 0.9998 | 0 | 480 | 0.081 | 0.044 | 0.035 | 4.78 / 16.12 | 7.93 / 21.76 | 446 | 26 | 16 | `compare-composite/capacity-composite-k2-H.all.json` |
| 8 | FAIL | FAIL | 0.273 | 0.9984 | 0 | 951 | 0.085 | 0.064 | 0.033 | 9.83 / 19.14 | 15.34 / 28.05 | 510 | 58 | 27 | `compare-composite/capacity-composite-k2-H.all.json` |
| 16 | FAIL | FAIL | 77.570 | 0.4511 | 0 | 827 | 0.044 | 0.082 | 0.037 | 22.81 / 45.61 | 32.77 / 59.24 | 642 | 62 | 28 | `compare-composite/capacity-composite-k2-H.all.json` |

N* griežtai = **0**, N* tol = **4**

### K = 2 — G — GPU + cuda-linear (PR #653) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 0.083 | 0.154 | 0.110 | 1.10 / 1.62 | 3.96 / 6.62 | 623 | 11 | 5 | `compare-composite/capacity-composite-k2-G.all.json` |
| 2 | FAIL | FAIL | 0.055 | 0.9995 | 60 | 240 | 0.061 | 0.446 | 0.069 | 1.43 / 2.05 | 7.01 / 9.57 | 737 | 26 | 5 | `compare-composite/capacity-composite-k2-G.all.json` |
| 4 | FAIL | FAIL | 2.773 | 0.9826 | 0 | 476 | 0.038 | 0.360 | 0.040 | 2.26 / 5.70 | 6.88 / 18.09 | 967 | 23 | 11 | `compare-composite/capacity-composite-k2-G.all.json` |

N* griežtai = **1**, N* tol = **1**

### K = 4 — C — CPU + host flow (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 0.671 | 0.100 | 0.179 | 2.08 / 2.85 | 6.23 / 8.06 | 258 | 11 | 4 | `compare-composite/capacity-composite-k4-C.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 0.631 | 0.171 | 0.143 | 3.51 / 4.16 | 9.57 / 11.40 | 262 | 20 | 4 | `compare-composite/capacity-composite-k4-C.all.json` |
| 4 | PASS | PASS | 0.000 | 1.0000 | 0 | 480 | 0.561 | 0.233 | 0.123 | 5.83 / 7.41 | 16.38 / 18.61 | 270 | 37 | 4 | `compare-composite/capacity-composite-k4-C.all.json` |
| 8 | FAIL | FAIL | 11.638 | 0.9375 | 0 | 933 | 0.531 | 0.155 | 0.065 | 13.76 / 17.56 | 26.48 / 32.77 | 288 | 37 | 8 | `compare-composite/capacity-composite-k4-C.all.json` |
| 16 | FAIL | FAIL | 95.596 | 0.3195 | 0 | 571 | 0.201 | 0.267 | 0.036 | 39.32 / 50.86 | 55.05 / 72.35 | 324 | 25 | 13 | `compare-composite/capacity-composite-k4-C.all.json` |

N* griežtai = **4**, N* tol = **4**

### K = 4 — H — GPU + host flow (MXL v1.1) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 0.379 | 0.099 | 0.121 | 6.75 / 12.19 | 9.83 / 15.86 | 406 | 35 | 8 | `compare-composite/capacity-composite-k4-H.all.json` |
| 2 | PASS | PASS | 0.000 | 1.0000 | 0 | 240 | 0.150 | 0.162 | 0.056 | 4.16 / 13.50 | 6.88 / 17.56 | 430 | 31 | 14 | `compare-composite/capacity-composite-k4-H.all.json` |
| 4 | FAIL | PASS | 0.109 | 0.9995 | 0 | 480 | 0.144 | 0.230 | 0.050 | 8.52 / 14.03 | 13.24 / 19.66 | 478 | 47 | 27 | `compare-composite/capacity-composite-k4-H.all.json` |
| 8 | FAIL | FAIL | 39.521 | 0.7317 | 0 | 684 | 0.109 | 0.262 | 0.048 | 19.66 / 36.18 | 29.10 / 46.66 | 576 | 65 | 30 | `compare-composite/capacity-composite-k4-H.all.json` |
| 16 | FAIL | FAIL | 115.707 | 0.2153 | 0 | 135 | 0.051 | 0.336 | 0.042 | 72.35 / 157.29 | 89.13 / 174.06 | 774 | 51 | 28 | `compare-composite/capacity-composite-k4-H.all.json` |

N* griežtai = **2**, N* tol = **4**

### K = 4 — G — GPU + cuda-linear (PR #653) (šaltinis: fair, išėjimas v210, deadline 30 ms)

| N | griežtai | tol | praradimai % | RT | teisingumo klaidos | verify mėginių | CPU/srautui (composite) | gfxgen CPU | gen CPU | proc p50/p99 ms | vėlavimas p50/p99 ms | ΔVRAM MB | GPU % | W | šaltinis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | PASS | 0.000 | 1.0000 | 0 | 120 | 0.127 | 1.029 | 0.130 | 1.56 / 2.13 | 7.54 / 9.57 | 735 | 29 | 6 | `compare-composite/capacity-composite-k4-G.all.json` |
| 2 | FAIL | FAIL | 0.036 | 0.9998 | 60 | 240 | 0.097 | 0.881 | 0.059 | 2.33 / 5.83 | 7.54 / 14.55 | 961 | 30 | 10 | `compare-composite/capacity-composite-k4-G.all.json` |
| 4 | FAIL | FAIL | 0.227 | 0.9989 | 0 | 480 | 0.059 | 0.333 | 0.049 | 3.64 / 3.90 | 8.32 / 11.66 | 1417 | 34 | 13 | `compare-composite/capacity-composite-k4-G.all.json` |

N* griežtai = **1**, N* tol = **1**

### Suvestinė — didžiausias N (N*), kai visi mažesni PASS

Reikšmė „griežtai / tol“. „+“ — didžiausia išbandyta pakopa praėjo (riba nepasiekta); kitu atveju pirmas FAIL ties kita pakopa. Pakopos, kurių nevykdė (VRAM riba arba 2 FAIL iš eilės), nėra N* dalis.

| K | C | H | G |
|---|---|---|---|
| 1 | 2 / 16 | 1 / 4 | 1 / 1 |
| 2 | 4 / 8 | 0 / 4 | 1 / 1 |
| 4 | 4 / 4 | 2 / 4 | 1 / 1 |


## Pastabos ir apribojimai

- **G rezultatai nepatikimi dėl teisingumo klaidų** (aukščiau). G N* = 1 reiškia „nepraėjo N=2 ir N=4“, o ne „GPU nepajėgia“:
  K=2/4 N=4 G eilutėse 0 teisingumo klaidų, bet 0,2–2,8 % praradimų prie nedidelių SM dažnių. G/C santykio prieš
  ištaisant N ≥ 2 neatitikimą ir kartojant su užfiksuotais dažniais nereikėtų daryti.
- **H** K=1 jau N=8 FAIL (`proc` p99 22 ms, artėja prie 30 ms termino).
- **C** gerai skalėjasi mažiems K (K=1 PASS_tol iki N=16), bet K=4 jau N=8 FAIL (11,6 % praradimų, 0,53 branduolio/srautui).
- Nemonotoniškumas (K=1 C N=4 griežtai FAIL, N=8 PASS; K=1 H N=2 FAIL, N=4 PASS) — WSL2 drebėjimas ir GPU dažnio perėjimai;
  pakartojimų nedaryta (viena pakopa vienam variantui). Pasikliauti reikia tendencija, ne pavieniu N.
- `verify`: ≥ 119 mėginių kiekvienoje nepraleistoje pakopoje (išskyrus didelės apkrovos K=4 N=16: C 571, H 135 — `check_dropped`).
- Nevykdyta: v210a išėjimo kampanija, pakartojimai, G be `verify`, G N=32.
- Logo kampanija vyko kitu energijos profiliu (`2026-11-11-gpu-vs-cpu.md`), todėl logo ir composite skaičiai tiesiogiai nepalyginami.
