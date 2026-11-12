# Kito proceso GPU įvykiai ir realaus laiko gamintojo vėlavimas (fiso_stall[G] „pliūpsnio“ priežastis)

**Data:** 2026-11-12. **Aplinka:** WSL2 Ubuntu 24.04, viena RTX 4070 Laptop GPU (8 GB, WDDM), CUDA 13.2, MXL 1.2.0 (PR #653),
binarai `/root/build/vramflow/func` (neperkompiliuoti). **Įrankiai:** `tests/perf/gpu_event_hiccup.py` (harness'as),
`tests/perf/gpu_ctx_probe.cpp` (mažiausias CUDA įvykių šaltinis). **Žali duomenys:** `/root/build/vramflow/ghv/`
(`m-G`, `m-C` — pagrindinė matrica; `n-G`, `n2-G`, `n-C` — siaurinimas; `l8-G` — logo N=8; `mit` — mitigacijos;
`fiso/0…9` — fiso_stall pakartojimai; kiekviename — `run.json`, `poll.json`, `smi.json`, `analysis.json`, `table.md`).

## Išvada trumpai

Priežastis — **ne** SIGSTOP ir ne pats proceso startas / pabaiga, o **GPU P-būsenos perėjimas P8 → P3/P4**, kurį sukelia
bet kurio kito proceso trumpas, bet platus GPU darbo pliūpsnis (cudaMemset, užpildymo kernel'is, visą GPU užimantis
kernel'is), kai GPU tuščiąja eiga nuslinkusi į P8 (210 MHz / atmintis 405 MHz). Perėjimo metu **visų** procesų GPU darbas
sustoja 50–350 ms: aukos commit'ai vėluoja, praleidžiami 1–16 indeksų. CPU auka (`--impl cpu --storage host`) tų pačių
įvykių metu nevėluoja niekada — tai GPU, ne OS planuoklio sutrikimas. Nuolatinė lengva GPU apkrova, laikanti GPU virš
P8, sutrikimą pašalino visiškai (0 / 105 įvykių, kontrolė be jos — 29 / 55). **Pataisa (§10):** paleidimo įvykių klasei
„0 / 105“ negalioja be išlygos — su keepwarm stebėtas vienas 37,7 ms vėlus tick'as `logo` starto metu (fiso_stall[G]);
tiksliniame 20 `logo --flows 8` G startų matavime — 0 / 20 (blogiausias 9,7 ms). Su keepwarm paleidimo įvykiai: 1 / 125.

## 1. Nesėkmingo paleidimo laiko juosta (`v13-mine/fiso-logo-G-stall`)

Indeksai — 50 Hz laukai; T(i) = i · 20 ms TAI.

| Kas | Indeksas / laikas |
|---|---|
| gen B pirmas / gen A pirmas | …886491 / …886493 |
| logo startas (t0) | …886508 |
| verify (N=8, 20 s) | …886573 – …887519 |
| SIGSTOP / SIGCONT gen B | t0 + 10 s / t0 + 15 s |
| post verify (10 s) | …887308 – …887754 (baigėsi 2 s prieš pliūpsnį) |
| **logo paskutinis indeksas** | **…887857**; `summary-logo.json` įrašytas 06:28:40.177 |
| **gen A vėlus commit** | **…887858** (34,7 ms) ir …887859 (89,2 ms), 3 praleisti |
| **gen B vėlus commit** | **…887858** (97,1 ms) — tas pats indeksas |
| gen B / gen A paskutinis (stop_all SIGTERM) | …887866 / …887871 |

Pliūpsnis prasidėjo indekse, einančiame iškart po paskutinio logo indekso: T(…858) = 40.160, t. y. ~17 ms prieš logo
santraukos įrašymą, ir **abu** gamintojai (vienas jų ką tik atsigavęs po SIGSTOP) vėlavo tuo pačiu metu. ±0,5 s lange
vienintelis įvykis — `logo` pabaiga (išvalymas po paskutinio tick'o). Kiti procesai tuo metu nei startavo, nei baigėsi.

## 2. Metodas

- Auka — `gen` 7 flow'ai 1080i50, G (gpu + cuda-linear) arba C (cpu + host), atskiras domenas `/dev/shm/vf-ghv-<hex>`.
- **Kiekvieno indekso** vėlavimas: pasyvus stebėtojas (atskiras Python procesas, be CUDA) kas ~1 ms skaito paskutinio aukos
  flow'o `data` failą (MXL flow būsena: poslinkis 200 — head indeksas, 208 — paskutinio rašymo TAI ns; patikrinama
  paleidžiant). Vėlavimas = rašymo laikas − T(i); head šuolis > 1 — praleisti indeksai. Patikra: 140 gen'o paties
  `vėlus commit` eilučių sutapo su stebėtojo verte 0,0 ms tikslumu; stebėtojas nė karto neužmigo > 5 ms (OS sutrikimų nebuvo).
- Įvykiai su ≥ 5 s tarpais, kiekvieno TAI pradžia / pabaiga; langas [pradžia − 0,2 s, pabaiga + 1,0 s]. Natūraliai
  pasibaigusiam procesui pradžia — santraukos įrašymo laikas (po jo seka flow'ų ir CUDA išvalymas).
- nvidia-smi (PATH, tada `/usr/lib/wsl/lib/nvidia-smi`) kas 100 ms: pstate, taktai — visuose paleidimuose, ir baziniuose.
- Pliūpsnis = lange vėlavimas > 30 ms (`--deadline-ms`) arba praleistas indeksas.

## 3. Pagrindinė matrica (10 kartų kiekvienas raundas + 3 baziniai po 60 s)

| Įvykio klasė | G auka: įvykių / pliūpsnių / blogiausias ms / praleista | C auka: įvykių / pliūpsnių / blogiausias ms / praleista |
|---|---|---|
| (a) antras gen G startas | 30 / 2 / 30,8 / 1 | 30 / 0 / 5,9 / 0 |
| (b) jo SIGTERM pabaiga | 20 / 0 / 7,0 / 0 | 20 / 0 / 5,9 / 0 |
| (c) kill -9 | 10 / 0 / 0,8 / 0 | 10 / 0 / 5,3 / 0 |
| (d) verify startas | 10 / 1 / 14,0 / 1 | 10 / 0 / 6,2 / 0 |
| (e) verify pabaiga | 10 / 0 / 0,9 / 0 | 10 / 0 / 6,0 / 0 |
| (f) logo N=1 startas / pabaiga | 10 / 0 / 19,9 / 0 · 10 / 0 / 1,1 / 0 | 10 / 0 / 5,5 / 0 · 10 / 0 / 6,0 / 0 |
| (f) logo N=7 startas / pabaiga | 10 / 1 / 18,6 / 1 · 10 / 0 / 1,4 / 0 | 10 / 0 / 5,5 / 0 · 10 / 0 / 5,5 / 0 |
| (g) zondas: exec / konteksto sukūrimas / sunaikinimas / pabaiga | 40 / 0 / 26,3 / 0 | 40 / 0 / 5,4 / 0 |
| (h) cudaMalloc + memset 64 MB / 512 MB / 2 GB | 10 / 0 / 1,1 · **10 / 10 / 257,6 / 67** · 10 / 0 / 27,6 | 30 / 0 / 5,9 / 0 |
| (h) cudaFree 64 / 512 / 2048 MB | 30 / 0 / 6,0 / 0 | 30 / 0 / 5,9 / 0 |
| (h) gyvo zondo exec / kontekstas / pabaiga | 30 / 0 / 20,4 / 0 | 30 / 0 / 6,0 / 0 |
| (i) antras gen C startas / SIGTERM | 20 / 0 / 6,0 / 0 | 20 / 0 / 5,8 / 0 |
| (i) tik CPU procesas (512 MB host; exec / užpildymas / pabaiga) | 30 / 0 / 5,8 / 0 | 30 / 0 / 5,8 / 0 |
| (j) SIGSTOP / SIGCONT antro gen G | 20 / 0 / 6,4 / 0 | 20 / 0 / 6,0 / 0 |
| **Iš viso** | **340 / 14** | **340 / 0** |
| Baziniai 3 × 60 s (be įvykių) | 0 pliūpsnių, max 7,4 ms | 0, max ~6 ms |

Papildomai: logo N=8 (7 aukos flow'ai + antro gen flow'as, 27 s kaip fiso_stall), 10 kartų — startas / pabaiga 0 / 0
pliūpsnių (max 8,2 ms). Visi 97 blogi G aukos indeksai buvo įvykių languose; savaiminių (už langų) — 0 per ~46 min G aukos.

## 4. Siaurinimas: kuri fazė?

Gyvas zondas su jau sukurtu kontekstu, viena operacija per įvykį (G auka; C auka — kontrolė, 65 įvykiai, 0 pliūpsnių).

| Operacija | įvykių | pliūpsnių | blogiausias ms | praleista |
|---|---:|---:|---:|---:|
| tik cudaMalloc 512 MB / 2 GB (be memset) | 10 / 5 | 0 / 0 | 1,1 / 0,7 | 0 |
| cudaMemset 512 MB jau rezervuotam bloku (1-as ir 2-as kartas) | 20 | 20 | 296,7 | 149 |
| cudaMalloc + memset 512 MB po free | 10 | 10 | 273,5 | 58 |
| memset 2 GB / 256 / 128 / 64 / 16 MB | 5 / 5 / 5 / 5 / 5 | 5 / 4 / 3 / 3 / 1 | 302 / 331 / 358 / 210 / 94 | 50 / 30 / 38 / 24 / 3 |
| memset 512 MB dalimis po 16 MB / 64 MB su sync | 5 / 5 | 5 / 5 | 206 / 162 | 38 / 15 |
| savas užpildymo kernel'is 512 MB / 2 GB | 5 / 5 | 5 / 5 | 320 / 279 | 57 / 45 |
| 1 gijos spin kernel'is 5…200 ms | 50 | 0 | 14,4 | 0 |
| visos GPU spin (4096 × 256) 50 / 200 ms per bloką | 5 / 5 | 1 / 5 | 185 / 277 | 8 / 51 |
| cudaFree | 25 | 0 | 5,9 | 0 |

Ką tai rodo: rezervavimas, atlaisvinimas, kontekstai ir signalai nevėlina; vėlina **platus GPU darbas**, nepriklausomai
nuo to, ar tai cudaMemset, ar savas kernel'is, ir beveik nepriklausomai nuo dydžio (16 MB kartais, ≥ 64 MB dažniausiai).
Pats operacijos laikas pliūpsnio metu išsitęsia iki to paties 100–360 ms (pvz. memset 128 MB: 7–8 ms be pliūpsnio,
330–360 ms su juo) — sustoja visa GPU, ne tik auka. 1 gijos kernel'is net 200 ms nevėlina.

**P-būsenos koreliacija** (nvidia-smi kas 100 ms, visi G paleidimai be mitigacijų): 84 iš 94 pliūpsnių turi P8 → P3/P5
perėjimą ±0,6 s (nvidia-smi jį parodo +70…+320 ms nuo pliūpsnio pradžios, t. y. jo pabaigoje); P8 → P3 perėjimų su
pliūpsniu — 80 / 156; perėjimų žemyn (P5 → P8) su pliūpsniu — 22 / 246 (gretimi įvykiai); visi pliūpsniai prasidėjo
būsenoje P8 (210 MHz, atmintis 405 MHz). C aukai tie patys 87 P8 → P3 perėjimai — 0 pliūpsnių.

## 5. fiso_stall[G] pakartojimas (10 × `functional.py --only fiso_stall`, FISO_VARIANTS=G, su nvidia-smi žurnalu)

Kontroliniai paleidimai — 10 / 10 be vėlių commit'ų. Su gedimu — 2 / 10 su pliūpsniu, abu kartus **abu** gen'ai tuo
pačiu indeksu:

| Paleidimas | gen A / gen B | Kada | P-būsena |
|---|---|---|---|
| 5 | 253,0 / 252,8 ms, 11 praleista | logo paskutinis indeksas + 3; T(i) = 29 ms po logo santraukos | P5 → P8 −520 ms, **P8 → P3 +291 ms** |
| 7 | 116,4 / 116,5 ms, 4 praleista | 217 ms po SIGCONT = post `verify` startas | **P8 → P3 +185 ms** |

Kituose 8: logo pabaiga be pliūpsnio, P8 → P3 perėjimas vyko +770…+890 ms (arba GPU tuo metu nebuvo P8). Tai ta pati
priežastis kaip §4: fiso_stall'e GPU spėja nuslinkti į P8 (paskutinėmis logo sekundėmis apkrova maža), o išvalymas /
naujo proceso startas paleidžia darbo pliūpsnį. Kad pliūpsnis kartais „prie logo pabaigos“, kartais „prie verify starto“ —
lemia ne įvykio rūšis, o tai, ar tuo metu GPU P8 ir ar perėjimas sutampa su aukos darbu.

## 6. Mitigacijos (be produkcinio kodo pakeitimų)

| Bandymas | įvykių | pliūpsnių | blogiausias ms |
|---|---:|---:|---:|
| Kontrolė: memset / fill / spinw seka be mitigacijos (5 paleidimai) | 55 | 29 | 287,2 |
| Ta pati seka + `gpu_ctx_probe busy 64 5` (užpildymas 64 MB kas 5 ms) | 50 | **0** | 13,2 |
| lifecycle (gen2, verify, logo 1/7) + busy | 55 | **0** | 18,7 |
| baziniai 2 × 60 s + busy | — | 0 | 6,3 |
| lifecycle + gyvas CUDA kontekstas kitame procese (`--keep-ctx`) | 55 | 0 | 18,9 |
| Išankstinis rezervavimas (memset jau rezervuotam blokui) | 20 | 20 | 296,7 |
| memset dalimis po 16 / 64 MB | 10 | 10 | 206 |

- **Nuolatinė lengva apkrova** laiko GPU P3/P4 (6–7 GHz atmintis): pliūpsniai 29 / 55 → 0 / 105. Kaina: GPU galia
  ~4,5–9 W (P8) → ~11–14 W (P3/P4), apie +6 W (nvidia-smi, 10 s su 7 flow'ų auka); aukos max be įvykių 6,3 ms. Tai įrodo priežastį (decisyvus bandymas) ir yra
  veikiantis apėjimas šioje mašinoje.
- Gyvas kontekstas: 0 / 55, bet ir be jo lifecycle pliūpsnių dažnis žemas (4 / 100) — **neįrodyta**, kad padeda.
- Išankstinis rezervavimas ir dalijimas dalimis — **nepadeda** (vėlina darbas, ne rezervavimas).
- `CUDA_DEVICE_SCHEDULE` tipo nustatymai — netirta: tai `cudaSetDeviceFlags` (proceso vidaus sinchronizacijos būdas), o
  sutrikimas — visos GPU pauzė; be kodo keitimo nustatyti nėra kaip.

## 7. Kas įrodyta / neįrodyta / čia netikrinama

**Įrodyta (šioje mašinoje):** GPU aukos 50–350 ms vėlavimai su praleistais indeksais atsiranda tik tada, kai kitas
procesas paleidžia platų GPU darbą, GPU būnant P8; tai visos GPU pauzė (vėluoja ir pats įvykio procesas, ir abu gen'ai
fiso_stall'e); CPU aukai — 0 iš 405 įvykių; laikant GPU virš P8 — 0 iš 105. SIGSTOP / SIGCONT, kill -9, SIGTERM,
konteksto sukūrimas / sunaikinimas, cudaMalloc, cudaFree, proceso startas / pabaiga patys savaime nevėlina (max < 30 ms).
Kodėl starto įvykiai kartais vėlina (3 / 80): cuda-linear flow'o kūrimas daro `cudaMemset` viso žiedo
(`CudaLinearBackend.cpp:1277`, 28 MB / flow'ui) — tai platus darbas, kaip §4 memset.

**Neįrodyta:** tikslus vidinis mechanizmas (atminties takto perjungimas P8 405 MHz → 7 GHz, WDDM / GPU planuoklio būsena
ar kas kita) — matome tik koreliaciją su P-būsena 100 ms tikslumu ir mitigacijos poveikį; kas konkrečiai logo išvalyme
paleidžia darbą (atvaizdavimų atjungimas, VRAM valymas tvarkyklėje) — nematuota atskirai. Tik ~51 % P8 → P3 perėjimų
sukelia pliūpsnį — kas lemia likusius, nežinoma.

**Netikrinta / negalima čia:** natyvus Linux (be WDDM / dxgkrnl), MPS, serverinės GPU (pastovūs taktai, persistence
mode, `nvidia-smi -lgc/-lmc` — taktų fiksavimas reikalauja administratoriaus teisių ir sistemos nustatymo keitimo, todėl
nedaryta), Windows HAGS nustatymas (nekeistas). Windows pusės kitų GPU klientų (naršyklė, kompozitorius) įtaka — ta pati
mechanika būtų tikėtina, bet nematuota.

## 8. Ką tai reiškia eksploatacijai

- Bendroje GPU (bent WSL2 / WDDM nešiojamame) **paleidžiant ar stabdant** kitą funkciją, kai GPU tuščiąja eiga P8,
  eteryje esančios GPU funkcijos gali sustoti iki ~0,35 s (17 laukų 50 Hz) — ne dėl izoliacijos klaidos funkcijoje.
  Funkcijų izoliacija (V10) teisinga: priskyrimo vartai tai jau rodo kaip aukštupio (`gen A`) sutrikimą.
- Praktinė taisyklė šiai mašinai: kol yra gyvas eteris, laikyti GPU virš P8 (nuolatinė lengva apkrova arba realus
  eterio krūvis, kuris pats neleidžia nuslinkti į P8); sunkūs inicializavimai (flow'ų kūrimas, dideli memset) — prieš
  eterį arba kai GPU jau „pabudusi“.
- Serveriams (natyvus Linux, serverinės GPU) reikia **pakartoti** šį bandymą (`gpu_event_hiccup.py`
  veikia be pakeitimų) ir patikrinti, ar pakanka persistence mode + fiksuotų taktų; tik tada rašyti „start/stop be
  poveikio“ į eksploatacijos reikalavimus.
- `fiso_stall[G]:upstream` INFO eilutėse vėlavimas, sutampantis su P8 → P3, yra aplinkos įvykis; vartų taisyklė
  (praradimai ≤ aukštupio `late_skips + late_commits`) jį jau teisingai priskiria.

## 9. Įrankis `vramflow-func keepwarm` ir harness'ai (po šio tyrimo)

Mitigacija tapo gamybiniu įrankiu: `vramflow-func keepwarm` (numatyta `--size-mb 64 --interval-ms 5`, `--duration 0` — iki
SIGINT / SIGTERM; kol veikia — tylus, pabaigoje viena santraukos eilutė). Užpildymas — `cudaMemsetAsync` + sync kas 5 ms
(absoliutus periodas). `functional.py` ir `capacity.py` jį paleidžia kiekvienam GPU variantui (H, G) ir sustabdo prieš
kiekvieną C variantą; `VRAMFLOW_GPU_KEEPWARM=0` išjungia; kiekvienas rezultatas turi lauką `keepwarm`.

**Kaina** (nvidia-smi kas 100 ms, 40 s, be kitų procesų; pakartota 2026-11-12 vakare tuo pačiu `kw_cost` scenarijumi,
žali duomenys `/root/build/vramflow/kw-cost/`). **Vienas skaičius, naudojamas visur (README, `capacity.py`): GPU užimtumas
19,5 %, galia +8,5 W (2,8 → 11,3 W).** Ankstesnis kodo komentaras „~13 %, ~+6 W“ buvo iš 10 s matavimo su 7 flow'ų auka (§5),
ne iš šio:

| | P-būsena | GPU užimtumas | galia | atminties taktas |
|---|---|---:|---:|---:|
| tuščia eiga | P8 (387 / 387 imčių) | 0 % | 2,8 W (p95 2,8) | 405 MHz |
| keepwarm 64 MB / 5 ms | P4 (379), P3 (8) | 19,5 % (maks. 21) | 11,3 W (p95 12,8) | 6027 MHz |

Vienas užpildymas — vid. 1,21 ms, maks. 2,49 ms (8969 užpildymai per 44,8 s, nė vienas ne ilgesnis už periodą). Pirmasis
matavimas (tas pats scenarijus): 2,6 → 11,2 W, 19,5 % — skirtumas tik tuščiosios eigos galioje (±0,2 W).

**Poveikis** — lemiamas bandymas pakartotas su pačiu įrankiu (`gpu_event_hiccup.py --keep-warm 64,5`; G auka 7 flow'ai;
seka `mallocraw 512; memset 512 ×2; fill 512; memset 128; memset 64; spinw 50; free` + zondo kontekstas / pabaiga — 10
įvykių paleidimui, 7 paleidimai kiekvienai pusei; žali duomenys `/root/build/vramflow/kw-effect/`):

| | įvykių | pliūpsnių (> 30 ms arba praleista) | blogiausias ms | praleista indeksų | blogų indeksų už langų |
|---|---:|---:|---:|---:|---:|
| be keepwarm | 70 | **35** (fill 7 / 7, memset 512 14 / 14, memset 128 5 / 7, memset 64 2 / 7, spinw 50 7 / 7) | 367,5 | 290 | 0 |
| su keepwarm | 70 | **0** | 21,5 (konteksto sukūrimas) | 0 | 0 |

Kas lieka ir su keepwarm: kito proceso įvykiai (CUDA konteksto sukūrimas, platus darbas) vis tiek prailgina kitų GPU darbą
iki ~20 ms (čia 21,5 ms; fiso_stall[G] — logo `proc_ms` maks. 14,3 ms, kai ties SIGCONT startuoja post `verify`). Tai jau
ne praradimai (< 30 ms), bet vėlavimo maksimumai. Izoliacijos vartai jų nebeatleidžia Δproc nariu (jis atleisdavo ir
tikrinamos funkcijos pačios sulėtėjimą — žr. `tests/functional/isolation.py`, `fiso_teeth`): post `verify` paleidžiamas prieš
gedimą (jo CUDA kontekstas kuriamas ne gedimo lange), o riba lyginama su to paties paleidimo langu prieš gedimą.

**Vartų patikimumas** (PASS / paleidimų; C variantas GPU nenaudoja — keepwarm jam niekada nepaleidžiamas, todėl „be“ —
tik G; žali duomenys `/root/build/vramflow/meas/`, kiekvienas `functional.json` įrašas turi `keepwarm`):

| Vartai | C (keepwarm niekada) | G su keepwarm | G be keepwarm |
|---|---:|---:|---:|
| fiso_kill | 5 / 5 | 5 / 5 (pirminė maks. riba: 4 / 5) | 5 / 5 |
| fiso_stall | 5 / 5 | 5 / 5 | 5 / 5 (pirminė maks. riba: 4 / 5) |
| fiso_composite | 5 / 5 | 5 / 5 | 5 / 5 |
| fiso_enc (su garsu) | 5 / 5 | 5 / 5 | 5 / 5 |
| fgold_logo | 10 / 10 | 10 / 10 | 10 / 10 |
| fgold_composite | 10 / 10 | **10 / 10** | **8 / 10** (6 `late_fields` + 2 / 16 `missing_fields`) |
| fgold_dec | 10 / 10 | 10 / 10 | 10 / 10 |

Žinomas nestabilumas `fgold_composite[G]` (≈ 1 iš 5, 2–8 vėlių laukų) be keepwarm pasikartojo 2 / 10, su keepwarm — 0 / 10.
Du fiso G nepavykimai buvo tik naujojo maks. vėlavimo vartų (praradimų 0): sveikas aukštupis (gen A) tuo metu pats vėlavo
17,0 / 22,4 ms (kontroliniame 6,9 / 9,8 ms). Riba papildyta Δup (aukštupio maks. vėlavimo padidėjimas, kaip praradimų
biudžetas); stulpeliuose — su ja perskaičiuota tų pačių įrašų (`before` / `after`) vartų išvada, skliaustuose — pirminė.

**`fgold_composite[G]`: 30 su keepwarm prieš 30 be** (2026-11-12 vakaras, pakaitomis su / be, tas pats binaras ir parinktys;
`tests/functional/functional.py --only fgold_composite`, `FGOLD_VARIANTS=G`, be keepwarm — `VRAMFLOW_GPU_KEEPWARM=0`; žali
duomenys `/root/build/vramflow/kw30/`, santrauka `results.txt`):

| | PASS | FAIL | kas krito |
|---|---:|---:|---|
| su keepwarm | **30 / 30** | 0 | — |
| be keepwarm | **27 / 30** | 3 | 2 `late_fields` + 2 `in_invalid`; 6 `late_fields` + 14 `missing_fields`; 6 `late_fields` + 4 `missing_fields` (visi — composite praradimai, verify teisingumo klaidų 0) |

Du paleidimai (vienas „su“, vienas „be“) nesiskaičiuoja ir pakartoti: jų metu buvo pakeisti santraukos raktų failai (`FAIL`
tik dėl raktų, praradimų 0). Išvada: be keepwarm šis nešiojamas GPU vartų nepraeina ~1 iš 10 paleidimų (anksčiau 2 / 10),
su keepwarm — 0 / 30. Todėl G vartai vykdomi su keepwarm, o `fgold_composite[G be keepwarm]` — tik INFO ranka (paleidžiama
kiekvieną kartą, vartų neveikia), kad nešiojamo GPU elgsena liktų matoma.

## 10. Paleidimo įvykis su keepwarm (pataisa „0 / 105“, 2026-11-12 vakaras)

**Kas nutiko.** `fiso_stall[G]` (savikontrolės vartai, keepwarm įjungtas) — sveikas gamintojas `gen` G parašė 37,7 ms vėlų
tick'ą (indeksas …916948) ~0,5 s po `logo --flows 8` G starto, 9,5 s prieš SIGSTOP; kiekvienas sveikas flow'as gavo po 1
`late_fields`, vėlavimo maks. 68,7 ms. Vartai lygino viso gyvenimo funkcijos praradimus su gedimo lango aukštupio biudžetu,
todėl paleidimo vėlavimas buvo priskirtas gedimui (FAIL). Tai **ne** izoliacijos klaida — bet tai keepwarm įvykis, kurio
„0 / 105“ neapėmė: tos lentelės įvykiai buvo kito proceso CUDA darbas jau veikiant aukai, o čia — funkcijos startas su 8
cuda-linear išėjimų kūrimu (8 × 28 MB `cudaMemset`, §7) ir `verify` startas po ~1 s.

**Tikslinis matavimas.** 20 `logo --flows 8` G startų šalia veikiančio `gen` G (7 + 1 flow'as), keepwarm 64 MB / 5 ms
įjungtas (`gpu_event_hiccup.py run … --victim G --rounds lifecycle --only logo8 --logo-n 8 --reps 20 --baseline 0
--keep-warm 64,5`; žali duomenys `/root/build/vramflow/ghv/kw-logo8/`; pasyvus stebėtojas — kiekvieno indekso vėlavimas):

| įvykis | įvykių | pliūpsnių > 30 ms | blogiausias ms | mediana ms | praleista |
|---|---:|---:|---:|---:|---:|
| `gen2G_start` (8-to įėjimo gamintojas) | 20 | 0 | 18,2 | 4,2 | 0 |
| **`logo8_start`** | **20** | **0** | **9,7** | 1,8 | 0 |
| `logo8_exit` | 20 | 0 | 1,4 | 0,9 | 0 |
| `gen2G_term` | 20 | 0 | 28,2 | 2,9 | 0 |
| fonas (be įvykių langų) | 21 743 indeksai | 0 | — | — | 0 |

**Išvada.** Su keepwarm paleidimo įvykių klasė — **1 / 125** pliūpsnis > 30 ms (fiso_stall[G] 1 iš 105 + 20 tikslinių 0 / 20;
tikslinis 0 / 20 reiškia tik, kad dažnis < ~15 % su 95 % pasikliautinumu (3 / n taisyklė), ne nulį). Paaiškinimas lieka §7: keepwarm sumažina,
bet nepašalina kito proceso plataus GPU darbo poveikio (§9: ~20 ms ilgėjimai lieka); retas 37,7 ms — tos pačios klasės
uodega. **Vartų pakeitimas** (`tests/functional/isolation.py`): funkcijos praradimai skaičiuojami tik gedimo lange (santraukos
`loss_s` — praradimai kiekvienai indekso laiko sekundei), tas pats langas kaip aukštupio biudžetas; kontrolinis langas
prasideda po funkcijos įšilimo (`--warmup` 5 s, buvo 3 s) **ir** po visų procesų paleidimo + 2 s; paleidimo vėlavimai
nebepriskiriami gedimui ir neiškreipia kontrolės.

## Atkartojimas

```
export FUNC_BIN=/root/build/vramflow/func/vramflow-func PROBE_BIN=/root/build/vramflow/perf-probe/gpu_ctx_probe
export LD_LIBRARY_PATH=/root/src/mxl-pr653/install/Linux-Clang-Release/lib:/root/build/vramflow/func
/usr/local/cuda-13.2/bin/nvcc -O2 -std=c++17 -x cu -o $PROBE_BIN tests/perf/gpu_ctx_probe.cpp
python3 tests/perf/gpu_event_hiccup.py run OUT --victim G --rounds lifecycle,ctxcpu,alloc --reps 10 --baseline 3
python3 tests/perf/gpu_event_hiccup.py run OUT --victim G --rounds narrow --reps 5 --baseline 0 --tag=-busy --keep-busy 64,5
python3 tests/perf/gpu_event_hiccup.py analyze OUT
# §9: tas pats su įrankiu ir vartų patikimumas
python3 tests/perf/gpu_event_hiccup.py run OUT --victim G --rounds narrow --reps 7 --baseline 0 --tag=-kw --keep-warm 64,5   --narrow "mallocraw 512;memset 512;memset 512;fill 512;memset 128;memset 64;spinw 50;free"
VRAMFLOW_GPU_KEEPWARM=0 FISO_VARIANTS=G python3 tests/functional/functional.py OUT --only fiso_kill,fiso_stall,fiso_composite,fiso_enc
```
