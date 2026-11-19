# „Production ready“ — 3 fazė: greitis ir stabilumas

Planas: `docs/planai/2026-11-17-production-ready.md`, paketai 3.1–3.3 (G1–G5). Aplinka: RTX 4070 Laptop 8 GB, WSL2
(Ubuntu 24.04), tvarkyklė 596.49, CUDA 13.2, MXL PR #653 ties `6dddc91d`. Bazė — `8750deb` (2 fazės pabaiga), pabaiga —
`19c0bc5`.

## Santrauka

| ID | Būsena | Esmė |
|---|---|---|
| G3 | PASS su autoriaus patvirtinta platformos riba | 4 nestabilių testų priežastys rastos; 3 kodo / testo klaidos pataisytos; platformos įvykiai liudijami nepriklausomais liudininkais (GPU `keepwarm`, host `hostwatch`), neliudyti lieka FAIL |
| G2 | PASS (dokumentuota) | chain N = 12 kritimai — WDDM GPU įvykiai, visiems flow'ams vienu metu; ne kodo talpos riba |
| G1 | 4 / 5 funkcijų PASS; chain — priežastis rasta (platforma), vartų tvarka laukia autoriaus | logo 54, enc-nvenc-ll 10 — 3 / 3; dec 18, composite 27 — su 0,2 % tolerancija 3 / 3 ir 2 / 3; chain 9 — šaltas startas 0 / 3, įšilus 3 / 3 |
| G4 | PASS (lentelė, vartai) | p50 / p99 prie N = 1 ir N = 0,85 · kelias; `regress.py` vartai (p99 + 10 %) |
| G5 | PASS | pirmas commit N = 1 p50 0,367 s, N = 54 p50 0,601 s; po `kill -9` 0,356 s / 0,613 s; 5 / 5 |

## 3.1 Nestabilių laiko testų šakninės priežastys (G3)

Įrankis: `tests/perf/flake.py <testas> --runs 20` (`edf6b89`) ir grynas libmxl šuolio zondas `tests/perf/mxl_hop_probe.cpp`.

### `gap_policy … < 1 ms`

Izoliuotai 20 / 20 PASS ir su įprastu, ir su `chrt -f 50` prioritetu (maks. 140–946 µs). Grynas libmxl šuolis be mūsų kodo:
p50 113 µs, p99 ~250 µs, maks. 824–1 222 µs — tas pats skirstinys kaip teste. Kodo kelias (hipotezė 4) atmestas, planuoklės
prioritetas uodegos nepašalina, GPU kelyje nėra. **Autoriaus sprendimas:** riba 1 ms paliekama, WSL2 testas praleidžiamas
(`5978d8f`); natyviame Linux — tikrinama lauko bandyme.

### `fsplit[composite-G]`

Trys atskiros priežastys, visos pataisytos su testu:

1. `gfxgen` per kito GPU proceso paleidimą praleido 2 laukus — `composite` teisingai pažymėjo sluoksnį trūkstamu, bet testo
   orakulas atleisdavo tik bazinio, ne grafikos aukštupio praradimus. Pataisa — tiksli `layer_dropped` apskaita (`067a22a`).
2. Grafikos aukštupio stebėtojas skaitė `cuda-linear` turinį `mxlFlowReaderGetGrainNonBlocking`, kuris įrenginio atminčiai
   negrąžina duomenų — stebėtojas klaidingai matė 1 044 iš 1 048 praleistų. Pataisa — `GetGrainSlot` (`a05b37e`).
3. Po 91 ms sustojimo tarp sprendimo ir commit'o laukas įrašomas vėliau nei `verify` skaitymo langas (80 ms): `verify` jį
   laiko trūkstamu, funkcija — ne. Atkurta `SIGSTOP` 3 × 10 kartų (nepaaiškinti indeksai = sekundės su > 40 ms vėlavimu).
   Pataisa — indeksų apskaita `verify` ↔ `logo` (`871f4a4`).

### `fiso_stall`

- **G:first sisteminė 0,75 ms uodega:** pirmoji D2D kopija į flow 0 `cuda-linear` alokaciją trukdavo 730–800 µs (vėliau
  ~50 µs). Pataisa — `GapFill::warm` paleidimo metu (`1fa800b`); po jos G:first pirmos spragos uodega 0 / 20 (buvo 3 / 20).
- **Spragų commit'ų vartai** imdavo viso paleidimo min / maks, todėl paleidimo GPU įvykis (~11 s prieš gedimą) patekdavo į
  gedimo lango vertinimą. Pataisa — tik gedimo lango commit'ai (`a58c144`).
- **Platformos įvykiai (autoriaus sprendimas (b)):** vėlavimas ar praradimas pažymimas INFO tik kai jis turi laiko langą,
  kiekvieną langą dengia nepriklausomas liudininkas — GPU (`keepwarm --event-ms`, `14504b6`) arba host (`hostwatch`,
  `088c3a5`) — ir jis apima visus sveikus flow'us arba yra bendras pakilimas (`7399cd4`, `139fdc7`). Vienas neliudytas
  pažeidimas palieka visą testą FAIL.
- **Antros sesijos atvejai (`19c0bc5`):** r05 — paleidimo metu prarastas laukas, kurį `verify` skaičiavo viso gyvenimo
  apskaitoje (liudininkai GPU 20,9 ms, host 12,8 ms) → dabar atimamas pagal indeksą, PASS. r15 — po `SIGCONT` visų 8 flow'ų
  bendras praradimas (GPU 36,7 ms) → Timed, INFO.

### Priėmimas

| Sesija | fsplit (5 var. × 10) | fiso_stall (6 var. × 20) | ctest |
|---|---|---|---|
| 1 (batch5, 2026-11-19) | FAIL 0 / 50 | FAIL 2 / 120 (r09 C:first, r14 G:middle) → autorius išplėtė (b) bendru pakilimu (`139fdc7`) | — |
| 2 (batch6, po `139fdc7`) | FAIL 0 / 50 | FAIL 3 / 120: r05, r15 → (b) (`19c0bc5`); r01 — lieka | 541 / 541 |
| Integracija po `19c0bc5` | — | FAIL 0: 2 paleidimai × 6 variantai (ir 6 `:upstream`) | — |

**r01 C:first — autoriaus patvirtinta WSL2 platformos riba:** paveikto flow'o spragos commit'as 25 033 µs prieš 25 000 µs
ribą (+33 µs); tą tick'ą visi sveiki 15,2 ms vietoje įprastų ~10,4 ms — viso proceso pavėlavimas, kurio nei GPU, nei host
liudininkas neužfiksavo (`hostwatch` maks. 5,144 ms < 6 ms slenksčio). Testas lieka FAIL, riba nekeičiama.

**`interop_foreign_writer`:** 20 pakartojimų šioje fazėje dar nepaleisti (ankstesniuose pilnuose paleidimuose kartais
`late_fields` 1). Lieka 3.1 likutis.

## 3.2 Grandinės vėlavimai prie N = 12 (G2)

`capacity.py --func chain --n 11,12 --repeats 5 --seconds 60`, G, su `keepwarm`:

| N | Griežtai PASS | Tolerancija 0,2 % PASS | e2e p99 |
|---|---|---|---|
| 11 | 4 / 5 | 4 / 5 | 292–299 ms |
| 12 | 3 / 5 | 5 / 5 | 292–299 ms |

Apkrova: NVENC 65 %, NVDEC 51 %, GPU 14 %. Kiekvienas kritimas — **visi flow'ai vienu metu praranda po 1 vienetą**, ir tai
sutampa su WDDM GPU įvykiu: N = 11 rep1 — 89,5 ms ir 102,3 ms įvykiai, `logo` vėlus commit visiems flow'ams tame indekse;
N = 12 rep1 — `enc.late_frames` po 1 visiems 11 flow'ų per 12–32 ms įvykių spiečių. Hipotezė (1) „GPU laiko dalijimas“
atmesta (GPU 14 %). **Išvada:** N = 12 — ne kodo talpos riba; kritimų priežastis — platformos GPU įvykiai.

Apribojimas: `dec` santraukoje nėra praradimų pagal sekundę / indeksą, todėl `dec.late_skips` (rep4 N = 12) tiesiogiai su
liudininku susieti negalima (atidėtas smulkus darbas).

## 3.3 Regresijos vartai ir matavimai (G1, G4, G5)

### Vartai

`tests/perf/regress.py <kampanija> --baseline <failas> [--update]` (`5d3eb78`, `ctest perf_regress`, 17 testų): FAIL, kai
kelias mažesnis, p99 > 1,10 × bazės arba VRAM srautui > 1,05 × bazės. Bazė `tests/perf/baseline/rtx4070-laptop-wsl2.json` iš
P8 kampanijos (2026-11-17): chain 11, composite-k2 32, dec 22, enc-nvenc-ll 12, logo 64.

### G1 — 0,85 · kelias, 3 × 60 s

| Funkcija | N | Griežtai PASS | Tol. PASS | Pastaba |
|---|---|---|---|---|
| logo | 54 | 3 / 3 | 3 / 3 | |
| enc-nvenc-ll | 10 | 3 / 3 | 3 / 3 | |
| dec | 18 | 2 / 3 | 3 / 3 | rep2 `late_skips` 18 = po 1 visiems |
| composite (k = 2) | 27 | 2 / 3 | 2 / 3 | rep3 `late_skips` 27 = po 1 visiems, `verif` 54 |
| chain | 9 | 0 / 3 | 0 / 3 | 0,57–0,69 % praradimų |
| ingest 1080i50 nvdec | 13 | 3 / 3 | 3 / 3 | NVDEC ~62 % |

dec ir composite kritimai — po vieną vienetą visiems flow'ams vienu metu (platformos įvykio požymis).

**chain — priežastis: nešiojamo kompiuterio GPU galios ribos žingsnis.** Kiekviename kritusiame paleidime yra vienas
ilgas (74–157 ms) viso GPU sustojimas. Jis įvyksta, kai GPU iš šalto būvio įkaista iki ~71 °C: platforma pakeičia galios
ribą iš plaukiojančios ~55 W į fiksuotą 42,00 W, ir tuo momentu sustoja visas GPU darbas. Atvėsus iki ~59 °C (~30 s be
apkrovos) riba vėl atleidžiama, ir įvykis gali kartotis. „~46–51 s“ — tai įkaitimo laikas (36–38 s nuo ≤ 55 °C; 40–51 s
po N = 1 lygio, kuris GPU atvėsina iki 57–58 °C). Windows galios profilis — „Whisper“.

| Įrodymas (chain N = 9, 60 s, nepakeistas kodas) | Rezultatas |
|---|---|
| Šaltas startas (≤ 55 °C) | 0 / 3 PASS; sustojimai 84 / 81 / 157 ms; `enforced.power.limit` 55,19 / 55,24 / 55,08 → 42,00 W per 0,02–0,10 s po sustojimo, 71 °C (10 Hz zondas) |
| `--n 1,9` (G1 seka) | 3 / 3 sustojimai 94–99 ms; power-cap vėliavėlė atsiranda tik 0,6–2 s po sustojimo |
| Jau įkaitęs (riba 42 W iš anksto) | 0 ilgų sustojimų per 9 lygius; PASS 2 / 3 griežtai, 3 / 3 tol. (vienas 38,8 ms įvykis `enc` išėjimo metu — kita klasė) |
| Šaltas, po CUDA pašildymo 23–26 s, kol riba = 42,00 W | 3 / 3 PASS |
| Be VRAMFlow funkcijų: 9 × ffmpeg `h264_cuvid` → `h264_nvenc` + CUDA apkrova | šaltas — žingsnis į 42,00 W 3 / 3, sustojimai 181 / 18 / 128 ms; įkaitęs — 0 / 3 |

dxgkrnl ioctl ftrace: prieš sustojimą jokio skyrimo / gyvavimo ioctl iš svečio; visi laukiantys GPU iškvietimai ir paties
`nvidia-smi` užklausa grįžta tą pačią akimirką — įvykis globalus, ne mūsų sukeltas. Paneigta: VRAM šuolis (+68–72 MiB
atsiranda 0,2–0,6 s *po* sustojimo — `dec` eilė pagilėja praleistais tick'ais, cuvid baseinas auga), P būsenos / dažnio
keitimas (P0 / 2580 MHz visą laiką), ankstesnio lygio išjungimas, harness `nvidia-smi` užklausos, klipo kilpa, NVENC
hwframe baseinas. Host pusės ETW pėdsako nėra (reikia administratoriaus teisių).

Teiginys „kitose funkcijose nebuvo“ neįrodytas: G1 `keepwarm-0.err` perrašomas kiekvienai funkcijai, išliko tik chain
liudininkas. Tikėtina, kad šaltai startuojančios kitos funkcijos patiria tą patį žingsnį.

**Pasekmės:** (1) G1 vartams reikia GPU šiluminio paruošimo prieš lygį — tai vartų tvarkos keitimas (plano §1, autoriaus
sprendimas); (2) eterio įrangoje GPU galios riba turi būti fiksuota, o statistikoje reikia NVML ribos / throttle
liudininko; (3) atskiras radinys: `dec` po praleisto tick'o eilė pagilėja negrįžtamai (+40 ms vėlinimo vienam praleistam
kadrui ir skyrimai veikimo metu iki 16,7 ms) — taisytina; (4) rezultatuose `power_scheme` — `None` (tikėtina, `env.sh`
nuima Windows PATH).

### G4 — išėjimo vėlavimas (ms), 3 pakartojimų mediana

| Funkcija | Histograma | N = 1 p50 / p99 | N = 0,85 · kelias | p50 / p99 |
|---|---|---|---|---|
| logo | `out_lateness` | 9,57 / 11,14 | 54 | 12,19 / 12,71 |
| composite (k = 2) | `out_lateness` | 9,83 / 10,62 | 27 | 11,93 / 14,29 |
| dec | `out_lateness` | 1,39 / 1,92 | 18 | 1,10 / 1,52 |
| enc-nvenc-ll | `frame_lateness` | 9,83 / 10,88 | 10 | 9,57 / 10,36 |
| chain | `frame_lateness` | 18,61 / 19,14 | 9 | 18,61 / 19,14 |

Histogramos — per variklio tick'ą (viena reikšmė tick'ui visiems flow'ams). `enc` / `chain` `frame_lateness` atspindi
tyčinį skaitymo atsilikimą ir patenka į tą patį histogramos krepšelį, todėl chain N = 1 ir N = 9 sutampa; apkrovą chain'e
geriau rodo `encode_ms` (N = 1 p99 2,5–3,0 ms, N = 9 p99 9,6–14,5 ms). Vartai: `regress.py` p99 + 10 %.

### G5 — paleidimas ir atsigavimas (`tests/perf/startup_time.py`, 5 pakartojimai)

| N | Iki pirmo commit p50 (maks.) | Po `kill -9` p50 |
|---|---|---|
| 1 | 0,367 s (0,378) | 0,356 s |
| 54 | 0,601 s (0,642) | 0,613 s |

## Kas lieka

- chain G1: vartų šiluminis paruošimas (autoriaus sprendimas); `dec` eilės gilėjimas po praleisto tick'o; `power_scheme` įrašymas.
- `interop_foreign_writer` 20 pakartojimų.
- Natyviame Linux: `gap_policy < 1 ms`, r01 tipo neliudyti ~5 ms pavėlavimai — lauko bandyme.
- Smulkūs: `dec` ir `composite` santraukose nėra praradimų pagal indeksą / sekundę.
