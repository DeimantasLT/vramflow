# VRAMFlow (lietuviškai)

**MXL media flow'ai GPU atmintyje ir jų apdorojimas ten pat — su tuo pačiu laiku ir elgsena kaip MXL ant CPU.**

Išsamus aprašas su visomis komandomis ir kodo pavyzdžiais — [README.md](README.md) (anglų k.). Čia — trumpai, ta pačia
tvarka.

[MXL](https://github.com/dmf-mxl/mxl) leidžia media funkcijoms keistis nesuspaustu vaizdu, garsu ir duomenimis per
bendrą atmintį (įprastai — RAM). Su `cuda-linear` backend'u iš
[dmf-mxl/mxl#653](https://github.com/dmf-mxl/mxl/pull/653) flow'as gali gyventi GPU atmintyje (VRAM), todėl viena GPU
funkcija perduoda kadrus kitai nekopijuodama per CPU.

| Dalis | Kas tai |
|---|---|
| `vramflow-func` | Paruoštos media funkcijos ant MXL flow'ų: generatorius, logotipas, H.264 dekodavimas, H.264 + AAC kodavimas, alfa grafikos kompozitorius, išėjimo patikra. Kiekviena veikia ant CPU arba GPU. |
| `libvramflow-gfx` | C biblioteka: RGBA kadrai iš bet kurios programos → standartinis MXL grafikos flow'as (`video/v210a`, fill + key). |
| `vramflow-verify` | Bandomasis writer'is ir reader'is, tikrinantys kiekvieną grain'ą (turinys, indeksas, laikas). |
| Testai ir įrankiai | Vienetų, GPU ir funkciniai testai; talpos matavimas (kiek flow'ų telpa). |

Viskas C++ / CUDA; Python — tik testų ir matavimų skriptai.

## 1. Pagrindinės sąvokos

- **Domenas** — katalogas bendroje atmintyje (`/dev/shm/...`); visi jį naudojantys procesai mato vienas kito flow'us.
- **Flow** — vienas srautas (pvz. vienas video signalas) su UUID ir aprašu.
- **Grain** — vienas flow'o vienetas; čia — vienas video **laukas** (1080i50 = 50 laukų/s). Indeksas skaičiuojamas iš
  TAI laiko, todėl visi procesai sutaria, kuris grain'as kuriam momentui.
- **Writer / reader** — writer'is sukuria flow'ą ir commit'ina grain'us, reader'ių gali būti kiek nori.
- **Saugykla** — `host` (RAM) arba `cuda-linear` (GPU atmintis). API abiem ta pati.
- **v210 / v210a** — 10 bitų 4:2:2 video; `v210a` turi dar key (alfa) plokštumą.

Trys variantai, renkami dviem parametrais:

| Variantas | `--impl` | `--storage` | Kada |
|---|---|---|---|
| **C** — CPU | `cpu` | `host` | Be GPU arba palyginimui su dabartine sistema |
| **H** — GPU, RAM flow'ai | `gpu` | `host` | GPU apdorojimas, bet kitoms programoms flow'ai reikalingi RAM'e |
| **G** — GPU, GPU flow'ai | `gpu` | `cuda-linear` | Geriausia: kadrai lieka GPU |

Visi trys duoda bitų tikslumu vienodą rezultatą (tikrinama prieš vieną nepriklausomą etaloną).

## 2–3. Reikalavimai ir surinkimas

Ubuntu 24.04 (natyviai arba WSL2), GPU variantams — NVIDIA GPU ir CUDA 13.x; Clang 19, CMake, Ninja, vcpkg, FFmpeg dev
bibliotekos. Sistemos TAI poslinkis turi būti 37 s. Surinkimas — trys žingsniai, žr. README.md 3 skyrių
(MXL su PR #653 → `10-build-mxl.sh` → `30-build-verify.sh -DMXLV_BUILD_FUNC=ON`).

## 4. Greita pradžia

README.md 4 skyriuje — penki pavyzdžiai, kuriuos galima nukopijuoti:

1. pirmas writer'is ir reader'is RAM'e (`vramflow-verify writer/reader --storage host`);
2. tas pats GPU atmintyje (`--storage cuda-linear`);
3. media funkcija: `gen` → `logo` → `verify` (be GPU: `--impl cpu --storage host`);
4. grafika su skaidrumu ant video: `gen` + `gfxgen` → `composite` → `verify`;
5. domeno turinys: `mxl-info -d <domenas> -l`.

## 5. Savo programoje

- **MXL writer'is / reader'is GPU atmintyje** — tas pats MXL API; writer'is paduoda
  `{"payload":{"backend":"cuda-linear","deviceUuid":"GPU-…"}}`, grain'o adresą duoda `mxlCudaLinearDevicePointer()`.
  Pavyzdys README.md 5.1, pilnas kodas `src/mxlio/verify_writer.cpp` ir `verify_reader.cpp`.
- **Grafika iš bet kurios programos** — `libvramflow-gfx` (`include/vramflow/gfx.h`, sutartis angliškai): flow aprašas
  per `vfgfx_flow_def`, `vfgfx_writer_create`, tada `vfgfx_submit` / kadro tikslumo `vfgfx_submit_at(index, …)` (biblioteka
  pati laiko ritmą) arba `vfgfx_write(index, …)` (ritmą valdai pats); RGBA arba BGRA kadrai, host arba CUDA atmintis
  (pasirinktinai su kviečiančiojo `cuda_stream`), būsena — `vfgfx_get_stats`. Leidimas 0.2.0, SONAME
  `libvramflow-gfx.so.1` (ABI epocha, simboliai `VFGFX_1`). Pavyzdys README.md 5.2.
- **Pavyzdys: OGraf grafika ant video** — `examples/ograf/` (README.md 5.3): headless naršyklė su permatomu fonu atvaizduoja
  [EBU OGraf](https://github.com/ebu/ograf) grafiką, `ograf_feed` per `vfgfx_submit` (ritmo režimas) paverčia ją `v210a`
  flow'u, `composite` uždeda ant `gen` video. Gyvam renderiui panaudotas jau įdiegtas Windows Edge (~4 kadrai/s);
  oficialių EBU pavyzdinių grafikų neatsisiųsta; kiti renderiai (pvz. GStreamer `wpesrc`) jungiami per stdin.
- Grafikos flow'as elgiasi kaip bet kuris kitas: sustojus šaltiniui, kompozitorius nieko neišgalvoja — tas išėjimo
  grain'as necommit'inamas, šaltiniui grįžus darbas tęsiasi.

## 6. Komandos

`vramflow-func gen | logo | dec | enc | gfxgen | composite | verify` — lentelė su pagrindiniais parametrais README.md
6 skyriuje; visi parametrai — `--help`. Kiekvienas paleidimas rašo `summary-<komanda>.json`, išėjimo kodas `0` — viskas
teisinga.

Paleidimo ir sustabdymo taisyklės (plačiau — README.md, „Start order, waits and restarts“ ir „Domain ownership“):

- **Paleidimo tvarka nesvarbi.** Skaitančios funkcijos (`logo`, `composite`, `verify`, `enc`) įėjimų laukia:
  `--input-wait-ms 0` (numatyta) — neribotai, kas 5 s rašo `laukiu flow …` (ir kodėl esamas flow netinka);
  `--input-wait-ms N` — ne ilgiau kaip N ms, tada išėjimo kodas `4` su trūkstamų flow'ų sąrašu.
- **Gamintojas dingo ir grįžo.** Skaitytojas tęsia darbą, spragos indeksus suskaičiuoja (`missing_fields`), grįžus
  gamintojui prisijungia pats (`reattached`). Grįžo kitu formatu — `--on-format-change exit` (numatyta, kodas `6`) arba
  `wait`. Spragos indeksai į paties skaitytojo išėjimą kol kas nerašomi (INVALID grain'ai — plano P5 dalis).
- **Domeno nuoma.** Kiekvienas `vramflow-func` procesas laiko `<domenas>.vramflow/<run-id>.lease` šalia domeno (`flock`,
  atsilaisvina ir po `kill -9`); `--run-id` — paleidimo vardas. Bandant sukurti flow, kurį jau rašo kitas gyvas
  paleidimas, klaidoje nurodomas jo run ID. Funkcijos domenų netrina; testų įrankiai (`tests/perf/domain.py`) kiekvienam
  paleidimui kuria atskirą domeną ir trina jį tik tada, kai nebėra gyvų savininkų.
- **Domeno kelias GPU (`cuda-linear`) flow'ams — ne ilgesnis kaip 48 simboliai** (skaitytojo lizdo
  `<domenas>/<flow>.mxl-flow/payload.sock` kelias turi tilpti į 107 simbolius); ilgesnį `vramflow-func` atmeta iškart.

## 8. Rezultatai (RTX 4070 Laptop 8 GB, WSL2 — orientaciniai)

| Funkcija | CPU (C) N\* | GPU su GPU flow'ais (G) N\* | CPU branduoliai flow'ui, C → G |
|---|---|---|---|
| Logotipo įkepimas, MXL → MXL | 16 | ≥ 64 (riba nepasiekta) | 0,057 → 0,005 |
| H.264 dekodavimas, TS/UDP → MXL | 8 | 24 (16 griežtai) | ~0,55 → ~0,04 |
| Pilna grandinė: TS → dekodavimas → logotipas → H.264 + AAC → TS | 4 | ≥ 12 (riba nepasiekta) | ~2,0 → ~0,33 |

N\* — didžiausias flow'ų skaičius, kai ši ir visos mažesnės pakopos praėjo (0 teisingumo klaidų, praradimai ≤ 0,2 %).
Pilnoje grandinėje ties N=12 riba — vienintelis NVENC variklis, CUDA branduoliai užimti ~10 %. Visos lentelės:
[`docs/rezultatai/2026-11-11-gpu-vs-cpu.md`](docs/rezultatai/2026-11-11-gpu-vs-cpu.md). Grafikos kompozitoriaus talpos
matavimai (N flow'ų × K sluoksnių):

Grafikos kompozitorius (`composite`), bazė + K sluoksnių → `v210`, 60 s pakopa, terminas 30 ms, tas pats N\* (tolerantiškai / griežtai):

| Sluoksnių K | C — CPU | H — GPU, RAM flow'ai | G — GPU, GPU flow'ai |
|---|---|---|---|
| 1 (logotipas 384×216) | 16 / 2 | 4 / 1 | 4 / 4 |
| 2 (logotipas + pilno ekrano) | 8 / 4 | 4 / 0 | 2 / 0 |
| 4 (logotipas + 3 × pilno ekrano) | 4 / 4 | 4 / 2 | 0 / 0 |

Lentelę skaityti atsargiai: Windows energijos profilis „Whisper“; G bėgimuose GPU dažnio vidurkis tik 210–690 MHz; kiekviena
pakopa matuota vieną kartą (be pakartojimų); G pakopos kartais nepraeina žemuose N, nors didesni N praeina, todėl N\* G
atveju pesimistiškas (didžiausias praėjęs N: 32, 32, 16 esant K = 1, 2, 4; K = 4, N = 32 pasiekė 8 GB VRAM ribą). Ar dažnis
paaiškina nesėkmes — netikrinta hipotezė. Pirmas G matavimas buvo negaliojantis (`gfxgen` klaida, ištaisyta) ir pakartotas.
Išsamiai: [`docs/rezultatai/2026-11-11-composite.md`](docs/rezultatai/2026-11-11-composite.md).

## Licencija ir autorystė

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Galima naudoti, keisti, platinti ir parduoti, taip pat
uždaruose produktuose. Platinant kodą ar jo išvestinį darbą, privaloma palikti NOTICE failą su autoriaus vardu.
Jei šis darbas panaudotas produkte, standarte ar publikacijoje, prašome nurodyti: **„VRAMFlow, autorius Deimantas
Kiršinas“** su nuoroda į šį repo.

Sukurta padedant DI programavimo asistentams (Claude). Autorius peržiūrėjo dizainą, visus matavimus atliko ant tikros
įrangos ir atsako už turinį.
