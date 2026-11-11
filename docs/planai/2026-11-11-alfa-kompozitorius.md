# Alfa grafika ir kompozitorius — įgyvendinimo planas

> **Vykdytojams:** privalomas sub-skill'as — superpowers:subagent-driven-development arba superpowers:executing-plans.
> Žingsniai pažymėti `- [ ]`. Įgyvendinimas — **naujoje sesijoje** (CLAUDE.md: specas ir įgyvendinimas atskirai).

**Tikslas:** biblioteka `libvramflow-gfx` (RGBA → standartinis `v210a` MXL flow, host ir `cuda-linear`, su
pasirinktiniu ritmo režimu) ir komanda `vramflow-func composite` (bazė + K sluoksnių → `v210` / `v210a`), C/H/G
variantai bitų tikslumu lygūs nepriklausomam etalonui, talpos matavimas N × K.

**Architektūra:** bendras rastro aprašas (`Raster`) apibendrina esamus `VideoIn`/`VideoOut` nuo „tik 1080i50
v210“ iki bet kokio v210 / v210a rastro. Matematika apibrėžta vieną kartą nepriklausomame etalone
(`src/gfx/gfx_ref.cpp`); CPU (`src/gfx/cpu/`) ir CUDA (`src/gfx/gfx_kernels.cu`) įgyvendinimai testuojami prieš jį.
Kompozitorius pernaudoja `logo` tick variklį (viena gija valdo MXL handle'us, vienas batch kernel'is laukui visiems N).

**Technologijos:** C++20, CUDA 13.2, MXL PR #653 (`6dddc91d92f5`) storage API, Catch2 3, CLI11, Python 3 (harness,
ctypes dūmų testas).

**Specas:** `docs/reikalavimai/2026-11-11-alfa-kompozitorius.md` (toliau **[S]**). Vykdytojas skaito abu.

## Bendri ribojimai

- `v210a` grain'as: fill (v210) iš karto, po jo key; key — 32 bitų LE blokai po 3 mėginius (bitai 0–9, 10–19,
  20–29), bitai 30–31 = 0, eilutė sulygiuota į 4 B, nenaudoti mėginiai = 0 [S §3; MXL `docs/Architecture.md:374`].
- Alfa flow'e **visada straight**, video diapazonas: key 64 = permatoma, 940 = nepermatoma [S §3, K1].
- Eilučių ilgiai: fill `ceil(w/48)·128` B; key `ceil(w/3)·4` B. 1080i50 laukas: fill 2 764 800 B, key
  1 382 400 B, grain 4 147 200 B.
- Interlaced: grain = laukas, lyginis indeksas = viršutinis laukas (paritetas 0 = kadro eilutės 0, 2, 4…).
- Laikas: indeksai iš TAI (`mxlGetTime`, `kFieldRate` 50/1); laukimas tik per esamą `wait_until`.
- Trūkstamas / `INVALID` / per vėlu įėjimas → to išėjimo grain'as tam indeksui **necommit'inamas**; skaitikliai
  `late_fields`, `too_late`, `in_invalid`, `read_errors` kaip `logo` [S §4.3].
- `x` lyginis ir ≥ 0, `y` ≥ 0 (kadro koordinatės); sluoksnio dalis už išėjimo ribų nukerpama.
- Sveikųjų skaičių formulės — §„Matematika“ žemiau; CPU, GPU ir etalonas: 0 skirtingų baitų.
- Kiekvienas naujas failas su SPDX antrašte (`2026 Deimantas Kiršinas`, `Apache-2.0`). Tik sintetinė medžiaga.
- Commit'ai lokaliai su `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; **jokio `git push`**.

## Matematika (pinned; etalonas įgyvendina tiksliai taip)

`A = 876`. Alfa skalė 0…A.

- RGBA → key: `key = 64 + (a8·876 + 127) / 255`.
- Premultiplied įėjimas → straight prieš konversiją: `c' = a8 == 0 ? 0 : min(255, (c·255 + a8/2) / a8)` kiekvienam R, G, B.
- RGB → Y/Cb/Cr kiekvienam pikseliui: esamas `ref::rgb_to_ycbcr10`.
- Chroma pora (pikseliai 2i, 2i+1): `cb = (a0·cb0 + a1·cb1 + (a0+a1)/2) / (a0+a1)`, analogiškai `cr`; jei
  `a0 + a1 == 0` → 512 (a — 8 bitų alfa). (Sveikaskaitė `ref::logo_from_rgba` taisyklės versija.)
- Key → alfa: `a = clamp(key − 64, 0, 876)`; su sluoksnio `opacity`: `o8 = lround(opacity·255)`,
  `ae = (a·o8 + 127) / 255`.
- Chroma alfa: `ac = (ae[2i] + ae[2i+1] + 1) >> 1` (sluoksnio pikseliai, kurie dengia išėjimo chroma mėginį).
- `v210` išėjimas (bazės alfa ignoruojama): `out = (b·(A − ae) + f·ae + A/2) / A`.
- `v210a` išėjimas (bazė `v210` = nepermatoma, `ab = A`; `--base none` = `ab = 0`, fill Y 64, C 512):
  `n = ae·A + ab·(A − ae)`; `ao = (n + A/2) / A`;
  `out = n == 0 ? (Y: 64, C: 512) : (f·ae·A + b·ab·(A − ae) + n/2) / n`; `key_out = ao + 64`.
- Visi fill rezultatai apkerpami į 4…1019. Tarpiniai — `uint64_t`.

## Peržiūros fokusas

1. Sluoksnis iš dalies už kadro (pvz., 384×216 ties x = 1800) → nukerpamas, jokio rašymo už buferio ribų.
2. Plotis, nesidalijantis iš 3 / 6 / 48 (1280, 382) → key padding = 0, fill grupės teisingos.
3. Grafikos writer'is perkraunamas veikiant (`FLOW_INVALID`) → tie laukai išmetami, kompozitorius atkuria reader'į
   ir tęsia, kai flow vėl atsiranda.
4. Ritmo režimas prieš pirmą `submit` → jokių grain'ų (įėjimas „trūksta“); po to kiekvienas laukas gauna paskutinį kadrą.
5. Laikrodžio šuolis atgal (WSL2) ritmo gijoje → tas pats indeksas necommit'inamas du kartus, jokio „runaway“.

Kiekvienas punktas turi testą jį valdančioje užduotyje (pažymėta **[PF-n]**).

---

### Užduotis 1: `v210a` ant `cuda-linear` — atitikties testas (vartai)

**Failai:**
- Sukurti: `tests/gpu/test_v210a_cuda_linear.cpp`
- Keisti: `CMakeLists.txt` (testo taikinys `vramflow-v210a-gpu-tests`, linkina `mxl::mxl`, `CUDA::cudart_static`)
- Sukurti: `upstream/U4-v210a-cuda-linear/` (README, patch prieš PR head, ctest.txt) — tik jei testas MXL medyje praeina arba randama klaida

**Sąsajos:** Produkuoja: patvirtinimą, kad `mxlFlowWriterGetStorageLayout` grąžina `planeCount == 2`,
`planes[0].size == fill`, `planes[1].size == key`, ir kad slice'ai bei `minValidSlices` veikia abiem plokštumoms.

- [ ] **1.1 Testas:** `TEST_CASE("v210a cuda-linear: dvi plokštumos, dydžiai")` — sukuria laikiną domeną
  `/dev/shm/vf-v210a-<pid>`, `v210a` 1080i50 flow su `{"payload":{"backend":"cuda-linear","deviceUuid":…}}`;
  `REQUIRE(layout.planeCount == 2)`, `planes[0].size == 2764800`, `planes[1].size == 1382400`.
- [ ] **1.2 Testas:** `"v210a cuda-linear: abi plokštumos perduodamos"` — writer'is įrašo pseudoatsitiktinį turinį į
  abi plokštumas (device), commit; reader'is (`mxlFlowReaderGetGrainSlot`) gauna slot'ą, `cudaMemcpy` į host,
  `CHECK` 0 skirtingų baitų abiem plokštumoms.
- [ ] **1.3 Testas:** `"v210a cuda-linear: slice'ai"` — commit'inama po 135 eilutes (`validSlices` 135, 270, 405,
  540); skaitymas su `minValidSlices = validSlices + 1` → `MXL_ERR_OUT_OF_RANGE_TOO_EARLY`; su `= validSlices` → OK.
- [ ] **1.4 Paleisti:** `ctest -R v210a` build kataloge. Laukta: 3/3 PASS. Jei FAIL — sistemingai ištirti
  (`superpowers:systematic-debugging`); maža klaida → pataisa `upstream/U4-…`; didelė → **STOP, klausti autoriaus** [S §4.5].
- [ ] **1.5** Tą patį testą (be mūsų priklausomybių) paruošti MXL Catch2 stiliumi `upstream/U4-…/` ir paleisti PR
  #653 medyje kaip U2/U3 (`scripts/wsl/20-upstream-tests.sh` stilius, ne root). Įrašyti `ctest.txt`.
- [ ] **1.6 Commit:** `test: v210a ant cuda-linear (dvi plokštumos, slice'ai) + U4 juodraštis`

### Užduotis 2: `Raster` ir `v210a` pakavimas

**Failai:**
- Sukurti: `src/gfx/raster.hpp` (header-only)
- Keisti: `src/func/flows.hpp`, `src/func/flows.cpp:90-165` (geometrijos patikros)
- Testas: `tests/unit/test_gfx_raster.cpp` (taikinys `mxlv-gfx-unit`, `cmake/gfx.cmake`)

**Sąsajos (produkuoja):**
```cpp
namespace vramflow::gfx {
struct Raster { int w, h; bool interlaced; bool alpha; };      // h — kadro aukštis
constexpr Raster kHd1080i{1920, 1080, true, false};
std::size_t fill_line_bytes(int w);    // ceil(w/48)*128
std::size_t key_line_bytes(int w);     // ceil(w/3)*4
int grain_lines(const Raster&);        // interlaced ? h/2 : h
std::size_t fill_bytes(const Raster&), key_bytes(const Raster&), grain_bytes(const Raster&);
std::uint32_t key_get(const std::uint32_t* key_line, int x);            // 10 bitų mėginys
void key_set(std::uint32_t* key_line, int x, std::uint32_t v);
bool raster_from_flow_def(const std::string& json, Raster& out, std::string& err);
// Bendri tipai visiems įgyvendinimams (ref, cpu, cuda):
enum class AlphaIn { Straight, Premultiplied };
struct Layer { const std::uint8_t* grain; Raster r; int x, y; float opacity; };   // grain — host arba device
struct CompositeJob { const std::uint8_t* base; Raster base_r; const Layer* layers; int k; Raster out_r; std::uint8_t* out; };
}
```
`VideoOut::create(..., const gfx::Raster& r = gfx::kHd1080i)` ir `VideoIn::open(..., const gfx::Raster* expect = nullptr)`
(nullptr → rastras perskaitomas iš flow ir saugomas `VideoIn::raster`); abu tikrina `planeCount == (alpha ? 2 : 1)`
ir plokštumų dydžius. Numatytosios reikšmės nekeičia esamų funkcijų elgsenos.

- [ ] **2.1 Testai:** `fill_line_bytes(1920) == 5120`, `(1280) == 3456`, `(384) == 1024`; `key_line_bytes(1920) == 2560`,
  `(1280) == 1708`, `(382) == 512`; `grain_bytes(kHd1080i with alpha) == 4147200`; `key_set/key_get` kiekvienam x
  0…1919 sutampa, bitai 30–31 lieka 0, paskutinio bloko padding (`w = 1280`: bitai 20–31) = 0 **[PF-2]**;
  `raster_from_flow_def` ant MXL `lib/tests/data/v210a_flow.json` → `{1920,1080,false,true}`.
- [ ] **2.2** Paleisti → FAIL (nėra simbolių). **2.3** Įgyvendinti. **2.4** `ctest -R gfx` PASS ir visas esamas
  `ctest` (42 testai) PASS. **2.5 Commit:** `gfx: Raster, v210a key pakavimas, flows bet kokiam rastrui`.

### Užduotis 3: nepriklausomas etalonas

**Failai:** sukurti `src/gfx/gfx_ref.{hpp,cpp}`; testas `tests/unit/test_gfx_ref.cpp`

**Sąsajos (produkuoja):**
```cpp
namespace vramflow::gfx::ref {
// Vienas laukas (paritetas p) iš progresyvinio RGBA kadro (w×h, stride B) → v210a grain'as (fill + key).
void rgba_to_v210a(const std::uint8_t* rgba, int w, int h, std::size_t stride, AlphaIn mode,
                   const Raster& r, int parity, std::uint8_t* grain);
// base: v210 arba v210a grain'as arba nullptr (--base none); out — v210 arba v210a pagal out_r.alpha.
void composite(const std::uint8_t* base, const Raster& base_r, const Layer* layers, int k,
               const Raster& out_r, int parity, std::uint8_t* out);
}
```
Rašoma tiesiai iš §„Matematika“ formulių, mėginių lygmeniu (išpakuoti → skaičiuoti → supakuoti), be optimizacijų.
**Negalima** kopijuoti kodo iš 4–5 užduočių (ir atvirkščiai).

- [ ] **3.1 Testai (ranka suskaičiuotos reikšmės):** key iš a8 0 → 64, 255 → 940, 128 → 504; premultiplied
  (200,100,50,128) → straight (398→255 apkirpta, 199, 100); `opacity 0` → išėjimas = bazė; sluoksnis a = A → išėjimas
  = sluoksnio fill; `v210a` + `--base none` su vienu sluoksniu → key ir fill = sluoksnio; grandinė: sluoksnis A su
  dvejetaine alfa (a8 ∈ {0, 255}) → composite(none, [A]) į v210a, tada composite(bazė, [tas v210a]) ==
  composite(bazė, [A]) bitais (bendram alfa atvejui apvalinimo skirtumas tik išmatuojamas ir užrašomas, ne
  reikalaujamas); nukirpimas ties x = 1800 nerašo už ribų
  (buferis su apsauginėmis zonomis, patikra) **[PF-1]**; paritetas 1 ima kadro eilutes 1, 3, 5….
- [ ] **3.2–3.4** FAIL → įgyvendinti → PASS. **3.5 Commit:** `gfx: nepriklausomas rgba→v210a ir composite etalonas`.

### Užduotis 4: CPU variantas (C)

**Failai:** sukurti `src/gfx/cpu/gfx_cpu.{hpp,cpp}` (`-O3 -march=x86-64-v3`, kaip `mxlv_media_cpu`);
testas `tests/unit/test_gfx_cpu.cpp`

**Sąsajos:** tos pačios signatūros kaip 3 užduoties, vardų erdvė `vramflow::gfx::cpu`, plius
`void composite_batch(const gfx::CompositeJob* jobs, int n, int parity, func::ThreadPool* pool)` (esamas
`ThreadPool` iš `func_common.hpp`; `nullptr` — viena gija).

- [ ] **4.1 Testai:** atsitiktiniai RGBA (visos a8 reikšmės 0…255), abu `AlphaIn`, pločiai 1920/1280/382, abu
  paritetai: `cpu::rgba_to_v210a` == `ref::` (0 skirtingų baitų); `composite`: K = 1, 2, 4, opacity {0, 0.5, 1},
  pozicijos {(0,0), (1488,64), (1800,900), (100,33)}, išėjimas v210 ir v210a, bazė v210 / v210a / none.
- [ ] **4.2–4.4** FAIL → įgyvendinti → PASS. **4.5 Commit:** `gfx: CPU rgba→v210a ir composite (C variantas)`.

### Užduotis 5: CUDA kernel'iai (H ir G)

**Failai:** sukurti `src/gfx/gfx_kernels.{hpp,cu}`; testas `tests/gpu/test_gfx_kernels.cpp`

**Sąsajos:**
```cpp
namespace vramflow::gfx::cuda {
cudaError_t rgba_to_v210a(const std::uint8_t* rgba_dev, int w, int h, std::size_t stride, AlphaIn mode,
                          const Raster& r, int parity, std::uint8_t* grain_dev, cudaStream_t s);
// Vienas kvietimas laukui visiems n išėjimams (batch, kaip logo_fields); jobs — host masyvas su device rodyklėmis,
// n ≤ kMaxBatch (64).
cudaError_t composite_batch(const CompositeJob* jobs, int n, int parity, cudaStream_t s);
}
```
`Layer` aprašai perduodami per `__grid_constant__` arba vieną device masyvą (kaip `media_kernels.cu`).

- [ ] **5.1 Testai:** tie patys atvejai kaip 4.1, GPU rezultatas (`cudaMemcpy` į host) == `ref::` 0 skirtingų baitų;
  `composite_batch` su n = 64 išėjimų, skirtingomis pozicijomis kiekvienam.
- [ ] **5.2–5.4** FAIL → įgyvendinti → PASS. **5.5** `compute-sanitizer --tool memcheck` ant `vramflow-gfx-gpu-tests`
  — 0 klaidų **[PF-1]**. **5.6 Commit:** `gfx: CUDA rgba→v210a ir composite batch`.

### Užduotis 6: biblioteka `libvramflow-gfx` (C ABI, ritmo režimas)

**Failai:** sukurti `include/vramflow/gfx.h`, `src/gfx/gfx_writer.cpp`; `cmake/gfx.cmake` (shared biblioteka
`vramflow-gfx`, `install(TARGETS …)` ir antraštė); testai `tests/gpu/test_gfx_writer.cpp`, `tests/gfx_ctypes_smoke.py`

**Sąsajos (produkuoja, tiksliai):**
```c
typedef struct vfgfx_writer_s* vfgfx_writer;
typedef enum { VFGFX_ALPHA_STRAIGHT = 0, VFGFX_ALPHA_PREMULTIPLIED = 1 } vfgfx_alpha;
typedef enum { VFGFX_MEM_HOST = 0, VFGFX_MEM_CUDA = 1 } vfgfx_mem;
typedef struct { const void* rgba; uint32_t width, height, stride; vfgfx_alpha alpha; vfgfx_mem mem; } vfgfx_frame;
/* 0 = OK; 1..1023 = mxlStatus; >= 2000 = VFGFX_ERR_* (INVALID_ARG 2001, GEOMETRY 2002, CUDA 2003, STATE 2004) */
int vfgfx_writer_create(mxlInstance inst, const char* flow_def_json, const char* writer_options_json, vfgfx_writer* out);
int vfgfx_write(vfgfx_writer w, uint64_t index, const vfgfx_frame* f);   /* rankinis: grain index (laukas index%2) */
int vfgfx_start_pacing(vfgfx_writer w, int64_t offset_ns);                 /* ritmo gija, kas laukas */
int vfgfx_submit(vfgfx_writer w, const vfgfx_frame* f);                    /* ritmo režimas: naujas kadras */
int vfgfx_stop_pacing(vfgfx_writer w);
uint64_t vfgfx_committed(vfgfx_writer w);                                  /* commit'intų grain'ų skaičius */
const char* vfgfx_last_error(vfgfx_writer w);
void vfgfx_writer_destroy(vfgfx_writer w, mxlInstance inst);
```
Saugykla — iš `writer_options_json` (host arba `cuda-linear`). `submit` konvertuoja kadrą iškart į du paruoštus
laukus (paritetai 0 ir 1, GPU arba CPU pagal saugyklą) ir po mutex'u pakeičia „paskutinį“; ritmo gija kas indeksą
(`wait_until`) kopijuoja atitinkamo pariteto paruoštą lauką į slot'ą ir commit'ina. Rankinis ir ritmo režimai
vienu metu negalimi (`VFGFX_ERR_STATE`). Ritmo gija vienintelė liečia MXL writer'į, kol veikia.

- [ ] **6.1 Testai:** rankinis `vfgfx_write` 4 laukams → reader'is gauna grain'us == `ref::rgba_to_v210a`;
  ritmo režimas: 0,5 s be `submit` → `vfgfx_committed == 0` **[PF-4]**; po vieno `submit` 1 s → ≥ 48 grain'ai, visi
  indeksai iš eilės, kiekvienas == paskutinio kadro atitinkamo pariteto laukas **[PF-4]**; dirbtinis laikrodžio
  šuolis atgal (testinis `wait_until` kabliukas / `ClockWatch`) → nė vienas indeksas necommit'inamas du kartus **[PF-5]**;
  `vfgfx_write` ritmo metu → 2004; host ir `cuda-linear` abu.
- [ ] **6.2** Python ctypes dūmų testas: įkelia `libvramflow-gfx.so`, sukuria host writer'į, įrašo vieną kadrą,
  `vfgfx_committed == 1`, `destroy`. Paleisti: `python3 tests/gfx_ctypes_smoke.py <build>/libvramflow-gfx.so` → `OK`.
- [ ] **6.3–6.5** FAIL → įgyvendinti → PASS (`ctest -R gfx_writer`, dūmų testas). **6.6 Commit:**
  `gfx: libvramflow-gfx C ABI, rankinis ir ritmo režimai`.

### Užduotis 7: `gfxgen` — sintetiniai grafikos šaltiniai

**Failai:** sukurti `src/func/gfxgen.cpp`, `src/gfx/gfx_content.hpp` (deterministinis RGBA turinys); keisti
`src/func/flows.{hpp,cpp}` (rolė `kRoleGfx = 5`, `gfx_def(n, k, Raster)`), `src/func/main_func.cpp`, `cmake/func.cmake`

**Sąsajos:**
- Flow ID: `flow_id(kRoleGfx, n * 8 + k)`, K ≤ 8.
- `void gfx_content(std::uint64_t seed, std::uint32_t n, std::uint32_t k, std::uint64_t frame, int w, int h, std::uint8_t* rgba)`
  — visos a8 reikšmės 0…255 pasikartoja kiekvienoje eilutėje; k lyginis — straight, nelyginis — premultiplied (RGB ≤ a).
- CLI: `vramflow-func gfxgen --domain D --flows N --layers K --layer-geom logo,full,full,full --mode pace|manual
  [--impl/--storage/--device-uuid/--duration/--out]`; `logo` = 384×216, `full` = 1920×1080, abu interlaced v210a.
  `pace` — statinis kadras (frame 0) ritmo režimu; `manual` — kadras = index/2 kiekvienam indeksui.
- Naudoja tik `libvramflow-gfx` (įrodo, kad biblioteka pakankama). `summary-gfxgen.json` su skaitikliais kaip `gen`.

- [ ] **7.1 Testas:** `tests/functional/functional.py` naujas `fgfx1`: `gfxgen --flows 2 --layers 2 --mode pace
  --duration 5` + `mxl-info` mato 4 `v210a` flow'us; `summary` — 0 klaidų, commit'ų ≥ 240 kiekvienam.
- [ ] **7.2–7.4** FAIL → įgyvendinti → PASS. **7.5 Commit:** `func: gfxgen sintetiniai v210a šaltiniai`.

### Užduotis 8: `composite` funkcija

**Failai:** sukurti `src/func/composite.cpp`; keisti `src/func/flows.hpp` (rolė `kRoleComp = 6`),
`main_func.cpp`, `func_common.{hpp,cpp}` (Opts: `layers`, `layer_xy`, `opacity`, `base`, `out_alpha`), `cmake/func.cmake`

**Sąsajos:**
- CLI: `vramflow-func composite --domain D --flows N --layers K --layer-xy X,Y (K kartų) [--opacity O (K kartų)]
  --base gen|none --out-format v210|v210a [--layer-flow K:UUID] [--deadline-ms 15 --read-offset-us …]
  [bendri argumentai]`.
  Įėjimai: bazė `flow_id(kRoleGen, n)` (arba none), sluoksniai `flow_id(kRoleGfx, n*8+k)`; išėjimas
  `flow_id(kRoleComp, n)`. Sluoksnių rastrai skaitomi iš flow'ų (`VideoIn::raster`).
- Tick: kaip `logo.cpp:132-180` (neblokuojantys get, kartojama kas 0,5 ms iki termino); išėjimas n commit'inamas tik
  jei bazė ir visi K sluoksniai `state == 1`; vienas `cuda::composite_batch` / `cpu::composite_batch` laukui.
- Įėjimo `read_errors` su `MXL_ERR_FLOW_INVALID` → reader'is uždaromas ir atidaromas iš naujo neblokuojančiai
  sekančiuose tick'uose (skaitiklis `reopened`) **[PF-3]**.

- [ ] **8.1 Testai (functional.py):** `fcomp1` — `gen` + `gfxgen --mode pace` + `composite --flows 2 --layers 2`
  10 s → išėjimo flow'ai egzistuoja, `late_fields == 0`; `fcomp2` — `gfxgen` užmušamas 2 s ir paleidžiamas iš naujo
  → `composite` neišeina, `reopened ≥ 1`, po paleidimo išėjimai vėl commit'inami **[PF-3]**; `fcomp3` — `--base none
  --out-format v210a` veikia be `gen`; `fcomp5` — grandinė: `composite --base none --out-format v210a` išėjimas
  paduodamas kaip sluoksnis antram `composite` su `gen` baze (`--layer-flow` perrašo k-tojo sluoksnio flow ID) →
  abu commit'ina, `late_fields == 0`.
- [ ] **8.2–8.4** FAIL → įgyvendinti → PASS. **8.5 Commit:** `func: composite (bazė + K sluoksnių, C/H/G)`.

### Užduotis 9: `verify --mode composite`

**Failai:** keisti `src/func/verify.cpp`

**Sąsajos:** `verify --mode composite --layers K --layer-xy … --opacity … --base gen|none --out-format …
--gfx-mode pace|manual --every M` — kiekvienam M-tajam išėjimo laukui atkuria laukiamą rezultatą CPU etalonu
(`ref::gen_field` / `gfx_content` → `ref::rgba_to_v210a` → `ref::composite`) ir lygina CRC32C; neatitikimai
skaičiuojami `verification_errors`, iškeliami `--dump` kaip esamuose režimuose.

- [ ] **9.1 Testas:** `fcomp4` — pilna grandinė su `verify --mode composite --every 5`, C, H ir G variantais po 10 s,
  `--gfx-mode manual` ir `pace` → `verification_errors == 0` visiems; tyčinis neatitikimas (`--opacity` skirtinga
  verify ir composite) → `verification_errors > 0` (įrodo, kad patikra veikia).
- [ ] **9.2–9.4** FAIL → įgyvendinti → PASS. **9.5 Commit:** `verify: composite režimas su CPU etalonu`.

### Užduotis 10: talpa N × K ir rezultatai

**Failai:** keisti `tests/perf/capacity.py` (`--func composite`, `--layers K`), `scripts/wsl/71-compare.sh`
(composite: variantai C, H, G); sukurti `docs/rezultatai/2026-11-1x-composite.md`

**Sąsajos:** pakopa paleidžia `gen` (bazė; C variante `--source fair` logika kaip `logo`), `gfxgen --mode pace`,
`composite`, `verify --mode composite`; funkcijos CPU = `composite` procesas; GPU zondas kaip dabar.
Geometrija [S §4.7]: K = 1 → `logo`; K = 2 → `logo,full`; K = 4 → `logo,full,full,full`.

- [ ] **10.1** Dūmų pakopa: `capacity.py … --func composite --n 1 --layers 2` C ir G → abu PASS, `verify` 0 klaidų.
- [ ] **10.2** Kampanija: K ∈ {1, 2, 4} × N ∈ {1, 2, 4, 8, 16, 32} × {C, H, G} (`71-compare.sh`, fone);
  VRAM biudžetas patikrinamas prieš didžiausias pakopas (`v210a` grain 4 147 200 B × ring'o grain'ai × flow'ai ≤ 7 GB).
- [ ] **10.3** `compare_report.py` → `docs/rezultatai/2026-11-1x-composite.md` + pastabos (energijos profilis, GPU
  dažniai, kaip `2026-11-11-gpu-vs-cpu.md`). **10.4 Commit:** `perf: composite N×K talpa C/H/G ir rezultatai`.

### Užduotis 11: OGraf pavyzdys ir dokumentacija

**Failai:** sukurti `examples/ograf/` (README, paleidimo skriptas, mažas C++ arba Python adapteris → `libvramflow-gfx`);
keisti `README.md`, `README.lt.md`, `upstream/README.md` (U4)

- [ ] **11.1 Tyrimas (prieš kodą):** patikrinti, kuris renderis šiandien be GUI duoda RGBA su alfa (pvz., GStreamer
  `wpesrc` su permatomu fonu, headless Chromium/CEF) ir kokia OGraf pavyzdinė grafika tinka; rezultatą (versijos,
  komanda, ar alfa tikrai permatoma) įrašyti `examples/ograf/README.md`.
- [ ] **11.2** Adapteris: renderio RGBA kadrai → `vfgfx_submit` ritmo režimu → `v210a` flow; demonstracija:
  `composite` su `gen` baze; `verify` netaikomas (turinys nedeterministinis) — patikra: išėjimas commit'inamas, key
  ne vien 64 / 940 (alfa tikrai perduodama), vienas laukas išsaugomas PNG peržiūrai.
- [ ] **11.3** README: bibliotekos API pavyzdys (10–15 eilučių C), `composite` komandos pavyzdys, rezultatų lentelė.
- [ ] **11.4 Commit:** `examples: OGraf → v210a; docs: gfx biblioteka ir composite`.

---

## Vykdymo pastabos

- Užduotis 1 — vartai: be jos G variantas `v210a` neegzistuoja. 2 → 3 → (4, 5 lygiagrečiai galimos) → 6 → 7 → 8 → 9 → 10; 11 po 6.
- Po 5, 8 ir 10 — `/code-review` (rizikinga: bitų tikslumas, tick laikas, talpos teiginiai).
- Kiekviena „PASS“ ataskaita — su ką tik paleistos komandos išvestimi (CLAUDE.md).
