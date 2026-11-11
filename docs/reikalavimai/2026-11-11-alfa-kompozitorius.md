# Alfa grafika ir kompozitorius MXL flow'ams (VRAMFlow)

- **Data:** 2026-11-11
- **Būsena:** dizainas (peržiūrėtas statytojo; K1, K2 atsakyti), laukia autoriaus patvirtinimo
- **Repo:** `vramflow` (šis repo). Žymėjimas: **[autorius]** — Deimanto sprendimas; **[mano]** — dizainerio
  sprendimas, kurį autorius gali atmesti; **[faktas]** — patikrinta šaltinyje (nuoroda).

## 1. Kam ir kodėl

Šiandien VRAMFlow `logo` funkcija logotipą „įkepa“: grafika yra funkcijos viduje (įtaisytas paveikslėlis), o ne
atskiras signalas. Tikroje gamyboje grafika (logotipai, titrai, OGraf / CasparCG grafika ir t. t.) ateina iš kitų
programų ir turi būti **atskiras signalas su alfa kanalu**, kurį galima sujungti su video taip, kaip reikia.

MXL tam jau turi formatą: `video/v210a` — viename grain'e fill (v210) ir key (alfa) plokštumos
[faktas: `mxl docs/Architecture.md:326,372-412`, `lib/internal/src/FlowParser.cpp:351-366,444-447`]. Trūksta:

1. paprasto būdo bet kokiai programai įrašyti savo grafiką į `v210a` flow (host arba GPU atmintyje);
2. funkcijos, kuri sujungia kelis sluoksnius į vieną rezultatą — kompozitoriaus.

**Esmė [autorius]:** tai žemo lygio sprendimas, kaip pats MXL — biblioteka ir komandos, jokių web ar kitų aplikacijų;
developeriai naudoja kaip nori. Grafikos flow yra **toks pat flow kaip visi kiti**, tik kitos paskirties: jokių
specialių taisyklių grafikai.

**Sėkmė:** developeris su keliomis eilutėmis kodo įrašo savo RGBA kadrus į `v210a` flow; viena komanda sujungia video
ir K grafikos sluoksnių į naują flow; rezultatas ant GPU ir ant CPU bitų tikslumu sutampa su nepriklausomu etalonu;
išmatuota, kiek išėjimų × sluoksnių telpa ant GPU palyginus su CPU (tas pats metodas kaip `docs/rezultatai/`).

## 2. Tikslai, ne-tikslai, anti-tikslai

**Tikslai**

- G1. C biblioteka (`libvramflow-gfx`): RGBA kadras → `v210a` grain'as MXL flow'e; host ir `cuda-linear` saugykla;
  įėjimo alfa — premultiplied arba straight (pasirenkama kiekvienam writer'iui).
- G2. Komanda `vramflow-composite`: bazinis video flow + K grafikos sluoksnių → išėjimo flow; parametrai nurodomi
  komandos argumentais paleidžiant, kaip kitų VRAMFlow funkcijų [autorius].
- G3. Išėjimas — `v210` (programa be alfos) **arba** `v210a` (sujungti sluoksniai su bendra alfa), pagal išėjimo
  flow aprašą; `v210a` išėjimą galima vėl paduoti kaip sluoksnį kitam kompozitoriui (grandinė) [mano, autorius pritarė
  principui „vėliau sujungti ir turėti rezultatą“].
- G4. Trys variantai kaip visur VRAMFlow: C (CPU + host), H (GPU + host), G (GPU + `cuda-linear`); bitų tikslumo
  etalonas, nepriklausomas nuo abiejų įgyvendinimų.
- G5. Pavyzdinis šaltinis: OGraf grafika → `v210a` flow, naudojant biblioteką [autorius: „pradžioje galima panaudoti
  kokį OGraf“].
- G6. Talpos matavimas: N išėjimų × K sluoksnių, C prieš G, tas pats `capacity.py` / `compare_report.py` metodas.

**Ne-tikslai (dabar)**

- Valdymas eteryje (fade, įjungimas/išjungimas, pozicijos keitimas veikiant). Parametrai fiksuojami paleidžiant.
  Ateities kryptis [mano]: valdymas per MXL duomenų / event flow, kad pakeitimai būtų susieti su grain indeksu.
- Garsas (kompozitorius garso neliečia; garso flow'ai praeina atskirai, kaip `chain` grandinėje).
- Savo HTML renderio kūrimas. OGraf prijungiamas kaip pavyzdys per esamą renderį.
- Mastelio keitimas (scaling), sukimas, spalvų erdvių konversijos tarp skirtingų flow'ų. Visi sluoksniai — to paties
  rastro ir spalvų erdvės kaip išėjimas (žr. §4.4), išskyrus poziciją (x, y).
- Kitos rezoliucijos nei bandymų stendo 1080i50 — kodas neturi jų uždrausti, bet testuojama tik 1080i50 (ir
  1080p25/50 bibliotekos vienetų testuose, jei pigu).

**Anti-tikslai (nesėkmė, net jei „veikia“)**

- Nestandartiniai flow'ai: `v210a` grain'as, kurio negalėtų teisingai perskaityti kitas MXL suderinamas įrankis.
- Grafikos flow'ai su kitokia laiko / vėlavimo semantika nei video flow'ai.
- Priklausomybė nuo naršyklės, web serverio ar GUI bibliotekos ar kompozitoriuje (OGraf pavyzdys — atskiras, neprivalomas).
- GPU ir CPU rezultatai, kurie skiriasi bent vienu bitu.

## 3. Ribojimai

- **Standartas [faktas: Architecture.md:374-376]:** key plokštuma eina iškart po fill; 32 bitų little-endian
  blokai, kiekviename 3 alfa mėginiai (bitai 0–9, 10–19, 20–29), bitai 30–31 nenaudojami; kiekviena eilutė
  sulygiuota į 4 baitus, nenaudoti mėginiai užpildomi 0. **Alfa yra „straight“ (ne premultiplied)** ir tame pačiame
  diapazone kaip fill: video diapazonas → key 64 (visiškai permatoma) … 940 (visiškai nepermatoma). Key eilutės ilgis
  = `ceil(w/3)·4` B (1920 → 2560 B).
- **Todėl [autorius, K1]:** pasirinkimas „premultiplied ar straight“ taikomas
  **bibliotekos įėjimui** (iš kur ateina RGBA). Į flow visada rašoma standartinė straight alfa. Kompozitorius skaito
  tik standartinius flow'us ir jokio papildomo flow žymėjimo nereikia.
- Interlaced: `v210a` leidžiamas interlaced, jei aukštis lyginis [faktas: FlowParser.cpp:370]; grain = laukas,
  kaip `v210` 1080i50 VRAMFlow'e (lyginis indeksas = viršutinis laukas, `docs/architektura/2026-11-10-gpu-media-funkcijos.md` §2).
- Laikas ir sinchronizacija — kaip MXL ant CPU: indeksai iš TAI (`mxlGetTime`), visi įėjimai sujungiami pagal **tą
  patį grain indeksą** (`docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md`).
- GPU 8 GB VRAM bandymų stende; `v210a` 1080i50 laukas = 2 764 800 B fill + 1 382 400 B key = 4 147 200 B
  (≈ 1,5× `v210` lauko) — ring'ų VRAM biudžetas skaičiuojamas plane.
- Licencija Apache-2.0, SPDX antraštės, kaip visas repo. Jokio tikro transliacijų turinio testuose — tik sintetinis.
- Rašymas ir skaitymas tik per MXL viešą API (ir PR #653 storage API `cuda-linear` atveju), kaip esamos funkcijos.

## 4. Kaip tai veikia (sutarta)

### 4.1 Biblioteka `libvramflow-gfx` (G1)

C API (C++ viduje, kaip MXL), kad būtų naudojama iš C, C++, Rust, Python (ctypes). Pristatoma tik C ABI ir
antraštė; Rust / Python bindings'ai nėra šio darbo dalis — tik Python ctypes dūmų testas, įrodantis, kad ABI
naudojamas [mano]. Minimali apimtis:

- sukurti grafikos writer'į: MXL instancija + `v210a` flow aprašas + saugykla (`host` | `cuda-linear` + GPU UUID);
- įrašyti kadrą: grain indeksas, RGBA 8 bitų buferis (host arba CUDA device rodyklė), eilutės žingsnis (stride),
  įėjimo alfa tipas (premultiplied | straight), spalvų diapazonas (video range, BT.709 — kaip esamas `rgb_to_ycbcr10`);
- interlaced flow'ams: vienas progresyvinis RGBA kadras → du laukai (du grain'ai), arba vienas laukas, jei šaltinis
  pateikia laukus (žr. §6);
- **ritmo režimas (pasirinktinis) [autorius]:** biblioteka pati, pagal TAI laiką, kiekvienam laukui commit'ina
  paskutinį gautą kadrą; programa tik pateikia naujus kadrus, kai jie atsiranda (statinis logotipas, 25 fps ar
  nepastovaus ritmo renderis, strigimas). Taip grafikos flow elgiasi kaip video šaltinis, kuris rašo visada.
  Paskutinis kadras commit'inamas be perskaičiavimo (paruoštas grain'as kopijuojamas ar perrašomas tame pačiame
  saugyklos tipe). Be šio režimo biblioteka tik konvertuoja ir įrašo nurodytą indeksą — ritmą valdo programa, kaip
  bet kurio MXL writer'io. Ritmo režimui reikia atskiros gijos su savo MXL writer'iu (writer'iai nėra thread-safe):
  programos kadrų pateikimas ir commit'ai sinchronizuojami bibliotekos viduje.

RGBA → v210a konversija yra GPU kernel'is ir CPU funkcija, abu bitų tikslumu lygūs etalonui.

### 4.2 Komanda `vramflow-composite` (G2, G3)

Nauja `vramflow-func` subkomanda arba atskiras binaras (statytojo pasirinkimas), su tais pačiais bendrais
argumentais kaip kitos funkcijos (`--domain`, `--impl`, `--storage`, `--device-uuid`, `--duration`, `--out` …).
Kiekvienam išėjimui:

- bazinis flow (`v210` arba `v210a`);
- K sluoksnių eilės tvarka iš apačios į viršų; kiekvienam: flow ID, pozicija `x,y` (sluoksnio viršutinis kairys
  kampas išėjimo kadre), neprivalomas bendras permatomumas `opacity` (0–1, pastovus);
- išėjimo flow aprašas (`v210` arba `v210a`).

Kad būtų galima matuoti talpą, vienas procesas turi galėti aptarnauti N nepriklausomų išėjimų (kiekvienas su savo
baze ir sluoksniais), naudodamas tą patį „tick“ variklį kaip `logo` (viena gija valdo MXL handle'us, vienas batch
kernel'is laukui visiems N; `docs/architektura/2026-11-10-gpu-media-funkcijos.md` §3).

### 4.3 Įėjimų vėlavimas ir dingimas [autorius]

Visi įėjimai (bazė ir grafika) traktuojami vienodai, lygiai kaip `logo` funkcija traktuoja savo video įėjimą: tick'as
laukia iki termino (`--deadline-ms`); jei bent vieno įėjimo grain'o tam indeksui iki termino nėra (arba jis
`INVALID`, arba per vėlu), **to išėjimo grain'as tam indeksui necommit'inamas** ir užskaitomas skaitikliuose
(`late_fields`, `too_late`, `in_invalid` — kaip dabar). Jokio „palaikyti paskutinį kadrą“ ar „praleisti sluoksnį“.

### 4.4 Kompozicijos matematika

- Operatorius „over“ ant straight alfos, iš apačios į viršų, sveikųjų skaičių aritmetika su tiksliai apibrėžtu
  apvalinimu, kad CPU, GPU ir etalonas sutaptų bitais. Formulę ir apvalinimą parenka statytojas ir užrašo plane;
  etalonas (`media_ref` stiliaus) įgyvendinamas atskirai nuo abiejų variantų.
- Alfa iš video diapazono: `a = clamp((key − 64) / 876, 0, 1)`, padauginta iš sluoksnio `opacity`.
- Chroma 4:2:2: vienam Cb/Cr mėginiui naudojama porinė alfa `(a0 + a1 + 1) >> 1` — ta pati taisyklė kaip esamame
  `logo` (`src/media/media_ref.cpp:101`) [mano].
- `x` turi būti lyginis (chroma poros); `y` interlaced flow'e — kadro eilutė, laukui perskaičiuojama pagal paritetą
  (kaip `logo_field`).
- `v210a` išėjimas: fill = sujungtas vaizdas, key = bendra „over“ alfa, užrašyta atgal straight, video diapazone.
  Išėjimas be bazės (tik grafikos submix) [mano]: `--base none` — kompozitorius viduje naudoja visiškai permatomą
  foną (key = 64) ir nelaukia papildomo įėjimo; galimas tik su `v210a` išėjimu.
- Sluoksnis gali būti mažesnis už išėjimą (pvz., 384×216 logotipas) — tada jo flow turi mažesnį rastrą; dalys už
  išėjimo ribų nukerpamos.

### 4.5 Variantai ir patikra (G4)

- C, H, G kaip `logo`: tie patys `--impl` / `--storage` argumentai; C nekuria CUDA konteksto.
- Patikra: `verify` režimas kompozitoriui — žinomas sintetinis turinys (bazės generatorius + sintetiniai grafikos
  šaltiniai su žinoma alfa, įskaitant visas 0…1023 key reikšmes ir kraštus), CRC etalonas kaip dabar.
- Vienetų testai: RGBA → v210a (abu alfa tipai), v210a pakavimas (1920 ir plotis, nesidalijantis iš 3, pvz., 1280),
  „over“ kraštutiniai atvejai (alfa 0, 1, opacity 0), grandinė (v210a išėjimas kaip sluoksnis).
- `v210a` ant `cuda-linear` (2 plokštumos) — iki šiol niekieno neištestuota; testas turi patikrinti abi plokštumas,
  slice'us ir `minValidSlices` ir būti tinkamas pasiūlyti upstream (kaip U2/U3). Mūsų kodas šiandien tikisi vienos
  plokštumos (`planeCount == 1`, `vramflow-verify` W4) — tai praplėsti.
- **Eilės tvarka ir rizika [mano]:** šis testas — pirma plano užduotis, prieš bibliotekos ir kompozitoriaus kodą.
  Jei `v210a` ant `cuda-linear` neveikia: maža klaida → taisoma kaip upstream pataisa (`upstream/U4-…`) su įrodymais;
  didelė (keičia PR #653 dizainą) → darbas stabdomas ir klausiama autoriaus (G variantas `v210a` atidedamas,
  matuojami C ir H).

### 4.6 OGraf pavyzdys (G5)

Atskiras pavyzdys `examples/ograf/` (ne bibliotekos ir ne kompozitoriaus dalis): esamas OGraf suderinamas renderis
su permatomu fonu → RGBA kadrai → `libvramflow-gfx` → `v210a` flow. Renderio pasirinkimas — statytojo (§6).

### 4.7 Talpa (G6)

`capacity.py` nauja funkcija `composite`: N išėjimų × K sluoksnių, C prieš G (ir H, jei pigu), GPU zondas, tas pats
PASS/PASS_tol apibrėžimas; rezultatai į `docs/rezultatai/`. Sluoksnių geometrija [mano] — tipinė eterio grafika:
K = 1 — logotipas 384×216; K = 2 — logotipas + pilno kadro grafika 1920×1080 (titrų / lower-third sluoksnis);
K = 4 — logotipas + 3 pilno kadro sluoksniai. Grafikos šaltiniai talpos bandyme — ritmo režimu.

## 5. Darbo eiga ir dokumentai

Pagal repo tvarką: šis dokumentas → planas `docs/planai/` (užduotys su priėmimo patikromis, test-first) →
įgyvendinimas → peržiūra. Commit'ai lokaliai; `git push` daro tik autorius. Upstream kandidatai (pvz., `v210a`
ant `cuda-linear` testas) ruošiami kaip `upstream/U*` aplankai su įrodymais.

## 6. Palikta statytojui

- Bibliotekos API vardai, ABI versijavimas, ar biblioteka statinė ar dinaminė.
- Ar `composite` — `vramflow-func` subkomanda ar atskiras binaras; argumentų sintaksė sluoksniams.
- Tikslios „over“ ir RGBA→YCbCr sveikųjų skaičių formulės ir apvalinimas (su sąlyga: bitų tikslumas tarp C, G ir
  etalono; formulė dokumentuota).
- Kaip progresyvinis grafikos kadras tampa dviem laukais (kadras 25 fps padalintas, ar šaltinis renderina 50 fps ir
  imamas kas antras eilutės rinkinys) — parinkti, kas teisingiau judančiai grafikai, ir užrašyti.
- OGraf renderis pavyzdžiui (pvz., GStreamer `wpesrc` su permatomu fonu, headless Chromium/CEF ar kt.) — patikrinti,
  kas šiandien realiai duoda RGBA su alfa be GUI, ir pasirinkti paprasčiausią.
- Kaip sintetiniai grafikos šaltiniai gaminami testams (biblioteka iš `gen` stiliaus generatoriaus).

## 7. Kas jau yra ir ką pernaudoti

- `src/media/media_ref.cpp` (`rgb_to_ycbcr10`, logo plokštumų paruošimas, porinė chroma alfa), `src/media/v210.hpp`,
  `media_kernels.cu` (`logo_fields` batch kernel'is), `src/media/cpu/cpu_media.cpp` (C variantas).
- `src/func/logo.cpp` (tick variklis, terminai, skaitikliai), `flows.{hpp,cpp}` (VideoIn/VideoOut, host ir
  `cuda-linear`), `verify.cpp`, `tests/perf/capacity.py`, `compare_report.py`.

## 8. Atsakyti klausimai

- **K1 [autorius]:** flow'uose alfa visada straight (standartas); premultiplied — tik bibliotekos įėjimo tipas,
  biblioteka jį konvertuoja.
- **K2 [autorius]:** grafikos ritmą kiekvienam laukui gali užtikrinti biblioteka (pasirinktinis ritmo režimas, §4.1).
