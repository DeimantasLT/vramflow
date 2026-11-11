# VRAMFlow (lietuviškai)

**Media funkcijos, kurių duomenys gyvena GPU atmintyje ant [MXL](https://github.com/dmf-mxl/mxl), ir bandymų stendas,
kuris įrodo, kad jos elgiasi lygiai taip pat kaip MXL ant CPU.**

Pagrindinis aprašymas — [README.md](README.md) (anglų k., ten ir komandų pavyzdžiai). Čia trumpai.

## Kam to reikia

MXL perduoda nesuspaustą vaizdą, garsą ir duomenis tarp media funkcijų per bendrą RAM. PR
[dmf-mxl/mxl#653](https://github.com/dmf-mxl/mxl/pull/653) prideda `cuda-linear` backend'ą: flow grain'ai laikomi GPU
atmintyje. Šis projektas atsako į tris klausimus:

1. **Ar teisinga?** Tie patys grain indeksai, laikas, slice'ai, sinchronizacijos grupės ir daliniai grain'ai kaip host
   MXL; turinys — bitų tikslumu.
2. **Ar stabilu?** Ilgi bandymai, sanitizeriai, laikrodžio šuoliai, paleidimas ir stabdymas, daug flow'ų.
3. **Kokia nauda?** Kiek pilnų transliacijos flow'ų telpa į vieną GPU, palyginus su CPU, darant tą patį darbą.

## Variantai

- **C** — CPU apdorojimas, host flow (kaip šiandien).
- **H** — GPU apdorojimas, host flow (MXL v1.1 + kopijos į GPU ir atgal).
- **G** — GPU apdorojimas, `cuda-linear` flow (PR #653): grain'ai neišeina iš GPU.

## Rezultatai (RTX 4070 Laptop 8 GB, WSL2 — orientaciniai)

| Funkcija | C N\* | G N\* | CPU branduoliai flow'ui, C → G |
|---|---|---|---|
| Logotipo įkepimas, MXL → MXL | 16 | ≥ 64 (riba nepasiekta) | 0,057 → 0,005 |
| H.264 dekodavimas, TS/UDP → MXL | 8 | 24 (16 griežtai) | ~0,55 → ~0,04 |
| Pilnas flow: TS → dekodavimas → logotipas → H.264 + AAC → TS | 4 | ≥ 12 (riba nepasiekta; žr. pastabą) | ~2,0 → ~0,33 |

GPU apkrova pilname flow ties N=12: CUDA branduoliai ~10 %, NVENC 64 %, NVDEC 50 %, ~34 W. Riba — vienintelis NVENC
variklis, didžioji GPU dalis laisva. Labai mažoje apkrovoje (N=2) nešiojamo kompiuterio GPU numeta dažnį
1605 → 210 MHz ir praleidžia terminus; nuo N=4 viskas praeina, todėl realaus laiko darbui serveryje reikia užfiksuotų
GPU dažnių. Dec ir grandinės matavimai vyko su Windows „Whisper“ (tyliuoju) profiliu. Visos lentelės:
[`docs/rezultatai/2026-11-11-gpu-vs-cpu.md`](docs/rezultatai/2026-11-11-gpu-vs-cpu.md).

N\* — didžiausias flow'ų skaičius, kai ši ir visos mažesnės pakopos praėjo (0 teisingumo klaidų, praradimai ≤ 0,2 %).
Autoritetingi skaičiai bus po pakartojimo natyviame Linux.

## Alfa grafikos kompozitorius

Grafika MXL'e — paprastas flow'as. MXL standartas jau turi `video/v210a`: v210 fill plokštuma + 10 bitų key plokštuma
(straight alfa, video diapazonas 64–940). VRAMFlow prideda įrankius tokiems flow'ams kurti ir sujungti — ant CPU,
ant GPU su host flow'ais ir ant GPU su `cuda-linear` (C / H / G), bitų tikslumu prieš vieną nepriklausomą etaloną.

- **`libvramflow-gfx`** (`include/vramflow/gfx.h`, C ABI): bet kuri programa (rendereris, OGraf/HTML puslapis,
  žaidimų variklis, skriptas) paduoda RGBA kadrus (host arba CUDA atmintis, straight arba premultiplied alfa) ir gauna
  standartinį `video/v210a` flow'ą. Rankinis režimas (`vfgfx_write(index, kadras)`) arba ritmo režimas
  (`vfgfx_submit(kadras)`; bibliotekos gija kas lauką pagal TAI commit'ina paskutinį kadrą).
- **`vramflow-func gfxgen`** — sintetiniai v210a šaltiniai (N × K flow'ų, logo 384×216 arba pilnas 1920×1080).
- **`vramflow-func composite`** — bazinis vaizdas + K grafikos sluoksnių (1–8; padėtis, nepermatomumas) → `v210` arba
  `v210a`. `--base none --out-format v210a` sukuria vien grafikos sluoksnį, `--layer-flow K:UUID` paduoda vieno
  kompozitoriaus išėjimą kitam — kompozitorius galima jungti grandine.
- **`vramflow-func verify --mode composite`** — atkuria laukiamą išėjimą CPU etalonu ir tikrina.

Grafikos flow'as elgiasi kaip bet kuris kitas: jei sluoksnio šaltinis sustoja ar perkraunamas, kompozitorius nieko
neišgalvoja — to išėjimo grain'as necommit'inamas (skaitytojai mato tarpą, kaip ir sustojus video writer'iui), o
šaltiniui grįžus kompozitorius vėl prisijungia.

Būsena: biblioteka, `gfxgen`, `composite` ir `verify --mode composite` padaryti ir ištestuoti (vienetų, GPU ir
funkciniai testai; visi 12 režimų × formatų derinių C/H/G praeina etalono patikrą). N × K talpos kampanija ir OGraf
pavyzdys — vykdomi.

## Kaip naudotis

1. Ubuntu 24.04 (natyviai arba WSL2), NVIDIA GPU, CUDA 13.x, Clang 19, CMake, Ninja, vcpkg, FFmpeg dev bibliotekos.
2. Kernel'io TAI poslinkis turi būti 37 s (žr. `docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md`).
3. MXL su PR #653 (komanda `6dddc91d92f5`) → `bash scripts/wsl/10-build-mxl.sh`.
4. Šis repo → `VERIFY_BUILD=$HOME/build/vramflow/func bash scripts/wsl/30-build-verify.sh -DMXLV_BUILD_FUNC=ON`.
5. Visos patikros → `bash scripts/wsl/run-all.sh --soak skip`.
6. GPU prieš CPU → `bash scripts/wsl/71-compare.sh <katalogas> logo|dec|chain <N sąrašas>`, ataskaita →
   `python3 tests/perf/compare_report.py report.md <katalogai…>`.

Naudojama tik sintetinė bandomoji medžiaga.

## Licencija ir autorystė

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Galima naudoti, keisti, platinti ir parduoti, taip pat
uždaruose produktuose. Platinant kodą ar jo išvestinį darbą, privaloma palikti NOTICE failą su autoriaus vardu.
Jei šis darbas panaudotas produkte, standarte ar publikacijoje, prašome nurodyti: **„VRAMFlow, autorius Deimantas
Kiršinas“** su nuoroda į šį repo.

Sukurta padedant DI programavimo asistentams (Claude). Autorius peržiūrėjo dizainą, visus matavimus atliko ant tikros
įrangos ir atsako už turinį.
