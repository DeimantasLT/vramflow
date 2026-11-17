# VRAMFlow (lietuviškai)

**MXL media flow'ai GPU atmintyje ir jų apdorojimas ten pat — su tuo pačiu laiku ir elgsena kaip MXL ant CPU.**

Išsamus aprašas su visomis komandomis ir kodo pavyzdžiais — [README.md](README.md) (anglų k.). Čia — trumpai, ta pačia
skyrių tvarka; jei kas nors skiriasi, teisingas README.md.

[MXL](https://github.com/dmf-mxl/mxl) leidžia media funkcijoms keistis nesuspaustu vaizdu, garsu ir duomenimis per
bendrą atmintį (įprastai — RAM). Su `cuda-linear` backend'u iš
[dmf-mxl/mxl#653](https://github.com/dmf-mxl/mxl/pull/653) flow'as gali gyventi GPU atmintyje (VRAM), todėl viena GPU
funkcija perduoda kadrus kitai nekopijuodama per CPU.

| Dalis | Kas tai |
|---|---|
| `vramflow-func` | Paruoštos media funkcijos ant MXL flow'ų: generatorius, tikrų šaltinių priėmimas su garsu (MPEG-TS per UDP / RTP / SRT, HLS), logotipas, H.264 dekodavimas, H.264 + AAC kodavimas, alfa grafikos kompozitorius, išėjimo patikra. Kiekviena veikia ant CPU arba GPU. |
| `libvramflow-gfx` | C biblioteka: RGBA kadrai iš bet kurios programos → standartinis MXL grafikos flow'as (`video/v210a`, fill + key). |
| `vramflow-verify` | Bandomasis writer'is ir reader'is, tikrinantys kiekvieną grain'ą (turinys, indeksas, laikas). |
| Testai ir įrankiai | Vienetų, GPU ir funkciniai testai; talpos matavimas (kiek flow'ų telpa). |

Viskas C++ / CUDA; Python — tik testų ir matavimų skriptai.

## 1. Pagrindinės sąvokos

- **Domenas** — katalogas bendroje atmintyje (`/dev/shm/...`); visi jį naudojantys procesai mato vienas kito flow'us.
- **Flow** — vienas srautas (pvz. vienas video signalas) su UUID ir aprašu (`flow_def.json`).
- **Grain** — vienas flow'o vienetas: interlaced formatui — **laukas** (1080i50 = 50 laukų/s), progresyviam — kadras.
  Indeksas skaičiuojamas iš TAI laiko, todėl visi procesai sutaria, kuris grain'as kuriam momentui.
- **Writer / reader** — writer'is sukuria flow'ą ir commit'ina grain'us, reader'ių gali būti kiek nori.
- **Saugykla** — `host` (RAM) arba `cuda-linear` (GPU atmintis). API abiem ta pati.
- **v210 / v210a** — 10 bitų 4:2:2 video; `v210a` turi dar key (alfa) plokštumą.

| Variantas | `--impl` | `--storage` | Kada |
|---|---|---|---|
| **C** — CPU | `cpu` | `host` | Be GPU arba palyginimui su dabartine sistema |
| **H** — GPU, RAM flow'ai | `gpu` | `host` | GPU apdorojimas, bet kitoms programoms flow'ai reikalingi RAM'e |
| **G** — GPU, GPU flow'ai | `gpu` | `cuda-linear` | Geriausia: kadrai lieka GPU |

Visi trys duoda bitų tikslumu vienodą rezultatą (tikrinama prieš vieną nepriklausomą etaloną).

## 2–3. Reikalavimai ir surinkimas

Ubuntu 24.04 (natyviai arba WSL2), GPU variantams — NVIDIA GPU su NVENC / NVDEC ir CUDA 13.x; Clang 19 (arba GCC 13+),
CMake ≥ 3.28, Ninja, vcpkg, `libavformat-dev libavcodec-dev libavutil-dev`, `ffmpeg`, Python 3 (tik testams). Sistemos TAI
poslinkis turi būti 37 s (kitaip funkcijos nestartuoja be `--allow-bad-clock`). Surinkimas — README.md 3 skyrius
(MXL su PR #653 → `10-build-mxl.sh` → `30-build-verify.sh -DMXLV_BUILD_FUNC=ON`).

## 4. Greita pradžia

README.md 4 skyriuje — pavyzdžiai, kuriuos galima nukopijuoti:

1. pirmas writer'is ir reader'is RAM'e (`vramflow-verify writer/reader --storage host`);
2. tas pats GPU atmintyje (`--storage cuda-linear`);
3. media funkcija: `gen` → `logo` → `verify` (be GPU: `--impl cpu --storage host`);
4. grafika su skaidrumu ant video: `gen` + `gfxgen` → `composite` → `verify`;
5. tikras šaltinis su garsu: sintetinis klipas (`62-make-av-clip.sh`) per vietinį UDP siuntėją (`tests/perf/ts_burst.py`) →
   `ingest` → `verify --mode ingest` (A/V poslinkis kiekvienam impulsui);
6. domeno turinys: `mxl-info -d <domenas> -l`.

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
  [EBU OGraf](https://github.com/ebu/ograf) grafiką, `ograf_feed` per `vfgfx_submit` paverčia ją `v210a` flow'u, `composite`
  uždeda ant `gen` video.
- Grafikos sluoksnis, kuris dingo, vėluoja ar grįžo kitu formatu, tam kadrui tampa permatomas (`layer_absent`) — programa
  (bazė) tęsiasi be spragos; sluoksnis laukiamas tik iki pirmo pjūvio a(i) ir vėl naudojamas po `--layer-return-frames`
  kadrų laiku (numatyta 12).

## 6. Komandos

`vramflow-func gen | ingest | logo | dec | enc | gfxgen | composite | verify | keepwarm` — lentelė su pagrindiniais parametrais
README.md 6 skyriuje; visi parametrai — `--help`. Kiekvienas paleidimas rašo `summary-<komanda>.json`.

- **Tarnyba arba matavimas.** `--duration 0` (numatyta funkcijoms) — veikia iki `SIGINT` / `SIGTERM`; `--duration S > 0` —
  matavimo paleidimas. Antras signalas — išeinama iškart (eilės neišleidžiamos, flow'ai atlaisvinami).
- **Formatai** (`--format`, ta pati reikšmė visai grandinei): `1080i50` (numatytasis), `1080i59.94`, `1080p25|50|29.97|59.94`,
  `720p…`, `2160p…`. Interlaced — grain'as yra laukas (viršutinis pirmas), progresyvus — kadras. Progresyvus 4:2:0 → 4:2:2 —
  kadro chroma (bitų tikslumu kaip FFmpeg bilinear), interlaced — lauko chroma. Galas į galą tikrinti 1080i50, 1080p25,
  1080p50 (ir 720p50 `logo`); BFF atmetamas.
- **Spragų politika** (rašytojai `gen`, `logo`, `composite`, `dec`): commit'inamas **kiekvienas** indeksas. Be turinio —
  `--on-loss invalid` (numatyta: INVALID grain'as su juodu turiniu, niekas neišgalvojama), `freeze` (paskutinis geras to paties
  pariteto ne ilgiau kaip `--hold-ms`, tada juoda) arba `black`. Ne šios funkcijos praradimai skaičiuojami atskirai:
  `gap_startup` (įėjimas dar neprisijungė), `gap_upstream` (įėjimas pats pateikė INVALID), `gap_align` (interlaced — iki kadro
  ribos).
- **Paleidimo tvarka nesvarbi.** `logo` numatytai startuoja iškart (`--start immediate`): išėjimai sukuriami tuoj pat, iki
  įėjimo — spragos pagal `--on-loss` (`gap_startup`, ne praradimas); `--start wait-all` — sena elgsena. `composite`, `enc`,
  `verify` įėjimų laukia: `--input-wait-ms 0` (numatyta) — neribotai, kas 5 s `laukiu flow …` (ir kodėl esamas flow netinka);
  `N` — ne ilgiau kaip N ms, tada exit `4` su trūkstamų flow'ų sąrašu.
- **Gamintojas dingo ir grįžo.** Skaitytojas tęsia, spragos indeksai commit'inami pagal `--on-loss` (`missing_fields`,
  `gap_invalid`), grįžus gamintojui prisijungia pats (`reattached`). Prisijungimas (reader'is ir `cuda-linear` atvaizdavimas)
  vyksta pagalbinėje gijoje, ne tick'e — tick'ui jis nieko nekainuoja. Grįžo kitu formatu — `--on-format-change exit`
  (numatyta, exit `6`) arba `wait`.
- **Grandinės laikas (briaunos sutartis).** Kiekvienas išėjimas flow aprašo žymoje skelbia, iki kada po `T(i)` grain'as
  commit'inamas (`urn:x-vramflow:tag:commit-bound-ns`) ir kiek turinys vėluoja nuo kilmės (`…:content-delay-ns`). Skaitytojas
  tick'ą atidaro `T(i) + E + ro` (E — didžiausia įėjimų riba), todėl niekas nevėluoja „pagal konstrukciją“, o vėlinimas
  kaupiasi matomai (50 Hz `gen → logo → logo → composite → enc`: 8; 16,5; 25; 33,5; 42 ms). Svetimam rašytojui be žymų —
  `--input-expect-ms`; skaityti anksčiau — `--input-offset K`; sukaupta delsa turi tilpti į domeno ring'ą (200 ms numatyta).
- **Domeno nuoma.** Kiekvienas procesas laiko `<domenas>.vramflow/<run-id>.lease` šalia domeno (`flock`, atsilaisvina ir po
  `kill -9`); `--run-id` — paleidimo vardas. Flow'o, kurį jau rašo kitas gyvas paleidimas, sukurti negalima (exit `2`,
  klaidoje — savininko run ID). Funkcijos domenų netrina. Nuomos katalogas atidaromas nesekant nuorodų ir atmetamas (exit
  `2`), jei tai ne katalogas, jei jo savininkas nėra nei šis procesas, nei domeno katalogo savininkas, arba jei svetimo
  katalogo teisės platesnės nei domeno; failai jame kuriami `O_EXCL | O_NOFOLLOW`, svetimi atidaromi tik paprasti (FIFO
  nepakabina). Prieiga prie katalogo — pagal RAŠYMO teisę į domeną (`0755` domenas → `0700` katalogas ir `0600` failai;
  `0770` → `0770` ir `0660` su domeno grupe), nuo `umask` nepriklauso: `flock` užtenka skaitomo failo, todėl visiems
  skaitomas užrakto failas leistų pašaliečiui pakabinti domeno procesus. Nuomos skelbiamos ir negyvos šalinamos tik po
  katalogo užraktu (`lease.lock`), užrakto laukiama ne ilgiau kaip 5 s. Katalogo vardas — iš kanoninio domeno kelio, todėl
  tas pats domenas kitu keliu (simbolinė nuoroda) turi tą pačią nuomą ir tą pačią knygą. Daugiau — `SECURITY.md`.
- **Domeno kelias GPU (`cuda-linear`) flow'ams — ne ilgesnis kaip 48 simboliai** (skaitytojo lizdo
  `<domenas>/<flow>.mxl-flow/payload.sock` kelias turi tilpti į 107 simbolius); ilgesnį `vramflow-func` atmeta iškart.
- **`ingest` (tikri šaltiniai su garsu).** `--input URL` (kartoti; `udp://`, `rtp://`, `srt://`, `http://…m3u8`, failas) → kiekvienam
  šaltiniui video flow'as (`07`, v210) ir garso flow'as (`08`, 48 kHz float32). Laiko modelis paprastai: kiekvienas paketas
  pažymimas vieną kartą, kai demuxeris jį perskaito — *išėjimo laikas = šaltinio laiko žyma + vienas šaltinio poslinkis*, bendras
  video ir garsui; poslinkį nustato pirmas video paketas taip, kad kadras būtų rodomas `--latency-ms` (numatyta 1000; HLS — ne
  mažiau kaip 3 segmentai) po atvykimo. Todėl kadras patenka į savo laiko žymos MXL indeksą, ne į atvykimo — pliūpsniai ir
  sulaikyti paketai vaizdo garso atžvilgiu nepastumia. Paketas į dekoderį (`--decoder nvdec | cuvid | sw`) paduodamas
  `--decode-lead-ms` (numatyta 400) prieš savo laiką. Šaltinio laikrodžiui skubant ar vėluojant garsas seka jį mėginio tikslumu
  (resampleris keičia greitį ne daugiau kaip `--max-comp-ppm`, numatyta ±0,1 %), o vaizdas pakartoja arba praleidžia visą kadrą
  (interlaced — laukų porą), kai nuo garso nutolsta pusę kadro — kaip frame synchronizer'is, A/V ≤ 1 laukas. Kadras, kuris nebespės,
  nededamas į dekoderį — praleidžiama iki kito rakto kadro (perkrova nesugriūna). Formatas nustatomas iš srauto prieš kuriant
  flow'us (laukų tvarka — iš pirmo IDR); 4:2:2, 10 bitų, BFF atmetami su įvardytu formatu (exit `2`); formatas pasikeitė vidury
  srauto — tas šaltinis sustabdomas, kiti tęsia, exit `6`. 50 Hz ir 25p šaltiniai — viename procese, 59,94 šeima — atskirame.
  **Saugumas:** HLS sargas — papildoma apsauga aplink libavformat HTTP / HLS klientą, ne smėlio dėžė (atsakymus vis tiek analizuoja
  ir nukreipimus seka pats libavformat), todėl HLS šaltiniai turi būti operatoriaus sukonfigūruoti, žinomi serveriai. `--input` schema — tik `udp`, `rtp`, `srt`, `http`, `https` arba vietinis failas (kita — exit `1`); kiekvienam
  libavformat atidarymui (ir įdėtiems) — `protocol_whitelist` / `format_whitelist`. HTTP(S) šaltinį pirma užklausia pats ingest; HLS
  atpažįstamas pagal turinį ir libavformat gauna būtent tą atsakymą (antros viršutinio grojaraščio užklausos nėra); ne HLS — tik MPEG-TS,
  užklausiamas antrą kartą sargo atidarymu (tiesioginis MPEG-TS — dvi užklausos sesijai; vienas atidarymas, M3U8 atsakymas atmetamas). Kiekvienas libavformat atidaromas atsakymas peržiūrimas (pirmi 4 KiB) be pakartotinės užklausos (peržiūrėti baitai
  atiduodami iš atminties); dydis imamas tik iš atsakymo antraščių, be ieškojimo (`Content-Length` ≥ 2^63 naujų užklausų nesukelia). M3U8 priimamas tik iš grojaraščio URL (viršutinis ir jo galutinis URL, patikrinto master variantai,
  jų perkrovimai); segmentas, raktas ar init sekcija, atsakę grojaraščiu, — atmetami, segmentų demuxeriui `hls` neleidžiamas.
  Kiekvienas toks grojaraštis perskaitomas ir patikrinamas VISAS (≤ `--hls-max-playlist`, 1 MiB) taip, kaip jį skaito libavformat
  (eilutės baigiasi LF, CR, CR LF arba NUL; atributai kabutėse ir be jų): master tik viršuje, ≤ 15 variantų, 10 000 nuorodų ir 4
  `#EXT-X-MAP` eilutės, nuorodos — tik http / https su `://` iš karto po schemos (`http:host/…` atmetamas visur, ir `--input`: libavformat host'ą imtų iš
  kelio), `#EXT-X-TARGETDURATION` ≤ `--hls-max-target-s` (30); bet kuri `#EXT-X-MEDIA` eilutė (ir be `URI`) — master lygmuo: viršuje ≤ 32,
  variantuose ir perkrovimuose — nė vienos; libavformat init sekcijų (`#EXT-X-MAP`) ir `#EXT-X-MEDIA` neatlaisvina iki sesijos pabaigos —
  init sekcijų sąmata ≤ 16 MiB sesijai; kiekvienam segmentui libavformat kopijuoja jo URL ir galiojančio AES-128 rakto URL (iki 4 KiB) —
  ši sąmata ≤ 16 MiB grojaraščiui ir ≤ 64 MiB visiems sesijos grojaraščiams (URL keliuose master įrašuose skaičiuojamas kiekvienam
  įrašui: libavformat kiekvienai `#EXT-X-STREAM-INF` ir `#EXT-X-MEDIA` su `URI` laiko atskirą grojaraštį); pažeidimas baigia tik tą sesiją. HTTP skaitymo riba 10 s, prisijungimo iš naujo atidarymas — ≤ `--input-wait-s` (30 s).
  Paketų eilės ribotos baitais (`--max-queue-mb` — kiekvienai eilei, šaltinis turi dvi: video ir garso; perteklius išmetamas ir skaičiuojamas, žymės — niekada), garso žiedas ≤
  `--max-audio-ring-mb` (256); slaptažodžiai ir užklausos reikšmės žurnaluose, santraukoje ir statistikoje — `***` (kabutėse — iki
  uždarančios kabutės; neišvaloma: tarpas užklausos reikšmėje be kabučių, paslaptis URL kelyje). SRT slaptažodis — `--srt-passphrase-from
  env:VARDAS | file:/kelias` (ne `/proc/*/cmdline`; `+` neleidžiamas — libavformat jį paverčia tarpu). **SSRF:** grojaraštis gali nukreipti procesą į bet kurį pasiekiamą http(s)
  host:port (ir loopback, debesų metaduomenų adresus), o bet kuris segmentas, raktas ar grojaraštis — HTTP nukreipimu (libavformat
  seka iki 8, FFmpeg 6.1 to išjungti neleidžia). `--hls-same-host` leidžia tik viršutinio grojaraščio serverį (autoritetas skaidomas tiksliai kaip libavformat `av_url_split`;
  host lyginamas kaip tekstas, be raidžių dydžio ir galinio taško, numatytas prievadas 80 / 443; autoritetą turi tik tiksliai
  mažosiomis rašytos schemos `http`, `https`, `crypto+http(s)` — libavformat TLS įjungia tik mažosiomis `https`, todėl nukreipimas į
  `HTTPS://host/` yra paprastas HTTP į 80 prievadą ir atmetamas). Apribojimas užsidaro: jei viršutinio grojaraščio host:port nustatyti
  nepavyksta (tuščias arba ilgesnis nei 1023 baitai host'as), šaltinis atmetamas, o ne paleidžiamas be apribojimo. Nepriklausomai nuo
  `--hls-same-host`, viršutinio `http(s)` atsakymo galutinis URL po nukreipimų turi prasidėti tiksliai `http://` arba `https://`
  (libavformat seka nukreipimą į bet kokią schemą, pvz. `ftp://`, ir vis tiek jungiasi HTTP į jos host:port) — kitaip šaltinis
  atmetamas. Su `--hls-same-host` tikrinamas tekstas, URL prieš atidarymą ir galutinis URL po nukreipimų prieš
  skaitant turinį — kitas serveris baigia sesiją, jo atsakymas nenaudojamas. Neužkertama: pačios nukreipimų užklausos (libavformat jas atlieka
  atidarymo viduje, prieš sargui pamatant galutinį URL; jų turinys nenaudojamas), host'ai, rodantys ne ten, kur atrodo (DNS rebinding,
  kitas tos pačios mašinos vardas ar IP — lyginamas tik host'o tekstas), ir AES-128 segmentų (`crypto+http…`) galutinis URL (nematomas —
  tikrinamas tik URL prieš atidarymą). URL apdorojimas modeliuotas pagal libavformat be `HAVE_DOS_PATHS`; Windows — ne tikslinė platforma.
  libav `https` numatytai sertifikatų netikrina (`tls_verify=0`). HTTP antraščių eilučių skaičius libavformat neribojamas, `Icy-*`
  reikšmės jungiamos be ribos (`http.c` n6.1:877-894; `Set-Cookie` irgi kaupiami) — priešiškas serveris gali auginti vieno
  atidarymo atmintį iki atidarymo pabaigos: ribojama tik atidarymo riba (`--input-wait-s`, prisijungiant iš naujo — 30 s) ir 10 s
  skaitymo riba, ne dydžiu. Prieš atidarymą atmesti URL (schema, `http:` be `//`, kitas serveris su `--hls-same-host`) į atidarymų
  spartos ribą (50 / s) neįskaičiuojami; jų skaičių sesijai riboja grojaraštis (≤ 10 000 įrašų) ir libavformat perkrovimų riba
  (`max_reload` 3, `hls.c` n6.1:1500-1503, 2568-2569). **libavformat nukreipimų kešo kilpa (FFmpeg ≤ 8.0) ir sargas:**
  FFmpeg 6.1 `http.c` kešuotas nukreipimas grįžta į atidarymo pradžią neskaičiuodamas nukreipimų ir netikrindamas pertraukimo
  kabliuko (`http.c` n6.1:362-373), o `301` / `308` kešuojami be galiojimo pabaigos (425-431): serveris, atsakantis
  `301 Location: <tas pats URL>` (ar A → B → A), suka atidarymą 100 % CPU be I/O — 10 s skaitymo riba, `--input-wait-s` ir
  sustabdymas nesuveikia, FFmpeg 6.1 kešo išjungti neleidžia (6.1 patikrinta šaltinyje; iki 8.0 — tas pats kelias, FFmpeg master pataisė:
  `max_redirects` tikrinamas ir kešo kelyje — čia nepertikrinta).
  Gijos libavformat viduje saugiai atšaukti negalima, o ji laikytų ir sustabdymą (procesas laukia visų šaltinių gijų). Todėl
  kiekvienas libavformat kvietimas stebimas: ilgiau nei jo pertraukimo riba (`--input-wait-s` atidarant, tyla nuo paskutinio paketo
  skaitant) + `--libav-watchdog-s` (numatyta 30 s) arba be pertraukimo kabliuko kvietimo `--libav-watchdog-s` (teisėti laukimai
  jį kviečia nuolat: kiekvienas skaitymo bandymas, HLS perkrovimo laukimas kas 100 ms) — žurnale šaltinis ir kvietimas, procesas iškart baigiamas exit `7` (ir sustabdymo metu); santrauka
  nerašoma, flow'ai lieka kaip po gedimo (kitas paleidimas juos surenka). Paleiskite su prižiūrėtoju (pvz. systemd
  `Restart=on-failure`); visi proceso šaltiniai paleidžiami iš naujo kartu. Atsarga dengia DNS užklausą (glibc: iki 5 s × 2 bandymai
  vienam vardų serveriui). **Patikimumas:** dreifo atsarga laikoma grįžtamuoju ryšiu (be perfiksavimo valandomis, ir HLS),
  slydimai — pateikimo tvarka, perfiksavimas — žurnale su priežastimi ir dydžiu, garsas — su 5 ms perėjimu; dekodavimas nuo
  IDR / recovery point / I pjūvio; NVDEC nepavykus — programinis dekoderis; dingęs / nedekoduojamas garsas po prisijungimo iš
  naujo — tyla; gyvi šaltinio skaitikliai `--stats-file` eilutėse (`flows[n].fn`). **Riba:** laikrodis fiksuojamas pagal pirmą
  video paketą, ne PCR — srautui su VBV vėlinimu ≥ `--latency-ms` latenciją reikia didinti.
- **Talpa ir priėmimo kontrolė** (`--admission auto | off | warn | enforce`, `--capacity-file`, `--admission-headroom`). Talpa —
  tik išmatuota: `PROFILE=knee scripts/linux/70-capacity.sh [katalogas]` randa kiekvienos funkcijos kelią
  (`tests/perf/capacity.py --n auto`, 3 pakartojimai), o pabaigoje `tests/perf/capacity_file.py <kampanija>` įrašo `capacity-<GPU UUID>.json` (NVDEC / NVENC Mpix/s su `bound`, NVENC sesijų skaičius,
  kai kelią nulėmė tvarkyklės riba, VRAM srautui kiekvienai funkcijai, GPU UUID ir tvarkyklė). Prieš kurdamas išėjimų flow'us
  procesas rezervuoja savo srautų paklausą (Mpix/s varikliui — 1080i50 / 1080p25 = 51,84 srautui, NVENC sesijos, VRAM baitai)
  knygoje šalia domeno (`<domenas>.vramflow/*.adm`, failas procesui su `flock` — po `kill -9` rezervacija dingsta iškart;
  skaitymas, sprendimas ir įrašymas — po vienu užraktu, todėl du procesai talpos neperskirsto). Riba — talpa × 0,85 (NVENC
  sesijoms — tikslus skaičius). Priimama iš eilės: `enforce` — pirmas netilpęs srautas ir visi po jo nepaleidžiami (jų flow'ų
  domene nėra), pranešime variklis ir skaičiai (`flow 4: atmestas — nvdec: prašoma 51,84, užimta 207,36, leidžiama 229,50
  Mpix/s`); nė vieno — exit `5` iškart, dalis — priimtieji veikia, pabaigoje exit `5`. Tikrinama: `ingest`, `dec` — NVDEC +
  VRAM (išmatuotas arba įvertis); `enc` NVENC — NVENC Mpix/s + sesijos + VRAM (jei išmatuotas); `enc` x264, `gen`, `gfxgen`,
  `logo`, `composite` — tik VRAM, jei talpos faile yra `functions.<žymė>`; `verify` ir `--impl cpu` — netaikoma. Numatyta `auto`
  (plano sprendimas K13): su talpos failu — kaip `enforce`, be failo — kaip `warn` (viena įspėjimo eilutė); aiškus `enforce` be
  failo — exit `1` („talpa neišmatuota, paleiskite 70-capacity“); talpos matavimo skriptai funkcijas leidžia su `--admission off`. Kita GPU (UUID) — failas nenaudojamas; kita tvarkyklė arba
  `lower_bound` — įspėjimas. Riba: knyga mato tik VRAMFlow procesus tame pačiame domene ir GPU; VRAM lyginamas su visa GPU
  atmintimi, ne su laisva.
- **Sąveika su MXL įrankiais.** `mxl-info` skaito visus VRAMFlow flow'us; `mxl-gst-sink` / `mxl-gst-cuda-sink` rodo `gen` ir
  `logo` išėjimus ir per aukštupio pertrauką laiko paskutinį gerą vaizdą (`--on-loss invalid`).

Exit kodai (variklio funkcijos; griežtas — `--strict-exit on` arba `auto` su `--duration S > 0`, švelnus — tarnyboje):

| Situacija | griežtas | švelnus |
|---|---:|---:|
| Tvarkinga pabaiga, nieko neprarasta | 0 | 0 |
| Kas nors prarasta (spraga, vėlavimas, praleistas indeksas, išėjimo klaida) | 4 | 0 |
| Sugedo vienas vienetas (kiti tęsia) | 2 | 0 |
| Sugedo visi vienetai; paleidimo ar vykdymo klaida | 2 | 2 |
| Netinkamos parinktys | 1 | 1 |
| Įėjimai neatsirado per `--input-wait-ms` | 4 | 4 |
| Įėjimas grįžo kitu formatu (`--on-format-change exit`) | 6 | 6 |
| Tik `ingest`: libavformat kvietimas užstrigo (nukreipimų kešo kilpa); iškart, be santraukos | 7 | 7 |
| Priėmimo kontrolė atmetė srautus (`--admission enforce` arba numatytoji `auto`, kai rastas talpos failas): nė vieno nepriimta — iškart; dalis priimta — jie veikia, pabaigoje (aukštesnė klaida — viršesnė) | 5 | 5 |
| Aiškiai nurodytas `--admission enforce` be tinkamo talpos failo (numatytoji `auto` tik įspėja) | 1 | 1 |

`verify` — exit `3`, jei yra teisingumo klaidų.

## 7. Testai ir matavimai

`bash scripts/linux/run-all.sh --soak skip` — viskas (zondai, upstream testai, vienetų / GPU / funkciniai testai, ataskaita);
talpa — `tests/perf/capacity.py` (ingest — `tests/perf/ingest_capacity.py`), CPU prieš GPU — `scripts/linux/71-compare.sh` +
`tests/perf/compare_report.py`; ingest testų klipai — `scripts/linux/62-make-av-clip.sh`. Tik sintetinė medžiaga.

## 8. Rezultatai (RTX 4070 Laptop 8 GB, WSL2 — orientaciniai)

| Funkcija | CPU (C) N\* | GPU su GPU flow'ais (G) N\* | CPU branduoliai flow'ui, C → G |
|---|---|---|---|
| Logotipo įkepimas, MXL → MXL | 16 | ≥ 64 (riba nepasiekta) | 0,057 → 0,005 |
| H.264 dekodavimas, TS/UDP → MXL | 8 | 24 (16 griežtai) | ~0,55 → ~0,04 |
| Pilna grandinė: TS → dekodavimas → logotipas → H.264 + AAC → TS | 4 | ≥ 12 (riba nepasiekta) | ~2,0 → ~0,33 |

N\* — didžiausias flow'ų skaičius, kai ši ir visos mažesnės pakopos praėjo (0 teisingumo klaidų, praradimai ≤ 0,2 %).
Pilnoje grandinėje ties N=12 nė vienas variklis neprisotintas (NVENC 63 %, NVDEC 51 %); 2026-11-17 kartojimas N=12
krenta dėl vėlavimų, priežastis nenustatyta (ankstesnė šio README versija klaidingai nurodė NVENC). Visos lentelės:
[`docs/rezultatai/2026-11-11-gpu-vs-cpu.md`](docs/rezultatai/2026-11-11-gpu-vs-cpu.md).

**Talpos kelias su pakartojimais (2026-11-17, P8):** tik G, 30 s pakopa, pakopa skaitoma tik jei praeina **visi 3**
pakartojimai (`capacity.py --n auto`).

| Funkcija | Kelias N | Ką nulėmė riba | VRAM srautui (R²) |
|---|---|---|---|
| Logotipo įkepimas | ≥ 64 | paieškos riba | 56,0 MB (1,000) |
| NVENC `ll` kodavimas | 12 | **tvarkyklės NVENC sesijų riba** | 126,2 MB (1,000) |
| H.264 dekodavimas (NVDEC) | 22 | NVDEC pralaidumas | 182,2 MB (0,997) |
| Pilna grandinė | 11 | vėlavimai ties 12, priežastis nenustatyta | 284,3 MB (0,999) |
| Kompozitorius, K = 2 | ≥ 32 | paieškos riba | 119,1 MB (1,000) |

GeForce tvarkyklė leidžia 12 vienalaikių NVENC sesijų; 13-a krenta (`incompatible client key`) ir nuverčia visą `enc`
procesą. Kompozitorius K = 2: nuo N\* 2 (lentelė žemiau, GPU 210–690 MHz) iki ≥ 32 su `keepwarm` (550–1112 MHz).
Išsamiai: [`docs/rezultatai/2026-11-17-p8-talpa.md`](docs/rezultatai/2026-11-17-p8-talpa.md).

Grafikos kompozitorius (`composite`), bazė + K sluoksnių → `v210`, 60 s pakopa, terminas 30 ms, N\* (tolerantiškai / griežtai):

| Sluoksnių K | C — CPU | H — GPU, RAM flow'ai | G — GPU, GPU flow'ai |
|---|---|---|---|
| 1 (logotipas 384×216) | 16 / 2 | 4 / 1 | 4 / 4 |
| 2 (logotipas + pilno ekrano) | 8 / 4 | 4 / 0 | 2 / 0 |
| 4 (logotipas + 3 × pilno ekrano) | 4 / 4 | 4 / 2 | 0 / 0 |

Lentelę skaityti atsargiai (energijos profilis „Whisper“, GPU dažnis G bėgimuose 210–690 MHz, kiekviena pakopa — kartą);
išsamiai: [`docs/rezultatai/2026-11-11-composite.md`](docs/rezultatai/2026-11-11-composite.md).

**Ingest** (sintetinis H.264 + AAC per UDP, 30 s pakopa, kiekvienas šaltinis tikrintas): NVDEC (G) 1080i50 ir 1080p25 — N\* ≥ 16
(pakopų galas), programinis (C) — ≥ 8; VRAM šaltiniui 135 MiB, CPU 0,10 (G) / 0,25 (C) branduolio; A/V kiekvienam impulsui
≤ 0,52 ms. Funkciniuose testuose A/V 0,0 ms per UDP, RTP, SRT ir HLS; 0,5 % greitesnis šaltinis — 13 kadro slydimų, A/V ≤ 18,8 ms,
be girdimų trūkių. Išsamiai: [`docs/rezultatai/2026-11-14-ingest.md`](docs/rezultatai/2026-11-14-ingest.md).

**GPU dažniai.** Nešiojamo kompiuterio GPU (WSL2 / WDDM) iš P8 į P3 pereina sustabdydamas visų procesų GPU darbą 50–350 ms;
kol eteris gyvas, laikykite `vramflow-func keepwarm` (64 MB kas 5 ms; +8,5 W) — testų įrankiai jį paleidžia patys. Serveriams
— užrakinti dažnius (nepatikrinta). Išsamiai:
[`docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md`](docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md).

## 9–11. Repo, upstream pataisos, ribos

- Struktūra: `include/vramflow/gfx.h` (viešas C API), `src/func/` (komandos), `src/engine/` (bendras tick variklis),
  `src/timing/` (PTS → TAI), `src/gfx/`, `src/media/` (CUDA, CPU, etalonas), `tests/`, `scripts/linux/` (`scripts/wsl/` —
  nukreipimai vienam leidimui), `upstream/`, `docs/` (lietuviškai).
- Upstream: [`upstream/README.md`](upstream/README.md) — U1–U4 PR #653 pataisų juodraščiai (su U1–U3 testai 136/136).
- Ribos: matuota viename nešiojamame GPU per WSL2; variantas H — tik logotipui, kompozitoriui ir transportui; 2 val. soak ir
  pilno sanitizer'ių paleidimo media funkcijoms dar nėra; NVENC šiame GPU koduoja progresyviai (x264 — tikras interlaced);
  VRAM flow'ai — tarp procesų toje pačioje mašinoje ir GPU.

## Licencija ir autorystė

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Galima naudoti, keisti, platinti ir parduoti, taip pat
uždaruose produktuose. Platinant kodą ar jo išvestinį darbą, privaloma palikti NOTICE failą su autoriaus vardu.
Jei šis darbas panaudotas produkte, standarte ar publikacijoje, prašome nurodyti: **„VRAMFlow, autorius Deimantas
Kiršinas“** su nuoroda į šį repo. Su FFmpeg (libx264) susietos programos platinamos pagal GPL (žr. NOTICE).

Sukurta padedant DI programavimo asistentams (Claude). Autorius peržiūrėjo dizainą, visus matavimus atliko ant tikros
įrangos ir atsako už turinį. MXL — EBU ir Linux Foundation projektas; šis repo nuo jų nepriklauso.
