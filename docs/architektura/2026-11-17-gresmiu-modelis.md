# VRAMFlow grėsmių modelis

**Būsena:** 2026-11-17, planas `docs/planai/2026-11-17-production-ready.md` (kriterijai S1, S7). Trumpa versija angliškai —
`SECURITY.md`. Kiekviena priemonė čia turi testą repo arba įrašą „nepriimta, nes …“; priemonė be testo nelaikoma priemone.

## 1. Kas saugoma

| # | Turtas | Ko reikia |
|---|---|---|
| A1 | Eterio signalas (flow turinys) | prieinamumas: kiti domeno procesai tęsia darbą; vientisumas — tiek, kiek pasitikima rašytoju |
| A2 | GPU ir host resursai (VRAM, NVDEC / NVENC talpa, CPU, `/dev/shm`) | nė vienas srautas ar šaltinis negali jų išeikvoti visiems |
| A3 | Host failų sistema | procesas nepaliečia nieko už savo domeno, nuomos katalogo ir išvesties katalogo |
| A4 | Paslaptys (SRT slaptažodis, URL žetonai) | nepatenka į žurnalus, santraukas ir `argv` |

## 2. Pasitikėjimo ribos

| # | Riba | Kas už jos | Kiek pasitikima |
|---|---|---|---|
| B1 | Tinklo įėjimai (`ingest`: TS per UDP / RTP / SRT, HLS per HTTP) | nuotolinio serverio baitai; juos skaito libav (trečiųjų šalių kodas) | nepasitikima |
| B2 | MXL domeno katalogas | kiti procesai, turintys teisę rašyti į domeną | pusiau: gali rašyti bet kokį turinį (tai MXL modelis — prieigą valdo Unix teisės), bet negali sugadinti skaitytojo atminties |
| B3 | `/dev/shm` (teisės 1777) | kiti vietiniai vartotojai gali kurti įrašus šalia domeno | nepasitikima |
| B4 | Operatoriaus konfigūracija (CLI, talpos failas, aplinka) | operatorius | pasitikima, bet klaida joje negali tyliai išjungti apsaugos |
| B5 | Tvarkyklė, CUDA, FFmpeg, MXL | priklausomybės | pasitikima; versijos prisegtos |

## 3. Grėsmės ir priemonės

| # | Grėsmė (riba) | Poveikis | Priemonė | Testas | Likutinė rizika |
|---|---|---|---|---|---|
| T-1 | Srautas ar grojaraštis išnaudoja libav parserio spragą (B1) | kodo vykdymas proceso teisėmis | protokolų ir formatų baltieji sąrašai mažina pasiekiamą libav kodą; build grūdinimas; paleidimas ne root | `tests/unit/test_url_policy.cpp`; `ctest build_hardening`; `scripts/linux/45-nonroot.sh` | libav spragos lieka: `ingest` skirtas tik žinomiems serveriams (README „Security rules“) |
| T-2 | Grojaraštis nukreipia į kitą schemą ar vidinį tinklą (B1) | failų skaitymas, prieiga prie vidinių resursų | nuorodos tik `http` / `https` po išsprendimo; šaltinio schema tik iš sąrašo; `--hls-same-host` | `test_url_policy.cpp`; fuzz `fuzz_playlist`, `fuzz_url` | be `--hls-same-host` leidžiamas bet kuris HTTP serveris; DNS perrišimas — operatoriaus tinklo klausimas |
| T-3 | Šaltinis išeikvoja atmintį: begalė segmentų, init sekcijų, milžiniški paketai (B1) | atminties išsekimas, kiti procesai nukenčia | `HlsLimits` (segmentai, grojaraščiai, baitai sesijai); paketų eilių ribos kiekiu ir baitais; garso žiedo riba | `test_url_policy.cpp`; `tests/unit/test_packet_queue.cpp`; fuzz invariantai | libav vidiniai buferiai už mūsų sąmatų ribų |
| T-4 | libav kvietimas pakimba (B1) | `ingest` nei dirba, nei baigiasi | sargas: exit 7 po `--libav-watchdog-s` | `tests/unit/test_libav_watchdog.cpp` | — |
| T-5 | Paslaptys žurnaluose ar `argv` (B1, B4) | slaptažodis matomas kitiems | `redact_url` visiems žurnalo tekstams; slaptažodis tik per `env:` arba `file:` | `test_url_policy.cpp`; fuzz `fuzz_url` (paslapties žymė) | — |
| T-6 | Kitas vartotojas pirmas sukuria `<domenas>.vramflow` arba padeda jame nuorodą ar FIFO (B3) | failo perrašymas proceso teisėmis; pakibimas; netikras užimtumas | katalogas tikrinamas (ne nuoroda, savininkas, teisės); failai `O_EXCL \| O_NOFOLLOW`; svetimi — tik paprasti, `O_NONBLOCK` | `tests/unit/test_safe_files.cpp` | pirmas sukurtas svetimas katalogas vis tiek neleidžia startuoti (exit 2): domeną laikyti tėviniame kataloge, kurį rašyti gali tik jo vartotojai. Domenas, rašomas visiems, lieka be apsaugos |
| T-6a | Pašalietis užrakina nuomos ar knygos failą: `flock` užtenka skaitymo aprašo (B3) | visi domeno procesai kimba ties knygos užraktu; mirusio proceso nuoma ir rezervacija lieka „gyvos“ | prieiga prie nuomos katalogo — tik tiems, kam domenas rašomas (`0755` domenas → `0700` katalogas); užrakto laukiama ribotai (5 s), negavus — rezervacija atmetama | `safe_files: pašalietis negali atidaryti …`; `safe_files: užimtas knygos užraktas nepakabina …` | grupės narys (turintis rašymo teisę) gali meluoti ar laikyti užraktą — tai pasitikėjimo riba B2 |
| T-6b | Negyvos nuomos šalinimas pagal vardą sutampa su naujos to paties vardo nuomos paskelbimu | pašalinama gyva nuoma: procesas dirba nematomas, tuo pačiu `run_id` gali startuoti kitas | skelbimas ir šalinimas — tik po katalogo užraktu `lease.lock`; šalinama tik jei vardas tebenurodo tą patį failą | `safe_files: tas pats run_id lygiagrečiai — nuomą gauna lygiai vienas ir ji lieka matoma` | — |
| T-7 | Tas pats domenas pasiekiamas kitu keliu (B4) | priėmimo kontrolė talpą išdalija du kartus | nuomos katalogo vardas iš kanoninio kelio | `ledger: tas pats domenas per simbolinę nuorodą — viena knyga` | skirtingi domenai toje pačioje GPU turi atskiras knygas (sprendimas D4). **Konteineriuose:** nuomos katalogas yra šalia domeno, todėl sumontavus tik patį domeną kiekvienas konteineris turi savo nuomas ir savo knygą — montuoti reikia tėvinį katalogą (`SECURITY.md`); tarp konteinerių dar nebandyta (planas 5.4) |
| T-8 | Piktybiškas ar sugedęs rašytojas: netikra grain'o antraštė, aprašas didesnis už saugyklą (B2) | skaitymas už saugyklos ribų (RAM ar VRAM) | `VideoIn::get` tikrina antraštę; `check_layout` — saugyklą prieš rastrą; kopijų dydžiai — iš savo rastro | `tests/gpu/test_hostile_writer.cpp` (host ir `cuda-linear`) | turinio teisingumas negarantuojamas; turinys skaitymo metu gali keistis (MXL to nesaugo) — tik vaizdo defektas |
| T-9 | Piktybiškas flow aprašas: neįmanomi matmenys ar dažnis, sugadinta sintaksė (B2) | perpildymas, avarija | matmenys 1…65536 ir dažnis 1…1 000 000 — griežti sveikieji; dydžiai 64 bitų | `test_hostile_writer.cpp`; fuzz `fuzz_flow_def` (su UBSan, nutraukiančiu paleidimą) | — |
| T-10 | Sugadintas talpos failas išjungia ribą (B4) | priėmimo kontrolė nebesaugo | laukai tikrinami: sveikasis skaičius leistinose ribose, baigtinis | `capacity: skaičius už lauko ribų atmetamas`; fuzz `fuzz_capacity` | talpos failą turi galėti rašyti tik operatorius |
| T-10a | Kito proceso knygos failas su neigiama ar sugadinta eilute (B2) | „užimta“ sumažėja visiems — talpa išdalijama du kartus | neigiamos, nebaigtinės ir sugadintos eilutės neskaičiuojamos | `ledger: neigiama ar sugadinta svetimo failo eilutė neskaičiuojama` | grupės narys gali ištrinti svetimą failą — pasitikėjimo riba B2 |
| T-11 | Per daug srautų vienai GPU (B4) | vėluoja visi išėjimai | priėmimo kontrolė: su talpos failu nauji srautai virš ribos nepaleidžiami | `tests/functional/overload.py` | kiti GPU naudotojai knygoje nematomi; termino vartai — ne šio plano |
| T-12 | Vieno srauto ar proceso gedimas | eterio praradimas kitiems | gedimas stabdo tik savo vienetą; kiti procesai nepriklausomi | `tests/functional/isolation.py`; `fout_error` | GPU lygio gedimas veikia visus tos GPU procesus |
| T-13 | Procesas paleistas `root` | bet kuri spraga — pilna prieiga | viskas veikia ne root | `scripts/linux/45-nonroot.sh` | `--rt-priority` reikia `CAP_SYS_NICE` |
| T-14 | Pakeista priklausomybė (B5) | — | MXL prisegta pagal commit; Catch2 ir CLI11 — iš tos pačios MXL vcpkg bazės; FFmpeg — iš distribucijos | — (planas 6.2: versijų failas) | binarinių paketų tikrumas — distribucijos atsakomybė |

## 4. Nepriimta, ir kodėl

- **Sandbox aplink libav (`seccomp`, vardų erdvės).** Tam reikėtų atskiro demux proceso arba labai plataus leidžiamų kvietimų
  sąrašo (CUDA tvarkyklė dirba per `ioctl`). Vietoje to — išorinė izoliacija: atskiras vartotojas, systemd direktyvos arba
  konteineris be papildomų teisių (`SECURITY.md`).
- **TLS sertifikato tikrinimas HLS šaltiniams.** libav jungiasi su `tls_verify=0`; autoriaus sprendimas — `ingest` skirtas
  žinomiems serveriams, apsauga čia tik papildoma.
- **Flow turinio autentiškumas (parašai).** Ne MXL modelio dalis: kas gali rašyti į domeną, tas gali rašyti turinį.
- **Viena priėmimo knyga visai GPU.** Sprendimas D4: knyga lieka domenui; riba dokumentuota.

## 5. Kas dar nepatikrinta

- Sanitizeriai visam host kodui ir `compute-sanitizer` visiems branduoliams (planas 2.1–2.3) — T-8 GPU pusė iki tol remiasi
  tik funkciniu testu.
- Klaidų kelių testai: VRAM išsekimas, CUDA klaida vidury darbo, pilnas `/dev/shm` (planas 2.5).
