# P4 priėmimo kontrolė (A dalis): talpos failas, knyga, `--admission` (2026-11-17)

**Planas:** `docs/planai/2026-11-11-production-grade.md` §P4, PR 3, 4, 5 (be `--shed`) ir `overload_enforce_synthetic`.
Termino vartai (`DeadlineGate`, PR 1–2), `--shed`, `overload_enforce_measured`, `overload_no_collapse`, `overload_shed_newest`
— kitame etape. **Aplinka:** RTX 4070 Laptop 8 GB, tvarkyklė 596.49, WSL2 Ubuntu 24.04.

## Kas tikrinama

Paklausa ir talpa — Mpix/s varikliui (plotis × aukštis × kadrų dažnis; 1080i50 / 1080p25 = 51,84 srautui), NVENC sesijos ir
VRAM baitai. Talpa — iš P8 talpos failo (`tests/perf/capacity_file.py`, schema 1). Riba: Mpix/s ir VRAM — talpa × 0,85
(`--admission-headroom`); NVENC sesijos — tikslus skaičius (GeForce tvarkyklės riba, 13-a sesija nuverčia visą `enc`
procesą — P8), atsarga netaikoma.

| Funkcija | NVDEC Mpix/s | NVENC Mpix/s | NVENC sesijos | VRAM |
|---|---|---|---|---|
| `ingest` | taip (ne `--decoder sw`) | — | — | `functions.ingest` (jei išmatuota) arba įvertis: (16 DPB + atsarga + kadrai iki `--decode-lead-ms` + 2) NV12 + ring'as v210 |
| `dec` | taip | — | — | `functions.dec` arba įvertis (16 + `--playout-frames` + 2 NV12 + ring'as) |
| `enc --encoder nvenc` | — | taip | 1 srautui | tik `functions.enc-nvenc-<tune>` |
| `enc --encoder x264`, `gen`, `gfxgen`, `logo`, `composite` | — | — | — | tik `functions.<žymė>` (`enc-x264-<preset>`, `composite-k<K>`, …); nėra — netikrinama, viena eilutė stderr |
| `verify`, bet kuri `--impl cpu` | netaikoma | | | |

Kai VRAM išmatuotas, proceso pastovi dalis (`vram_offset_bytes`) rezervuojama pirma, tada `vram_per_flow_bytes` srautui.
VRAM lyginamas su visa GPU atmintimi (`gpu.vram_bytes` faile arba `cudaDeviceProp.totalGlobalMem`), ne su laisva.

**Knyga:** `<domenas>.vramflow/adm-<pid>-<hex>.adm` — failas procesui su `flock(LOCK_EX)`; skaitymas, sprendimas ir
įrašymas — po bendru `admission.lock`. Užrakinamas svetimas failas = miręs procesas: pašalinamas, nebeskaičiuojamas.
Skaičiuojamos tik tos pačios GPU (UUID) rezervacijos.

**Priėmimas iš eilės:** `enforce` — pirmas netilpęs srautas ir visi po jo atmetami (jų flow'ai domene nesukuriami); nė
vieno — exit `5` iškart (su santrauka), dalis — priimtieji veikia, pabaigoje exit `5` (klaidos kodas `rc ≠ 0` ir sugedęs
vienetas — viršesni, praradimas `4` — ne). Pranešime: variklis, prašoma, užimta, leidžiama. `ingest` priima po zondavimo
(formatas žinomas) ir prieš kuriant flow'us; kitos funkcijos — `main_func` prieš subkomandą.

## Sprendimai (šio etapo)

1. **Numatyta `--admission auto`** (plano sprendimas K13: `enforce`, kai yra talpos failas; `warn`, kai jo nėra; `off` — tik
   rankiniams bandymams). Su tinkamu talpos failu numatytoji veikia kaip `enforce` — eteryje esantys srautai saugomi iškart,
   kai GPU išmatuota; be failo — kaip `warn`: **viena** įspėjimo eilutė paleidžiant (user story 7 kriterijus siūlė kas 10 s —
   supaprastinta), viršijus aiškų `warn` — eilutė srautui, rezervacija vis tiek įrašoma, kad kiti procesai matytų tikrą
   užimtumą. Pirma šio etapo versija numatytąja laikė `warn` visada — tai neatitiko K13 ir pataisyta tą pačią dieną
   (testai: `admission: numatytoji auto …`, `overload_default[capacity]`). Talpos matavimo skriptai (`tests/perf/capacity.py`,
   `ingest_capacity.py`) funkcijas leidžia su `--admission off`, kad permatavimo neribotų senas talpos failas.
2. **Exit `5`** — laisvas `vramflow-func` (7 — libav sargas; `vramflow-verify` 5 = konfliktas — kitas binaras).
   `enforce` be tinkamo talpos failo — exit `1` (konfigūracija).
3. NVENC sesijos — atskiras resursas (`engine == "nvenc_sessions"`), `bound: lower_bound` reikšmės naudojamos kaip riba
   su įspėjimu, kita tvarkyklė (NVML) — įspėjimas su abiem versijomis, kita GPU (UUID) — failas nenaudojamas.
4. Talpos failo vieta: `--capacity-file` (failas arba katalogas su `capacity-<UUID>.json`), numatyta — `VRAMFLOW_CAPACITY`.
5. JSON — mažas skaitytuvas `admission.cpp` viduje (projekte bendro JSON skaitytuvo nėra; nauja priklausomybė netinka).

## Testų skaičiai

Vieneto (`mxlv-admission-unit`, 22 atvejai): 16-tas 1080p25 srautas (829,44) priimamas, 17-tas (881,28) atmetamas
`nvdec`, riba 850; 6 GiB / 135 MiB — 38-tas (5130 MiB) priimamas, 39-tas atmetamas `vram`, riba 5222,4 MiB; sesijos 12 —
13-a atmetama `nvenc_sessions`, nors Mpix/s leistų; talpa 8 srautų, procesai 5 + 5 → antras gauna 1; `kill -9` →
rezervacija nebeskaičiuojama (< 100 ms; 30 kartų iš eilės be klaidų); 8 procesai vienu metu, vieta 4 → priimti lygiai 4;
P8 failas (RTX 4070 Laptop) kaip fiktūra. `vramflow-glue-tests [exit]`: dalis atmesta → 5 abiem režimais.

Funkciniai (`tests/functional/overload.py`):

- `overload_enforce_synthetic` — dirbtinis talpos failas (NVDEC 270 · 0,85 = 229,5 Mpix/s), 8 `ingest` procesai po 1 UDP
  šaltinį: pirmi 4 — 60 s, exit 0, `late_frames` 0, `skip_to_key` 0, `missing_frames` 0, po 1709 dekoduotų kadrų; 5–8 —
  exit 5, „nvdec: prašoma 51,84, užimta 207,36, leidžiama 229,50 Mpix/s“; domene 8 flow'ai (4 video + 4 garso).
- `overload_no_capacity` — `enforce` be failo → exit 1 ir „talpa neišmatuota, paleiskite 70-capacity“; numatytoji (`auto`) be
  failo → `warn`, 1 eilutė.
- `overload_default[capacity]` — `dec --flows 3` be `--admission` parinkties, talpa 1 srautui: priimtas 1, atmesti 2, exit 5
  (numatytoji `auto` su talpos failu = `enforce`).
- `overload_partial` — `dec --flows 3`, talpa 1 srautui: domene 1 flow, exit 5; `enc --flows 3`, sesijų riba 2: „flow 2:
  atmestas — nvenc_sessions: prašoma 1, užimta 2, leidžiama 2 sesijų“.

Švarus build (PR 5 commit'as) — 0 įspėjimų; `ctest` 481 / 482 (nepavyko `gap_policy: … < 1 ms`: 1,161 ms, žinomas laiko
testas; 3 pakartojimai iš eilės — PASS). Visas funkcinis rinkinys vieną kartą: 318 PASS, 32 INFO, 2 FAIL —
`fiso_stall[G:middle]` ir `fsplit[composite-G]`, abu iš žinomų laiko testų sąrašo (jų tikrinamas kodas nekeistas).

Pakartota atskirai tą pačią dieną. Ant `8bd366d`: švarus build — 0 įspėjimų, `ctest` 482 / 482, visas funkcinis rinkinys
(352 testai, 2 h 25 min) — 1 FAIL: `fiso_stall[G:first]` (spragos commit'as ties 25 034 µs, riba 25 000 µs; žinomas laiko
testas). Po numatytosios `auto` pataisos: švarus build — 0 įspėjimų, `ctest` 483 / 483, `overload_*` ir `fcomp1` — 9 / 9
PASS (`overload_default[capacity]`: exit 5, `mode: enforce`, priimtas 1, atmesti 2); `capacity.py --func logo --n 2` su
`VRAMFLOW_CAPACITY` aplinkoje — PASS, priėmimo eilučių harness'o procesų stderr — 0.

Plano `skip_events == 0` dar netikrinamas — skaitiklis atsiras su termino vartais; vietoje jo — esamas `skip_to_key`.

## Kas nepatikrinta

- Su tikru (ne dirbtiniu) talpos failu prie kelio (`overload_enforce_measured`: `2·N_k` srautų → priimta `floor(0,85·N_k)`,
  visiems `late == 0`) — kitame etape kartu su termino vartais.
- VRAM įverčio tikslumas `ingest` (nėra `functions.ingest` matavimo; P8 `dec` išmatavo 182,2 MiB srautui, `ingest` įvertis 1080i50 su
  numatytomis parinktimis — 29 NV12 paviršiai + 11 v210 grain'ų ≈ 116 MiB, be proceso dalies). Kol nėra matavimo, VRAM riba `ingest` — apytikslė.
- Kiti GPU naudotojai ir kiti domenai ant tos pačios GPU knygoje nematomi; daug GPU — tik `--device-uuid` pasirinkta.
- Tvarkyklės versijos neatitikimas patikrintas tik vieneto lygiu (fiktūra 596.49 prieš kitą tekstą); tikroje mašinoje
  versijos sutapo.
