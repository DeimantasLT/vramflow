# Gedimų izoliacija — „prieš“ skaičiai (V0b)

Data: 2026-11-12. Kodas: `main` prieš tick variklio perkėlimą (planas `docs/planai/2026-11-12-tick-variklis.md` §9 V0b, §10).
Testai: `tests/functional/isolation.py` (`fiso_kill`, `fiso_stall`, `fiso_composite`, `fiso_enc`), būsena `INFO`.
Aplinka: WSL2 Ubuntu 24.04, RTX 4070 Laptop, 12 loginių branduolių, sintetinė laikmena (`gen`/`gfxgen`), 1080i50 (50 laukų/s).

## Metodika

- N = 8 flow'ai. Vienas aukštupio rašytojas (`gen` B arba `gfxgen` B) turi tik paskutinį flow'ą (`--flow-base 7`), kitas procesas — likusius
  7 („sveiki“). `kill -9` arba `SIGSTOP` (5 s, paskui `SIGCONT`) — 10-ą sekundę po funkcijos starto.
- Kiekvienas zondas paleidžiamas du kartus: **kontrolinis** (be gedimo, „prieš“) ir **su gedimu** („po“); kitaip tokie patys.
  Funkcija veikia 24 s, `--warmup 9`, todėl `out_lateness` histograma abiem atvejais apima nuo 9-os sekundės (1 s iki gedimo ir viskas po jo).
  Skaitikliai (`late_fields`, `late_skips`, …) kaupiami visam paleidimui; kontroliniame jie 0.
- `verify --every 5` (20 s) skaito visus N išėjimus. Lentelėse „sveiki“ = 7 flow'ų suma, „paveiktas“ = 8-as flow'as.
- `out_lateness` (enc — `frame_lateness`): p50 / p99 / maks., ms. Histograma logaritminė (kvantas ~2 %).
- Du pakartojimai (pirma reikšmė / antra reikšmė po įstrižainės), kai skiriasi. Aplinkos jitter'is: kontrolinio maks. kinta 8,3–14,2 ms.
- G = `gpu` + `cuda-linear` (enc — NVENC), C = `cpu` + `host` (enc — x264 ultrafast).

## fiso_kill — `gen` B nužudomas (`logo` N = 8)

| | sveiki: `late_fields` / `late_skips` / `lapped` | `out_lateness` prieš | `out_lateness` po | paveiktas `logo`: `missing_fields` | `verify` paveiktam: present / missing / invalid (prieš → po) |
|---|---|---|---|---|---|
| C | 0 / 0 / 0 | 6,5 / 9,6 / 12,6 (6,4 / 7,9 / 8,3) | 6,2 / 7,8 / 14,3 (6,0 / 7,3 / 13,8) | 706 (707), `detached` 1 | 947 / 0 / 0 → 437 / 510 / 0 |
| G | 0 / 0 / 0 | 5,3 / 10,1 / 11,3 (7,3 / 11,1 / 12,8) | 6,4 / 8,8 / 15,4 (6,4 / 8,8 / 12,6) | 713 (715), `detached` 1 | 948 / 0 / 0 → 420 / 527 / 0 |

`verify` sveikiems: `missing` 0, `invalid` 0, `mismatch_fields` 0. `logo` ir `verify` baigiasi exit 4 (paveikto flow'o spraga).

## fiso_stall — `gen` B sustabdomas 5 s (SIGSTOP → SIGCONT), procesas gyvas

| | sveiki: `late_fields` / `late_skips` / `lapped` | `out_lateness` prieš | `out_lateness` po | paveiktas `logo`: `late_fields` / `late_skips` | `verify` sveikiems: invalid; paveiktam: present / missing / invalid |
|---|---|---|---|---|---|
| C | 0 / **875** / 0 | 6,5 / 9,6 / 12,6 (6,4 / 7,9 / 8,3) | 6,4 / **32,2** / 32,3 (6,4 / 32,2 / 32,5) | 125 / 125 | 875; 697 / 247 / 3 |
| G | 0 / **875** / 0 | 5,3 / 10,1 / 11,3 (7,3 / 11,1 / 12,8) | 7,5 / **36,2** / 38,2 (7,7 / 36,2 / 36,8) | 125 / 125 | 875; 697 / 247 / 3 |

875 = 7 flow'ų × 125 praleistų laukų: 5 s × 50 Hz = 250 laukų, praleidžiamas kas antras (**50 %**), visiems sveikiems, ne tik paveiktam. Abu pakartojimai
sutampa tiksliai. `verify` sveikuose flow'uose mato 875 `invalid` grain'ų, `missing` 0, `mismatch_fields` 0. Po `SIGCONT` papildomų praradimų nėra
(sveikų `late_skips` tiksliai 875). Tikų skaičius histogramoje: 626 vietoje 751.

## fiso_composite — `gfxgen` B (vieno išėjimo sluoksnio šaltinis) nužudomas, `composite` N = 8, K = 1

| | sveiki: `late_fields` / `late_skips` / `lapped` | `out_lateness` prieš | `out_lateness` po | paveiktas `composite`: `missing_fields` | `verify` paveiktam: present / missing / invalid (prieš → po) |
|---|---|---|---|---|---|
| C | 0 / 0 / 0 | 6,5 / 7,9 / 8,7 (6,4 / 8,3 / 9,5) | 6,4 / 12,2 / 14,1 (6,2 / 8,8 / 12,8) | 704 (706), `flow_invalid` 1 | 948 / 0 / 0 → 443 / 504 / 0 |
| G | 0 / 0 / 0 | 6,8 / 12,5 / 14,2 (6,6 / 12,2 / 13,5) | 7,7 / 11,7 / 14,8 (7,7 / 12,5 / 18,3) | 715 (711), `flow_invalid` 1 | 948 / 0 / 0 → 422 / 525 / 0 |

## fiso_enc — `gen` B nužudomas (`enc` N = 8, `ts_sink` 8 prievadai)

| | sveiki: `frames_encoded` / `late_frames` / `missing_frames` | `frame_lateness` prieš | `frame_lateness` po | paveiktas: `missing_frames` / `frames_encoded` | TS sveikiems: min. video PES / CC / sync klaidos |
|---|---|---|---|---|---|
| C | 4200 (7 × 600) / 0 / 0 | 8,8 / 10,4 / 11,1 (8,8 / 10,4 / 10,7) | 8,2 / 10,1 / 11,0 (8,2 / 9,8 / 10,3) | 390 / 210 | 600 / 0 / 0 |
| G | 4200 / 0 / 0 | 1,9 / 2,7 / 3,2 (1,9 / 2,7 / 4,0) | 1,5 / 2,7 / 2,8 (1,5 / 2,7 / 3,0) | 394 (395) / 206 (205) | 600 / 0 / 0 |

## Išvada: kas šiandien nutinka sveikiems flow'ams

Kai aukštupio rašytojas **miršta** (`kill -9`), kiti N − 1 flow'ai to beveik nejaučia: `late_fields`, `late_skips`, `lapped`, `missing` ir `invalid` visuose
keturiuose zonduose ir abiejuose variantuose lygūs 0, `enc` sveikų srautų kadrų skaičius pilnas (600 per srautą), TS be CC/sync klaidų, o `out_lateness`
p99 po gedimo neviršija p99 prieš daugiau nei triukšmo ribose (kontrolinio maks. tarp paleidimų svyruoja 8,3–14,2 ms; maks. po gedimo — nuo −0,2 iki +5,5 ms
nuo kontrolinio, p99 — nuo −2,3 iki +4,3 ms). Plane užrašyto „≈ +10 ms pirmame tick'e po `kill -9`“ [iš kodo, nevykdyta] **neatkartojome**: funkcijos
mirusį įėjimą aptinka prieš laukimą (`InputSlot::state() != Attached`, `poll_inputs`) ir laukia tik gyvų. Kai rašytojas **užstringa, bet gyvas** (SIGSTOP),
kiekvieną tick'ą funkcija laukia jo iki `--deadline-ms` (30 ms) — ilgiau už 20 ms periodą, todėl ciklas atsilieka ir praleidžia kas antrą lauką **visiems**
flow'ams: 875 = 7 × 125 sveikų lauko praleidimų (50 % per 5 s), `out_lateness` p99 šoka nuo ~10 iki 32–36 ms, o `verify` sveikuose flow'uose mato 875
`invalid` grain'ų. Praradimai baigiasi su `SIGCONT`. Tai ir yra „prieš“ stulpelis §10: `fiso_stall` — 875 (tikslas 0); `fiso_kill` ir `fiso_composite`
— jau 0 (V10/V11 jų neturi pabloginti; p99 po gedimo turi likti ≤ p99 prieš + Wr + 1 ms).

Atkartojimas: `functional.py <out> --only fiso_kill,fiso_stall,fiso_composite,fiso_enc` (`FISO_VARIANTS=C,G`, `FISO_N=8`); struktūruoti skaičiai —
`functional.json` laukuose `before` / `after`.
