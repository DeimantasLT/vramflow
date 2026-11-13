# Gedimų izoliacija — „po“ (V10: `logo` bendrame tick variklyje, izoliacija įjungta)

**Data:** 2026-11-12. **Kodas:** `65497fd` (V10) + priskyrimo vartai (šis įrašas). **Aplinka:** ta pati kaip
„prieš“ (`docs/rezultatai/2026-11-12-izoliacija-pries.md`): WSL2, viena GPU, N = 8 flow'ų, `logo` 27 s (`--warmup 9`),
gedimas ties 10 s, SIGSTOP 5 s. Žalių duomenų vieta — `functional.json` kiekvieno paleidimo kataloge (`before` / `after`).
Variklio ribos 50 Hz: ro = 0,5 ms, Wr = 5 ms, D = 30 ms (`--deadline-ms`; B = 17,5, C = 30 ms), M = 3, Late histerezė 8.

## fiso_kill / fiso_stall — sveiki 7 flow'ai

| | sveiki `late_fields` / `late_skips` / `lapped` PRIEŠ | PO | `out_lateness` p99 be gedimo → su gedimu PRIEŠ | PO |
|---|---|---|---|---|
| kill C | 0 / 0 / 0 | 0 / 0 / 0 | 9,6 → 7,8 ms | 7,7 → 7,4 ms |
| kill G | 0 / 0 / 0 | 0 / 0 / 0 | 10,1 → 8,8 ms | 10,6 → 9,0 ms |
| stall C | 0 / **875** / 0 | 0 / 0 / 0 | 9,6 → **32,2** ms | 7,7 → 7,4 … 7,7 ms |
| stall G | 0 / **875** / 0 | 0 / 0 / 0 * | 10,1 → **36,2** ms | 10,6 → 10,9 … 11,1 ms |

\* žr. „Aukštupio sutrikimas fiso_stall[G]“ — 2 iš 6 paleidimų sveiki flow'ai prarado po 1–2 indeksus, kuriuos pats
sveikas aukštupis (`gen` A) pateikė per vėlai arba praleido.

Paveiktas flow'as: `kill` — `missing_fields` 856 / 862, `detached` 1; `stall` — `late_fields` 3, `missing_fields`
247, `stalls` 1; po SIGCONT atskiras `verify` (10 s, `--every 1`) — 447 / 448 imčių, `missing` = `invalid` = `mismatch` = 0.
Sveikiems `verify`: `missing` = `invalid` = `mismatch_fields` = 0 (išskyrus aukštupio atvejus žemiau).

## `logo` `out_lateness` prieš ir po migracijos (be gedimo, N = 8, p50 / p99 ms)

| | prieš (senas ciklas) | V9 (variklis, legacy ribos) | V10 (izoliacija) |
|---|---|---|---|
| C | 6,5 / 9,6 | 6,36 / 7,93 | 6,10 / 7,67 |
| G | 5,3 / 10,1 | 5,31 / 10,62 | 5,31 … 5,96 / 10,6 … 11,9 |

## Vartų taisyklė (priskyrimas, ne atlaisvinimas)

Tikrinama savybė: funkcija sveikiems flow'ams nepriduria **savo** praradimų. Sveiko aukštupio (`gen` A) santraukoje
yra `late_skips` ir naujas įvykio skaitiklis `late_commits` (commit'ai vėlesni nei `--deadline-ms` po T(i); `gen`
stderr — eilutė su indeksu ir TAI laiku). Kiekvienam sveikam flow'ui: funkcijos praradimai (`late_fields`,
`late_skips`, `lapped`, `too_late`, `in_invalid`, `read_errors`, `missing_fields`) ir `verify` `missing + invalid`
≤ `late_skips + late_commits` (be aukštupio sutrikimų — 0, kaip anksčiau); `mismatch_fields` == 0 visada; p99 riba ir
paveikto flow'o tvirtinimai nepakito. Aukštupio skaičiai — atskira INFO eilutė (`fiso_*[X]:upstream`).

Pakartojimai po taisyklės: fiso_kill C / G, fiso_stall C / G — 1 / 1 kiekvienas (aukštupis be sutrikimų; didžiausias
`gen` A commit'o vėlavimas su gedimu 4,6 … 12,5 ms).

## Aukštupio sutrikimas fiso_stall[G]

Ką matėme: 3 iš 7 fiso_stall[G] paleidimų (2 mano, 1 koordinatoriaus) sveikas `gen` A turėjo vieną tick'ą su commit'o
vėlavimu 36,7 / 45,0 / 47 ms (be gedimo — ≤ 9 ms), kartais ir `late_skips` = 1. Indeksas prarastas aukštupyje — `logo`
jo atkurti negali (D = 30 ms). Kai sutrikimas sutapo, ir `logo` `proc_ms` maks. buvo 50 ms (ta pati GPU). C variante
(be CUDA) — nė karto.

Priežasties bandymas (`tests/perf/cross_stall.py`, po 3 pakartojimus, tik du gamintojai, be `logo`): A / B variantai
G/G, C/G, H/G, G/C (B sustabdomas 5 s) ir G/G be gedimo — **visuose 15 paleidimų `late_commits` = 0**, maks. vėlavimas
≤ 14,8 ms. Vien „sustabdytas CUDA procesas vėlina kitą CUDA procesą“ hipotezė šiuo bandymu **nepatvirtinta** (0 iš 9
sustabdymų su CUDA B). Sutrikimas pasitaiko tik pilname scenarijuje, kuriame yra trečias CUDA procesas (`logo` G) —
dažnis ≈ 1 iš 3.

Kas neįrodyta: priežastis (WSL2 GPU paravirtualizacija ir kelių kontekstų laiko dalijimas, kai vienas kontekstas
sustabdytas, yra tik hipotezė); vėlaus tick'o laikas nuo SIGSTOP / SIGCONT dabar registruojamas (`late_ticks` su `vs_stop_ms`,
`vs_cont_ms`), bet nuo jo įjungimo sutrikimas dar nepasikartojo. Patikrinti natyviame Linux (ne WSL2) — ar dingsta.

## Atnaujinimas 2026-11-13: enc C vartai talpos ribose (T1) ir favsync_late_audio[C] (T2)

### T1 — fiso_enc[C] N = 8 → 6

Ankstesnėje patikroje fiso_enc[C:*] krisdavo ≈ 1 iš 3 pilnų paleidimų: visų sveikų srautų maks. vėlavimas gedimo sekundę
25,85–25,99 ms prieš ribą 25,28 ms (kontrolė 11,31, Δup 7,47), be jokių praradimų. x264 atsarga šioje mašinoje
(`tests/perf/enc_headroom.py`: ta pati grandinė kaip fiso_enc, be gedimo, x264 ultrafast, 1080i50 + garsas, 2 pakartojimai):

| N | sistemos CPU, % iš 12 loginių | enc CPU, % vieno | frame_lateness p50 / p99 / max, ms | sekundžių maks. mediana, ms |
|---|---|---|---|---|
| 2 | 7,6 … 8,2 | 76 | 5,0 … 5,7 / 6,8 … 7,5 / 7,3 … 8,2 | 6,3 … 7,4 |
| 4 | 17,4 | 176 | 5,8 … 6,4 / 8,5 … 9,0 / 10,2 … 11,3 | 8,0 … 8,4 |
| 6 | 26,7 … 27,1 | 280 | 7,4 … 7,7 / 9,3 / 9,6 … 12,6 | 8,8 … 9,0 |
| 8 | 41,7 … 42,0 | 396 … 400 | 8,8 / 10,6 … 10,9 / 12,9 … 13,1 | 9,9 … 10,0 |

Praradimų 0 visur. WSL2 rodo 12 loginių CPU = 6 branduoliai × 2 SMT: N = 8 užima ≈ 5,0 loginio CPU (5 iš 6 fizinių
branduolių, jei kiekviena gija — atskirame), N = 6 — ≈ 3,3. Pasirinkta **N = 6** (`isolation.py` `ENC_C_N`; kiti vartai — 8).

Vartų atsarga — mažiausias (riba − sveiko srauto vėlavimas) per gedimo lango sekundes:

| N | paleidimų (× 3 vietos) | PASS | mažiausia atsarga | mediana |
|---|---|---|---|---|
| 6 | 10 × 3 = 30 | **30 / 30** | 3,86 ms | 5,78 ms |
| 8 | 3 × 3 = 9 | 8 / 9 (1 — visi sveiki `frames_encoded` 646 < 649) | 1,87 ms | 6,03 ms |

fiso_teeth[enc-C] prie N = 6: injekcija 30 ms — vartai FAIL 16 / 16, 10 ms — FAIL 16 / 16 (lango p99 taisyklė 16 / 16, maks.
vėlavimo taisyklė 13 / 16), be injekcijos — PASS 16 / 16. Meta-testas (reikalauja, kad 10 ms pagautų ABI taisyklės) —
13 / 16 (8 / 10 + 5 / 6). Kodėl maks. vėlavimo taisyklė kartais nesuveikia: 10 ms injekcija pavėlina ir sveiką aukštupį — `gen` (C, turinys
CPU) commit'o vėlavimas pakyla 2–5 ms KIEKVIENĄ gedimo lango sekundę (kontrolėje maks. 6,6 ms; išmatuota 6 paleidimais);
hipotezė (nepatikrinta): pavėlintas enc darbas (weave, x264) persislenka ant `gen` kito lauko generavimo CPU laiko. Δup šį funkcijos sukeltą aukštupio vėlavimą priskiria aukštupiui,
ir atsarga lieka 0 … 1,5 ms. Tai žinomas vartų „Ko vartai NEAPTINKA“ atvejis (Δup „paaiškina“), ne N pasekmė; taisymas —
gamintojų ir tikrinamos funkcijos CPU rinkinių atskyrimas harness'e (neįgyvendinta).
