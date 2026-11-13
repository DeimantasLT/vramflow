# Sąveika su MXL įrankiais (P5b, K4) — matavimai

**Data:** 2026-11-13. **Mašina:** WSL2 Ubuntu 24.04, RTX 4070 Laptop, viena GPU. **MXL:** `/root/src/mxl-pr653` (PR #653),
įrankiai `install/Linux-Clang-Release/bin`. **VRAMFlow:** `main` po briaunos laiko sutarties (V24).
**Testai:** `tests/functional/interop.py` (`interop_info`, `interop_sink_host`, `interop_sink_cuda`, `interop_foreign_writer`),
papildomi matavimai — vienkartiniai skriptai (scratchpad `explore.py`, `explore2.py`; neįtraukti į repozitoriją).

## Kaip matuota

- GStreamer sink'ai paleisti po **Xvfb** (`:9x`, 1920×1080×24, juodas fonas, be langų tvarkyklės). Ką rodo ekranas —
  `xwd -root` kopijos 5 Hz: `move` — kadras pasikeitė nuo ankstesnės kopijos, `still` — nepasikeitė, `black` — nepasikeitė
  ir vidutinis šviesis < 24. Langas — 3 s prieš gedimą, 3 s gedimo, 4 s po gamintojo grįžimo (2 s po paleidimo).
- Gedimas: aukštupio `gen` `kill -9`, po 3 s paleidžiamas iš naujo; tarp jo ir sink'o — `logo` su `--on-loss X` (jis rašo
  kiekvieną indeksą). „Be VRAMFlow spragų“ — sink'as skaito `gen` tiesiai (rašytojo nebėra) arba `logo --on-loss leave`
  (rašytojas gyvas, bet indeksų nerašo).
- Vėlinimas — sink'o paties eilutės `Video latency initialized / increase detected` (didžiausia reikšmė; `mxl-gst-sink`
  skaito `--read-delay` = 40 ms atgal). `mxl-gst-cuda-sink` vėlinimo nespausdina.
- Formatai: `gen --format` 1080i50, 1080p25, 1080p50, 720p50, 1080i59.94; v210a — `gfxgen` (`--layer-geom full`).

## Įdiegti paketai (tik testų įrankiai, ne build priklausomybės)

| Paketas | Versija | Kam |
|---|---|---|
| `xvfb` | 2:21.1.12-1ubuntu1.9 | virtualus X ekranas sink'ams |
| `x11-apps` | 7.7+11build3 | `xwd` ekrano kopijoms |
| `xserver-common` | 2:21.1.12-1ubuntu1.9 | `xvfb` priklausomybė |
| `x11-xkb-utils` | 7.7+8build2 | `xvfb` priklausomybė |
| `libxfont2` | 1:2.0.6-1+deb13u1build0.24.04.2 | `xvfb` priklausomybė |
| `libfontenc1` | 1:1.1.8-1build1 | `xvfb` priklausomybė |
| `libxcb-damage0` | 1.15-1ubuntu2 | `x11-apps` priklausomybė |

GStreamer 1.24 (`gstreamer1.0-plugins-base/good/bad/ugly`, `gstreamer1.0-x`, `gstreamer1.0-gl`) jau buvo įdiegtas — nieko
nediegta. `xauth` (1:1.1.2-1build1) jau buvo. Diegta `apt-get install --no-install-recommends xvfb xauth x11-apps`.

## Rezultatai

### (1) `mxl-info`

| Flow | host | cuda-linear |
|---|---|---|
| video 1080i50, 1080p25, 1080p50, 720p50, 1080i59.94 (`gen`) | exit 0, `Format: Video`, `Active: true`, head > 0 | tas pats |
| garsas 48 kHz float32 (`gen --audio 1`) | exit 0, `Format: Audio` | tas pats |
| v210a (`gfxgen`) | exit 0, `Format: Video`, `Active: true` | tas pats |

MXL priima flow aprašus su VRAMFlow žymomis (`urn:x-vramflow:tag:commit-bound-ns`, `…content-delay-ns`): flow'ai sukuriami,
`flow_def.json` jas išsaugo; `mxl-info` žymų nespausdina (ir nepraneša klaidos).

### (2), (3) `mxl-gst-sink` (host) ir `mxl-gst-cuda-sink` (cuda-linear v210), 1080p50

| Situacija | `mxl-gst-sink` (host) | `mxl-gst-cuda-sink` (cuda-linear) |
|---|---|---|
| `gen` tiesiai, be gedimo | rodo (29/30 kopijų juda), vėlinimas ≤ 41–42 ms | rodo (29/30) |
| `logo` išėjimas, be gedimo | rodo, ≤ 41–50 ms | rodo |
| gedimas, `--on-loss invalid` (numatyta: INVALID + juodas turinys) | **neužstringa**: INVALID grain'us praleidžia, ekrane **sustingęs paskutinis geras kadras** (15/15 `still`), 0 „TOO EARLY“ eilučių; gamintojui grįžus — iškart juda | tas pats: sustingęs paskutinis kadras, grįžus juda |
| gedimas, `--on-loss freeze` (`--hold-ms 1000`) | ~1 s sustingęs (5 kopijos), tada juoda; grįžus juda | tas pats |
| gedimas, `--on-loss black` | juoda (13–14/15) iškart; grįžus juda | juoda (15/15); grįžus juda |
| `--start immediate` (sink paleistas, kol `logo` rašo tik paleidimo INVALID) | ekranas juodas (langas neatidarytas — nė vieno galiojančio kadro), `gen` paleidus — rodo | tas pats |
| **be VRAMFlow spragų**: rašytojas nužudytas (`gen` tiesiai) | rodo paskutinį kadrą, bet **sukasi tuščiai**: per 3 s — ~1,1 mln. `TOO EARLY` WARN eilučių (99 MB žurnalo), grįžus `gen` — `Reconnected`, rodo | **baigia darbą**: `mxlFlowReaderGetGrainSlot failed with status 11` (perkurtas flow) — sink'as negrįžta |
| **be VRAMFlow spragų**: rašytojas gyvas, indeksų nerašo (`logo --on-loss leave`) | tas pats tuščias sukimasis (~1,15 mln. `TOO EARLY` eilučių / 3 s), grįžus — `TOO LATE` → persilygina, rodo | stovi ant to paties indekso (kas 1 ms bando), grįžus — rodo |

Pastabos: `mxl-gst-sink` vėlinimas ≈ jo `--read-delay` (40 ms) + 1–10 ms. Kadangi jis skaito fiksuotu 40 ms atsilikimu, flow'as,
kurio skelbiama riba didesnė (`commit-bound-ns` > 40 ms — pvz. penkta VRAMFlow video pakopa po šaltinio: 8 + 4 · 8,5 = 42 ms,
**apskaičiuota, nematuota**), jam reikalautų didesnio `--read-delay`.

### (4) Atvirkštinė kryptis: `mxl-gst-testsrc` → `logo` → sink

- `mxl-gst-testsrc` **interlaced nepalaiko** („This application does not support interlaced flows“) — naudotas 1080p50.
- **Rastas VRAMFlow defektas (pataisytas, `cf3abc5`):** testsrc (picojson) perrašo flow aprašą su JSON pabėgimu
  `"media_type":"video\/v210"`; VRAMFlow jį atmesdavo („nepalaikomas media_type video\/v210“) ir niekada neprisijungdavo.
  Dabar `json_str` atkoduoja pabėgimus.
- Svetimas flow be žymų: `logo --input-expect-ms 12` → fazė E = 12 ms (`expect_us` 12000, `inputs_untagged` 1), skelbiama riba
  20,5 ms, `logo` 0 praradimų (`fields_ok` 800–1000), `out_lateness` p50 13,0 / p99.9 13,7 ms; `mxl-gst-sink` jo išėjimą rodo
  (24/25 kopijų juda), ≤ 47 ms.
- testsrc pats perrašo grouphint žymą savąja (`mxl-gst-testsrc-group:video`).

### (5) Interlaced ir v210a

| Įrankis | interlaced v210 (1080i50) | v210a progresyvus (1080p50) |
|---|---|---|
| `mxl-gst-sink` | atsisako: „does not support interlaced flows“, exit 0 | veikia ir rodo (`gfxgen pace` — statinis kadras, 15/15 `still`; vėlinimas ≤ 41 ms); ar naudoja key plokštumą — nevertinta |
| `mxl-gst-cuda-sink` | **nepatikrina** `interlace_mode`: paleidžia `width=1920,height=1080,framerate=50/1` konvejerį lauko grain'ams ir veikia (vaizdo turinys nevertintas; laukas — ne kadras, todėl teisingas būti negali) | atsisako: „displays video/v210 only“, exit 1 |
| `mxl-video-probe` | host — OK (100 grain'ų, `grain_errors=0`, 50 fps, vėlinimas 2,3–5,1 ms); cuda-linear — `MXL_ERR_UNSUPPORTED_OPERATION` (įrankis neskaito GPU atminties) | host — OK; cuda-linear — `UNSUPPORTED_OPERATION` |
| `mxl-info` | OK | OK |

## Ar INVALID grain'ai saugūs MXL įrankiams?

**Taip.** Abu etaloniniai sink'ai INVALID grain'ą praleidžia ir skaito kitą indeksą: neužstringa, nesisuka tuščiai (0 „TOO
EARLY“ eilučių), gamintojui grįžus rodo iškart. Priešingai — **be** INVALID (nerašomi indeksai, `leave`) `mxl-gst-sink`
per 3 s išspausdina ~1,1 mln. WARN eilučių (100 % CPU vienam branduoliui), o `mxl-gst-cuda-sink` perkurto flow'o atveju
baigia darbą. Taigi K3 „commit'inti kiekvieną indeksą“ yra būtina, o numatytoji `--on-loss invalid` **nereikalauja keitimo dėl
užstrigimo**.

Vienas dalykas autoriaus sprendimui (numatytoji nekeista): INVALID grain'o **juodo turinio** šie sink'ai nerodo — jie kartoja
paskutinį gerą kadrą, todėl operatorius ekrane mato „sustingusį“ vaizdą, ne „nėra signalo“. Jei matomas „nėra signalo“ (juoda)
reikalingas ir MXL įrankiams — tai `--on-loss black`; `freeze` elgiasi taip pat, kaip šie sink'ai su INVALID, tik juoda po
`--hold-ms`.

## Testai (stabilūs — įtraukti)

`interop_info` (12 atvejų), `interop_sink_host` ir `interop_sink_cuda` (po 6: gen, logo, outage/invalid|freeze|black,
start-immediate), `interop_foreign_writer`. Neįtraukta (nestabilu arba ne mūsų sutartis): tuščio sukimosi eilučių skaičius,
`mxl-gst-cuda-sink` interlaced vaizdo turinys, vėlinimo skaičiai (tik užrašomi). Jei trūksta įrankio, Xvfb ar `xwd` — SKIP su
priežastimi.
