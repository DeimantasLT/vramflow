# Bendras „tick“ variklis (VRAMFlow) — įgyvendinimo planas

> **Įgyvendinimo būsena (2026-11-13):** 0–3 etapai ir V19, V21 — įgyvendinti; **V17** (spragų politika K3: `--on-loss`,
> `--hold-ms`, ritmo gija iki pirmo `submit` — INVALID) — įgyvendinta; **V18** (`--start immediate` — `logo`; `composite` ir
> `enc` lieka wait-all: `composite` išėjimo rastras priklauso nuo sluoksnių įėjimų, `enc` MXL išėjimų neturi) — įgyvendinta;
> **V20** (`--deadline-ms` 0 = auto `3·P/4`; `gen` „vėlaus commit'o“ slenkstis — tas pats; WSL2 kampanijos — 30 aiškiai)
> — įgyvendinta. Žemiau — pirminis planas.
>
> **Būsena (pirminė):** planas, **laukia autoriaus patvirtinimo**. Kodas nekeistas, niekas nekompiliuota ir nepaleista (tuo metu kitas
> agentas taisė kompiliatoriaus įspėjimus; GPU neliesta). Teiginiai apie VRAMFlow kodą — iš skaitymo `git show HEAD:<kelias>`,
> HEAD = `98b92c1`. Teiginiai apie MXL — iš `/root/src/mxl-pr653` @ `6dddc91d92f5` ir kiekvienas pažymėtas
> **[patikrinta: failas:eilutė]** arba **[prielaida]**. Skaičiai, kurių niekas nematavo, pažymėti **[iš kodo, nevykdyta]**.
> **Vykdytojams:** superpowers:executing-plans arba subagent-driven-development; kiekvienas PR (V0a…V24) — atskira šaka, pirma
> testai (TDD), commit'ai lokaliai, **jokio `git push`**. Elgseną keičiantys PR (4 etapas) nepradedami be autoriaus sprendimų §12.

**Šaltiniai:** auditas `docs/rezultatai/2026-11-11-kodo-auditas.md` (toliau **[A]**, radiniai **A1…A28**); bazės planas
`docs/planai/2026-11-11-production-grade.md` (**[B]**, darbai P1–P10, sprendimai K1–K17); tinklo reikalavimai ir planas
`docs/reikalavimai/2026-11-12-tinklas-ir-grupes.md` §7 (**[R]**) ir `docs/planai/2026-11-12-tinklas-ir-grupes.md` §4 E0 (**[P]**);
pirminė architektūra `docs/architektura/2026-11-10-gpu-media-funkcijos.md` §3 (**[F]**); laikas
`docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md` (**[L]**).

---

## 1. Tikslas

Vienas komponentas `mxlv::engine::TickEngine` pakeičia septynias išsiskyrusias ciklo kopijas (`gen`, `logo`, `composite`, `enc`,
`dec`, `gfxgen`, bibliotekos ritmo gija). Variklis valdo laikrodį, įėjimus, išėjimus ir politikas; funkcija lieka tik tuo, kuo ji
skiriasi nuo kitų — `process(batch)` ir keliais kabliukais. Variklyje vieną kartą įgyvendinama:

1. **Gedimų izoliacija tarp flow'ų** — paruošti flow'ai apdorojami ir commit'inami laiku; neparuošto įėjimo laukiama tik ribotą
   grain'o periodo dalį; užstrigęs įėjimas kitiems nebekainuoja nieko.
2. **Spragų politika K3** vienoje vietoje visiems rašytojams (ir bibliotekos ritmo gijai): kiekvienas indeksas commit'inamas.
3. **Įšvirkščiamas laikrodis** — būsenų automatas ir terminai tikrinami virtualiu laiku, be GPU ir be MXL.
4. **Statistikos kabliukas** (P7), **priėmimo kabliukas** (P4), **adresavimas pagal (domenas, flow UUID)**, **briaunos laiko
   sutartis** (A20) ir **tarnybos gyvavimo ciklas** (A7) — sąsajos dabar, kad tinklas, grupės ir NMOS jų nelaužytų.

Kas **nekeičiama** ([A] §3): kernel'iai ir etalonai, vienas batch'as visiems flow'ams, commit tik po vieno
`cudaStreamSynchronize` per paleidimą, zero-copy skaitymas su „lap“ patikra, `check_layout` ir `info.index == index` patikra,
viena gija valdo visus MXL handle'us, darbininkų gijos (`dec`, `enc`) MXL neliečia, K16 (vienas procesas — viena GPU),
taisyklė „išėjimo indeksas = įėjimo indeksas“ ([F] §3 (c)).

## 2. Ką tai išsprendžia

| Radinys | Kaip | Kuriame žingsnyje |
|---|---|---|
| **A1** nėra izoliacijos | Ribų laiko juosta A / B / C (§6 D2), kvorumas, pertraukiklis, gyvumo patikra ties A riba | V4, V5, V10 |
| **A3** septynios kopijos | Vienas `TickEngine`; funkcijos — `TickFunction`; CTest `arch_single_loop` neleidžia aštuntos | V8–V16 |
| **A7** testo, ne tarnybos gyvavimas | `--duration 0` = iki signalo; `--strict-exit`; tvarkingas sustabdymas | V19 |
| **A11** testai ant tikro laikrodžio | `Clock` sąsaja + `VirtualClock`; ritmo testai perkeliami į virtualų laiką | V1, V4–V6, V14b |
| **A12** tikrinama „0 praradimų“, ne elgsena po praradimo | `fiso_*` funkciniai testai, virtualaus laiko degradacijos testai | V0b, V4, V5, V10 |
| **A13** K3 neįgyvendintas | `OutputPort::fill_gap` + `GapState`; INVALID commit'inamas tą akimirką, kai aišku, kad turinio nebus | V3, V7, V17 |
| **A17** `dec` perpildymai priskiriami laikrodžiui pagal sumą | Variklio laikrodžio įvykių langas `TickContext::clock_event_recent()`; suminė nuolaida šalinama | V13 |
| **A18** 0,5 ms apklausa, bendras 30 ms terminas | Laukimas MXL futex'u ant vieno „ašinio“ įėjimo; ribos — periodo dalys | V4, V10, V20 |
| **A19** numanomas RT elgesys | Variklio gija — vienintelė RT gija: `--rt-priority`, `--cpu-affinity` vienoje vietoje (CUDA laukimo režimas lieka P8) | V19 |
| **A20** nėra briaunos laiko sutarties | `EdgeTiming{index_offset, expect_ns}`, skelbiamas vėlinimas, rinkinio lygiavimas, TAI preflight | sąsaja V4; įgyvendinimas V24 |
| A10 (128 ritmo gijos) | Bendras ritmo vykdytojas: viena gija vienam (dažnis, offset, GPU), viena kopijų eilė, vienas sync | V14 |
| A25 (vienas signalas), A27 (santykinis miegas) | `sigwait` cikle; `mxlSleepUntil` | V19, V16 |

Neapima (lieka savo darbuose): A2 (`enc` garso PTS — variklis duoda tik `end_unit(…, Gap)` kabliuką), A4, A5, A6, A8, A9, A14,
A15 (darbininkų gyvavimas; variklis duoda `fail_unit`), A16, A21–A24, A26, A28.

## 3. Kas yra šiandien (HEAD `98b92c1`)

### 3.1 Septyni ciklai

| Ciklas | Kur | Įėjimai | Išėjimai | Kas tik jo (lieka funkcijoje) |
|---|---|---|---|---|
| `gen` | `src/func/gen.cpp:78-155` | — | N video + N garso | `gen_fields`, garso sinusas |
| `logo` | `src/func/logo.cpp:163-281` | N × 1 | N | `logo_fields`, savitikros injekcija, H variantas |
| `composite` | `src/func/composite.cpp:210-376` | N × (bazė + K) | N | darbų sąrašas, `composite_batch` |
| `enc` | `src/func/enc.cpp:421-596` | N video (2 grain'ai kadrui) + N garso | — (darbininkų eilės) | weave, D2H, `Item` į eilę |
| `dec` | `src/func/dec.cpp:269-402` | — (darbininkų eilės) | N | atkūrimo buferis, split |
| `gfxgen` (manual) | `src/func/gfxgen.cpp:240-286` | — | N·K per `vfgfx_write` | turinio paruošimas, prefetch |
| ritmo gija | `src/gfx/gfx_writer.cpp:272-323` | — (paskutinis `submit`) | 1 kiekvienai gijai | paruošto lauko kopija |

Iš viso ≈ 773 ciklo eilutės. `verify` (`src/func/verify.cpp:369-…`) — stebėtojas su savo atsilikimu; **sąmoningai lieka
atskiras** (nepriklausomas testų orakulas: variklio klaida neturi pasislėpti tikrintuve).

Kopijos jau išsiskyrė (visa — [iš kodo]):

- Praleidimų apskaita: `gen` ir `gfxgen` — proceso lygio (`gen.cpp:152`, `gfxgen.cpp:283`); `logo`, `composite`, `dec` — kiekvienam
  flow'ui (`logo.cpp:278`); `enc` — kadrais su kitais vardais (`enc.cpp:593`); ritmo gija — savo `PacingStats`.
- Laikrodžio šuoliai: `ClockWatch` funkcijose; `dec` papildomai turi savo globalų langą ir **suminę** nuolaidą pabaigoje
  (`dec.cpp:56-62, 424-433`); ritmo gija — tik `clock_backward` per kabliukus.
- `early_wakes` ir diagnostinė eilutė apie nesulauktą įėjimą — tik `composite` (`composite.cpp:225, 270-279`).
- Re-attach skaitikliai: `composite` — `reopened` kiekvieną tick'ą (`:213-219`); `logo`, `enc` — `reattached` pabaigoje.
- Exit 4 sąrašai — keturi skirtingi (`logo.cpp:295-298`, `composite.cpp:387-390`, `enc.cpp:629-633`, `dec.cpp:442-447`).
- `enc` neturi `--deadline-ms` ir `--read-offset-us` parinkčių (`main_func.cpp:104-119`) — visada 30 ms ir 500 µs.
- Skaitikliai — `std::map<std::string, uint64_t>` karštame kelyje (`func_common.hpp:96-99`): kiekvienas `inc("…")` sukuria
  `std::string` ir ieško medyje; vardams > 15 simbolių (`audio_underrun_frames`) — dar ir alokacija **[prielaida: libstdc++ SSO]**.

### 3.2 Ką P6 jau pataisė ir ko ne (svarbu izoliacijos įrodymui)

`runtime::InputSlot` (`src/runtime/input_slot.cpp:128-153`) po 10 ms `TOO_EARLY` iš eilės tikrina `mxlIsFlowActive` ir nužudyto
rašytojo flow atjungia. Todėl A1 scenarijus HEAD'e skyla į du:

| Scenarijus | HEAD elgsena [iš kodo, nevykdyta] |
|---|---|
| Rašytojas nužudytas (`kill -9`, segfault) | **Vieną** tick'ą visi proceso flow'ai laikomi ≈ 10 ms (commit ≈ T(i) + 12 ms, be praleidimų); nuo kito tick'o — laisvi |
| Rašytojas **gyvas, bet nerašo** (SIGSTOP, pakibęs GPU, perkrautas dekoderis, lėtas tinklas) | `flock` laikomas → `mxlIsFlowActive` = true → **kiekvieną** tick'ą visi laukia iki T(i) + 30 ms, commit ≈ T(i) + 31,5 ms, `mxlGetCurrentIndex` jau i + 2 → `late_skips` kas antram laukui **visiems** flow'ams, kol procesas neperkrautas. Tai tiksliai [A] 1 radinio scenarijus — jis HEAD'e tebegalioja |

## 4. MXL sutartis, kuria remiasi dizainas

| # | Faktas | Žyma |
|---|---|---|
| M1 | `mxlFlowReaderGetGrainSlot(…, timeoutNs, …)`: 0 — grįžta nelaukęs; niekada negrąžina `MXL_ERR_TIMEOUT` — nesulaukus `MXL_ERR_OUT_OF_RANGE_TOO_EARLY` | [patikrinta: `lib/include/mxl/flow.h:690-711`] |
| M2 | Terminas = `CLOCK_REALTIME` dabar + timeout; laukiama **santykiniu** `FUTEX_WAIT` ant flow'o `syncCounter`; po kiekvieno pabudimo likutis perskaičiuojamas iš `CLOCK_REALTIME` | [patikrinta: `lib/src/flow.cpp:19-22`; `lib/internal/src/Sync.cpp:30-34, 68-111`; `PosixDiscreteFlowReader.cpp:266-288`] |
| M3 | Commit: `headIndex = index`, grain'o antraštė (su `flags`) nukopijuojama, `lastWriteTime` = TAI, `syncCounter++`, `wakeAll` | [patikrinta: `PosixDiscreteFlowWriter.cpp:217-233`] |
| M4 | Skaitytojo rezultatas: `index > head` → `TOO_EARLY`; `index < head − grainCount + 2` → `TOO_LATE`; kitaip `OK`, jei `validSlices ≥ min(prašyta, totalSlices)` **arba** yra `MXL_GRAIN_FLAG_INVALID` | [patikrinta: `PosixDiscreteFlowReader.cpp:213-264`] |
| M5 | Necommit'intas indeksas: rašytojui atidarant vėlesnį indeksą, praleisti pažymimi `validSlices = 0` + INVALID; skaitytojas juos mato tik po **kito commit'o** (iki tol `TOO_EARLY`) | [patikrinta: `PosixDiscreteFlowWriter.cpp:145-162` + M4] |
| M6 | Rašytojas sustojo: `TOO_EARLY` be galo; `FLOW_INVALID` tik kai pasikeitė flow duomenų failo inode arba failo nebėra | [patikrinta: `PosixDiscreteFlowReader.cpp:187-201, 295-306`] |
| M7 | Sisteminiai kvietimai skaitant: sėkmingas skaitymas — vienas `futimens`; `TOO_EARLY` — vienas `stat` | [patikrinta: `PosixDiscreteFlowReader.cpp:31-37, 189-198, 300-305`] |
| M8 | `mxlFlowReaderGetRuntimeInfo` — `headIndex`, `lastWriteTime` iš bendros atminties, be sisteminio kvietimo | [patikrinta: `lib/src/flow.cpp:350-359`; `PosixDiscreteFlowReader.cpp:75-88`] |
| M9 | `mxlIsFlowActive` = `open` + `flock(LOCK_EX\|LOCK_NB)` + `close` flow duomenų failui | [patikrinta: `lib/src/flow.cpp:29-66`] |
| M10 | INVALID grain'as, commit'intas su `validSlices == totalSlices`, pastumia `_lastCommittedIndex`; „Writing grain marked as invalid is the proper way to make the ringbuffer move forward“ | [patikrinta: `PosixDiscreteFlowWriter.cpp:225-229`; `flow.h:27-33`] |
| M11 | Atidaryti indeksą ≤ paskutinio commit'into → `MXL_ERR_INVALID_ARG` | [patikrinta: `PosixDiscreteFlowWriter.cpp:140-143`] |
| M12 | Sync grupė: skaitytojai tikrinami **paeiliui** su bendru terminu; grąžinama **pirma** nesėkmė, nepasakant, kuris skaitytojas; lėčiausias perkeliamas į sąrašo pradžią; vienas `timestamp` visiems | [patikrinta: `FlowSynchronizationGroup.cpp:77-132`; `flow.h:930-1019`] |
| M13 | Ring'o gylis = 200 ms × grain'ų dažnis (50 Hz — 10, 25 Hz — 5 grain'ai); keičiamas domeno parinktimi | [patikrinta: `Instance.cpp:124, 302, 433`] |
| M14 | `mxlSleepUntil` — absoliutus `clock_nanosleep(CLOCK_TAI, TIMER_ABSTIME)` | [patikrinta: `lib/src/time.cpp:70-75`; `Thread.cpp:56-61`] |
| M15 | Flow aprašas įrašomas toks, koks paduotas (nežinomos žymos išlieka); `grouphint` žyma privaloma | [patikrinta: `FlowManager.cpp:94-100`; `FlowParser.cpp:164-172`] |
| M16 | Adresas `mxl:///<domeno-kelias>?id=<uuid>` | [patikrinta: `docs/Addressability.md:13, 39`] |
| M17 | Garso (continuous) skaitytojas — tas pats futex mechanizmas | [patikrinta: `PosixContinuousFlowReader.cpp:163-176`] |
| M18 | Paskutinis rašytojas atlaisvindamas ištrina flow | [patikrinta: `Instance.cpp:211-229`] |
| M19 | Viena gija vienu metu gali laukti tik **vieno** flow (MXL neturi „lauk bet kurio“; `syncCounter` adresas — ne viešas API) | [patikrinta: viešame `flow.h` tokios funkcijos nėra; M12] |
| M20 | Laikrodžiui šokus atgal per Δ blokuojantis skaitymas gali pailgėti iki Δ, jei tuo metu futex pabunda be mūsų grain'o | [patikrinta: seka iš M2; [L] §1.4] — tikras elgesys **[nevykdyta]** |
| M21 | `mxl-gst-sink` elgsena su INVALID grain'ais | **[prielaida — matuoja P5b]** |

## 5. Architektūra

```
            procesas = viena funkcija = viena GPU (K16)                     kiti procesai
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  TickEngine — VIENA gija vienam grain'ų dažniui; jai priklauso visi MXL handle'ai      │
 │                                                                                        │
 │  įėjimai            ribų laiko juosta           funkcija            išėjimai           │
 │ ┌───────────┐  head ┌──────────────────┐ batch ┌──────────────┐    ┌───────────────┐   │
 │ │ InputPort │──────►│ kvorumas → A     │──────►│ TickFunction │───►│ lap patikra   │   │   MXL flow
 │ │ automatas │◄──────│ vėluojantys → B  │       │ ::process    │    │ commit / K3   │───┼──► (cuda-linear / host)
 │ │ Live/Late/│ futex │ pernešti → C     │       │ (1 sync)     │    │ OutputPort    │   │
 │ │ Stalled/… │ (ašis)│ spraga → K3      │       └──────▲───────┘    └───────────────┘   │
 │ └─────▲─────┘       └────────▲─────────┘              │ darbininkai (dec / enc eilės)  │
 │       │ (domenas, UUID)      │ Clock (System | Virtual)                                │
 │ ┌─────┴──────────────────────┴─────────────────────────────────────────────────────┐   │
 │ │ StatsHub: skaitikliai, būsenos (BCP-008 vardai), histogramos ──► StatsSink (P7)    │   │
 │ └────────────────────────────────────────────────────────────────────────────────────┘ │
 │   valdymo eilė: retarget / enable / add / remove / stop  ◄── signalai, conn/*.json     │
 │   priėmimo kabliukas (P4 knyga, viena GPU)               ◄── add_unit                  │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

**Sluoksniai (priklausomybės tik žemyn):**

| Biblioteka | Turinys | Priklauso nuo | Kodėl atskirai |
|---|---|---|---|
| `mxlv_engine` | `Clock`, ribų matematika, įėjimų automatas, `GapState`, `TickEngine`, `StatsHub` | `mxlv_timing` | **Be MXL, be CUDA, be CLI11** → vienetų testai be GPU ir be domeno (kaip `mxlv_timing`) |
| `mxlv_engine_mxl` | `MxlVideoInput` (ant `InputSlot`), `MxlVideoOutput` (ant `VideoOut`), `MxlAudioInput`, `SystemClock` | `mxlv_engine`, `mxlv_runtime`, `mxlv_flows` | Vienintelė vieta, kur variklis kalba su MXL |
| `vramflow-func`, `libvramflow-gfx` | `TickFunction` įgyvendinimai; biblioteka — ritmo vykdytojas | abi aukščiau | Biblioteka variklį linkuoja statiškai (PIC), eksportuoja tik `vfgfx_*` |

**Gijos.** Viena variklio gija procesui ir grain'ų dažniui (šiandien — tiksliai viena). Kelių dažnių procese (būsimas `ingest`)
— po vieną variklį dažnio grupei; kiekvienas MXL handle priklauso lygiai vienam varikliui. Bibliotekoje — vienas variklis
kiekvienam (grain'ų dažnis, `offset_ns`, GPU). Darbininkų gijos (`dec`, `enc`, CPU baseinas) nekinta. Statistiką rašo atskira
gija (P7), kuri kartu yra ir „širdies plakimas“: jei variklio gija stringa (pakibęs `cudaStreamSynchronize`), eilutės toliau
rašomos su `"state":"Unhealthy"` ir `stalled_ms`.

**Vienetas (unit)** — izoliacijos ir apskaitos atomas: vienas išėjimas (arba sink'as) su savo įėjimų rinkiniu. `logo`: 1 įėjimas
→ 1 išėjimas; `composite`: bazė + K → 1; `enc`: video (2 grain'ai) + garsas → eilė; `gen`, `dec`, ritmo gija: šaltinis → 1.
Visi varikliai tiksi **grain'ų dažniu**; vienetas gali dalyvauti kas `stride`-ą tick'ą (`enc` interlaced: `stride = 2`,
`residue = 1` — tick'as ties apatiniu lauku, kaip šiandien `i_bot`).

## 6. Dizaino sprendimai

### D1. Variklio forma ir gijų modelis

**Rekomendacija.** Vienas `TickEngine` su vienu ciklo įgyvendinimu; funkcijos — įskiepiai (`TickFunction`). Viduje variklis
sudėtas iš mažų grynos logikos dalių (ribos, automatas, spragų būsena, statistika), kurias galima tikrinti po vieną. Dekodavimas
ir kodavimas į tą patį kurpalį nespraudžiami: variklis žino tik **briaunas**, ne funkcijas —

- *šaltinio vienetas* (`gen`, `dec`, `ingest`, ritmo gija): MXL įėjimų nėra; ar turinys yra, funkcija pasako `begin_tick` metu
  (`ctx.no_content(unit)`); darbininkų eilės lieka funkcijos viduje;
- *sink vienetas* (`enc`): MXL išėjimų nėra; `process` suaudžia, `end_unit` deda į darbininko eilę;
- *transformacijos vienetas* (`logo`, `composite`, `mv`): MXL abiejose pusėse.

Batch'as: variklis paduoda `Batch{index, units[]}` (≤ N vienetų, kiekvienas su savo K įėjimų rodyklėmis ir atidarytais išėjimo
slot'ais); funkcija iš jo sudaro kernel'io darbų masyvą (kaip šiandien, by-value). Vienam indeksui — ne daugiau kaip 3 `process`
kvietimai (A, B, C), įprastai 1; kernel'io paleidimų skaičius ir toliau nepriklauso nuo N.

**Atmesta:** įrankių rinkinys be bendro ciklo (lieka 7 ciklai; P4 / P7 / K3 vėl × 7) · variklis, valdantis ir CUDA
(pažeidžia „kernel'ių sluoksnis geras“; variklis privalo būti testuojamas be GPU) · gija kiekvienam flow'ui ar įėjimui (128
gijos pabunda tą pačią akimirką — A10; MXL handle — vienam kviečiančiajam) · procesas kiekvienam flow'ui (po CUDA kontekstą) ·
`dec` / `enc` kaip „MXL → MXL“ vienetai su tarpiniais flow'ais (papildomas ring'as ir kopija) · grafo karkasas su keliomis
funkcijomis viename procese (ne šio darbo apimtis; viena funkcija procese lieka).

**Autoriaus sprendimas:** migracijos eilė (K-T13), ar `verify` lieka už variklio (K-T15).

### D2. Įėjimo būsenų automatas ir izoliacijos taisyklė

**Būsenos:** `Disabled` (išjungtas valdymo — IS-05 `master_enable: false`), `Waiting` (dar niekada neprijungtas), `Live`
(prijungtas; papildoma žymė `Late`), `Stalled` (prijungtas, rašytojas gyvas, bet M tick'ų iš eilės nepateikė grain'o),
`Detached` (buvo prijungtas, rašytojo nebėra), `Mismatch` (grįžo kitu formatu, politika `exit` — galutinė).

**Žymėjimai.** `r` — variklio grain'ų dažnis, `T(i) = time_of(i, r)`, `P = T(1) − T(0)`. `ro` = `--read-offset-us`. `E` — laukiamas
įėjimų vėlinimas (didžiausias `expect_ns` tarp Live įėjimų; kol briaunos sutartis neįjungta — 0). `Φ = E + ro` — variklio fazė.
`Wr` = `--ready-wait-ms`, `D` = `--deadline-ms`, `G` — apsauga B fazės apdorojimui, `M` = `--stall-after`.

Numatytosios reikšmės — sveikaisiais skaičiais nuo P: **`Wr = P/4`, `D = 3·P/4`, `G = 3·P/20`, `M = 3`**; `D_eff = min(D, 2·P − G)`.

| Grain'ų dažnis | P | Wr | D | G | didžiausias D (2P − G) |
|---|---|---|---|---|---|
| 25/1 | 40 000 000 ns | 10,0 ms | 30,0 ms | 6,0 ms | 74,0 ms |
| 30000/1001 | 33 366 667 ns | 8,34 ms | 25,03 ms | 5,01 ms | 61,73 ms |
| 50/1 | 20 000 000 ns | 5,0 ms | 15,0 ms | 3,0 ms | 37,0 ms |
| 60000/1001 | 16 683 333 ns | 4,17 ms | 12,51 ms | 2,50 ms | 30,86 ms |

Pastaba: 15 ms prie 50 Hz — tai [F] §3 pirminė reikšmė („< lauko periodo, kad kitas tick nevėluotų“); 30 ms prie 25 Hz —
šiandieninė numatytoji.

**Ribos indeksui c** (visos — absoliutūs TAI laikai):

```
t_open(c) = T(c) + Φ                                   // tick'o atidarymas
A(c)      = t_open(c) + Wr                              // ilgiausiai tiek paruošti laukia neparuoštų
B(c)      = min(T(c) + E + D_eff, t_open(c+1) − G)      // paskutinė riba savame periode
C(c)      = T(c) + E + D_eff, tik jei > B(c)            // pernešimas (kai --deadline-ms ilgesnis už periodą)
```

**Taisyklės (pinned; įgyvendinimas — vienas, testai tvirtina reikšmes):**

- **T1 Atidarymas.** Tick'as i atidaromas ties `t_open(i)`, niekada anksčiau. Jei variklis pavėlavo — iškart; ribos nepasislenka.
- **T2 Vieneto indeksas.** Vienetas turi ne daugiau kaip vieną neišspręstą indeksą `cur`. Atidarant tick'ą i: laisvas vienetas
  gauna `cur = i`; vienetas, vis dar laukiantis `i − 1` (pernešimas), tick'e i nedalyvauja, kol `i − 1` neišspręstas; išsprendus
  — prisijungia prie i, jei i ribos dar nepraėjo, kitaip i — spraga (`carry_overrun`). Todėl vieno rašytojo indeksai visada
  didėja po vieną.
- **T3 Klasė tick'o pradžioje.** `Blocked` — bent vienas būtinas įėjimas ne `Live` arba šaltinis pranešė `no_content`;
  `Quorum` — visi būtini įėjimai `Live` be žymės `Late`; `Straggler` — visi `Live`, bent vienas `Late`.
- **T4 Blocked.** `Stalled` įėjimui — vienas `head()` skaitymas (be sisteminio kvietimo, M8); jei jo grain'as paruoštas ir kiti
  įėjimai paruošti — vienetas apdorojamas A fazėje. Kitaip spraga commit'inama **tick'o atidarymo metu** (`T(i) + Φ`).
- **T5 A paleidimas.** Pirmą akimirką, kai paruošti **visi `Quorum`** vienetai, bet ne vėliau kaip `A(i)`. Į batch'ą patenka visi
  tuo metu paruošti vienetai (ir `Straggler`). Paruoštų nėra — paleidimo nėra.
- **T6 Po A.** Kiekvienam `Quorum` įėjimui, neparuoštam A paleidimo metu: `writer_alive()` (M9) = ne → `Detached`, jo vienetas —
  spraga dabar; = taip → žymė `Late`.
- **T7 B paleidimas.** Pirmą akimirką, kai paruošti visi likę tick'o i vienetai, bet ne vėliau kaip `B(i)`. Jei `C(i)` nėra —
  nesulaukę vienetai: spraga (`late`), jų neparuoštiems įėjimams `miss_run++`.
- **T8 C paleidimas (pernešimas).** Jei `C(i)` yra — nesulaukę vienetai lieka neišspręsti per kito tick'o atidarymą; paleidžiama,
  kai visi paruošti, bet ne vėliau kaip `C(i)`; po to — spraga ir `miss_run++`. Daugiausia vienas perneštas indeksas vienetui.
- **T9 Laukimas.** Tarp įvykių gija blokuojasi tik iki **artimiausio** būsimo įvykio iš {`t_open(i+1)`, atvirų tick'ų A, B, C}
  (§D3). Niekada ilgiau.
- **T10 Pertraukiklis.** `miss_run ≥ M` → `Stalled`: nelaukiama visai. `Stalled` → `Live + Late`, kai `head` padidėjo nuo
  praėjusio tick'o (rašytojas vėl gamina). `Late` nuimama, kai įėjimas buvo paruoštas A paleidimo metu.
- **T11 Variklio vėlavimas.** Po tick'o i: `next = i + 1`; `cur = index_of(dabar − E, r)`; jei `cur > next` — praleista
  `cur − next` indeksų, `next = cur` (jokio pasivijimo). Klasė: tame tick'e aptiktas laikrodžio šuolis → `clock_skips`; iki
  `--warmup` → `warmup_skips`; kitaip `late_skips`. (Kai `E = 0`, tai tiksliai šiandieninė taisyklė su `mxlGetCurrentIndex`.)
- **T12 Lap patikra ir commit.** Po `process`: kiekvienam naudotam įėjimui `lapped(max(head, now_idx), j, slotCount)` → išėjimas
  atšaukiamas, `lapped++`; kitaip commit. Tvarka kaip šiandien.
- **T13 Sustabdymas.** `stop` tikrinamas po kiekvieno laukimo; atidaryti grain'ai atšaukiami; neišspręsti vienetai — ne praradimas.

**Perėjimai:**

| Iš | Įvykis | Į | Skaitikliai |
|---|---|---|---|
| `Waiting` / `Detached` | `attach_poll` prijungė (gyvas rašytojas, `head` žinomas, formatas tinka; ne dažniau kaip kas 100 ms) | `Live + Late` | `reattached++` (tik iš `Detached`) |
| `Waiting` / `Detached` | formatas kitas, politika `exit` | `Mismatch` | variklis baigia, exit 6 |
| `Live` | paruoštas A paleidimo metu | `Live` (be `Late`), `miss_run = 0` | `ok++` |
| `Live` | neparuoštas ties A, rašytojas negyvas | `Detached` | `detaches++`, `lost++` |
| `Live` | neparuoštas ties A, rašytojas gyvas | `Live + Late` | — |
| `Live + Late` | paruoštas iki paskutinės ribos (B arba C) | `Live + Late`, `miss_run = 0` | `late++` |
| `Live (+ Late)` | neparuoštas iki paskutinės ribos | `Live + Late`; kai `++miss_run ≥ M` → `Stalled` | `lost++`; `stalls++` |
| `Live` | `get` → `FLOW_INVALID` | `Detached` | `detaches++` |
| `Live` | `get` → `TOO_LATE` arba `OK + INVALID` | `Live` | `too_late++` / `invalid++`; vienetas — spraga |
| `Stalled` | `head` padidėjo | `Live + Late`, `miss_run = 0` | — |
| `Stalled` | rašytojas negyvas (tikrinama kas 100 ms) | `Detached` | `detaches++` |
| bet kuri | `set_enabled(false)` / `set_enabled(true)` | `Disabled` / `Waiting` | — |
| bet kuri | `retarget(ref)` | `Waiting` | `reattached` nedidėja |

Naujai prijungtas įėjimas pradeda su `Late`: kol neįrodė, kad ateina laiku, kvorumo jis nelaiko.

**Ryšys su esamomis parinktimis.** `--read-offset-us` lieka (tai `ro`; su futex laukimu jo nebereikia tam, kad pirmas bandymas
nebūtų tuščias — A18, — bet reikšmė nekeičiama, kol neišmatuota). `--deadline-ms` lieka ir reiškia D — paskutinę ribą nuo
`T(i) + E`; reikšmė virš `2P − G` apkarpoma su įspėjimu paleidžiant. Nauja: `--ready-wait-ms` (Wr), `--stall-after` (M).

**Įrodymas skaičiais, 50 Hz** (P = 20 ms, ro = 0,5 ms, Wr = 5 ms, D = 15 ms, G = 3 ms, M = 3; `proc` — A fazės apdorojimas iki
commit'o; pavyzdžiui imu blogiausią išmatuotą p99 = 4,10 ms — `composite` G, K = 2, N = 32, WSL2
(`docs/rezultatai/2026-11-11-composite.md`); sveikas įėjimas ateina ≈ T + 1,2 ms — [A] 18):

| Tick'as | Miręs įėjimas k (`kill -9`) | Kitų N − 1 paleidimas | Kitų commit'as ne vėliau | Gija laisva |
|---|---|---|---|---|
| i (pirmas po mirties) | `Live` → ties A (T + 5,5) `writer_alive()` = ne → `Detached`, spraga | T + 5,5 ms | T + 5,5 + 4,1 = **T + 9,6 ms** | nuo T + 9,6 < `t_open(i+1)` = T + 20,5 |
| i + 1 ir toliau | `Detached` (spraga ties T + 0,5; bandymas prisijungti kas 100 ms) | T + 1,2 ms (kaip be gedimo) | T + 5,3 ms | — |

| Tick'as | Gyvas, bet užstrigęs įėjimas k (SIGSTOP) | Kitų N − 1 paleidimas | Gija laisva |
|---|---|---|---|
| i | ties A — gyvas → `Late`; laukiama iki B = T + 15; spraga; `miss_run = 1` | T + 5,5 ms | T + 15 ms < T + 20,5 |
| i + 1, i + 2 | `Late` → ne kvorume; laukiama iki B; `miss_run = 2, 3` → `Stalled` | T + 1,2 ms | T + 15 ms |
| i + 3 ir toliau | `Stalled`: vienas `head()` skaitymas, spraga ties T + 0,5 | T + 1,2 ms | T + 5,3 ms |

Išvados (galioja bet kuriam N, nes kito įėjimo būsena į kitų vienetų ribas neįeina):

1. Sveikų vienetų commit'as bet kuriame tick'e ≤ `T(i) + ro + Wr + proc` = T + 9,6 ms < P. Praleidimas (T11) įvyktų tik
   baigus vėliau nei `T(i+1) + P/2` = T + 30 ms ⇒ **0 `late_skips`, 0 prarastų grain'ų**.
2. Miręs įėjimas kitus palaiko **vieną kartą**, ne ilgiau kaip Wr − 1,2 = 4,3 ms; gyvas užstrigęs — taip pat vieną kartą
   (nuo antro tick'o jis `Late` ir kvorumo nelaiko). HEAD: atitinkamai ≈ 10 ms vieną kartą ir **30 ms kiekvieną tick'ą su 50 %
   praleidimų** (§3.2).
3. Gija niekada neblokuojama po `B(i) ≤ t_open(i+1) − G`, todėl kitas tick'as atidaromas laiku, jei `proc_B ≤ G`. Viršijus —
   skaitiklis `guard_overrun`, o laiku esantys vienetai vis tiek paleidžiami ne vėliau kaip `A(i+1)` (riba absoliuti).
4. 59,94 Hz: ≤ 0,5 + 4,17 + 4,1 = 8,77 ms < 16,68 ms. 25 Hz: ≤ 0,5 + 10 + 4,1 = 14,6 ms < 40 ms.
5. Su `--deadline-ms 30` prie 50 Hz (taip šiandien paleidžiami `composite` testai): B = T + 17,5, C = T + 30 = `T(i+1)` + 10.
   Tick'as i + 1 atidaromas ties T + 20,5, kiti vienetai paleidžiami ties `T(i+1)` + 1,2; k vienetas lieka prie i iki C. Kitiems —
   tie patys skaičiai kaip aukščiau.

Riba, kurios šis žingsnis neišsprendžia: tame viename tick'e sveikų vienetų išėjimai vėluoja iki 9,6 ms; grandinėje, kur žemiau
yra ≥ 4 pakopos su ta pačia D nuo `T(i)`, paskutinė gali nebesulaukti (≈ +2,5 ms kiekvienai pakopai). Tai A18 / A20 problema —
ją išsprendžia `expect_ns` (D7), ne ilgesnis terminas.

**Atmesta:** vienas bendras terminas visiems (šiandien) · tik trumpesnis terminas be dviejų fazių ([A] 1c vienas pats: kiti vis
tiek laikomi 0,6·P kiekvieną tick'ą) · adaptyvios ribos pagal išmatuotą vėlavimą (nedeterministiška, slepia blogėjantį
aukštupį, netestuojama tiksliomis reikšmėmis) · neparuoštą atmesti iškart be B fazės (prarandami grain'ai iš tik drebančio
aukštupio — 12–16 ms p99 jau išmatuota `composite` G) · terminą ≥ P apkarpyti be pernešimo (esami testai su `--deadline-ms 30`
WSL2 aplinkoje pradėtų kristi dėl 17–30 ms sutrikimų).

**Autoriaus sprendimas:** numatytosios Wr, D, M (K-T2, K-T3); ar palaikyti pernešimą (K-T1).

### D3. Kaip laukti

**Rekomendacija — „ašinis“ (pivot) laukimas:** blokuotis MXL skaitymu ant **vieno** įėjimo, kuris tikėtina ateis paskutinis,
ribotu laiku; kitus tikrinti be sisteminių kvietimų.

```
wait_ready(S, until):                       // S — laukiami (vienetas, įėjimas, indeksas j)
  kartoti:
    kiekvienam s ∈ S:  h = port.head()                       // M8: bendra atmintis; MXL_UNDEFINED_INDEX → neparuoštas
                       jei h ≥ j_s: port.probe(j_s)          // OK | INVALID | TOO_LATE → s išspręstas (M7: 1 futimens)
    jei fazės aibė paruošta arba dabar ≥ until: grįžti
    ašis = argmax_{s ∈ S} ewma_late(s)                       // lygybė → didžiausias įėjimo numeris
    port(ašis).wait(j_ašis, min(until − dabar, slice))       // M2: futex ant to flow; pabunda nuo commit'o (M3)
```

`ewma_late` — įėjimo atvykimo vėlavimo (`lastWriteTime − T(j)`, M8) eksponentinis vidurkis; naudojamas **tik ašiai parinkti**,
ne riboms. `slice` = 1 ms tik tada, kai vienu metu laukia dvi skirtingos aibės (pvz. pernešta i − 1 ir einamoji i); kitaip —
visas likęs laikas. MXL pats daro tą patį sync grupėje — lėčiausią skaitytoją kelia į priekį (M12).

**Kaina** (N įėjimų iš vieno aukštupio proceso, sveika būsena; [iš kodo, nevykdyta]):

| | Šiandien (apklausa 0,5 ms) | Ašinis laukimas |
|---|---|---|
| Sisteminiai kvietimai per tick'ą | ≈ 2 ratai × N `stat` + N `futimens` + 2 `clock_nanosleep` (N = 32: ≈ 98; N = 128: ≈ 386) | 1 `futex` + N `futimens` (33; 129) |
| Pabudimai laukiant | iki 2000 / s | ≈ 1 tick'ui (50–100 / s) |
| Pridėtas vėlavimas pakopai | iki 0,5 ms (apklausos žingsnis) | futex pabudimas (dešimtys µs) **[prielaida — matuoti]** |
| Blogiausias atvejis | — | N nepriklausomų aukštupių blogiausia tvarka: iki N blokuojančių kvietimų (vis tiek ≤ N sisteminių kvietimų) |
| Neteisinga ašis (ji mirė šiame tick'e) | — | laukiama iki A ribos — tai tas pats vienkartinis Wr iš D2 |

Sustabdymo reakcija: ≤ artimiausia riba ≤ P. Laikrodžio šuolis atgal gali pailginti vieną blokuojantį kvietimą iki Δ (M20) —
variklis tai aptinka po fakto (T12 laikrodžio taisyklė §D5) ir priskiria laikrodžiui; pataisoma tik upstream'e (K-T12).

**Atmesta:** 0,5 ms apklausa (šiandien; lieka kaip `--wait poll` diagnostikai ir migracijos metu) · MXL sync grupė visiems
įėjimams (viskas arba nieko, vienas `timestamp` — netinka `index_offset`, nepasako, kuris skaitytojas nesulaukė: M12; tinka tik
rinkiniui be poslinkių — palikta kaip galima optimizacija) · pagalbinė gija kiekvienam įėjimui (A10 pavyzdys; handle — vienam
kviečiančiajam) · `FUTEX_WAITV` tiesiai ant `syncCounter` (ne viešas API: M19) · sukimasis be miego.

**Autoriaus sprendimas:** ar siūlyti upstream'ui „lauk bet kurio“ / būseną kiekvienam skaitytojui (K-T12).

### D4. Spragų politika (K3) vienoje vietoje

Vienintelis kelias, kuriuo bet kuris rašytojas „nepateikia“ turinio — `OutputPort::fill_gap(index, action)`; veiksmą parenka
`GapState::decide` (gryna logika). Tai pakeičia [B] §P5 `GapPolicy{Leave, Invalid, Repeat}` ir `VideoOut::skip`.

| Politika (`--on-loss`) | Kas commit'inama | Kaina | Skaitytojas mato |
|---|---|---|---|
| `invalid` (numatyta, K3) | `flags \|= MXL_GRAIN_FLAG_INVALID`, `validSlices = totalSlices`, turinys neliečiamas | 0 GPU darbo; `open` + `commit` | `OK` + INVALID iškart (M4, M10) |
| `freeze` | paskutinio galiojančio **to paties pariteto** grain'o kopija iš savo ring'o (interlaced: `i − 2`; progresyvus: `i − 1`); ne ilgiau kaip `--hold-ms` nuo paskutinio gero, tada `black` | 1 kopija grain'ui (1080i50: 2,76 MB) | galiojantį grain'ą |
| `black` | iš anksto paruoštas juodas grain'as (Y = 64, Cb = Cr = 512; v210a — key = 0, t. y. skaidrus) | 1 kopija; VRAM + 1 grain'as kiekvienam rastrui (1080i: 2,76 MB; 1080p: 5,53 MB; UHD: 22,1 MB) | galiojantį grain'ą |
| `leave` (tik migracijai) | nieko (šiandieninė elgsena) | 0 | `TOO_EARLY`, kol nebus kito commit'o (M5) |

Papildomo VRAM `freeze` nereikia: šaltinis — ankstesnis grain'as tame pačiame ring'e (`step ≤ 2 ≤ slotCount − 2`, M13). Jei
to pariteto galiojančio grain'o ring'e nebėra — `black`. `freeze` ir `black` kopijos eina į funkcijos stream'ą ir
sinchronizuojamos kartu su tos fazės vieninteliu sync (host flow — `memcpy`).

**Kada commit'inama spraga:** `Blocked` vienetui — tick'o atidarymo metu (T4); vėluojančiam — ties jo paskutine riba (T7 / T8);
šaltiniui be turinio (`dec` underrun, ritmo gija be `submit`) — tick'o atidarymo metu.

**Startas, kol nėra pirmo įėjimo grain'o.** Režimas `immediate` (rekomenduojamas po migracijos): išėjimai sukuriami iškart
(visų esamų funkcijų išėjimo formatas žinomas iš parinkčių), variklis tiksi, vienetai be įėjimų — pagal politiką: `invalid` →
INVALID; `black` → juoda; `freeze` → juoda (nėra ko kartoti). Šie grain'ai skaičiuojami `gap_startup`, ne praradimu.
`--input-wait-ms N > 0` išlieka: jei būtinas įėjimas per N ms nė karto neprisijungė — exit 4. Režimas `wait-all` (šiandien):
variklis nepradeda, kol neprisijungė visi įėjimai.

**Kai įėjimas grįžta:** vienetas tęsia pirmame tick'e, kuriame visi jo įėjimai paruošti. Interlaced išėjimui su `freeze` /
`black` turinys grįžta **tik lyginiu indeksu** (kadro riba) — kitaip vienas kadras būtų sudėtas iš seno viršutinio ir naujo
apatinio lauko. Su `invalid` — iškart.

**Variklio paties praleisti indeksai** (T11) neužpildomi: MXL juos pažymi INVALID atidarant kitą indeksą (M5) — nulinė kaina
tada, kai variklis ir taip vėluoja. Tai vienintelė išimtis iš „`freeze` / `black` ⇒ visada galiojantis grain'as“.

**Biblioteka.** Ritmo režime prieš pirmą `vfgfx_submit` — spragų politika (šiandien: nieko; `gfx.h` tekstas „Iki pirmo submit
grain'ų nėra“ keičiamas tame pačiame PR). Po `submit` paskutinio kadro kartojimas — ne spraga (grafika statinė).
Garso (continuous) išėjimų politika — P3 sprendimas (tyla ar nerašyti); sąsaja tam palikta.

**Atmesta:** `leave` kaip galutinė (A13) · visur kartoti (slepia gedimą — [B] §P5) · INVALID su `validSlices = 0` (nepastumia
`_lastCommittedIndex` — M10) · pildyti variklio praleistus indeksus (kopijos tada, kai jau vėluojama) · atskiras „sustingusio
kadro“ buferis kiekvienam išėjimui (VRAM be reikalo).

**Autoriaus sprendimas:** `freeze` be istorijos — `black` ar INVALID (K-T4); kadro riba grįžtant (K-T5); praleistų indeksų
nepildymas (K-T6); `immediate` kaip numatytasis (K-T7); numatytoji lieka `invalid`, kol P5b neparodė kitaip (M21).

### D5. Laikrodis

`Clock` sąsaja (§7) su `SystemClock` (`mxlGetTime`, `CLOCK_MONOTONIC_RAW`, `mxlSleepUntil` ne ilgesniais kaip 50 ms gabalais —
A27) ir `VirtualClock` (testams: `advance`, `step`). Įėjimų laukimas taip pat eina per sąsają (`InputPort::wait`), todėl
vienetų testuose scenarijus („įėjimas k ateina T + 26 ms“, „rašytojas miršta ties t“) vykdomas be tikro miego: laikas
pasistumia iki artimiausio įvykio. Visas variklio vienetų rinkinys trunka milisekundes ir nepriklauso nuo GPU apkrovos.

Laikrodžio žingsniai (modelis — [L] §1.4, [B] §P1 10 punktas):

| Įvykis | Aptikimas | Variklio elgsena |
|---|---|---|
| Žingsnis (bet kuris) | `StepDetector`: (TAI − MONOTONIC_RAW) pokytis > 5 ms tarp patikrų (kaip `ClockWatch`, `flows.cpp:84-96`); tikrinama atidarant tick'ą ir po kiekvieno blokuojančio laukimo | atviri tick'ai uždaromi: neišspręsti vienetai — spraga su priežastimi `clock`; `on_clock_step(Δ)` funkcijai (`ingest` → `SourceClock::on_clock_step`, `dec` → perpildymų langas); `clock_steps++` |
| Atgal per Δ | `t_open(i)` toli ateityje | gija miega iki `t_open(i)`; joks indeksas nerašomas du kartus (M11); `clock_backward++`; statistikoje sinchronizacijos būsena `Unhealthy`, kol laukiama |
| Pirmyn per Δ | T11: `cur > next` | praleisti indeksai → `clock_skips` (ne `late_skips`); jokio pasivijimo |
| Slew'as | nematomas | — |
| TAI poslinkis ≠ 37 s | `adjtimex` paleidžiant | exit 2, nebent `--allow-bad-clock`; įrašas statistikoje (A20; [L] §3.3) |

**Atmesta:** platesni slenksčiai esamuose tikro laiko testuose (problema lieka — A11) · kabliukai kiekviename cikle (šiandien
yra tik ritmo gijoje) · MXL pakeitimas testuose per linkerį (trapi, netikrina sąsajos).

**Autoriaus sprendimas:** leidimas perrašyti tikro laiko ritmo testus į virtualų laiką (K-T9).

### D6. Statistikos kabliukas (P7)

- Skaitikliai — fiksuotos struktūros, indeksuojamos numeriu (ne `std::map<std::string>`); vardai priskiriami tik išvedant.
  Pabaigos santraukos (`summary-*.json`) raktai **lieka tie patys** (adapteris `func::to_summary`).
- Rašo tik variklio gija. Tick'o pabaigoje (`StatsHub::end_tick`, po commit'ų, už bet kokio laukimo ribų), jei praėjo
  intervalas, ji nukopijuoja momentinę kopiją (128 flow'ų ≈ 55 KB) ir perduoda `StatsSink::publish` — P7 `StatsPublisher`
  ją formuoja ir rašo savo gijoje. Tick'as niekada nedaro I/O; jei ankstesnė kopija dar nepaimta — `dropped_lines++`.
- Kiekvienam įėjimui: `ok`, `late`, `lost`, `invalid`, `too_late`, `read_errors`, `reattached`, `stalls`, `detaches`, būsena,
  atvykimo vėlavimo p50 / p99. Kiekvienam išėjimui: `ok`, `late` (commit'inta B / C fazėje), `gap_invalid`, `gap_frozen`,
  `gap_black`, `gap_startup`, `lapped`, `late_skips`, `clock_skips`, `warmup_skips`, commit'o vėlavimo p50 / p99 (64 kibirų
  histograma, 256 B). Varikliui: `wake_err`, `wait_ms`, `proc_ms`, `out_lateness` (esamos histogramos), paleidimų skaičius,
  `held_ms_max` (kiek ilgiausiai paruoštus laikė neparuoštas), `guard_overrun`.
- Eilutės schema lieka `"v":1` ([B] §P7); pridedami tik neprivalomi laukai ([P] E0): viršuje `"state"`, `"group"`; `flows[]`
  įrašuose `"dir":"in"|"out"`, `"unit"`, `"state"`, `"in_state"`, `"domain"`, `"bundle"`, `"role"`.

Būsenų vardai — BCP-008 (`runtime::Health`, tas pats enum kaip [P] E0), skaičiuojami intervalui:

| | `Healthy` | `PartiallyHealthy` | `Unhealthy` | `Inactive` |
|---|---|---|---|---|
| Įėjimas | `Live`, `late = lost = invalid = 0` | `Live`, `ok > 0` ir (`late`, `lost` ar `invalid` > 0) | `Waiting`, `Stalled`, `Detached`, `Mismatch` arba `ok = 0` | `Disabled` |
| Išėjimas | visi indeksai `ok` | `ok > 0` ir buvo spragų; arba spragą dengia `freeze` / `black` | `ok = 0` | vienetas išjungtas |
| Procesas | visi `Healthy` | mišru | visi išėjimai `Unhealthy` arba variklio gija stringa | nėra vienetų |

`statusReportingDelay` (3 s) ir perėjimų skaitikliai NMOS prasme — sidecar'o reikalas ([R] M4); variklis duoda neapdorotą
intervalo būseną ir `state_changes`.

**Atmesta:** HTTP / Prometheus variklyje (K9) · atominiai skaitikliai, skaitomi kitos gijos (kaina kiekvienam prieaugiui ir
nesuderinta kopija tarp skaitiklių) · pilna 15 KB histograma kiekvienam flow'ui kiekvienam intervalui.

### D7. Briaunos laiko sutartis (A20)

Bazinė taisyklė lieka: **išėjimo indeksas = įėjimo indeksas**, vėlavimas matomas kaip commit'o laiko poslinkis ([F] §3 (c)).
Sąsaja dabar turi du dydžius kiekvienam įėjimui ir du — kiekvienam išėjimui:

| Dydis | Reikšmė | Numatyta | Kada naudojama |
|---|---|---|---|
| `EdgeTiming::index_offset` (grain'ais, ≤ 0) | įėjimo indeksas `j = i + index_offset` | 0 | kertant laikrodžio domeną (tinklo `bridge`), kai vėlavimas netelpa į ring'ą |
| `EdgeTiming::expect_ns` | įėjimo grain'as j laukiamas commit'intas iki `T(j) + expect_ns` | 0; vėliau — iš aukštupio žymos | grandinės gylis nebeeikvoja bendro biudžeto: fazė `Φ = max(expect_ns) + ro` |
| `OutputTiming::commit_bound_ns` | viršutinė šio išėjimo commit'o riba po `T(i)`: `Φ + Wr + G` | skelbiama | žemupio `expect_ns` |
| `OutputTiming::content_delay_ns` | turinio vėlinimas nuo kilmės: didžiausias per rinkinio narius `content_delay_in − index_offset · P` | skelbiama | A/V lygiavimas po perindeksavimo |

Skelbiama flow aprašo žymose (M15): `urn:x-vramflow:tag:commit-bound-ns`, `urn:x-vramflow:tag:content-delay-ns` (po vieną
dešimtainę eilutę), ir statistikoje. Svetimam flow be žymų — `--input-expect-ms` (numatyta 0).

**Rinkinio (video + garsas + key) lygiavimas.** Rinkinio nariai (tas pats `grouphint` rinkinio vardas) privalo turėti tą patį
`content_delay`. Vienetas ima `align = max(content_delay)` per narius ir kiekvieną narį skaito su papildomu poslinkiu
`(align − content_delay_k)` jo paties vienetais (video — grain'ais, garsas — mėginiais per `sample_begin`); išėjimas skelbia
`content_delay = align`. Viena fazė varikliui: vienetai su mažesniu `expect_ns` savo įėjimus tiesiog randa jau paruoštus.
MXL sync grupė rinkiniui be poslinkių gali būti naudojama kaip optimizacija; variklis nuo jos nepriklauso.

Dabar įgyvendinama tik tai, kas nieko nekainuoja: `index_offset` (vienas sudėjimas) ir `expect_ns` ribose (numatyta 0).
Žymų rašymas, `--input-offset`, lygiavimas ir TAI preflight — V24, prieš tinklo E3.

**Atmesta:** perindeksuoti visada (laužo [F] §3 (c), pridėtų vėlinimą kiekvienai pakopai) · sutartis iš išmatuoto vėlavimo
(nedeterministiška) · vienas `read_offset` procesui (šiandien) · fazių grupės viename variklyje (sudėtinga; kol kas skirtingo
gylio grandinės — skirtinguose procesuose, kaip ir dabar).

**Autoriaus sprendimas:** žymų vardai ir „skelbiama riba, ne matavimas“ (K-T14); viena fazė varikliui (K-T10).

### D8. Tarnybos gyvavimo ciklas (A7)

- **Numatyta — veikti iki sustabdymo:** `--duration 0`. `--duration N > 0` — matavimo režimas: N sekundžių ir tvarkinga pabaiga.
- **Exit kodas:** `--strict-exit auto|on|off`; `auto` = `on`, kai `--duration > 0` (harness'ai visada jį nurodo — jų elgsena
  nekinta: praradimai → 4), ir `off` tarnybos režime (0 — tvarkingas sustabdymas; ≠ 0 — tik fatalu: 1 parinktys, 2 paruošimas
  ar vykdymo klaida, 5 priėmimas atmetė, 6 formato pasikeitimas). Praradimai tarnybos režime eina tik į statistiką.
- **Sustabdymas** (SIGINT / SIGTERM arba `Stop` komanda): baigiamas einamasis laukimas (≤ P), atidaryti grain'ai atšaukiami,
  `TickFunction::finish` (darbininkų eilių išleidimas ≤ `--drain-ms`, numatyta 500), paskutinė statistikos eilutė ir santrauka,
  išėjimai atlaisvinami (flow ištrinamas — M18), įėjimai, nuoma, rezervacija. Antras signalas — greitas išėjimas atlaisvinus
  flow'us (A25: `sigwait` cikle).
- **Vieneto gedimas** (K17: formatas pasikeitė vidury srauto, darbininkas mirė): `ctx.fail_unit(u, kodėl)` — vienetas
  sustabdomas, jo išėjimai atlaisvinami (žemupys mato `Detached` ir elgiasi pagal savo politiką), kiti tęsia; proceso exit
  kodas pabaigoje ≠ 0.
- **RT:** `--rt-priority N` (SCHED_FIFO variklio gijai), `--cpu-affinity SĄRAŠAS`; numatyta išjungta.

**Atmesta:** palikti 30 s numatytąją · exit 4 už bet kurį praradimą tarnyboje (`Restart=on-failure` vieną pavėlavusį lauką
laikytų gedimu) · sugedusio vieneto išėjime be galo rašyti INVALID (flow atrodytų gyvas orkestratoriui).

**Autoriaus sprendimas:** K-T8, K-T11.

### D9. Priėmimas ir adresavimas

- **Priėmimo kabliukas:** `add_unit` prieš kurdamas išėjimus kviečia `EngineConfig::admit(UnitSpec, why)`; P4 jį įgyvendina
  per `Ledger` — **viena knyga vienai GPU visame serveryje** (`<šaknis>/.gpu/<gpu-uuid>/`, [R] K-14), įraše `group`. Paklausą
  funkcija įrašo į `UnitSpec::demand` (išėjimo ring'o VRAM = `grain_bytes × slotCount`, NVDEC / NVENC Mpix/s). Atmestas
  vienetas nesukuriamas (jo flow'ų domene nėra), kiti dirba; `enforce` — exit 5 pabaigoje ([B] §P4). Jau priimti niekada
  nesuprastinami (K13).
- **Adresavimas:** įėjimas — `runtime::FlowRef{domain, flow_id}` ([P] E0): `UUID` arba `mxl:///<domenas>?id=<uuid>` (M16);
  šiandieninė trumpoji forma (`--in-role`, `--flow-base`) lieka ir išverčiama į `FlowRef` vienoje vietoje (`make_flow_id`:
  be `--group` / `--bundle` — senasis `4d584c47-00RR-…`, su jais — UUIDv5 iš (grupė, rinkinys, rolė)). Kitas domenas —
  per `InstanceCache` (viena `mxlInstance` domenui procese).
- **Keitimas veikiant:** `TickEngine::post(Retarget | Enable | AddUnit | RemoveUnit | Stop)` iš bet kurios gijos; pritaikoma
  tarp tick'ų. Tuo pačiu keliu bus aptarnaujamas `conn/<receiver-id>.json` (IS-05) ir bibliotekos `vfgfx_start_pacing` /
  `vfgfx_stop_pacing`.

**Atmesta:** knyga domenui (du domenai vienoje GPU talpą skaičiuotų du kartus) · įėjimai tik pagal rolę ir numerį (nėra kaip
perjungti veikiant) · signalai ar lizdai į funkciją ([P] §3.4).

### D10. Migracija

Principas: **pirma perkėlimas be elgsenos pokyčio, tada elgsenos pokyčiai po vieną.** Variklis nuo pirmos dienos turi
`EngineConfig::legacy()` — reikšmes, su kuriomis jis elgiasi kaip šiandieninis ciklas (viena riba = terminas, apklausa 0,5 ms,
be pertraukiklio, `leave`, `wait-all`). Eilė ir vartai — §9.

**Atmesta:** viskas vienu PR (neperžiūrima, nėra kur atsitraukti) · perkelti kartu su nauja elgsena (regresijos nebeatskiriamos
nuo sprendimų) · pradėti nuo `logo` be skeleto patikros ant `gen` (žr. K-T13).

## 7. Sąsajos

```cpp
// src/engine/clock.hpp — be MXL
namespace mxlv::engine {
using Index = std::uint64_t;
class Clock {
public:
    virtual ~Clock() = default;
    virtual std::uint64_t now_tai() const = 0;                    // TAI ns
    virtual std::int64_t now_mono_raw() const = 0;                // CLOCK_MONOTONIC_RAW ns (žingsnių detektoriui)
    virtual void sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) = 0;   // absoliučiai; ≤ 50 ms gabalais
};
class VirtualClock final : public Clock {                         // testams
public:
    explicit VirtualClock(std::uint64_t start_tai_ns);
    void advance(std::uint64_t ns);                               // abu laikrodžiai
    void step(std::int64_t delta_ns);                             // tik TAI (laikrodžio žingsnis)
};
struct StepDetector {                                             // buvęs func::ClockWatch
    static constexpr std::int64_t kStepNs = 5'000'000;
    std::int64_t check(const Clock&);                             // 0 — nebuvo; kitaip Δ ns (su ženklu)
    std::uint64_t steps = 0, back_ns = 0;
};

// src/engine/cuts.hpp — gryna matematika
struct Cuts { std::int64_t open, a, b, c; bool has_c; };          // TAI ns
struct CutConfig {
    timing::Rational rate; std::int64_t read_offset_ns, expect_ns, ready_wait_ns, deadline_ns, guard_ns; bool single_phase;
};
constexpr std::int64_t period_ns(timing::Rational r);             // T(1) − T(0)
CutConfig default_cuts(timing::Rational rate);                    // Wr = P/4, D = 3P/4, G = 3P/20, ro = 500 µs
bool clamp_deadline(CutConfig&, std::string& warn);               // D_eff = min(D, 2P − G); true — apkarpyta
Cuts cuts_for(Index c, const CutConfig&);
Index due_index(std::uint64_t now_tai, const CutConfig&);         // index_of(now − expect_ns, rate) — T11

// src/engine/ports.hpp — be MXL tipų
enum class InState : std::uint8_t { Disabled, Waiting, Live, Stalled, Detached, Mismatch };
enum class Probe : std::uint8_t { Ready, Invalid, NotYet, TooLate, Gone, Error };
enum class Need : std::uint8_t { Required, Wanted, Optional };    // Wanted: laukiama, bet vienetas tęsia be jo (enc garsas)
enum class InMode : std::uint8_t { Exact, Latest };               // Latest: naujausias ne vėlesnis nei j (mv plytelė)
struct GrainRef { const void* ptr = nullptr; std::uint32_t flags = 0; };
struct EdgeTiming { std::int32_t index_offset = 0; std::int64_t expect_ns = 0; };

class InputPort {                                                  // kviečia tik variklio gija
public:
    virtual ~InputPort() = default;
    virtual bool attach_poll(std::uint64_t now_tai) = 0;           // neblokuoja; true — prijungtas; ≤ 1 bandymas / 100 ms
    virtual bool mismatch(std::string& why) const = 0;
    virtual Index head() = 0;                                      // be sisteminio kvietimo; ~0 — nežinoma
    virtual std::uint64_t last_write_tai() = 0;                    // ašies parinkimui
    virtual Probe probe(Index j, GrainRef& out) = 0;               // neblokuoja
    virtual Probe wait(Index j, std::uint64_t timeout_ns, GrainRef& out) = 0;   // blokuoja ≤ timeout
    virtual bool writer_alive() = 0;                               // tik ties A riba ir kas 100 ms Stalled būsenoje
    virtual bool lapped(Index j, Index now_idx) = 0;
    virtual void detach() = 0;
    virtual void retarget(const runtime::FlowRef&) = 0;            // [P] E0
};
enum class GapPolicy : std::uint8_t { Leave, Invalid, Freeze, Black };
enum class GapAction : std::uint8_t { None, Invalid, Repeat, Black };
enum class OpenResult : std::uint8_t { Ok, AlreadyCommitted, Error };   // AlreadyCommitted — laikrodis atgal (M11)
class OutputPort {
public:
    virtual ~OutputPort() = default;
    virtual OpenResult open(Index i, void*& slot) = 0;
    virtual bool commit() = 0;                                     // validSlices = totalSlices
    virtual void cancel() = 0;
    virtual bool fill_gap(Index i, GapAction a, Index repeat_from) = 0;   // kopijos — į funkcijos stream'ą
    virtual bool flush_gaps() = 0;                                 // vienas sync fazei (host — nieko)
    virtual bool interlaced() const = 0;
    virtual std::uint32_t slot_count() const = 0;
};

// src/engine/gap_state.hpp — gryna logika
struct GapState {
    GapPolicy policy = GapPolicy::Invalid; std::int64_t hold_ns = 1'000'000'000;
    // Grąžina veiksmą indeksui i; repeat_from — šaltinio indeksas (tik Repeat).
    GapAction decide(Index i, std::int64_t t_index_ns, bool interlaced, std::uint32_t slot_count, Index& repeat_from);
    bool may_resume(Index i, bool interlaced) const;               // freeze / black + interlaced → tik lyginis i
    void on_content(Index i, std::int64_t t_index_ns);             // commit'intas tikras turinys
    void on_gap(Index i, GapAction done);
};

// src/engine/engine.hpp
struct InputSpec { InputPort* port; EdgeTiming timing; std::uint8_t grains = 1; Need need = Need::Required; InMode mode = InMode::Exact; };
struct Demand { double nvdec_mpix_s = 0, nvenc_mpix_s = 0; std::uint64_t vram_bytes = 0; };   // = [B] §P4 Demand
struct UnitSpec {
    std::string name;                          // statistikai (flow ID arba rinkinys)
    std::vector<InputSpec> in;                 // tuščia — šaltinis
    std::vector<OutputPort*> out;              // tuščia — sink
    std::uint32_t stride = 1, residue = 0;     // dalyvauja, kai i % stride == residue (enc interlaced: 2, 1)
    Demand demand;
};
enum class Outcome : std::uint8_t { Ready, GapBlocked, GapLate, GapNoContent, GapClock, Lapped, Skipped };
struct UnitTick {
    std::uint32_t unit; Index index; std::uint8_t phase;           // 0 = A, 1 = B, 2 = C
    std::span<const GrainRef> in;              // Σ grains, įėjimų tvarka; nesulaukto Wanted / Optional ptr == nullptr
    std::span<void* const> out;                // atidaryti išėjimo slot'ai
};
struct Batch { Index index; std::uint8_t phase; std::span<const UnitTick> units; };

class TickContext {
public:
    Clock& clock(); Index index() const;
    void no_content(std::uint32_t unit);                           // šaltinis: šiam indeksui turinio nėra → K3
    void fail_unit(std::uint32_t unit, std::string why);           // K17
    void fatal(int exit_code, std::string why);
    void count(std::uint32_t unit, CounterId id, std::uint64_t n = 1);
    void hist(HistId id, std::uint64_t ns);
    bool clock_event_recent(std::uint64_t window_ns) const;        // A17
};
class TickFunction {
public:
    virtual ~TickFunction() = default;
    virtual void begin_tick(Index, TickContext&) {}                // t_open: šaltiniai paruošia turinį
    // Visi paruošti vienetai vienu batch'u; ≤ 3 kvietimai indeksui. Funkcija pati daro vieną GPU sync prieš grįždama.
    virtual bool process(const Batch&, TickContext&) = 0;          // false → fatal jau nustatytas
    virtual void end_unit(const UnitTick&, Outcome, TickContext&) {}   // po commit'o / spragos: garsas, enc eilė
    virtual void end_tick(Index, TickContext&) {}
    virtual void on_clock_step(std::int64_t delta_ns, TickContext&) {}
    virtual void finish(TickContext&) {}                           // sustabdant: eilių išleidimas
};

enum class StartMode : std::uint8_t { Immediate, WaitAll };
enum class WaitMode : std::uint8_t { Pivot, Poll };
struct EngineConfig {
    CutConfig cuts;                            // dažnis, ro, Wr, D, G
    std::uint32_t stall_after = 3;             // 0 — pertraukiklis ir Late žymės išjungti (legacy)
    GapPolicy on_loss = GapPolicy::Invalid; std::int64_t hold_ns = 1'000'000'000;
    StartMode start = StartMode::Immediate; WaitMode wait = WaitMode::Pivot;
    std::int64_t input_wait_ns = 0;            // 0 — neribotai
    double duration_s = 0, warmup_s = 5;
    std::uint32_t skip_align = 1;              // dec interlaced: 2 (po praleidimo tęsti nuo viršutinio lauko)
    std::function<bool(const UnitSpec&, std::string& why)> admit;   // P4; tuščias — priimti
    // Testų kabliukas (bibliotekos PacingHooks::wait): jei nustatytas, tick'as atidaromas juo, ne Clock::sleep_until.
    std::function<bool(Index, std::int64_t offset_ns, const std::atomic<bool>& stop)> open_wait;
    static EngineConfig legacy(timing::Rational rate, std::int64_t read_offset_ns, std::int64_t deadline_ns);
};
enum class EndReason : std::uint8_t { Stopped, Duration, Fatal, InputsTimeout, FormatChange };
struct RunResult { EndReason reason; int exit_code; std::string fatal; Index first_index, last_index; bool any_loss; };

struct Retarget { std::uint32_t unit, input; runtime::FlowRef ref; };
struct Enable { std::uint32_t unit, input; bool on; };
struct RemoveUnit { std::uint32_t unit; };
struct Stop {};
using Command = std::variant<Retarget, Enable, UnitSpec /* AddUnit */, RemoveUnit, Stop>;

class TickEngine {
public:
    TickEngine(const EngineConfig&, Clock&, StatsHub&, const std::atomic<bool>& stop);
    bool add_unit(const UnitSpec&, std::uint32_t& unit, std::string& why);   // prieš run(); false — priėmimas atmetė
    RunResult run(TickFunction&);                                   // kviečianti gija tampa variklio gija
    std::future<bool> post(Command);                                // iš bet kurios gijos; pritaikoma tarp tick'ų
};

// src/engine/stats.hpp
struct InCounters  { std::uint64_t ok, late, lost, invalid, too_late, read_errors, reattached, stalls, detaches; };
struct OutCounters { std::uint64_t ok, late, gap_invalid, gap_frozen, gap_black, gap_left, gap_startup, lapped,
                                   late_skips, clock_skips, warmup_skips, commit_errors; };
struct Snapshot;                                                    // POD masyvai + intervalas + būsenos
class StatsSink { public: virtual ~StatsSink() = default; virtual void publish(const Snapshot&) = 0; };   // privalo neblokuoti
class StatsHub {
public:
    explicit StatsHub(double interval_s, StatsSink* sink = nullptr);
    CounterId register_counter(std::string name);                   // prieš run(); funkcijos skaitikliai (frames_queued …)
    HistId register_hist(std::string name);
    void end_tick(std::uint64_t now_tai);                           // variklio gija
    runtime::Health health() const;
};
}  // namespace mxlv::engine

// src/engine/mxl_ports.hpp — vienintelė vieta su MXL
namespace mxlv::engine {
class SystemClock final : public Clock { /* mxlGetTime, clock_gettime(CLOCK_MONOTONIC_RAW), mxlSleepUntil */ };
class MxlVideoInput final : public InputPort {                      // ant runtime::InputSlot (+ writer_alive(), detach())
public: MxlVideoInput(runtime::InputSlot& slot, mxlInstance inst);
};
using GrainCopy = std::function<bool(void* dst, const void* src, std::size_t bytes)>;   // D2D į funkcijos stream'ą arba memcpy
class MxlVideoOutput final : public OutputPort {                    // ant func::VideoOut
public: MxlVideoOutput(func::VideoOut& out, GrainCopy copy, std::function<bool()> flush, const void* black_grain);
};
}
namespace mxlv::func {
engine::EngineConfig engine_config(const Opts& o);                  // parinktys → konfigūracija (viena vieta)
void to_summary(const engine::StatsHub&, const engine::RunResult&, Stats& out);   // tie patys raktai kaip šiandien
int exit_code_for(const engine::RunResult&, const Opts& o);         // --strict-exit
}
```

Papildymai esamame kode (maži): `runtime::InputSlot::writer_alive()`, `InputSlot::detach_now()` (dabar privatus `detach`);
`func::VideoOut::commit_invalid()`; `Opts` laukai naujoms parinktims; `func::ClockWatch` tampa `engine::StepDetector` apvalkalu.

**Kaip septyni ciklai sėda į sąsają:**

| Funkcija | Vienetai | `begin_tick` | `process` | `end_unit` |
|---|---|---|---|---|
| `gen` | N šaltinių | — | `gen_fields` / CPU / H | garsas (`AudioOut::write`) |
| `logo` | N: 1 įėjimas → 1 išėjimas | — | `logo_fields`, injekcija | — |
| `composite` | N: bazė + K → 1 | — | darbų sąrašas, `composite_batch` | `reopened`, diagnostika |
| `enc` | N sink: video (`grains = 2`, `stride = 2`) + garsas (`Wanted`) | — | weave, D2H | `Item` į darbininko eilę; spragai — vieta A2 pataisai |
| `dec` | N šaltinių, `skip_align = 2` interlaced | kadras iš eilės; `no_content` = underrun | split | — |
| ritmo gija | 1 šaltinis kiekvienam writer'iui, bendras variklis | `no_content`, kol nėra `submit` | visų kopijos viename stream'e, vienas sync | `committed++`, `on_commit` kabliukas |
| `gfxgen` manual | N·K šaltinių be variklio valdomų išėjimų | turinio paruošimas / prefetch | `vfgfx_write` kiekvienam | — |

## 8. Parinktys ir numatytosios reikšmės

| Parinktis | Reikšmė | Migracijos metu (`legacy`) | Po 4 etapo | Pastaba |
|---|---|---|---|---|
| `--read-offset-us` | `ro` | 500 (`gen`: 0) | tas pats | lieka |
| `--deadline-ms` | D, paskutinė riba nuo `T(i) + E` | 30 | 0 = auto `3·P/4` | lieka; > `2P − G` apkarpoma su įspėjimu; pridedama ir `enc` |
| `--ready-wait-ms` | Wr | = terminas (viena riba) | 0 = auto `P/4` | nauja |
| `--stall-after` | M | 0 (išjungta) | 3 | nauja |
| `--wait` | `futex` \| `poll` | `poll` | `futex` | diagnostikai |
| `--on-loss` | `invalid` \| `freeze` \| `black` (\| `leave`) | `leave` | `invalid` | K3 |
| `--hold-ms` | `freeze` trukmė | — | 1000 | K3 |
| `--start` | `immediate` \| `wait-all` | `wait-all` | `immediate` | K-T7 |
| `--input-wait-ms` | kiek laukti įėjimų | 0 (neribotai) | tas pats | lieka |
| `--on-format-change` | `exit` \| `wait` | `exit` | tas pats | lieka |
| `--duration` | sekundės; 0 = iki signalo | 30 | 0 | A7 |
| `--strict-exit` | `auto` \| `on` \| `off` | `on` | `auto` | nauja |
| `--warmup` | histogramų pradžia | 5 | tas pats | lieka |
| `--drain-ms` | eilių išleidimas sustabdant | — | 500 | nauja |
| `--stats-interval-s`, `--stats-out` | P7 | — | 1, išjungta | [B] §P7 |
| `--input REF` | `UUID` arba `mxl:///…?id=…` | — | šalia `--in-role` | [P] E0 |
| `--input-offset K`, `--input-expect-ms X` | `index_offset`, `expect_ns` | — | 0, auto iš žymos | V24 |
| `--admission` | `off` \| `warn` \| `enforce` | — | pagal K13 | [B] §P4 |
| `--rt-priority`, `--cpu-affinity` | variklio gija | — | išjungta | A19 |
| `--allow-bad-clock` | praleisti TAI patikrą | — | išjungta | A20 |

Vidinės konstantos (ne parinktys): G = `3·P/20`; žingsnio slenkstis 5 ms; prisijungimo bandymas kas 100 ms
(`InputSlot::kRetryNs`); blokuojančio laukimo gabalas 1 ms, kai laukia dvi aibės.

## 9. Migracijos žingsniai

**Vartai kiekvienam PR** (jei nepasakyta kitaip): 0 kompiliatoriaus įspėjimų; pilnas `ctest` (185) žalias; funkciniai
`functional.py` + `startstop.py` (variantai C ir G) žali; `fgold_*` žali (nuo V0a). „Bitas į bitą“ tikrinama taip: `verify
--every 1` (kiekvienas grain'as prieš nepriklausomą CPU etaloną) + nė vienas santraukos raktas nedingo ir nepakeitė prasmės
(nauji leidžiami) + praradimų skaitikliai be gedimo lieka 0.

**Jau nepriklauso nuo ciklo (lieka žali be pakeitimų):** vienetų testai (`test_common`, `test_format`, `test_gfx_*`,
`test_media*`, `test_source_clock`, `test_timing_index`); GPU kernel'ių ir atitikties testai (`test_kernels`,
`test_media_kernels`, `test_gfx_kernels`, `test_v210a_cuda_linear`); `test_flows` ir `test_runtime` (variklis juos naudoja, ne
keičia); `functional.py` F5–F8 (atskiras `vramflow-verify` binaras); `startstop.py` (tikrina iš išorės — taip parašyta jo
antraštėje); `chroma_psnr_*`, `build_arch`, `gfx_writer_c_header`, `gfx_writer_ctypes_smoke`. **Priklauso nuo ciklo ir yra
regresijos vartai:** `fgfx1`, `fcomp1…6`, `fcli1`, `ffmt_logo`, `ffmt_comp`, `ffmt_enc`, `test_gfx_writer` ritmo testai.

### 0 etapas — testai ir „prieš“ skaičiai (tik testai)

| PR | Kas | Testai ir tvirtinimai |
|---|---|---|
| **V0a** `test(functional): golden gates` (~300) | `tests/functional/golden.py`; santraukos raktų sąrašai `tests/functional/data/summary_keys_<rolė>.json` iš HEAD | `fgold_logo`, `fgold_composite` (N = 2, 10 s, C ir G, `verify --every 1`): `samples == present == (last_index − first_index + 1) · N`, `mismatch == 0`; funkcijos santraukoje Σ `fields_ok` == tiek pat, visi praradimų skaitikliai 0; raktų aibė == įrašytai. `fgold_dec` (klipas → `dec` → `verify --mode dec --ref`, N = 1): `samples > 0`, CRC neatitikimų 0, `parity_mismatch == 0` |
| **V0b** `test(functional): fault isolation probes` (~350) | `tests/functional/isolation.py`; kol kas būsena `INFO` su skaičiais (kaip esamas F6b) | `fiso_kill`: `gen` A (N − 1 flow) + `gen` B (1 flow, kitas `--flow-base`) → `logo` N → `verify` N; 10-ą sekundę `kill -9` B. Užrašoma kitiems N − 1: `late_fields`, `late_skips`, `out_lateness` p99 prieš ir po; paveiktam — ką mato `verify`. `fiso_stall`: tas pats su SIGSTOP 5 s ir SIGCONT. `fiso_composite`: nužudomas vieno išėjimo sluoksnio šaltinis. `fiso_enc`: du `gen` → `enc` N, nužudomas vienas. N = 8, C ir G. Rezultatas — „prieš“ stulpelis §10 |

### 1 etapas — variklio branduolys (be MXL, be GPU; `mxlv-engine-unit`)

| PR | Kas | Testai ir tvirtinimai (virtualus laikas, tikslios reikšmės) |
|---|---|---|
| **V1** `feat(engine): clock and cut points` (~300) | `clock.hpp`, `cuts.hpp`, `cmake/engine.cmake` | `cuts: numatytosios pagal dažnį` — 50/1: P = 20 000 000, `a − T` = 5 500 000, `b − T` = 15 000 000, C nėra; 60000/1001: P = 16 683 333, Wr = 4 170 833, D = 12 512 499, G = 2 502 499; 25/1: 10 / 30 / 6 ms. `cuts: --deadline-ms 30 prie 50 Hz` — `b − T` = 17 500 000, `c − T` = 30 000 000. `cuts: apkarpymas` — D = 60 ms → 37 000 000 ir įspėjimo tekstas su abiem skaičiais. `cuts: legacy` — viena riba ties T + 30 ms, B ir C nėra. `step_detector` — +4,9 ms → 0; +5,1 ms → žingsnis; −1,29 s → `back_ns` = 1 290 000 000. `due_index` — sutampa su `mxlTimestampToIndex`, kai `expect_ns = 0` (10⁴ atsitiktinių laikų) |
| **V2** `feat(engine): input state machine` (~250) | `input_state.hpp` | Kiekviena §6 D2 perėjimų lentelės eilutė — atskiras atvejis su skaitiklių reikšmėmis; `M = 3`: trys paskutinės ribos praleidimai iš eilės → `Stalled`, `stalls == 1`; du praleidimai + vienas sėkmingas → `miss_run == 0`; naujai prijungtas — `Late` |
| **V3** `feat(engine): gap policy state` (~200) | `gap_state.hpp` | `invalid` → visada `Invalid`. `freeze`, interlaced: geri 100, 101; spragos 102…: `Repeat` iš 100, 101, 102, …; ties `T(i) − T(101) > 1000 ms` → `Black`. `freeze` be istorijos → `Black`. `freeze`, šaltinio nebėra ring'e (`slot_count = 5`, praleisti 4 indeksai) → `Black`. `may_resume`: interlaced + `freeze` — nelyginis i → false, lyginis → true; `invalid` → visada true. `leave` → `None` |
| **V4** `feat(engine): tick loop, phases A/B, pivot wait` (~400) | `engine.hpp/.cpp`, `ports.hpp`, testų pakaitalai `ScriptedInput`, `RecordingOutput`, `RecordingFunction` | `engine: sveikas tick'as` — N = 4, įėjimai T + 1,2 ms: 1 paleidimas tiksliai ties T + 1 200 000 ns; 1 blokuojantis laukimas; 4 commit'ai indeksu i. `engine: miręs įėjimas` — N = 32, 50 Hz, k = 6 negyvas: tick'e i paleidimas ties T + 5 500 000 su 31 vienetu, k → `Detached`, `held_ns == 4 300 000`; tick'uose i + 1…i + 49 paleidimas ties T + 1 200 000, `held_ns == 0`; visiems 31: indeksai iš eilės, `late_skips == 0`. `engine: gyvas užstrigęs` — tick'e i: A ties 5,5 ms, spraga ties T + 15 000 000; i + 1, i + 2: A ties 1,2 ms; i + 3: `Stalled`, po A gija nebelaukia (kitas įvykis — `t_open(i + 4)`); `stalls == 1`. `engine: užstrigęs grįžta` — `head` padidėja → `Live + Late`; pirmas laiku atėjęs grain'as nuima `Late`. `engine: vėluojantis gyvas 8 ms` — A su kitais, B ties T + 8 000 000; išėjimo `late == 1`, `lost == 0`. `engine: sustabdymas laukiant` — išeina per vieną laukimą, atidaryti grain'ai atšaukti. `engine: duration` — 10 s prie 50 Hz → 500 ± 1 tick'ų; 0 → tik `stop`. Invariantas visuose: kiekvieno išėjimo commit'inti indeksai griežtai didėja; ≤ 3 `process` kvietimai indeksui |
| **V5** `feat(engine): carry, clock steps, skips` (~300) | pernešimas (C), T11, T12 | `engine: pernešimas` — `--deadline-ms 30`, 50 Hz, įėjimas k ateina T(i) + 26 ms: k@i commit'intas ties 26 ms, kiti i + 1 paleisti ties `T(i+1)` + 1,2 ms, k@(i+1) po to; rašytojo k seka i, i + 1 be spragos. `engine: variklis pavėlavo 45 ms` — abu laikrodžiai +45 ms: `late_skips == 2` kiekvienam išėjimui, jokio pasivijimo. `engine: šuolis pirmyn +1,3 s` — `clock_skips == 65`, `late_skips == 0`, `on_clock_step(+1 300 000 000)` vieną kartą. `engine: šuolis atgal −1,3 s` — virtualus miegas 1,3 s, joks indeksas du kartus, pirmas po šuolio = paskutinis + 1, `clock_backward == 1`. `engine: skip_align 2` — po praleidimo kitas indeksas lyginis |
| **V6** `feat(engine): stats hub` (~350) | `stats.hpp/.cpp` | `stats: skirtumai, ne sumos` — `ok` 50, 100, 150 → kopijose 50, 50, 50. `stats: būsenos` — lentelė §6 D6 po vieną atvejį kiekvienam langeliui; vardai tiksliai `Healthy`, `PartiallyHealthy`, `Unhealthy`, `Inactive`. `stats: tick'as neblokuojamas` — `publish`, kuris niekada negrįžta laiku: 1000 `end_tick` < 1 ms kiekvienas, `dropped_lines > 0`. `stats: be alokacijų tick'e` — skaičiuojantis alokatorius: 0 alokacijų po `run()` pradžios |

### 2 etapas — MXL adapteriai

| PR | Kas | Testai ir tvirtinimai (tikras MXL, host saugykla — GPU nereikia; `vramflow-engine-mxl-tests`) |
|---|---|---|
| **V7** `feat(engine): MXL ports, SystemClock, VideoOut gap commits` (~400) | `mxl_ports.hpp/.cpp`; `VideoOut::commit_invalid`; `InputSlot::writer_alive`, `detach_now` | `engine_mxl: INVALID skaitomas kaip OK + INVALID iškart` — indeksai 100, 101 (`fill_gap(Invalid)`), 102: skaitant 101 su timeout 0 → `OK`, žymė INVALID, `info.index == 101`; skaitant 101 **prieš** commit'inant 102 — taip pat `OK` (tai ir yra skirtumas nuo `leave`). `engine_mxl: Repeat` — grain'as 102 baitas į baitą == 100 (interlaced), be žymės. `engine_mxl: Black` — Y = 64, C = 512 visame grain'e; v210a key = 0. `engine_mxl: head() be sisteminių kvietimų` — 10⁵ kvietimų, `strace -c` ar `getrusage` skaitiklis nepakito (jei netinka — matavimas, ne testas). `engine_mxl: blokuojantis laukimas pabunda nuo commit'o` — kita gija commit'ina po 5 ms; `wait` grįžta `Ready`; žymė `realtime`, tvirtinama tik < P. `engine_mxl: 8 flow'ai, vienas rašytojas nužudytas` (fork + `kill -9`, `legacy` išjungtas) — kitų 7 išėjimų indeksai iš eilės, `late_skips == 0`; aštunto — INVALID kiekvienam indeksui po mirties |

### 3 etapas — perkėlimas (elgsena nekinta, kol nepasakyta kitaip)

| PR | Kas | Papildomi tvirtinimai |
|---|---|---|
| **V8** `refactor(gen): run on TickEngine` (~250) | `gen.cpp` ciklas → `GenFn`; `func::engine_config`, `to_summary` | `summary-gen.json` raktai nepakito (`fields_written`, `audio_blocks`, `gen_ms`, `commit_lateness`, `clock_backward`, `clock_steps`, `late_skips`, `warmup_skips`); `fgold_*` — `gen` yra visų jų šaltinis |
| **V9** `refactor(logo): run on TickEngine (legacy timing)` (~350) | `LogoFn`; C, H, G variantai | `fgold_logo`; `ffmt_logo` visais formatais; `fstart_order`, `fstart_restart`, `fstart_logo_restart` nepakitę (`reattached`, `missing_fields`); `out_lateness` p50 ± 0,3 ms nuo V0b „prieš“ (INFO) |
| **V10** `feat(engine): isolation on` (~200) | `engine_config` nebe `legacy` ribose: Wr = P/4, M = 3, `--wait futex`; D lieka 30 ms, `--on-loss leave`, `--start wait-all` | `fiso_kill` ir `fiso_stall` tampa **vartais** `logo`: kitiems N − 1 — `late_fields == 0`, `late_skips == 0`, `lapped == 0`, `verify`: `missing == 0`, `invalid == 0`, `mismatch == 0`; `out_lateness` p99 po gedimo ≤ p99 prieš + Wr + 1 ms; paveiktam — `missing_fields > 0`, `stalls ≥ 1` (`fiso_stall`), po SIGCONT 0 praradimų 10 s. `fgold_logo` nepakitęs |
| **V11** `refactor(composite): run on TickEngine` (~400) | `CompositeFn`; šalinamas atskiras `reopened` sekimas | `fcomp1…6`, `ffmt_comp`, `fgold_composite`; `fiso_composite` — vartai: kiti išėjimai 0 praradimų |
| **V12** `refactor(enc): run on TickEngine` (~400) | `EncFn` (sink, `stride`, garsas `Wanted`); `--deadline-ms`, `--read-offset-us` ir `enc` | `ffmt_enc`; `ts_sink`: kadrų skaičius, CC klaidų 0, A/V kaip prieš; `fiso_enc` — vartai: kitų srautų `frames_encoded` ≥ 25 · t − 1, `late_frames == 0` |
| **V13** `refactor(dec): run on TickEngine` (~350) | `DecFn` (šaltinis, `skip_align`); A17: perpildymai laikrodžiui priskiriami tik per `clock_event_recent(3 s)`; suminė nuolaida šalinama, PASS taisyklė — į harness'ą | `fgold_dec`; talpos `--func dec` viena pakopa (N = 4) kaip prieš. Naujas vienetų testas: žingsnis −40 ms 1-ą minutę, perpildymas 50-ą → skaičiuojamas `overrun_frames`, ne laikrodžiui |
| **V14** `refactor(gfx): shared pacing executor` (~400) | `gfx_writer.cpp`: `pace_loop_body` → vienas variklis kiekvienam (dažnis, `offset_ns`, GPU); visų writer'ių kopijos viename stream'e, vienas sync; `PacingHooks` lieka: `now` → `Clock`, `wait` → `EngineConfig::open_wait`, `on_commit` → `end_unit`, `fail_commit` → išėjimo apvalkalas (writer'is su kabliukais gauna savo variklį) | Visi 17 `test_gfx_writer` testų nepakeisti ir žali; `fgfx1`, `fcomp6`. Naujas: `gfx_writer: 32 rašytojai — viena ritmo gija` — `/proc/self/task` skaičius padidėjo 1, ne 32; visi 32 commit'ina kiekvieną indeksą |
| **V14b** `test(gfx): pacing tests on virtual clock` (~250) | 4 tikro laiko testai (`:363`, `:499`, `:539`, `:563`) → `VirtualClock`; likę tikro laiko — `LABELS realtime`, `RESOURCE_LOCK gpu` | Tvirtinimai griežtėja: 50 tick'ų → **tiksliai** 50 commit'ų (vietoje `≥ 48` per 1000 ms); `late_skips == 0` tiksliai. **Tik su autoriaus leidimu (K-T9)** |
| **V15** `refactor(gfxgen): manual mode on TickEngine` (~200) | `GfxgenFn` | `fgfx1` (manual), `fcomp6`; santraukos raktai nepakitę |
| **V16** `chore(engine): remove loop helpers, single-loop guard` (~150) | `wait_until` → `mxlSleepUntil` (A27); negyvas kodas | CTest `arch_single_loop` (`tests/arch/check_single_loop.sh`): `src/func/{gen,logo,composite,enc,dec,gfxgen}.cpp` ir `src/gfx/gfx_writer.cpp` neturi `wait_until(` nei `mxlSleepForNs(` |

### 4 etapas — elgsenos pakeitimai (kiekvienas atskirai; testų lūkesčiai keičiami tame pačiame PR ir tik su autoriaus žinia)

| PR | Kas | Testai ir tvirtinimai |
|---|---|---|
| **V17** `feat: gap policy K3` (~350) | `--on-loss`, `--hold-ms`; numatyta `invalid` visiems rašytojams ir ritmo gijai; `gfx.h` tekstas | `fiso_kill` paveiktam flow'ui: `verify` `invalid` == spragos indeksų skaičius ± 2, `missing == 0`; su `--on-loss freeze`: `invalid == 0`, `missing == 0`, pirmą sekundę grain'ai == paskutiniam geram (pagal paritetą), vėliau juodi; su `black`: juodi iškart. Žemupio skaitymas spragos indekse grįžta per < 1 ms (ne po periodo). **Keičiami lūkesčiai:** `fstart_restart` (`missing_fields` → `gap_invalid`), `test_gfx_writer:363` („be submit grain'ų nėra“ → INVALID grain'ai), `verify` skaitiklių lūkesčiai |
| **V18** `feat: immediate start` (~250) | `--start immediate` numatyta | `fstart_order`: `logo` išėjimo flow atsiranda per ≤ 1 s nuo `logo` starto (prieš `gen`); iki `gen` — `gap_startup > 0`, praradimų 0; po `gen` starto — kaip anksčiau. Neigiamas: `--input-wait-ms 1500` be gamintojo → exit 4 per ≤ 2 s (kaip `fstart_wait`) |
| **V19** `feat: service lifetime` (~300) | `--duration 0`, `--strict-exit`, `sigwait` cikle, `--drain-ms`, `--rt-priority`, `--cpu-affinity` | `fsvc_run`: `logo` be `--duration` veikia 40 s, SIGTERM → exit 0 per ≤ P + 600 ms, santrauka parašyta. `fsvc_loss`: tarnybos režime vienas `kill -9` aukštupyje → exit 0; su `--duration 20` tas pats → exit 4. `fsvc_second_signal`: du SIGTERM → procesas baigiasi per ≤ 200 ms, flow'ų domene nelieka |
| **V20** `feat: timing defaults by grain rate` (~150) | `--deadline-ms` auto = `3·P/4`; harness'ai WSL2 kampanijoms nurodo 30 aiškiai | `fiso_stall` su numatytosiomis: paveiktas išėjimas commit'ina spragą ties T + 15 ms ± 1 ms (50 Hz). `ffmt_*` 25p: nepakito (auto = 30 ms) |
| **V21** `feat: live stats sink` | [B] §P7 PR-1 ir PR-3 tampa `StatsSink` įgyvendinimu | [B] §P7 testai; papildomai: Σ `ok` per eilutes == santraukos `fields_ok`; būsena `Unhealthy` per ≤ 2 intervalus po `kill -9` |
| **V22** `feat: flow addressing and control queue` | [P] E0 PR-1, PR-2 per `TickEngine::post` | [P] E0 testai (`flow_ref`, `retarget`, `set_enabled`); papildomai `engine: retarget tarp tick'ų` — kitų vienetų indeksai be spragos |
| **V23** `feat: admission hook` | su [B] §P4 | [B] §P4 testai; `engine: atmestas vienetas` — jo `OutputPort::open` niekada nekviestas, kiti dirba |
| **V24** `feat: edge timing contract` (~350) | žymos, `--input-offset`, `--input-expect-ms`, rinkinio lygiavimas, TAI preflight | `engine: index_offset −2` — skaitomas `j = i − 2`, išėjimo `content_delay` = 2P. `engine: expect 9 ms` — `t_open` = T + 9,5 ms, sveikas vienetas paleidžiamas be laukimo. `engine: rinkinio lygiavimas` — video `content_delay` 40 ms, garso 0 → garsas skaitomas 1920 mėginių anksčiau. Funkcinis: 5 pakopų grandinė su `fiso_kill` viršuje — apačioje 0 praradimų (be žymų — užrašomas skirtumas) |

Priklausomybės: V1–V6 lygiagrečiai galimi V2, V3, V6 (po V1); V7 po V4; V8 po V7; V10 po V9; V11–V13 po V10 bet kuria tvarka;
V14 po V7; V17 po V14; V21–V23 — kai pasiekiami P7, E0, P4. P6 likę PR ir [B] P7 / P4 / P5 PR-2…3 **neturi būti daromi atskirai
kiekviename cikle** — jie tampa V17, V21, V23.

## 10. Matavimai (prieš / po)

„Prieš“ — HEAD `98b92c1`; kur neišmatuota, pirmas matavimas — V0b. „Po“ — tikslas ir kartu vartai.

| Matas | Prieš | Po (tikslas) | Kaip |
|---|---|---|---|
| Kitų N − 1 flow'ų `late_skips` + `late_fields`, kai vienas šaltinis gyvas, bet užstrigęs (SIGSTOP 5 s) | ≈ 50 % laukų praleista visiems [iš kodo, nevykdyta] | **0** | `fiso_stall`, N = 8 ir 32, `logo` G |
| Kitų N − 1 commit'o vėlavimas pirmame tick'e po `kill -9` | ≈ +10 ms [iš kodo] | ≤ Wr (5 ms prie 50 Hz), vieną tick'ą | `fiso_kill`, `out_lateness` maks. |
| Kitų N − 1 `out_lateness` p99 po gedimo | — (V0b) | ≤ p99 be gedimo + 1 ms | `fiso_*` |
| Sisteminiai kvietimai tick'ui, `logo` G N = 32 | ≈ 98 [iš kodo] | ≈ 33 | `strace -c -f` 10 s |
| Variklio gijos pabudimai per sekundę laukiant | iki 2000 | ≤ 150 | `perf stat -e sched:sched_switch` |
| Grain'o vėlinimas pakopoje, `logo` G N = 8, p50 / p99 | 1,2 ms p50 (README §8) | ≤ „prieš“ (laukiama −0,2…0,5 ms) | `out_lateness` |
| Kernel'io paleidimai tick'ui p50 / maks. | 1 / 1 | 1 / ≤ 3 | variklio skaitiklis |
| CPU branduolių flow'ui, `logo` G N = 32 ir 64 | README §8: 0,005 | ne blogiau kaip +3 % | `capacity.py` |
| `composite` G K = 2 N = 32, `out_lateness` p50 / p99 | 10,88 / 15,86 ms (`docs/rezultatai/2026-11-11-composite.md`) | ne blogiau; griežtas PASS ≥ 3 iš 5 | `capacity.py --func composite` |
| `gfxgen --flows 32 --layers 4 --mode pace`: ritmo gijų skaičius | 128 | 1 | `/proc/<pid>/task` |
| Tas pats: `composite` `late_fields` per 60 s | — (išmatuoti) | ≤ „prieš“ | A10 hipotezės patikra |
| Variklio pridėtinė kaina tick'ui vienetui (be MXL) | — | < 200 ns | `tests/perf/bench_engine.cpp`, virtualus laikas |
| Statistikos kaina: `logo` N = 32 p99 su `--stats-interval-s 1` ir be | — | skirtumas ≤ 1 % | [B] §P7 |
| `--on-loss` kaina, N = 8, visi flow'ai spragoje: commit p50 / p99, MB/s | `leave`: 0 | `invalid`: ≈ 0; `freeze` / `black` G: išmatuoti; host: išmatuoti (žr. R9) | [B] §P5 lentelė |
| Sustabdymo laikas SIGTERM → exit, p50 / maks. | — (išmatuoti; `dec` iki 20 s — A15, ne variklis) | ≤ P + `--drain-ms` | `fsvc_run` |
| Ciklo kodo eilutės | ≈ 773 septyniose vietose | viena vieta | `wc -l` + `arch_single_loop` |
| Variklio vienetų testų trukmė | — | < 1 s visam rinkiniui, 0 tikro miego | `ctest -R engine` |

Ant bet kurios NVIDIA GPU: tie patys skriptai; `fiso_*` ir `fgold_*` turi C variantą, kuriam GPU nereikia.

## 11. Rizikos

| # | Rizika | Poveikis | Mažinimas |
|---|---|---|---|
| R1 | Perkeliant pasikeičia skaitiklių vardai, exit kodai ar laikai | Harness'ai ir ataskaitos lūžta | `legacy()` konfigūracija, raktų sąrašai iš HEAD (V0a), `fgold_*` kiekviename PR |
| R2 | MXL blokuojančio skaitymo terminas ant `CLOCK_REALTIME` (M2, M20) | Laikrodžio šuolis atgal gali pailginti vieną laukimą iki Δ | Laukimai ≤ artimiausia riba; `StepDetector` priskiria laikrodžiui; siūlymas upstream'ui (K-T12) |
| R3 | `futimens` kiekvienam sėkmingam skaitymui (M7) | `composite` N = 128, K = 8 — iki 1152 sisteminių kvietimų tick'ui **[iš kodo; kaina neišmatuota]** | Išmatuoti V7; jei reikšminga — siūlymas upstream'ui riboti dažnį |
| R4 | Pernešimo (C) logika — daugiau būsenų | Indeksų tvarkos klaida viename rašytojuje | Daugiausia vienas perneštas indeksas; invariantų tvirtinimai kiekviename V4 / V5 teste; be `--deadline-ms > P` kelias nenaudojamas |
| R5 | Bibliotekos ritmo gija ant bendro variklio | `vfgfx_stop_pacing` turi likti sinchroninis; užraktų tvarka (`lat_m` laikomas per batch'o sync); išimtis vieno writer'io kelyje neturi stabdyti kitų | `RemoveUnit` per `future`; užraktai didėjančia writer'ių tvarka; esami 17 testų nepakeisti; išimtis → `fail_unit` |
| R6 | Dvi fazės drebančioje aplinkoje — du paleidimai tick'ui | `proc` p99 auga | Matuojama (§10); `--ready-wait-ms` derinamas; reikšmė = terminas grąžina vieną paleidimą |
| R7 | `invalid` numatytoji keičia tai, ką mato žemupys ir etaloniniai skaitytojai (M21) | `mxl-gst-sink` gali strigti ir ant INVALID | P5b matuoja; jei stringa — numatytoji `freeze` (K3); V17 atskirai nuo perkėlimo |
| R8 | `immediate` startas keičia paleidimo semantiką | Testai ir orkestravimas, kurie laukia „flow atsiras, kai bus signalas“ | Atskiras PR (V18); `--start wait-all` lieka |
| R9 | `freeze` / `black` host flow'ams — pilna grain'o kopija kiekvienam flow'ui kiekvieną tick'ą | Kai dingsta šaltinis, maitinęs visus N: N = 64 × 2,76 MB = 177 MB tick'ui (8,8 GB/s) vienoje gijoje | Kopijos per esamą `ThreadPool`; išmatuoti; `invalid` nieko nekainuoja; GPU flow'ams — D2D |
| R10 | Kitas agentas tuo pačiu metu keičia tuos pačius failus (įspėjimai, P6) | Konfliktai | Variklis — nauji failai (`src/engine/`); funkcijų PR tik po jų darbo; E0 (2) derinti, ne lygiagrečiai |
| R11 | Viena fazė varikliui (D7) | Skirtingo gylio įėjimai viename procese gauna didžiausią vėlinimą | Dokumentuota; fazių grupės — vėliau, sąsaja (`expect_ns` vienetui) jau yra |
| R12 | Virtualus laikas nepagauna tikrų futex lenktynių | Klaida tik tikrame vykdyme | V7 testai ant tikro MXL, `fiso_*`, sanitizerių skriptas (`50-sanitizers.sh`) variklio testams |
| R13 | Numatytosios Wr, D, M parinktos iš kodo ir dviejų matavimų lentelių | Gali reikėti keisti po V0b | Reikšmės — parinktys; V20 atskiras; sprendimas K-T2 / K-T3 po matavimo |

## 12. Klausimai autoriui

| # | Klausimas | Rekomendacija | Kiti variantai (kodėl ne) |
|---|---|---|---|
| **K-T1** | Ar palaikyti `--deadline-ms` ilgesnį už periodą (pernešimas, C riba)? | **Taip**, iki `2P − G` (vienas perneštas indeksas). Esami `composite` testai ir talpos kampanijos WSL2 naudoja 30 ms prie 50 Hz | Apkarpyti iki < P (paprasčiau ~150 eil., bet 17–30 ms sutrikimai WSL2 taptų praradimais) |
| **K-T2** | Numatytasis terminas po migracijos | `3·P/4` (15 ms prie 50 Hz — kaip [F]; 30 ms prie 25 Hz — kaip dabar); WSL2 kampanijos nurodo 30 aiškiai | Palikti 30 ms visiems (žemupys apie spragą sužino dvigubai vėliau); `0,6·P` ([A]; mažiau atsargos drebėjimui) |
| **K-T3** | `Wr = P/4`, `M = 3` | Kaip parašyta; patvirtinti po V0b skaičių | `Wr = P/8` (mažiau laikoma, dažniau du paleidimai); `M = 1` (viena lėta akimirka — iškart `Stalled`) |
| **K-T4** | `freeze` be nė vieno gero grain'o (startas) | **`black`** — operatorius, pasirinkęs `freeze`, nori visada galiojančio signalo | INVALID (taip buvo [B] §P5 `Repeat` aprašyme) |
| **K-T5** | Interlaced: po spragos turinys grįžta tik kadro riba (`freeze` / `black`) | **Taip** — vienas palyginimas, nėra kadro iš dviejų laikų | Iškart (paprasčiau; vienas „šukuotas“ kadras) |
| **K-T6** | Variklio paties praleisti indeksai su `freeze` / `black` | **Nepildyti** — lieka INVALID per MXL (M5) | Pildyti iki `slotCount − 2` (kopijos tada, kai jau vėluojama) |
| **K-T7** | Išėjimus kurti prieš įėjimų atsiradimą (`immediate`) kaip numatytąją | **Taip**, po migracijos (V18); to reikia P9 („canvas pildomas iškart“) ir NMOS Sender'iams | Palikti `wait-all` (vienas neatsiradęs įėjimas neleidžia startuoti visiems N) |
| **K-T8** | `--duration` numatyta 0 ir `--strict-exit auto` | **Taip** | Palikti 30 s; exit kodą sieti tik su aiškia parinktimi |
| **K-T9** | Leidimas (a) perrašyti 4 tikro laiko ritmo testus į virtualų laiką (tvirtinimai griežtėja), (b) žymėti likusius `realtime`, (c) pakeisti lūkesčius testuose, kuriuos liečia K3 (V17 sąrašas) | **Taip** visiems trims | Palikti tikrą laiką (A11 lieka) |
| **K-T10** | Viena fazė varikliui (D7) | **Taip** dabar | Fazių grupės viename variklyje (sudėtingiau; nėra naudotojo) |
| **K-T11** | Sugedusio vieneto (K17) išėjimai | **Atlaisvinti flow** — žemupys mato `Detached` | Be galo rašyti INVALID (flow atrodo gyvas) |
| **K-T12** | Upstream siūlymai MXL (siunčia autorius, kaip U1–U4): (a) skaitytojo terminas ant monotoninio laikrodžio; (b) sync grupė, grąžinanti būseną kiekvienam skaitytojui, arba „lauk bet kurio“; (c) `futimens` ne kiekvienam skaitymui | Paruošti `upstream/U5…U7` su įrodymais iš V7 matavimų | Nesiūlyti (ribojimai lieka visiems MXL naudotojams) |
| **K-T13** | Perkėlimo eilė | `gen` → `logo` → izoliacija įjungiama → `composite` → `enc` → `dec` → ritmo gija → `gfxgen`. `gen` pirmas, nes be įėjimų: mažiausias PR, o jis yra visų funkcinių testų šaltinis — skeleto klaida matoma iškart | [A] eilė (`logo` pirmas): vertė anksčiau vienu PR, bet skeletas ir įėjimai tikrinami kartu |
| **K-T14** | Briaunos sutartis: skelbiama **riba** (`commit-bound-ns`), ne išmatuotas vėlavimas; žymų vardai `urn:x-vramflow:tag:…` | **Taip** — deterministiška, testuojama tiksliomis reikšmėmis | Matuoti ir prisitaikyti (slepia blogėjimą) |
| **K-T15** | `verify` lieka už variklio | **Taip** — nepriklausomas orakulas | Perkelti (aštunta kopija dingtų, bet variklio klaida galėtų slėptis abiejose pusėse) |

Blokuoja: **K-T1** (V5), **K-T13** (V8), **K-T9** (V14b, V17), **K-T4…K-T7** (V17, V18), **K-T2 / K-T3** (V10, V20), **K-T8** (V19).

## 13. Kas šiame plane nepatikrinta ir ko jis neapima

- **Niekas nevykdyta:** nei build, nei testai, nei matavimai. §3.2 ir §6 D2 / D3 skaičiai „šiandien“ — išvados iš kodo; pirmas
  tikras matavimas — V0b.
- **M20, M21** ir futex pabudimo vėlavimas (D3 lentelė) — prielaidos, kol neišmatuota V7.
- `proc` p99 = 4,10 ms įrodyme paimtas iš vienos WSL2 lentelės (`composite` G, K = 2, N = 32); kitai apkrovai skaičius bus
  kitas, bet išvada priklauso tik nuo `ro + Wr + proc < P`.
- Ar `mxlFlowReaderGetRuntimeInfo` kopijos dydis (visa `mxlFlowInfo`) juntamas prie 1152 įėjimų — neišmatuota.
- Darbo medis, kurį tuo metu keitė kitas agentas, neskaitytas — remtasi `HEAD`.
- Neapima: A2 (`enc` garso PTS), A4 (pasenę flow'ai), A5 (`gfx.h` ABI — tik pastaba, kad K3 keičia ritmo režimo tekstą),
  A6 (kelios GPU), A8 (build; pastaba: `mxlv_engine` turi savo `cmake/engine.cmake` ir nepriklauso nuo `MXLV_BUILD_FUNC`),
  A15 (darbininkų gyvavimas), A19 CUDA laukimo režimo dalis (P8), garso išėjimų spragų politika (P3).

## 14. Sprendimai (2026-11-12)

Autorius sprendimus delegavo su kriterijumi „praktiškas, universalus pasirinkimas transliacijai; pirmiausia tvirta bazė“.
Priimtos §12 rekomendacijos:

| Klausimas | Sprendimas |
|---|---|
| K-T1 | Pernešimas palaikomas, `--deadline-ms` iki `2P − G` |
| K-T2 | Numatytasis terminas `3·P/4`; WSL2 kampanijos 30 ms nurodo aiškiai |
| K-T3 | `Wr = P/4`, `M = 3`; peržiūrima po V0b skaičių |
| K-T4 | `freeze` be gero grain'o → `black` |
| K-T5 | Interlaced po spragos turinys grįžta kadro riba |
| K-T6 | Variklio paties praleisti indeksai nepildomi |
| K-T7 | `--start immediate` numatyta (po migracijos, V18) |
| K-T8 | `--duration 0` ir `--strict-exit auto` numatyta (V19) |
| K-T10 | Viena fazė varikliui |
| K-T11 | Sugedusio vieneto išėjimai atlaisvinami |
| K-T12 | Upstream siūlymai U5…U7 ruošiami su įrodymais; siunčia autorius |
| K-T13 | Eilė: `gen` → `logo` → izoliacija → `composite` → `enc` → `dec` → ritmo gija → `gfxgen` |
| K-T14 | Briaunos sutartis skelbia ribą, žymos `urn:x-vramflow:tag:…` |
| K-T15 | `verify` lieka už variklio |
| K-T9 | Leista (autorius 2026-11-12: „daryk, kaip atrodo geriausia“): 4 tikro laiko ritmo testai perrašomi į virtualų laiką su griežtesniais tvirtinimais; K3 liečiamų testų lūkesčiai keičiami tame pačiame PR, kiekvienas pakeitimas išvardijamas commit'o žinutėje |
| Apimtis | Bazei dabar: 0–3 etapai, V17 (spragų politika), V18 (startas), V19 (tarnybos režimas), V20, V21 (statistika). **Atidėta iki tinklo etapo:** V22 (adresavimas ir valdymo eilė), V23 (priėmimo kabliukas — su P4), V24 (briaunos laiko sutartis) |

Pastaba: A2 (`enc` garso PTS) jau pataisyta (`344f530`, `AudioTimeline`); griežti įspėjimai įjungti (`34c39d3`), todėl
„0 įspėjimų“ vartai dabar reiškia `-Wall -Wextra -Wpedantic -Werror`. Bazinis ctest — 193.
