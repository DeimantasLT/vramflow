# Tinklas, flow grupės ir NMOS valdymas (VRAMFlow) — reikalavimai

- **Data:** 2026-11-12 (tyrimas atliktas 2026-11-11)
- **Būsena:** juodraštis, **laukia autoriaus sprendimų** (§9). Kodas nekeistas, niekas nekompiliuota ir nepaleista.
- **Repo:** `vramflow`. Planas: `docs/planai/2026-11-12-tinklas-ir-grupes.md` (toliau **[P]**; faktų lentelė — [P] §1).
- **Etapas:** PO bazės sutvirtinimo (`docs/planai/2026-11-11-production-grade.md`, P1–P10). Autoriaus žodžiais: „svarbiausia
  dabar — tvirta bazė“. Šis dokumentas tik planuoja kitą etapą ir pasako, ką bazėje reikia palikti atvirą (§7).
- **Žymėjimas:** **[autorius]** — autoriaus sprendimas ar žodžiai; **[mano]** — dizainerio siūlymas, kurį autorius gali
  atmesti; **[faktas]** — patikrinta 2026-11-11 šaltinyje (nuoroda arba [P] §1 eilutė **F-nn**); **[užrašai]** — iš 2026 m.
  pranešimų užrašų, šiandien nepatikrinta.

## 1. Kam ir kodėl

Šiandien VRAMFlow veikia viename serveryje ir vienoje GPU: `cuda-linear` flow'o rašytojas ir skaitytojas turi būti toje
pačioje mašinoje, nes GPU atmintis skaitytojui perduodama per failo deskriptorių flow'o kataloge
[faktas: `mxl docs/PluggablePayloads.md:90-96`, F-15]. Tikroje gamyboje funkcijos gyvena skirtinguose serveriuose, o
viename serveryje dirba kelios laidos / komandos, kurios neturi viena kitai trukdyti.

**Autoriaus užduotis [autorius]:** suplanuoti (1) RDMA ir (2) TCP sprendimą, kai „GPU galima dalintis tarp skirtingų
serverių“; (3) grupavimą — flow'us galima grupuoti ir dalinti tarp grupių; (4) NMOS standartų integraciją. Sprendimų
kriterijus: praktiškas, universalus pasirinkimas transliacijai — išsaugotas sinchronas ir lūpų sinchronas, nesuspaustas
video ir garsas, profesionalūs standartai, paprasti žemo lygio įrankiai, ant kurių stato developeriai (jokių web aplikacijų).

**Ką realiai reiškia „dalintis GPU tarp serverių“ [mano, faktais paremta]:** nutolusios GPU atminties „prijungti kaip
vietinės“ neįmanoma — nei CUDA, nei RDMA to neduoda; RDMA yra greitas *kopijavimas*, ne bendras atvaizdavimas. Todėl
dalijamasi **flow'ais**, ne atmintimi: flow'as, gyvenantis serverio A GPU, *replikuojamas* į serverio B GPU arba RAM taip,
kad B funkcijoms jis atrodytų vietinis ir turėtų tuos pačius indeksus (taip MXL ir apibrėžia replikaciją
[faktas: `mxl docs/Timing.md:83`, F-13]).

**Sėkmė:** viena komanda serveryje A ir viena serveryje B perkelia pasirinktus flow'us (arba visą šaltinio rinkinį: video +
garsas) per TCP arba RDMA; B pusėje indeksai, flow ID ir turinys tapatūs A pusei; A/V lieka sulygiuoti; nutrūkus ryšiui
niekas negriūva ir skaitytojai nekaba; kelios grupės viename serveryje nemato viena kitos flow'ų, kol jie nepaskelbti;
visa tai (neprivalomai) matoma ir valdoma per NMOS IS-04 / IS-05; pridėtas vėlavimas ir talpa išmatuoti tuo pačiu metodu
kaip `docs/rezultatai/`.

## 2. Tikslai, ne-tikslai, anti-tikslai

**Tikslai**

- G1. `bridge` funkcija (replikatorius): vietinis flow (host arba `cuda-linear`) → nutolęs flow (host arba `cuda-linear`),
  naudojant **MXL Fabrics** (TCP ir RDMA „verbs“ teikėjai), nekuriant savo medijos protokolo.
- G2. TCP veikia visur ir testuojamas vienoje mašinoje (loopback, du domenai). RDMA: iš pradžių per host buferį
  (VRAM → RAM → NIC → RAM → VRAM), tiesiai VRAM → NIC → VRAM (GPUDirect) — kai tai leidžia geležis ir upstream Fabrics.
- G3. Indeksai ir laikas išsaugomi: B pusės grain'o indeksas == A pusės; niekas neperžymima laiku; vieno šaltinio video ir
  garsas lieka sulygiuoti mėginio tikslumu.
- G4. Atsparumas: tempas pagal šaltinį, ribotas pasivijimas po sutrikimo, aiški elgsena dingus ryšiui, automatinis
  atsistatymas, išmatuotas laikrodžių poslinkis tarp serverių.
- G5. Grupės: vardinis flow'ų rinkinys su savo domenu, teisėmis, GPU atminties apskaita ir priėmimo kvota; šaltinio
  rinkinys (video + garsas + key) skaitomas kartu ir sinchroniškai; pasirinkti flow'ai paskelbiami kitoms grupėms —
  tame pačiame serveryje **be kopijos**, tarp serverių — per `bridge`.
- G6. Paprastas žemo lygio valdymas: CLI + failai; viskas veikia be jokio registro ar valdiklio.
- G7. NMOS kaip **neprivalomas** sluoksnis: IS-04 (atradimas), IS-05 + BCP-007-03 (jungimas, įskaitant `bridge`),
  BCP-002-01/-02 (grupavimas, identifikacija), vėliau IS-12 + BCP-008-01/-02 (būsenos stebėjimas).
- G8. Matavimai: pridėtas vėlavimas, CPU branduoliai vienam flow, flow'ų skaičius jungčiai, kopijų kaina — C / H / G
  stiliaus palyginimas (TCP prieš RDMA per host buferį prieš GPUDirect).

**Ne-tikslai (šiame etape)**

- Nutolusios GPU atminties atvaizdavimas kaip vietinės; GPU skaičiavimo dalijimas tarp serverių (nutolęs kernel'ių
  vykdymas, vGPU, laiko dalijimas) — tai planuoklės / K8s reikalas.
- Suspaustas pernešimas tarp serverių (JPEG XS, H.264, SRT) — MXL viduje tik nesuspausta; suspaustas kelias lieka `enc` / `ingest`.
- WAN, internetas, šifravimas duomenų plokštumoje (Fabrics jo neturi [faktas: F-10]) — tik izoliuotas medijos tinklas.
- AWS EFA teikėjas (Fabrics jį turi, bet neturime kur testuoti); multicast.
- Orkestratorius, kuris pats sprendžia, kada ir ką replikuoti (K8s operatorius, kaip Qvest `mxl-k8s` [iš dalies patikrinta,
  F-41]) — pateikiame įrankius, kuriuos toks orkestratorius kviečia.
- Savas NMOS registras ar valdiklis; IS-07, IS-08, IS-11 (žr. §6.4).
- ST 2110 / RTP šliuzas (atskiras darbas; `bridge` neša tik MXL flow'us).

**Anti-tikslai (nesėkmė, net jei „veikia“)**

- Savas tinklo medijos protokolas ten, kur tinka MXL Fabrics; privati Fabrics ar MXL atšaka (fork), kurios negalima grąžinti upstream.
- Replikuotas flow su kitais indeksais, kitu flow ID ar pakeistu turiniu nei šaltinis (bent vienas bitas).
- Laiko perpaišymas: `bridge` niekada nerašo indekso pagal B serverio „dabar“.
- Skaitytojas B pusėje, kuris kabo arba griūva, kai nutrūksta ryšys.
- Tyli degradacija: perėjimas iš GPUDirect į host buferį, iš RDMA į TCP ar paketų metimas be skaitiklio ir būsenos.
- Funkcijos, kurios neveikia be NMOS registro, HTTP serverio ar demono.
- Web aplikacija, GUI ar nuolatinis valdymo demonas kaip privaloma dalis.
- Grupių „izoliacija“, kuri remiasi tik susitarimu dėl flow vardų viename bendrame domene; griežta izoliacija (failų
  teisės arba konteinerių tomai) turi būti įmanoma ir patikrinta testais, ne tik pažadėta.

## 3. Sąvokos

| Sąvoka | Reikšmė šiame dokumente | Kuo paremta |
|---|---|---|
| **Serveris (host)** | Viena OS su savo laikrodžiu ir GPU. NMOS — *Node*. | — |
| **Domenas** | MXL domenas: tmpfs katalogas su `*.mxl-flow/` katalogais. Prieigą valdo Unix teisės. | [faktas: `Architecture.md`] |
| **Domeno ID** | UUID faile `<domenas>/domain_def.json` (`id`, `label`, `description`, `tags`). Taip du konteineriai su skirtingais keliais sutaria, kad tai tas pats domenas. | [faktas: BCP-007-03 v1.0.0, F-30] |
| **Grupė** | Vardinis VRAMFlow vienetas vienai laidai / komandai / nuomininkui: privatus domenas `<šaknis>/<grupė>/`, eksporto domenas `<šaknis>/<grupė>.pub/`, būsenos katalogas `<šaknis>/<grupė>.vramflow/`, Unix grupė (gid), kvota. Izoliacijos, apskaitos ir priėmimo vienetas. | [mano] |
| **Rinkinys (bundle)** | Vieno šaltinio flow'ai, kurie turi būti skaitomi kartu: `Video`, `Audio`, `Key` … NMOS „natural group“: žymė `urn:x-nmos:tag:grouphint/v1.0` = `"<rinkinys>:<rolė>"`. | [faktas: BCP-002-01 v1.0.0, grouphint registras, F-35; MXL reikalauja bent vienos tokios žymės, F-19] |
| **Sync grupė** | MXL API objektas (`mxlFlowSynchronizationGroup`): skaitytojas laukia, kol *visi* jo pridėti flow'ai turi duomenis tam pačiam laikui. Proceso viduje, ne failuose. | [faktas: `mxl/flow.h:923-1019`, F-18] |
| **Eksportas** | Flow'o paskelbimas kitoms grupėms: flow sukuriamas eksporto domene (`.pub`). Tik eksportuoti flow'ai gali būti skaitomi kitos grupės ar perduodami į kitą serverį. | [mano] |
| **Bridge (replikatorius)** | `vramflow-func bridge`: pusė **tx** (šaltinio serveryje; Fabrics *initiator*) ir pusė **rx** (paskirties serveryje; Fabrics *target*). | [faktas: Fabrics modelis, F-09] |
| **TargetInfo / pasiūlymas (offer)** | Fabrics *target* adresas ir atminties raktai, kuriuos tx turi gauti iš rx „šalutiniu kanalu“. Pasiūlymas = TargetInfo + sąsajos parametrai + flow aprašas. | [faktas: `FabricsDeveloperGuide.md`, F-09] |
| **Per host buferį (host-staged)** | Kelias VRAM → RAM → tinklas → RAM → VRAM. Veikia su šiandieniniu Fabrics. | [mano] |
| **GPUDirect** | Kelias VRAM → NIC → VRAM be kopijų į RAM (libfabric `FI_HMEM`, NVIDIA GPUDirect RDMA). | [faktas: F-23, F-24] |
| **Pridėtas vėlavimas** | Laikas nuo grain'o commit'o šaltinio flow'e (A) iki commit'o replikuotame flow'e (B). Matuojamas dalimis, kurių kiekviena — vieno serverio laikrodyje ([P] §3.6), todėl nepriklauso nuo laikrodžių poslinkio. | [mano] |
| **Laikrodžių poslinkis δ** | B serverio TAI − A serverio TAI. Turinio vientisumui nesvarbus, bet B skaitytojams, skaitantiems „dabar“, prideda δ prie vėlavimo. | [faktas: `docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md` §2.3] |

## 4. Ribojimai

- **MXL viduje viskas nesuspausta** (v210 / v210a 10 bitų, float32 garsas). Srautai [faktas: formulė `FabricsBandwidth.md`, F-28]:
  1080i50 arba 1080p25 v210 — 138,24 MB/s = **1,106 Gb/s**; 1080p50 — 2,212 Gb/s; 720p50 — 0,995 Gb/s; 720p25 — 0,498 Gb/s;
  2160p50 — 8,85 Gb/s; 1080i50 v210a — 1,659 Gb/s; garsas 2 kan. 48 kHz float32 — 3,07 Mb/s. Laidui pridėti ~2–5 % antraščių.
  ⇒ **1 GbE netelpa nė vienas HD 50 Hz flow**; per 1 GbE tarp dviejų mašinų galima testuoti tik 720p25 (≈ 50 % jungties).
- **Tik sintetinė arba vieša testinė medžiaga**; jokio neeterinio LRT turinio; niekas nesiunčiama į išorinius servisus.
- **Fabrics šiandien nepriima `cuda-linear` flow'ų** — nei PR #653 medyje, nei `main` [faktas: F-04, F-05]; GPU atminties
  palaikymas Fabrics'e — atviras, neperžiūrėtas PR #654, tik „verbs“ teikėjui [faktas: F-06]. libfabric TCP teikėjas
  `FI_HMEM` neturi iš viso [faktas: F-22]. ⇒ TCP keliui GPU flow'ai **visada** eina per host buferį.
- **Geležis čia:** nešiojamas kompiuteris (WSL2 Ubuntu 24.04, RTX 4070 Laptop 8 GB, be RDMA NIC; WSL2 branduolyje nėra
  soft-RoCE: `CONFIG_RDMA_RXE is not set` [faktas: F-25]); darbo stotis (RTX 3090, 1 GbE, natyvus Linux). Abi GPU — GeForce;
  NVIDIA dokumentuoja GPUDirect RDMA „Tesla ir Quadro“ klasei [faktas: F-24]. ⇒ GPUDirect čia nepatikrinamas.
- **VRAM biudžetas (8 GB plokštė, ~6,5 GB laisva):** replikuotas flow B pusėje yra antras ring'as. Kai abi pusės toje pačioje
  GPU (testas vienoje mašinoje): +26,4 MiB VRAM vienam 1080i50 flow'ui (10 grain'ų), +52,7 MiB — 1080p50; tarpiniai host
  ring'ai: +52,9 MiB RAM vienam 1080i50 flow'ui (po vieną kiekvienoje pusėje). N = 8 × 1080i50: +211 MiB VRAM, +423 MiB RAM.
- **Laikrodis:** keli serveriai turi turėti bendrą laiko šaltinį, kuris „neslysta“ [faktas: `Timing.md:14`, F-14]. Mūsų ribos —
  kaip `laikas-ir-sinchronizacija.md` §1.3: NTP / chrony ≤ 1 ms (L2) pakanka, nes indeksus parenka vienas rašytojas; PTP (L3)
  reikalingas tik kai du serveriai *nepriklausomai* žymi laiku.
- **Fabrics duomenų plokštuma nešifruota ir neautentifikuota**; TargetInfo raktas — ne saugumo priemonė [faktas: F-10].
  Apsauga — tinklo lygiu (atskiras medijos VLAN, ACL).
- **Flow'o teisės:** MXL kiekvienam flow katalogui prideda skaitymo teisę grupei ir „kitiems“ [faktas: `FlowManager.cpp:59-64`,
  F-17]. ⇒ izoliacijos riba yra **domeno katalogas**, ne flow katalogas.
- **`cuda-linear` flow GPU atmintį šiandien gauna tik to paties UID procesas** (PR #653 `6dddc91d`): lizdo `payload.sock`
  leidimai išvedami iš laikino (0700) katalogo dar prieš flow paskelbimą, o `fchmod` po `bind` kelio teisių nekeičia
  [faktas: F-16 — kodas perskaitytas, lizdo elgsena išbandyta; su GPU nevykdyta]. Tai griežčiau, nei skelbia PR aprašas
  („…only to a process that may read the flow directory“). Host flow'ams ribojimo nėra — galioja įprastos failų teisės.
- Licencija Apache-2.0, SPDX antraštės; `git push` daro tik autorius; naujos priklausomybės — tik autoriaus sprendimu
  (tikslus pin, ≥ 30 d. senumo): libfabric, nmos-cpp (§9 K-1, K-9).
- Rašymas ir skaitymas tik per MXL viešą API, PR #653 storage API ir `mxl/fabrics.h`.

## 5. Kaip tai veikia (siūloma; detalės — [P] §3)

### 5.1 A. Transportas tarp serverių

1. **Vienas mechanizmas — MXL Fabrics; du teikėjai:** `tcp` (visur) ir `verbs` (RoCEv2). Fabrics duoda: viso ring'o
   replikaciją su išsaugotais indeksais (grain'o antraštė keliauja kartu), RMA „remote write“, garso (continuous) flow'us,
   1 → N paskirčių, prisijungimo būsenų mašiną su pakartotiniu jungimusi, tik antraštės siuntimą INVALID grain'ams
   [faktas: F-09, F-11, F-12, F-47].
2. **Ko Fabrics neduoda ir prideda VRAMFlow:** (a) `cuda-linear` galai — per host buferį: tx nukopijuoja grain'ą iš VRAM
   į tarpinį host flow, kurį Fabrics replikuoja; rx įkelia į VRAM flow tuo pačiu indeksu; (b) TargetInfo perdavimas:
   failas → mažas TCP „rendezvous“ → (neprivalomai) NMOS IS-05; (c) tempas pagal šaltinį ir ribotas pasivijimas
   (≤ pusė ring'o vienu pliūpsniu — MXL taisyklė [faktas: `Timing.md:89`]); (d) spragų politika B pusėje pagal bazės K3
   (`--on-loss invalid|freeze|black`, kiekvienas indeksas commit'inamas); (e) rinkinio (video + garsas) perdavimas vienu
   procesu; (f) laikrodžių poslinkio matavimas; (g) gyva statistika, priėmimo kontrolė, grupių teisės.
3. **Galų matrica** (visi keturi deriniai — tas pats kodas): host → host (grynas Fabrics, jokių papildomų kopijų),
   `cuda-linear` → host (B pusėje flow'ą skaito CPU funkcijos ir MXL įrankiai), host → `cuda-linear`
   (CPU serveriai maitina GPU serverį), `cuda-linear` → `cuda-linear`.
4. **RDMA:** `--provider verbs` su tuo pačiu host buferio keliu — nulis branduolio kopijų tinkle, bet dvi PCIe kopijos.
   **GPUDirect** (`--stage gpu`) — atskiras, vėlesnis etapas; įjungiamas tik kai Fabrics praneša tokį gebėjimą (PR #654
   ar jo įpėdinis upstream), NIC ir GPU jį palaiko. Kol to nėra — aiški klaida arba aiškiai pažymėtas `stage: host`.
5. **Ką palaikyti iš „GPU dalijimo“ reikšmių [mano; autoriaus sprendimas K-2]:**

   | # | Reikšmė | Siūlymas |
   |---|---|---|
   | R1 | A serverio GPU flow → B serverio GPU (replikacija) | **Taip** — pagrindinis atvejis |
   | R2 | A serverio GPU flow → B serverio RAM | **Taip** — pigu, reikalinga CPU funkcijoms ir MXL įrankiams |
   | R3 | Keli serveriai (su GPU ar be) maitina vieną GPU serverį, rezultatai grįžta atgal | **Taip** — R1 / R2 abiem kryptimis + priėmimas pagal grupę |
   | R4 | Vienas šaltinis → keli serveriai (1 → N) | **Taip, antrame žingsnyje** — Fabrics tai turi |
   | R5 | Nutolusi GPU atmintis kaip vietinė | **Ne** — neįmanoma |
   | R6 | GPU skaičiavimo dalijimas tarp serverių | **Ne** — ne šio įrankio lygis |

### 5.2 B. Grupės ir dalijimasis

1. **Du lygiai, kurių nereikia painioti [mano]:**
   - **Rinkinys** = vieno šaltinio flow'ai (video, garsas, key). Standartinis pagrindas — BCP-002-01 „natural group“ ir
     `grouphint` žymė, kurią VRAMFlow jau rašo. Skaitomas kartu per MXL sync grupę.
   - **Grupė** = operatoriaus apibrėžtas vienetas (laida, komanda). BCP-002-01 tokių grupių **neapima**
     („group may be defined by a user … not in scope“ [faktas: F-35]) ⇒ tai VRAMFlow lygio sąvoka, atvaizduota į MXL domeną.
2. **Grupė = domenas.** `<šaknis>/<grupė>/` (privatus), `<šaknis>/<grupė>.pub/` (eksportas), `<šaknis>/<grupė>.vramflow/`
   (nuoma, kvota, jungčių failai, statistika — kaip bazės K8). Kiekvienas domenas turi `domain_def.json` su stabiliu UUID.
   Atmesta: vienas domenas visiems + tik žymės (nėra izoliacijos); domenas kiekvienam rinkiniui (šimtai domenų, sync grupė
   tarp domenų sudėtingesnė).
3. **Izoliacija — du lygiai [mano; K-6]:**
   - **Organizacinė** (numatyta; visos grupės tuo pačiu UID — kūrimo mašina, viena paslaugos paskyra, arba konteineriai,
     kur ribą nubrėžia tai, kurie domenų katalogai į konteinerį atvaizduoti): atskiri domenai, kvotos, nuomos; funkcija
     paliečia kitos grupės flow tik tada, kai jai aiškiai nurodytas kitas domenas. Zero-copy dalijimasis veikia visiems flow'ams.
   - **Griežta** (atskiras Unix naudotojas ir gid kiekvienai grupei; domeno katalogas `0750`): host flow'ai saugomi failų
     teisėmis; privatūs `cuda-linear` flow'ai kitam UID neprieinami iš principo (§4). Kaina: `cuda-linear` flow **eksportuoti
     kitam UID šiandien neįmanoma** — kol upstream nepataisyta (K-13), toks eksportas daromas per host flow
     (vietinis `bridge` `cuda-linear` → host į `.pub` domeną; skaitytojas, jei reikia, pats įkelia į savo GPU atmintį).
4. **Apskaita ir priėmimas pagal grupę:** VRAM vienai grupei = jos domenų `cuda-linear` ring'ų suma (dydis įrašytas
   `payload.cuda.json` [faktas: F-15] — GPU nereikia) + funkcijų rezervacijos (P4 knyga). Kvota — `group.json`.
   Priėmimas tikrina **abi** ribas: grupės kvotą ir visos GPU išmatuotą talpą.
5. **Dalijimasis tame pačiame serveryje — be kopijos:** eksportuotą flow kita grupė skaito tiesiai iš `.pub` domeno
   (host flow — `mmap`; `cuda-linear` — tas pats VRAM ring'as per `payload.sock`, kai rašytojas ir skaitytojas dirba tuo
   pačiu UID — žr. 3 punktą). Įėjimas nurodomas MXL adresu
   `mxl:///<domeno-kelias>?id=<uuid>` [faktas: `Addressability.md`, F-20]. Eksportas sprendžiamas kuriant flow
   (`--export`). Atmesta kaip numatytasis kelias: bind-mount / simbolinė nuoroda į kitą domeną (reikia root arba konteinerio
   vykdyklės; tinka K8s, lieka kaip galimybė); kopija per vietinį `bridge` (dvigubas VRAM) — tik atsarginis variantas.
6. **Dalijimasis tarp serverių:** `bridge` iš A grupės `.pub` domeno į B grupės domeną; flow ID ir `grouphint` tie patys.
   Taisyklė viena abiem atvejams: **grupę palieka tik eksportuoti flow'ai**.
7. **Flow ID:** šiandien ID išvedamas iš rolės ir eilės numerio (`4d584c47-00RR-4000-8000-…`, `src/func/flows.cpp:17-21`) —
   du serveriai su tuo pačiu `gen` pagamintų vienodus ID ir susidurtų viename domene. Siūloma: deterministinis UUIDv5 iš
   (grupė, rinkinys, rolė) — unikalus tarp grupių, stabilus tarp paleidimų (to reikia ir NMOS) [mano; K-15].
8. **Valdymo paviršius:** `vramflow-ctl group|bundle|flow|connect …` + failai būsenos kataloge; viską galima padaryti ir
   `mkdir` / `chgrp` / teksto redaktoriumi — įrankis tik patogumas ir patikra.

### 5.3 D. NMOS valdymo plokštuma (neprivaloma)

1. **Atskiras procesas (sidecar) `vramflow-nmos` [mano; K-9]:** skaito tai, ką VRAMFlow ir taip turi failuose (domenai,
   `flow_def.json`, `domain_def.json`, nuomos, jungčių failai, P7 statistikos eilutės) ir kalba NMOS. Medijos funkcijose
   nėra HTTP. Be sidecar'o viskas veikia kaip iki šiol.
2. **IS-04:** serveris = Node; funkcijos paleidimas (`run_id`) = Device; kiekvienas rašomas flow = Source + Flow + Sender
   (`transport: urn:x-nmos:transport:mxl`, `interface_bindings: []`, `manifest_href: null`); kiekvienas funkcijos įėjimas =
   Receiver su BCP-004-01 gebėjimais [faktas: BCP-007-03 v1.0.0, F-30]. `VideoFormat` → Flow laukai; `grouphint` — kaip faile.
   Numatytas režimas — **registruotas** (registro adresas nurodomas parinktimi arba randamas per DNS-SD); peer-to-peer
   (mDNS) — tik mažam stendui viename tinklo segmente.
3. **IS-05 + BCP-007-03:** parametrai `mxl_domain_id`, `mxl_flow_id`; `master_enable` paleidžia / stabdo skaitymą ar rašymą.
   BCP-007-03 v1.0.0 **neapibrėžia** nei Fabrics, nei TargetInfo, nei GPU atminties tipo — tik pasako, kad tas pats
   `mxl_flow_id` gali būti skirtinguose domenuose, nes flow'ai replikuojami [faktas: F-30]. `bridge` valdomas tais pačiais
   ištekliais + gamintojo parametrais `ext_…` (schema juos leidžia [faktas: F-31]): rx pusės Sender'io `/active` skelbia
   TargetInfo, valdiklis jį įrašo į tx pusės Receiver'į — t. y. valdiklis tampa „šalutiniu kanalu“ vietoje failo.
   AMWA IN-001 (reikalavimai MXL valdymo API: nustatyti / gauti flow ir domeno ID, idempotentiški start / stop, pasiekiamų
   domenų sąrašas, būsena `started` / `stopped` [faktas: F-34]) dengiami tais pačiais IS-05 parametrais ir `master_enable`.
4. **Grupės:** rinkinys → `grouphint` ant Sender / Receiver (BCP-002-01); grupė → domeno ID (`mxl_domain_id`) ir
   `domain_def.json` žymės; Node / Device — BCP-002-02 gamintojo, produkto ir egzemplioriaus žymės.
5. **Stebėjimas:** P7 statistika ir P9 aliarmai → BCP-008-01 `NcReceiverMonitor` / BCP-008-02 `NcSenderMonitor` per IS-12
   (ryšio, sinchronizacijos, srauto būsenos; vėluojančių / prarastų skaitikliai) [faktas: F-37].
6. **Saugumas:** BCP-003-01 (TLS) ir IS-10 (autorizacija) taikomi tik sidecar'o HTTP / WebSocket; žemo lygio įrankiai
   tinklo API neturi. Autorizacijos serveris, sertifikatai, valdiklis — objekto (facility) reikalas.
7. **Praleidžiama:** IS-11 (suderinamumo valdymas reguliuojamiems siuntėjams — MXL flow formatas nekintamas), IS-08
   (garso kanalų žemėlapis — kol nėra garso maršrutizavimo funkcijos), IS-07 (įvykiai / tally — tik jei P9 aliarmams
   prireiks įvykių; būsenai užtenka BCP-008), IS-14, IS-13 (dar ne leidimas).

## 6. User stories ir priėmimo kriterijai

Bendros sąlygos visiems kriterijams, jei nepasakyta kitaip: sintetinis `gen` šaltinis (barkodas = indeksas, CRC etalonas),
formatas 1080i50 (ring'as 10 grain'ų, laukas 20 ms), trukmė 60 s po 5 s apšilimo, „viena mašina“ = du domenai ir
`127.0.0.1`. „Pateikta“ = commit'inta be `MXL_GRAIN_FLAG_INVALID`. Išėjimo kodai: 4 — įėjimo nesulaukta, 5 — priėmimas
atmetė (kaip bazėje); nauji: **7** — transportas neprieinamas, **8** — šalys nesuderinamos.

### 6.1 A — transportas

---

# [N1] Pernešimas per TCP: developeris gali perkelti flow į kitą serverį ir gauti tą patį flow

# Kontekstas (Context)
Kaip medijos funkcijų developeris,
Noriu viena komanda kiekvienoje pusėje perkelti flow (host arba `cuda-linear`) per paprastą TCP tinklą į kito serverio RAM arba GPU,
Kad funkcijos kitame serveryje jį skaitytų kaip vietinį, su tais pačiais indeksais ir tuo pačiu turiniu.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. GPU → GPU per TCP (teigiamas)
Duota (Given): domene A veikia `gen --storage cuda-linear --flows 4`; domene B — `bridge --side rx --storage cuda-linear`; `bridge --side tx --provider tcp`
Kai (When): sistema veikia 60 s
Tada (Then): `verify --mode gen` domene B: 0 turinio (CRC) klaidų, kiekvieno grain'o barkodas == jo indeksas, pateikta ≥ 99,8 % indeksų (≥ 2 994 iš 3 000 kiekvienam flow)
Ir (And): B pusės flow ID ir `flow_def.json` baitas į baitą sutampa su A pusės; pridėtas vėlavimas p99 ≤ 10 ms (privaloma riba ≤ 20 ms = 1 laukas; išmatuota reikšmė užrašoma)

[ ] 2. Visi saugyklų deriniai (teigiamas)
Duota (Given): tas pats bandymas su deriniais host → host, `cuda-linear` → host, host → `cuda-linear`
Kai (When): kiekvienas veikia 60 s
Tada (Then): kiekvienam galioja 1 kriterijaus ribos
Ir (And): host → host derinyje skaitikliai `d2h_bytes == 0` ir `h2d_bytes == 0` (jokių papildomų kopijų)

[ ] 3. Alfa ir formatai (kraštinis)
Duota (Given): `v210a` flow (dvi plokštumos) ir formatai 1080p25, 1080p50
Kai (When): perduodama 20 s
Tada (Then): abi plokštumos (fill ir key) sutampa su šaltiniu (0 skirtingų baitų tikrintuose grain'uose); `grain_rate` ir rastras B pusėje tapatūs

[ ] 4. Paskirties nėra (neigiamas)
Duota (Given): rx pusė nepaleista
Kai (When): paleidžiu tx su `--input-wait-ms 3000`
Tada (Then): tx baigiasi kodu 4 per ≤ 3,5 s, pranešime — adresas ir prievadas
Ir (And): domene B nesukurta nė vieno flow; šaltinio flow'ai ir jų skaitytojai nepaliesti (`late == 0`)

[ ] 5. Flow jau užimtas (neigiamas)
Duota (Given): domene B jau yra gyvas to paties flow ID rašytojas
Kai (When): paleidžiu rx tam flow
Tada (Then): rx baigiasi kodu ≠ 0, pranešime — savininko `run_id` (bazės P6 nuoma)
Ir (And): esamas flow nepaliestas: jo skaitytojas per tą laiką nepraranda nė vieno grain'o

[ ] 6. Nesuderinamos šalys (neigiamas)
Duota (Given): tx ir rx su skirtingu teikėju arba skirtingu ring'o grain'ų skaičiumi
Kai (When): bandoma jungtis
Tada (Then): abi pusės baigiasi kodu 8 per ≤ 2 s; pranešime abi reikšmės; nė vienas grain'as neperduotas

# Papildoma informacija (Optional)
Techninės pastabos: Fabrics `tcp` teikėjas, RMA; `cuda-linear` galai — per tarpinį host flow ([P] §3.2).
Rizikos / Apribojimai: 1 GbE netelpa HD 50 Hz — tarp dviejų mašinų šis kriterijus tikrinamas 720p25.
Instrukcijos: README skyrius „Tarp serverių“.

---

# [N2] Rinkinys: operatorius gali perkelti šaltinį (video + garsas) neprarasdamas lūpų sinchrono

# Kontekstas (Context)
Kaip transliacijos inžinierius,
Noriu perkelti visą šaltinio rinkinį vienu veiksmu,
Kad kitame serveryje video ir garsas liktų sulygiuoti taip pat tiksliai kaip šaltinyje.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. A/V mėginio tikslumu (teigiamas)
Duota (Given): rinkinys `src0` = 1080i50 video + 2 kan. 48 kHz garsas su žymėmis (blyksnis ir 1 mėginio impulsas kas sekundę); `bridge --bundle src0`
Kai (When): perduodama 60 s
Tada (Then): B pusėje kiekvienai sekundei impulso mėginys nuo blyksnio lauko pradžios nutolęs 0 mėginių (privaloma riba: |A/V| ≤ 1 laukas = 20 ms)
Ir (And): garso mėginiai B pusėje bitas į bitą lygūs A pusės; video — kaip N1.1

[ ] 2. Rinkinys skaitomas kartu (teigiamas)
Duota (Given): B pusėje skaitytojas atidaro rinkinį vardu (`BundleIn`, MXL sync grupė)
Kai (When): skaito 60 s
Tada (Then): kiekviename žingsnyje visi nariai turi tą patį laiką (video indeksas i, garso mėginiai nuo `sample_begin(i)`) arba žingsnis užskaitomas `bundle_incomplete`; `bundle_incomplete ≤ 0,2 %`

[ ] 3. Vienas narys dingo (neigiamas)
Duota (Given): garso rašytojas A pusėje sustabdomas 5 s ir paleidžiamas vėl
Kai (When): rinkinys toliau perduodamas
Tada (Then): video perduodamas be spragų; garso B pusėje tais 5 s nėra (spraga, ne nuliai); `BundleIn` tuos žingsnius žymi `bundle_incomplete`
Ir (And): garsui grįžus A/V poslinkis vėl 0 mėginių; nė vienas procesas nebaigė darbo

[ ] 4. Neegzistuojantis rinkinys (neigiamas)
Duota (Given): rinkinio vardas, kurio domene nėra
Kai (When): paleidžiu tx su `--input-wait-ms 2000`
Tada (Then): kodas 4 per ≤ 2,5 s, pranešime rinkinio vardas ir domene rasti rinkiniai

---

# [N3] Ryšio nutrūkimas: operatorius gali pasikliauti, kad nutrūkus jungčiai niekas negriūva ir niekas nekaba

# Kontekstas (Context)
Kaip operatorius,
Noriu, kad dingus tinklui tarp serverių replikuoti flow'ai elgtųsi kaip šaltinis be signalo ir patys atsistatytų,
Kad kitos funkcijos tame serveryje toliau dirbtų, o aš matyčiau, kas ir kiek laiko buvo prarasta.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. 5 s nutrūkimas (neigiamas)
Duota (Given): veikiantis `bridge`, 1 flow, rx su `--on-loss invalid` (numatyta)
Kai (When): jungtis nutraukiama 5 s ir atstatoma
Tada (Then): nė vienas procesas nebaigia darbo; B pusėje kiekvienas nutrūkimo indeksas commit'intas su `MXL_GRAIN_FLAG_INVALID` ne vėliau kaip per `--deadline-ms` (30 ms) po savo laiko; skaitytojas su 40 ms timeout'u negauna nė vienos `TOO_EARLY` klaidos
Ir (And): statistikoje `reconnects == 1`, `link_down_ms` = 5 000 ± 500, būsena `Unhealthy` nutrūkimo metu

[ ] 2. Atsistatymas (teigiamas)
Duota (Given): 1 kriterijaus būsena
Kai (When): jungtis atstatyta
Tada (Then): galiojantys grain'ai vėl pateikiami per ≤ 2 s; po atsistatymo vienu pliūpsniu atsiunčiama ≤ 5 grain'ai (pusė ring'o); indeksai niekada nemažėja
Ir (And): per 5 s pridėtas vėlavimas p99 grįžta į ≤ (reikšmė prieš nutrūkimą + 2 ms)

[ ] 3. Paskirties procesas paleistas iš naujo (neigiamas)
Duota (Given): veikiantis `bridge`; rx gauna `kill -9` ir po 3 s paleidžiamas vėl
Kai (When): rx vėl klauso (naujas TargetInfo)
Tada (Then): tx pats gauna naują pasiūlymą (rendezvous arba pasikeitęs failas) ir tęsia per ≤ 5 s; `reconnects == 1`; tx procesas tas pats (PID nepakito)

[ ] 4. Per lėta jungtis (neigiamas)
Duota (Given): jungties pralaidumas apribotas iki 50 % flow'o srauto
Kai (When): veikia 60 s
Tada (Then): tx nekaupia eilės: `skipped > 0`, pridėtas vėlavimas p99 ≤ 5 laukai (100 ms — pusė ring'o), proceso RSS pokytis ≤ 5 %
Ir (And): B pusėje praleisti indeksai — INVALID, ne seni kadrai; būsena `PartiallyHealthy`

[ ] 5. Vėluojantis grain'as po spragos (kraštinis)
Duota (Given): indeksas i B pusėje jau pažymėtas INVALID
Kai (When): jo duomenys atkeliauja vėliau
Tada (Then): jis nebecommit'inamas antrą kartą, užskaitomas `too_late`; vėlesni indeksai nepaveikti

---

# [N4] RDMA: inžinierius gali perjungti tą patį pernešimą į RoCEv2 ir pamatyti, ką tai duoda

# Kontekstas (Context)
Kaip sistemų inžinierius,
Noriu tą pačią komandą paleisti su RDMA teikėju,
Kad CPU apkrova ir vėlavimas sumažėtų ten, kur yra RDMA tinklo plokštės, o ten, kur jų nėra, gaučiau aiškų atsakymą.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Verbs per host buferį (teigiamas)
Duota (Given): serveris su RDMA įrenginiu (funkciniam testui tinka soft-RoCE), `--provider verbs`
Kai (When): perduodama 60 s
Tada (Then): galioja N1.1 turinio ir indeksų ribos
Ir (And): rezultatuose yra eilutė „tcp prieš verbs“: pridėtas vėlavimas p50 / p99, CPU branduolių vienam flow abiejose pusėse

[ ] 2. Nėra RDMA įrenginio (neigiamas)
Duota (Given): mašina be RDMA įrenginio
Kai (When): paleidžiu su `--provider verbs`
Tada (Then): kodas 7 per ≤ 1 s; pranešime — rastų sąsajų sąrašas (teikėjas ir adresas)
Ir (And): tylaus perėjimo į TCP nėra; su `--provider auto` pasirenkamas `tcp` ir tai parašoma stderr bei statistikoje (`provider: "tcp"`)

[ ] 3. Tikra RoCE jungtis (teigiamas; tik lab'e)
Duota (Given): du serveriai su ≥ 25 GbE RoCEv2
Kai (When): perduodami N flow'ų 1080p50, N didinamas iki pirmo nesėkmingo lygio
Tada (Then): užrašytas didžiausias N be praradimų ir pridėtas vėlavimas p99; lūkestis [užrašai]: ~2 ms vienam flow ant 25 GbE; riba PASS: p99 ≤ 5 ms

---

# [N5] GPUDirect: inžinierius gali įjungti VRAM → NIC → VRAM ten, kur tai palaikoma, ir niekada negauna to tyliai išjungto

# Kontekstas (Context)
Kaip sistemų inžinierius,
Noriu, kad tarp dviejų GPU serverių kadrai keliautų be kopijų į RAM, kai geležis ir programinė įranga tai leidžia,
Kad vėlavimas ir CPU / PCIe apkrova būtų mažiausi, o kai neleidžia — tiksliai žinočiau kodėl.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Be host kopijų (teigiamas; tik tinkamoje geležyje)
Duota (Given): Fabrics su GPU atminties gebėjimu, RDMA NIC ir GPU, palaikanti GPUDirect RDMA, abiejose pusėse; `--stage gpu`
Kai (When): perduodama 60 s
Tada (Then): `d2h_bytes == 0` ir `h2d_bytes == 0` abiejose pusėse; turinys ir indeksai — kaip N1.1
Ir (And): pridėtas vėlavimas p99 ≤ to paties bandymo su `--stage host` reikšmė

[ ] 2. Nepalaikoma (neigiamas)
Duota (Given): GeForce GPU, TCP teikėjas arba Fabrics be GPU atminties gebėjimo
Kai (When): paleidžiu su `--stage gpu`
Tada (Then): kodas 7 per ≤ 2 s; pranešime įvardyta, ko trūksta (Fabrics gebėjimas / teikėjas / atminties registracijos klaida su libfabric kodu)
Ir (And): su `--stage auto` dirbama per host buferį, o statistikoje ir pirmoje stderr eilutėje — `stage: "host"` ir priežastis

---

# [N6] GPU serveris kaip bendra paslauga: keli serveriai gali maitinti vieną GPU, o perkrova atmetama, ne paskirstoma

# Kontekstas (Context)
Kaip operatorius,
Noriu, kad keli serveriai siųstų flow'us į vieną GPU serverį, kuriame dirba GPU funkcijos,
Kad brangi GPU būtų naudojama bendrai, bet jau priimti srautai nenukentėtų, kai kas nors atsiunčia per daug.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Du siuntėjai, viena GPU (teigiamas)
Duota (Given): GPU serveryje rx dviem grupėms; du tx (imituojami du serveriai), po 2 flow'us, talpa pakankama
Kai (When): veikia 60 s, o GPU serveryje `logo` apdoroja visus 4 replikuotus flow'us
Tada (Then): visi 4 `logo` išėjimai: 0 turinio klaidų, `late == 0`; kiekvienos grupės flow'ai — savo domene

[ ] 2. Viršyta grupės kvota (neigiamas)
Duota (Given): grupės kvota — 2 flow'ų VRAM; siuntėjas prašo 3
Kai (When): rx bando priimti trečią
Tada (Then): trečias atmetamas kodu 5; pranešime — `group`, ribojantis išteklius (`vram`), prašyta / užimta / leidžiama baitais; tx gauna tą pačią priežastį
Ir (And): pirmi 2 flow'ai ir kitos grupės flow'ai per tą laiką nepraranda nė vieno grain'o

[ ] 3. Viršyta GPU talpa, kvotos pakanka (neigiamas)
Duota (Given): dvi grupės, kiekvienos kvota 3 flow'ai, išmatuota GPU talpa — 4
Kai (When): abi prašo po 3
Tada (Then): priimami pirmi 4 (0,85 atsarga taikoma kaip bazės P4), kiti atmetami su `scope: "gpu"`

---

# [N7] Laikrodžiai: inžinierius mato laiko poslinkį tarp serverių ir gali neleisti dirbti su blogu laiku

# Kontekstas (Context)
Kaip sistemų inžinierius,
Noriu, kad `bridge` išmatuotų ir parodytų laikrodžių poslinkį tarp serverių,
Kad sinchronizacijos problemas pamatyčiau anksčiau nei žiūrovas, o blogai sinchronizuotas serveris nepatektų į grandinę.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Poslinkis matuojamas (teigiamas)
Duota (Given): du galai toje pačioje mašinoje (tikrasis poslinkis 0)
Kai (When): veikia 60 s
Tada (Then): kiekvienoje statistikos eilutėje `clock_offset_us` ir `path_delay_us`; |`clock_offset_us`| ≤ 200
Ir (And): B pusės skaitytojas, skaitantis „dabar“ su 30 ms terminu: `late == 0`

[ ] 2. Didelis poslinkis (neigiamas)
Duota (Given): imituotas 2 s poslinkis (testinis kabliukas matavime)
Kai (When): veikia be `--max-clock-offset-ms`
Tada (Then): kas 10 s įspėjimas stderr; `clock_offset_us` = 2 000 000 ± 1 000; turinys ir indeksai B pusėje vis tiek tapatūs (N1.1); spragų laikmatis neskelbia klaidingų praradimų (`lost == 0`)
Ir (And): su `--max-clock-offset-ms 5` tx baigiasi kodu 8 per ≤ 5 s, pranešime išmatuota ir leidžiama reikšmė

### 6.2 B — grupės

---

# [G1] Grupės kūrimas: operatorius gali sukurti grupę ir matyti, kas joje yra

# Kontekstas (Context)
Kaip operatorius,
Noriu viena komanda sukurti grupę su savo domenu, teisėmis ir kvota ir bet kada pamatyti jos flow'us ir rinkinius,
Kad kiekviena laida turėtų savo aiškią erdvę, o aš neturėčiau atsiminti UUID.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Sukūrimas (teigiamas)
Duota (Given): tuščia šaknis `/dev/shm/vf`
Kai (When): `vramflow-ctl group create --root /dev/shm/vf --name studio1 --gid <G> --vram-mb 2048`
Tada (Then): yra `studio1/` (režimas 0750, grupė G), `studio1/domain_def.json` (galiojantis pagal BCP-007-03 schemą: `id`, `label`, `description`, `tags`), `studio1.vramflow/group.json`
Ir (And): `vramflow-ctl group list` rodo `studio1` su domeno ID; pakartota ta pati komanda — kodas 0 ir niekas nepakinta (idempotentiška; `id` tas pats)

[ ] 2. Sąrašai (teigiamas)
Duota (Given): grupėje veikia `gen` su video ir garsu, rinkinys `cam1`
Kai (When): `vramflow-ctl bundle list --group studio1`
Tada (Then): išvestyje `cam1` su nariais `Video` ir `Audio`, jų flow ID, formatu, saugykla (`host` / `cuda-linear`) ir savininko `run_id`; `--json` duoda tą patį kaip vieną JSON objektą

[ ] 3. Blogas vardas (neigiamas)
Duota (Given): vardai `Studio 1`, `a:b`, `` (tuščias), 64 simbolių eilutė, `x.pub`
Kai (When): bandau kurti
Tada (Then): kiekvienu atveju kodas 2, pranešime leidžiamų simbolių taisyklė; failų sistemoje nieko nesukurta

[ ] 4. Konfliktas (neigiamas)
Duota (Given): `studio1` jau yra su kita `--gid` ar kvota
Kai (When): kuriu dar kartą su kitais parametrais
Tada (Then): kodas ≠ 0, pranešime abu parametrų rinkiniai; esama grupė nepakeista (keitimui — atskira `group set`)

[ ] 5. Trynimas su gyvais savininkais (neigiamas)
Duota (Given): grupėje veikia bent viena funkcija (gyva nuoma)
Kai (When): `vramflow-ctl group remove --name studio1`
Tada (Then): kodas ≠ 0, išvardyti gyvi `run_id`; nieko neištrinta

---

# [G2] Rinkiniai: developeris gali pažymėti šaltinio flow'us kaip rinkinį ir skaityti juos kartu

# Kontekstas (Context)
Kaip medijos funkcijų developeris,
Noriu, kad vieno šaltinio video, garsas ir key būtų standartiniu būdu pažymėti kaip vienas rinkinys,
Kad mano funkcija ir bet kuris NMOS valdiklis juos rastų ir jungtų kartu.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Žymė pagal standartą (teigiamas)
Duota (Given): `gen --bundle cam1 --audio 1`
Kai (When): sukuriami flow'ai
Tada (Then): `flow_def.json` žymė `urn:x-nmos:tag:grouphint/v1.0` = `["cam1:Video"]` ir `["cam1:Audio"]`; `mxl-info` flow'us atidaro be klaidų
Ir (And): flow ID tarp dviejų paleidimų su tais pačiais (grupė, rinkinys, rolė) yra tas pats

[ ] 2. Skaitymas kartu (teigiamas)
Duota (Given): funkcija atidaro `--bundle cam1`
Kai (When): skaito 60 s
Tada (Then): kiekvienas žingsnis gauna video indeksą i ir garso mėginius `[sample_begin(i), sample_begin(i+1))`; `bundle_incomplete == 0`

[ ] 3. Dvi vienodos rolės (neigiamas)
Duota (Given): rinkinyje `cam1` jau yra rolė `Video`
Kai (When): antras rašytojas kuria flow su `cam1:Video`
Tada (Then): kodas ≠ 0 prieš sukuriant flow; pranešime esamo flow ID ir savininkas (rolė rinkinyje unikali — registro taisyklė)

[ ] 4. Dvitaškis varde (neigiamas)
Duota (Given): `--bundle "cam:1"`
Kai (When): paleidžiu
Tada (Then): kodas 2; pranešime, kad `:` rezervuotas

---

# [G3] Izoliacija: operatorius gali būti tikras, kad viena grupė nemato ir neišstumia kitos

# Kontekstas (Context)
Kaip operatorius, viename GPU serveryje laikantis kelias laidas,
Noriu, kad grupės viena kitos flow'ų nematytų ir negalėtų užimti viena kitos GPU atminties dalies,
Kad vienos komandos klaida ar perkrova nepaveiktų kitos eterio.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Svetimas skaitytojas, griežta izoliacija (neigiamas)
Duota (Given): grupės `studio1` (atskiras UID, gid G1, domenas 0750) privačiame domene yra host flow ir `cuda-linear` flow; naudotojas u2 nepriklauso G1
Kai (When): u2 bando skaityti abu
Tada (Then): host flow — skaitytojo kūrimas nepavyksta (prieigos klaida); `cuda-linear` — `MapSlots` nepavyksta ir deskriptorius neišduodamas
Ir (And): rašytojas ir teisėti skaitytojai per tą laiką nepraranda nė vieno grain'o

[ ] 1a. Kita grupė nepasiekiama netyčia, organizacinė izoliacija (neigiamas)
Duota (Given): abi grupės tuo pačiu UID; rinkinys `cam1` yra tik `studio1` domene
Kai (When): paleidžiu `logo --group studio2 --bundle cam1 --input-wait-ms 2000`
Tada (Then): kodas 4 per ≤ 2,5 s; funkcija neskaito `studio1` flow'ų (jos nuomoje ir statistikoje nėra `studio1` domeno); skaityti kitos grupės flow galima tik aiškiai nurodžius jo eksporto adresą `mxl:///…`

[ ] 2. Savas skaitytojas kitame procese ar konteineryje (teigiamas)
Duota (Given): tas pats UID kaip rašytojo, kitas procesas (arba konteineris su atvaizduotu grupės domenu, be bendros IPC / PID erdvės)
Kai (When): skaito `cuda-linear` flow 60 s
Tada (Then): `MapSlots` pavyksta; 0 CRC klaidų

[ ] 2a. Kitas UID toje pačioje Unix grupėje (neigiamas; žinomas MXL PR #653 ribojimas)
Duota (Given): naudotojas u1 ≠ rašytojo UID, su G1 kaip pagrindine grupe; MXL be upstream pataisos
Kai (When): skaito `cuda-linear` flow
Tada (Then): skaitytojas baigiasi kodu ≠ 0 per ≤ `--input-wait-ms`; pranešime — kad `cuda-linear` flow prieinamas tik rašytojo UID ir kad reikia host eksporto; host flow tame pačiame domene tas pats naudotojas skaito be klaidų
Ir (And): kai MXL turės pataisą, tas pats testas turi praeiti kaip teigiamas (0 CRC klaidų per 60 s) — testas pats praneša, kuri elgsena aptikta

[ ] 3. Apskaita pagal grupę (teigiamas)
Duota (Given): `studio1` — 4 × 1080i50 `cuda-linear` flow'ai, `studio2` — 2
Kai (When): `vramflow-ctl group usage`
Tada (Then): `studio1` ring'ų VRAM = 4 × ring'o dydis ± 2 MiB kiekvienam (lygiavimo paklaida), `studio2` = 2 ×; suma sutampa su `nvidia-smi` pokyčiu ± 10 %

[ ] 4. Kvota (neigiamas)
Duota (Given): `studio1` kvota išnaudota, `studio2` — ne
Kai (When): `studio1` paleidžia dar vieną `cuda-linear` rašytoją su `--admission enforce`
Tada (Then): kodas 5 su `group: "studio1"`; flow nesukurtas
Ir (And): tuo pačiu metu `studio2` naujas rašytojas priimamas

---

# [G4] Dalijimasis be kopijos: grupė gali paskelbti flow kitai grupei tame pačiame serveryje

# Kontekstas (Context)
Kaip laidos inžinierius,
Noriu pasirinktą flow (pvz., programos išėjimą) paskelbti kitai grupei be jokio kopijavimo,
Kad antra laida jį naudotų be papildomos GPU atminties ir be papildomo vėlavimo.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Zero-copy skaitymas (teigiamas)
Duota (Given): `studio1` rašytojas su `--export` (flow `studio1.pub` domene, `cuda-linear`); `studio2` funkcija tuo pačiu UID (organizacinė izoliacija arba konteineriai) su `--input mxl:///dev/shm/vf/studio1.pub?id=<uuid>`
Kai (When): skaito 60 s
Tada (Then): 0 CRC klaidų; GPU atminties naudojimas atidarius skaitytoją padidėja ≤ 2 MiB (antro ring'o nėra); `d2h_bytes == 0`
Ir (And): skaitymo vėlavimas p99 nuo to paties flow skaitymo grupės viduje skiriasi ≤ 0,2 ms

[ ] 2. Tas pats host flow'ui (teigiamas)
Duota (Given): eksportuotas host flow; `studio2` — kitas UID (griežta izoliacija), turintis eksporto domeno skaitymo teisę
Kai (When): `studio2` skaito
Tada (Then): 0 CRC klaidų; `/dev/shm` užimtumas nepadidėja (± 1 MiB)

[ ] 2a. GPU flow kitam UID — per host eksportą (teigiamas atsarginis kelias)
Duota (Given): griežta izoliacija; `studio1` turi `cuda-linear` flow ir paleidžia vietinį `bridge` (`cuda-linear` → host) į `studio1.pub`
Kai (When): `studio2` (kitas UID) skaito eksportuotą host flow 60 s
Tada (Then): 0 CRC klaidų, indeksai tapatūs šaltiniui; pridėtas vėlavimas p99 ≤ 5 ms; `studio1` statistikoje matoma kopijos kaina (`d2h_bytes`)

[ ] 3. Neeksportuotas flow (neigiamas)
Duota (Given): flow tik privačiame `studio1` domene
Kai (When): `studio2` funkcija nurodo jį per `mxl:///…/studio1?id=…`
Tada (Then): kodas 4 (arba prieigos klaida iškart), pranešime — kad flow neeksportuotas arba neprieinamas; jokio dalinio skaitymo

[ ] 4. Skaitytojas negali sugadinti (neigiamas)
Duota (Given): `studio2` procesas su eksportuotu flow
Kai (When): bando jam sukurti rašytoją arba paleidžia `mxlGarbageCollectFlows` savo instancijoje
Tada (Then): rašytojo sukurti nepavyksta; flow failai ir `payload.sock` lieka; `studio1` skaitytojai nepraranda nė vieno grain'o

---

# [G5] Dalijimasis tarp serverių: grupė gali perduoti rinkinį kitos grupės serveriui

# Kontekstas (Context)
Kaip laidos inžinierius,
Noriu eksportuotą rinkinį perduoti kitai grupei kitame serveryje ta pačia `bridge` komanda,
Kad dalijimasis tarp serverių turėtų tas pačias taisykles kaip ir viename serveryje.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Rinkinys tarp grupių (teigiamas)
Duota (Given): `studio1` (serveris A) eksportavo rinkinį `pgm`; `bridge --side tx --group studio1 --bundle pgm`; serveryje B `bridge --side rx --group studio2`
Kai (When): veikia 60 s
Tada (Then): `studio2` domene yra tie patys flow ID su ta pačia `grouphint` žyme; galioja N1.1 ir N2.1 ribos
Ir (And): `vramflow-ctl bundle list --group studio2` rodo `pgm` su kilme (šaltinio serveris ir domeno ID)

[ ] 2. Privatus flow nepaleidžiamas (neigiamas)
Duota (Given): rinkinys yra tik privačiame `studio1` domene
Kai (When): paleidžiu tx
Tada (Then): kodas 2, pranešime — „neeksportuotas“; rx pusėje nieko nesukurta

### 6.3 D — NMOS

---

# [M1] Atradimas: valdymo sistema gali matyti VRAMFlow funkcijas ir flow'us per NMOS IS-04

# Kontekstas (Context)
Kaip objekto (facility) valdymo sistemos integratorius,
Noriu, kad VRAMFlow flow'ai ir funkcijų įėjimai atsirastų NMOS registre kaip standartiniai MXL Sender'iai ir Receiver'iai,
Kad juos matyčiau ir jungčiau tuo pačiu valdikliu kaip ir kitų gamintojų MXL funkcijas.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Registracija (teigiamas)
Duota (Given): veikia registras (pvz., `nmos-cpp-registry` ant `127.0.0.1`) ir `vramflow-nmos --root /dev/shm/vf --registry <url>`
Kai (When): grupėje paleidžiamas `gen` (video + garsas) ir `logo`
Tada (Then): per ≤ 5 s registro Query API rodo Node, po Device kiekvienam paleidimui, Source + Flow + Sender kiekvienam rašomam flow ir Receiver `logo` įėjimui; Sender: `transport == "urn:x-nmos:transport:mxl"`, `interface_bindings == []`, `manifest_href == null`
Ir (And): Flow `frame_width`, `frame_height`, `interlace_mode`, `grain_rate`, `media_type` sutampa su `VideoFormat`; `grouphint` žymė — kaip `flow_def.json`

[ ] 2. Atitiktis (teigiamas)
Duota (Given): AMWA `nmos-testing` toje pačioje mašinoje (be mDNS skelbimų į tinklą)
Kai (When): paleidžiamas rinkinys IS-04-01 (jame yra ir BCP-002-01 / BCP-002-02 patikros)
Tada (Then): 0 FAIL (WARN ir „not implemented“ išvardijami ataskaitoje); BCP-007-03-01 — žr. M2.4

[ ] 3. Registro nėra (neigiamas)
Duota (Given): registras nepasiekiamas arba sidecar'as nužudytas (`kill -9`)
Kai (When): medijos funkcijos veikia 60 s
Tada (Then): `late == 0`, 0 turinio klaidų, CLI ir failų valdymas veikia kaip be NMOS
Ir (And): sidecar'as (jei gyvas) bando registruotis iš naujo ir po registro grįžimo ištekliai atsiranda per ≤ 15 s

[ ] 4. Išteklių išnykimas (kraštinis)
Duota (Given): užregistruoti ištekliai
Kai (When): funkcija sustabdoma tvarkingai / nužudoma
Tada (Then): jos Sender / Receiver / Flow / Source / Device pašalinami iš registro per ≤ 2 s (tvarkingai) arba per ≤ 15 s (nužudyta; pagal nuomos išnykimą)

---

# [M2] Jungimas: valdiklis gali nurodyti funkcijai, kurį MXL flow skaityti (IS-05 + BCP-007-03)

# Kontekstas (Context)
Kaip valdymo sistemos integratorius,
Noriu per IS-05 nurodyti Receiver'iui domeno ID ir flow ID ir įjungti / išjungti skaitymą,
Kad maršrutizavimas vyktų standartiniu būdu, o ne per funkcijos perkrovimą su kitais argumentais.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Aktyvavimas (teigiamas)
Duota (Given): `logo` su laisvu įėjimu; du galimi šaltiniai F1, F2 toje pačioje grupėje
Kai (When): valdiklis PATCH `/staged`: `master_enable: true`, `activation.mode: "activate_immediate"`, `transport_params: [{"mxl_domain_id": <D>, "mxl_flow_id": <F2>}]`
Tada (Then): per ≤ 1 s funkcija skaito F2 (išėjimo barkodas atitinka F2), `/active` rodo tas pačias reikšmes, IS-04 Receiver'io `version` pasikeitė
Ir (And): tas pats pasiekiamas be NMOS: `vramflow-ctl connect --receiver <id> --flow <F2>` (tas pats jungties failas)

[ ] 2. Išjungimas (teigiamas)
Duota (Given): aktyvus Receiver'is
Kai (When): `master_enable: false`
Tada (Then): skaitymas sustoja per ≤ 1 s; funkcija toliau gyva; išėjimas elgiasi pagal `--on-loss`

[ ] 3. Negalimos reikšmės (neigiamas)
Duota (Given): užklausos su (a) domeno ID, kurio šis Node nepasiekia, (b) `mxl_flow_id: "auto"` Receiver'iui, (c) `transport_file` su duomenimis
Kai (When): PATCH `/staged`
Tada (Then): kiekviena atmetama HTTP 400; `/staged` ir `/active` nepakito; funkcija skaito tą patį flow kaip prieš tai

[ ] 4. Atitiktis (teigiamas)
Duota (Given): `nmos-testing`
Kai (When): IS-05-01, IS-05-02, BCP-007-03-01
Tada (Then): 0 FAIL

---

# [M3] Bridge per NMOS: valdiklis gali sujungti du serverius be failų nešiojimo

# Kontekstas (Context)
Kaip valdymo sistemos integratorius,
Noriu `bridge` paleisti ir sustabdyti per IS-05, o TargetInfo perduoti standartiniu IS-05 parametrų keliu,
Kad replikacija tarp serverių būtų valdoma iš tos pačios sistemos kaip ir visos kitos jungtys.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Jungtis per ext parametrus (teigiamas)
Duota (Given): rx pusės Sender'io `/active` turi `ext_vf_fabrics_target_info` ir `ext_vf_fabrics_provider`
Kai (When): valdiklis tas reikšmes ir `mxl_flow_id` įrašo į tx pusės Receiver'io `/staged` su `master_enable: true`
Tada (Then): replikacija prasideda per ≤ 2 s; galioja N1.1 ribos
Ir (And): `master_enable: false` ją sustabdo per ≤ 1 s

[ ] 2. Sugadintas TargetInfo (neigiamas)
Duota (Given): `ext_vf_fabrics_target_info` — ne TargetInfo
Kai (When): PATCH
Tada (Then): HTTP 400 su priežastimi; tx būsena nepakito; rx toliau laukia

[ ] 3. Be NMOS (neigiamas valdiklio priklausomybės testas)
Duota (Given): sidecar'as nepaleistas
Kai (When): `bridge` paleidžiamas su `--connect` / `--offer-in`
Tada (Then): veikia kaip N1 — NMOS nėra būtinas nė vienam `bridge` režimui

---

# [M4] Būsena: stebėjimo sistema gali matyti kiekvieno įėjimo ir išėjimo sveikatą standartiniu modeliu

# Kontekstas (Context)
Kaip eterio stebėjimo sistemos integratorius,
Noriu vėluojančių ir prarastų grain'ų, ryšio ir sinchronizacijos būseną gauti per BCP-008 modelį,
Kad VRAMFlow aliarmai atsirastų toje pačioje sistemoje kaip ir ST 2110 įrenginių.

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Sveika būsena (teigiamas)
Duota (Given): veikia `logo` su `--stats-out`; sidecar'as su IS-12
Kai (When): viskas tvarkoje 30 s
Tada (Then): įėjimo `NcReceiverMonitor`: `overallStatus == Healthy`, `connectionStatus == Healthy`, `streamStatus == Healthy`; skaitikliai lygūs P7 statistikos sumoms

[ ] 2. Šaltinis dingo (neigiamas)
Duota (Given): 1 kriterijaus būsena
Kai (When): šaltinio rašytojas nužudomas
Tada (Then): per ≤ 2 statistikos intervalus `connectionStatus == Unhealthy`, `connectionStatusTransitionCounter` padidėja 1, `overallStatus == Unhealthy`, žinutėje — flow ID
Ir (And): šaltiniui grįžus būsena tampa `Healthy` ne anksčiau kaip po `statusReportingDelay` (3 s)

[ ] 3. Statistikos nėra (neigiamas)
Duota (Given): funkcija paleista be `--stats-out`
Kai (When): sidecar'as kuria monitorių
Tada (Then): `overallStatus == Unhealthy` su žinute „nėra statistikos iš funkcijos“, ne `Healthy`

[ ] 4. Atitiktis (teigiamas)
Duota (Given): `nmos-testing`
Kai (When): IS-12-01, BCP-008-01-01, BCP-008-02-01
Tada (Then): 0 FAIL

### 6.4 Kas neįeina į user stories

IS-07, IS-08, IS-11, IS-14, EFA, multicast, WAN, šifravimas, automatinis orkestratorius — žr. §2 ne-tikslus.

## 7. Ryšys su baze (ką palikti atvirą jau dabar)

Šis etapas **remiasi**: P1 (indeksų ir laiko matematika; spragų laikmatis), P2 (`VideoFormat`, `sample_begin`,
`format_from_flow_def`), P5 (`GapPolicy`, `VideoOut::skip`, K3 `--on-loss`), P6 (`InputSlot`, `DomainLease`, `OpenPolicy`),
P7 (statistikos eilutės), P4 (priėmimas, knyga), P8 (talpos failas). Be P5–P7 `bridge` kurti neverta.

Bazėje jau dabar reikia palikti atvira (detalės ir signatūros — [P] §4, etapas E0) [mano; autoriaus sprendimas K-14, K-15]:

1. **Įėjimas adresuojamas (domenas, flow UUID)**, ne (rolė, numeris): `InputSlot` turi leisti pakeisti taikinį veikiant
   (kitas flow; kitas domenas — per instancijų talpyklą) ir būti išjungiamas (IS-05, grupių dalijimasis).
2. **Priėmimo knyga — vienai GPU visame serveryje**, ne vienam domenui: su dviem domenais vienoje GPU atskiros knygos
   skaičiuotų talpą du kartus. Įraše — `group`.
3. **Statistikos schema v1** — pridedami tik neprivalomi laukai (`group`, `bundle`, `domain`, `links[]`); būsenų vardai
   iš karto kaip BCP-008 (`Healthy`, `PartiallyHealthy`, `Unhealthy`, `Inactive`).
4. **Flow ID** — deterministinis pagal (grupė, rinkinys, rolė); `grouphint` = `"<rinkinys>:<rolė>"`.
5. **Nuomos failas** — `raktas=reikšmė` eilutės, nežinomi raktai ignoruojami (bus `group=`, `bundle=`, `export=`).
6. **Vienas UID `cuda-linear` flow'ams** — README ir skaitytojo klaidos tekstas turi tai pasakyti aiškiai (kol upstream
   nepataisyta); bazės testai neturi remtis prielaida, kad kitas UID gali atidaryti `cuda-linear` flow.
7. **`domain_def.json`** rašomas kiekvienam domenui, kurį sukuria VRAMFlow įrankiai ir harness'as.

## 8. Palikta statytojui

- Komandų ir parinkčių vardai (`bridge --side tx|rx` ar dvi subkomandos; `vramflow-ctl` ar `vramflow-func group …`).
- Rendezvous žinučių laukai ir versijavimas (su sąlyga: viena JSON eilutė žinutei, laukas `v`, nežinoma versija → kodas 8).
- Tarpinio host flow vieta (`<grupė>.vramflow/wire-<run_id>/` ar kita), su sąlyga, kad ji trinama su nuoma.
- Ar `cuda-linear` → host įkėlimas / iškėlimas naudoja esamą `cuda_upload.hpp` kelią, ar atskirą prisegtą buferį — pagal matavimą.
- UUIDv5 vardų erdvės UUID ir vardo eilutės formatas (su sąlyga: testas lygina su nepriklausomu Python `uuid.uuid5`).
- Kaip harness'e nutraukiama jungtis (tinklo vardų erdvė + `veth`, `tc netem`, ar tarpinis procesas) — kas veikia WSL2 ir natyviai.
- NMOS sidecar'o vidinė struktūra ir kaip `nmos-cpp` įtraukiamas į build'ą (jei K-9 patvirtintas).

## 9. Atviri klausimai autoriui

| # | Klausimas | Rekomendacija | Kiti variantai (kodėl ne) |
|---|---|---|---|
| K-1 | Nauja priklausomybė: **libfabric** ir MXL build su `MXL_ENABLE_FABRICS_OFI=ON` | libfabric **v2.5.1** iš šaltinio — tą pačią versiją MXL naudoja savo devcontainer'yje [faktas: F-21] | Ubuntu 24.04 paketas 1.17.0 (per senas ir MXL su juo netestuoja); v2.7.0 (2026-09-18, naujausia, bet MXL su ja nepatvirtinta) |
| K-2 | Kurias „GPU dalijimo“ reikšmes palaikyti | R1 + R2 + R3, R4 antrame žingsnyje; R5, R6 — ne (§5.1) | Tik R1 (per siaura: CPU serveriai ir MXL įrankiai lieka be kelio) |
| K-3 | TargetInfo perdavimas | Trys lygiai: failas (pirmas etapas) → mažas TCP rendezvous (atsistatymui be išorinio veikėjo) → NMOS IS-05 `ext_` (neprivaloma) | Tik failai (po rx perkrovimo reikia išorinio skripto); tik NMOS (įrankiai nebeveiktų be registro — anti-tikslas) |
| K-4 | GPU atminties RDMA: laukti upstream ar daryti patiems | Dabar — per host buferį su nepakeistu Fabrics. GPUDirect — tik per upstream (padėti ištestuoti PR #654 lab'e, siūlyti pataisas kaip U1–U4). Jokios privačios Fabrics atšakos | Savas `FI_HMEM` kelias šalia Fabrics (dubliuoja upstream darbą, neišlaikomas); laukti ir nieko nedaryti (TCP / RDMA per host veikia jau dabar) |
| K-5 | Grupės modelis | Grupė = domenas (+ `.pub` eksporto domenas); rinkinys = `grouphint` (§5.2) | Vienas domenas + žymės (be izoliacijos); domenas rinkiniui (per daug domenų) |
| K-6 | Izoliacijos lygis ir teisių modelis | Numatyta **organizacinė** izoliacija (vienas UID, atskiri domenai, kvotos; konteineriuose — atvaizduojami tik reikalingi domenai) — su ja veikia zero-copy visiems flow'ams. **Griežta** (UID ir gid kiekvienai grupei) — parinktis; `cuda-linear` eksportas kitam UID joje eina per host flow, kol upstream nepataisė lizdo teisių (K-13) | Griežta kaip numatytoji (šiandien neleistų dalintis GPU flow'ais be kopijos); tik konteineriai (netinka plikam serveriui); ACL (`setfacl`) — galima vėliau |
| K-7 | Numatytoji elgsena dingus ryšiui ir per lėtai jungčiai | Dingus ryšiui — `invalid` (kaip bazės K3); per lėta jungtis — šokti prie naujausio (ribotas vėlavimas, praleisti indeksai INVALID) | Kaupti ir vytis (vėlavimas auga — eteriui blogiau nei spraga); `freeze` kaip numatytoji (slepia gedimą) |
| K-8 | Laikrodžių reikalavimas tarp serverių | Matuoti visada; įspėti, kai \|δ\| > 1 ms; drausti tik su `--max-clock-offset-ms`. Dokumentuoti: chrony ≤ 1 ms minimumas, PTP — rekomenduojama | Reikalauti PTP visada (neįmanoma WSL2 ir debesyje); nematuoti (problema pasimato tik eteryje) |
| K-9 | NMOS įgyvendinimas | Atskiras sidecar'as ant **sony/nmos-cpp** (Apache-2.0; aktyvus, paskutinis commit'as 2026-11-09; turi IS-04/05/12, BCP-007-03, BCP-008). Pin: commit `b6c83670` (2026-10-05, 37 d.) [faktas: F-38]. Neprivalomas CMake taikinys | NVIDIA NvNmos (Apache-2.0, paprastesnis C API, bet papildomas sluoksnis ant nmos-cpp); savas minimalus IS-04 / IS-05 (HTTP serveris, DNS-SD, registracijos klientas — daug kodo, saugumo paviršius, atitikties testai mūsų sąskaita); NMOS medijos funkcijų viduje (HTTP kiekviename procese — prieš „žemo lygio“ principą) |
| K-10 | NMOS apimtis ir eilė | (1) failai „paruošti NMOS“ (be priklausomybių) → (2) IS-04 + IS-05 tik būsena → (3) IS-05 valdymas ir `bridge` → (4) IS-12 + BCP-008 po P7 → (5) TLS. Praleisti IS-07, IS-08, IS-11, IS-14 | Pradėti nuo IS-12 (nėra ką stebėti, kol P7 nepadarytas) |
| K-11 | Saugumo lygis | Sidecar'e TLS (BCP-003-01) — parinktis, lab'e išjungta; IS-10 — tik jei to reikalauja objektas; rendezvous prievadas — tik medijos tinklo sąsajoje + leidžiamų adresų sąrašas | Privalomas TLS visur (sertifikatų infrastruktūros čia nėra) |
| K-12 | Geležis RDMA / GPUDirect patikrai | Reikia žinoti: ar lab klasteryje yra RoCE NIC (ConnectX-4 ar naujesnės) ir profesionalios / duomenų centrų klasės GPU; ar galima gauti ≥ 10 GbE jungtį tarp nešiojamo kompiuterio ir darbo stoties. Be to: RDMA tik soft-RoCE darbo stotyje, tarp mašinų — tik 720p25 | — |
| K-13 | Upstream radiniai | Paruošti kaip `upstream/U5-…` su įrodymais ir pasiūlyti PR #653 autoriui: (a) lizdo leidimai išvedami iš laikino 0700 katalogo prieš paskelbimą ⇒ deskriptorių gauna tik rašytojo UID, nors aprašas žada „kas gali skaityti flow katalogą“; (b) `fchmod` po `bind` nekeičia kelio teisių; (c) `SO_PEERCRED` tikrina tik pagrindinę grupę. Siunčia autorius | Nutylėti (liktume su ribojimu, kuris kliudo ir kitiems MXL naudotojams); taisyti savo MXL kopijoje (privati atšaka — anti-tikslas) |
| K-14 | Priėmimo knygos vieta (keičia bazės P4 dizainą) | Viena knyga vienai GPU visame serveryje (`<šaknis>/.gpu/<gpu-uuid>/`), įraše `group` ir `domain` | Palikti `<domenas>.vramflow/` (su keliais domenais talpa skaičiuojama kelis kartus) |
| K-15 | Flow ID schema (keičia bazę) | UUIDv5 iš (grupė, rinkinys, rolė); senas `--flow-base` lieka testams | Palikti fiksuotus ID (susidūrimai tarp serverių ir grupių) |

## 10. Kas jau yra ir ką pernaudoti

- MXL: `mxl/fabrics.h` (target / initiator, TargetInfo), `tools/mxl-fabrics-demo` (pavyzdys abiem pusėms), sync grupės
  (`mxl/flow.h:923-1019`), `mxl-info` (jau skaito `domain_def.json`, `tools/mxl-info/main.cpp:662`).
- VRAMFlow: `src/func/flows.{hpp,cpp}` (`VideoIn` / `VideoOut` host ir `cuda-linear`, `AudioOut`), `src/func/cuda_upload.hpp`
  (host ↔ GPU kopijos — H variantas), `src/runtime/` (`InputSlot`, `DomainLease`, `OpenPolicy`), `src/timing/`
  (`index_of`, `time_of`), `verify` (barkodas, CRC, vėlavimas), `tests/perf/capacity.py`, `compare_report.py`,
  `tests/perf/domain.py`; upstream radinių tvarka `upstream/U*`.
- Dokumentai: `docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md` §2.3 (Fabrics ir laikas, testas T7),
  `docs/architektura/2026-11-11-gpu-vs-cpu-ir-upstream.md` (S16 — Fabrics trūkumas).
