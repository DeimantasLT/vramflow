# [P4] Perkrova: operatorius gali būti tikras, kad per didelė apkrova nesugriauna jau veikiančių srautų

> **Būsena: Laukia autoriaus patvirtinimo.** Planas: `docs/planai/2026-11-11-production-grade.md` §P4.
> Brief'as: `docs/reikalavimai/2026-11-11-production-grade.md` (L4, L10, L11, P4).

# Kontekstas (Context)
Kaip operatorius, paleidžiantis daug srautų ant vienos GPU,
Noriu, kad sistema žinotų išmatuotą kiekvieno variklio (NVDEC, NVENC, VRAM) talpą, atsisakytų priimti daugiau, nei
gali aptarnauti, o pavėlavęs srautas pasivytų praleisdamas iki artimiausio rakto kadro,
Kad jau veikiantys išėjimai niekada neužgestų visi kartu ir aš iškart matyčiau, kas nepriimta ir kodėl.

---

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Dviguba apkrova su priėmimo kontrole (teigiamas)
Duota (Given): išmatuota talpa N srautų (talpos failas iš talpos harness'o) ir `--admission enforce`
Kai (When): paleidžiu 2 × N ingest srautų
Tada (Then): priimami pirmi `floor(0,85 × N)` srautų ir kiekvienas jų per 60 s turi 0 pavėluotų kadrų
Ir (And): likę srautai nepaleidžiami, jų flow'ų domene nėra

[ ] 2. Atmetimas yra aiškus (neigiamas)
Duota (Given): talpa užimta iki ribos
Kai (When): bandau paleisti dar vieną srautą
Tada (Then): jis atmetamas su exit kodu 5
Ir (And): pranešime yra ribojantis variklis (`nvdec`, `nvenc` arba `vram`), prašyta, užimta ir leistina reikšmė skaičiais

[ ] 3. Tas pats ant bet kurios GPU (teigiamas)
Duota (Given): dirbtinis talpos failas su riba 4 srautams
Kai (When): paleidžiu 8 srautus
Tada (Then): pirmi 4 veikia 60 s su `late = 0` ir `skip_events = 0`
Ir (And): 5–8 atmesti; `mxl-info -l` rodo lygiai 8 flow'us (4 video + 4 garso)

[ ] 4. Perkrova be priėmimo kontrolės negriauna visko (neigiamas)
Duota (Given): `--admission off` ir 2 × N srautų
Kai (When): sistema veikia 60 s perkrauta
Tada (Then): nė vienas išėjimas neturi ilgesnės nei 2 s spragos iš eilės
Ir (And): kiekvienas srautas pateikia ≥ 40 % savo grain'ų, o praleidimai matomi skaitikliuose `skip_events` ir gyvoje statistikoje (būsena `Degraded`)

[ ] 5. Atsigavimas po perkrovos (teigiamas)
Duota (Given): 4 kriterijaus būsena
Kai (When): sustabdau perteklinius N srautų
Tada (Then): likę srautai per ≤ 5 s grįžta į 0 pavėluotų kadrų ir būseną `Normal`

[ ] 6. Pavėlavęs srautas praleidžia iki rakto kadro (teigiamas)
Duota (Given): vienas srautas, kurio paketai sulaikomi 2 s ir tada atsiunčiami pliūpsniu
Kai (When): pliūpsnis atkeliauja
Tada (Then): pavėlavę paketai nedekoduojami (`dropped_packets > 0`, `skip_events = 1`)
Ir (And): išėjimas atsinaujina nuo pirmo rakto kadro, kuris spėja į savo terminą, ne vėliau kaip per 1 GOP + 200 ms

[ ] 7. Nėra talpos failo (kraštinis, neigiamas)
Duota (Given): talpa šiai GPU neišmatuota
Kai (When): paleidžiu su `--admission enforce`
Tada (Then): procesas baigiasi su exit kodu ≠ 0 ir pranešimu, kaip išmatuoti talpą
Ir (And): su `--admission warn` srautai paleidžiami ir kas 10 s rašomas įspėjimas „talpa neišmatuota“

[ ] 8. Keli procesai dalijasi ta pačia GPU (daug elementų)
Duota (Given): du ingest procesai tame pačiame domene ir toje pačioje GPU, talpa 8 srautai
Kai (When): pirmas rezervuoja 5, antras prašo 5
Tada (Then): antras gauna tik tiek, kiek telpa iki `0,85 × 8` ribos (1 srautą), likę atmetami su 2 kriterijaus pranešimu

[ ] 9. Nužudytas procesas neužima talpos (neigiamas)
Duota (Given): procesas su rezervacija nutraukiamas `kill -9`
Kai (When): kitas procesas tikrina užimtumą
Tada (Then): mirusio proceso rezervacija nebeskaičiuojama per ≤ 100 ms
Ir (And): naujas srautas priimamas

[ ] 10. Pasenęs talpos failas (kraštinis)
Duota (Given): talpos failas išmatuotas su kita tvarkyklės versija arba kita GPU
Kai (When): paleidžiu ingest
Tada (Then): rašomas įspėjimas su abiem versijomis; kita GPU (UUID nesutampa) → failas nenaudojamas

---

# Papildoma informacija (Optional)
UI/UX nuorodos: nėra (komandinė eilutė + JSON eilučių statistika).
Techninės pastabos: paklausa ir talpa skaičiuojama megapikseliais per sekundę kiekvienam varikliui ir VRAM baitais;
rezervacijų knyga — failai su `flock` nuomos kataloge šalia domeno; `min_decode` matuojamas kiekvienam šaltiniui.
Riba „≥ 40 %“ (4 kriterijus) — pasiūlymas, kalibruojamas pirmu matavimu ir tada fiksuojamas.
Rizikos / Apribojimai: ilgas GOP (iki 10 s) reiškia ilgą praleidimą — tai matoma, ne slepiama; automatinis srautų
prioritetas ir paskirstymas tarp kelių GPU — kitam etapui. `--shed newest` — pasirinktinis, numatyta išjungta.
Instrukcijos: README skyrius „Capacity and admission“ su talpos matavimo komanda ir exit kodų lentele.
