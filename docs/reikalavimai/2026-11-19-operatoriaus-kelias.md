# Operatoriaus kelias: įdiegti, paleisti, gauti signalą (VRAMFlow) — reikalavimai

- **Data:** 2026-11-19
- **Būsena:** juodraštis, laukia autoriaus patvirtinimo. Kodas nekeistas.
- **Repo:** `vramflow`. Susiję: `docs/planai/2026-11-17-production-ready.md` (5.1 soak, 5.3 diegimas, 5.4 konteineriai,
  6.1 operatoriaus vadovas) — šis dokumentas juos sujungia į vieną naudotojo kelią ir iškelia prieš 4 fazę (VRAM).
- **Žymėjimas:** **[autorius]** — autoriaus žodžiai ar sprendimas; **[mano]** — siūlymas, kurį autorius gali atmesti;
  **[faktas]** — patikrinta 2026-11-19 kode ar matavime.

## 1. Kam ir kodėl

**[autorius]** „Man svarbiausia, kad paleidžia signalą ir jis stabiliai eina, ir galima jį nuskaityti ir pasiimti. Žmogus
pasiima, įsirašo, paleidžia, duoda inputą ir gauna outputą. Viskas turi veikti stabiliai, tiksliai, kokybiškai.“
Taikinys — **natyvus Linux serveris su NVIDIA GPU**, diegimas — **Docker atvaizdas** [autorius, 2026-11-19]. Standartas
turi tikti bet kuriai NVIDIA GPU, ne vienam kompiuteriui [autorius].

**Kas šiandien trukdo [faktas]:**

- Diegimas: reikia rankomis klonuoti MXL PR #653, sukompiliuoti su vcpkg, sukompiliuoti VRAMFlow ir kiekviename terminale
  nustatyti `LD_LIBRARY_PATH` / `PATH` (README §3).
- Grandinę reikia jungti pačiam keliomis komandomis su sutampančiomis rolėmis ir flow ID.
- `enc` išvestis įrašyta kode: `udp://127.0.0.1:<port-base + n>` (`src/func/enc.cpp:255`) — signalo į kitą mašiną, SRT ar
  multicast išsiųsti neįmanoma.
- GPU paruošimas (laikrodžiai, galios riba, `keepwarm`) — žinomas tik iš README §8 ir matavimų ataskaitų.

**Sėkmė:** serveryje su NVIDIA GPU ir Docker operatorius įvykdo vieną `docker run` su konfigūracijos failu, kuriame
nurodyti įėjimai (SRT / UDP MPEG-TS), apdorojimas (logotipas arba grafika) ir išėjimai (SRT / UDP / multicast); po
≤ 5 s išėjime yra H.264 + AAC TS, kurį priima `ffplay` / `ffprobe` / vMix / kitas imtuvas; 2 h be praleistų kadrų,
A/V ≤ 1 laukas; klaida pasakoma žmonių kalba, ne kodu.

## 2. Tikslai, ne-tikslai, anti-tikslai

**Tikslai**

- T1. `enc` išvestis — bet kuris FFmpeg palaikomas TS adresas kiekvienam srautui: `udp://host:port` (unicast ir multicast,
  `ttl`), `srt://host:port?mode=caller|listener` (slaptažodis ne iš argv, kaip `ingest`).
- T2. `vramflow-func run <config>` — vienas procesas-prižiūrėtojas paleidžia `ingest` → `logo` arba `composite` → `enc`
  teisinga tvarka, vienoje domeno direktorijoje, su `cuda-linear` flow'ais; nukritusį vaiką perkrauna (esamos atkūrimo
  taisyklės), sustabdo visus tvarkingai (SIGTERM → visi baigiasi ≤ 5 s, domenas švarus).
- T3. Konfigūracija — vienas JSON failas (be naujos priklausomybės) [mano]; klaidos su lauko keliu ir priežastimi.
- T4. `vramflow-func gpu-check` — paleidžiama automatiškai prieš `run`: GPU modelis, tvarkyklė, CUDA, NVENC / NVDEC,
  NVENC sesijų riba, persistence mode, ar galios riba ir dažniai fiksuoti, ribojimo priežastys, TAI poslinkis 37 s.
  Išvada: `tinka` / `tinka su įspėjimais` / `netinka` + ką padaryti. Be konkrečių vatų ar laipsnių kode — tik būsenų
  stabilumas ir NVML vėliavos [autorius: „bet kuriam su NVIDIA GPU“].
- T5. GPU stebėjimas veikimo metu: NVML galios ribos, temperatūros ir ribojimo priežasčių pokyčiai rašomi į statistiką
  su TAI laiku — eteryje platformos sustojimas matomas kaip įvykis, o ne kaip nepaaiškinti praleisti kadrai.
- T6. Docker atvaizdas (`docker run --gpus all`): MXL, VRAMFlow, FFmpeg viduje; domenas — `tmpfs` konteineryje; jokių
  build įrankių galutiniame atvaizde; versija `vramflow-func --version` ir `versions_json`.
- T7. Stabilumo įrodymas: 2 h soak'as būtent šios grandinės (`run` su konfigūracija), N = 0,85 · kelias, išėjimas
  tikrinamas `verify --mode ingest` per antrą `ingest` iš išėjimo TS (uždaras ratas).
- T8. Operatoriaus vadovas (`docs/operations.md`, EN): 1 puslapio „pirmas paleidimas“, konfigūracijos laukai, klaidos,
  ką daryti.

**Ne-tikslai**

- Web UI, NMOS valdymas, keli serveriai, RDMA (atskiras dokumentas `2026-11-12-tinklas-ir-grupes.md`).
- `.deb` paketas (autorius pasirinko Docker).
- Kelių konteinerių dalijimasis `cuda-linear` flow'ais (5.4) — šiame etape visa grandinė viename konteineryje.
- VRAM optimizavimas (4 fazė) — po šio kelio.

**Anti-tikslai** (net jei techniškai veikia — nesėkmė)

- Paleidimas, kuriam reikia `LD_LIBRARY_PATH`, rankinio flow ID ar žinoti, kas yra rolė.
- Tylus kadrų praradimas: kiekvienas praleistas laukas ar platformos sustojimas — matomas statistikoje.
- Slaptažodžiai komandų eilutėje ar žurnaluose.
- Konfigūracijos laukas, kuris tyliai ignoruojamas.

## 3. User stories

### [OK-1] Išvestis į bet kurį adresą: operatorius gali nusiųsti signalą kitur

**Kontekstas.** Kaip operatorius, noriu nurodyti kiekvieno išėjimo TS adresą (UDP unicast / multicast, SRT), kad signalą
pasiimtų kita mašina ar programa.

**Priėmimo kriterijai**

[ ] 1. UDP į kitą adresą
Duota: `enc --output udp://<adresas>:5000` vienam srautui. Kai: paleidžiama 10 s. Tada: `ffprobe` tame adrese mato
H.264 1920×1080 ir AAC 48 kHz. Ir: 0 `late_frames`.

[ ] 2. Multicast su TTL
Duota: `--output "udp://239.1.1.1:5000?ttl=4"`. Kai: imtuvas prisijungia prie grupės. Tada: gauna TS. Ir: santraukoje
įrašytas tikslas be slaptažodžių.

[ ] 3. SRT caller ir listener
Duota: `srt://…?mode=listener` ir išorinis `ffmpeg -i srt://…?mode=caller`. Kai: imtuvas prisijungia po 3 s. Tada: TS
gaunamas; iki prisijungimo `enc` nekrenta ir nepraranda įėjimo laukų. Ir: slaptažodis — tik `--srt-passphrase-from`.

[ ] 4. Keli srautai — keli adresai
Duota: N = 3, `--output` kartojamas 3 kartus. Tada: kiekvienas srautas savo adresu. Ir: `--output` skaičius ≠ N →
exit 2, klaida „--output: 2 adresai, srautų 3“.

[ ] 5. Blogas adresas
Duota: `--output ftp://x` arba `udp://` be porto. Tada: exit 2 prieš kuriant flow'us, žinutė su adresu ir priežastimi.

[ ] 6. Atgalinis suderinamumas
Duota: be `--output`. Tada: elgsena kaip dabar (`udp://127.0.0.1:<port-base + n>`); esami testai žali.

### [OK-2] Viena komanda: operatorius gali paleisti visą grandinę iš konfigūracijos

**Kontekstas.** Kaip operatorius, noriu vieno konfigūracijos failo ir vienos komandos, kad nereikėtų žinoti, kaip
sujungti `ingest`, `logo` / `composite` ir `enc`.

**Priėmimo kriterijai**

[ ] 1. Minimali konfigūracija
Duota: JSON su 1 įėjimu (`srt://…` arba `udp://…`), `logo: builtin`, 1 išėjimu. Kai: `vramflow-func run cfg.json`. Tada:
≤ 5 s iki pirmo išėjimo TS paketo. Ir: išėjimo vaizdas su logotipu (`verify` / PSNR prieš CPU etaloną kaip `fchain`).

[ ] 2. Keli srautai
Duota: 4 įėjimai, 4 išėjimai. Tada: kiekvienas įėjimas → savo išėjimas, A/V ≤ 1 laukas kiekviename.

[ ] 3. Blogas konfigūracijos failas
Duota: nežinomas laukas, trūkstamas `outputs`, netinkamas JSON. Tada: exit 2 prieš paleidžiant vaikus; žinutė su lauko
keliu (`inputs[1].url: …`). Ir: nežinomas laukas — klaida, ne ignoravimas.

[ ] 4. Tvarkingas sustabdymas
Duota: veikianti grandinė. Kai: SIGTERM (`docker stop`). Tada: visi vaikai baigiasi ≤ 5 s, domenas tuščias, exit 0.

[ ] 5. Vaiko kritimas
Duota: veikianti grandinė. Kai: `kill -9` `logo` vaikui. Tada: prižiūrėtojas jį perkrauna; išėjimas atsistato ≤ 2 s;
praleisti laukai suskaičiuoti statistikoje. Ir: po 5 kritimų per 60 s — visa grandinė baigiasi exit 2 su priežastimi
(ne begalinis ciklas) [mano: ribos konfigūruojamos].

[ ] 6. Įėjimas dingsta ir grįžta
Duota: SRT šaltinis nutrūksta 10 s. Tada: išėjimas tęsiasi pagal esamą `ingest` politiką (juoda / paskutinis kadras),
neužstringa; šaltiniui grįžus — vėl tikras vaizdas, A/V ≤ 1 laukas.

[ ] 7. Grafika vietoje logotipo
Duota: `composite` su `libvramflow-gfx` sluoksniu (OGraf pavyzdys). Tada: tie patys 1–4 kriterijai.

### [OK-3] GPU patikra: operatorius sužino, ar mašina tinka eteriui

**Kontekstas.** Kaip diegėjas, noriu prieš paleidimą sužinoti, ar ši mašina ir GPU tinka, kad problemos neišlįstų eteryje.

**Priėmimo kriterijai**

[ ] 1. Tinkama mašina — `gpu-check` exit 0, `tinka`, sąrašas: modelis, tvarkyklė, CUDA, NVENC / NVDEC, sesijų riba.
[ ] 2. Nefiksuoti dažniai / galios riba — `tinka su įspėjimais`, ir konkreti komanda, kaip sutvarkyti (`nvidia-smi -pm 1`,
`-lgc`, `-pl`), arba „nešiojamas kompiuteris — galios riba keičiama platformos, naudoti tik testams“.
[ ] 3. Nėra GPU / NVENC / blogas TAI poslinkis — `netinka`, exit 2, priežastis.
[ ] 4. `run` paleidžia `gpu-check` pats; `netinka` → nepaleidžia; `--skip-gpu-check` — tik su įspėjimu žurnale.
[ ] 5. Veikimo metu galios ribos / temperatūros / ribojimo priežasčių pokytis → statistikos eilutė su TAI; vienetų
testas su NVML imitacija (pokytis → įvykis, be pokyčio → nieko).
[ ] 6. `keepwarm` įjungiamas automatiškai tik kai `gpu-check` nustatė, kad dažniai nefiksuoti.

### [OK-4] Docker atvaizdas: diegėjas gali įdiegti viena komanda

**Kontekstas.** Kaip diegėjas, noriu vieno Docker atvaizdo, kad nereikėtų nieko kompiliuoti serveryje.

**Priėmimo kriterijai**

[ ] 1. `docker run --rm --gpus all <atvaizdas> vramflow-func --version` → versija ir MXL SHA.
[ ] 2. `docker run --gpus all -v ./cfg.json:/cfg.json <atvaizdas> run /cfg.json` → OK-2 kriterijus 1 konteineryje.
[ ] 3. Konteineris ne root (UID ≠ 0); domenas `tmpfs`; po `docker stop` exit 0.
[ ] 4. Atvaizde nėra kompiliatorių ir `-dev` paketų; dydis įrašytas ataskaitoje.
[ ] 5. Atvaizdo build — viena komanda (`docker build`), atkuriama (prisegtos versijos, MXL SHA).

### [OK-5] Stabilumas: autorius gali įrodyti, kad grandinė eina 2 h

**Priėmimo kriterijai**

[ ] 1. `run` su N = 0,85 · chain kelias, 2 h, uždaras ratas (išėjimo TS → antras `ingest` → `verify --mode ingest`):
0 late, 0 missing, A/V ≤ 1 laukas, VRAM augimas ≤ 32 MiB, RSS ≤ 1 %, fd augimas 0 (plano 5.1 ribos).
[ ] 2. Tas pats su `logo` kritimu kas 60 s, 30 min: kiekvienas kritimas atsistato, praradimai = tik kritimo langai.
[ ] 3. Platformos GPU įvykiai, jei įvyksta, pažymėti T5 įvykiais ir neįskaičiuoti tyliai.

## 4. Žinomi kodo darbai, iškilę 3 fazėje (įtraukti)

- `dec` / `ingest` eilė po praleisto tick'o pagilėja negrįžtamai (+40 ms vėlinimo kiekvienam praleistam kadrui) —
  2 h soak'e kauptųsi; taisyti prieš OK-5 [faktas: 3 fazės ataskaita §3.3]. Patikrinti, ar `ingest` turi tą patį elgesį.
- `power_scheme` rezultatuose `None` (`env.sh` PATH).
- `interop_foreign_writer` 20 pakartojimų (3.1 likutis).

## 5. Priklausomybės ir rizikos

- Docker build ir bazinis CUDA atvaizdas (keli GiB) — nauja priklausomybė; siuntimas tik autoriui leidus (plano §1).
  Vietinis testas WSL2 reikalauja Docker Desktop WSL integracijos (laukia autoriaus).
- NVML — jau yra su tvarkykle (`libnvidia-ml.so`), atskiro paketo nereikia [mano, patikrinti build'e].
- SRT išvestis per FFmpeg 6.1.1 `libsrt` — patikrinti, ar sisteminis FFmpeg su ja sukompiliuotas (`ingest` SRT veikia —
  tikėtina taip).
- Natyvaus Linux serverio čia nėra: OK-3 kriterijus 1 ir OK-5 galutinai patvirtinami tik lauko bandyme.

## 6. Eiliškumas [mano]

OK-1 → OK-2 → OK-3 → (dec eilės pataisa) → OK-4 → OK-5 → operatoriaus vadovas → 4 fazė (VRAM).
Pirmas apčiuopiamas rezultatas — OK-1 + OK-2: viena komanda WSL2 duoda TS išėjimą bet kuriuo adresu.
