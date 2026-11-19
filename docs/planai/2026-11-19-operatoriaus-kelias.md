# Operatoriaus kelias — įgyvendinimo planas

> **Vykdytojams:** privalomas sub-skill'as — `superpowers:subagent-driven-development` arba `superpowers:executing-plans`.
> Žingsniai pažymėti `- [ ]`.

**Tikslas:** žmogus Linux serveryje su NVIDIA GPU paleidžia vieną `docker run` su konfigūracijos failu ir gauna stabilų
H.264 + AAC MPEG-TS išėjimą bet kuriuo UDP / multicast / SRT adresu.

**Architektūra:** esamos funkcijos (`ingest` → `logo` / `composite` → `enc`) lieka atskiri procesai viename domene su
`cuda-linear` flow'ais. Naujas prižiūrėtojas `vramflow-run` (Python, tik standartinė biblioteka) iš konfigūracijos
sudaro komandas, paleidžia jas teisinga tvarka, perkrauna nukritusius ir tvarkingai stabdo. `enc` gauna laisvą išvesties
adresą su atskira siuntimo gija. `vramflow-func gpu-check` ir NVML stebėjimas veikimo metu daro GPU būseną matomą.
Viskas supakuojama į Docker atvaizdą.

**Technologijos:** C++20, CUDA 13.2, FFmpeg 6.1.1 (sisteminis, `libsrt`), NVML (per esamą `dlopen`,
`src/func/stats_lines.cpp`), Catch2, CLI11, Python 3 (stdlib), Docker (`nvidia/cuda` bazė).

**Specas:** `docs/reikalavimai/2026-11-19-operatoriaus-kelias.md` (toliau **[R]**; user stories OK-1…OK-5).

## Globalūs apribojimai

- Kodo komentarai — lietuviškai; README, `docs/operations.md` — angliškai; `docs/rezultatai` — lietuviškai.
- 0 kompiliatoriaus įspėjimų; `clang-tidy` (`scripts/linux/55-clang-tidy.sh`) — 0 radinių naujame kode.
- TDD produkciniam kodui: kiekvienas [R] priėmimo kriterijus → testas, matytas krentantis prieš kodą.
- Testų ribos ir vartų taisyklės nekeičiamos; esami testai nekeičiami, išskyrus kai elgsena keičiasi pagal [R] (tada —
  ledger'yje `Ruling:`).
- `include/vramflow/gfx.h` nekeičiamas.
- Naujos priklausomybės — tik Docker bazinis atvaizdas (Task 8, autoriaus leidimas siuntimui); jokių naujų C++ / Python
  bibliotekų.
- Slaptažodžiai — tik `--srt-passphrase-from env:… | file:…`; santraukose ir žurnaluose — `redact_url`.
- Be `--output` `enc` elgiasi tiksliai kaip dabar (`udp://127.0.0.1:<port-base + n>`).
- Sintetinė medžiaga testuose (`scripts/linux/62-make-av-clip.sh`); jokio LRT turinio.
- Commit'ai — angliški Conventional Commits su `Assisted-by: Claude Code (claude-opus-5-5)`; push po patikros leidžiamas.
- Platformos ribos: matavimai WSL2 — orientaciniai; OK-3.1 ir OK-5 galutinai — natyviame Linux (lauko bandymas).

## Peržiūros fokusas

1. SRT imtuvas atsijungia ir vėl prisijungia eteryje — `enc` neturi užstrigti, kodavimas ir įėjimo skaitymas tęsiasi,
   praleisti paketai suskaičiuoti, ryšys atsistato (Task 2, testas `foutput_srt_reconnect`).
2. Konfigūracijoje du įėjimai su tuo pačiu UDP portu arba du išėjimai į tą patį adresą — klaida prieš paleidimą, ne
   tylus konfliktas (Task 4, `test_config_duplicate_endpoints`).
3. `docker stop` (SIGTERM, 10 s Docker riba) — visi vaikai baigiasi ≤ 5 s, domenas išvalytas, ne SIGKILL po 10 s
   (Task 4 `frun_stop`, Task 8 `check_image.sh` žingsnis 3).
4. Įėjimo šaltinis neateina iš viso paleidžiant (operatorius suklydo adresu) — `run` neturi kabėti tyliai: aiški žinutė
   po `input_wait_s`, exit 2 (Task 4, `frun_no_source`).
5. `gpu-check` mašinoje, kur NVML nepasiekiamas (konteineris be `--gpus`) — `netinka` su priežastimi „GPU nematoma
   konteineriui (`--gpus all`)“, ne segfault (Task 5, `gpu_verdict: nvml unavailable`).

---

### Task 1: `enc --output` ir `--audio-role` — adresas kiekvienam srautui, garsas iš `ingest`

**[R] OK-1 kriterijai 4–6; T1. Spragos:** išvestis įrašyta kode (`src/func/enc.cpp:255`); garsas skaitomas tik iš
`kRoleAudio` (`enc.cpp:451, 466, 482`), o `ingest` rašo `kRoleIngestAudio` (`src/func/flows.hpp:25`).

**Failai:**
- Modify: `src/ingest/url_policy.hpp`, `src/ingest/url_policy.cpp` — `output_url_allowed`
- Modify: `src/func/func_common.hpp` (laukai `outputs`, `audio_role`, `out_srt_passphrase_from`), `src/func/main_func.cpp`
  (`enc` parinktys), `src/func/enc.cpp` (URL parinkimas, validacija prieš flow'us)
- Test: `tests/unit/test_url_policy.cpp`, `tests/functional/functional.py` (`foutput_args`)

**Sąsajos:**
- Produces: `bool mxlv::ingest::output_url_allowed(const std::string& url, std::string& why)`;
  `std::string mxlv::func::enc_output_url(const FuncOpts& o, std::uint32_t n)` — `o.outputs[n]` arba senas
  `udp://127.0.0.1:%d?pkt_size=1316&buffer_size=4194304`; UDP adresui be `pkt_size` prideda `pkt_size=1316`.
- CLI: `enc --output URL` (kartojamas, 0 arba lygiai `--flows` kartų), `--audio-role N` (numatyta 3 = `kRoleAudio`),
  `--srt-passphrase-from env:V|file:P`.

- [ ] **1.1 Krentantys vieneto testai** (`test_url_policy.cpp`, `[enc][url]`):
  `output_url_allowed` priima `udp://10.0.0.5:5000`, `udp://239.1.1.1:5000?ttl=4`, `srt://h:9000?mode=listener`,
  `srt://h:9000?mode=caller&latency=200000`; atmeta su `why` netuščiu: `ftp://x`, `udp://h` (be porto), `udp://h:0`,
  `udp://h:70000`, `srt://h:1?passphrase=x` (`why` turi „--srt-passphrase-from“), `/tmp/a.ts`, `rtp://h:1`,
  `srt://h:1?mode=rendezvous`.
- [ ] **1.2** `mxlv-unit "[enc][url]"` → FAIL (nėra funkcijos).
- [ ] **1.3** Įgyvendinti `output_url_allowed` (naudoti `url_scheme`, `url_authority`); `enc_output_url`.
- [ ] **1.4** `mxlv-unit "[enc][url]"` → PASS.
- [ ] **1.5 Krentantys funkciniai** `foutput_args` (be GPU darbo, tik argumentų kelias): `--flows 3` su 2 `--output` →
  exit 2, stderr turi `--output: 2 adresai, srautų 3`; `--output ftp://x` → exit 2 per < 1 s, domene neatsiranda flow'ų;
  be `--output` — `ffprobe udp://127.0.0.1:<port>` mato srautą (senas kelias).
- [ ] **1.6** Įgyvendinti CLI ir validaciją prieš `create_output`; `--audio-role` naudoti visose trijose vietose.
- [ ] **1.7 Krentantis funkcinis** `fenc_ingest_audio`: `gen`-AV klipas → UDP → `ingest` → `enc --in-role 7 --audio-role 8
  --output udp://127.0.0.1:<p>` → antras `ingest` → `verify --mode ingest`: visi sinchro impulsai, A/V ≤ 1 laukas.
  Prieš 1.6 — FAIL (garso flow'as neatsiranda / `enc` laukia `kRoleAudio`).
- [ ] **1.8** `ctest` visas + `functional.py --only foutput_args,fenc_ingest_audio,fchain` → PASS; esami `enc` testai žali.
- [ ] **1.9** Commit: `feat(enc): --output URL per stream, --audio-role for ingest audio`.

### Task 2: `enc` siuntimo gija — SRT listener, atsijungimai, perjungimas

**[R] OK-1 kriterijai 1–3; peržiūros fokusas 1.** Šiandien `avio_open2` ir `av_interleaved_write_frame` kviečiami kodavimo
kelyje: SRT `listener` blokuotų iki imtuvo, nutrūkus ryšiui — klaida ir procesas krenta.

**Failai:**
- Create: `src/func/ts_sender.hpp`, `src/func/ts_sender.cpp` — viena gija vienam srautui
- Modify: `src/func/enc.cpp` (muxer'is rašo į `AVIOContext` su `write_packet` callback'u → `TsSender::push`),
  `cmake/func.cmake` (įtraukti failą, vieneto testą)
- Test: `tests/func/test_ts_sender.cpp` (`vramflow-glue-tests`), `tests/functional/functional.py`

**Sąsajos:**
- Produces:
  ```cpp
  struct SenderStats { std::uint64_t bytes_sent, packets_dropped, reconnects; bool connected; };
  class TsSender {  // muxer'is → eilė (≤ 2 s pagal muxrate) → gija → avio
   public:
    using OpenFn = std::function<int(AVIOContext**)>;  // testuose — imitacija
    TsSender(std::string url_redacted, OpenFn open, std::size_t max_queue_bytes);
    void push(const std::uint8_t* p, std::size_t n);  // niekada neblokuoja; pilna eilė → seniausi 188 B kartotiniai išmetami
    SenderStats stats() const;
    void stop();  // ≤ 1 s
  };
  ```
- Santraukos laukai `enc[n]`: `out_url` (redaguotas), `out_bytes`, `out_dropped_packets`, `out_reconnects`,
  `out_connected`; `--stats-file` eilutėje tie patys.
- Pakartotinis prisijungimas: atidėjimas 0,5 → 1 → 2 → … ≤ 5 s; išmetama tik iki TS paketo ribos (188 B), kad imtuvas
  gautų sveikus paketus.

- [ ] **2.1 Krentantys vieneto testai** `ts_sender`: (a) `open` grąžina klaidą 3 kartus, paskui pavyksta → `reconnects` 3,
  `push` niekada neužtrunka > 1 ms; (b) rašymas grąžina klaidą viduryje → atsijungia, vėl prisijungia, `connected` true;
  (c) eilė pilna → `packets_dropped` > 0, išsiųstų baitų skaičius dalijasi iš 188; (d) `stop()` per užblokuotą `open` ≤ 1 s.
- [ ] **2.2** `vramflow-glue-tests "[ts_sender]"` → FAIL (nėra klasės).
- [ ] **2.3** Įgyvendinti `TsSender`; `enc.cpp` — `avio_alloc_context` su `write_packet` → `push`.
- [ ] **2.4** `vramflow-glue-tests "[ts_sender]"` → PASS (10 kartų iš eilės, `--rng-seed` skirtingi).
- [ ] **2.5 Krentantys funkciniai:** `foutput_udp` (išėjimas į `udp://127.0.0.2:<p>` — `ffprobe` mato H.264 1920×1080 ir
  AAC 48 kHz, 0 `late_frames`); `foutput_srt_listener` (imtuvas `ffmpeg -i srt://…?mode=caller` prisijungia po 3 s → gauna
  TS; `enc` exit 0, 0 `late_frames`, įėjimo praradimų 0); `foutput_srt_reconnect` (imtuvas nužudomas po 10 s, paleidžiamas
  po 5 s → vėl gauna TS, `out_reconnects` ≥ 1, `enc` neužstrigo, `late_frames` 0); `foutput_multicast` (`udp://239.1.1.1:<p>?ttl=1`
  — SKIP su priežastimi, jei `ip route get 239.1.1.1` nerodo maršruto).
- [ ] **2.6** `ctest` + `functional.py --only foutput_,fenc_ingest_audio,fchain,fiso_enc` → PASS.
- [ ] **2.7** Commit: `feat(enc): non-blocking TS sender with SRT listener and reconnect`.

### Task 3: `ingest` eilės patikra po praleisto tick'o

**[R] §4.** 3 fazėje rasta: `dec` po praleisto tick'o eilė pagilėja (+40 ms vienam praleistam kadrui) ir nebeatsitraukia
iki `playout_frames + fps` ribos (`src/func/dec.cpp:351–370`, `clock_queue`). `dec` — testinis kelias; produkte —
`ingest`. Pirmiausia nustatyti, ar `ingest` elgiasi taip pat.

**Failai:** priklauso nuo 3.1; jei taisoma — `src/func/dec.cpp` ir / ar `src/func/ingest.cpp`, testas
`tests/functional/ingest.py`.

- [ ] **3.1 Matavimas (`superpowers:systematic-debugging`):** `ingest` su `--test-drop-every` analogu neegzistuoja —
  naudoti `SIGSTOP` 100 ms `ingest` procesui 5 kartus per 60 s; matuoti `ingest` latenciją (commit − T(i)) prieš ir po.
  Rezultatas ledger'yje: ar latencija grįžta į bazinę ≤ 2 s po sustojimo.
- [ ] **3.2 Jei negrįžta:** krentantis testas `fingest_stall_recovers` (latencija po kiekvieno sustojimo grįžta ≤ bazinė
  + 1 laukas per 2 s) → pataisa → PASS 20 / 20. Jei grįžta: `dec` pataisa tik su tuo pačiu testu `fdec_stall_recovers`,
  arba `Ruling:` atidėti (dec — ne produkto kelias).
- [ ] **3.3** Commit (jei buvo pataisa): `fix(ingest|dec): drain playout queue back to target after a stall`.

### Task 4: `vramflow-run` — prižiūrėtojas iš konfigūracijos

**[R] OK-2 visi kriterijai; T2, T3; peržiūros fokusas 2–4.**

**Ruling (plano lygiu):** [R] T2 rašė `vramflow-func run`; prižiūrėtojas rašomas Python (stdlib `json`, `subprocess`,
`signal`), nes griežta JSON validacija C++ be naujos bibliotekos — šimtai eilučių savo analizatoriaus; komanda
`vramflow-run <config>`. Kaina, jei klystu: atvaizde reikia `python3` (~30 MB).

**Failai:**
- Create: `tools/vramflow-run/vramflow_run/config.py` (schema, validacija → `Plan`), `…/graph.py` (`Plan` → komandų sąrašas),
  `…/supervise.py` (paleidimas, perkrovimai, stabdymas), `tools/vramflow-run/vramflow-run` (įėjimo taškas),
  `config/examples/logo-srt.json`, `config/examples/logo-udp-4.json`
- Test: `tools/vramflow-run/test_config.py`, `tools/vramflow-run/test_graph.py` (`ctest run_config`, `run_graph`),
  `tests/functional/run.py` (`frun_*`, registruoti `functional.py`)
- Modify: `CMakeLists.txt` (ctest registracija, `install` į `bin/`)

**Konfigūracijos schema (v1, griežta — nežinomas laukas yra klaida):**
```json
{
  "version": 1,
  "format": "1080i50",
  "storage": "cuda-linear",
  "inputs":  [{"url": "srt://0.0.0.0:9000?mode=listener", "srt_passphrase_from": "env:SRT_IN"}],
  "process": {"logo": {"file": "/cfg/logo.png", "x": 1600, "y": 60}},
  "outputs": [{"url": "udp://239.1.1.1:5000?ttl=4", "vbitrate": 7500000, "abitrate": 192000}],
  "restart": {"max": 5, "window_s": 60},
  "input_wait_s": 20
}
```
`process` — lygiai vienas iš `logo` (`file` arba `"builtin"`) ar `composite` (`layers` — `libvramflow-gfx` šaltiniai);
`inputs` ir `outputs` vienodo ilgio 1…16.

**Sąsajos:**
- `config.load(path) -> Plan` arba `ConfigError(path_in_json: str, why: str)`; `str(err)` = `"inputs[1].url: …"`.
- `graph.commands(plan, domain: str, exe: str) -> list[Proc]`, `Proc(name, argv, env)`; tvarka: `ingest` (`--flow-base 0`,
  rolės 7 / 8) → `logo --in-role 7` arba `composite` → `enc --in-role 2|6 --audio-role 8 --output …` (Task 1 parinktys).
- `supervise.run(plan, procs) -> int` (exit kodas): prieš paleidimą `gpu-check` (Task 5; `netinka` → exit 2); vaikas
  nukrito → perkrauti su tais pačiais argv; > `restart.max` per `window_s` → visi stabdomi, exit 2, priežastis stderr;
  SIGTERM / SIGINT → SIGTERM vaikams atvirkštine tvarka, po 4 s — SIGKILL, domenas pašalinamas, exit 0.

- [ ] **4.1 Krentantys `test_config.py`:** minimali konfigūracija → `Plan` su 1 įėjimu; nežinomas laukas `inputs[0].urll`
  → `ConfigError("inputs[0].urll", "nežinomas laukas")`; trūksta `outputs` → klaida „privalomas“; `inputs` 2, `outputs` 1 →
  `"outputs: 1, inputs: 2 — turi sutapti"`; abu `logo` ir `composite` → klaida; netinkamas JSON → klaida su eilute ir
  stulpeliu; `url` su `passphrase=` → klaida „naudokite srt_passphrase_from“; **`test_config_duplicate_endpoints`:** du
  įėjimai `udp://…:9000` arba du išėjimai tuo pačiu adresu → klaida su abiem keliais; 17 įėjimų → klaida „≤ 16“.
- [ ] **4.2 Krentantys `test_graph.py`:** 1 įėjimas + logo → 3 `Proc` tvarka `ingest, logo, enc`; `enc` argv turi
  `--audio-role 8` ir `--output <url>`; 4 įėjimai → vienas `ingest` su 4 `--input` ir `--flows 4` visur; slaptažodis
  neatsiranda jokiame argv, tik `env`.
- [ ] **4.3** `python3 test_config.py`, `test_graph.py` → FAIL; įgyvendinti `config.py`, `graph.py` → PASS.
- [ ] **4.4 Krentantys funkciniai `tests/functional/run.py`:** `frun_logo_udp` (OK-2.1: ≤ 5 s iki pirmo paketo; išėjimas
  per antrą `ingest` + `verify --mode ingest` — A/V ≤ 1 laukas; logotipo sritis prieš CPU etaloną kaip `fchain`);
  `frun_multi4` (OK-2.2); `frun_stop` (OK-2.4: SIGTERM → visi vaikai ≤ 5 s, domenas tuščias, exit 0); `frun_child_kill`
  (OK-2.5: `kill -9 logo` → išėjimas atsistato ≤ 2 s; 6 kritimai per 60 s → exit 2, stderr „logo nukrito 6 kartus per 60 s“);
  `frun_source_gap` (OK-2.6: SRT šaltinis nutrūksta 10 s → išėjimas tęsiasi, grįžus A/V ≤ 1 laukas); `frun_no_source`
  (fokusas 4: neteisingas portas → po `input_wait_s` exit 2, žinutė su redaguotu URL); `frun_composite` (OK-2.7).
- [ ] **4.5** Įgyvendinti `supervise.py` ir įėjimo tašką → visi `frun_*` PASS; `ctest` PASS.
- [ ] **4.6** Commit: `feat(run): vramflow-run supervisor with strict JSON config`.

### Task 5: `vramflow-func gpu-check` ir automatinis `keepwarm`

**[R] OK-3 kriterijai 1–4, 6; T4; peržiūros fokusas 5.** Jokių konkrečių vatų ar laipsnių — tik NVML vėliavos ir
stabilumas.

**Failai:**
- Create: `src/func/gpu_check.hpp`, `src/func/gpu_check.cpp`
- Modify: `src/func/stats_lines.cpp` (NVML `dlopen` lentelė: pridėti `nvmlDeviceGetPersistenceMode`,
  `nvmlDeviceGetEnforcedPowerLimit`, `nvmlDeviceGetPowerManagementLimitConstraints`, `nvmlDeviceGetCurrentClocksEventReasons`,
  `nvmlDeviceGetEncoderCapacity`, `nvmlDeviceGetName`), `src/func/main_func.cpp` (subkomanda), `cmake/func.cmake`
- Test: `tests/func/test_gpu_check.cpp`

**Sąsajos:**
```cpp
struct GpuFacts { bool nvml_ok; std::string why_nvml, name, driver; int cuda_major; bool nvenc, nvdec;
                  bool persistence; bool clocks_locked; bool power_limit_stable;  // 3 s iš eilės nekinta
                  bool laptop;  // power_limit nestabilus ir min != max apribojimuose
                  int nvenc_session_limit;  // -1 — nežinoma
                  double tai_offset_s; };
enum class Verdict { Ok, Warn, Bad };
struct Check { Verdict v; std::vector<std::string> lines; std::vector<std::string> fixes; bool want_keepwarm; };
Check gpu_verdict(const GpuFacts& f);   // grynoji funkcija — testuojama be GPU
GpuFacts gpu_facts();                    // NVML + CUDA + CLOCK_TAI
```
- CLI: `vramflow-func gpu-check [--json]` → exit 0 (`tinka`), 0 (`tinka su įspėjimais`, stderr įspėjimai), 2 (`netinka`).
  `--json` → `{"verdict":"ok|warn|bad","lines":[…],"fixes":[…],"keepwarm":bool}` — jį skaito `vramflow-run`.

- [ ] **5.1 Krentantys `test_gpu_check.cpp` (`[gpu_check]`):** visi gerai → `Ok`, `want_keepwarm` false; `persistence`
  false → `Warn`, `fixes` turi `nvidia-smi -pm 1`; `clocks_locked` false → `Warn`, `want_keepwarm` true, `fixes` turi
  `nvidia-smi -lgc`; `laptop` true → `Warn` su eilute „nešiojamas kompiuteris — galios riba keičiama platformos, tik
  testams“; `nvenc` false → `Bad`; `tai_offset_s` 0 → `Bad` su „TAI poslinkis 0 s, reikia 37 s“; `nvml_ok` false su
  `why_nvml` → `Bad`, eilutė „GPU nematoma (konteineriui reikia `--gpus all`)“; `cuda_major` < 13 → `Bad`.
- [ ] **5.2** → FAIL; įgyvendinti `gpu_verdict`, `gpu_facts`, CLI → PASS.
- [ ] **5.3** Rankinis patikrinimas šioje mašinoje: `vramflow-func gpu-check` — išvestis ledger'yje (tikėtina `Warn`,
  `laptop`).
- [ ] **5.4 Funkcinis** `fgpu_check_json` (išvestis — galiojantis JSON, `verdict` ∈ {ok, warn, bad}); `vramflow-run`
  integracija: `want_keepwarm` → paleidžia `vramflow-func keepwarm --duration 0` pirmą ir stabdo paskutinį
  (`test_graph.py` atvejis); `--skip-gpu-check` → įspėjimas stderr.
- [ ] **5.5** Commit: `feat(gpu-check): GPU readiness verdict, keepwarm only when clocks float`.

### Task 6: GPU būsenos stebėjimas veikimo metu

**[R] OK-3 kriterijus 5; T5.**

**Failai:**
- Create: `src/func/gpu_watch.hpp`, `src/func/gpu_watch.cpp`
- Modify: `src/func/stats_lines.cpp` (10 Hz NVML imtis statistikos gijoje), santraukos `gpu_events`
- Test: `tests/func/test_gpu_watch.cpp`

**Sąsajos:**
```cpp
struct GpuSample { std::uint64_t tai_ns; unsigned power_limit_mw; unsigned temp_c; unsigned long long reasons; };
struct GpuEvent { std::uint64_t tai_ns; std::string what; };  // "power_limit 55190→42000 mW", "reasons +hw_slowdown"
class GpuWatch { public: std::optional<GpuEvent> step(const GpuSample& s); };
```
- Įvykis — galios ribos pokytis > 1 %, bet kuri nauja ribojimo priežastis (išskyrus `GpuIdle`, `ApplicationsClocksSetting`);
  stderr eilutė `[gpu] įvykis: TAI <s> <what>` (formatas kaip `keepwarm` / `hostwatch` liudininkų, kad `isolation.py`
  galėtų juos skaityti vėliau); santraukoje `gpu_events` ≤ 256.

- [ ] **6.1 Krentantys testai:** pirmas mėginys → nieko; tas pats → nieko; riba 55 190 → 42 000 → įvykis su abiem
  skaičiais; 55 190 → 55 000 (0,3 %) → nieko; nauja priežastis `SW_POWER_CAP` → įvykis; jos dingimas → nieko.
- [ ] **6.2** → FAIL; įgyvendinti → PASS; `ctest` PASS.
- [ ] **6.3 Patikrinimas:** chain N = 9 šaltas startas (Fable metodas, `scratchpad/fable/`) → `logo` / `enc` stderr turi
  `[gpu] įvykis` su `power_limit …→42000` ±0,2 s nuo ilgo sustojimo.
- [ ] **6.4** Commit: `feat(stats): NVML power-limit and throttle events with TAI`.

### Task 7: Matavimų procedūra — GPU nusistovėjimas prieš lygį

**Autoriaus sprendimas 2026-11-19:** niuansai — vykdytojo sprendimas, standartas bendras visoms NVIDIA GPU.

**Failai:** Modify `tests/perf/capacity.py` (prieš kiekvieną lygį — CUDA apkrova, kol `enforced.power.limit` ir
`clocks.sm` nekinta 3 s, ≤ 60 s; stulpeliai `gpu.csv`: `enforced.power.limit`, `temperature.gpu`,
`clocks_event_reasons.active`; lygio rezultate `gpu_settle_s`, `power_limit_steps`); `tests/perf/test_capacity_file.py`;
`scripts/linux/env.sh` (`power_scheme` — `powercfg.exe` pilnu keliu `/mnt/c/Windows/System32/`).

- [ ] **7.1 Krentantys testai** `test_capacity_file.py`: `settle([...])` grąžina sekundes, kai paskutinės 3 s reikšmės
  pastovios; nepastovios 60 s → `None` ir lygio įspėjimas; žingsnis lange → `power_limit_steps` 1.
- [ ] **7.2** → FAIL → įgyvendinti → PASS.
- [ ] **7.3 Patikrinimas:** chain N = 9, 3 × 60 s nuo šalto starto → 3 / 3 PASS (Fable: pašildžius 3 / 3).
- [ ] **7.4** G1 kartojimas visoms 5 funkcijoms 3 × 60 s; `docs/rezultatai/2026-11-19-production-ready-faze3.md` §3.3
  atnaujinti; `regress.py` prieš bazę.
- [ ] **7.5** Commit: `test(perf): settle GPU power state before each level, log limit and throttle reasons`.

### Task 8: Docker atvaizdas

**[R] OK-4. Sustojimas:** prieš 8.3 — autoriaus leidimas siųsti bazinį atvaizdą (`nvidia/cuda:13.x-devel-ubuntu24.04` ir
`-runtime`, keli GiB) ir Docker Desktop WSL integracija.

**Failai:** Create `docker/Dockerfile` (dvi stadijos: builder — MXL `6dddc91d` per `scripts/linux/10-build-mxl.sh`,
VRAMFlow `Release`; runtime — `nvidia/cuda:13.x-runtime-ubuntu24.04` + `libavformat60 libavcodec60 libavutil58
libsrt1.5-openssl python3`, `cmake --install` medis, naudotojas `vramflow` UID 10001, `ENTRYPOINT ["vramflow-run"]`),
`docker/README.md`, `tests/arch/check_image.sh`; Modify `cmake/install.cmake` ir `cmake/version.cmake` (plano 5.3:
`--version`, `RUNPATH $ORIGIN/../lib`).

- [ ] **8.1 Krentantis** `ctest install_tree` (plano 5.3): `cmake --install` į laikiną katalogą; `bin/vramflow-func --version`
  veikia be `LD_LIBRARY_PATH`; `readelf -d` `RUNPATH` turi `$ORIGIN` ir neturi build kelio.
- [ ] **8.2** Įgyvendinti install / version → PASS.
- [ ] **8.3 Krentantis** `tests/arch/check_image.sh <image>`: (1) `--version` su MXL SHA; (2) `run /cfg.json` su
  `config/examples/logo-udp-4.json` — išėjimas priimamas iš šeimininko `ffprobe`; (3) `docker stop` → exit 0 per ≤ 5 s;
  (4) `id -u` ≠ 0; (5) `dpkg -l | grep -E 'gcc|clang|-dev '` tuščias; (6) atvaizdo dydis — į ataskaitą.
- [ ] **8.4** `docker build` → `check_image.sh` PASS.
- [ ] **8.5** Commit: `feat(docker): runtime image with vramflow-run entrypoint`.

### Task 9: 2 h soak'as uždaru ratu

**[R] OK-5; plano 5.1.**

**Failai:** Create `tests/perf/soak.py`, `tests/perf/drift.py`, `tests/perf/test_drift.py` (plano 5.1 sąsaja); scenarijus —
`vramflow-run` su N = 9 (0,85 · chain kelias) konfigūracija; ratas: AV klipas → `ffmpeg` UDP → `vramflow-run` → išėjimas
UDP → antras `ingest` → `verify --mode ingest`.

- [ ] **9.1 Krentantys** `test_drift.py`: tiesinis augimas 1 MiB / 10 min per 2 h → slinkis 12 MiB; triukšmas be augimo →
  |slinkis| < 1 MiB; fd 40 → 41 → FAIL.
- [ ] **9.2** → FAIL → įgyvendinti → PASS.
- [ ] **9.3** Paleidimai (GPU — nieko kito): 2 h; 30 min su `logo` `kill -9` kas 60 s. Priėmimas — [R] OK-5.1–3 ribos.
- [ ] **9.4** Ataskaita `docs/rezultatai/2026-11-2x-soak.md`; README §8; commit `test(perf): 2 h closed-loop soak of vramflow-run`.

### Task 10: Operatoriaus vadovas ir likučiai

**[R] T8; 3.1 likutis.**

- [ ] **10.1** `docs/operations.md` (EN): „First run in 5 minutes“ (Docker), konfigūracijos laukai su pavyzdžiais, `gpu-check`
  išvestis ir ką daryti, exit kodai, statistika ir `[gpu] įvykis`, žinomos ribos (nešiojami kompiuteriai, NVENC sesijų
  riba GeForce kortose).
- [ ] **10.2** README §4 pradžioje — 10 eilučių „Run it“ su `docker run`; nuoroda į `docs/operations.md`.
- [ ] **10.3** `flake.py interop_foreign_writer --runs 20` → rezultatas 3 fazės ataskaitoje.
- [ ] **10.4** Galutinė peržiūra: `/code-review high` visam šio plano diff'ui; Critical / Important — vienas taisymų
  praėjimas su TDD.
- [ ] **10.5** Commit: `docs(operations): operator guide and quick start`.

## Eiliškumas ir sustojimai

Task 1 → 2 → 4 (pirmas apčiuopiamas rezultatas: `vramflow-run cfg.json` WSL2 duoda TS bet kuriuo adresu) → 5 → 6 → 3 →
7 → 8 (**sustojimas:** autoriaus leidimas Docker atvaizdui) → 9 → 10. Task 3 ir 7 nepriklauso nuo 4–6.
