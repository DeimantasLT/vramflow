# vramflow-verify — architektūra ir verifikacijos planas

- **Data:** 2026-11-10
- **Tikslas:** įrodymais (ne teiginiais) parodyti, kad MXL SDK su GPU grain payload'ais (atviras PR `dmf-mxl/mxl#653`) vienoje mašinoje veikia teisingai ir stabiliai.
- **Rezultatas:** nepriklausomas įrankis `vramflow-verify` + vienos komandos harness `run-all.sh`, sukuriantis įrodymų paketą su PASS/FAIL kiekvienam kriterijui.
- **Skaitytojas:** inžinierius, kuris tai įgyvendins. Dokumentas turi būti pakankamas koduoti be spėjimų.

Žymėjimas: **[patikrinta: šaltinis]** — perskaityta šaltinyje 2026-11-10 (failas:eilutė PR SHA `6dddc91d`, jei nenurodyta kitaip). **[prielaida]** — nepatikrinta, turi būti patvirtinta vykdant (nurodyta, kuriame etape).

---

## 0. Santrauka ir svarbiausi radiniai skaitant kodą

PR #653, head `thomastrue/mxl:feature/cuda-grain-payloads`, SHA `6dddc91d92f56c3c9d90fa089c95d8f262bc2c92`, 72 failai, būsena open [patikrinta: api.github.com/repos/dmf-mxl/mxl/pulls/653].

Skirtumai nuo to, kas buvo žinoma prieš skaitant kodą (**įgyvendintojas privalo remtis šia lentele, ne užduoties aprašu**):

| # | Radinys | Šaltinis |
|---|---|---|
| R1 | CMake vėliavėlė yra **`MXL_ENABLE_CUDA_PAYLOAD`** (ne `MXL_ENABLE_CUDA`). Numatyta OFF. | [patikrinta: `CMakeLists.txt:7`, `lib/CMakeLists.txt:83`, `lib/tests/CMakeLists.txt:135`, `tools/mxl-gst/CMakeLists.txt:124`] |
| R2 | Be CUDA toolkit'o build'as **nebetyli** — `find_package(CUDAToolkit REQUIRED)` nutraukia konfigūraciją. Recenzento pastaba „silently drops" šiam SHA nebegalioja. | [patikrinta: `lib/payload-backends/cuda-linear/CMakeLists.txt:7-9`] |
| R3 | `mxl-internal-tests` link klaida (`mxlCreateInstance` per `dladdr`) buvo `PayloadBackendRegistry.cpp`; šio failo nebėra, loader'is dabar ima `dladdr(&locationAnchor)`. Tikėtina, kad sutaisyta — **patvirtinti F1**. | [patikrinta: `lib/internal/src/PayloadBackend.cpp:47`; PR komentaras danbouzaSWS 2026-10-16] / [prielaida: kad link'inasi] |
| R4 | Catch2 žymos: **`[cuda payload]`**, **`[cuda container]`**, **`[payload backend]`**, **`[mxl flows][grain storage]`**. Žymų `[cuda]` ir `[payload-plugin]` šiame SHA nėra. | [patikrinta: `lib/tests/cuda/test_cuda_payload.cpp:229,297,313,340,377,410,448`; `lib/internal/tests/test_payload_backend.cpp`; `lib/tests/test_grain_storage.cpp`] |
| R5 | CUDA testai be GPU **ne krenta, o SKIP'inami** (`SKIP("No CUDA device.")`). „Peer GPU" testas vienos GPU mašinoje visada SKIP. SKIP ≠ PASS — harness privalo tai atskirti. | [patikrinta: `test_cuda_payload.cpp:233,317,344,381,414,452`] |
| R6 | Storage tipo vardas — **`com.nvidia.cuda-device-pointer`**; device pointer'is ir device indeksas skaitomi iš `reserved` baitų per **neinstaliuojamą** header'į `lib/payload-backends/cuda-linear/CudaLinearStorage.h`. | [patikrinta: `CudaLinearStorage.h:18,29-37,43-47,62-78`; `cuda-linear/CMakeLists.txt:58-61` instaliuoja tik `.so`] |
| R7 | Antras writer'is backend flow'ui gauna **`MXL_ERR_CONFLICT`** (ne „prisijungia"). Tas pats nutinka **po `kill -9` paleidus writer'į iš naujo**, kol flow katalogas nepašalintas GC — nes `publishFlowDirectory` nepavyksta ir esamas flow yra v2. | [patikrinta: `lib/internal/src/FlowManager.cpp:418-436`; `flow.h:461-462`] / [prielaida: restart elgsena — patvirtinti F6] |
| R8 | `mxlFlowWriterCommitGrain` su `grain.index != atidarytas` grąžina `MXL_ERR_INVALID_ARG`; `OpenGrainSlot` su `index <= lastCommitted` — `MXL_ERR_INVALID_ARG`. Todėl gedimo injekcija „wrong index" daroma **turinio lygyje** (payload antraštėje), ne per `mxlGrainInfo.index`. | [patikrinta: `lib/internal/src/PosixDiscreteFlowWriter.cpp:140-143,212-215`] |
| R9 | Praleidus indeksą writer'is pažymi praleistus slot'us `MXL_GRAIN_FLAG_INVALID`, `validSlices=0`, **bet GPU atmintyje lieka senas turinys**. Reader'is gauna `MXL_STATUS_OK` + INVALID vėliavėlę. | [patikrinta: `PosixDiscreteFlowWriter.cpp:145-162`; `PosixDiscreteFlowReader.cpp:234-235`] |
| R10 | Reader'is **netikrina**, kad `grainInfo.index == prašytas` — grąžina slot'o `index % grainCount` antraštę. Skaitomas langas: `index >= head - grainCount + 2` → skaitomi `grainCount - 1` grain'ai. | [patikrinta: `PosixDiscreteFlowReader.cpp:218-256`] |
| R11 | `mxl-gst-cuda-sink` naudoja `autovideosink` — **reikia ekrano**. Automatizacijai: `xvfb-run`. | [patikrinta: `tools/mxl-gst/cuda_sink.cpp:98-101`; `docs/Tools.md:594`] |
| R12 | Backend'as `dlopen("libcuda.so.1")` ir prieš alokaciją tikrina `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED`; jei 0 → `MXL_ERR_UNSUPPORTED_OPERATION`. Tai tiksliai tas kelias, kurį testuoja žingsnis 0. | [patikrinta: `CudaLinearBackend.cpp:366,398-405`] |
| R13 | `docs/PayloadStorage.md` skyrius „Future work" vis dar aprašo `cudaMalloc` + `cudaIpcGetMemHandle`; **faktinis kodas ir `docs/PluggablePayloads.md` — VMM + POSIX FD**. Tikėti kodu. | [patikrinta: `CudaLinearBackend.cpp:10-18,216`; `docs/PluggablePayloads.md:88-94`; `docs/PayloadStorage.md:272-275`] |
| R14 | WSL-Ubuntu CUDA repo **deprecated**, bet jame yra `cuda-toolkit-13-2` (13.2.0/13.2.1/13.2.2) **ir `cuda-toolkit-13-3`** — būtina pin'inti versiją, nediegti `cuda-toolkit`/`cuda` meta paketų. | [patikrinta: developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/Packages; docs.nvidia.com/cuda/wsl-user-guide §3.4.2] |

Recenzento (danbouzaSWS, native Linux, RTX 5070 Ti, senesnė CUDA-IPC versija) atskaitos skaičiai palyginimui: device flow commit→verified P50 110–126 µs, P99 207–327 µs (1 flow); host flow su D2H+H2D P50 313–696 µs [patikrinta: PR #653 komentaras 2026-10-16]. Mūsų WSL2 slenksčiai sąmoningai laisvesni (§6).

---

## 1. Patikrinta MXL semantika šiame SHA

| Teiginys | Statusas |
|---|---|
| Flow = `<domain>/<uuid>.mxl-flow/`; backend flow kataloge papildomai `payload.json` (libmxl), `payload.cuda.json` (backend: version, export=`posix-fd`, deviceUuid, slotCount, stride, allocSize) ir `payload.sock`. | [patikrinta: `docs/PluggablePayloads.md:43-55`; `CudaLinearBackend.cpp:74-82,978-983`; `test_cuda_payload.cpp` socket kelias] |
| Backend flow: flow data version **2** (`FLOW_DATA_VERSION_PAYLOAD_BACKEND`), grain failuose tik 8 KiB antraštė. Host flow lieka v1. | [patikrinta: `lib/internal/include/mxl-internal/Flow.hpp:21,27`; `FlowManager.cpp:372`] |
| Viena CUDA VMM alokacija visam ring'ui (`cuMemCreate`, `CU_MEM_ALLOCATION_TYPE_PINNED`, `CU_MEM_LOCATION_TYPE_DEVICE`, dydis apvalinamas iki `cuMemGetAllocationGranularity(MINIMUM)`), eksportas `cuMemExportToShareableHandle(..., CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)`; slot'ai lygiuojami 256 B. | [patikrinta: `CudaLinearBackend.cpp:244-286,417-420`; `docs/PluggablePayloads.md:90`] |
| Reader'is `MapSlots` metu jungiasi prie `payload.sock`, gauna FD per `SCM_RIGHTS`, importuoja `cuMemImportFromShareableHandle`. Tame pačiame procese kaip writer'is — dalijasi per vidinį registrą (rodyklės sutampa). | [patikrinta: `CudaLinearBackend.cpp:296-322,1089-1108`; `test_cuda_payload.cpp:283`] |
| Socket'as FD duoda tik procesui, kurio uid = socket'o savininkas, arba gid su `S_IWGRP`, arba kai `S_IWOTH`. | [patikrinta: `CudaLinearBackend.cpp:733-749`] |
| Plugin'o paieška: `MXL_PAYLOAD_PLUGIN_PATH` (absoliutūs keliai, `:`), tada libmxl katalogas. | [patikrinta: `docs/PluggablePayloads.md:59-61`; `PayloadBackend.cpp:44-59`] |
| Writer opcijos: `{"payload":{"backend":"cuda-linear","deviceUuid":"GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}}`. Be `deviceUuid` → `MXL_ERR_INVALID_ARG`; nematomas UUID → `MXL_ERR_NOT_FOUND`; data flow → `MXL_ERR_UNSUPPORTED_OPERATION`. Reader'is: `""` arba `{"payload":{"deviceUuid":"…"}}`; `backend` raktas reader'iui → `MXL_ERR_INVALID_ARG`. | [patikrinta: `test_cuda_payload.cpp:478-501`; `lib/internal/src/FlowOptionsParser.cpp:120-170`] |
| `mxlFlowWriterOpenGrain` / `mxlFlowReaderGetGrain*` ne-host storage → `MXL_ERR_UNSUPPORTED_OPERATION`. | [patikrinta: `PosixDiscreteFlowWriter.cpp:90-93`; `PosixDiscreteFlowReader.cpp:117-120`; `test_cuda_payload.cpp:269,287`] |
| `payloadLocation`/`deviceIndex` deprecated, visada 0 / -1. | [patikrinta: `lib/include/mxl/flowinfo.h:85-97`] |
| Ring dydis: `grainCount = historyDuration · num / (1e9 · den)`, `historyDuration` numatytas 200 ms, raktas `urn:x-mxl:option:history_duration/v1.0` domeno `options.json`. 50p → 10 slot'ų, skaitomi 9. | [patikrinta: `lib/internal/src/Instance.cpp:38,124,302,433`] |
| Timeout'as blokuojančiame skaityme = `MXL_ERR_OUT_OF_RANGE_TOO_EARLY`; `FLOW_INVALID` nustatomas tik po TOO_EARLY (inode palyginimas). | [patikrinta: `PosixDiscreteFlowReader.cpp:187-200,295-306`; `flow.h:706-707`] |
| `mxlGetTime()` = `CLOCK_TAI`; commit įrašo `lastWriteTime` (TAI) ir didina `syncCounter` + futex wake. | [patikrinta: `lib/src/time.cpp:11-14`; `PosixDiscreteFlowWriter.cpp:217-233`] |
| GC: flow aktyvus, jei ant data failo laikomas `flock`; `mxlGarbageCollectFlows` / `mxl-info -g`. | [patikrinta: `Instance.cpp:373-404`; `mxl.h:92-93`; `docs/Tools.md:23`] |
| flow_def: privalomas netuščias `label`, grouphint žyma; pavyzdys `lib/tests/data/v210_flow.json`. | [patikrinta: `lib/internal/src/FlowParser.cpp:172,250-254`] |
| v210 eilutės ilgis `ceil(w/48)·128` (1920 → 5120 B; 1080p grain = 5 529 600 B = 1 382 400 × uint32). | [prielaida: formulė iš MXL v1.1 žinių; `getV210LineLength` kūnas neskaitytas — **tikrinti vykdymo metu**: `info.grainSize == 5529600`] |
| Writer'iai laiko `flock(LOCK_SH)`, reader'iai lock-free; handle'ai ne thread-safe. | [prielaida: MXL v1.1 semantika; GC kodas (`LOCK_EX|LOCK_NB`) su tuo dera] |
| Indeksas = round(TAI_ns·num/(1e9·den)). | [prielaida: v1.1; įrankis **nenaudoja** savo formulės darbui — tik `mxlGetCurrentIndex`/`mxlTimestampToIndex`/`mxlIndexToTimestamp`] |

---

## 2. Žingsnis 0 — `cuda-fd-probe` (be MXL). Rizika Nr. 1

**Kodėl:** NVIDIA WSL gidas nedokumentuoja `cuMemExportToShareableHandle` su POSIX FD ant WSL2. Žinomi WSL ribojimai: Unified Memory nepilnas ir concurrent CPU/GPU access nepalaikomas, pinned system memory ribota, NVML dalinis (GPU utilization, active compute process neveikia), nėra GPU filtravimo pagal indeksą [patikrinta: docs.nvidia.com/cuda/wsl-user-guide §4.1–4.2]. Apie VMM/FD eksportą gide **nieko** — nei „veikia", nei „neveikia" [patikrinta: tas pats puslapis].

**Kas:** `probe/cuda_fd_probe.cpp` — vienas failas, tik CUDA **driver API** (`cuda.h`, link'as `-lcuda`; jokio nvcc kernel'io, kad nesėkmės priežastis būtų vienareikšmė). Kartoja tiksliai backend'o kvietimų seką (`CudaLinearBackend.cpp:244-322,387-452`).

**Procesų modelis:** `main` sukuria `socketpair(AF_UNIX, SOCK_STREAM)` ir `fork()` **prieš bet kokį CUDA kvietimą** (CUDA po fork'o neveikia). Tėvas = eksportuotojas, vaikas = importuotojas. Antras režimas `--role exporter|importer --socket <kelias>` — du nepriklausomai paleisti procesai per vardinį Unix socket'ą `/dev/shm/…` (kaip MXL).

**Žingsniai (kiekvienas rašo `{"step","call","curesult","curesult_name","ok"}` į stdout JSON Lines):**

| ID | Procesas | Veiksmas | PASS sąlyga |
|---|---|---|---|
| P0.1 | abu | `cuInit(0)`, `cuDeviceGetCount`, `cuDeviceGet(0)`, `cuDeviceGetName`, `cuDeviceGetUuid`, `cuDeviceTotalMem` | `CUDA_SUCCESS`, ≥1 įrenginys |
| P0.2 | exp | `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)` | == 1 |
| P0.3 | exp | `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED)` | == 1 |
| P0.4 | exp | `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`; `cuMemGetAllocationGranularity(MINIMUM)` su prop `{PINNED, requestedHandleTypes=POSIX_FD, location={DEVICE,0}}` | SUCCESS, granularity > 0 (užrašyti reikšmę) |
| P0.5 | exp | `cuMemCreate(size = roundup(55 296 000, gran))`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess(READWRITE)` | SUCCESS |
| P0.6 | exp | užpildo: `cuMemcpyHtoD` su host buferiu `b[i] = (uint8)(i*131 + 7)`; `cuCtxSynchronize` | SUCCESS |
| P0.7 | exp | `cuMemExportToShareableHandle(&fd, h, POSIX_FD, 0)`; `fstat(fd)` | SUCCESS, fd ≥ 0 |
| P0.8 | exp→imp | `sendmsg` su `SCM_RIGHTS` (fd + alloc dydis); imp `recvmsg` | gautas fd ≥ 0 |
| P0.9 | imp | `cuMemImportFromShareableHandle(&h2, (void*)(uintptr_t)fd, POSIX_FD)`, `close(fd)`, reserve/map/setAccess | SUCCESS |
| P0.10 | imp | `cuMemcpyDtoH` viso dydžio, palyginti **kiekvieną baitą** su formule | 0 neatitikimų |
| P0.11 | imp→exp | imp perrašo `b[i] = (uint8)(i*17 + 3)`, sync, praneša; exp nuskaito ir lygina | 0 neatitikimų (abiem kryptim ta pati fizinė atmintis) |
| P0.12 | abu | exp: unmap/addressFree/release ir **išeina**; imp po 500 ms vėl skaito ir lygina | 0 neatitikimų, jokios CUDA klaidos (atmintis gyva, kol yra importas — MXL writer'io crash scenarijus) |
| P0.13 | abu | ciklas ×200: create→export→send→import→verify→release abiejose pusėse | FD skaičius (`/proc/self/fd`) pradžioje == pabaigoje abiejuose; `cuMemGetInfo` free pabaigoje ≥ pradžios − 16 MiB |
| P0.14 | info | `cudaIpcGetMemHandle` kelias (legacy IPC) — tik informacinis, į sprendimą neįeina | užrašyti rezultatą |

Išėjimo kodas: 0 = P0.1–P0.13 PASS; 10 = P0.2/P0.3 atributas 0; 11 = create/export klaida; 12 = import/map klaida; 13 = duomenų neatitikimas; 14 = nuotėkis; 1 = kita.

**Sprendimų lentelė:**

| Rezultatas | Ką darome |
|---|---|
| 0 | Tęsiame visą planą. |
| 10 / 11 / 12 (FD eksportas/importas WSL2 nepalaikomas) | **Pivot A.** Ataskaitoje: „PR #653 cuda-linear backend'as ant WSL2 (driver 596.49, kernel 6.6.87.2) neveikia: `<kvietimas>` → `<CUresult>`". Vykdome tik: (a) upstream ne-CUDA testus (`[payload backend]`, `[grain storage]` — jie eina per `test-file` backend'ą be GPU [patikrinta: `docs/PayloadStorage.md:65-77,127`]), (b) F2 kaip įrodymą, kad CUDA testai krenta/skip'inami su ta pačia klaida, (c) bazinį matavimą §7 (host flow + `cudaMemcpy`), (d) sanitizer'ius mūsų kernel'iams. GPU kelio teisingumo išvada: **NEĮRODYTA šioje mašinoje**; rekomendacija — native Linux mašina su NVIDIA GPU. [prielaida: kad tokia prieinama] |
| 13 (baitai nesutampa) | **STOP.** Tai driver'io lygio duomenų korupcija; MXL testuoti beprasmiška. Surinkti įrodymus, pranešti NVIDIA/PR autoriui. |
| 14 (nuotėkis) | Tęsiame, bet soak'o slenksčiams pridedame žymą „žinomas driver'io nuotėkis X B/ciklą"; F9 vertinamas atskaičius probe bazę. |

Žingsnio 0 kaina: ~150 eilučių C++, <1 val. **Daroma pirmiausia, prieš MXL build'ą.**

---

## 3. Repo išdėstymas (`vramflow`)

MXL šaltinis **už** mūsų repo ribų: `~/src/mxl-pr653` (WSL Linux FS). Mūsų repo fiziškai yra Windows diske (`/mnt/c/…/vramflow` — **kelyje yra tarpas**), todėl: build katalogas `~/build/vramflow`, įrodymai rašomi į `~/evidence/` ir pabaigoje kopijuojami į `evidence/`; visi skriptai kabutėse cituoja kelius. [prielaida: 9p našumas ir tarpai — įprastos WSL problemos]

```
vramflow/
├─ CMakeLists.txt               # C++20 + CUDA; opcijos žemiau
├─ CMakePresets.json            # wsl-release, wsl-asan
├─ vcpkg.json                   # catch2, cli11 (tas pats builtin-baseline kaip MXL: 9e44ec0e…)
├─ config/
│  ├─ thresholds.json           # VISI slenksčiai (§6) — ne hardcode
│  └─ matrix.json               # N sąrašas, trukmės, formatas, soak
├─ probe/cuda_fd_probe.cpp      # §2
├─ src/
│  ├─ common/
│  │  ├─ pattern.hpp            # word(), fold64 — __host__ __device__ inline
│  │  ├─ header.hpp             # GrainHeader encode/decode/validate
│  │  ├─ crc32c.hpp             # programinis CRC32C, host+device
│  │  ├─ index_math.hpp         # langas, gap skaičiavimas (grynos funkcijos)
│  │  ├─ histogram.hpp          # log-linear latency histograma
│  │  ├─ csv.hpp  json.hpp      # rašytojai be priklausomybių
│  │  ├─ flowdef.hpp            # flow_def JSON generatorius
│  │  └─ clock.hpp              # TAI/MONO, TAI offset, clock-step detektorius
│  ├─ gpu/
│  │  ├─ kernels.cu  kernels.hpp  # fill, finalize_header, verify, corrupt
│  │  └─ gpu_ctx.cpp            # įrenginys pagal UUID, stream, mažas pinned rezultato buferis
│  ├─ mxlio/
│  │  ├─ slots.hpp              # SlotView: device ptr (cuda-linear) arba host ptr
│  │  ├─ writer.cpp             # §4.4
│  │  └─ reader.cpp             # §4.5
│  ├─ modes/  writer_main.cpp reader_main.cpp multi_main.cpp gpu_info.cpp hog.cpp
│  └─ main.cpp                  # CLI11 subkomandos
├─ tests/
│  ├─ unit/                     # Catch2, be GPU (§5.2)
│  ├─ gpu/                      # Catch2, žyma [gpu] (§5.3)
│  ├─ selftest/selftest.sh      # gedimų injekcijos (§5.1)
│  └─ tools/v11_reader_probe.cpp # F4, kompiliuojamas prieš v1.1.0 (neprivalomas)
├─ scripts/
│  ├─ setup-wsl.md              # §9 žingsniai (žmogui; nieko nediegia automatiškai)
│  ├─ build-mxl.sh              # checkout SHA + build + install
│  ├─ run-all.sh                # §8
│  ├─ stages/NN-*.sh
│  ├─ lib/common.sh             # run_step(), evidence keliai, timeout'ai
│  └─ report.py                 # python3 stdlib; CSV → report.md/report.json
├─ docs/architektura/           # šis dokumentas
└─ evidence/                    # .gitignore (išskyrus galutinius report.md)
```

**CMake kintamieji (konfigūruojami, ne hardcode):**

| Kintamasis | Numatyta | Paskirtis |
|---|---|---|
| `MXL_ROOT` | `~/src/mxl-pr653/install/Linux-GCC-Release` | `find_package(mxl CONFIG REQUIRED)` → `mxl::mxl` [patikrinta: `docs/Building.md` „Using with CMake"; installDir `CMakePresets.json:13`] |
| `MXL_SOURCE_DIR` | `~/src/mxl-pr653` | include `lib/payload-backends/cuda-linear` (dėl `CudaLinearStorage.h`, R6) |
| `MXL_V11_ROOT` | tuščia | jei nurodyta — statomas `v11_reader_probe` (F4) |
| `MXLV_ENABLE_SANITIZERS` | OFF | `-fsanitize=address,undefined` host kodui |

Link'inimas: `mxl::mxl` + `CUDA::cudart_static` — tas pats modelis kaip PR testų (procese gyvena dvi statinės cudart kopijos: plugin'e ir mūsų exe) [patikrinta: `lib/tests/cuda/CMakeLists.txt:24-28`]. Plugin'as randamas šalia `libmxl.so` instaliacijos `lib/` kataloge [patikrinta: `cuda-linear/CMakeLists.txt:58-61`]; jei ne — `MXL_PAYLOAD_PLUGIN_PATH`.

Kompiliavimo laiko apsaugos mūsų kode: `static_assert(sizeof(mxlGrainStorageLayout)==512)`, `sizeof(mxlGrainStorage)==256`, `sizeof(mxlGrainInfo)==4096`, `sizeof(GrainHeader)==64` [patikrinta dydžiai: `flow.h:151,279,324`; `docs/PayloadStorage.md:21-27`].

---

## 4. `vramflow-verify` dizainas

### 4.1 Subkomandos

| Komanda | Paskirtis |
|---|---|
| `gpu-info` | JSON: ordinal, vardas, UUID formatu `GPU-8-4-4-4-12` (iš `cudaGetDeviceProperties().uuid`, kaip `CudaTestUtils.hpp:16-38`), total/free VRAM, VMM ir POSIX-FD atributai. **UUID imamas iš čia, ne iš `nvidia-smi -L`** [prielaida: `nvidia-smi -L` WSL'e UUID rodo, bet nepasikliaujame]. |
| `writer` | vienas flow, vienas procesas |
| `reader` | vienas flow, vienas procesas |
| `multi --role writer\|reader --flows N` | N flow'ų viename procese, **po vieną giją kiekvienam handle'ui** |
| `hog --leave-free-mb X` | užima VRAM (`cudaMalloc` blokais po 64 MiB), laiko iki SIGTERM (F8) |

**Kodėl `multi`:** kiekvienas CUDA procesas turi savo kontekstą (~150–400 MB VRAM [prielaida]) ir WSL'e nėra MPS, tad GPU dalijamas laiko pjūviais tarp procesų. 32 writer + 32 reader procesai netilptų į 8 GB ir matuotų kontekstų perjungimą, ne MXL. Skalavimas daromas **2 procesais** (1 multi-writer + 1 multi-reader), kad kelias tarp procesų (FD importas) liktų tikras.

Bendros opcijos: `--domain DIR` (privaloma, tikrinama `mxlIsTmpFs` [patikrinta: `mxl.h:109-110`]), `--device-uuid`, `--storage cuda-linear|host`, `--width 1920 --height 1080 --rate 50/1`, `--flows N`, `--flow-index-base K`, `--run-seed U64`, `--duration SEC`, `--warmup SEC` (statistika nuo warmup pabaigos), `--out DIR`, `--read-timeout-ms 100`, `--stall-timeout-ms 2000`, `--reopen-timeout-ms 10000`, `--gc-before-create`, `--host-crosscheck-every K` (reader; 0 = išjungta), `--cycles C` (F9: kurti/naikinti handle'ą C kartų), `--exit-on-first-error`.

Deterministiniai identifikatoriai: flow `n` → id `4d584c56-0000-4000-8000-%012x` (n), `label = "vramflow-verify flow n"`, grouphint `"vramflow-verify-<run-seed hex>:Video n"`, `flow_seed = splitmix64(run_seed ^ n)`. flow_def pagal `lib/tests/data/v210_flow.json` šabloną (media_type `video/v210`, `interlace_mode: progressive`, components Y/Cb/Cr 10 bit).

Išėjimo kodai: `0` švaru; `1` naudojimo klaida; `2` aplinkos/MXL/CUDA fatali klaida; `3` **aptikta verifikacijos klaida** (mismatch, checksum, header, index, content-index); `4` tik pajėgumo klaidos (gap/too_late/stall be verifikacijos klaidų); `5` `MXL_ERR_CONFLICT` kuriant writer'į.

### 4.2 Deterministinis turinys

Payload = `n = grainSize/4` little-endian `uint32` žodžių (1080p v210: 1 382 400). Žodžiai `[0,16)` — antraštė (64 B), `[16,n)` — raštas.

```cpp
// pattern.hpp — __host__ __device__
inline uint64_t splitmix64(uint64_t x){ x+=0x9E3779B97F4A7C15ull; uint64_t z=x;
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; return z^(z>>31); }
inline uint32_t word(uint64_t seed, uint64_t index, uint32_t i){
  uint64_t k = seed ^ (index * 0x9E3779B97F4A7C15ull);
  uint32_t x = uint32_t(k ^ (k>>32)) + i * 0x9E3779B1u;
  x^=x>>16; x*=0x85EBCA6Bu; x^=x>>13; x*=0xC2B2AE35u; x^=x>>16;   // murmur3 fmix32
  return x & 0x3FFFFFFFu; }                                        // legalus v210 (bitai 30-31 = 0)
// fold64 = Σ_{i=16}^{n-1} uint64(w_i) * (2*i+1)  (mod 2^64) — priklauso nuo pozicijos, sudedamas bet kokia tvarka
```

Auksinės reikšmės unit testams (apskaičiuota nepriklausomu Python skriptu 2026-11-10):

| Išraiška | Reikšmė |
|---|---|
| `splitmix64(0)`, `splitmix64(1)` | `e220a8397b1dcdaf`, `910a2dec89025cc1` |
| `word(0,0,16)` | `197b8fee` |
| `word(1,2,16)` | `2b573fc4` |
| `word(0x0123456789ABCDEF, 88123456789, 17)` | `09dd12f4` |
| `word(0x0123456789ABCDEF, 88123456789, 1382399)` | `240e5868` |
| `fold64(seed=1,index=2,n=64)` | `000001acda15a704` |
| `fold64(0x0123456789ABCDEF, 88123456789, n=1382400)` | `9f7fd5fa9f07281b` |
| `crc32c("123456789")` | `e3069283` |

Antraštė (64 B, little-endian, `#pragma pack(1)` nereikia — natūralus lygiavimas):

| Offset | Tipas | Laukas | Pastaba |
|---|---|---|---|
| 0 | u32 | `magic` | `0x4D584C56` („MXLV") |
| 4 | u16 | `version` | 1 |
| 6 | u16 | `header_size` | 64 |
| 8 | u64 | `flow_seed` | |
| 16 | u64 | `grain_index` | indeksas, kuriam sugeneruotas turinys |
| 24 | u64 | `origin_tai_ns` | writer'io `mxlGetTime()` prieš pat fill kernel'io paleidimą |
| 32 | u64 | `payload_fold64` | apskaičiuota GPU **prieš** bet kokią injekciją |
| 40 | u32 | `payload_words` | n |
| 44 | u32 | `writer_pid` | |
| 48 | u64 | `seq` | writer'io commit'ų skaitiklis nuo 0 (restart'as matomas kaip seq šuolis atgal) |
| 56 | u32 | `reserved` | 0 (**jokio „čia injekcija" bito** — detektorius neturi būti informuojamas) |
| 60 | u32 | `header_crc32c` | CRC32C baitų 0..59 |

`mxlGrainInfo.reserved` nenaudojamas („Do not use", `flow.h:150-151`), nors commit'as jį ir nukopijuoja į shm (`PosixDiscreteFlowWriter.cpp:221`).

### 4.3 CUDA kernel'iai (`kernels.cu`)

Visi dirba vienos gijos `cudaStream_t`; sinchronizacija — `cudaStreamSynchronize(stream)` (ne `cudaDeviceSynchronize`, kad N gijų neblokuotų viena kitos; PR įrankiai naudoja `cudaDeviceSynchronize` [patikrinta: `cuda_testsrc.cpp:326`] — mūsų variantas griežtesnis tik tuo, kad sinchronizuoja vienintelį stream'ą, rašantį į tą slot'ą).

| Kernel | Parametrai | Veiksmas |
|---|---|---|
| `k_fill` | `uint32_t* slot, seed, index, n, uint64_t* d_fold` | grid = ceil((n−16)/256) blokų × 256 gijų; rašo `slot[i]=word(...)` kai i∈[16,n); bloko redukcija per `__shared__` + vienas `atomicAdd(unsigned long long)` į `*d_fold` (prieš tai `d_fold` nunulinamas `cudaMemsetAsync`) |
| `k_finalize_header` | `uint32_t* slot, GrainHeader h (by value), const uint64_t* d_fold` | 1 gija: įrašo `h.payload_fold64=*d_fold`, skaičiuoja CRC32C, rašo 64 B į `slot[0..16)` |
| `k_corrupt` | `uint32_t* slot, uint32_t pos, uint32_t xor_mask` | 1 gija: `slot[pos] ^= mask` (injekcijai) |
| `k_verify` | `const uint32_t* slot, seed, index, n, VerifyResult* d_res` | lygina kiekvieną žodį i∈[16,n) su `word()`; `atomicAdd` į `mismatch_words`; `atomicMin` į `first_bad_word`; skaičiuoja faktinį `fold64`; gija 0 nukopijuoja 64 B antraštę į `d_res->header` |

`VerifyResult { u64 mismatch_words; u64 actual_fold64; u32 first_bad_word (init 0xFFFFFFFF); u32 pad; u8 header[64]; }` — 96 B; nunulinamas prieš kiekvieną kvietimą; parsiunčiamas `cudaMemcpyAsync(D2H)` į mažą buferį (bandyti `cudaHostAlloc` 4 KiB; nepavykus — pageable; WSL pinned ribojimas šiam dydžiui nereikšmingas [prielaida]).

Vienam grain'ui reader'is daro **vieną** kernel'į + vieną 96 B D2H + vieną stream sync.

### 4.4 Writer'is — tiksli MXL kvietimų seka

Signatūros: `flow.h` nurodytos eilutės. Šablonas — `lib/tests/cuda/cuda_writer_helper.cpp:113-175` ir `tools/mxl-gst/cuda_testsrc.cpp:263-330`.

| # | Kvietimas | Klaidų tvarkymas |
|---|---|---|
| W1 | `mxlCreateInstance(domain, "")` (`mxl.h:83-84`) | `NULL` → exit 2 |
| W2 | (jei `--gc-before-create`) `mxlGarbageCollectFlows(instance)` (`mxl.h:92-93`) | ne OK → įspėjimas, tęsti |
| W3 | `mxlCreateFlowWriter(instance, flowDef, options, &writer, &cfg, &created)` (`flow.h:469-471`); `options` = cuda-linear JSON arba `NULL` (host) | `INVALID_ARG` → exit 2 (blogas def/opcijos); `UNSUPPORTED_OPERATION` → exit 2 su užuomina „plugin nerastas arba GPU neeksportuoja FD — žr. probe"; `NOT_FOUND` → exit 2 („UUID nematomas"); **`CONFLICT` → exit 5**; kita → exit 2. Jei `created==false` (įmanoma tik host) → exit 2, nebent `--allow-join` |
| W4 | `mxlFlowWriterGetStorageLayout(writer, &layout)` (`flow.h:782-783`) | ne OK → exit 2. Tikrinti: `version==1`, `size==512`, `storageType` == `MXL_CUDA_LINEAR_STORAGE_TYPE` arba `MXL_PAYLOAD_STORAGE_HOST_POINTER`, `slotCount == cfg.discrete.grainCount`, `planeCount==1`, `planes[0].pitch >= cfg.discrete.sliceSizes[0]` |
| W5 | `mxlFlowWriterMapSlots(writer, layout.slotCount, slots.data())` (`flow.h:801-802`) | ne OK → exit 2 |
| W6 | pakartotinai W4 (device indeksas galioja po map'o — taip daro `cuda_sink.cpp:248-250`); `cudaSetDevice(mxlCudaLinearLayout(layout).deviceIndex)` **toje gijoje, kuri dirbs**; `cudaStreamCreate` | CUDA klaida → exit 2 |
| W7 | sanity: kiekvienam slot'ui `p = mxlCudaLinearDevicePointer(slots[s])`; `cudaPointerGetAttributes(p).type == cudaMemoryTypeDevice` (kaip `test_cuda_payload.cpp:262`); `p % 256 == 0`; rodyklės skirtingos ir `p[s+1]-p[s] >= grainSize` | pažeidimas → exit 3 (`slot_layout_error`) |
| W8 | `index = mxlGetCurrentIndex(&cfg.common.grainRate)` (`time.h:26-27`) | `MXL_UNDEFINED_INDEX` → exit 2 |
| **ciklas** | | |
| W9 | `mxlFlowWriterOpenGrainSlot(writer, index, &info, &slot)` (`flow.h:818-819`) | `INVALID_ARG` (index ≤ paskutinis commit'intas → laikrodis šoko atgal): `clock_backward++`, `index = last+1`, kartoti; kita → exit 2. Tikrinti `slot < slotCount`, `info.index == index`, `info.grainSize == planes[0].size` |
| W10 | `t0 = mxlGetTime()`; `k_fill` → `k_finalize_header` (→ injekcijos §5.1) → `cudaStreamSynchronize` | CUDA klaida → `cuda_errors++`, `mxlFlowWriterCancelGrain`, exit 2 |
| W11 | `info.validSlices = info.totalSlices;` `mxlFlowWriterCommitGrain(writer, &info)` (`flow.h:771-772`) | ne OK → exit 2 |
| W12 | `seq++`, `index++`; `now = mxlGetCurrentIndex()`; jei `now > index` → `writer_late_skips += now-index`, `index = now` (praleisti; MXL pats pažymės INVALID, R9); `mxlSleepForNs(mxlGetNsUntilIndex(index, &rate))` (`time.h:35-36,64-65`) | — |
| W13 | SIGINT/SIGTERM (blokuoti `sigprocmask`, tikrinti `sigtimedwait` cikle) → `mxlReleaseFlowWriter`, `mxlDestroyInstance` | — |

**Commit kontraktas:** device darbas baigtas **prieš** W11 [patikrinta: `docs/PluggablePayloads.md:94`; `cuda_testsrc.cpp:325-326`]. Host storage (bazė): `slots[s].host.pointer` [patikrinta: `flow.h:285-289,318`], W10 = `k_fill`+`k_finalize_header` į gijos scratch device buferį, tada `cudaMemcpy(D2H)` į host slot'ą.

Writer'io CSV fiksuoja `gpu_op_us` (W10 trukmė) ir `period_jitter_us` (faktinis commit laikas − `mxlIndexToTimestamp(index)`).

### 4.5 Reader'is — tiksli seka ir atsilikimo politika

Šablonas — `lib/tests/cuda/cuda_reader_helper.cpp:62-128`, `tools/mxl-gst/cuda_sink.cpp:227-325`. Žingsniai žymimi **RD1–RD12** (kad nesipainiotų su radiniais R1–R14 iš §0).

| # | Kvietimas | Klaidų tvarkymas |
|---|---|---|
| RD1 | `mxlCreateInstance(domain, "")` | `NULL` → exit 2 |
| RD2 | `mxlCreateFlowReader(instance, flowId, options, &reader)` (`flow.h:496-497`); options `""` | `FLOW_NOT_FOUND` → kartoti kas 50 ms iki `--wait-flow-ms` (numatyta 10 s), tada exit 2; `INVALID_ARG`/`PERMISSION_DENIED`/kita → exit 2 |
| RD3 | `mxlFlowReaderGetConfigInfo(reader, &cfg)` (`flow.h:553-554`) | ne OK → exit 2 |
| RD4 | `mxlFlowReaderGetStorageLayout(reader, &layout)` (`flow.h:660-661`) | `UNSUPPORTED_OPERATION` → exit 2 („backend nepakraunamas reader'io procese"); tikrinimai kaip W4 |
| RD5 | `mxlFlowReaderMapSlots(reader, layout.slotCount, slots.data())` (`flow.h:687-688`) | `NOT_FOUND` (prie `payload.sock` niekas neklauso → writer'is miręs [patikrinta: `CudaLinearBackend.cpp:819,841`]) → būsena REOPEN; `UNSUPPORTED_OPERATION`/kita → exit 2 |
| RD6 | pakartoti RD4; `cudaSetDevice(deviceIndex)`; `cudaStreamCreate`; sanity kaip W7 | |
| RD7 | `mxlFlowReaderGetRuntimeInfo(reader, &rt)` (`flow.h:565-566`); `index = rt.headIndex` jei writer'is jau rašė, kitaip `mxlGetCurrentIndex` | |
| **ciklas** | | |
| RD8 | `st = mxlFlowReaderGetGrainSlot(reader, index, MXL_GRAIN_VALID_SLICES_ALL, timeoutNs, &info, &slot)` (`flow.h:709-711`); `t_wake = mxlGetTime()` | žr. politiką žemiau |
| RD9 | `mxlFlowReaderGetRuntimeInfo` → jei `rt.headIndex == index` → `t_commit = rt.lastWriteTime` (tikslus commit laikas), kitaip `t_commit` nežinomas | |
| RD10 | `k_verify` + D2H + `cudaStreamSynchronize`; `t_ver = mxlGetTime()` | CUDA klaida → `cuda_errors++`, exit 2 |
| RD11 | `mxlFlowReaderGetRuntimeInfo` → **lap patikra**: jei `rt.headIndex > index + grainCount - 2` → rezultatas atmetamas (`lapped_discarded++`), nes slot'as galėjo būti perrašomas tikrinimo metu (langas iš `PosixDiscreteFlowReader.cpp:227`) | |
| RD12 | vertinimas (§4.6), `index++` | |

**Politika pagal `st` (RD8):**

| `st` | Veiksmas |
|---|---|
| `MXL_STATUS_OK`, `info.flags & MXL_GRAIN_FLAG_INVALID` | `grains_invalid++`, `gap_grains++`; payload **netikrinamas**; `index++` |
| `MXL_STATUS_OK` | RD9–RD12 |
| `MXL_ERR_OUT_OF_RANGE_TOO_EARLY` | `too_early_timeouts++`; **nešokti į priekį** — laukti to paties indekso. Jei laukiama > `--stall-timeout-ms`: `stalls++`, `mxlIsFlowActive(instance, flowId, &active)` (`flow.h:510-511`) → užrašyti; resync `index = max(index, rt.headIndex)`; `gap_grains += skirtumas` |
| `MXL_ERR_OUT_OF_RANGE_TOO_LATE` | `too_late_events++`; `new = rt.headIndex`; `gap_grains += new - index`; `index = new` (šuolis į head) |
| `MXL_ERR_FLOW_INVALID` | `reopens++`; `cudaStreamSynchronize`; `mxlReleaseFlowReader`; būsena REOPEN: kartoti RD2–RD7 kas 100 ms iki `--reopen-timeout-ms`; nepavykus → exit 4. Seni slot'ų aprašai ir rodyklės **nebenaudojami** [patikrinta: `flow.h:675-676`] |
| `MXL_ERR_INVALID_FLOW_READER`, `MXL_ERR_INVALID_ARG`, `MXL_ERR_UNKNOWN`, kita | exit 2 (mūsų arba bibliotekos klaida; `MXL_ERR_UNKNOWN` — užrašyti stderr) |

`timeoutNs` = `--read-timeout-ms` (100 ms = 5 kadrai 50p). Funkcija niekada negrąžina `MXL_ERR_TIMEOUT` [patikrinta: `flow.h:706-707`].

### 4.6 Ką reader'is tikrina kiekvienam grain'ui (skaitikliai)

| Skaitiklis | Sąlyga | Klasė |
|---|---|---|
| `index_mismatch` | `info.index != index` (R10: biblioteka to netikrina) | VERIFIKACIJA |
| `slot_mismatch` | `slot != index % layout.slotCount` arba `slot >= slotCount` | VERIFIKACIJA |
| `size_mismatch` | `info.grainSize != planes[0].size` arba `validSlices != totalSlices` be INVALID vėliavėlės | VERIFIKACIJA |
| `hdr_fail` | magic/version/header_size/`header_crc32c`/`payload_words`/`flow_seed` neteisingi | VERIFIKACIJA |
| `content_index_mismatch` | `hdr.grain_index != index` (pasenęs arba svetimas turinys slot'e) | VERIFIKACIJA |
| `mismatch_grains`, `mismatch_words` | `res.mismatch_words > 0` (lyginama su `word(seed, index, i)` — **prašytu** indeksu) | VERIFIKACIJA |
| `checksum_fail` | `res.actual_fold64 != hdr.payload_fold64` | VERIFIKACIJA |
| `dup` | `hdr.seq == ankstesnio patikrinto seq` arba `hdr.grain_index <=` paskutinis patikrintas | VERIFIKACIJA |
| `seq_break` | `hdr.seq - prev.seq != index - prev.index` ir nebuvo gap/reopen | informacinis (writer restart) |
| `gap_grains` | indeksai, kurių nepavyko patikrinti (INVALID, TOO_LATE, stall) | PAJĖGUMAS |
| `stale_max_ms` | `t_ver - mxlIndexToTimestamp(index)` maksimumas | metrika |
| `host_crosscheck_fail` | kas K-tą grain'ą: visas payload D2H, CPU perskaičiuoja `word()` ir `fold64` **atskira skaliarine implementacija** | VERIFIKACIJA (detektoriaus detektorius) |

Latencijos (histograma 1 µs…10 s, 32 sub-kibirai/oktavai):
- `lat_commit2ver = t_ver − t_commit` (tik kai RD9 davė tikslų `t_commit`) — **pagrindinė metrika**;
- `lat_wake = t_wake − t_commit`; `lat_e2e = t_ver − hdr.origin_tai_ns`; `gpu_op = t_ver − t_wake`.

### 4.7 Išvestis

`<out>/flow-<n>-<role>.csv` — 1 eilutė/s:

```
ts_tai_ns,ts_mono_ns,flow,role,head_index,cur_index,grains_ok,grains_invalid,gap_grains,
too_late_events,too_early_timeouts,stalls,lapped_discarded,reopens,
index_mismatch,slot_mismatch,size_mismatch,hdr_fail,content_index_mismatch,
mismatch_grains,mismatch_words,checksum_fail,dup,seq_break,host_crosscheck_fail,cuda_errors,
lat_c2v_p50_us,lat_c2v_p99_us,lat_c2v_max_us,lat_wake_p50_us,lat_e2e_p50_us,lat_e2e_p99_us,
gpu_op_p50_us,gpu_op_p99_us,stale_max_ms,writer_late_skips,clock_backward,period_jitter_p99_us
```
(skaitikliai kumuliaciniai; latencijos — tos sekundės; writer'iui netaikomi stulpeliai tušti.)

`<out>/proc-<role>.csv` — 1 eilutė/s procesui: `ts_tai_ns,rss_kb,fd_count,threads,cpu_user_ms,cpu_sys_ms,vram_free_mb,vram_total_mb,tai_minus_mono_ns,tai_offset_s` (`/proc/self/statm`, `/proc/self/fd`, `getrusage`, `cudaMemGetInfo`, `adjtimex().tai`).

`<out>/summary-<role>.json`: versijos (`mxlGetVersion` [patikrinta: `mxl.h:70-71`], CUDA runtime/driver versijos, GPU vardas/UUID), argumentai, layout'as (storageType, slotCount, stride, device ptr 0-ojo slot'o), kiekvienam flow — galutiniai skaitikliai, P50/P99/P99.9/max visam periodui po warmup, histogramos kibirai, pirmųjų 10 klaidų įrašai (`index, slot, first_bad_word, expected, got, hdr dump`), `exit_code`, `verdict`.

---

## 5. Testai

### 5.1 Verifikatoriaus savitikra (pirmieji TDD testai — detektorius privalo aptikti)

Writer'io injekcijos; reader'is paleidžiamas be jokių specialių opcijų. Kiekvienas atvejis: 1 flow, 20 s, 50p (≈1000 grain'ų), po 2 s warmup.

| Opcija | Ką daro writer'is | Laukiama reader'io pusėje (PASS sąlyga) |
|---|---|---|
| (jokios) — kontrolė | — | visi VERIFIKACIJOS skaitikliai = 0; exit 0 |
| `--corrupt-every N` (N=50) | po `k_finalize_header`, prieš sync: `k_corrupt(slot, pos = 16 + (index*2654435761 % (n-16)), mask = 1 << (index%30))` — **vienas bitas** | `mismatch_grains == injekcijų sk.` (±1 dėl warmup ribos), `mismatch_words == mismatch_grains`, `checksum_fail == mismatch_grains`, `first_bad_word == pos` kiekvienam; `hdr_fail == 0`; exit 3 |
| `--corrupt-header-every N` | `k_corrupt(slot, pos=4, mask=1)` (seed laukas) | `hdr_fail == injekcijų sk.`; exit 3 |
| `--skip-every N` (N=50) | praleidžia indeksą (W9 kviečiamas su `index+1`) | `grains_invalid == injekcijų sk.`, `gap_grains == injekcijų sk.`, VERIFIKACIJOS skaitikliai 0; exit 4 |
| `--wrong-index-every N` | turinį ir antraštę generuoja indeksui `index-1`, commit'ina kaip `index` | `content_index_mismatch == injekcijų sk.`; `mismatch_grains == injekcijų sk.` (nes lyginama su prašytu); `hdr_fail == 0`; exit 3 |
| `--stale-every N` | W10 visai nevykdo (jokio device rašymo), commit'ina | `content_index_mismatch == injekcijų sk.` (slot'e — turinys iš `index - grainCount`); exit 3 |
| `--commit-before-sync-every N` (+`--late-write-ms 5`) | `k_fill` tik žodžiams `[16, n/2)`, sync, **commit**, `sleep(late-write-ms)`, tada `k_fill` `[n/2, n)` + `k_finalize_header` | ≥ 90 % injekcijų aptinkama kaip `mismatch_grains` arba `hdr_fail` (reader'is pabunda per ≪ 5 ms); exit 3. Slenkstis 90 %, nes tai lenktynės [prielaida] |
| `--dup-every N` | antraštėje `seq` nepadidinamas | `dup == injekcijų sk.`; exit 3 |

Papildomai: reader'is su `--host-crosscheck-every 10` kontrolėje → `host_crosscheck_fail == 0`; su `--corrupt-every 50` → CPU kryžminė patikra aptinka tas pačias pozicijas, kurias GPU.

`tests/selftest/selftest.sh` paleidžia visus atvejus, lygina `summary-reader.json` su laukiamomis reikšmėmis, grąžina 0 tik jei **visi** atvejai elgiasi kaip lentelėje. **Jei savitikra nepraeina — visi tolesni „0 mismatches" rezultatai negalioja ir harness sustoja.**

### 5.2 Unit testai (Catch2, be GPU, be MXL)

| Testas | Tikrina |
|---|---|
| `pattern` | auksinės reikšmės iš §4.2; `word()` ≤ `0x3FFFFFFF`; skirtingi seed/index/i → skirtingos reikšmės (10⁶ imčių kolizijų dalis < 1e-3) |
| `fold64` | auksinės reikšmės; dviejų žodžių sukeitimas keičia sumą; vieno bito pakeitimas keičia sumą |
| `crc32c` | `"123456789"` → `e3069283`; tuščia → `00000000` |
| `header` | encode→decode round-trip; kiekvieno lauko sugadinimas → `validate()` grąžina konkrečią klaidą; `sizeof==64`; offset'ai per `offsetof` |
| `index_math` | lango funkcija `readable(head, idx, grainCount)` sutampa su `PosixDiscreteFlowReader.cpp:227` (lentelė: grainCount=10, head=100 → skaitomi 92..100); gap skaičiavimas po TOO_LATE; `slot = idx % count` |
| `index_math_mxl` (link'inamas su libmxl) | `mxlTimestampToIndex`/`mxlIndexToTimestamp` round-trip 50/1 ir 30000/1001; `mxlIndexToTimestamp(i+1) − mxlIndexToTimestamp(i)` = 20 000 000 ns (50p) |
| `histogram` | žinomų imčių P50/P99/P99.9 ≤ 3,2 % paklaida; merge; tuščia |
| `csv` / `json` | antraštė = §4.7; skaičių formatavimas be locale; JSON escape; išvestis parsinama `python3 -c json.load` |
| `flowdef` | sugeneruotas JSON turi id, format, label ≠ "", grouphint `"<group>:<role>"`, grain_rate, components; (su libmxl) `mxlCreateFlowWriter` jį priima host režimu `/dev/shm` domene |
| `clock` | TAI offset skaitymas; clock-step detektorius ant sintetinės sekos |

### 5.3 GPU testai (Catch2 `[gpu]`; be MXL; `cudaMalloc` buferiai)

| Testas | Tikrina |
|---|---|
| `k_fill` vs host | D2H ir palyginimas su host `word()` visiems n ∈ {64, 1024, 1 382 400}; `d_fold` == host `fold64` |
| `k_finalize_header` | D2H 64 B → `header.validate()` OK; CRC sutampa su host |
| `k_verify` švarus | `mismatch_words==0`, `actual_fold64==hdr.payload_fold64`, `first_bad_word==0xFFFFFFFF` |
| `k_verify` aptinka | 1 bitas pirmame (16), paskutiniame (n−1) ir atsitiktiniame žodyje → `mismatch_words==1`, teisingas `first_bad_word`; 1000 sugadintų žodžių → 1000 |
| `k_verify` kitas indeksas/seed | `mismatch_words` ≈ n−16 |
| nelygus n | n, kuris nesidalija iš 256 (pvz. 1 000 003): be out-of-bounds (paleidžiama ir po `compute-sanitizer memcheck`) |
| stream izoliacija | 8 gijos × savo stream × 500 iteracijų fill+verify → 0 klaidų |

---

## 6. Testų matrica ir PASS/FAIL slenksčiai (ši mašina)

Visi slenksčiai — `config/thresholds.json`. Rezultato būsenos: **PASS / FAIL / NOT_RUN (su priežastimi) / INFO**. NOT_RUN niekada neskaičiuojamas kaip PASS.

### 6.1 Funkciniai (užduotyje „F1..F7" apima 9 punktus — numeruoju F1..F9)

| ID | Testas | Vykdymas | PASS |
|---|---|---|---|
| F1 | Upstream ctest nuosekliai | `ctest --test-dir ~/src/mxl-pr653/build/Linux-GCC-Release -j1 --output-on-failure --output-junit <ev>/ctest.xml` | 0 failed; visi target'ai (įskaitant `mxl-internal-tests`, R3) susilink'ino; skipped sąrašas užfiksuotas |
| F2 | CUDA testai tikrai įvyko | tiesiogiai `mxl-cuda-tests "[cuda payload]" -r junit` + `mxl-internal-tests "[payload backend]"` + `mxl-tests "[grain storage]"` [prielaida: binarų vardai — patikslinti iš build'o] | „Writer and reader in one process", „Reader in another process", „A refused second writer…", „Descriptors in an unknown format…", „Invalid options" — **passed, ne skipped**. „Reader on a peer GPU" ir `[cuda container]` → NOT_RUN (1 GPU / nėra docker) — raportuojama atskirai |
| F3 | GStreamer CUDA testsrc→sink | `mxl-gst-cuda-testsrc -d <dom> -v <flow.json> --device-uuid <uuid>` + `xvfb-run -a mxl-gst-cuda-sink -d <dom> -v <id>` 30 s; turinys ne mūsų rašto, todėl tikrinamas tik `headIndex` judėjimas per `mxl-info` | abu procesai gyvi 30 s, exit 0 po SIGINT; `headIndex` padidėjo ≥ 0,95·30·fps; stderr be `error`; jei Xvfb/`autovideosink` neveikia WSL'e → NOT_RUN su priežastimi (R11) |
| F4 | Senas v1.1.0 reader'is atmeta v2 flow | `v11_reader_probe` (link'intas su v1.1.0 `libmxl.so`, tag `v1.1.0` = `8f3f1096e8` [patikrinta: api.github.com tags]) prieš veikiantį cuda-linear writer'į | `mxlCreateFlowReader` ≠ OK **arba** `mxlFlowReaderGetGrain` ≠ OK. **FAIL**, jei grąžina OK grain'us (tyli klaida — recenzento radinys Nr. 2 senai versijai). NOT_RUN, jei `MXL_V11_ROOT` nenurodytas |
| F5a | Antras writer'is | veikia writer A + reader; paleidžiamas writer B tam pačiam flow id | B exit 5 (`CONFLICT`) per < 5 s; `stat -c %i payload.sock` nepakito; reader'io VERIFIKACIJOS skaitikliai 0 ir `gap_grains` 0 per visą laiką; naujas reader'is po to prisijungia ir patikrina 100 grain'ų |
| F5b | Dviejų writer'ių lenktynės kuriant (**recenzentų netestuota**) | ×50: du writer'iai startuoja vienu metu (`flock` barjeras + `exec`), tas pats flow id, tuščias domenas | kiekvienoje iteracijoje: lygiai vienas gyvas writer'is, kitas exit 5; naujas reader'is patikrina 100 grain'ų su 0 klaidų; domene nelieka `.tmp` katalogų ir socket'ų be klausytojo |
| F6a | Writer `kill -9` | writer + reader; `kill -9` writer'iui | reader'is: nėra crash, nėra CUDA klaidų; `stalls ≥ 1`; `mxlIsFlowActive == false` per ≤ 2 s; paskutinis geras grain'as perskaitytas GPU 100 kartų po crash → 0 mismatch (atmintis gyva per importą, P0.12); VERIFIKACIJOS skaitikliai 0 |
| F6b | Restart be GC | po F6a naujas writer'is tuo pačiu id **be** `--gc-before-create` | **Užfiksuoti faktą** (INFO): laukiama exit 5 `CONFLICT` (R7). Bet koks rezultatas be crash = INFO; crash = FAIL |
| F6c | Restart su GC | naujas writer'is su `--gc-before-create` | flow sukurtas; senas reader'is gauna `FLOW_INVALID` → `reopens == 1` per ≤ `reopen-timeout`; po reopen 500 grain'ų su 0 VERIFIKACIJOS klaidų; naujas `payload.sock` inode ≠ senas |
| F6d | Naujas reader'is prie mirusio flow | po F6a (be restart) paleisti naują reader'į | švari klaida (exit 2 arba 4) per ≤ `wait`+`reopen` timeout; nėra pakibimo, nėra crash |
| F7 | Reader `kill -9` | writer + 2 reader'iai; `kill -9` vienam; ×20 | writer'io `writer_late_skips` nepadidėjo, exit 0 pabaigoje; antro reader'io skaitikliai 0; writer'io `fd_count` po 20 ciklų == pradinis (±0) |
| F8 | VRAM spaudimas | `hog --leave-free-mb 150`; (a) veikiantis flow tęsia 30 s; (b) bandoma sukurti naują flow | (a) VERIFIKACIJOS skaitikliai 0; (b) arba sėkmė, arba **švari** klaida (`mxlCreateFlowWriter` ≠ OK, exit 2), domene nelieka naujo flow katalogo nei temp katalogo, jokio crash. Jei WDDM persimeta į shared memory ir klaidos nėra → INFO „WDDM oversubscription" [prielaida] |
| F9 | FD / atminties nuotėkis | (a) `reader --cycles 1000` (create→layout→map→1 grain verify→release) prieš vieną writer'į; (b) `writer --cycles 200` (create→map→10 grain'ų→release) | (a) reader'io ir **writer'io** `fd_count` pabaigoje == pradžioje; reader'io RSS augimas < 5 MB; `vram_free` pabaigoje ≥ pradžios − 16 MB; (b) tas pats writer'iui; domene 0 likusių flow katalogų |

### 6.2 Sanitizer'iai

| ID | Testas | Vykdymas | PASS |
|---|---|---|---|
| S1 | compute-sanitizer memcheck | `compute-sanitizer --tool memcheck --error-exitcode 86 --target-processes all …` : (a) `tests/gpu` binaras; (b) `multi` in-process writer+reader 1 flow, 320×180 @ 5 fps, 200 grain'ų (sulėtėjimas 10–100× [prielaida] → realaus laiko slenksčiai išjungti `--no-rt-thresholds`); (c) atskiri writer ir reader procesai | (a),(b): 0 klaidų. (c) reader'is su importuota VMM atmintimi: jei memcheck rodo out-of-bounds importuotai sričiai, o (b) švaru → INFO „galimas false positive importuotai atminčiai" [prielaida], ne FAIL |
| S2 | racecheck | tas pats (a),(b), `--tool racecheck` | 0 hazard'ų mūsų kernel'iuose (`__shared__` redukcija) |
| S3 | synccheck | tas pats, `--tool synccheck` | 0 klaidų |
| — | initcheck | **nevykdomas**: „initcheck tool does not support IPC allocations" | [patikrinta: docs.nvidia.com/compute-sanitizer Known Issues] |
| S4 | ASan+UBSan host | MXL: `cmake --preset Linux-Clang-Debug -DMXL_ENABLE_CUDA_PAYLOAD=ON -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`; ctest ne-CUDA testai + mūsų unit testai + selftest kontrolė 1 flow 60 s su `ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=1` | ne-CUDA testai: 0 ataskaitų. Su CUDA: 0 ataskaitų **MXL/mūsų kadruose**; jei ASan nesuderinamas su libcuda WSL'e (neinicijuojasi) → CUDA dalis NOT_RUN [prielaida: `protect_shadow_gap=0` reikalingas] |

Compute Sanitizer WSL2 palaikomas („Added support for WSL2"), bet coredump'ai WSL2 nepalaikomi [patikrinta: docs.nvidia.com/compute-sanitizer Release Notes / Known Issues]. Paketas `cuda-sanitizer-13-2` įeina į `cuda-command-line-tools-13-2` [patikrinta: wsl-ubuntu Packages].

### 6.3 VRAM biudžetas (8 GB)

- Vienas 1080p50 v210 flow: 10 slot'ų × 5 529 600 B = 55 296 000 B → apvalinama iki granularity. Jei granularity = 2 MiB → **54 MiB (56,6 MB)** [patikrinta formulė: `CudaLinearBackend.cpp:251-259`; prielaida: 2 MiB — tikrąją reikšmę duoda P0.4].
- Reader'io importas **nedubliuoja** fizinės atminties (P0.11 tai įrodo).
- `multi` N=32: 32 × 56,6 MB = **1,81 GB** + 2 CUDA kontekstai (~0,3–0,8 GB [prielaida]) + rezultatų buferiai (< 1 MB) ≈ **2,2–2,6 GB**. Windows darbalaukis/WDDM užima ~0,5–1,5 GB [prielaida]. Telpa su > 3 GB atsarga.
- Bazė (host flow) N=32: device scratch 2 × 32 × 5,5 MB = 0,35 GB VRAM; **`/dev/shm` 32 × 55,3 MB = 1,77 GB RAM** — patikrinti `df /dev/shm` etape 00.

### 6.4 Skalavimas, soak, pakartojamumas

Konfigūracija: 1080p50 v210, `multi` writer + `multi` reader (2 procesai), warmup 10 s, matavimas 120 s; N ∈ {1, 4, 8, 16, 32}.

Slenksčiai vienam bandymui (visi flow'ai kartu), **kai N ≤ pajėgumo**:

| Metrika | Slenkstis |
|---|---|
| Visi VERIFIKACIJOS skaitikliai (§4.6) | **= 0** (bet kuriame N; ne nulis = FAIL nepriklausomai nuo apkrovos) |
| `cuda_errors`, `reopens`, `stalls` | = 0 |
| `gap_grains`, `too_late_events`, `grains_invalid`, `writer_late_skips` | = 0 |
| patikrintų grain'ų dalis | ≥ 99,9 % nuo `trukmė × fps × N` (likutis — tik `lapped_discarded`) |
| `lat_commit2ver` P50 | ≤ 1 ms |
| `lat_commit2ver` P99 | ≤ 5 ms |
| `lat_commit2ver` P99.9 | ≤ 20 ms (1 kadras 50p) |
| writer `period_jitter` P99 | ≤ 5 ms |
| `clock_step_events` | = 0 (kitaip bandymas kartojamas; žr. rizika K3) |

Latencijos slenksčiai — [prielaida]: parinkti ~10× virš recenzento native Linux skaičių dėl WSL2/WDDM laiko pjūvių; po pirmo paleidimo peržiūrėti ir užfiksuoti `thresholds.json` (keitimas — tik su įrašu ataskaitoje).

- **Pajėgumas N\*** = didžiausias N, kuriame visi slenksčiai PASS. Lygiai, kuriuose FAIL tik PAJĖGUMO metrikos, raportuojami kaip „virš pajėgumo" (ne teisingumo klaida). Jei N=1 nepraeina latencijos slenksčių, bet VERIFIKACIJA = 0 → teisingumas PASS, našumas FAIL (atskiros eilutės ataskaitoje).
- **Soak N_soak** = vienu lygiu žemiau N\* (jei N\* = 1 → 1). Trukmė: **2 h dabar**, 24 h vėliau (`--soak 2h|24h|skip`).
- **Pakartojamumas:** 3 × 120 s bandymai su N_soak, tarp jų pilnas procesų užbaigimas ir `mxl-info -g`.

Soak slenksčiai (papildomai prie lentelės aukščiau; pirmos 5 min — warmup, neįeina):

| Metrika | Slenkstis |
|---|---|
| VERIFIKACIJOS skaitikliai | = 0 per visas 2 h |
| `gap_grains` | = 0, išskyrus grain'us ±2 s nuo užfiksuoto `clock_step_event` (jie išvardijami atskirai; > 3 clock step → soak negalioja, kartoti) |
| RSS (abu procesai) | mediana(paskutinės 10 min) − mediana(pirmos 10 min po warmup) ≤ **1 %** ir tiesinės regresijos nuolydis ≤ 1 %/2h |
| `fd_count` | pabaigoje == pradžioje (**±0**) |
| VRAM: `vram_total − vram_free` (įrankio `cudaMemGetInfo`) | medianų skirtumas ≤ max(1 %, 32 MB) (Windows programos kelia triukšmą — soak metu uždaryti naršykles; `nvidia-smi memory.used` — antras šaltinis) |
| latencijos dreifas | P99 paskutinę valandą ≤ 1,5 × P99 pirmą valandą |
| GPU temperatūra | užfiksuota; jei P99 latencija koreliuoja su throttling'u → pastaba (ne FAIL) |

Pakartojamumo slenksčiai: visi 3 bandymai PASS; P50 `lat_commit2ver` sklaida (max−min)/mediana ≤ 25 %; CPU branduoliai/flow sklaida ≤ 20 %.

### 6.5 Išorinės metrikos (harness)

| Metrika | Įrankis | WSL pastaba |
|---|---|---|
| CPU branduoliai/flow | `pidstat -u -h -p <pid> 1` → (`%usr+%system`)/100/N; antras šaltinis — įrankio `getrusage` | veikia [prielaida: `sysstat` paketas] |
| RSS, FD | įrankio `proc-*.csv` + `pidstat -r`; `ls /proc/<pid>/fd \| wc -l` | veikia |
| VRAM | `nvidia-smi --query-gpu=timestamp,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -l 1` | `memory.used` ir temperatūra — tikimasi veiks; `utilization.gpu` ir per-procesinė atmintis — **neveikia WSL** [patikrinta: wsl-user-guide §4.2]; laukai `[N/A]` priimami; **patvirtinti etape 00** |
| TAI offset | `adjtimex` (įrankis) | WSL2 gali būti 0 vietoj 37 s. Vienos mašinos testui tai nesvarbu (writer ir reader naudoja tą patį laikrodį) → **WARN + įrašas manifest'e, ne FAIL**. Sistemos nustatymų nekeičiame. |

---

## 7. Bazė palyginimui (GPU kelio nauda)

Tas pats įrankis, `--storage host` (v1.1 kelias; `OpenGrainSlot`/`GetGrainSlot` veikia ir host storage — `flow.h:694,807`):

| Variantas | Writer | Reader | Ką matuoja |
|---|---|---|---|
| **G** (PR kelias) | `k_fill` tiesiai į slot'o device ptr | `k_verify` tiesiai iš device ptr | zero-copy GPU→GPU |
| **B1** (host flow, GPU galai) | `k_fill` į scratch device buferį → `cudaMemcpy` D2H į host slot'ą | `cudaMemcpy` H2D iš host slot'o į scratch → `k_verify` | ką kainuotų tas pats GPU→GPU srautas per v1.1 host flow (2 PCIe kopijos po 5,53 MB) |

Kopijos — iš pageable atminties (reader'io mapping'as read-only, `cudaHostRegister` jį atmeta — recenzento pastaba; WSL'e pinned atmintis ribota) → B1 yra **realistiška, ne optimistinė** bazė; tai pažymima ataskaitoje.

Vykdoma N ∈ {1, 4, 8, 16, 32}, tos pačios trukmės. Ataskaitos lentelė G vs B1 kiekvienam N: `lat_commit2ver` P50/P99/P99.9, `lat_e2e` P50, CPU branduoliai/flow (writer, reader), RSS, pajėgumas N\*, apskaičiuotas PCIe srautas (B1: 2 × 5,53 MB × fps × N; G: 0). Bazės slenksčių nėra (INFO), išskyrus: VERIFIKACIJOS skaitikliai = 0 (tai kartu patikrina įrankį v1.1 kelyje).

Pivot A atveju (§2) vykdoma tik B1.

---

## 8. `run-all.sh` ir įrodymų paketas

`scripts/run-all.sh [--soak 2h|24h|skip] [--only NN,NN] [--skip NN] [--mxl-root …] [--out …]`. Kiekvienas etapas — `scripts/stages/NN-*.sh`, kviečiamas per `run_step` (užrašo komandą, stdout/stderr, exit kodą, pradžios/pabaigos TAI, taiko `timeout`). Etapo nesėkmė nesustabdo kitų, **išskyrus** 01 (pivot), 03 ir 04 (be veikiančio detektoriaus toliau nėra prasmės). Prieš kiekvieną etapą: naujas tuščias domenas `/dev/shm/mxlv-<run>-<NN>`; po etapo — procesų užbaigimas, `mxl-info -g`, domeno šalinimas.

| NN | Etapas | Turinys |
|---|---|---|
| 00 | env | versijos ir prielaidų patikra: MXL `git rev-parse HEAD` == `6dddc91d…`, `git status --porcelain` tuščias; mūsų repo SHA; `nvcc --version`; `nvidia-smi` (driver); `uname -r`; `/proc/version` (WSL); `wsl.exe --version` (jei interop); `gpu-info`; `gcc/clang/cmake/ninja` versijos; `ldd libmxl-payload-cuda-linear.so`; `/dev/shm` tmpfs ir laisva vieta; TAI offset; RAM; `nvidia-smi --query-gpu` laukų prieinamumas. Kietos sąlygos (FAIL → stop): GPU matomas, `/dev/shm` tmpfs, plugin'as egzistuoja |
| 01 | probe | §2; rezultatas lemia PIVOT |
| 02 | build | mūsų build (`wsl-release`), `ctest -N`; įrašomos kompiliavimo vėliavėlės |
| 03 | unit | §5.2 + §5.3 (`-r junit`) |
| 04 | selftest | §5.1 |
| 05 | upstream | F1, F2 |
| 06 | functional | F3–F9 |
| 07 | sanitizers | S1–S3; S4, jei yra ASan build'as (`MXL_ASAN_ROOT`), kitaip NOT_RUN |
| 08 | baseline | §7 B1 |
| 09 | scaling | §6.4 G, N = 1,4,8,16,32 → N\*, N_soak |
| 10 | repeat | 3 × 120 s |
| 11 | soak | 2 h (arba 24 h / skip) |
| 12 | report | `report.py` → `report.md`, `report.json`, `SHA256SUMS` |

Orientacinė trukmė be soak: ~50 min [prielaida]; su 2 h soak ~3 h.

**Paketo išdėstymas:**

```
evidence/<UTC yyyymmddThhmmssZ>_<mxlsha7>_<toolsha7>/
├─ manifest.json        # run id, argumentai, visų SHA, versijos, GPU (vardas, UUID, VRAM), driver,
│                       # kernel, WSL versija, CUDA toolkit, TAI offset, granularity iš P0.4, pivot būsena
├─ env/                 # neapdorotos komandų išvestys (nvidia-smi.txt, nvcc.txt, uname.txt, ldd.txt, …)
├─ config/              # thresholds.json ir matrix.json kopijos (tiksliai tos, kurios naudotos)
├─ stages/NN-name/
│  ├─ cmd.txt  stdout.log  stderr.log  exit_code  timing.json
│  ├─ result.json       # [{id, title, status, measured, threshold, evidence_files[]}]
│  └─ <case>/           # flow-*.csv, proc-*.csv, summary-*.json, pidstat.txt, nvidia-smi.csv, junit.xml
├─ report.md            # žr. žemiau
├─ report.json          # mašininė santrauka
└─ SHA256SUMS           # visų paketo failų
```

`report.md` (generuojamas, ne rašomas ranka): (1) antraštė su SHA/versijomis; (2) **verdiktų lentelė** — viena eilutė kiekvienam kriterijui: ID, pavadinimas, PASS/FAIL/NOT_RUN/INFO, išmatuota, slenkstis, nuoroda į failą; (3) atskiros suvestinės „Teisingumas" ir „Našumas"; (4) G vs B1 lentelė; (5) skalavimo lentelė; (6) soak tendencijos (pirmos/paskutinės 10 min medianos, nuolydžiai); (7) NOT_RUN sąrašas su priežastimis; (8) nukrypimai nuo plano ir clock-step įvykiai. `report.py` **niekada** nerašo PASS be `result.json` įrašo su išmatuota reikšme.

---

## 9. WSL aplinkos paruošimas (dokumentuojama `scripts/setup-wsl.md`; vykdo žmogus)

1. Nauja distribucija: `wsl --install -d Ubuntu-24.04` (atskira nuo esamų). **Nediegti jokio Linux NVIDIA driver'io** (`cuda-drivers`, `nvidia-open`) — WSL naudoja Windows driver'io `libcuda.so` stub'ą [patikrinta: wsl-user-guide §3.4]. `nvidia-smi` yra `/usr/lib/wsl/lib/` [patikrinta: ten pat §4.1].
2. Paketai (pagal `.devcontainer/Dockerfile` ir `examples/Dockerfile.cuda-tests` šiame SHA) [patikrinta]:
   ```
   sudo apt-get update && sudo apt-get install -y --no-install-recommends \
     build-essential cmake ninja-build git curl wget zip unzip tar pkg-config ca-certificates \
     bison flex ccache clang-19 lld-19 llvm-19 \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-x \
     sysstat xvfb python3 jq
   ```
   CMake: MXL reikalauja ≥ 3.24, preset'ai ≥ 3.23 [patikrinta: `CMakeLists.txt:4`; `CMakePresets.json:3-7`]; Ubuntu 24.04 apt cmake 3.28 [prielaida]. Rust, libfabric, doxygen **nereikalingi** (`MXL_ENABLE_FABRICS_OFI=OFF` numatyta; `-DBUILD_DOCS=OFF`).
3. CUDA toolkit **13.2** iš WSL-Ubuntu repo (pin'inta; Windows driver 596.49 palaiko iki CUDA 13.2):
   ```
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
   sudo apt-get install -y cuda-compiler-13-2 cuda-cudart-dev-13-2 cuda-driver-dev-13-2 \
        cuda-nvml-dev-13-2 cuda-sanitizer-13-2
   export PATH=/usr/local/cuda-13.2/bin:$PATH
   ```
   Keyring ir paketai egzistuoja [patikrinta: HTTP 200 keyring; Packages]. **Nediegti `cuda-toolkit` ar `cuda`** (atneštų 13.3). Jei minimalaus rinkinio CMake `FindCUDAToolkit` neužtenka → `cuda-toolkit-13-2` (pilnas, be driver'io) [prielaida: minimalus rinkinys pakankamas]. Atsarginis variantas, jei 13.2 runtime nesuderinamas su driver'iu: `cuda-toolkit-12-9` (PR Dockerfile stato su 12.8.1 [patikrinta: `examples/Dockerfile.cuda-tests`]).
4. vcpkg (kaip MXL devcontainer'yje): `git clone https://github.com/microsoft/vcpkg ~/vcpkg && ~/vcpkg/bootstrap-vcpkg.sh -disableMetrics; export VCPKG_ROOT=~/vcpkg` [patikrinta: `.devcontainer/Dockerfile` „Install vcpkg"; preset'ų `toolchainFile` = `$env{VCPKG_ROOT}/…`]. Priklausomybės iš `vcpkg.json`: catch2, stduuid, spdlog, fmt, picojson, cli11, ada-url [patikrinta].
5. MXL build (`scripts/build-mxl.sh`):
   ```
   git clone https://github.com/dmf-mxl/mxl ~/src/mxl-pr653 && cd ~/src/mxl-pr653
   git fetch origin pull/653/head && git checkout --detach 6dddc91d92f56c3c9d90fa089c95d8f262bc2c92
   cmake --preset Linux-GCC-Release -DMXL_ENABLE_CUDA_PAYLOAD=ON -DBUILD_DOCS=OFF \
         -DCUDAToolkit_ROOT=/usr/local/cuda-13.2
   cmake --build build/Linux-GCC-Release
   cmake --install build/Linux-GCC-Release        # → install/Linux-GCC-Release
   ```
   Preset'ų vardai: `Linux-GCC-Debug|Release`, `Linux-Clang-Debug|Release` [patikrinta: `CMakePresets.json`]. Build'e turi atsirasti `build/Linux-GCC-Release/lib/libmxl.so*`, `lib/libmxl-payload-cuda-linear.so`, `lib/tests/cuda/mxl-cuda-tests` [patikrinta: keliai iš `examples/Dockerfile.cuda-tests`].
6. Neprivaloma (F4): `git worktree add ~/src/mxl-v1.1.0 v1.1.0` + build → `MXL_V11_ROOT`. Neprivaloma (S4): Clang-Debug ASan build → `MXL_ASAN_ROOT`.
7. Prieš soak (rankinis sąrašas, harness tik tikrina/įspėja): maitinimas iš tinklo; Windows miego režimas išjungtas; uždarytos GPU naudojančios programos; `.wslconfig` RAM ≥ 12 GB [prielaida].

---

## 10. Įgyvendinimo užduotys (eilės tvarka, test-first)

Kiekviena užduotis: pirma testas (raudonas), tada kodas, tada priėmimo patikra su **parodyta komandos išvestimi**.

| # | Užduotis | Priėmimo patikra |
|---|---|---|
| T0 | WSL aplinka pagal §9 (1–4) | `nvcc --version` rodo 13.2; `/usr/lib/wsl/lib/nvidia-smi` rodo RTX 4070 |
| T1 | `probe/cuda_fd_probe.cpp` + mini CMake | exit kodas ir JSONL išvestis; **sprendimas pagal §2 lentelę užrašomas prieš tęsiant** |
| T2 | MXL build pagal §9.5; `build-mxl.sh` | `git rev-parse HEAD` = `6dddc91d…`; `ls` rodo plugin'ą; `ctest -N` sąrašas |
| T3 | Repo skeletas: CMake, preset'ai, vcpkg.json, tuščias `main` su CLI11, Catch2 runner | `cmake --preset wsl-release && cmake --build` OK; `vramflow-verify --help` |
| T4 | `pattern.hpp`, `crc32c.hpp` + unit testai su auksinėmis reikšmėmis | `ctest -R pattern\|crc` žalia |
| T5 | `header.hpp` + testai | žalia |
| T6 | `index_math.hpp`, `histogram.hpp`, `clock.hpp` + testai | žalia |
| T7 | `csv.hpp`, `json.hpp`, `flowdef.hpp` + testai (įsk. libmxl host priėmimą) | žalia |
| T8 | `kernels.cu` + GPU testai §5.3 | `ctest -L gpu` žalia; `compute-sanitizer --tool memcheck` ant GPU testų: 0 klaidų |
| T9 | `gpu-info` | UUID sutampa su `mxl-cuda-tests` naudojamu (`CudaTestUtils.hpp` formatas) |
| T10 | Writer host režimu (W1–W13, `--storage host`) | `mxl-info` rodo judantį head; CSV rašomas |
| T11 | Reader host režimu (RD1–RD12, politika) + kontrolinis selftest | 20 s: visi skaitikliai 0, exit 0 |
| T12 | Injekcijos §5.1 + `selftest.sh` (host režimu) | visi atvejai kaip lentelėje |
| T13 | `--storage cuda-linear` writer'iui ir reader'iui (W3–W7, RD4–RD6) | selftest kartojamas cuda-linear režimu — tie patys rezultatai |
| T14 | `multi` režimas (gija per handle) | N=4, 30 s: 0 klaidų; GPU testas „stream izoliacija" žalias |
| T15 | Reopen/stall/TOO_LATE keliai + `--gc-before-create`, `--cycles`, `hog` | rankiniai F6a/F6c/F9 scenarijai praeina |
| T16 | `scripts/lib/common.sh`, `run-all.sh`, etapai 00–04 | paketas sukuriamas; `result.json` kiekvienam etapui |
| T17 | Etapai 05–07 (F1–F9, S1–S3), `v11_reader_probe` | kiekvienas F/S turi `result.json` su būsena |
| T18 | Etapai 08–11 + `matrix.json`/`thresholds.json` | N\*, N_soak apskaičiuoti; 10 min bandomasis soak |
| T19 | `report.py` + testas ant fiksuoto pavyzdinio paketo | `report.md` lentelė; dirbtinai sugadinus CSV → FAIL eilutė |
| T20 | Pilnas `run-all.sh --soak 2h`; `/code-review`; `code-simplifier` | galutinis paketas + `SHA256SUMS` |

T10–T12 sąmoningai daromi **host režimu pirma**: detektorius įrodomas kelyje, kuris nepriklauso nuo PR GPU kodo, ir tik tada (T13) taikomas GPU keliui.

---

## 11. Rizikos, švelninimas ir sustojimo sąlygos

| # | Rizika | Tikimybė | Švelninimas |
|---|---|---|---|
| K1 | **POSIX FD eksportas/importas neveikia WSL2** (nedokumentuota) | vidutinė–didelė | žingsnis 0 prieš viską; Pivot A (§2) |
| K2 | Driver 596.49 ↔ toolkit 13.2.x nesuderinamumas (statinė cudart) | maža | etapas 00 + probe; atsarginis toolkit 12.9 |
| K3 | WSL2 laikrodžio šuoliai (host miegas, laiko sinchronizacija) → indeksų šuoliai, netikri gap'ai | vidutinė (soak) | clock-step detektorius (`tai_minus_mono`), įvykiai anotuojami; miegas išjungtas; > 3 šuoliai → soak kartojamas |
| K4 | TAI offset = 0 WSL2 | didelė | vienai mašinai nesvarbu; WARN + manifest; **nekeisti sistemos** |
| K5 | WDDM laiko pjūviai tarp 2 CUDA procesų, nėra MPS → latencijos uodegos | didelė | `multi` (tik 2 procesai); laisvi, konfigūruojami slenksčiai; teisingumas ir našumas vertinami atskirai |
| K6 | Laptopo terminis throttling'as / baterija | vidutinė | temperatūra ir galia loguojami; soak tik iš tinklo; N_soak vienu lygiu žemiau N\* |
| K7 | `nvidia-smi`/NVML laukai neprieinami WSL | vidutinė | pagrindinis VRAM šaltinis — `cudaMemGetInfo` įrankyje; `[N/A]` priimami |
| K8 | WDDM VRAM oversubscription → F8 neduoda OOM | vidutinė | F8 rezultatas INFO, jei klaidos nėra |
| K9 | compute-sanitizer false positive importuotai VMM atminčiai; sulėtėjimas | vidutinė | in-process variantas kaip atskaita; mažas formatas; reader'io rezultatas INFO |
| K10 | ASan nesuderinamas su libcuda WSL | vidutinė | CUDA dalis NOT_RUN; ne-CUDA testai privalomi |
| K11 | `mxl-gst-cuda-sink` be ekrano | vidutinė | `xvfb-run`; kitaip F3 NOT_RUN |
| K12 | PR juda (naujas push pakeičia API) | didelė | SHA pin'intas; harness tikrina SHA; atnaujinimas — atskiras, sąmoningas žingsnis su šio dokumento R lentelės peržiūra |
| K13 | Mūsų įrankio klaida duoda netikrą „0 klaidų" | — | §5.1 savitikra privaloma prieš kiekvieną paleidimą; CPU kryžminė patikra; unit auksinės reikšmės apskaičiuotos nepriklausomai |
| K14 | Mūsų įrankio klaida duoda netikrus mismatch'us (lap lenktynės) | — | RD11 lap patikra; `lapped_discarded` atskiras skaitiklis |
| K15 | Repo kelias su tarpu ir ant `/mnt/c` | didelė | build ir evidence Linux FS; kabutės skriptuose |
| K16 | `v210` stride / binarų vardai / minimalus CUDA paketų rinkinys — prielaidos | maža | tikrinama vykdymo metu (assert'ai, etapas 00) |
| K17 | Vienos GPU mašina: peer-GPU ir cross-container keliai nepatikrinami | faktas | NOT_RUN ataskaitoje; išvada galioja tik „vienas GPU, vienas host, be konteinerių" |

**Sustojame ir pranešame (netęsiame automatiškai), jei:**
1. Probe exit 13 (baitų neatitikimas driver'io lygyje).
2. Savitikra (§5.1) nepraeina — detektorius nepatikimas.
3. Kontrolinis bandymas (be injekcijų) cuda-linear režimu duoda bet kokį VERIFIKACIJOS skaitiklį > 0 esant N=1 — tai **pagrindinis radinys**: fiksuojame pirmų 10 klaidų dump'us, kartojame 3 kartus, tikriname su `--host-crosscheck-every 1`, ir tik tada raportuojame PR autoriams.
4. MXL build'as SHA `6dddc91d` nepavyksta su `MXL_ENABLE_CUDA_PAYLOAD=ON` (tada raportuojame build klaidą su visa išvestimi; netaisome upstream kodo savo nuožiūra).
5. Bet kuris procesas crash'ina (SIGSEGV/SIGABRT) libmxl ar plugin'o viduje — surenkame backtrace (`gdb -batch`), raportuojame.

Pivot A (probe 10/11/12) nėra „stop" — tai kitoks, mažesnis planas su aiškia išvada „GPU kelias šioje platformoje neįrodytas".

---

## 12. Ko šis planas neįrodo

- Kelių GPU (peer access) ir kelių konteinerių/pod'ų scenarijų.
- Native Linux našumo (WSL2/WDDM skaičiai nėra perkeliami į serverį).
- v210a (2 plokštumos), interlaced, kitų raiškų — tik 1080p50 v210 (parametrai konfigūruojami, bet matrica jų neapima).
- Fabrics/RDMA iš device atminties (PR to nedaro).
- 24 h stabilumo — iki kol bus įvykdytas `--soak 24h`.
