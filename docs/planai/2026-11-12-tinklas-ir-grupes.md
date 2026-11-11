# Tinklas, flow grupės ir NMOS valdymas (VRAMFlow) — įgyvendinimo plano juodraštis

> **Būsena:** plano juodraštis, **laukia autoriaus sprendimų** (reikalavimų §9, K-1…K-15). Kodas nekeistas, niekas
> nekompiliuota ir nepaleista. Teiginiai apie kodą paremti skaitymu (failas:eilutė), apie išorę — šiandien atidarytais
> šaltiniais (§1). Kur reikia vykdymo — pažymėta **[nepatikrinta čia]**. Funkcijų kūnų plane nėra — tik sąsajos.
> **Vykdytojams:** kiekvienas etapas (E0…E11) — atskira šaka ir nauja sesija; pirmiausia testai (TDD), commit'ai lokaliai,
> `git push` daro tik autorius. Etapai E2+ nepradedami, kol nebaigti bazės P5, P6, P7 (žr. §5).

**Reikalavimai:** `docs/reikalavimai/2026-11-12-tinklas-ir-grupes.md` (toliau **[R]**; user stories N1–N7, G1–G5, M1–M4).
**Bazė:** `docs/planai/2026-11-11-production-grade.md` (toliau **[B]**; P1–P10, sprendimai K1–K17).

**Tikslas:** flow'ai, gyvenantys vieno serverio GPU ar RAM, tampa prieinami kitų serverių funkcijoms per TCP ir RDMA su
tais pačiais indeksais ir turiniu; flow'ai grupuojami, izoliuojami ir dalijami tarp grupių; visa tai neprivalomai matoma
ir valdoma per NMOS. Žemo lygio: funkcija `bridge`, biblioteka `mxlv_group`, CLI `vramflow-ctl`, failai; jokio demono.

**Architektūros kryptis (8 eilutės):**

1. **Medijos kelias — tik MXL Fabrics** (teikėjai `tcp` ir `verbs`); VRAMFlow neturi savo medijos protokolo.
2. **`cuda-linear` galai — per tarpinį host flow** (VRAM → RAM → Fabrics → RAM → VRAM), nes Fabrics šiandien atmeta
   flow'us su payload backend'u; **GPUDirect** (`--stage gpu`) — tik per upstream Fabrics, kai atsiras gebėjimas.
3. **`bridge` niekada neperžymi laiko:** indeksas B pusėje == indeksas A pusėje; rinkinys (video + garsas) — vienas procesas.
4. **Šalutinis kanalas TargetInfo perdavimui — trys lygiai:** failas → TCP rendezvous (JSON eilutės) → NMOS IS-05 `ext_`.
5. **Grupė = MXL domenas** (+ `.pub` eksporto domenas + `.vramflow` būsena); **rinkinys = `grouphint`**; izoliacija —
   organizacinė (numatyta: vienas UID, atskiri domenai, kvotos) arba griežta (UID / gid domeno kataloge, konteinerių
   tomai); dalijimasis viename serveryje — skaitymas iš `.pub` be kopijos (`cuda-linear` — tik tam pačiam UID, F-16).
6. **Viena priėmimo knyga vienai GPU** visame serveryje, su grupės kvota; VRAM apskaita iš `payload.cuda.json`.
7. **NMOS — atskiras sidecar procesas** ant `nmos-cpp`, skaitantis VRAMFlow failus; funkcijose HTTP nėra.
8. **Matavimai — plano dalis:** pridėtas vėlavimas (dalimis, kiekviena vieno laikrodžio), CPU branduoliai flow'ui,
   flow'ų skaičius jungčiai; TCP prieš verbs prieš GPUDirect.

**Technologijos:** C++20, CUDA 13.2, MXL PR #653 (`6dddc91d92f5`) + `MXL_ENABLE_FABRICS_OFI=ON`, libfabric (K-1),
Catch2 3, CLI11, Python 3 (harness); NMOS sidecar'ui — `sony/nmos-cpp` (K-9). Naujos priklausomybės — tik autoriaus sprendimu.

---

## 1. Šiandien patikrinti faktai (2026-11-11)

Statusas: **✔** — patikrinta šiandien nurodytame šaltinyje; **◐** — iš dalies; **✘** — nepatikrinta (užrašai arba ankstesnės
žinios). Vietiniai keliai be priešdėlio — MXL medis `/root/src/mxl-pr653` (WSL, HEAD `6dddc91d`, 2026-11-09).

| # | Teiginys | Šaltinis | Statusas |
|---|---|---|---|
| F-01 | Naujausias MXL leidimas — v1.1.0 (žymos: v1.0.2, v1.1.0-rc1, v1.1.0); jame atsirado Fabrics ir sync grupės | `git ls-remote --tags https://github.com/dmf-mxl/mxl`; https://github.com/dmf-mxl/mxl/releases | ✔ |
| F-02 | MXL `main` HEAD `3aaa8ed6`, paskutinis commit'as 2026-11-09 | `git ls-remote … refs/heads/main`; GitHub API `repos/dmf-mxl/mxl/commits` | ✔ |
| F-03 | PR #653 („Add pluggable CUDA linear grain payload support“) — atviras; jo galva `6dddc91d92f5` = mūsų vietinis medis | https://github.com/dmf-mxl/mxl/pull/653; `git ls-remote … refs/pull/653/head` | ✔ |
| F-04 | PR #653: „Fabrics refuses these flows…“; kode — `MXL_ERR_UNSUPPORTED_OPERATION` flow'ams su payload backend'u | PR #653 aprašas; `lib/fabrics/ofi/src/internal/Region.cpp:159-165` | ✔ |
| F-05 | `main` Fabrics taip pat atmeta GPU atmintį; `FI_HMEM` filtruojamas teikėjams verbs, shm, efa | `main` `Region.cpp:159-162` („GPU memory is not currently supported…“); `ProviderConfig.cpp:96,112,128` (vietinis ir `main`) | ✔ |
| F-06 | PR #654 („Add fabrics FI_HMEM support for CUDA grain payloads“) — atviras nuo 2026-08-05, paskutinis force-push 2026-09-27, peržiūrų nėra; tik verbs; prideda `MXL_FABRICS_IFACE_CAP_HMEM`, atskirus regionus (host antraštė + CUDA payload), CUDA „bounce“ buferius, nes IPC importuota atmintis „not peermem-registerable“; testų planas nepažymėtas | https://github.com/dmf-mxl/mxl/pull/654; `refs/pull/654/head` = `a61be1b8` | ✔ |
| F-07 | PR #746 („Separate grain payload storage from the discrete flow ring“) — atviras (2026-10-02, paskutinis push 2026-11-11); pirmas iš serijos, viešo API nekeičia | https://github.com/dmf-mxl/mxl/pull/746 | ✔ |
| F-08 | #594 („Architecture: device hosted grain mechanism“) — atvira, v1.2 gairė; kryptis: įrenginio atmintis kaip „kitas domenas“, tarp domenų — replikatorius; Fabrics atsiliepimas: per-commit metaduomenys host'e ⇒ antras remote write; sprendimo nėra | https://github.com/dmf-mxl/mxl/discussions/594 | ✔ |
| F-09 | Fabrics modelis: *target* (gavėjas, apgaubia flow writer'į) skelbia TargetInfo; *initiator* (siuntėjas, apgaubia reader'į) daro RMA remote write; teikėjai tcp / verbs (`FI_EP_MSG`), efa / shm (`FI_EP_RDM`); įgyvendinta: discrete per RMA tiesiai į media buferį, continuous per RMA + bounce buferį; send/recv atsarginis kelias neįgyvendintas; teikėjo biblioteka pati neparenka | `docs/Fabrics.md` §2, §5.1.4, §10; `docs/FabricsGettingStarted.md` §3; `docs/FabricsDeveloperGuide.md` | ✔ |
| F-10 | TargetInfo raktas „is not a security mechanism“; „Media transfers are unencrypted“; saugoti tinklo lygiu | `docs/FabricsDeveloperGuide.md` („Serializing the TargetInfo“) | ✔ |
| F-11 | Vienas initiator gali turėti kelis target'us (1 → N) | `docs/FabricsDeveloperGuide.md` („Setting up an initiator“) | ✔ |
| F-12 | RC target po nutolusio `shutdown` grįžta į klausymą; RC initiator galinis taškas persikuria iš `Idle` (pakartotinis jungimasis) | `RCTarget.cpp:165-168`; `RCInitiator.cpp:269-271, 386` (skaityta, nevykdyta) | ✔ |
| F-13 | „MXL Flow replication between hosts will preserve the indexing of the grains… flow IDs, grain indexes and ring buffer size and indexes“ | `docs/Timing.md:83` | ✔ |
| F-14 | Keli host'ai: laiko šaltiniai neturi „slip against each other“; jitter'is priimtinas; PTP nereikalaujamas | `docs/Timing.md:6,14` | ✔ |
| F-15 | `cuda-linear`: vienas rašytojas; deskriptorius per `payload.sock`; tik ta pati mašina („A copy between nodes is a separate domain on each node. RDMA from device memory is later work“); `payload.cuda.json` turi `deviceUuid`, `slotCount`, `stride`, `allocSize`; garsas — tik host | `docs/PluggablePayloads.md:55,65,90-96`; `CudaLinearBackend.cpp:978-983` | ✔ |
| F-16 | `payload.sock` (PR #653 `6dddc91d`): backend'o `prepare` kviečiamas laikiname `mkdtemp` kataloge (0700) **prieš** paskelbimą; lizdas ten `bind`'inamas ir iškart `applySocketMode()` iš to katalogo teisių išveda režimą (⇒ 0600) ir kviečia `fchmod` klausymo deskriptoriui; `peerMayRead()` lygina `SO_PEERCRED` uid / (pagrindinį) gid su `fstat(_listen)` ⇒ **deskriptorių gauna tik rašytojo UID**. Eksperimentas (be GPU): po `bind` su `umask 022` kelio režimas 0755; `fchmod(fd, 0666)` kelio režimo **nekeičia**, `fstat(fd)` rodo 0666; kitas UID jungdamasis gauna `EACCES`; po `chmod(kelias, 0666)` — prisijungia. Išvada apie UID — iš kodo; su GPU ir dviem naudotojais nevykdyta | `lib/internal/src/FlowManager.cpp:345, 411, 418`; `CudaLinearBackend.cpp:1255-1258, 606-635, 711-750`; Python bandymas WSL2 laikiname kataloge (branduolys 6.6.87.2) | ◐ |
| F-17 | MXL kiekvienam flow katalogui prideda `g+rx`, `o+rx`; flow failai kuriami režimu 0664 (& ~umask); skaitytojui reikia tik skaitymo teisės | `lib/internal/src/FlowManager.cpp:59-64`; `SharedMemory.cpp:30`; `FlowReader.cpp:33-37`; `docs/Architecture.md:42` (`main`) | ✔ |
| F-18 | Sync grupės API: `mxlCreateFlowSynchronizationGroup`, `…AddReader`, `…AddPartialGrainReader`, `…WaitForDataAt(group, timestamp, timeoutNs)` | `lib/include/mxl/flow.h:923-1019` | ✔ |
| F-19 | Flow aprašas privalo turėti bent vieną `urn:x-nmos:tag:grouphint/v1.0` reikšmę formatu `<group>:<role>[:device\|node]` | `lib/internal/src/FlowParser.cpp:130-186, 258` | ✔ |
| F-20 | MXL adresavimas: `mxl://<authority>/<domain-path>?id=<flow>&id=…`; domeno kelio vertimas — ne MXL reikalas | `docs/Addressability.md` | ✔ |
| F-21 | libfabric naujausias — v2.7.0 (2026-09-18); MXL devcontainer'is stato **v2.5.1** iš šaltinio; Ubuntu 24.04 paketas — 1.17.0 | GitHub API `repos/ofiwg/libfabric/releases/latest`; `.devcontainer/scripts/common/libfabric/install.sh`; `apt-cache policy libfabric-dev` | ✔ |
| F-22 | libfabric v2.7.0: tcp teikėjas — `FI_MSG \| FI_RMA`, **be `FI_HMEM`**; verbs — `FI_HMEM` sąlyginai (dma-buf arba peer-mem); shm — turi `FI_HMEM` | `prov/tcp/src/xnet_attr.c:36-39`; `prov/verbs/src/verbs_info.c:1571`, `verbs_init.c:736-745, 850`; `prov/shm/src/smr_attr.c:40-41` (žyma v2.7.0) | ✔ |
| F-23 | `fi_mr_regattr`: `iface = FI_HMEM_CUDA`, `device.cuda` = CUdevice; `FI_HMEM_DEVICE_ONLY`; `FI_MR_DMABUF` (registracija per dma-buf deskriptorių) | https://ofiwg.github.io/libfabric/main/man/fi_mr.3.html | ✔ |
| F-24 | NVIDIA GPUDirect RDMA dok. (atnaujinta 2026-11-02): „available on both Tesla and Quadro GPUs“; reikia `nvidia-peermem` + RDMA branduolio palaikymo (arba MLNX_OFED), ConnectX-3 VPI ar naujesnių; abu įrenginiai po tuo pačiu PCIe root complex; veikia su `cudaMalloc()` puslapiais; GeForce neminima | https://docs.nvidia.com/cuda/gpudirect-rdma/ | ✔ |
| F-25 | Šio WSL2 branduolys 6.6.87.2: `CONFIG_INFINIBAND=m`, `CONFIG_RDMA_RXE is not set`, `CONFIG_RDMA_SIW is not set` ⇒ soft-RoCE be savo branduolio neveiks | `zcat /proc/config.gz` (WSL) | ✔ |
| F-26 | Soft-RoCE (`rxe`) neturi dma-buf MR registracijos (faile nėra „dmabuf“); `mlx5` — turi | `torvalds/linux` master `drivers/infiniband/sw/rxe/rxe_verbs.c` (0 atitikmenų), `hw/mlx5/main.c` (4) | ✔ |
| F-27 | Čia MXL sukompiliuotas **be** Fabrics (`MXL_ENABLE_FABRICS_OFI` numatyta OFF; `libmxl-fabrics*` nėra; libfabric neįdiegtas) | `CMakeLists.txt:6`; `find build install …`; `scripts/wsl/10-build-mxl.sh` (VRAMFlow) | ✔ |
| F-28 | v210 eilutė `ceil(w/48)·128` B; `v210a` key `ceil(w/3)·4` B; RoCEv2 antraštės 82 B paketui (MTU biudžete 44 B). Išvestinė: 1080i50 = 1,106 Gb/s; 1080p50 = 2,212; 720p25 = 0,498; 2160p50 = 8,85; 1080i50 v210a = 1,659 Gb/s | `docs/FabricsBandwidth.md` §1–2; aritmetika | ✔ |
| F-29 | NMOS leidimai (žymos): IS-04 v1.3.3, IS-05 v1.2.0, IS-07 v1.0.1, IS-08 v1.0.1, IS-10 v1.0.1, IS-11 v1.0.0, IS-12 v1.0.1, IS-14 v1.0.0, MS-05-02 v1.0.0, BCP-002-01 v1.0.0, BCP-002-02 v1.0.0, BCP-003-01 v1.0.1, BCP-004-01 v1.0.0, **BCP-007-03 v1.0.0**, BCP-008-01 v1.0.0, BCP-008-02 v1.0.0 | `git ls-remote --tags https://github.com/AMWA-TV/<spec>` kiekvienam | ✔ |
| F-30 | BCP-007-03 v1.0.0: transportas `urn:x-nmos:transport:mxl`; IS-04 ≥ v1.3, IS-05 ≥ v1.2; `interface_bindings: []`; `manifest_href: null`, `/transportfile` → 404; parametrai `mxl_domain_id`, `mxl_flow_id` (viena koja); Sender'is abu priima `null` ir `"auto"`, Receiver'io `mxl_flow_id` — tik `null`; `master_enable` paleidžia / stabdo skaitymą ar rašymą; kiekvienas domenas privalo turėti `domain_def.json` {`id`, `label`, `description`, `tags`}; „As MXL Flows may be replicated across MXL Domains… different `mxl_domain_id` values“. **Nė žodžio** apie Fabrics, TargetInfo, GPU atmintį, IS-12 | https://raw.githubusercontent.com/AMWA-TV/bcp-007-03/v1.0.0/docs/NMOS-With-MXL.md (https://specs.amwa.tv/bcp-007-03/) | ✔ |
| F-31 | BCP-007-03 MXL transporto parametrų schema leidžia papildomus `^ext_[a-zA-Z0-9_]+$` (string / boolean / null / number) | `…/bcp-007-03/v1.0.0/APIs/schemas/sender_transport_params_mxl.json`, `sender_transport_params_ext.json` | ✔ |
| F-32 | IS-05 v1.2.0 `/transporttype` leidžia gamintojo URN (ne `urn:x-nmos:`) | `…/is-05/v1.2.0/APIs/schemas/transporttype-response-schema.json` | ✔ |
| F-33 | NMOS transportų registre (`main`, `1b780c23`) `urn:x-nmos:transport:mxl` **dar nėra** (yra rtp, rtp.mcast, rtp.ucast, dash, mqtt, websocket, ndi, usb) | `…/nmos-parameter-registers/main/transports/README.md` | ✔ |
| F-34 | IN-001 (reikalavimai MXL valdymo API, ne specifikacija): nustatyti / gauti `MXLFlowID`, `MXLDomainID`; idempotentiški start / stop; pasiekiamų domenų sąrašas; būsenos `started` / `stopped`; domeno ID → kelio atvaizdavimas — už ribų | https://raw.githubusercontent.com/AMWA-TV/in-001/main/docs/Overview.md | ✔ |
| F-35 | BCP-002-01: „natural group“ apibrėžia pats Node; vartotojo ar automatikos apibrėžtos grupės — „not in scope“. `grouphint`: `<group-name>:<role-in-group>[:<group-scope>]`, `:` draudžiamas, rolė grupėje unikali, žymė **nekintama**, sritis `device` (numatyta) arba `node` | `…/bcp-002-01/v1.0.0/docs/1.0. Natural Grouping.md`; `…/nmos-parameter-registers/main/tags/grouphint.md` | ✔ |
| F-36 | BCP-002-02: Node ir Device privalo turėti po vieną gamintojo, produkto ir egzemplioriaus ID žymę (derinys unikalus) | `…/bcp-002-02/v1.0.0/docs/Overview.md` (tikslūs žymių URN — iš registro, ✘ šiandien neatidaryta) | ◐ |
| F-37 | BCP-008-01: `NcReceiverMonitor` — `linkStatus`, `connectionStatus`, `externalSynchronizationStatus`, `streamStatus` (+ žinutės, perėjimų skaitikliai), `overallStatus`, `statusReportingDelay` numatyta 3 s, `GetLostPacketCounters`, `GetLatePacketCounters`; privalomi IS-12 ir MS-05-02. BCP-008-02: `NcSenderMonitor` — `linkStatus`, `transmissionStatus`, `externalSynchronizationStatus`, `essenceStatus`, `GetTransmissionErrorCounters` | `…/bcp-008-01/v1.0.0/docs/Overview.md`; `…/bcp-008-02/v1.0.0/docs/Overview.md` | ✔ |
| F-38 | `sony/nmos-cpp`: Apache-2.0; `master` `c98184da`, paskutinis commit'as 2026-11-09; žymų nėra; README: IS-04/05/07/08/09/10/12/13/14, BCP-002-01/02, 003-01/02, 004-01, 006-01, **007-03**, 008-01/02, MS-05-01/02 (IS-11 nėra); `Development/nmos/mxl.h` yra; reikia Conan 2, Boost, OpenSSL, DNS-SD (Avahi arba mDNSResponder); commit'as `b6c83670` datuotas 2026-10-05 | https://github.com/sony/nmos-cpp; GitHub API (`commits`, `contents/Development/nmos`) | ✔ |
| F-39 | `AMWA-TV/nmos-testing`: Apache-2.0; paskutinis commit'as 2026-11-11; rinkiniai (failai): `IS0401`, `IS0403`, `IS0501`, `IS0502`, `IS1201`, `BCP00301`, `BCP0070301`, `BCP0070302`, `BCP0080101`, `BCP0080201`; BCP-002-01/-02 tikrinami IS-04 Node rinkinyje; mDNS skelbimus galima išjungti (`ENABLE_DNS_SD`, `DNS_SD_MODE`) | GitHub API `repos/AMWA-TV/nmos-testing/contents/nmostesting/suites`; https://github.com/AMWA-TV/nmos-testing | ✔ |
| F-40 | `NVIDIA/nvnmos`: Apache-2.0, paskutinis push 2026-11-04 | GitHub API `repos/NVIDIA/nvnmos` | ✔ |
| F-41 | Qvest `mxl-k8s` (orkestravimas ir tarp-mazginis MXL pernešimas K8s) ir `go-mxl` paskelbti atviru kodu 2026 m. birželį; repozitorijos adresas ir būklė šiandien nerasti | https://www.advanced-television.com/?p=286238 | ◐ |
| F-42 | Riedel demonstracija ~2 ms vienam flow ant 25 GbE; ~30 HD flow'ų vienai 100 G jungčiai; 1080p59,94 video + garsas ≈ 2,8 Gb/s | 2026 m. pranešimų užrašai | ✘ |
| F-43 | MXL SDK pats `domain_def.json` nekuria (`main` `Architecture.md` jo nemini); `mxl-info` jį skaito | `tools/mxl-info/main.cpp:662`; `main` `docs/Architecture.md` (grep) | ✔ |
| F-44 | Fabrics API: `MXL_FABRICS_API_VERSION 0`; target'o konfigūracija ima `writer`, initiator'iaus — `reader` (naudotojo buferių API nėra); parinktis `cqDepth`; rekomenduojamas dviejų fazių CQ ištuštinimas | `lib/fabrics/include/mxl/fabrics.h:19,127-141`; `docs/Fabrics.md` §11 | ✔ |
| F-45 | VRAMFlow: flow ID = `4d584c47-00RR-4000-8000-<n>`; `grouphint` = `vramflow-func-<seed>-<role>:Video <n>`; nuoma `<domenas>.vramflow/<run_id>.lease`; bazės plano `Ledger::open(domain, device_uuid)` | `src/func/flows.cpp:17-21,26,39`; `src/runtime/domain_lease.hpp`; [B] §P4 | ✔ |
| F-46 | Fabrics regionas discrete flow'ui = grain'o antraštė (8 192 B) + payload viename faile; „immediate data“ neša ring'o lizdą ir slice; tikrąjį indeksą target'as skaito iš perkeltos antraštės; Fabrics nenaudoja `CLOCK_TAI` | `Region.cpp:153-180`; `docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md` §2.3 (patikrinta 2026-11-10) | ◐ |
| F-47 | INVALID grain'ui siunčiama tik antraštė: `mxlFabricsInitiatorTransferGrain(initiator, index, 0, 0)` | `docs/FabricsDeveloperGuide.md` („Transferring discrete flows“) | ✔ |
| F-48 | Soft-RoCE (`rdma_rxe`) natyviame Ubuntu yra `linux-modules-extra` pakete; veikia per vieną mašiną | ankstesnės žinios | ✘ |

**Ką Fabrics duoda ir ko neduoda (santrauka).** *Duoda:* host discrete (v210, v210a, slice'ai) ir continuous (garsas) flow'ų
replikaciją su išsaugotais indeksais per tcp / verbs / efa / shm; 1 → N; pakartotinį jungimąsi; tik antraštę INVALID grain'ams.
*Neduoda:* `cuda-linear` flow'ų (F-04, F-05); `FI_HMEM` per TCP iš principo (F-22); TargetInfo pernešimo; tempo, spragų
politikos, orkestravimo; saugumo (F-10); atsarginio send/recv kelio (F-09). *VRAMFlow prideda:* viską iš [R] §5.1 p. 2.

**Ar RDMA su GPU atmintimi realu šiandien?** Per host buferį — taip, su bet kuria RoCE plokšte (funkciškai — ir su soft-RoCE
natyviame Linux). Tiesiai VRAM → NIC → VRAM — **ne mūsų geležyje ir ne su šiandieniniu upstream**: reikia (a) Fabrics su
`FI_HMEM` (PR #654 — atviras, neperžiūrėtas, F-06), (b) `mlx5` klasės NIC abiejose pusėse (F-24, F-26), (c) GPU, kuriai
NVIDIA dokumentuoja GPUDirect RDMA (Tesla / Quadro klasė; mūsų — GeForce, F-24), (d) `nvidia-peermem` arba dma-buf kelio.

---

## 2. Ribojimai (privalomi; iš [R] §4 ir [B] §1)

- Commit'ai: Conventional Commits anglų kalba, maži; draft PR ≤ ~400 pakeistų eilučių; **jokio `git push`**.
- Pirma klausti: naujos priklausomybės (libfabric, nmos-cpp — tikslus pin, ≥ 30 d.), testų trynimas / perrašymas, viskas
  destruktyvaus, tinklo sąsajų ar branduolio modulių keitimas mašinoje (`modprobe rdma_rxe`, `ip netns` — tik su leidimu).
- Tik sintetinė arba vieša testinė medžiaga. MXL viduje — tik nesuspausta. 0 kompiliatoriaus įspėjimų.
- Kodo komentarai lietuviškai; README / ADR angliškai.
- **Srautai ir jungtys** (F-28; 80 % jungties užpildymas): 10 GbE — 7 × 1080i50 arba 3 × 1080p50; 25 GbE — 18 arba 9;
  100 GbE — 72 × 1080i50, 36 × 1080p50 arba 9 × 2160p50. 1 GbE — tik 720p25 (0,498 Gb/s).
- **VRAM / RAM biudžetas vienam replikuotam flow** (10 grain'ų ring'as): tarpinis host ring'as kiekvienoje pusėje —
  `10 × (8 192 + grain'as)`: 1080i50 → 26,5 MiB RAM; B pusės VRAM ring'as — 26,4 MiB (1080i50), 52,7 MiB (1080p25 / p50),
  39,6 MiB (1080i50 v210a), 211 MiB (2160p50). Testas vienoje mašinoje, N = 8 × 1080i50: +211 MiB VRAM, +423 MiB RAM
  ⇒ telpa į ~6,5 GB laisvos VRAM su esamais 8 šaltinio ring'ais (211 MiB). Ring'o dydis visada iš `layout.slotCount`.

---

## 3. Architektūra

### 3.1 Vaizdas

```
 Serveris A  (grupė studio1)                          Serveris B  (grupė studio2)
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ gen / ingest / composite             │            │            logo / composite / enc    │
 │      │ commit(i)                     │            │                 ▲ GetGrainSlot(i)    │
 │      ▼                               │            │                 │                    │
 │ [F: cuda-linear]  studio1.pub/       │            │ [F: cuda-linear]  studio2/           │
 │      │ MapSlots, D→H kopija          │            │      ▲ H→D kopija, commit(i)         │
 │      ▼        bridge --side tx       │            │      │        bridge --side rx       │
 │ [F: host]  studio1.vramflow/wire-…/  │  Fabrics   │ [F: host]  studio2.vramflow/wire-…/  │
 │   reader ─► initiator ───────────────┼── RMA ────►┼─► target ─► writer                   │
 │                                      │ tcp|verbs  │                                      │
 │   rendezvous klientas ◄──────────────┼─ JSON/TCP ─┼──► rendezvous klausytojas            │
 └──────────────────────────────────────┘            └──────────────────────────────────────┘
   garsas (host, continuous): reader ─► initiator ── RMA ──► target ─► writer (be tarpinio flow)
```

Kai galas yra host flow, tarpinio flow nėra: tx initiator'ius skaito patį šaltinio flow, rx target'as rašo tiesiai į
paskirties flow (grynas Fabrics, 0 papildomų kopijų). `--stage gpu` (E7) tarpinius host flow'us pakeičia Fabrics GPU regionais.

### 3.2 A. Transportas — sprendimai

| Sprendimas | Pasirinkta | Atmesta (viena eilute) |
|---|---|---|
| Medijos protokolas | MXL Fabrics (`mxl/fabrics.h`), teikėjai `tcp`, `verbs`; `shm` — tik testams | Savas TCP protokolas (dubliuoja Fabrics, nesuderinamas su kitais MXL įrankiais); ST 2110 / RTP (kitas darbas; neišsaugo MXL indeksų tiesiogiai) |
| `cuda-linear` galai | Tarpinis host flow tame pačiame serveryje (`<grupė>.vramflow/wire-<run_id>/`, režimas 0700), tas pats flow ID | Privati Fabrics atšaka su `FI_HMEM` (neišlaikoma; F-06 dar keisis); laukti upstream ir nieko nedaryti |
| Kas kuria paskirties flow | rx, iš gauto `flow_def.json` **pažodžiui** (baitas į baitą), tada target'ai | Perkurti aprašą iš `VideoFormat` (rizika prarasti laukus, pvz. žymes) |
| TargetInfo kanalas | (1) failas `--offer-out` / `--offer-in` (tx stebi failo pasikeitimą), (2) rendezvous `--listen` / `--connect`, (3) NMOS `ext_` (E9) | Tik failas (po rx perkrovimo reikia išorinio skripto); bendra failų sistema tarp serverių (nauja infrastruktūra) |
| Tempas | tx seka šaltinį: kiekvienam grain'o dažniui — tick gija, kaip kitose funkcijose; siunčia indeksą, kai jis commit'intas | Siųsti „kiek leidžia jungtis“ iš istorijos (pliūpsniai > 50 % ring'o pažeidžia `Timing.md:89`) |
| Pasivijimas ir lėta jungtis — viena taisyklė | Skrydyje ≤ `max_inflight` grain'ų (numatyta 2, riba `slotCount / 2`); kai atsiranda vietos, siunčiami **naujausi** `min(laukiantys, laisvos vietos)` indeksai, senesni praleidžiami (`skipped`). Vėlavimas ≤ `max_inflight` × vieno grain'o pernešimo laikas | Persiųsti viską, kas dar ring'e (vėlavimas auga, pliūpsnis > 50 % ring'o); neribota eilė (atmintis auga) |
| Spragos B pusėje | [B] K3: kiekvienas indeksas commit'inamas; `--on-loss invalid` (numatyta) \| `freeze` \| `black` per `VideoOut::skip` | Necommit'inti (skaitytojai kaba — [B] L8) |
| Kada indeksas „prarastas“ | `time_of(i) + D0 + deadline`, kur `D0` — slenkantis (10 s) mažiausias `atvykimas − time_of(indeksas)`; nepriklauso nuo laikrodžių poslinkio | Absoliutus B laikrodis (klaidingi praradimai, kai δ > termino) |
| Rinkinys | Vienas tx ir vienas rx procesas visam rinkiniui; garsas siunčiamas ties video grain'o ribomis (`sample_begin(i)`…`sample_begin(i+1)`) | Atskiri procesai kiekvienam flow (nėra bendro gedimo vaizdo) |
| Laikrodžių poslinkis | Rendezvous matuoja 4 laiko žymėmis (kaip NTP), ima mažiausio vėlinimo imtį iš 8; kas 10 s kartoja | Pasikliauti, kad chrony sukonfigūruotas (nematoma) |
| Suderinamumo patikra | Prieš pirmą grain'ą lyginama: Fabrics teikėjas, gebėjimai, `maxMessageSize`, `slotCount`, grain'o dydis, plokštumų skaičius; neatitikimas → kodas 8 | Leisti Fabrics'ui nulūžti vėliau (neaiški klaida) |
| 1 → N | Vienas tx, keli `--connect` / `--offer-in` (Fabrics `AddTarget` kiekvienam); E4 pabaigoje | N atskirų tx (N kartų D→H kopijų) |

**Ko `bridge` nedaro:** nekeičia turinio, nekonvertuoja formatų, neperžymi laiko, nesprendžia, ką replikuoti (tai nurodo
operatorius, skriptas ar NMOS valdiklis).

### 3.3 B. Grupės — sprendimai

Išdėstymas (šaknis `--root`, numatyta `$VRAMFLOW_ROOT` arba `/dev/shm/vramflow`):

```
<šaknis>/
  .node/node.json                 # serverio ID (UUID), etiketė — stabilus NMOS Node ID
  .gpu/<gpu-uuid>/                # priėmimo knyga vienai GPU (visos grupės)            [K-14]
  <grupė>/                        # privatus MXL domenas: domain_def.json, *.mxl-flow/   (0750, gid)
  <grupė>.pub/                    # eksporto MXL domenas (sukuriamas pirmo eksporto metu) (0755 arba 0750 su „share“ gid)
  <grupė>.vramflow/               # būsena: *.lease, group.json, conn/, wire-<run_id>/, stats/
  <grupė>.pub.vramflow/           # eksporto domeno nuomos ([B] K8 taisyklė „<domenas>.vramflow“)
```

| Sprendimas | Pasirinkta | Atmesta |
|---|---|---|
| Kas yra grupė | MXL domenas + būsenos katalogas + Unix gid + kvota | Tik žymė (nėra izoliacijos); sync grupė (tai proceso objektas, ne nuosavybės vienetas) |
| Kas yra rinkinys | `grouphint` = `<rinkinys>:<rolė>`; rolės: `Video`, `Audio`, `Key`, `Gfx <k>`, `Data` | Atskiras manifestų failas (antras tiesos šaltinis šalia `flow_def.json`) |
| Rinkinio skaitymas | `BundleIn`: MXL sync grupė + bendras laikas `WaitForDataAt(time_of(i))` | Kiekvienas flow atskirai (nėra „visi arba nė vienas“) |
| Flow ID | UUIDv5(`VRAMFLOW_NS`, `"<grupė>/<rinkinys>/<rolė>"`); `--flow-base` lieka testams | Atsitiktinis UUID (nestabilus tarp paleidimų — blogai NMOS ir skriptams) |
| Domeno ID | UUIDv5(`VRAMFLOW_NS`, `"<node-id>/<grupė>[.pub]"`), įrašomas į `domain_def.json` | Atsitiktinis (po `tmpfs` išvalymo pasikeistų) |
| Izoliacijos riba | Domeno katalogas (MXL flow katalogus daro `o+rx`, F-17). Du lygiai: **organizacinė** (vienas UID; atskiri domenai, kvotos; konteineriuose — tik atvaizduoti domenai) — numatyta; **griežta** (UID + gid grupei, domenas 0750) — parinktis | Teisės flow katalogui (MXL jas perrašo); griežta kaip numatytoji (žr. kitą eilutę) |
| `cuda-linear` prieiga | Tik rašytojo UID (F-16) — priimama kaip faktas: organizacinėje izoliacijoje zero-copy veikia; griežtoje GPU flow kitam UID eksportuojamas per host flow (vietinis `bridge` `cuda-linear` → host į `.pub`). Radinys → `upstream/U5` | Savas `chmod` / `umask` apėjimas (nepadeda: `peerMayRead` vis tiek atmeta kitą UID); taisyti MXL kopiją (privati atšaka) |
| Dalijimasis viename serveryje | Flow kuriamas `.pub` domene (`--export`); kita grupė skaito per `mxl:///…?id=` — be kopijos | Bind-mount flow katalogo į kitą domeną (root arba konteinerio vykdyklė; lieka galimybe K8s); vietinė kopija `bridge --provider shm` (2 × VRAM) |
| Dalijimasis tarp serverių | `bridge` iš `.pub` į gavėjo grupės domeną; privatūs flow'ai nepaleidžiami (kodas 2) | Leisti bet kurį flow (izoliacija taptų susitarimu) |
| VRAM apskaita | `Σ allocSize` iš `payload.cuda.json` grupės domenuose (F-15) + knygos rezervacijos | NVML pagal procesą (WSL2 neduoda — [B] P7 rizika) |
| Priėmimas | Dvi ribos: grupės kvota (`group.json`) ir GPU talpa (talpos failas × 0,85); `Decision.scope` = `group` \| `gpu` | Tik GPU riba (viena grupė suvalgo viską) |
| Valdymas | `vramflow-ctl` (CLI) + failai; visi failai žmogui skaitomi JSON / `raktas=reikšmė` | Demonas su API; web sąsaja |

### 3.4 D. NMOS — sprendimai

| Sprendimas | Pasirinkta | Atmesta |
|---|---|---|
| Kur gyvena NMOS | Atskiras procesas `vramflow-nmos` (neprivalomas CMake taikinys `VRAMFLOW_ENABLE_NMOS`, numatyta OFF) | NMOS kiekvienoje funkcijoje (HTTP ir DNS-SD medijos procese) |
| Biblioteka | `sony/nmos-cpp`, pin `b6c83670` (F-38); statoma atskirai, funkcijos su ja nesusietos | NvNmos (papildomas sluoksnis, mažiau kontrolės IS-12 modeliui); savas IS-04 / IS-05 (žr. [R] K-9) |
| Iš kur sidecar'as žino būseną | Tik iš failų: domenai (`domain_def.json`, `flow_def.json`), nuomos (`run_id`, `flow=`, `input=`), `conn/*.json`, `stats/*.jsonl` | IPC su funkcijomis (naujas ryšys, kurį reikia išlaikyti) |
| Kaip IS-05 pasiekia funkciją | Sidecar'as rašo `conn/<receiver-id>.json`; funkcija jį stebi tame pačiame cikle kaip `InputSlot::poll` ir kviečia `retarget` | Signalai / lizdai į funkciją |
| Registras | Registruotas režimas su nurodytu adresu (`--registry URL`) — numatyta; DNS-SD (unicast, mDNS) ir P2P — nmos-cpp galimybės, įjungiamos parinktimi | Tik mDNS (neveikia tarp VLAN, WSL2, pod'ų) |
| `bridge` per IS-05 | BCP-007-03 ištekliai + `ext_vf_fabrics_target_info`, `ext_vf_fabrics_provider`, `ext_vf_fabrics_iface` (F-31) | Gamintojo transporto URN atskiram tinklo šuoliui (F-32 leidžia, bet 4 ištekliai vienam tiltui ir jokio valdiklio palaikymo); laukti AMWA apibrėžimo |
| GPU atminties matomumas | Gamintojo žymės ant Flow / Sender: `urn:x-vramflow:tag:payload-storage` (`host` \| `cuda-linear`), `urn:x-vramflow:tag:gpu-uuid`; Receiver'iui — importuojami tipai | Nieko (valdiklis nežinotų, kad reikia `bridge`); standarto šiandien nėra (F-08, F-30) |

Išteklių atvaizdavimas:

| VRAMFlow / MXL | NMOS | ID (UUIDv5 vardas) | Pastabos |
|---|---|---|---|
| Serveris | Node | `node` iš `.node/node.json` | BCP-002-02 žymės; `clocks`: `clk0` (vidinis arba PTP, jei žinomas) |
| Funkcijos paleidimas (`run_id`) | Device (`urn:x-nmos:device:pipeline`) | `device/<node>/<grupė>/<run_id>` | `controls`: IS-05 `sr-ctrl`, vėliau IS-12 `ncp`; stabilus tik su aiškiu `--run-id` |
| Rašomas flow | Source + Flow + Sender | `source/<mxl-flow-id>`, `flow/<node>/<domeno-id>/<mxl-flow-id>`, `sender/…` | `transport: urn:x-nmos:transport:mxl`; Flow iš `flow_def.json` be MXL papildomų laukų (`channel_count`); `mxl_flow_id` ≠ IS-04 Flow ID (leidžiama, F-30) |
| Funkcijos įėjimas | Receiver | `receiver/<node>/<run_id>/<įėjimo-vardas>` | `caps.media_types`, `constraint_sets` iš `VideoFormat` (BCP-004-01) |
| Grupė | — (domeno ID IS-05 parametruose) | `domain_def.json` `id` | IS-05 `/constraints` išvardija šiam Node pasiekiamus domenų ID |
| Rinkinys | `grouphint` ant Sender, Receiver, Source, Flow | — | Sritis `device`; žymė nekintama (F-35) |
| `bridge` tx | Device su Receiver'iu (skaito F domene A) | kaip aukščiau | `ext_vf_fabrics_*` Receiver'io parametruose |
| `bridge` rx | Device su Source + Flow + Sender (rašo F domene B) | kaip aukščiau | Flow `parents` = kilmės Flow ID, jei žinomas; `ext_vf_fabrics_*` Sender'io `/active` |
| P7 statistika, P9 aliarmai | IS-12 blokas su `NcReceiverMonitor` / `NcSenderMonitor` | `touchpoints` → IS-04 ID | E10 |

Būsenų atvaizdavimas į BCP-008 (E10) [mano]:

| BCP-008 laukas | VRAMFlow šaltinis | `Healthy` / `PartiallyHealthy` / `Unhealthy` |
|---|---|---|
| `connectionStatus` (Receiver) | `InputSlot` būsena, `late`, `lost` | `Attached` ir 0 prarastų intervale / veikia atkūrimas (`freeze`, pakartotinis jungimasis) / `Detached`, `Waiting` arba prarasta > 0 |
| `GetLostPacketCounters`, `GetLatePacketCounters` | `lost`, `late` suminiai | po vieną skaitiklį kiekvienam flow |
| `streamStatus` | INVALID grain'ai, formato neatitikimas | 0 INVALID / INVALID > 0 / `Mismatch` |
| `externalSynchronizationStatus` | laikrodžio stebėjimas (P1 `ClockWatch`, `bridge` `clock_offset_us`) | \|δ\| ≤ 1 ms ir 0 žingsnių / δ viršija ribą / laikrodžio žingsnis; `NotUsed`, kai šaltinis nežinomas |
| `linkStatus` | tik `bridge`: Fabrics jungties būsena; kitiems — `AllUp` | `AllUp` / `SomeDown` (1 → N) / `AllDown` |
| `transmissionStatus` (Sender) | rašytojo `skipped`, `too_late`, `bridge` `sent` klaidos | 0 / > 0 su atkūrimu / nerašoma |
| `essenceStatus` (Sender) | įėjimo turinio galiojimas (`in_invalid`), P9 aliarmai (juoda, sustingęs, tyla) | pagal aliarmų politiką |

### 3.5 C. Ryšys su baze

| Bazės darbas | Ko reikia šiam etapui | Būklė |
|---|---|---|
| P1 laikas | `index_of`, `time_of` spragų laikmačiui ir tempui; laikrodžio stebėjimas | padaryta |
| P2 formatai | `VideoFormat`, `format_from_flow_def`, `sample_begin` (garso ribos) | padaryta |
| P5 sutartis | `GapPolicy`, `VideoOut::skip`, `--on-loss` — rx spragų elgsena | P5a prieš E3 |
| P6 start / stop | `InputSlot` (tx įėjimai, re-attach), `DomainLease`, `OpenPolicy`; **+ E0 pakeitimai** | vyksta |
| P7 statistika | JSON eilutės; `links[]`; būsenų vardai — **+ E0** | prieš E4 |
| P4 priėmimas | `Ledger`, `Capacity`, `Demand`; **+ E0: knyga vienai GPU, grupės kvota** | prieš E4 / E5 |
| P8 talpa | talpos failas; nauja funkcija `bridge` harness'e | prieš E4 matavimus |
| P9 multivieweris | aliarmai → BCP-008 `essenceStatus` | prieš E10 |

### 3.6 Kaip matuojamas pridėtas vėlavimas

Kiekviena dalis — vieno serverio laikrodyje, todėl laikrodžių poslinkis neįeina:

- `send = t_done(i) − t_src(i)` (A laikrodis): nuo momento, kai tx skaitytojas gavo grain'ą i, iki Fabrics užbaigimo
  (remote write patvirtintas); apima D→H kopiją.
- `commit = t_commit(i) − t_cq(i)` (B laikrodis): nuo užbaigimo perskaitymo rx pusėje iki commit'o paskirties flow'e;
  apima H→D kopiją.
- **Pridėtas vėlavimas = send + commit.** Kryžminė patikra vienoje mašinoje: `verify` vėlavimo (dabar − `origin_tai`)
  skirtumas tarp domeno B ir domeno A turi sutapti su `send + commit` ± 0,5 ms.

---

## 4. Etapai

Eilė: **E0 → E1 → E2 → E3 → E4 → E5 → E6 → (E7)**; NMOS šaka **E8 → E9 → E10 → E11** gali eiti lygiagrečiai nuo E1.

| Etapas | Kas | Priklauso nuo | Nešiojamas (WSL2, 4070) | Darbo stotis (3090, natyvus) | Abi per 1 GbE | Lab |
|---|---|---|---|---|---|---|
| E0 | Bazės sąsajų atvėrimas | P4, P6, P7 dizainas | vienetų testai | ✔ | — | — |
| E1 | Grupės, rinkiniai, `vramflow-ctl` | P6 | ✔ (didžioji dalis be GPU) | ✔ | — | — |
| E2 | MXL + Fabrics build; `bridge` host → host, TCP, failų režimas | K-1, P6 | ✔ loopback | ✔ | 720p25 | — |
| E3 | `cuda-linear` galai, v210a, garsas, rinkinys, tempas, spragos | P5 | ✔ (viena GPU abiem pusėms) | ✔ | 720p25: 3090 ↔ 4070 | — |
| E4 | Nutrūkimai, rendezvous, laikrodis, statistika, priėmimas, 1 → N | P7, P4 | ✔ (jungties nutraukimas — jei branduolys leidžia) | ✔ | ✔ (tikras δ) | — |
| E5 | Dalijimasis tarp grupių, teisės, apskaita | E1, P4 | ✔ (du Unix naudotojai) | ✔ | — | — |
| E6 | `verbs` per host buferį | E3 | ✘ (F-25) | soft-RoCE **[nepatikrinta čia]** | ✘ | tikri NIC |
| E7 | GPUDirect | upstream, geležis | ✘ | ✘ (GeForce, be RDMA NIC) | ✘ | tik čia |
| E8 | NMOS failai + sidecar IS-04, IS-05 būsena | E1, K-9 | ✔ (registras ant `127.0.0.1`) | ✔ | ✔ | — |
| E9 | IS-05 valdymas, `bridge` per `ext_` | E0, E4, E8 | ✔ | ✔ | ✔ | — |
| E10 | IS-12 + BCP-008 | P7, (P9), E8 | ✔ | ✔ | — | — |
| E11 | TLS (BCP-003-01), IS-10 pagal poreikį | E8 | ✔ (savi sertifikatai) | ✔ | — | objekto PKI |

---

### E0. Bazės sąsajų atvėrimas (maži pakeitimai bazėje, kol ji dar daroma)

**Tikslas:** kad P4 / P6 / P7, kai bus baigti, neblokuotų tinklo, grupių ir NMOS. Tai [B] plano pataisos, ne nauja funkcija.

**Failai:** keisti `src/runtime/input_slot.hpp/.cpp`, `domain_lease.hpp/.cpp`; [B] §P4 (`Ledger`), §P7
(schema); `src/func/flows.hpp/.cpp` (ID ir `grouphint` generavimas per vieną vietą); `src/func/func_common.hpp` (`Opts`),
`main_func.cpp`. Sukurti `src/runtime/flow_ref.hpp` (header-only), `src/runtime/instance_cache.hpp/.cpp`, `tests/unit/test_flow_ref.cpp`.

**Sąsajos**

```cpp
namespace mxlv::runtime {
struct FlowRef { std::string domain; std::string flow_id; };      // domain — kelias; tuščias = funkcijos --domain
// "UUID" | "mxl:///abs/kelias?id=UUID" (MXL Addressability; authority privalo būti tuščia, lygiai vienas id).
bool parse_flow_ref(const std::string& s, const std::string& default_domain, FlowRef& out, std::string& err);
std::string format_flow_ref(const FlowRef& r);                    // visada mxl:/// forma

// Viena mxlInstance kiekvienam domenui procese (InputSlot::poll instanciją gauna iš kvietėjo — src/runtime/input_slot.hpp).
class InstanceCache {
public:
    mxlInstance get(const std::string& domain, std::string& err);   // sukuria pirmą kartą; tas pats kelias → ta pati instancija
    void release_all();
};
class InputSlot {   // papildymai; esamas konstruktorius (std::string flow_id, expect, on_change) ir poll(inst, now) lieka
public:
    void retarget(std::string flow_id);       // kitas poll(): atsijungia nuo seno ir jungiasi prie naujo; reattached() nedidėja;
                                              // kitas domenas = kvietėjas kitam poll() paduoda kitą instanciją iš InstanceCache
    void set_enabled(bool on);                // IS-05 master_enable; false → būsena Disabled, get() → MXL_ERR_FLOW_NOT_FOUND
    // State papildomas: Disabled
};
enum class Health : std::uint8_t { Inactive, Healthy, PartiallyHealthy, Unhealthy };   // vardai = BCP-008
const char* to_string(Health h);

struct GroupQuota { std::uint64_t vram_bytes = 0; double nvdec_mpix_s = 0, nvenc_mpix_s = 0; };   // 0 — neribota
struct Decision { bool admit; std::string engine; std::string scope; double used, requested, limit; };  // scope: "gpu" | "group"
class Ledger {
public:
    // ledger_dir — viena visam serveriui ir GPU: "<šaknis>/.gpu/<gpu-uuid>"; nebe domeno nuomos katalogas.
    static bool open(const std::string& ledger_dir, const std::string& device_uuid, Ledger& out, std::string& err);
    Decision reserve(const std::string& id, const std::string& group, const Demand& d, const Capacity& c, const GroupQuota* q);
    void release(const std::string& id);
    Demand total(const std::string& group = {}) const;            // tuščia — visos grupės
};
std::string default_ledger_dir(const std::string& root, const std::string& device_uuid);

class DomainLease {   // papildymai
public:
    bool set(const std::string& key, const std::string& value);   // „raktas=reikšmė“ eilutė; nežinomi raktai skaitant ignoruojami
    bool add_input(const std::string& name, const std::string& receiver_id, const std::string& caps_json);
};
}
namespace mxlv::func {
// Viena vieta flow ID ir grouphint generavimui (naudoja visos funkcijos).
std::string make_flow_id(const Opts& o, std::string_view bundle, std::string_view role, int legacy_role, std::uint32_t n);
std::string make_grouphint(std::string_view bundle, std::string_view role);    // "<bundle>:<role>"
}
```

CLI (bendros parinktys): `--input REF` (kartojama; `UUID` arba `mxl:///…?id=…`) šalia esamų `--in-role` / `--flow-base`;
`--group G`, `--root R` (jei duota — `--domain` išvedamas), `--bundle NAME`, `--export`.

P7 eilutės schema lieka `"v":1`; nauji **neprivalomi** laukai: viršuje `"group"`, `"state"`; kiekviename `flows[]` —
`"domain"`, `"bundle"`, `"role"`, `"state"`; naujas masyvas `"links"` (E4). Skaitytojai nežinomus laukus ignoruoja.

**Testai** (`tests/unit/test_flow_ref.cpp`, `tests/gpu/test_runtime.cpp`, `tests/unit/test_admission.cpp`)

- `flow_ref: tik UUID` — `"4d584c47-0001-4000-8000-000000000000"` + numatytas `/dev/shm/a` → `{"/dev/shm/a", tas pats UUID}`.
- `flow_ref: mxl URI` — `"mxl:///dev/shm/vf/studio1.pub?id=<uuid>"` → domenas `/dev/shm/vf/studio1.pub`; pirmyn-atgal per `format_flow_ref` duoda tą pačią eilutę.
- `flow_ref: atmetimas` — `mxl://host/…` (authority), du `id=`, nė vieno `id=`, UUID su `{}` ar didžiosiomis — visi `false` su tekstu.
- `input_slot: retarget` — prijungtas prie F1; `retarget(F2)` → per ≤ 200 ms `Attached` prie F2; `reattached() == 0`; grain'ai po to — iš F2 (barkodas).
- `input_slot: retarget į kitą domeną` — F2 kitame domene (instancija iš `InstanceCache`) → `Attached`; senas skaitytojas paleistas (atvirų deskriptorių skaičius nepadidėjo).
- `instance_cache` — du kvietimai tam pačiam keliui → tas pats `mxlInstance`; neegzistuojantis domenas → `nullptr` su tekstu.
- `input_slot: set_enabled(false)` — būsena `Disabled` per ≤ 1 `poll`; `get()` → `MXL_ERR_FLOW_NOT_FOUND` (kaip kitos neprijungtos būsenos); `set_enabled(true)` → `Attached` per ≤ 200 ms.
- `ledger: du domenai, viena GPU` — A (domenas 1) rezervuoja 4 GiB, B (domenas 2) su talpa 6 GiB × 0,85 = 5,1 GiB prašo 2 GiB → atmesta, `scope == "gpu"`, `used == 4 GiB`.
- `ledger: grupės kvota` — kvota 1 GiB, grupė jau turi 900 MiB, prašo 200 MiB → atmesta, `scope == "group"`, `limit == 1 GiB`; kita grupė tuo pačiu metu priimama.
- `lease: nežinomi raktai` — nuomos failas su eilute `ateitis=1` → `live_owners` ir `flow_owner` veikia kaip anksčiau.
- `flow_id: suderinamumas atgal` — be `--group` / `--bundle` `make_flow_id` grąžina seną `4d584c47-00RR-4000-8000-…` (esami testai nekinta).

**PR:** (1) `FlowRef` + `--input` + `InstanceCache` (~300); (2) `InputSlot::retarget`, `set_enabled` (~250); (3) [B] §P4 / §P7
teksto pataisos + `Ledger` signatūra, jei P4 dar nepradėtas (~150); (4) `make_flow_id` / `make_grouphint` (~200).

**Rizikos:** P6 šiuo metu daromas kito agento — E0 (2) derinti su juo, ne lygiagrečiai tuose pačiuose failuose.

---

### E1. Grupės ir rinkiniai

**Tikslas:** [R] G1, G2 ir G3.3: grupės sukūrimas, `domain_def.json`, vardai, rinkinių sąrašas, `BundleIn`, apskaita iš failų.

**Failai:** sukurti `src/group/group.hpp/.cpp`, `src/group/uuid5.hpp` (header-only; SHA-1 — savas arba iš OpenSSL, kuris jau
yra per FFmpeg? — **statytojas patikrina; nauja priklausomybė draudžiama be klausimo**), `src/runtime/bundle_in.hpp/.cpp`,
`src/ctl/main_ctl.cpp`, `cmake/group.cmake`, `tests/unit/test_group.cpp`, `tests/gpu/test_bundle_in.cpp`,
`tests/functional/groups.py`. Keisti: `tests/perf/domain.py` (`make_domain` rašo `domain_def.json`), `gen.cpp`
(`--bundle`, garsas tame pačiame rinkinyje), README.

**Sąsajos**

```cpp
namespace mxlv::group {
inline constexpr const char* kNamespaceUuid = "…";              // fiksuotas VRAMFLOW_NS (parenka statytojas, įrašo į ADR)
bool valid_name(std::string_view s);    // [a-z0-9]([a-z0-9._-]{0,61}[a-z0-9])?; be ':' ir '/'; nesibaigia ".pub", ".vramflow"
bool valid_role(std::string_view s);    // 1…32 spausdinami ASCII simboliai be ':'
std::string uuid5(std::string_view ns_uuid, std::string_view name);                 // mažosios raidės, 8-4-4-4-12
std::string flow_uuid(std::string_view group, std::string_view bundle, std::string_view role);   // uuid5(NS, "g/b/r")
std::string domain_uuid(std::string_view node_id, std::string_view group, bool pub);

struct Paths { std::string root, name, domain, pub_domain, state_dir, pub_state_dir; };
Paths paths(const std::string& root, const std::string& name);

struct DomainDef { std::string id, label, description; std::map<std::string, std::vector<std::string>> tags; };
bool read_domain_def(const std::string& domain, DomainDef& out, std::string& err);
bool write_domain_def(const std::string& domain, const DomainDef& d, std::string& err);   // laikinas failas + rename; kitas esamas id → klaida

struct Hint { std::string bundle, role, scope; };               // scope: "" | "device" | "node"
bool parse_hint(std::string_view s, Hint& out, std::string& err);
std::string make_hint(const Hint& h);

struct Config { int v = 1; std::string name; long gid = -1; unsigned mode = 0750, pub_mode = 0755; runtime::GroupQuota quota; };
enum class CreateResult { Created, Unchanged, Conflict, Error };
CreateResult create(const std::string& root, const Config& c, std::string& err);
bool load(const std::string& root, const std::string& name, Config& out, std::string& err);
bool ensure_pub(const std::string& root, const std::string& name, std::string& err);       // sukuria <grupė>.pub su domain_def.json
bool remove(const std::string& root, const std::string& name, std::string& err);           // tik be gyvų nuomų abiejuose domenuose
bool node_id(const std::string& root, std::string& out, std::string& err);                 // skaito arba sukuria .node/node.json

struct Member { std::string flow_id, role, media_type, storage, owner_run_id, device_uuid; vramflow::Rational rate; bool exported; };
using Bundles = std::map<std::string, std::vector<Member>>;
bool list_bundles(const Paths& p, Bundles& out, std::string& err);                          // skaito flow_def.json, payload.json, nuomas
struct Usage { std::uint32_t flows = 0; std::uint64_t ring_vram_bytes = 0, reserved_vram_bytes = 0, host_bytes = 0; };
bool usage(const Paths& p, const std::string& device_uuid, const std::string& ledger_dir, Usage& out, std::string& err);  // be CUDA
}
namespace mxlv::runtime {
struct BundleTick { mxlStatus status; std::uint64_t video_index; std::uint64_t audio_begin; std::uint32_t audio_count; };
class BundleIn {
public:
    bool open(mxlInstance inst, const std::string& domain, const std::string& bundle, const OpenPolicy& p,
              const std::atomic<bool>& stop, std::string& err);
    BundleTick wait(std::uint64_t video_index, std::uint64_t timeout_ns);   // sync grupė: visi nariai arba TOO_EARLY
    InputSlot* member(std::string_view role);                               // nullptr — tokios rolės nėra
    std::uint64_t incomplete() const;
};
}
```

CLI (`vramflow-ctl`; visos komandos turi `--root`, `--json`; išėjimo kodai: 0, 2 — blogi argumentai, 3 — konfliktas, 1 — kita):

```
vramflow-ctl group create --name G [--gid N] [--mode 0750] [--pub-mode 0755] [--vram-mb M] [--nvdec-mpix-s X] [--nvenc-mpix-s Y]
vramflow-ctl group list | show --name G | set --name G [...] | remove --name G
vramflow-ctl group usage [--name G] [--device-uuid GPU-…]
vramflow-ctl bundle list --group G        vramflow-ctl flow list --group G [--exported]
vramflow-ctl connect --group G --receiver ID (--flow REF | --disable)          # E9; rašo conn/<ID>.json
```

**Testai** (`tests/unit/test_group.cpp`, be GPU)

- `group: vardai` — galioja `studio1`, `a`, `cam-3.b_2`; negalioja ``, `Studio`, `a:b`, `a/b`, `.x`, `x.`, `x.pub`, `x.vramflow`, 64 simbolių eilutė.
- `group: keliai` — `paths("/dev/shm/vf", "studio1")` → `/dev/shm/vf/studio1`, `…/studio1.pub`, `…/studio1.vramflow`, `…/studio1.pub.vramflow`.
- `uuid5: prieš nepriklausomą etaloną` — 20 vardų: `uuid5(NS, vardas)` == Python `uuid.uuid5` reikšmėms iš `tests/unit/data/uuid5_vectors.json` (generuoja `tests/tools/gen_uuid5.py`); versijos skaitmuo `5`; žinomas vektorius: DNS vardų erdvė + `"www.example.com"` → `2ed6657d-e927-568b-95e1-2665a8aea6a2` (patikrinta 2026-11-11 su Python `uuid.uuid5`).
- `flow_uuid: savybės` — tas pats (g, b, r) → tas pats ID; pakeitus bet kurį narį — kitas; `flow_uuid("a", "b/c", "d") != flow_uuid("a/b", "c", "d")` (skirtukas varduose draudžiamas — tikrinama, kad `valid_name` tai užtikrina).
- `domain_def: pirmyn-atgal` — parašytas ir perskaitytas tapatus; faile lygiai raktai `id`, `label`, `description`, `tags`; `id` atitinka `^[0-9a-f]{8}-…$`.
- `domain_def: nekeičia svetimo` — domene jau yra `domain_def.json` su kitu `id` → `write_domain_def` klaida, failas nepakito (baitai).
- `hint: parse` — `"cam1:Video"` → {cam1, Video, ""}; `"cam1:Audio 1:device"` gerai; `"cam1"`, `"a:b:c:d"`, `":x"`, `"cam1:"`, `"cam1:Video:studio"` → klaida su tekstu.
- `group: create idempotentiška` — pirmas kvietimas `Created`, antras su tais pačiais `Unchanged` (failų `mtime` nepakito), su kita kvota `Conflict`.
- `group: režimai` — po `create` su `gid` G: `stat(domenas).st_mode & 0777 == 0750`, `st_gid == G`; `.vramflow` — 0770.
- `usage: iš failų` — netikras domenas su 3 flow katalogais, kurių `payload.cuda.json` `allocSize` 27 648 000, 27 648 000, 55 296 000 ir `deviceUuid` X → `ring_vram_bytes == 110 592 000`, `flows == 3`; kitas `deviceUuid` → 0.
- `list_bundles` — domenas su `cam1:Video`, `cam1:Audio`, `cam2:Video` → 2 rinkiniai, `cam1` turi 2 narius; flow be gyvos nuomos → `owner_run_id` tuščias.

**Testai su MXL** (`tests/gpu/test_bundle_in.cpp`; host ir `cuda-linear`)

- `bundle_in: visi nariai` — video 50/1 ir garsas 48 kHz rašomi sinchroniškai: 500 žingsnių, visi `MXL_STATUS_OK`, `audio_begin == 960·i`, `audio_count == 960`, `incomplete() == 0`.
- `bundle_in: garsas vėluoja` — garso rašytojas atsilieka 3 laukais: `wait(i, 10 ms)` → `MXL_ERR_OUT_OF_RANGE_TOO_EARLY`; `incomplete()` didėja po 1; video narys per `member("Video")` vis tiek skaitomas.
- `bundle_in: 59,94` — 60000/1001: `audio_count` seka 801, 801, 800, 801, 801 (5 grain'ai = 4 004 mėginiai, kaip [B] PF-5).
- `bundle_in: rinkinio nėra` — `open` su `OpenPolicy{wait_ms = 0}` → `false` per < 100 ms, tekste rasti rinkiniai.

**Funkciniai** (`tests/functional/groups.py`): G1.1–G1.5 ir G2.1–G2.4 iš [R] pažodžiui; papildomai `mxl-info -d <domenas> -l`
po `group create` → exit 0 (MXL nesutrinka dėl `domain_def.json`; F-43).

**Matavimai:** `vramflow-ctl bundle list` trukmė su 64 flow'ais (tikslas < 50 ms); `BundleIn::wait` kaina žingsnyje (< 10 µs virš atskirų skaitymų).

**PR:** (1) `uuid5` + vardai + testai (~250); (2) `domain_def`, `Paths`, `create` / `load` / `remove` (~350); (3) `list_bundles`,
`usage` (~300); (4) `vramflow-ctl group|bundle|flow` (~300); (5) `BundleIn` + testai (~350); (6) `gen --bundle`, harness'o
`domain_def.json`, `groups.py`, README (~300).

**Rizikos:** `grouphint` keitimas keičia `flow_def.json` — esami testai, lyginantys aprašą, atnaujinami tame pačiame PR
ir tik su autoriaus žinia. SHA-1 šaltinis (žr. Failai). Rolės `Video <n>` su tarpais — leidžiama (registras draudžia tik `:`).

---

### E2. MXL su Fabrics ir `bridge` host → host per TCP (failų režimas)

**Tikslas:** įrodyti grandinę su nepakeistu Fabrics: [R] N1.2 (host → host), N1.4–N1.6. Tai iš esmės `mxl-fabrics-demo`
elgsena VRAMFlow funkcijos pavidalu (tick variklis, skaitikliai, nuoma).

**Pirmas žingsnis — patikra, ne kodas:** sukompiliuoti MXL PR #653 medį su `-DMXL_ENABLE_FABRICS_OFI=ON` ir libfabric (K-1),
paleisti `mxl-fabrics-info` ir dokumentuotą pavyzdį §6.1 (`mxl-fabrics-demo`, tcp, `::1`) su host flow. Jei tai neveikia
WSL2 — **STOP**, ataskaita autoriui (alternatyva: tas pats žingsnis darbo stotyje). **[nepatikrinta čia]** (F-27).

**Failai:** sukurti `src/net/fabrics_link.hpp/.cpp` (plonas RAII apvalkalas aplink `mxl/fabrics.h`), `src/net/offer.hpp/.cpp`,
`src/net/pace.hpp` (header-only, gryna logika), `src/func/bridge.cpp`, `cmake/net.cmake` (taikinys `mxlv_net`, statomas tik
kai rastas `mxl::mxl-fabrics`; be jo `bridge` subkomandos nėra ir `--help` tai pasako), `tests/unit/test_pace.cpp`,
`tests/unit/test_offer.cpp`, `tests/gpu/test_fabrics_link.cpp`, `tests/functional/bridge.py`,
`scripts/linux/12-build-libfabric.sh`. Keisti: `scripts/…/10-build-mxl.sh` (`-DMXL_ENABLE_FABRICS_OFI`, valdoma kintamuoju),
`main_func.cpp`, `func_common.hpp` (`Opts`), `flows.hpp` (rolė `10` — bridge), README.

**Sąsajos**

```cpp
namespace mxlv::net {
enum class Provider : std::uint8_t { Auto, Tcp, Verbs, Shm };
enum class Stage : std::uint8_t { Auto, Host, Gpu };
struct Iface { Provider provider; std::string node, service; std::uint64_t caps = 0, max_msg = 0; std::string device; };
bool parse_provider(const std::string& s, Provider& out);
bool list_ifaces(mxlInstance inst, Provider want, std::vector<Iface>& out, std::string& err);   // mxlFabricsGetInterfaces
bool pick_iface(const std::vector<Iface>& all, Provider want, const std::string& node, Iface& out, std::string& err);  // Auto: verbs > tcp

struct FlowOffer { std::string flow_id; std::string target_info; std::uint32_t slots = 0; std::uint64_t grain_bytes = 0;
                   std::uint32_t planes = 0; std::string storage; };
struct Offer { int v = 1; std::string run_id, group, domain_id; Iface iface; std::vector<FlowOffer> flows; std::int64_t tai_ns = 0; };
std::string offer_to_json(const Offer& o);
bool offer_from_json(const std::string& s, Offer& out, std::string& err);        // v != 1 arba trūksta lauko → klaida su lauko vardu
enum class Compat { Ok, Provider, MaxMsg, Slots, GrainBytes, Planes };
Compat check_compat(const Iface& local, const FlowOffer& remote, std::uint32_t local_slots, std::uint64_t local_grain_bytes,
                    std::uint32_t local_planes, const Iface& remote_iface);

// Tempas: kuriuos indeksus siųsti. Gryna funkcija.
struct PaceIn { bool has_last; std::uint64_t last_sent; std::uint64_t head; std::uint32_t inflight, max_inflight; };
struct PaceOut { bool send; std::uint64_t first, last; std::uint64_t skipped; };
PaceOut pace(const PaceIn& in);      // siunčia naujausius min(head − last_sent, max_inflight − inflight); likę — skipped

class TargetLink {      // rx: vienas flow, apgaubia mxlFlowWriter (host)
public:
    bool setup(mxlInstance inst, mxlFlowWriter writer, const Iface& bind, std::uint32_t cq_depth, std::string& target_info, std::string& err);
    mxlStatus poll_grain(std::uint64_t& index);                      // ReadGrainNonBlocking; NOT_READY — nieko
    mxlStatus poll_samples(std::uint64_t& head, std::size_t& count);
    Health health() const;
};
class InitiatorLink {   // tx: vienas flow, apgaubia mxlFlowReader (host)
public:
    bool setup(mxlInstance inst, mxlFlowReader reader, const Iface& bind, std::uint32_t cq_depth, std::string& err);
    bool add_target(const std::string& target_info, std::string& err);
    bool remove_target(const std::string& target_info, std::string& err);
    mxlStatus send_grain(std::uint64_t index, std::uint16_t first_slice, std::uint16_t end_slice);   // (i, 0, 0) — tik antraštė
    mxlStatus send_samples(std::uint64_t head, std::size_t count);
    mxlStatus progress(std::uint32_t timeout_ms);                    // 0 — neblokuoja
    bool connected() const;
};
}
namespace mxlv::func { int run_bridge(const Opts&); }
```

CLI (E2 poaibis):

```
vramflow-func bridge --side rx --domain D --flow-def FILE [--flow-def FILE …] --offer-out FILE
                     [--provider tcp|verbs|shm|auto] [--node ADDR] [--service PORT] [--cq-depth N]
vramflow-func bridge --side tx --domain D --flow REF [--flow REF …] --offer-in FILE
                     [--provider …] [--node ADDR] [--max-inflight 2]
  + bendros: --run-id, --input-wait-ms, --duration, --warmup, --out, --stats-interval-s, --stats-out
```

Išėjimo kodai: 0; 2 — argumentai; 4 — įėjimo ar pasiūlymo nesulaukta; 7 — transportas neprieinamas; 8 — nesuderinama.

**Testai**

- `pace: normalus žingsnis` — `{has_last, last 100, head 101, inflight 0/5}` → `send`, `first == last == 101`, `skipped == 0`.
- `pace: pasivijimas ribose` — last 100, head 103, inflight 0/5 → 101…103, `skipped == 0`.
- `pace: pasivijimas virš ribos` — last 100, head 120, inflight 0/5 → 116…120, `skipped == 15`.
- `pace: pirmas žingsnis` — `has_last == false`, head 500 → tik 500.
- `pace: laikrodis atgal` — head 99, last 100 → `send == false`.
- `pace: skrydžio riba` — inflight 2/2 → `send == false`, `skipped == 0` (sprendžia kitas žingsnis).
- `pace: viena laisva vieta` — last 100, head 104, inflight 1/2 → siunčiamas tik 104, `skipped == 3`.
- `pace: numatytoji riba` — `max_inflight` didesnis už `slotCount / 2` (pvz. 8 iš 10) → argumentų klaida (kodas 2).
- `offer: pirmyn-atgal` — 3 flow'ų pasiūlymas → tapatus; `"v": 2` → klaida „versija“; be `target_info` → klaida su lauko vardu; 1 MiB eilutė → klaida (riba 256 KiB).
- `compat` — skirtingas teikėjas → `Provider`; `slots` 10 prieš 12 → `Slots`; `grain_bytes` 2 764 800 prieš 5 529 600 → `GrainBytes`; viskas vienoda → `Ok`.
- `fabrics_link: tcp per ::1` (`tests/gpu/test_fabrics_link.cpp`; GPU nereikia, bet reikia MXL) — host 1080i50 flow: 100 grain'ų su žinomu CRC → target pusėje 100 indeksų ta pačia tvarka, payload CRC lygūs, `info.index` lygūs; INVALID grain'as (`send_grain(i, 0, 0)`) → target'e `flags & MXL_GRAIN_FLAG_INVALID`.
- `fabrics_link: garsas` — continuous 48 kHz 2 kan.: 100 × 960 mėginių → target'e tie patys `headIndex` ir bitai.
- `fabrics_link: backend flow atmetamas` — `cuda-linear` flow skaitytojas → `setup` `false`, tekste „payload backend“ (fiksuoja F-04; kai upstream pasikeis, testas kris ir pasakys, kad E7 galima pradėti).
- Funkciniai (`bridge.py`): `bridge_tcp_host_host` ([R] N1.2: 60 s, 4 flow'ai, ≥ 2 994 iš 3 000, 0 CRC klaidų, `d2h_bytes == h2d_bytes == 0`); `bridge_flow_def_identical` (`cmp` A ir B `flow_def.json` → 0); `bridge_rx_absent` (N1.4); `bridge_flow_conflict` (N1.5); `bridge_incompatible` (N1.6); `bridge_no_fabrics_build` (binaras be `mxlv_net`: `vramflow-func bridge` → kodas 2, tekste kaip įjungti).

**Matavimai** (`docs/rezultatai/<data>-bridge-tcp.md`): `N | formatas | send p50 / p99 ms | commit p50 / p99 ms | CPU branduolių flow'ui tx / rx | pralaidumas MB/s | praleista`;
N = 1, 2, 4, 8 × 1080i50 ir 1, 2, 4 × 1080p50 per loopback; tas pats `mxl-fabrics-demo` kaip atskaitos taškas (N = 1).

**PR:** (1) libfabric + MXL Fabrics build skriptai, patikros ataskaita (~120 + dokumentas); (2) `pace`, `offer`, `compat` + testai
(~350); (3) `TargetLink` / `InitiatorLink` + testai (~400); (4) `run_bridge` host → host, failų režimas (~400); (5) `bridge.py`,
README, rezultatai (~300).

**Rizikos:** Fabrics API versija 0 (F-44) — gali keistis; visi kvietimai tik per `fabrics_link`. PR #653 medis + Fabrics
kartu upstream netestuoti. libfabric TCP per WSL2 loopback — našumas nežinomas. Tx procese tame pačiame flow yra ir
skaitytojas, ir (E3) rašytojas — leidžiama, bet tikrinama `fabrics_link` teste.

**Klausimai:** K-1.

---

### E3. `cuda-linear` galai, v210a, garsas, rinkinys, tempas, spragos

**Tikslas:** [R] N1.1–N1.3, N2 (visi), N5.2 (`--stage` elgsena be GPUDirect).

**Sprendimai**

- Tarpinis host flow: tx — rašytojas + skaitytojas (initiator'iui) `wire-<run_id>/` domene; rx — rašytojas (target'ui) ten pat.
  Kopijos: tx `VideoIn` lizdas → `cudaMemcpy` D→H tiesiai į tarpinio grain'o buferį (`OpenGrain`) → commit → `send_grain`;
  rx po užbaigimo: tarpinio grain'o buferis → `cudaMemcpyAsync` H→D į `VideoOut` lizdą → sinchronizacija → commit tuo pačiu
  indeksu. Skaitikliai `d2h_bytes`, `h2d_bytes`.
- `--storage as-source|host|cuda-linear` (rx): `as-source` (numatyta) ima iš pasiūlymo; `cuda-linear` be `--device-uuid` → klaida.
- `--stage auto|host|gpu`: E3 — `gpu` visada kodas 7 su tekstu „Fabrics be GPU atminties gebėjimo (MXL PR #654)“; `auto` → `host`.
- v210a: abi plokštumos vienu grain'u (Fabrics pilnam grain'ui daro vieną remote write, `Fabrics.md` §6.1.4).
- Spragos: `GapClock` (žr. §3.2) + `VideoOut::skip(i, policy)`; garsui spragos nerašomos (kaip [B] P3).
- Rinkinys: `--bundle NAME` išskleidžiamas per `group::list_bundles`; garso porcija žingsnyje i: `[sample_begin(i), sample_begin(i+1))`.

**Failai:** sukurti `src/net/gap_clock.hpp` (header-only), `src/net/staging.hpp/.cpp`, `tests/unit/test_gap_clock.cpp`,
`tests/gpu/test_staging.cpp`. Keisti: `bridge.cpp`, `flows.hpp/.cpp` (`AudioIn`, jei dar nėra — skaitymas mėginių ribomis),
`verify.cpp` (režimas `gen` kitame domene — jau veikia per `--domain`), `bridge.py`.

**Sąsajos**

```cpp
namespace mxlv::net {
struct GapConfig { std::int64_t deadline_ns = 30'000'000; std::int64_t window_ns = 10'000'000'000; };
class GapClock {
public:
    GapClock(timing::Rational rate, const GapConfig& c);
    void on_arrival(std::uint64_t index, std::int64_t arrival_tai_ns);
    bool ready() const;                                       // bent vienas atvykimas
    std::int64_t base_delay_ns() const;                       // slenkantis minimumas (atvykimas − time_of(indeksas))
    std::int64_t lost_at(std::uint64_t index) const;          // time_of(i) + base_delay + deadline
    // Indeksai (after, upto], kurių terminas jau praėjo ir kurie neatvyko.
    std::uint64_t due(std::uint64_t last_committed, std::int64_t now_tai_ns) const;
};
// Tarpinis host flow tame pačiame serveryje, tas pats flow ID, domenas wire-<run_id>.
class Staging {
public:
    bool create(mxlInstance wire_inst, const std::string& flow_def_json, std::string& err);   // rašytojas (+ skaitytojas tx pusėje)
    mxlFlowWriter writer() const;  mxlFlowReader reader() const;
    // tx: device → host; grąžina MXL statusą. rx: host → device.
    mxlStatus download(const mxlio::SlotView& src, std::uint64_t index, const mxlGrainInfo& info, cudaStream_t s);
    mxlStatus upload(std::uint64_t index, const mxlio::SlotView& dst, mxlGrainInfo& info, cudaStream_t s);
    std::uint64_t d2h_bytes() const, h2d_bytes() const;
};
}
```

CLI papildymai: rx `--storage`, `--device-uuid`, `--stage`, `--on-loss`, `--hold-ms`, `--deadline-ms`; tx `--bundle NAME`, `--stage`.

**Testai**

- `gap_clock: bazinis vėlinimas` — atvykimo amžiai 3,0; 3,2; 2,9; 3,1 ms → `base_delay_ns == 2 900 000`; `lost_at(i) == time_of(i) + 32 900 000`.
- `gap_clock: laikrodžių poslinkis` — tie patys amžiai + 2 s → `base_delay_ns == 2 002 900 000`; per 100 iš eilės atvykusių indeksų `due()` visada 0 (jokių klaidingų praradimų).
- `gap_clock: spraga` — indeksai 100, 101, 104 (102 ir 103 neatvyko): ties `lost_at(102)` → `due == 1`, ties `lost_at(103)` → `due == 2`.
- `gap_clock: nėra atvykimų` — `ready() == false`, `due() == 0` (nieko neskelbiama prarasta prieš pirmą grain'ą).
- `gap_clock: langas` — po 10 s amžiai pakyla nuo 3 iki 8 ms → `base_delay` per ≤ 10 s pasiekia 8 ms; per jungties tylą reikšmė nekinta.
- `staging: D→H→D bitų tikslumu` (`tests/gpu/test_staging.cpp`) — 1080i50 v210 ir v210a, 1080p50: šaltinio lizdo CRC == tarpinio host grain'o CRC == paskirties lizdo CRC; 100 grain'ų; `d2h_bytes == 100 × grain'o dydis`.
- `staging: slice'ai` — `info.validSlices == totalSlices` (540 laukui, 1 080 kadrui) abiejose pusėse.
- `staging: mažas grain'as` — 382×40 (61 440 B, < 64 KiB — [B] P10 H-B atvejis) → CRC lygūs (kopijos sinchronizuojamos prieš commit).
- Funkciniai: `bridge_tcp_matrix` (4 saugyklų deriniai × N1.1 ribos); `bridge_v210a` (N1.3); `bridge_formats` (1080i50, 1080p25, 1080p50: po 20 s, 0 CRC klaidų); `bridge_bundle_av` (N2.1: 60 impulsų, poslinkis 0 mėginių, garso bitai lygūs); `bridge_bundle_read` (N2.2: `bundle_incomplete ≤ 0,2 %`); `bridge_bundle_member_gone` (N2.3); `bridge_bundle_missing` (N2.4); `bridge_stage_gpu_refused` (N5.2: kodas 7, tekste „PR #654“; su `auto` — `stage: "host"` statistikoje); `bridge_chain` (domene B `logo` ant replikuoto flow + `verify --mode logo`: 0 klaidų, `late == 0` su `--deadline-ms 30`).

**Matavimai:** lentelė „galų derinys | N | send p50 / p99 | commit p50 / p99 | D→H MB/s | H→D MB/s | CPU branduolių | VRAM MB flow'ui | RAM MB flow'ui“;
1080i50 N = 1, 2, 4, 8; 1080p50 N = 1, 2, 4; palyginimas su tuo pačiu N be `bridge` (logo tiesiai domene A) — kiek
pablogėja `logo` lauko p99. Kopijos kelias: pageable prieš prisegtą (`cudaHostRegister`) tarpinį buferį — tik jei WSL2 leidžia **[nepatikrinta čia]**.

**PR:** (1) `GapClock` + testai (~250); (2) `Staging` + testai (~350); (3) `bridge` su `cuda-linear` galais, `--storage`, `--stage`
(~400); (4) spragų politika rx + v210a (~250); (5) garsas ir `--bundle` (~350); (6) funkciniai testai + rezultatai (~350).

**Rizikos:** vėluojantis remote write į lizdą, kuris jau pažymėtas INVALID, kai tarpinis flow == paskirties flow (host
galas) — INVALID grain'o turinys ir taip neapibrėžtas, bet testas `bridge_late_after_gap` (E4) tai užfiksuoja. Dvi PCIe kopijos
vienam grain'ui — matuojama, ne spėjama. `cudaHostRegister` WSL2 ribojimai.

---

### E4. Atsparumas: nutrūkimai, rendezvous, laikrodis, statistika, priėmimas, 1 → N

**Tikslas:** [R] N3 (visi), N6, N7; G5 mechanika.

**Sprendimai**

- Rendezvous: viena TCP jungtis tarp tx ir rx, JSON eilutės (`\n`), kiekviena su `"v":1,"t":…`. Žinutės: `want` (tx → rx: `run_id`,
  `group`, flow aprašai pažodžiui, `slots`, `grain_bytes`, `planes`, `storage`, pageidaujamas teikėjas), `offer` (rx → tx: §E2
  `Offer`), `reject` (rx → tx: `code`, `reason`, `scope`, `engine`, `used`, `requested`, `limit`), `ping` / `pong` (4 laiko žymės),
  `bye`. Jungties praradimas = signalas tx gauti naują `offer`. Didžiausia eilutė 256 KiB. `--allow CIDR` (kartojama) rx pusėje;
  be jo — tik `--listen` sąsajos tinklas.
- Failų režimas lieka: tx stebi `--offer-in` (`mtime` + dydis kas 200 ms) ir perskaito pasikeitusį.
- Būsenos: `Healthy` (jungtis yra, 0 praleistų intervale), `PartiallyHealthy` (`skipped > 0` arba vyksta pasivijimas),
  `Unhealthy` (jungties nėra), `Inactive` (išjungta per IS-05 / dar nepaleista).
- Priėmimas rx pusėje prieš kuriant flow'us: `Demand.vram_bytes` = paskirties ring'as (jei `cuda-linear`); knyga ir kvota — E0.
- 1 → N: tx priima kelis `--connect` / `--offer-in`; kiekvienam — atskira būsena ir skaitikliai; vieno target'o gedimas
  nestabdo kitų (tikrinama testu; jei Fabrics to neleidžia — atskiri `InitiatorLink` kiekvienam target'ui).

**Failai:** sukurti `src/net/rendezvous.hpp/.cpp`, `src/timing/offset.hpp` (header-only), `tests/unit/test_rendezvous.cpp`,
`tests/unit/test_offset.cpp`, `tests/functional/bridge_faults.py`, `tests/tools/netns_link.sh` (jungties nutraukimas:
tinklo vardų erdvė + `veth`; **tik su autoriaus leidimu**, kitaip testas `SKIP` su priežastimi). Keisti: `bridge.cpp`,
`StatsPublisher` (masyvas `links`), `tests/perf/capacity.py` (funkcija `bridge`), `compare_report.py`.

**Sąsajos**

```cpp
namespace mxlv::timing {
struct OffsetSample { std::int64_t offset_ns, delay_ns; };                       // offset = nutolęs − vietinis
OffsetSample estimate_offset(std::int64_t t1, std::int64_t t2, std::int64_t t3, std::int64_t t4);
class OffsetFilter { public: void add(OffsetSample s); bool valid() const; OffsetSample best() const; };   // mažiausio delay imtis iš 8
}
namespace mxlv::net {
enum class MsgType : std::uint8_t { Want, Offer, Reject, Ping, Pong, Bye };
struct Msg { MsgType type; std::string body_json; };
bool encode(const Msg& m, std::string& line, std::string& err);                  // viena eilutė su '\n'
bool decode(std::string_view line, Msg& out, std::string& err);                  // v != 1, nežinomas t, > 256 KiB → klaida
class RendezvousServer {        // rx
public:
    bool listen(const std::string& host_port, const std::vector<std::string>& allow_cidr, std::string& err);
    bool poll(Msg& out, int& peer_id);                 // neblokuoja; viena gija (ne tick gija)
    bool send(int peer_id, const Msg& m);
    void close(int peer_id);
};
class RendezvousClient {        // tx
public:
    bool connect(const std::string& host_port, std::uint32_t timeout_ms, std::string& err);
    bool send(const Msg& m);  bool poll(Msg& out);  bool alive() const;
};
struct LinkStats { std::string peer, provider, stage; runtime::Health state; std::uint64_t reconnects, link_down_ms, sent, recv,
                   skipped, src_missing, lost, too_late, bytes, d2h_bytes, h2d_bytes; std::int64_t clock_offset_us, path_delay_us; };
}
```

Statistikos `links[]` elementas (P7 eilutėje): `{"peer":"10.0.0.2:5800","provider":"tcp","stage":"host","state":"Healthy",
"reconnects":0,"link_down_ms":0,"sent":50,"skipped":0,"src_missing":0,"bytes":138649600,"send_p50_ms":1.9,"send_p99_ms":3.1,
"d2h_bytes":138240000,"clock_offset_us":12,"path_delay_us":180}`; rx pusėje vietoje `sent` / `send_*` — `recv`, `lost`,
`too_late`, `commit_p50_ms`, `commit_p99_ms`, `h2d_bytes`.

CLI papildymai: rx `--listen HOST:PORT`, `--allow CIDR`, `--admission off|warn|enforce`, `--group G`; tx `--connect HOST:PORT`
(kartojama — 1 → N), `--max-clock-offset-ms M` (0 — tik įspėti), `--reconnect-wait-s S` (−1 neribotai), `--group G`.
Pilna `bridge` parinkčių aibė po E4 = E2 + E3 + šios.

**Testai**

- `offset: formulė` — t1 = 1 000 000, t2 = 1 500 000, t3 = 1 510 000, t4 = 1 030 000 ns → `offset_ns == 490 000`, `delay_ns == 20 000`.
- `offset: filtras` — 8 imtys su delay {20, 90, 15, 300, 40, 25, 18, 60} µs → `best()` yra imtis su 15 µs; < 8 imčių → `valid() == false` tik kol 0 imčių.
- `rendezvous: žinutės pirmyn-atgal` — visi 6 tipai; `{"v":2,…}` → klaida „versija“; eilutė be `\n` per 256 KiB → klaida ir jungtis uždaroma.
- `rendezvous: leidžiami adresai` — `--allow 127.0.0.0/8`: jungtis iš `127.0.0.1` priimama, iš `::1` atmetama (uždaryta per ≤ 100 ms be atsakymo).
- `rendezvous: want → offer → ping` (dvi gijos viename teste) — `offer` gaunamas per ≤ 200 ms; po 8 `ping` `clock_offset_us` ∈ [−200, 200].
- `rendezvous: reject` — rx knyga pilna → tx gauna `reject` su `code == 5`, `scope`, `engine`, trimis skaičiais; tx baigiasi kodu 5.
- Funkciniai (`bridge_faults.py`):
  - `bridge_link_drop` — [R] N3.1 + N3.2: 5 s nutrūkimas; B skaitytojas su 40 ms timeout'u: 0 `TOO_EARLY`; INVALID indeksų skaičius 250 ± 25; `reconnects == 1`; `link_down_ms` 5 000 ± 500; galiojantys grain'ai per ≤ 2 s; pirmas pliūpsnis ≤ 5 grain'ai; indeksai monotoniški.
  - `bridge_rx_restart` — N3.3: `kill -9` rx, po 3 s paleidžiamas; tx tęsia per ≤ 5 s; tx PID tas pats; abiem režimais (rendezvous ir failas).
  - `bridge_tx_restart` — `kill -9` tx: rx gyvas, B pusėje INVALID pagal `--on-loss`; naujas tx prisijungia ir tęsia; rx `reconnects == 1`.
  - `bridge_slow_link` — N3.4: pralaidumas 50 % (`tc tbf` vardų erdvėje arba `SKIP`): `skipped > 0`, `send_p99_ms ≤ 100` (5 laukai), RSS pokytis ≤ 5 %.
  - `bridge_late_after_gap` — N3.5: tx sulaikomas (`SIGSTOP`) 100 ms ir paleidžiamas: rx `too_late ≥ 1`, nė vienas indeksas necommit'intas du kartus, vėlesni — galiojantys.
  - `bridge_on_loss_freeze` — `--on-loss freeze --hold-ms 1000`, 3 s nutrūkimas: pirma 1 s — paskutinio gero grain'o kopijos (CRC lygus), toliau — juoda; nė vieno INVALID.
  - `bridge_clock_offset` — N7.1 ir N7.2 (testinis kabliukas `VRAMFLOW_TEST_CLOCK_OFFSET_NS=2000000000` tik matavime): įspėjimas kas 10 s; `clock_offset_us` 2 000 000 ± 1 000; `lost == 0`; su `--max-clock-offset-ms 5` — kodas 8 per ≤ 5 s.
  - `bridge_admission_group` — N6.2: kvota 2 ring'ai (55 296 000 B), prašoma 3 → trečias `reject`, `scope == "group"`; pirmi du: 0 prarastų.
  - `bridge_admission_gpu` — N6.3: dirbtinis talpos failas (VRAM 4 ring'ai / 0,85), dvi grupės po 3 → priimta 4, `scope == "gpu"`.
  - `bridge_fan_in` — N6.1: du tx į vieną rx serverį, `logo` ant 4 flow'ų: `late == 0`, 0 klaidų.
  - `bridge_fan_out` — vienas tx, du rx (du domenai): abu gauna ≥ 99,8 %; vienas rx nužudomas → kitas nepraranda nė vieno grain'o.
  - `bridge_two_hosts` (tik su dviem mašinomis; kitaip `SKIP`) — 720p25, 3090 → 4070 per 1 GbE, 10 min: 0 CRC klaidų, ≥ 99,8 %, užrašyti `clock_offset_us` (min / maks.), `send` ir `commit` p99; privaloma riba: pridėtas vėlavimas p99 ≤ 40 ms (1 kadras).

**Matavimai:** atsistatymo laikas (ms; p50 / maks. iš 20 bandymų) po jungties nutrūkimo, rx perkrovimo, tx perkrovimo;
`clock_offset_us` stabilumas per 10 min tarp dviejų mašinų (ir prieš `chronyc tracking`); talpa: `capacity.py --func bridge`
kelio N per loopback (TCP) kiekvienam galų deriniui; 1 → N kaina (CPU flow'ui prieš N).

**PR:** (1) `offset` + testai (~150); (2) rendezvous žinutės + serveris / klientas + testai (~400); (3) `bridge` su rendezvous,
pakartotinis jungimasis, failo stebėjimas (~400); (4) `links[]` statistika, būsenos (~250); (5) priėmimas rx + `reject` (~300);
(6) 1 → N (~300); (7) `bridge_faults.py` + `netns_link.sh` (~400); (8) `capacity.py --func bridge` + rezultatai (~300).

**Rizikos:** jungties nutraukimui reikia root ir branduolio galimybių (`veth`, `netns`, `tbf`) — WSL2 **[nepatikrinta čia]**;
be jų lieka procesų žudymo testai. Fabrics pakartotinio jungimosi laikas (F-12) nežinomas — 2 s riba gali reikalauti
`MakeProgress` dažnio derinimo. Rendezvous — neautentifikuotas (K-11).

**Klausimai:** K-3, K-7, K-8, K-11.

---

### E5. Dalijimasis tarp grupių tame pačiame serveryje, teisės, apskaita

**Tikslas:** [R] G3 (visi), G4 (visi), G5.2.

**Sprendimai:** §3.3. `--export` → flow kuriamas `<grupė>.pub` domene (`ensure_pub`), nuoma — `.pub.vramflow`. Skaitytojui
eksporto domene: atskira `mxlInstance` (E0 `InstanceCache`); skaitytojas niekada nekviečia `mxlGarbageCollectFlows`
svetimam domenui. Kai `cuda-linear` flow atidaryti nepavyksta dėl UID (F-16), skaitytojo klaida tai pasako ir siūlo host
eksportą. Griežtoje izoliacijoje `--export` su `cuda-linear` — įspėjimas paleidžiant („kitas UID šio flow neatidarys“).

**Failai:** keisti `flows.cpp` (`VideoOut::create` — lizdo teisės; domeno parinkimas pagal `--export`), `func_common.cpp`
(instancijų talpykla pagal domeną), `bridge.cpp` (tx atmeta neeksportuotus, kodas 2), `main_ctl.cpp` (`group usage`).
Sukurti `tests/functional/sharing.py`, `tests/tools/as_user.sh` (paleidimas kitu UID / gid; reikia root — WSL čia root),
`upstream/U5-payload-sock-mode/` (README, eksperimento skriptas, siūlomas pataisymas — siunčia autorius).

**Testai** (`sharing.py`; du testiniai naudotojai `vf-u1` (gid G1), `vf-u2` (kita grupė), sukuriami ir pašalinami testo — **su leidimu**)

- `share_isolation_host` — G3.1: `vf-u2` skaitytojas privačiam host flow → kūrimas nepavyksta; rašytojo `fields_ok` nepakito.
- `share_isolation_cuda` — G3.1: `vf-u2` `MapSlots` → klaida; deskriptorius neišduotas; rašytojo `fields_ok` nepakito.
- `share_no_accidental_cross_group` — G3.1a: tas pats UID; `logo --group studio2 --bundle cam1 --input-wait-ms 2000`, kai `cam1` yra tik `studio1`: kodas 4 per ≤ 2,5 s; nuomos faile ir statistikoje nėra `studio1` domeno kelio.
- `share_same_uid_other_process` — G3.2: tas pats UID, atskiras procesas: `cuda-linear` 60 s, 0 CRC klaidų.
- `share_cuda_other_uid_same_group` — G3.2a: `vf-u1` (pagrindinė grupė G1, kitas UID): šiandien laukiama klaida su aiškiu tekstu; testas užrašo aptiktą elgseną (`same-uid-only` arba `group-readable`) į rezultatus ir krenta tik jei elgsena nei viena, nei kita (pirmas paleidimas — F-16 patikra vykdymu; rezultatas → `upstream/U5`).
- `share_cuda_other_uid_after_chmod` (diagnostinis, į `upstream/U5`) — tas pats, bet prieš skaitymą `chmod 0666 <flow>/payload.sock`: užrašoma, ar `MapSlots` vis tiek atmetamas (laukiama: taip — atmeta `peerMayRead`).
- `share_export_zero_copy_cuda` — G4.1 (tas pats UID): `nvidia-smi` naudojama atmintis prieš / po skaitytojo: pokytis ≤ 2 MiB; `d2h_bytes == 0`; 0 CRC klaidų; skaitymo p99 skirtumas ≤ 0,2 ms.
- `share_export_zero_copy_host` — G4.2 (kitas UID, eksporto domenas 0750 su bendra gid): 0 CRC klaidų; `/dev/shm` užimtumo pokytis ≤ 1 MiB.
- `share_export_cuda_via_host` — G4.2a: vietinis `bridge --provider shm` (jei `shm` netinka — `tcp` per `127.0.0.1`) `cuda-linear` → host į `.pub`; kitas UID skaito: 0 CRC klaidų, indeksai tapatūs, pridėtas vėlavimas p99 ≤ 5 ms.
- `share_private_not_readable` — G4.3: `--input mxl:///…/studio1?id=…` iš kitos grupės → kodas 4 arba prieigos klaida per ≤ `--input-wait-ms`.
- `share_reader_cannot_damage` — G4.4: skaitytojo grupės procesas bando `mxlCreateFlowWriter` tam pačiam ID `.pub` domene → klaida; po bandymo flow katalogo failų sąrašas ir `payload.sock` inode nepakitę.
- `share_usage` — G3.3: 4 + 2 flow'ai → `group usage --json`: `ring_vram_bytes` = 4 × ir 2 × `allocSize` (iš `payload.cuda.json`); suma prieš `nvidia-smi` pokytį ± 10 %.
- `share_quota` — G3.4: kvota = 4 ring'ai; penktas rašytojas su `--admission enforce` → kodas 5, `group == "studio1"`; `studio2` rašytojas priimamas.
- `share_bridge_private_refused` — G5.2.

**Matavimai:** skaitymo vėlavimas grupės viduje prieš per `.pub` (turi nesiskirti); VRAM vienam papildomam skaitytojui (MB).

**PR:** (1) `--export`, `ensure_pub`, instancijos pagal domeną (~300); (2) UID ribojimo klaidos tekstas ir įspėjimas + testai (~150); (3) `group usage`
+ kvotos kelias funkcijose (~300); (4) `sharing.py` + `as_user.sh` (~350); (5) `upstream/U5` medžiaga (dokumentas + skriptas).

**Rizikos:** F-16 išvada apie UID padaryta iš kodo — pirmas E5 žingsnis ją patikrina vykdymu (`share_cuda_other_uid_same_group`).
Upstream gali pakeisti lizdo logiką (leidimai pagal paskelbtą katalogą, `SO_PEERGROUPS`) — tada griežta izoliacija gaus
zero-copy eksportą be mūsų pakeitimų (testas tai aptiks).
Testams reikia kurti OS naudotojus. `0755` eksporto domenas = matomas visiems serverio naudotojams — numatyta 0750 su
atskira „share“ gid, jei autorius taip nuspręs (K-6).

---

### E6. RDMA (`verbs`) per host buferį

**Tikslas:** [R] N4. Kodo beveik nereikia (`--provider verbs` jau eina per `fabrics_link`) — tai patikros ir matavimų etapas.

**Žingsniai:** (1) darbo stotyje (natyvus Ubuntu, **su leidimu**): `modprobe rdma_rxe`, `rdma link add rxe0 type rxe netdev <if>`,
`mxl-fabrics-info -p verbs` → bent viena sąsaja **[nepatikrinta čia, F-48]**; (2) `bridge.py` ir `bridge_faults.py` su
`--provider verbs` (tie patys kriterijai); (3) lab'e su tikromis RoCE plokštėmis: QoS pagal `FabricsGettingStarted.md` §7
(DSCP 26 → prioritetas 3, PFC), talpos kopėčios.

**Testai:** `bridge_verbs_host_host`, `bridge_verbs_matrix` (N4.1 = N1.1 ribos); `bridge_verbs_absent` (N4.2: mašinoje be RDMA
įrenginio kodas 7 per ≤ 1 s, tekste sąsajų sąrašas; `--provider auto` → `tcp`, parašyta stderr ir statistikoje);
`bridge_verbs_cq_depth` — N = 8 flow'ų, `--cq-depth` 8 prieš 256: su per mažu gyliu jungtis nutrūksta (laukiama pagal
`Fabrics.md` §11.1) ir `bridge` atsistato (`reconnects ≥ 1`), su 256 — `reconnects == 0`.

**Matavimai:** „tcp prieš verbs“ lentelė: `send` / `commit` p50 / p99, CPU branduolių flow'ui abiejose pusėse, kelio N.
Soft-RoCE skaičiai žymimi „funkcinis, ne našumo“ (programinis RDMA paprastai lėtesnis už TCP). Lab'e: kelio N 1080p50 ant
25 / 100 GbE; lūkestis ~2 ms flow'ui (F-42, nepatikrinta), PASS riba p99 ≤ 5 ms.

**Rizikos:** `rxe` stabilumas; `FI_MR_CACHE_MONITOR=disabled`, kurį nustato Fabrics (`FabricInstance.cpp:22`), ir tmpfs
puslapių prisegimas („Domain on ext4 / `/tmp` is expected to fail MR pin… use `/dev/shm`“ — PR #654 pastaba, F-06).

---

### E7. GPUDirect (VRAM → NIC → VRAM) — sąlyginis

**Pradžios sąlygos (visos):** (1) upstream MXL turi Fabrics GPU atminties kelią (PR #654 ar įpėdinis sulietas, arba autorius
nusprendžia testuoti PR šaką lab'e); (2) lab'e yra `mlx5` NIC ir GPU su GPUDirect RDMA palaikymu abiejose pusėse (K-12);
(3) E6 praėjo ant tų pačių mašinų. Indikatorius repo viduje: testas `fabrics_link: backend flow atmetamas` (E2) pradeda kristi.

**Kas keičiasi:** `Stage::Gpu` — tarpinio host flow nėra; `InitiatorLink` / `TargetLink` gauna `cuda-linear` skaitytoją / rašytoją
tiesiai; gebėjimas tikrinamas per Fabrics sąsajų vėliavėles (`MXL_FABRICS_IFACE_CAP_HMEM` pagal F-06 — vardas gali keistis).
Jei upstream reikalauja proceso vietinių `cudaMalloc` buferių (F-06: IPC importuota atmintis neregistruojama) — viena GPU
kopija kiekvienoje pusėje lieka (D→D, ne per PCIe) ir tai parašoma rezultatuose.

**Testai:** N5.1 (`d2h_bytes == 0`, `h2d_bytes == 0`, N1.1 ribos, p99 ≤ `--stage host` reikšmė tame pačiame stende);
`bridge_gpu_refused_reason` — trys neigiami variantai (TCP teikėjas; Fabrics be gebėjimo; registracijos klaida): kodas 7 ir
skirtingas, konkretus tekstas kiekvienam.

**Matavimai:** `stage host` prieš `stage gpu`: pridėtas vėlavimas, CPU branduolių flow'ui, PCIe apkrova, kelio N.

**Rizikos:** ar `cuMemCreate` (VMM) atmintis, kurią naudoja `cuda-linear` backend'as, registruojama per `nvidia-peermem` /
dma-buf — neaišku (NVIDIA dok. kalba apie `cudaMalloc` puslapius, F-24); PCIe topologija (tas pats root complex); IOMMU.
**Šis etapas gali likti neįgyvendintas — tai priimtina; `--stage auto` tada visada reiškia `host`.**

---

### E8. NMOS: failai „paruošti NMOS“ ir sidecar su IS-04 (+ IS-05 būsena)

**Tikslas:** [R] M1. Mažiausias naudingas žingsnis: valdiklis **mato** viską, ką daro VRAMFlow, ir dabartines jungtis.

**Dalis be priklausomybių (galima iškart po E1):** `domain_def.json` visur; stabilūs ID; `grouphint` pagal registro taisykles;
nuomos faile `input=` eilutės (įėjimo vardas, Receiver ID, gebėjimai); `.node/node.json`; BCP-002-02 reikšmės
(`manufacturer`, `product`, `instance-id`) faile `.node/node.json` — tikslūs žymių URN iš registro tikrinami prieš rašant (F-36 ◐).

**Sidecar'as:** `vramflow-nmos --root R [--group G …] --registry URL | --dns-sd unicast|mdns | --p2p [--http-port 3212]
[--poll-ms 500] [--label TEXT]`. Kas `--poll-ms` nuskaito failus, lygina su ankstesne būsena ir atnaujina nmos-cpp modelį
(`version` keičiama tik pasikeitus turiniui). IS-05: `/constraints` — pasiekiamų domenų ID ir esamas flow ID; `/active` —
tikra būsena iš `conn/*.json` ir nuomų; `PATCH /staged`, kuris ką nors keistų, E8 etape atmetamas (HTTP 400 su tekstu
„valdymas neįjungtas“) — valdymas ateina E9.

**Failai:** sukurti `src/nmos/model.hpp/.cpp` (failai → išteklių aprašai; **be** nmos-cpp, testuojama be tinklo),
`src/nmos/main_nmos.cpp` (nmos-cpp klijai), `cmake/nmos.cmake`, `tests/unit/test_nmos_model.cpp`,
`tests/functional/nmos.py`, `scripts/linux/14-build-nmos-cpp.sh`, `docs/architektura/<data>-nmos.md` (EN, ADR).

**Sąsajos** (`model` — gryna logika; JSON kaip `std::string`, kad nebūtų priklausomybės nuo nmos-cpp tipų)

```cpp
namespace mxlv::nmos {
struct Ids { std::string node, device, source, flow, sender, receiver; };
std::string device_id(std::string_view node, std::string_view group, std::string_view run_id);
std::string source_id(std::string_view mxl_flow_id);
std::string flow_id(std::string_view node, std::string_view domain_id, std::string_view mxl_flow_id);
std::string sender_id(std::string_view node, std::string_view domain_id, std::string_view mxl_flow_id);
std::string receiver_id(std::string_view node, std::string_view run_id, std::string_view input_name);

struct Resource { std::string type, id, json; };                 // type: node|device|source|flow|sender|receiver
struct Snapshot { std::vector<Resource> resources; };
// Nuskaito šaknį (grupes, domenus, nuomas, conn/) ir pagamina IS-04 v1.3 išteklius. Be tinklo ir be GPU.
bool build_snapshot(const std::string& root, const std::vector<std::string>& groups, Snapshot& out, std::string& err);
struct Diff { std::vector<Resource> added, modified; std::vector<std::pair<std::string, std::string>> removed; };
Diff diff(const Snapshot& before, const Snapshot& after);
// MXL flow_def.json → IS-04 Flow (pašalina MXL papildomus laukus, prideda source_id, device_id, version).
bool flow_from_mxl_def(const std::string& flow_def_json, const Ids& ids, std::string& out_json, std::string& err);
bool receiver_caps(const vramflow::VideoFormat& f, bool alpha, std::string& out_json);       // BCP-004-01 constraint_sets
struct Connection { bool master_enable; std::string mxl_domain_id, mxl_flow_id; std::map<std::string, std::string> ext; };
bool read_connection(const std::string& state_dir, const std::string& receiver_id, Connection& out, std::string& err);
}
```

**Testai**

- `nmos_model: ID stabilumas` — tie patys įėjimai → tie patys ID; kitas `node` → visi kiti; `flow_id` skiriasi tarp dviejų domenų tam pačiam MXL flow.
- `nmos_model: Flow iš flow_def` — 1080i50 v210: `frame_width == 1920`, `frame_height == 1080`, `interlace_mode == "interlaced_tff"`, `grain_rate == {25, 1}`, `media_type == "video/v210"`, `colorspace == "BT709"`; garsas: nėra `channel_count` rakto, `sample_rate == {48000, 1}`, `bit_depth == 32`; žymė `grouphint` perkelta nepakeista.
- `nmos_model: Sender` — `transport == "urn:x-nmos:transport:mxl"`, `interface_bindings == []`, `manifest_href == null`, `flow_id` rodo į Flow, `subscription.active == true`.
- `nmos_model: Receiver` — `transport` tas pats; `format == "urn:x-nmos:format:video"`; `caps.media_types == ["video/v210"]`; `constraint_sets` turi rastrą ir `grain_rate`; be jungties `subscription == {sender_id: null, active: false}`.
- `nmos_model: schema` — kiekvienas pagamintas išteklius galioja prieš IS-04 v1.3.3 JSON schemas (schemos — testo duomenys iš `AMWA-TV/is-04` žymos `v1.3.3`).
- `nmos_model: diff` — pridėtas flow → `added` 3 ištekliai (source, flow, sender); pasikeitusi tik jungtis → `modified` 1 (receiver); funkcija dingo → `removed` visi jos ištekliai; niekas nepakito → tuščias `Diff` ir `version` tie patys.
- Funkciniai (`nmos.py`; `nmos-cpp-registry` ir `nmos-testing` ant `127.0.0.1`, mDNS išjungtas; jei įrankių nėra — `SKIP` su priežastimi): M1.1 (≤ 5 s iki pasirodymo Query API; laukai lygūs), M1.2 (IS-04-01: 0 FAIL), M1.3 (registras išjungtas 30 s: funkcijų `late == 0`; grįžus — ištekliai per ≤ 15 s; sidecar'as `kill -9`: funkcijos nepaveiktos), M1.4 (≤ 2 s / ≤ 15 s); IS-04-03 (peer-to-peer) — tik jei įjungtas `--p2p` ir mDNS veikia stende.

**Matavimai:** sidecar'o CPU (tikslas < 1 % branduolio su 64 flow'ais, `--poll-ms 500`) ir RSS; laikas nuo flow sukūrimo iki registro.

**PR:** (1) NMOS paruošti failai (nuomos `input=`, `node.json`) (~250); (2) `model` + testai (~400); (3) nmos-cpp build skriptas +
`main_nmos` IS-04 (~400); (4) IS-05 tik būsena (~300); (5) `nmos.py` + ADR (~350).

**Rizikos:** nmos-cpp build svoris (Conan 2, Boost, OpenSSL, DNS-SD biblioteka) ir žymų nebuvimas (pin pagal SHA). BCP-007-03
labai nauja; `urn:x-nmos:transport:mxl` dar ne registre (F-33) — gali keistis. DNS-SD WSL2 nepatikimas — todėl `--registry URL`.
`nmos-testing` paleidimas (Docker ar Python iš šaltinio) **[nepatikrinta čia]**.

**Klausimai:** K-9, K-10.

---

### E9. IS-05 valdymas ir `bridge` per `ext_` parametrus

**Tikslas:** [R] M2, M3.

**Sprendimai:** Receiver'io aktyvavimas → sidecar'as atomiškai rašo `conn/<receiver-id>.json` → funkcija (`InputSlot::retarget`,
`set_enabled`) per ≤ 1 s → funkcija patvirtina nuomos faile (`conn.<name>=<flow-ref>|disabled`) → sidecar'as atnaujina `/active`
ir IS-04 `subscription`. Atmetama HTTP 400: nepasiekiamas domeno ID; Receiver'io `mxl_flow_id: "auto"`; `transport_file` su
duomenimis; formatas, kurio funkcija nepriima (pagal gebėjimus). Sender'iui: `mxl_flow_id` / `mxl_domain_id` keisti negalima
(400, „cannot be applied in the current operating context“); `master_enable: false` → funkcija nebecommit'ina į tą išėjimą.
Planuoti aktyvavimai (`activate_scheduled_*`) — pagal nmos-cpp galimybes; jei reikia papildomo darbo — atidedama.
`bridge`: rx Sender'io `/active` skelbia `ext_vf_fabrics_target_info` (base64), `ext_vf_fabrics_provider`,
`ext_vf_fabrics_iface`; tx Receiver'is juos priima `/staged`; tx `bridge` paleidžiamas su `--offer-in conn` (laukia jungties failo).

IN-001 (F-34) atitikimas: flow ir domeno ID nustatymas / gavimas — IS-05 parametrai; idempotentiški start / stop —
`master_enable`; pasiekiami domenai — `/constraints`; būsena `started` / `stopped` — `/active`; struktūrizuotos klaidos —
IS-05 klaidos objektas (`code`, `error`, `debug`).

**Failai:** keisti `src/nmos/main_nmos.cpp`, `model.cpp`, `bridge.cpp` (`--offer-in conn`, `--offer-out conn`), `main_ctl.cpp`
(`connect`), funkcijų ciklai (jungties failo stebėjimas — viena bendra pagalbinė `runtime::ConnWatch`). Sukurti
`src/runtime/conn_watch.hpp/.cpp`, `tests/gpu/test_conn_watch.cpp`.

**Sąsajos**

```cpp
namespace mxlv::runtime {
struct ConnState { bool present = false, master_enable = true; FlowRef ref; std::map<std::string, std::string> ext; std::uint64_t serial = 0; };
class ConnWatch {
public:
    ConnWatch(std::string state_dir, std::string receiver_id);
    bool poll(ConnState& out);          // true — pasikeitė nuo praeito karto; ne dažniau kaip kas 100 ms; neblokuoja
};
bool write_conn(const std::string& state_dir, const std::string& receiver_id, const ConnState& s, std::string& err);  // laikinas + rename
}
```

**Testai**

- `conn_watch: pasikeitimas` — parašytas failas → `poll` per ≤ 200 ms `true` su ta pačia `FlowRef`; nepakitęs failas → `false`; sugadintas JSON → `false`, klaida į žurnalą, ankstesnė būsena galioja.
- `conn_watch: be failo` — `present == false`; funkcija elgiasi pagal CLI argumentus (NMOS nebūtinas).
- Funkciniai (`nmos.py`): M2.1 (PATCH → išėjimo barkodas atitinka F2 per ≤ 1 s; `vramflow-ctl connect` duoda tą patį), M2.2, M2.3 (trys 400 atvejai; būsena nepakito), M2.4 (IS-05-01, IS-05-02, BCP-007-03-01: 0 FAIL), M3.1 (replikacija per ≤ 2 s; N1.1 ribos), M3.2 (400), M3.3 (be sidecar'o veikia).

**Rizikos:** BCP-007-03 „`master_enable: false` MUST stop the MXL write operation“ prieš [B] K3 („commit'inti kiekvieną indeksą“) —
sprendimas: IS-05 išjungimas yra aiškus operatoriaus veiksmas, todėl rašymas stabdomas; skaitytojai tai mato kaip dingusį
šaltinį (`InputSlot` → `Detached`). `ext_` parametrai — gamintojo; kiti valdikliai jų nesupras (jiems lieka failų / rendezvous kelias).

---

### E10. IS-12 + BCP-008-01 / -02 (po P7; aliarmai — po P9)

**Tikslas:** [R] M4. Sidecar'as skaito P7 JSON eilutes (`stats/*.jsonl` arba `--stats-out` kelius iš nuomų) ir palaiko
MS-05-02 modelį: šakninis blokas → blokas kiekvienam Device → `NcReceiverMonitor` kiekvienam įėjimui, `NcSenderMonitor`
kiekvienam išėjimui, su `touchpoints` į IS-04 ID. Atvaizdavimas — §3.4 lentelė. `statusReportingDelay` numatyta 3 s;
perėjimai į blogesnę būseną — iškart, į geresnę — po delsos; perėjimų skaitikliai; `autoResetCountersAndMessages`.

**Sąsajos:** `namespace mxlv::nmos { struct MonitorState { runtime::Health overall, link, connection, sync, stream; std::string
message; std::uint64_t lost, late; std::uint64_t transitions[4]; }; MonitorState fold(const MonitorState& prev, const
StatsLine& line, std::int64_t now_tai_ns, std::int64_t reporting_delay_ns); bool parse_stats_line(std::string_view json,
StatsLine& out, std::string& err); }` — gryna logika, testuojama be nmos-cpp.

**Testai:** `monitor: sveika` (30 eilučių be praradimų → visi `Healthy`); `monitor: praradimas` (eilutė su `lost: 3` →
`connection == Unhealthy` toje pačioje eilutėje, `transitions[connection] == 1`); `monitor: delsa į geresnę` (po praradimo
švarios eilutės: 2,9 s — vis dar `Unhealthy`, 3,0 s — `Healthy`); `monitor: overall = blogiausia` (`PartiallyHealthy` +
`Unhealthy` → `Unhealthy`); `monitor: nėra statistikos` (3 intervalai be eilučių → `Unhealthy`, žinutėje „nėra statistikos“);
`monitor: žinutė „Previously: …“` grįžus į `Healthy`. Funkciniai: M4.1–M4.4 (IS-12-01, BCP-008-01-01, BCP-008-02-01: 0 FAIL).

**Rizikos:** BCP-008 sritys sukurtos tinklo srautams; MXL atvaizdavimas — mūsų interpretacija (specifikacijos MXL nemini,
F-30) — užrašyti ADR ir, jei autorius nori, pasiūlyti AMWA.

---

### E11. Saugumas: TLS, autorizacija

**Tikslas:** sidecar'o HTTP ir WebSocket per TLS (BCP-003-01 v1.0.1: TLS 1.2+, sertifikatų SAN) — parinktis `--tls-cert`,
`--tls-key`, `--tls-ca`; numatyta išjungta. IS-10 (JWT tikrinimas) — tik jei objektas to reikalauja; autorizacijos serveris —
objekto. Rendezvous ir Fabrics lieka be šifravimo (F-10) — dokumentuojama: atskiras VLAN, `--allow`.

**Testai:** `nmos_tls` — `nmos-testing` BCP-003-01 su savais sertifikatais: 0 FAIL; neigiamas: be TLS parinkčių HTTPS
prievadas neatidarytas; su netinkamu sertifikatu sidecar'as baigiasi ≠ 0 su tekstu.

---

## 5. Eilė bazės atžvilgiu

1. **Dabar (bazės metu):** tik E0 pataisos bazės plane ir kode, kur P4 / P6 / P7 dar nebaigti, ir autoriaus sprendimai K-1…K-15.
2. **Po P6:** E1 (grupės; didžioji dalis be GPU) ir E8 dalis be priklausomybių.
3. **Po P5a + P6:** E2 (pirmas tikras rizikos mažinimas — ar Fabrics veikia mūsų aplinkoje).
4. **Po P5:** E3. **Po P7 + P4 (+ P8):** E4, E5.
5. **Kai yra geležis:** E6 (darbo stotis — soft-RoCE), E7 (lab).
6. **NMOS:** E8 po E1 ir K-9; E9 po E4; E10 po P7 (aliarmai — po P9); E11 paskutinis.

## 6. Ko čia patikrinti negalima

- Tikras RDMA (RoCEv2) ir jo našumas; QoS / PFC; 25 / 100 GbE talpa (E6 lab dalis, N4.3).
- GPUDirect RDMA: nėra nei upstream palaikymo, nei tinkamos geležies (E7, N5.1).
- Soft-RoCE nešiojamame kompiuteryje (WSL2 branduolys be `rxe`, F-25); darbo stotyje — tik funkciškai.
- HD 50 Hz tarp dviejų mašinų (1 GbE): tik 720p25; 1080 tarp mašinų — reikia ≥ 2,5 / 10 GbE.
- PTP tarp serverių (L3); ilgalaikis laikrodžių dreifas tarp tikrų serverių (WSL2 laikrodis — žr. `laikas-ir-sinchronizacija.md` §0).
- Dviejų GPU vienoje mašinoje atvejai (peer access); AWS EFA; Kubernetes (hostPath, Multus / SR-IOV).
- Tikri NMOS valdikliai ir registrai objekte; DNS-SD per VLAN; IS-10 su tikru autorizacijos serveriu.
- Šiame plane **niekas nevykdyta**: MXL su Fabrics čia nesukompiliuotas (F-27); `mxl-fabrics-demo` nepaleistas; nmos-cpp ir
  `nmos-testing` nepaleisti; jungties nutraukimo įrankiai WSL2 nepatikrinti.

## 7. Rizikos (bendros)

| # | Rizika | Poveikis | Mažinimas |
|---|---|---|---|
| R1 | Fabrics API v0 ir PR #653 / #654 / #746 serija keičiasi (F-03…F-07) | Perrašymas | Visi Fabrics kvietimai `src/net/fabrics_link`; MXL SHA prisegtas; testas, fiksuojantis F-04 |
| R2 | Fabrics su PR #653 medžiu WSL2 nesikompiliuoja arba TCP per loopback per lėtas | E2 stringa | E2 pirmas žingsnis — patikra prieš kodą; atsarginė vieta — darbo stotis |
| R3 | Dvi PCIe kopijos + TCP kopijos suvalgo GPU kelio naudą | Mažesnė talpa nei tikėtasi | Matuojama E3; host → host ir GPUDirect lieka kaip kraštiniai taškai palyginimui |
| R4 | Laikrodžiai (WSL2 žingsniai, δ tarp mašinų) | Klaidingi praradimai, vėlavimas | `GapClock` nepriklauso nuo δ; poslinkis matuojamas; preflight iš `laikas-ir-sinchronizacija.md` T0 |
| R5 | Flow ID ir `grouphint` keitimas paliečia visas funkcijas ir testus | Regresijos | E0 su suderinamumu atgal (be `--group` — seni ID) |
| R6 | Priėmimo knygos perkėlimas (K-14) po to, kai P4 jau parašytas | Dvigubas darbas | Nuspręsti prieš P4 įgyvendinimą |
| R7 | `cuda-linear` prieinamas tik rašytojo UID (F-16; išvada iš kodo) | Griežtoje izoliacijoje GPU flow'ų negalima dalinti be kopijos | Numatyta organizacinė izoliacija / konteineriai; host eksportas; radinys U5 upstream; testai `share_*` aptinka pasikeitimą |
| R8 | Rendezvous ir Fabrics be autentifikacijos | Svetimas siuntėjas medijos tinkle | `--allow`, atskiras VLAN, dokumentacija; TLS tik valdymo plokštumai |
| R9 | NMOS: sunki priklausomybė, jauna specifikacija, gamintojo `ext_` | Palaikymo kaina | Neprivalomas taikinys; `model` be nmos-cpp; failų kelias visada veikia |
| R10 | 1 → N: lėtas gavėjas stabdo kitus | Vieno gedimas plinta | Testas `bridge_fan_out`; jei reikia — atskiri initiator'iai |
| R11 | Per didelė apimtis vienam etapui po bazės | Nebaigti darbai | Etapai nepriklausomi; kiekvienas duoda patikrinamą rezultatą; E7 ir E10–E11 gali likti vėlesniam laikui |

## 8. Klausimai autoriui

Žr. [R] §9 (K-1…K-15). Blokuoja: **K-1** (E2), **K-14**, **K-15** (E0 — prieš P4 įgyvendinimą), **K-5** (E1), **K-9** (E8).
Neblokuoja pradžios: K-2, K-3, K-4, K-6, K-7, K-8, K-10, K-11, K-12, K-13.
