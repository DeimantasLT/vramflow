# Tikslus laikas ir sinchronizacija vramflow projekte

- **Data:** 2026-11-11 · **Būsena:** pasiūlymas (niekas neįdiegta, nekeista, WSL būsena neliesta — tik skaitymo komandos)
- **Susiję:** [`2026-11-10-vramflow-verify.md`](2026-11-10-vramflow-verify.md) ([V]), [`2026-11-10-gpu-media-funkcijos.md`](2026-11-10-gpu-media-funkcijos.md) ([F])
- **Reikalavimas:** sinchronizacija ir tikslus laikas — privaloma, „lygiai kaip CPU/host MXL visomis prasmėmis". MXL naudoja mašinos laiką; jei mašinos sinchronizuotos — jos gyvena tame pačiame laike.

Žymėjimas: **[patikrinta: šaltinis]** — perskaityta šaltinyje arba išmatuota šioje mašinoje 2026-11-10 23:50–23:58 EET (MXL: `/root/src/mxl-pr653`, SHA `6dddc91d`). **[prielaida]** — nepatikrinta, turi būti patvirtinta vykdant.

---

## 0. Svarbiausia išvada: šios mašinos laiko problema — ne TSC ir ne Hyper-V, o du laikrodžio valdytojai

Pradinė hipotezė („VM tsc eina 4,3 % per greitai, Hyper-V time sync kas ~31 s šoka atgal") **nepasitvirtino**. Išmatuota:

| # | Faktas | Įrodymas |
|---|---|---|
| M1 | Neapdorotas TSC yra tikslus: `/dev/ptp_hyperv` (host laikas) prieš `CLOCK_MONOTONIC_RAW` per 40 s: **−2,1 ppm** | [patikrinta: zondas `clockprobe.py`, 40 s] |
| M2 | `CLOCK_MONOTONIC` prieš `CLOCK_MONOTONIC_RAW`: **+44 569 ppm (+4,46 %)** — t. y. greitina ne geležis, o branduolio NTP korekcija | [patikrinta: tas pats zondas] |
| M3 | `adjtimex`: `tick` = 10414…10464 (norma 10000) ir **keičiasi kas kelias sekundes**, `freq` šokinėja −50…+16 ppm → kažkas aktyviai „slew'ina" laikrodį ~+4,2…4,6 % | [patikrinta: `adjtimex(modes=0)` × 7 per 40 s] |
| M4 | WSL pats paleidžia **`chronyd`** (mini-init, už distro PID vardų srities) su `refclock PHC /dev/ptp0 poll 3 dpoll -2 offset 0`, `makestep 1.0 3`, `leapsectz right/UTC`, `rtcsync` | [patikrinta: `microsoft/WSL` `src/linux/init/main.cpp:3731-3779` (master); WSL 2.5.10.0] |
| M5 | Ubuntu distro viduje veikia **`systemd-timesyncd`** → `ntp.ubuntu.com`, poll 32 s, paskutinis offset −1,95 s; jis žingsniuoja laikrodį, kai nuokrypis didelis | [patikrinta: `timedatectl timesync-status`, `systemctl is-active`] |
| M6 | Windows host laikrodis **0,89 s priekyje UTC**; `W32Time` tarnyba **nepaleista** (host laikrodis laisvai plaukia) | [patikrinta: `w32tm /stripchart /computer:time.windows.com` → −0,896 s; `w32tm /query /status` → 0x80070426] |
| M7 | `phc − realtime` pjūklas: +755 ms → −363 ms per ~25 s, tada šuolis −1,38 s atgal → +892 ms | [patikrinta: zondas] |
| M8 | Hyper-V `hv_utils` laiką žingsniuoja tik gavęs `ICTIMESYNCFLAG_SYNC` (boot/restore) arba — su `timesync_implicit=1` — kai svečias **atsilieka ≥ 5 s** | [patikrinta: `WSL2-Linux-Kernel` `drivers/hv/hv_util.c:356-375, 416-419` (linux-msft-wsl-6.6.y); cmdline `hv_utils.timesync_implicit=1`] |
| M9 | TAI−UTC = 37 s (`adjtimex.tai = 37`) | [patikrinta]; nustato WSL `chronyd` per `leapsectz right/UTC` [prielaida: mechanizmas] |

**Mechanizmas:** WSL `chronyd` tempia svečio laikrodį prie *host* laiko (UTC + 0,89 s), `systemd-timesyncd` kas 32 s žingsniu grąžina prie *NTP* laiko (UTC). Tarp žingsnių chrony vejasi ~1 s paklaidą leistinu greičiu (numatytas `maxslewrate` 83 333 ppm [patikrinta: chrony.conf 4.5 doc]) — iš čia „+4,3 %" ir „−1,3 s kas 31 s". Tai dviejų valdytojų kova, kurią Ubuntu WSL dokumentacija aprašo tiesiogiai: NTP klientas distro viduje „gali sukelti clock skew, jei naudoja kitus serverius nei Windows", siūloma `systemctl disable systemd-timesyncd.service` [patikrinta: ubuntu.com/wsl/docs/stable/explanation/time-sync].

**Pasekmės ankstesniems matavimams:** (a) `ClockWatch` lygina TAI su `steady_clock` (= `CLOCK_MONOTONIC`, kuris slew'inamas kartu su TAI) — todėl greičio paklaidos nematė, matė tik žingsnius [patikrinta: `src/func/flows.cpp:47-55`]; (b) `ffmpeg -re` „4,3 % greitesnis" — tas pats slew'as, ne atskiras reiškinys; (c) talpos (capacity) skaičiai, gauti su 4,4 % pagreitintu laikrodžiu, yra **pesimistiniai ~4,4 %** (realiu laiku buvo generuojama 52,2 lauko/s) [prielaida: išvestinė iš M2].

---

## 1. Kaip MXL apibrėžia laiką ir ką reiškia „teisingai"

### 1.1 Apibrėžimai (kodas ir dokumentai)

| Tema | Kas parašyta | Šaltinis |
|---|---|---|
| Epocha | Ring buferio indeksas atitinka laiko žymą nuo SMPTE ST 2059-1 (PTP) epochos; MXL **nereikalauja** PTP šaltinio, tik naudoja epochą ir TAI | [patikrinta: `docs/Timing.md:6`] |
| Laikrodis | `mxlGetTime()` = `clock_gettime(CLOCK_TAI)`; 37 s konstanta naudojama **tik** jei `CLOCK_TAI` neapibrėžtas kompiliavimo metu; vykdymo metu TAI poslinkis netikrinamas | [patikrinta: `lib/src/time.cpp:11-14`; `ClockHelpers.hpp:22-24, 33-47`; issue #609 — atvira, TSC sprendimas: tik įspėjimas `mxl-info`] |
| Indeksas | `index = (t·num + 0,5e9·den) / (1e9·den)` — **apvalinimas iki artimiausio**, ne `floor` | [patikrinta: `IndexConversion.hpp:12-23`] |
| Atgal | `T(index) = (index·den·1e9 + num/2) / num` | [patikrinta: `IndexConversion.hpp:25-34`] |
| Miegas | `mxlSleepForNs` — santykinis `clock_nanosleep(CLOCK_TAI, 0)`; `mxlSleepUntil` — `TIMER_ABSTIME` ant `CLOCK_TAI` | [patikrinta: `time.cpp:65-75`; `Thread.cpp:42, 60`] |
| Skaitytojo timeout | Deadline = `CLOCK_REALTIME + timeoutNs`; laukimas — santykinis `FUTEX_WAIT`, po kiekvieno pabudimo likutis perskaičiuojamas iš `CLOCK_REALTIME` | [patikrinta: `lib/src/flow.cpp:19-22`; `Sync.cpp:33, 68-100, 118`] |
| Rašytojas | Commit įrašo `lastWriteTime` = TAI; indeksą parenka **aplikacija** | [patikrinta: `PosixDiscreteFlowWriter.cpp:222`; `Timing.md:22`] |
| Sync grupė | `WaitForDataAt(ts)`: kiekvienam flow `expectedIndex = timestampToIndex(rate, ts)`, laukia, kol `headIndex ≥ expectedIndex`; TAI naudoja tik eilės euristikai | [patikrinta: `FlowSynchronizationGroup.cpp:77-132`; `Architecture.md:278-279`] |
| Replikacija | „MXL Flow replication between hosts will preserve the indexing of the grains" | [patikrinta: `Timing.md:83`] |

### 1.2 Ko MXL reikalauja iš laikrodžio

- Vienas host'as: „any stable time sources such as NTP or SMPTE ST 2059 is adequate" [patikrinta: `Timing.md:12`].
- Keli host'ai: laiko šaltiniai „does not slip against each other over time" (vienodas dažnis); jitter'is priimtinas [patikrinta: `Timing.md:14`]. Debesyse — „sub millisecond accuracy" [`Timing.md:16`].
- Rašytojas privalo pats „pace itself" pagal host laikrodį, rašyti ne daugiau 50 % ring'o vienu pliūpsniu, o jei išorinis šaltinis neturi bendro dažnio — įgyvendinti re-sync (drop/repeat/resample) [patikrinta: `Timing.md:22, 89-90`].
- Prižiūrėtojų pozicija: MXL „assumes a realtime environment with exact and appropriately synchronized clocks"; TAI poslinkis — host'o konfigūracijos reikalas [patikrinta: issue #609, K. Hoffmann 2026-07-23].
- **Skaitinių ribų (ppm, µs) MXL dokumentuose nėra** [patikrinta: `grep` per `docs/`]. Galutinį modelį rengia JT-DMF End-to-End Synchronisation WG [`Timing.md:8`].

### 1.3 Mūsų „teisingai" (išvestinės ribos)

| Lygis | Reikalavimas | Kodėl |
|---|---|---|
| L0 visur | TAI−UTC = 37 s; **0 laikrodžio žingsnių** darbo metu; vienas laikrodžio valdytojas | 37 s klaida = 1850 laukų poslinkis; žingsnis = sustojimas arba skylė |
| L1 vienas host'as | dažnio paklaida prieš etaloną ≤ 50 ppm trumpalaikė, ≤ 5 ppm vidutinė [prielaida: mūsų riba] | 50 ppm = 1 laukas per 400 s prieš išorinį 50 Hz šaltinį — re-sync logika turi veikti, bet retai |
| L2 keli host'ai, NTP | tarpusavio poslinkis ≤ 1 ms, dažnis užrakintas | ≤ 5 % lauko periodo; pakanka video, kai indeksai ateina iš vieno rašytojo (Fabrics) |
| L3 keli host'ai, PTP | sisteminis laikrodis prie GM ≤ 10 µs (PHC ≤ 1 µs) [prielaida: ST 2059-2 tikslas 1 µs — PDF šioje sesijoje nenuskaitytas] | pusė 48 kHz mėginio = 10,4 µs: tik tada du host'ai, žymintys `index(T_now)`, sutaria mėginio tikslumu |

### 1.4 Kas lūžta, kai laikrodis šoka ar dreifuoja

| Įvykis | Rašytojas | Skaitytojas | Sync grupė | Fabrics |
|---|---|---|---|---|
| Žingsnis atgal Δ | `index(T_now)` grįžta atgal → `OpenGrain` su indeksu ≤ head atmetamas; TAI ritmu einantis rašytojas **stovi Δ** (mūsų išmatuota ~1,3 s) | laukia ilgiau; MXL timeout'ai gali **pailgėti iki Δ** (deadline `CLOCK_REALTIME`) [patikrinta: `Sync.cpp:80-90`] | `WaitForDataAt` lygina tik indeksus → tik timeout'o iškraipymas | initiator'ius ima `mxlGetCurrentIndex` → `TOO_EARLY` kilpa Δ [patikrinta: `demo.cpp:414-428`] |
| Žingsnis pirmyn Δ | praleidžia Δ·rate indeksų (skylė); jei Δ > ring'o istorija — visi skaitytojai `TOO_LATE` | timeout'ai suveikia per anksti; reikia re-sync į head | tas pats | target'e skylė |
| Slew'as (dažnio paklaida ε) | indeksų erdvė vientisa, bet realus kadrų dažnis = 50·(1+ε); išoriniai šaltiniai/imtuvai (UDP, SDI, ekranas) persipildo arba badauja | nemato (tas pats laikrodis) | nemato | tarp host'ų — latencijos dreifas iki ring'o ribos |
| TAI poslinkis 0 vietoje 37 | indeksai 37 s „praeityje" | su teisingu host'u nesusitinka niekada | — | pastovus `TOO_LATE`/`TOO_EARLY` |

### 1.5 MXL dokumentacijos spragos (verta upstream doc PR)

1. `Timing.md:62-66` rodo `GrainIndex = Timestamp / GrainDurationNs` (floor), kodas apvalina iki artimiausio (`IndexConversion.hpp:16-17`); `mxl-info` komentaras teigia, kad grain'as apima `[T(i), T(i+1))` (`tools/mxl-info/main.cpp:122`) — trys skirtingi apibrėžimai, skirtumas ½ grain'o.
2. TAI poslinkio prielaida nedokumentuota (issue #609): nėra nei reikalavimo, nei „kaip nustatyti" (chrony `leapsectz`/`leapseclist`, `phc2sys`).
3. Nėra žodžio apie laikrodžio žingsnius; reikėtų: „laikrodis po starto tik slew'inamas"; skaitytojų timeout'ai skaičiuojami per `CLOCK_REALTIME` (turėtų būti monotoninis).
4. Nėra skaitinių reikalavimų (ppm, poslinkis tarp host'ų) — tik „stable" ir „does not slip".
5. `Fabrics*.md` visai nemini laiko: ar target'ui reikia sinchronizuoto laikrodžio, kas nutinka skaitytojui su `mxlGetCurrentIndex` esant poslinkiui [patikrinta: `grep -i "tai\|ptp\|clock"` — 0 atitikmenų apie laiką].
6. Continuous flow: `Architecture.md:191, 255` sako „index is inclusive" (mėginiai `head−255…head`), kodas ima pozicijas `[index−count, index)` (`PosixContinuousFlowReader.cpp:135-136`, `PosixContinuousFlowWriter.cpp:101-102`) → **1 mėginio dviprasmybė**, kuriam laiko momentui priklauso mėginys.
7. Sync grupė: diskrečiam flow „duomenys ties T" = grain'as, apimantis laiką *po* T; continuous — mėginiai *iki* T. Asimetrija neaprašyta (žr. §3 T5).
8. VM gairės: vienas laikrodžio valdytojas; WSL jau turi savo `chronyd`.

---

## 2. Teisingas laiko nustatymas

### 2.1 (a) Ši mašina — WSL2 / Hyper-V

**Variantai (reitinguoti):**

| # | Variantas | Vertinimas |
|---|---|---|
| **A1** | **Išjungti `systemd-timesyncd` Ubuntu distro; palikti WSL `chronyd` → `/dev/ptp0` (hyperv PHC)** | **Rekomenduojama.** Tai Microsoft numatytas kelias (M4) ir Ubuntu dokumentuota rekomendacija. Lieka vienas valdytojas, tik slew. Svečio tikslumas = host tikslumas. |
| **A2** | + sutvarkyti Windows host laiką (`W32Time` nuolat veikiantis, artimas NTP šaltinis) | **Reikalinga kartu su A1**, kitaip svečias tiksliai seks netikslų host'ą (dabar +0,89 s, laisvai plaukiantis). |
| A3 | Savas `chrony` Ubuntu distro su `refclock PHC /dev/ptp_hyperv` (Azure dokumentuota eilutė) kaip valdytojas | **Atmesta šiai mašinai:** WSL `chronyd` išjungti nėra dokumentuoto būdo → būtų du `chronyd` ant vieno branduolio. Azure VM (ne WSL) — taip, tai teisingas kelias [patikrinta: learn.microsoft.com/azure/virtual-machines/linux/time-sync]. |
| A4 | `chrony` Ubuntu distro **tik stebėjimui** (`-x`, laikrodžio nevaldo) | Naudinga verifikacijai: `chronyc tracking/sources` rodo poslinkį iki PHC ir NTP. Neprivaloma — tą patį daro mūsų zondas. |
| A5 | Clocksource → `hyperv_clocksource_tsc_page` | **Atmesta:** TSC tikslus −2,1 ppm (M1); keisti nėra ko. |
| A6 | Išjungti Hyper-V žingsnius (`hv_utils.timesync_implicit=0` per `.wslconfig` `kernelCommandLine`) | **Nereikia ir neveikia patikimai** (pranešama, kad nuo WSL 2.4.12 ignoruojama; WSL issue #12765) [patikrinta: Ubuntu WSL doc]. Žingsnis įvyksta tik atsilikus ≥ 5 s (po host miego) — testų metu tai draudžiama būsena bet kuriuo atveju. |

**Siūlomos komandos (NEVYKDYTA; vykdo užsakovas, po GPU benchmark'o pabaigos):**

```bash
# A1 — WSL Ubuntu-24.04, root
systemctl disable --now systemd-timesyncd.service
timedatectl show -p NTP -p NTPSynchronized          # tikimasi NTP=no
# grąžinimas: systemctl enable --now systemd-timesyncd.service
```

```powershell
# A2 — Windows, administratoriaus PowerShell (sistemos nustatymas — reikia patvirtinimo)
Set-Service w32time -StartupType Automatic
Start-Service w32time
w32tm /config /manualpeerlist:"<NTP_SERVERIS>,0x8" /syncfromflags:manual /update
w32tm /resync /force      # vienkartinis ~0,9 s žingsnis: daryti PRIEŠ testus, ne jų metu
w32tm /query /status /verbose
# grąžinimas: Set-Service w32time -StartupType Manual; Stop-Service w32time
```

Papildomai (neprivaloma, „high accuracy" profilis iki ~1 ms): `MinPollInterval=6`, `MaxPollInterval=6`, `UpdateInterval=100`, `FrequencyCorrectRate=2` (`HKLM\SYSTEM\CurrentControlSet\Services\W32Time\Config`), `SpecialPollInterval=64` (`…\TimeProviders\NtpClient`), tada `w32tm /config /update` [patikrinta: learn.microsoft.com „Configuring systems for high accuracy"].

**Po pakeitimo:** svečią pervesti į švarią būseną `wsl --shutdown` (WSL `chronyd` startuoja iš naujo, `makestep 1.0 3` leidžia pradinį žingsnį) — arba palaukti, kol chrony suslew'ins (~1 s paklaida per ≤ ~15 s) [prielaida].

**Tikėtinas rezultatas** [prielaida, tikrinama §3 T1]: `tick` = 10000, `|freq|` < 50 ppm, `phc − realtime` |poslinkis| < 1 ms be pjūklo, 0 žingsnių; `ffmpeg -re` ir TAI ritmas sutampa.

**Kiti klausimai:**
- *TAI poslinkis:* dabar 37 (M9). Kadangi tai priklauso nuo WSL vidinio `chronyd`, mūsų įrankiai privalo patys tikrinti `adjtimex().tai == 37` starte (§3.3). Atsarginis rankinis nustatymas — `adjtimex` su `ADJ_TAI` (kaip `linuxptp` `clockadj.c:211-220`).
- *systemd WSL'e:* įjungtas (`/etc/wsl.conf` `[boot] systemd=true`, PID 1 = systemd) [patikrinta]; `systemctl disable` išlieka per perkrovimus. `.wslconfig` keisti nereikia (`vmIdleTimeout=-1` jau yra).
- *Išliekančios ribos:* (1) host miegas/hibernacija → svečias atsilieka → Hyper-V žingsnis pirmyn; (2) bet koks host laiko žingsnis (rankinis `w32tm /resync`) → svečio `chronyd` **slew'ins iki 8,3 %** kelias–keliolika sekundžių, nes `makestep` galioja tik pirmiems 3 atnaujinimams, o WSL init `chrony.conf` perrašo kiekvieno VM starto metu (konfigūruoti negalima) [patikrinta: `main.cpp:3766-3774`]. Todėl: testų metu — jokio miego ir jokių rankinių resync; monitorius (T1) tai pagauna.
- *Ko WSL neduos:* PTP lygio (µs) tikslumo prieš išorinį GM — nėra PTP NIC. Nešiojamas kompiuteris tinka L0/L1 (vieno host'o) verifikacijai; L2/L3 — tik serveriuose.

### 2.2 (b) Fiziniai serveriai

**P1 — yra PTP grandmaster ir NIC su PHC (tikslinė produkcinė schema, L3):**

`/etc/linuxptp/ptp4l-st2059.conf` (linuxptp neturi paruošto SMPTE profilio [patikrinta: `linuxptp/configs/` sąrašas]; reikšmės sutampa su NVIDIA „Media" profiliu [patikrinta: DOCA Firefly doc v2.2.0]):

```ini
[global]
domainNumber            127
clientOnly              1          # linuxptp < 4.0: slaveOnly 1
priority1               128
priority2               127
logAnnounceInterval     -2
announceReceiptTimeout  3
logSyncInterval         -3
logMinDelayReqInterval  -3
delay_mechanism         E2E
network_transport       UDPv4
hybrid_e2e              1
dscp_event              46
dscp_general            46
time_stamping           hardware
tx_timestamp_timeout    30
[<NIC>]
```

```bash
ptp4l -f /etc/linuxptp/ptp4l-st2059.conf -i <NIC>            # PHC ← GM (PTP/TAI skalė)
phc2sys -s <NIC> -c CLOCK_REALTIME -w                        # sisteminis laikrodis ← PHC
```

- `domainNumber` ir intervalai **turi sutapti su studijos PTP GM** (ST 2110 tinklo nustatymais) — suderinti su tinklo savininkais [prielaida: 127/−2/−3 yra ST 2059-2 numatytosios].
- `phc2sys -w` (be `-O`) UTC poslinkį ima iš `ptp4l` (`currentUtcOffset`) ir **nustato branduolio TAI poslinkį** [patikrinta: phc2sys doc; `phc2sys.c:1138-1141` → `sysclk_set_tai_offset`]. Su `-O -37` TAI poslinkis nenustatomas → `CLOCK_TAI` = UTC → MXL klaida 37 s. Patikra: `pmc -u -b 0 'GET TIME_PROPERTIES_DATA_SET'` (`currentUtcOffset 37`, `currentUtcOffsetValid 1`, `ptpTimescale 1`).
- PHC laiko TAI, `CLOCK_REALTIME` — UTC, `CLOCK_TAI` = REALTIME + 37 = PHC. MXL skaito `CLOCK_TAI` → sutampa su ST 2110 RTP laiko žymų epocha.
- **Privaloma išjungti kitus valdytojus:** `systemd-timesyncd`, `chronyd` (arba palikti `chronyd -x` stebėjimui) — NVIDIA rašo tą patį: „NTP must be disabled" [patikrinta: DOCA Firefly doc].
- `phc2sys -S 0.0` nenaudoti aklai: numatytai po starto nebežingsniuoja (`step_threshold` 0), startinis žingsnis > 20 µs leidžiamas [patikrinta: phc2sys doc] — MXL procesus startuoti tik po `phc2sys` konvergencijos (systemd `After=` + patikra).

**P2 — chrony kaip vienintelis valdytojas, PHC kaip šaltinis** (kai norima NTP atsargos ir `chronyc` stebėjimo): `ptp4l` kaip P1, vietoje `phc2sys`:

```conf
refclock PHC /dev/ptpN tai poll 0 dpoll -2 stratum 1 prefer
leapsectz right/UTC                 # Ubuntu 24.04: reikia paketo tzdata-legacy (kataloge right/ nėra [patikrinta WSL distro])
# chrony ≥ 4.6: leapseclist /usr/share/zoneinfo/leap-seconds.list   [patikrinta: chrony NEWS 4.6]
makestep 0.1 3
rtcsync
```

`tai` opcija reikalauja `leapsectz`; ta pati direktyva nustato branduolio TAI poslinkį (tik `leapsecmode system`) [patikrinta: chrony.conf 4.5 doc]. Ubuntu 24.04 turi chrony 4.5 (be `leapseclist`) [patikrinta: `apt-cache policy`].

**P3 — nėra PTP (chrony → vietinis NTP serveris), L1/L2:** `server <ntp-serveris> iburst minpoll 4 maxpoll 6`, `leapsectz right/UTC` (arba `leapseclist`), `makestep 1.0 3`. Windows DC tikslumas — ms eilės [prielaida]; vienam host'ui pakanka pagal `Timing.md:12`. Patikra: `chronyc tracking` („System time", „Frequency", „Leap status"), TAI = 37.

**Tikslumas transliacijai:** PTP su aparatinėmis žymomis — „sub-microsecond" [patikrinta: DOCA Firefly doc — bendras teiginys]; tikslinė ST 2059-2 riba 1 µs tarp įrenginių [prielaida]. `phc2sys` prideda dešimtis–šimtus ns iki kelių µs priklausomai nuo apkrovos [prielaida] → sisteminis laikrodis ≤ 10 µs yra realus PASS slenkstis.

### 2.3 (c) Keli host'ai su MXL Fabrics / RDMA

| Klausimas | Atsakymas | Šaltinis |
|---|---|---|
| Ar replikacija išsaugo indeksus? | **Taip.** Initiator'ius RMA įrašu perkelia visą grain'ą su antrašte; „immediate data" neša tik ring'o slotą (16 b) ir slice; target'as tikrąjį indeksą perskaito iš perkeltos grain'o antraštės. Continuous — `headIndex` keliauja antraštėje. | [patikrinta: `Timing.md:83`; `ImmData.cpp:16-20`; `ProtocolIngressRMA.cpp:66-75, 156-161`; `Fabrics.md:346`] |
| Ar pats Fabrics naudoja laiką? | **Ne.** `lib/fabrics` naudoja tik `steady_clock` timeout'ams; jokio `CLOCK_TAI`. | [patikrinta: `grep` per `lib/fabrics`] |
| Kas parenka indeksą? | Šaltinio host'o rašytojas (jo laikrodis). Demo initiator'ius skaito nuo `mxlGetCurrentIndex` (šaltinio laikrodis), demo target'as commit'ina gautu indeksu. | [patikrinta: `demo.cpp:414-428, 831-841`; `FabricsDeveloperGuide.md:293-300`] |
| Ar target'ui reikia sinchronizuoto laiko? | **Duomenų vientisumui — ne, skaitytojams — taip.** Skaitytojas target'e, naudojantis `mxlGetCurrentIndex`, mato latenciją `L + δ` (L — pernešimo vėlinimas, δ = target − source laikrodis). | išvestinė iš kodo |
| δ > 0 (target priekyje) | kiekvienas skaitymas laukia δ ilgiau; jei δ + L > timeout → `TOO_EARLY`/timeout kilpa | [patikrinta: `PosixDiscreteFlowReader.cpp:229-288`] |
| δ < 0 (target atsilieka) | `mxl-info` rodo latenciją „ateityje" (raudona); skaitoma senesnė istorija; kai \|δ\| > ring'o istorija → `TOO_LATE` | [patikrinta: `tools/mxl-info/main.cpp:117-153`] |
| Dažnio nesutapimas | latencija dreifuoja iki ring'o ribos — tai tiesiogiai draudžia `Timing.md:14` | [patikrinta] |
| Ką sako Fabrics dokumentai? | **Nieko apie laiką** (spraga §1.5-5). | [patikrinta] |
| RDMA/libfabric reikalavimai laikui | Nėra; RoCE NIC (ConnectX/BlueField) kartu yra PTP PHC → tą patį prievadą naudoti `ptp4l` (P1). BlueField: `mlxconfig … set REAL_TIME_CLOCK_ENABLE=1`. | [patikrinta: DOCA Firefly doc]; ConnectX atitikmuo [prielaida] |

**Taisyklė klasteriui:** visi MXL host'ai (source ir target) — tas pats GM, P1 schema, L3; skaitytojų timeout'ai ≥ L + 2·max\|δ\|; ring'o istorija ≥ 4·(L + max\|δ\|) [prielaida: mūsų atsarga].

---

## 3. Verifikacijos planas (čia ir serveriuose)

Visi testai rašo įrodymus į `evidence/`; slenksčiai — į `config/thresholds.json` (naujas blokas `"time"`).

| ID | Testas | Kaip | PASS (WSL po A1+A2) | PASS (serveris P1) |
|---|---|---|---|---|
| **T0** | Preflight (kiekvieno paleidimo pradžioje) | `adjtimex`: `tai`, `tick`, `freq`, `status`; `systemctl is-active systemd-timesyncd`; clocksource | `tai==37`; `tick==10000`; `\|freq\|<100 ppm`; timesyncd neaktyvus | + `ptp4l` `SLAVE`, `currentUtcOffsetValid 1`; `chronyd` nevaldo |
| **T1** | Laikrodžio kokybės monitorius (fone visų testų metu, 4 Hz) | `TAI − MONOTONIC_RAW`, `MONOTONIC − MONOTONIC_RAW`, `PHC − REALTIME`, `adjtimex` | žingsnių (> 1 ms tarp imčių) = **0**; dažnis prieš RAW \|ε\| ≤ 50 ppm (p99), ≤ 5 ppm vid.; \|PHC−REALTIME\| ≤ 1 ms | žingsnių 0; \|ε\| ≤ 1 ppm; `phc2sys` offset rms ≤ 1 µs, max ≤ 10 µs; `ptp4l` offset rms ≤ 100 ns [prielaida] |
| **T2** | Rašytojo laiko tikslumas | `gen`: `wake_err = t_wake − T(i)` ir `commit − T(i)` pasiskirstymas, 10 min | `wake_err` p50 ≤ 0,2 ms, p99 ≤ 2 ms; commit p99 < 20 ms (1 laukas); 0 praleistų indeksų | p99 ≤ 0,5 ms (RT branduolys/izoliuoti CPU) |
| **T3** | Ilgalaikis ritmas | per 10 min: `(last_index − first_index) / (ΔMONOTONIC_RAW · 50)` | 1 ± 50·10⁻⁶ | 1 ± 5·10⁻⁶ |
| **T4** | Tarp-procesinis sutarimas | A rašo grain'us su `origin_tai_ns` ([V] antraštė), B skaito: `\|T(index) − origin_tai\|`, B `mxlGetCurrentIndex ∈ {index, index+1}`; papildomai 2 procesai, žadinami vienu `eventfd`, lygina `index(T_now)` | 100 % sutampa, išskyrus imtis ≤ 50 µs nuo ribos | tas pats; tarp host'ų: nesutapimų dalis ≤ δ/period |
| **T5** | Daugelio flow sync grupė | N GPU video (cuda-linear) + N host audio; kas lauką `WaitForDataAt(T(i))` video grupei ir `WaitForDataAt(T(i+1))` audio grupei; tada `GetGrainSlot(i, 0)` ir `GetSamplesNonBlocking((i+1)·960, 960)` | visi `OK`; `info.index == i`; barkodas == `i mod 2²⁴` visuose N; audio turinys = generatoriaus funkcija mėginiams `[i·960, (i+1)·960)`; 0 neatitikimų per 10 min, N = 1, 8, 32 | tas pats |
| **T6a** | A/V mėginio tikslumu (MXL lygis) | kiekvienai sekundei k: video laukas `i0 = 50k` (blyksnis + barkodas) → `s0 = i0·960`; audio langas `[s0−2400, s0+2400)`; lyginama su deterministiniu etalonu `e(s) = amp(s)·sin(2π·1000·(s mod 48000)/48000)`; ieškomas poslinkis `d ∈ [−48, 48]`, minimizuojantis Σ\|x[s+d] − e(s)\| | **d = 0** visoms sekundėms, liekana ≤ 1e-6 (t. y. 0 mėginių, griežčiau nei „≤ 1") | tas pats |
| **T6b** | A/V po koduoti/dekoduoti | `enc` → TS → `ffmpeg -copyts` dekodavimas: blyksnio kadro PTS ir pyptelėjimo pradžia (kryžminė koreliacija su 10 ms etalonu, sub-mėginio interpoliacija) | \|PTS_a − PTS_v\| ≤ **1 ms** (dabar [F] §7 riba 5 ms — griežtinti); `PTS_v mod 90000 == 0` blyksnio kadre (PTS pririštas prie TAI sekundės) | tas pats |
| **T7** | Fabrics (tik serveriuose) | source `gen` → initiator → target; target'e `verify`: indeksai ir barkodai identiški source; `mxl-info` latencija | — | 0 indeksų neatitikimų; latencija pastovi (dreifas ≤ 1 grain/val.), niekada „ateityje" |
| **T8** | Tyčinis laikrodžio žingsnis (atskirai, su leidimu, tik bandomojoje aplinkoje) | ±2 s žingsnis veikiant `gen`+`logo`+`verify` | nėra crash; `clock_steps == 1`; nė vienas indeksas neįrašytas du kartus; indeksai monotoniški; atsigavimas ≤ Δ + ring'o trukmė; visi praleisti laukai priskirti `clock_skips`, ne `late_skips` | tas pats; **produkcijoje žingsnių būti negali** |

Pastabos:
- T5 asimetrija [prielaida, patvirtinama pačiu T5]: `WaitForDataAt(T(i))` video flow laukia grain'o `i` (laikas `[T(i), T(i+1))`), o audio flow — tik `headIndex ≥ i·960` (mėginiai *iki* `T(i)`). Viena grupė su vienu `T` neduoda lauko `i` garso; todėl dvi grupės arba vienas laukimas ties `T(i+1)` (+20 ms latencijos).
- T6a mėginio konvencija: pozicija `p` ↔ TAI `p/48000` s (pusatviras `[index−count, index)`, kaip kode). Tai mūsų pasirinkimas, kol upstream neišspręs §1.5-6.
- T1 zondo prototipas (tik skaito laikrodžius): laikinas prototipas — perkelti į `scripts/wsl/80-clock-probe.py`.

### 3.3 Kur turi keistis mūsų kodas

| Vieta | Pakeitimas |
|---|---|
| `src/func/flows.cpp:47-55` `ClockWatch::stepped` | Lyginti su `CLOCK_MONOTONIC_RAW`, ne `steady_clock` (kuris slew'inamas kartu su TAI ir slepia dažnio paklaidą). Pridėti dažnio įvertį (ppm per slenkantį 5 s langą) → `clock_rate_ppm_max`; žingsnio slenkstis 1 ms. |
| `src/func/flows.cpp:57-67` `wait_until` | Paskutinį miegą daryti absoliutų (`mxlSleepUntil` / `clock_nanosleep(CLOCK_TAI, TIMER_ABSTIME)`) — mažesnis `wake_err`; 50 ms gabalus palikti `stop` reakcijai. „Šuolio" euristika `target − now > 40 ms` — pakeisti `ClockWatch` rezultatu. |
| naujas `clock_preflight()` (`src/common/clock.hpp`, kviečiamas iš `main_func.cpp` ir `main.cpp`) | `adjtimex`: `tai != 37` → exit 2 (libmxl netikrina, #609); `\|tick − 10000\| > 1` arba `\|freq\| > 100 ppm` → exit 2, nebent `--allow-bad-clock`; rezultatas į suvestinę. |
| `src/mxlio/common_io.cpp:171-172, 233-234` | proc CSV papildyti: `mono_raw_ns`, `tick`, `freq_ppm`, `adj_status`, `maxerror_us` (adjtimex jau kviečiamas). |
| `src/func/gen.cpp:48-50, 89` | pridėti `wake_err` histogramą su ženklu (`t_wake − T(i)`); `commit_lateness` palikti. |
| `src/func/gen.cpp:78-82`, `src/func/flows.hpp:94-108` | komentare užfiksuoti mėginio konvenciją (§3 T6a); nekeisti logikos. |
| `src/func/enc.cpp:193-194, 209, 229-237` | PTS = TAI indeksas (jau taip). Pridėti: PTS monotoniškumo skaitiklį (`pts_backward`), ir `flash_pts_mod_90000` patikrą T6b; laikrodžio žingsnio atveju — `discontinuity_indicator` arba srauto restartas [prielaida: sprendimas po T8]. |
| `src/func/dec.cpp:224-233` | UDP → MXL šliuzas: palikti re-sync (drop/repeat) pagal `Timing.md:22`, bet skaičiuoti atskirai `resync_drop`/`resync_repeat`; po A1 `ffmpeg -re` perviršio neturi likti — tai T3 dalis. |
| `config/thresholds.json:18` `max_clock_steps: 3` | sinchronizacijos testams **0**; 3 palikti tik talpos pakopoms kaip „aplinkos triukšmą", kol A1 neįdiegta. Naujas blokas `"time"` su §3 slenksčiais. |
| [F] §7 A/V riba 5 ms | → 1 ms (T6b) ir 0 mėginių (T6a). |

### 3.4 PTS modelis (`mxlv_timing`, planas §P1)

Ingest funkcijos (P3) įėjimo laiką verčia į MXL laiką viena biblioteka `src/timing/` (be MXL, libav ir GPU; testai
`tests/unit/test_source_clock.cpp`, `test_timing_index.cpp`):

- **Vienas poslinkis šaltiniui** (`SourceClock`): `TAI = ns(PTS išvyniotas) + off`, bendras visiems šaltinio flow'ams —
  video ir garsas su tuo pačiu PTS gauna tą patį TAI (lūpų sinchronas pagal konstrukciją). Žymima demux metu
  (`observe()` kiekvienam paketui), todėl vėlesnis poslinkio pokytis neliečia jau kelyje esančių kadrų.
- **Fiksavimas** ties pirmu video paketu: `off = a + L − ns(DTS)`, po to kvantuojama į kadrų tinklelį (`align_rate`) —
  kiekvienas kadras krinta tiksliai ant grain'o laiko, 90 kHz apvalinimas (±1 tikas) indeksų nekeičia.
- **33 bitų wrap** išvyniojamas kiekvienam srautui; PTS — DTS atžvilgiu (PTS ir DTS gali būti skirtingose wrap pusėse).
  Naujo srauto (ar srauto po nutrūkimo) pirmas paketas pririšamas prie paskutinio šaltinio paketo — visi srautai
  išvynioti vienodai.
- **Nutrūkimas** (DTS šuolis > 1 s pirmyn arba bet koks atgal): perfiksuojama ties kitu video paketu; jei atvykimas
  tolydus (`|tęsinys − naujas| ≤ relock_slack`), išėjimo laikas tęsiasi be šuolio, kitaip — naujas fiksavimas.
- **Dreifas (K6):** atsarga `m = dts_tai − a`, 10 s langų minimumai; 3 langai iš eilės už `L ± 200 ms` → poslinkis ±1
  kadras (`Event::Slipped`, interlaced lieka laukų poromis, nes tinklelis kadrų). Įvertis `ClockStats::drift_ppm`
  (mažiausių kvadratų nuolydis per 30 langų) — juo P3 nustato `swr_set_compensation` garsui. Pirmi ~200 ms dreifo
  sugeriami be slydimo (200 ppm per 1 val. → 13 slydimų, ne 18).
- **Laikrodžio šuolis** (`ClockWatch` → `on_clock_step(Δ)`): `off += round(Δ / Tf) · Tf`.
- **Garsas** (`AudioCursor`): mėginio indeksas `round(TAI · 48000 / 10⁹)`; ≤ 1 ms nuokrypis tęsiamas be plyšio,
  didesnis — spraga arba persidengimas (mėginiais).

---

## 4. Rizikos ir darbai natyviuose Linux serveriuose

| # | Rizika | Poveikis | Priemonė |
|---|---|---|---|
| K1 | Du laikrodžio valdytojai (timesyncd + chrony; chrony + phc2sys; WSL chronyd + bet kas) | pjūklas, žingsniai — kaip čia išmatuota | T0 preflight; viena tarnyba per host'ą; kitos `disable` arba `-x` |
| K2 | TAI poslinkis nenustatytas (šviežias Ubuntu be `leapsectz`; `phc2sys -O`) | 37 s indeksų klaida, tylus | `clock_preflight` exit 2; P1/P2/P3 konfigūracija |
| K3 | Host laiko žingsnis / miegas (nešiojamas) | 8 % slew arba žingsnis svečiui | testų metu `powercfg` be miego [prielaida], jokio `w32tm /resync`; T1 pagauna |
| K4 | Ankstesni talpos rezultatai gauti su +4,4 % laikrodžiu | N\* pesimistinis; `late_skips` dalis galėjo būti laikrodžio | pakartoti talpos pakopas po A1+A2 |
| K5 | WSL `chrony.conf` nevaldomas, WSL atnaujinimas gali pakeisti elgseną | regresija | T0/T1 kiekviename paleidime; WSL versija į evidence |
| K6 | GM skelbia neteisingą `currentUtcOffset` arba `Valid=0` | TAI klaida visame klasteryje | `pmc` patikra diegiant; aliarmas |
| K7 | PTP paketai be QoS / per ne-PTP komutatorius | µs → šimtai µs | boundary/transparent clock komutatoriai, DSCP 46 |
| K8 | MXL timeout'ai ant `CLOCK_REALTIME` | žingsnio metu timeout'ai iškraipomi | 0 žingsnių politika; upstream PR (§1.5-3) |
| K9 | ST 2059-2 reikšmės ir 1 µs tikslas šiame dokumente — iš atminties | galimas neatitikimas studijos PTP GM | suderinti su ST 2110 tinklo savininkais prieš diegiant |

**Natyvaus Linux darbo stotis (chrony → NTP)** [prielaida: konfigūracija nematyta šioje sesijoje]: patikrinti `chronyc tracking`, `chronyc sources -v`, TAI = 37 (`leapsectz right/UTC` arba `leapseclist`), `systemctl is-active systemd-timesyncd` = inactive; paleisti T0–T6. Lygis L1/L2.

**Lab klasteris:** išsiaiškinti, ar yra PTP GM ir PHC NIC (`ethtool -T <if>`); jei taip — P1 visuose mazguose (K8s: `ptp4l`/`phc2sys` kaip host tarnybos arba DaemonSet su `hostNetwork` + `CAP_SYS_TIME`; pod'ai laiką paveldi iš host branduolio); jei ne — P3 į vieną bendrą NTP šaltinį ir tik L2. Tada T0–T8, T7 su RDMA.

---

## 6. Kas nepatikrinta

- A1+A2 poveikis (niekas nevykdyta) — visi „tikėtini rezultatai" yra [prielaida] iki T0/T1.
- Likę ~2 570 ppm tarp M2 (44 569) ir `tick` indėlio (~42 000–46 000, kinta) — nepaaiškinta tiksliai; priskiriama chrony `tick` kaitai matavimo metu [prielaida].
- ST 2059-2:2021 lentelės (PDF nepavyko nuskaityti); ConnectX `REAL_TIME_CLOCK_ENABLE`; `phc2sys` tikslumo skaičiai.
- T5 sync grupės elgsena su `cuda-linear` flow'ais ir continuous semantika — tik iš kodo skaitymo.
- natyvaus Linux darbo stoties ir klasterio faktinė laiko konfigūracija.
