# [P9] GPU multivieweris: operatorius gali stebėti daug VRAM flow'ų viename vaizde su aliarmais, be kopijų į host atmintį

> **Būsena: Patvirtinta (autorius delegavo sprendimus, 2026-11-11; žr. plano §6).** Planas: `docs/planai/2026-11-11-production-grade.md` §P9.
> Brief'as: `docs/reikalavimai/2026-11-11-production-grade.md` (L15, P9).

# Kontekstas (Context)
Kaip MCR operatorius arba inžinierius, derinantis grandinę,
Noriu viena komanda (`vramflow-func mv`) gauti sumažintų flow'ų tinklelį su aliarmais (juoda, sustingęs vaizdas, tyla,
nėra signalo) ir garso lygiais, sukomponuotą GPU atmintyje ir pateiktą kaip MXL flow,
Kad matyčiau visų srautų būklę realiu laiku, o stebėjimas neapkrautų sistemos host kopijomis (šiandien 13–20 MB kadrui).

---

# Priėmimo kriterijai (Acceptance Criteria)

[ ] 1. Tinklelis iš 16 įėjimų (teigiamas)
Duota (Given): 16 veikiančių 1080i50 video flow'ų GPU atmintyje
Kai (When): paleidžiu `vramflow-func mv --grid 4x4 --canvas 1920x1080 --format 1080p50`
Tada (Then): sukuriamas canvas flow (v210, cuda-linear), kuriame per 60 s pateikta ≥ 99,8 % grain'ų
Ir (And): kas 25-tas canvas kadras bitas į bitą sutampa su CPU etalonu; funkcijoje D2H kopijų 0 baitų

[ ] 2. Mišrūs formatai (teigiamas)
Duota (Given): įėjimai 1080i50, 1080p25 ir 720p50 viename tinklelyje
Kai (When): multivieweris veikia 60 s
Tada (Then): visos plytelės rodomos, canvas dažnis lieka 50/1, išėjime nėra spragų

[ ] 3. Dingęs įėjimas (neigiamas)
Duota (Given): veikiantis tinklelis
Kai (When): vieno įėjimo gamintojas nutraukiamas
Tada (Then): per ≤ 1 s tos plytelės būsena tampa `no_signal` (statistikoje ir nupiešta plytelėje)
Ir (And): kitos plytelės ir canvas išėjimas nepaliesti (0 papildomų prarastų grain'ų); gamintojui grįžus plytelė atsinaujina per ≤ 1 s

[ ] 4. Juodas vaizdas (teigiamas)
Duota (Given): įėjimas, kurio turinys tampa juodas (Y vidurkis ≤ 80, maksimumas ≤ 128, 10 bitų)
Kai (When): tai trunka 2 s
Tada (Then): aliarmas `black` įsijungia per 2,0 ± 0,1 s
Ir (And): pirmas nejuodas kadras jį išjungia per ≤ 0,1 s

[ ] 5. Sustingęs vaizdas (teigiamas)
Duota (Given): įėjimas, kuris kartoja tą patį kadrą (interlaced atveju — tuos pačius du laukus)
Kai (When): tai trunka 2 s
Tada (Then): aliarmas `frozen` įsijungia per 2,0 ± 0,1 s

[ ] 6. Judantis vaizdas nėra „sustingęs“ (neigiamas)
Duota (Given): įėjimas su judančiu testiniu vaizdu, kuriame keičiasi bent vienas plytelės mėginys per kadrą
Kai (When): multivieweris veikia 60 s
Tada (Then): aliarmas `frozen` neįsijungia nė karto; `black` taip pat ne

[ ] 7. Tyla ir garso lygiai (teigiamas)
Duota (Given): garso flow su 1 kHz sinusu −20 dBFS
Kai (When): skaitau statistiką
Tada (Then): rodomas pikas −20,0 ± 0,1 dBFS ir RMS −23,0 ± 0,1 dBFS abiem kanalams
Ir (And): garsui nutilus (RMS < −60 dBFS) aliarmas `silent` įsijungia per 5,0 ± 0,1 s; signalas −59 dBFS jo nesukelia

[ ] 8. Per daug įėjimų tinkleliui (neigiamas)
Duota (Given): tinklelis 4x4
Kai (When): nurodau 17 įėjimų
Tada (Then): procesas baigiasi su exit kodu ≠ 0 ir pranešimu „17 įėjimų netelpa į 4x4“; canvas flow nesukuriamas

[ ] 9. Tuščias tinklelis (kraštinis)
Duota (Given): nė vienas įėjimas dar neegzistuoja
Kai (When): paleidžiu multiviewerį
Tada (Then): canvas flow pildomas iškart, visos plytelės `no_signal`
Ir (And): įėjimams atsiradus bet kokia tvarka, plytelės užsipildo per ≤ 1 s kiekviena

[ ] 10. Plytelė už canvas ribų (neigiamas)
Duota (Given): išdėstymo failas, kuriame plytelė išeina už canvas
Kai (When): paleidžiu multiviewerį
Tada (Then): klaida su plytelės numeriu ir koordinatėmis; nieko nerašoma už buferio ribų

[ ] 11. Zero-copy rodymas (teigiamas; tik natyviame Linux, jei autorius patvirtins K14)
Duota (Given): canvas flow ir `vramflow-view`
Kai (When): rodau 60 s
Tada (Then): D2H kopijų 0 baitų visame kelyje
Ir (And): pateikimo vėlavimas p99 ≤ 1 kadras

---

# Papildoma informacija (Optional)
UI/UX nuorodos: aliarmų piešimas (rėmelis, lygių stulpeliai) — pagal LRT brand sistemą (`lrt-dizainas`), suderinama atskirai.
Techninės pastabos: mastelis v210 → v210 ploto vidurkiu, etalonas + CPU + CUDA bitų tikslumu; aliarmų statistika — GPU
redukcijos kernel'is; būsenos į JSON eilučių statistiką (planas §P7). VRAM: canvas 1080p50 ring'as 52,7 MiB,
3440×1440 — 126,6 MiB.
Rizikos / Apribojimai: CUDA–GL / Vulkan rodymas WSL2 nepatikrinamas — 11 kriterijus tikrinamas tik natyvioje
mašinoje ir tik po sprendimo K14; iki tol canvas rodomas bet kuriuo MXL vartotoju. Garsumas pagal EBU R 128 (LUFS),
tikrasis pikas, subtitrų / ANC aliarmai — kitam etapui. Aliarmų slenksčiai — pasiūlymas (planas K15).
Instrukcijos: README §6 papildomas `mv` komanda ir skyriumi „Monitoring“.
