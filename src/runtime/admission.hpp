// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Priėmimo kontrolė (planas docs/planai/2026-11-11-production-grade.md §P4, user story P4): paklausa ir talpa — megapikseliais
// per sekundę kiekvienam varikliui (NVDEC, NVENC; 1080i50 / 1080p25 = 51,84 Mpix/s srautui), NVENC sesijomis (GeForce
// tvarkyklės riba: išmatuota 12, 13-a sesija nuverčia visą enc procesą — docs/rezultatai/2026-11-17-p8-talpa.md) ir VRAM
// baitais. Talpa — iš P8 talpos failo (tests/perf/capacity_file.py, schema 1). Atmesta: spėti talpą iš GPU modelio lentelės.
//
// Ribos: Mpix/s ir VRAM — talpa × headroom (numatyta 0,85); NVENC sesijos — tiksliai išmatuotas skaičius (kieta tvarkyklės
// riba, atsarga netaikoma). Talpa 0 — nežinoma, tas variklis netikrinamas. `bound: lower_bound` (tikra talpa didesnė) —
// reikšmė vis tiek naudojama kaip riba (atsargiai), apie tai įspėjama (capacity_warnings).
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/format.hpp"
#include "runtime/safe_files.hpp"

namespace mxlv::runtime {

// Vieno srauto (arba proceso pastovios dalies) paklausa.
struct Demand {
    double nvdec_mpix_s = 0, nvenc_mpix_s = 0;
    int nvenc_sessions = 0;
    std::uint64_t vram_bytes = 0;
};
Demand operator+(const Demand& a, const Demand& b);

// Funkcijos (talpos failo `functions.<žymė>`) išmatuotas VRAM modelis: proceso VRAM = offset + N · per_flow.
struct FunctionCapacity {
    std::uint64_t vram_per_flow_bytes = 0, vram_offset_bytes = 0;  // 0 — neišmatuota
    int knee = 0;
};

struct Capacity {
    double nvdec_mpix_s = 0, nvenc_mpix_s = 0;  // 0 — nežinoma (netikrinama)
    std::uint64_t vram_bytes = 0;               // visa GPU atmintis (faile `gpu.vram_bytes` arba iš įrenginio); 0 — netikrinama
    int nvenc_sessions = 0;                     // 0 — nežinoma / be ribos
    double headroom = 0.85;
    std::string source;  // talpos failo kelias (pranešimams)
    std::string gpu_name, gpu_uuid, driver, format, measured_utc;
    bool nvdec_lower_bound = false, nvenc_lower_bound = false;
    std::map<std::string, FunctionCapacity> functions;
};

// Sprendimas: admit; atmetus — ribojantis variklis ("nvdec" | "nvenc" | "nvenc_sessions" | "vram") ir skaičiai jo vienetais
// (Mpix/s, sesijos, baitai): jau užimta, prašoma, leidžiama.
struct Decision {
    bool admit = true;
    std::string engine;
    double used = 0, requested = 0, limit = 0;
};

// Ar `req` telpa šalia jau užimto `used`. Tikrinama tvarka: nvdec, nvenc, nvenc_sessions, vram (pirmas viršytas — atmetimo
// priežastis).
Decision decide(const Demand& used, const Demand& req, const Capacity& c);
// „nvdec: prašoma 51,84, užimta 829,44, leidžiama 850,00 Mpix/s“ (VRAM — MB; sesijos — vienetais).
std::string describe(const Decision& d);

// path — talpos failas arba katalogas (tada `<katalogas>/capacity-<device_uuid>.json`). Klaida (nėra failo, ne ta schema,
// trūksta lauko, failas kitai GPU — UUID nesutampa) → false ir err. device_uuid tuščias — UUID netikrinamas.
bool load_capacity(const std::string& path, const std::string& device_uuid, Capacity& out, std::string& err);
// Tas pats iš teksto (testams); source nekeičiamas.
bool parse_capacity(const std::string& json, Capacity& out, std::string& err);
// Įspėjimai apie naudojamą talpą: kita tvarkyklės versija nei dabartinė (abi versijos tekste; running tuščias — nežinoma),
// `lower_bound` varikliai.
std::vector<std::string> capacity_warnings(const Capacity& c, const std::string& running_driver);

// Kadro megapikseliai per sekundę: plotis × aukštis × kadrų dažnis (interlaced — du laukai sudaro vieną kadrą).
double mpix_s(const vramflow::VideoFormat& f);
// Ingest / dec srauto paklausa: NVDEC Mpix/s (nvdec == false — programinis dekoderis, 0) ir VRAM įvertis:
// surfaces NV12 kadrų (dekoderio DPB + atsarga + laikomi kadrai; plotis / aukštis lygiuoti 64 / 32) + ring_grains v210 grain'ų
// (išėjimo flow'o ring'as cuda-linear). Garso flow'ai — host atmintyje (neskaičiuojami).
Demand ingest_demand(const vramflow::VideoFormat& f, int surfaces, std::uint32_t ring_grains, bool nvdec);

// Rezervacijų knyga tarp procesų (planas §P4; PF-4): nuomos kataloge `<domenas>.vramflow/` (P6) po vieną failą procesui
// (`adm-<pid>-<8 hex>.adm`), laikomą su flock(LOCK_EX) — branduolys užraktą atlaisvina ir po `kill -9`, todėl mirusio proceso
// rezervacija nebeskaičiuojama (jo failas pašalinamas kito skaitymo metu). Skaitymas + sprendimas + įrašymas — po bendru
// `admission.lock` užraktu (atominis tarp procesų). Skaičiuojamos tik tos pačios GPU (device_uuid) rezervacijos.
// Turinys: `device=`, `pid=`, po eilutę `r <id> <nvdec Mpix/s> <nvenc Mpix/s> <sesijos> <VRAM B>` kiekvienai rezervacijai.
// Riba: knyga mato tik VRAMFlow procesus tame pačiame domene (kitų GPU naudotojų ir kitų domenų — ne).
class Ledger {
public:
    Ledger() = default;
    Ledger(Ledger&& o) noexcept;
    Ledger& operator=(Ledger&& o) noexcept;
    Ledger(const Ledger&) = delete;
    Ledger& operator=(const Ledger&) = delete;
    ~Ledger();

    static bool open(const std::string& domain, const std::string& device_uuid, Ledger& out, std::string& err);
    // Atominė (flock): užimta = visų gyvų šios GPU procesų rezervacijos (ir šio); telpa → įrašoma. force — įrašoma ir netilpus
    // (--admission warn: kiti procesai turi matyti tikrą užimtumą); grąžinamas tas pats sprendimas. Tas pats id — pakeičiamas.
    Decision reserve(const std::string& id, const Demand& d, const Capacity& c, bool force = false);
    void release(const std::string& id);
    Demand total() const;  // tik gyvų procesų (ir šio)
    Demand own() const;    // tik šio Ledger rezervacijos
    bool held() const { return fd_ >= 0; }
    const std::string& path() const { return path_; }
    void close();  // atlaisvina užraktą ir pašalina failą (rezervacijos dingsta)

private:
    Demand others(bool may_unlink) const;  // may_unlink — tik laikant knygos užraktą (negyvų procesų failai šalinami)
    bool write_own();
    int fd_ = -1;
    LeaseDir ld_;  // patikrintas nuomos katalogas (runtime/safe_files.hpp); viskas per openat / unlinkat
    std::string dir_, path_, name_, device_;
    std::vector<std::pair<std::string, Demand>> res_;
};

// --admission auto | off | warn | enforce (planas K13; numatyta auto: enforce su talpos failu, warn be jo —
// docs/rezultatai/2026-11-17-p4-admission.md).
enum class AdmissionMode { Off, Warn, Enforce };
bool parse_admission_mode(const std::string& s, AdmissionMode& out);
// Numatytoji --admission auto (planas K13): yra talpos failas → enforce; nėra → warn. Aiškiai nurodytas režimas
// (off | warn | enforce) nuo talpos failo nepriklauso. Nežinomas vardas → false.
bool resolve_admission_mode(const std::string& s, bool have_capacity, AdmissionMode& out);

// Srautų priėmimas iš eilės. Pirma rezervuojama proceso pastovi dalis `process` (funkcijos VRAM offset; nulinė — praleidžiama),
// tada srautai flows[0], flows[1] … (id: "<prefix>-proc", "<prefix>-<k>"). enforce: pirmas netilpęs srautas ir VISI tolesni
// atmetami (priimamas prefiksas — „pirmi N“; tolesnis, kuris būtų tilpęs, pažymimas engine == "order"); netilpus pastoviai
// daliai — atmetami visi (process_refused). warn: visi priimami ir įrašomi į knygą (force), viršijimai skaičiuojami `over`.
// off — nieko netikrina ir nerašo.
struct AdmitResult {
    std::size_t admitted = 0;
    std::size_t over = 0;              // warn: priimta viršijus talpą
    bool process_refused = false;
    std::vector<Decision> decisions;   // kiekvienam srautui; process_refused — visiems ta pati pastovios dalies priežastis
};
AdmitResult admit_in_order(Ledger& l, const std::string& prefix, const Demand& process, const std::vector<Demand>& flows,
                           const Capacity& c, AdmissionMode mode);

}  // namespace mxlv::runtime
