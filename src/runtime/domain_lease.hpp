// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// DomainLease — paleidimo (proceso) nuoma MXL domene (planas §P6, sprendimas K8).
//
// Failas `<domenas>.vramflow/<run_id>.lease` ŠALIA domeno (MXL domeno katalogas lieka tik MXL; mxl-info ir GC jo neliečia),
// laikomas su flock(LOCK_EX): branduolys atlaisvina užraktą ir po `kill -9`. Gyvas savininkas = failas, kurio užrakto negalima
// gauti. Turinys: `run_id=`, `pid=`, po vieną `flow=<id>` kiekvienam šio paleidimo sukurtam išėjimo flow (savininko paieškai,
// kai antras paleidimas bando sukurti tą patį flow). Harness'as (tests/perf/domain.py) domeną trina tik be gyvų savininkų.
// flock ant tmpfs (/dev/shm) veikia, ant tinklo FS — ne; nuomos katalogas visada tame pačiame FS kaip domenas.
#include <string>
#include <vector>

namespace mxlv::runtime {

// "<domenas>.vramflow" (galiniai '/' nuimami).
std::string lease_dir(const std::string& domain);
// Numatytasis run_id: "<role>-<pid>-<8 hex>".
std::string make_run_id(const std::string& role);
// run_id: 1…64 simboliai [A-Za-z0-9._-].
bool valid_run_id(const std::string& run_id);

// cuda-linear flow reader'is jungiasi prie `<domenas>/<uuid>.mxl-flow/payload.sock` (Unix lizdas, sun_path — 107 simboliai;
// 59 iš jų — flow katalogas ir lizdo vardas). Ilgesnis kanoninis domeno kelias: writer'is flow sukuria, bet joks reader'is
// kitame procese prie jo neprisijungia (vartotojas lauktų be galo). Todėl riba tikrinama paleidžiant.
constexpr std::size_t kMaxCudaDomainPath = 48;
// Tuščias tekstas — tinka (arba domeno dar nėra: tai praneš MXL); kitaip klaida su faktiniu ilgiu ir riba.
std::string cuda_domain_path_error(const std::string& domain);

class DomainLease {
public:
    DomainLease() = default;
    DomainLease(DomainLease&& o) noexcept;
    DomainLease& operator=(DomainLease&& o) noexcept;
    DomainLease(const DomainLease&) = delete;
    DomainLease& operator=(const DomainLease&) = delete;
    ~DomainLease();

    // Sukuria nuomos failą ir jį užrakina. Klaida (run_id jau naudojamas gyvo proceso, netinkamas, FS klaida) → err.
    static bool acquire(const std::string& domain, const std::string& run_id, DomainLease& out, std::string& err);
    // Gyvų savininkų run_id (surūšiuoti); negyvų nuomos failai pašalinami.
    static std::vector<std::string> live_owners(const std::string& domain);
    // Gyvo savininko, užregistravusio flow_id, run_id; tuščia — nėra.
    static std::string flow_owner(const std::string& domain, const std::string& flow_id);

    // Užregistruoja išėjimo flow šiam paleidimui (flow=<id> eilutė).
    bool add_flow(const std::string& flow_id);
    const std::string& run_id() const { return run_id_; }
    bool held() const { return fd_ >= 0; }
    // Atlaisvina užraktą ir pašalina failą.
    void release();

private:
    int fd_ = -1;
    std::string path_, run_id_;
};

}  // namespace mxlv::runtime
