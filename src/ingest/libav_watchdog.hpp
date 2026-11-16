// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// R13-2: libav kvietimų sargas (watchdog). libav ≤ 8.0 http nukreipimų kešas gali sukti giją amžinai: kešo pataikymas — `goto redo`
// be nukreipimų skaitiklio ir be pertraukimo patikros (http.c n6.1:362-373), 301 / 308 kešuojami be galiojimo pabaigos (425-431), todėl
// „A → 301 A“ (ar A → B → A) — 100 % CPU be I/O: rw_timeout, atidarymo riba ir sustabdymas niekada nesuveikia. FFmpeg 6.1 kešo
// išjungti neleidžia (FFmpeg master pataisė: kešo kelyje irgi tikrinamas max_redirects). Gijos libav viduje saugiai atšaukti negalima,
// o ji laikytų ir sustabdymą (join) — todėl sargas, pamatęs užstrigusį kvietimą, baigia VISĄ procesą aiškiu exit kodu (prižiūrėtojas
// paleidžia iš naujo).
//
// Gryna logika be libav / MXL (tests/unit/test_libav_watchdog.cpp): laikas — per laikrodžio funkciją.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace mxlv::ingest {

// Vieno šaltinio libav kvietimo būsena (rašo šaltinio gija ir jos pertraukimo kabliukas, skaito sargo gija).
struct CallWatch {
    std::string name;                         // šaltinio vardas žurnalui (be paslapčių)
    std::uint64_t margin_ns = 0;              // atsarga virš pertraukimo ribos ir didžiausias tarpas be pertraukimo kabliuko
    std::atomic<const char*> what{nullptr};   // dabartinis libav kvietimas (statinė eilutė); nullptr — ne libav viduje
    std::atomic<std::uint64_t> since{0};      // įėjimo į kvietimą laikas
    std::atomic<std::uint64_t> kill_at{0};    // pertraukimo riba + margin_ns; 0 — pertraukimo ribos nėra (tik pažangos taisyklė)
    std::atomic<std::uint64_t> last_poll{0};  // paskutinis pertraukimo kabliuko kvietimas (libav jį kviečia kiekviename laukime)
};

enum class Stuck : std::uint8_t {
    No,
    Deadline,    // kvietimas tęsiasi ilgiau nei jo pertraukimo riba + atsarga (kabliukas turėjo jį nutraukti seniai)
    NoProgress,  // libav viduje, bet pertraukimo kabliukas nekviestas ilgiau nei atsarga (teisėti laukimai jį kviečia ≤ 100 ms)
};

// Ar kvietimas užstrigo dabar (now). Ne libav viduje (what == nullptr) — niekada.
inline Stuck call_stuck(const char* what, std::uint64_t since, std::uint64_t kill_at, std::uint64_t last_poll, std::uint64_t margin_ns,
                        std::uint64_t now) {
    if (what == nullptr || since == 0) return Stuck::No;
    if (kill_at != 0 && now > kill_at) return Stuck::Deadline;
    const std::uint64_t alive = last_poll > since ? last_poll : since;
    if (margin_ns != 0 && now > alive && now - alive > margin_ns) return Stuck::NoProgress;
    return Stuck::No;
}

// Sargo gija: kas poll_ms tikrina visus šaltinius; užstrigus — on_stuck(šaltinis, kvietimas, priežastis, trukmė ns). Numatytas on_stuck
// (nenurodytas): eilutė į stderr ir std::_Exit(exit_code) — be destruktorių ir santraukos (kitos gijos dar dirba; MXL flow'us kitas
// paleidimas surenka mxlGarbageCollectFlows). Sargas turi gyventi ilgiau už visas šaltinių gijas (ir jų join).
class LibavWatchdog {
public:
    using Clock = std::function<std::uint64_t()>;
    using OnStuck = std::function<void(const std::string& name, const char* what, Stuck why, std::uint64_t elapsed_ns)>;
    LibavWatchdog(std::uint64_t margin_ns, int exit_code, Clock now, OnStuck on_stuck = {}, int poll_ms = 200);
    ~LibavWatchdog();
    LibavWatchdog(const LibavWatchdog&) = delete;
    LibavWatchdog& operator=(const LibavWatchdog&) = delete;
    // Naujas stebimas šaltinis; rodyklė galioja iki sargo sunaikinimo.
    CallWatch* add(const std::string& name);
    std::uint64_t margin_ns() const { return margin_; }

private:
    void run();
    std::uint64_t margin_;
    int exit_code_;
    Clock now_;
    OnStuck on_stuck_;
    int poll_ms_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::deque<CallWatch> w_;  // deque: emplace_back nejudina esamų elementų (rodyklės stabilios)
    std::thread th_;
};

}  // namespace mxlv::ingest
