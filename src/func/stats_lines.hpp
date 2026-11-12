// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Gyva stebėsena (planas docs/planai/2026-11-11-production-grade.md §P7, K9; tick variklio planas V21): variklio StatsSink,
// rašantis JSON eilutes (viena eilutė per --stats-interval, numatyta 1 s) į --stats-file (`-` — stderr) ir operatoriaus
// įvykius (vienetas sugedo / atsigavo, įėjimas atsijungė / prisijungė / užstrigo, išėjimo klaida, formato pokytis, laikrodžio
// žingsnis) iš karto į stderr su UTC laiku. Abu kviečia StatsHub publikavimo gija — tick'as nelaukia ir nealokuoja.
// GPU varikliai (SM, NVENC, NVDEC, VRAM) — NVML per dlopen (libnvidia-ml.so.1, tvarkyklės dalis; tipai — CUDA įrankių
// rinkinio nvml.h); be NVML laukai null.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>

#include "engine/stats.hpp"

namespace mxlv::func {

struct Opts;

// NVML zondas (planas §P7 EngineProbe). Kviečiamas tik publikavimo gijoje.
struct GpuSample {
    bool valid = false;
    int sm_pct = -1, mem_pct = -1, nvenc_pct = -1, nvdec_pct = -1;  // < 0 — nepalaikoma
    std::uint64_t vram_used = 0, vram_total = 0;
};
class GpuProbe {
public:
    ~GpuProbe();
    bool open(const std::string& device_uuid, std::string& err);  // false — be NVML (laukai null)
    GpuSample sample();

private:
    void* lib_ = nullptr;
    void* dev_ = nullptr;
    void* f_util_ = nullptr;
    void* f_enc_ = nullptr;
    void* f_dec_ = nullptr;
    void* f_mem_ = nullptr;
    void* f_shutdown_ = nullptr;
};

class StatsLines final : public engine::StatsSink {
public:
    // role — santraukos rolė (logo, enc …); stats_file tuščias — eilučių nėra (įvykiai į stderr — visada).
    StatsLines(const Opts& o, std::string role);
    ~StatsLines() override;
    void publish(const engine::Snapshot&) override;
    void event(const engine::Event&, const engine::StatsNames&) override;
    double interval_s() const { return interval_s_; }
    std::uint64_t lines() const { return lines_.load(std::memory_order_acquire); }

private:
    std::string role_, run_id_, device_uuid_;
    double interval_s_ = 1.0;
    FILE* f_ = nullptr;
    bool own_ = false;
    std::atomic<std::uint64_t> lines_{0};  // rašo publikavimo gija
    GpuProbe gpu_;
    bool gpu_tried_ = false, gpu_ok_ = false;
    bool gpu_wanted_ = true;
    std::string buf_;
};

// „2026-11-12T10:00:00.123Z“ iš TAI ns (UTC = TAI − 37 s; MXL laikas — TAI).
std::string utc_of_tai(std::uint64_t tai_ns);
// Operatoriaus žurnalo eilutė į stderr: „[vramflow <laikas>] <role>: <tekstas>“ (ne tick gijoje).
void log_line(const char* role, const std::string& text);

}  // namespace mxlv::func
