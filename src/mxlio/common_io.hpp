// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bendros writer'io/reader'io dalys: opcijos, flow aprašai, skaitikliai, CSV/JSON išvestis, proceso metrikos.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include "common/histogram.hpp"
#include "mxlio/cuda_uuid.hpp"  // cuda_uuid_string, cuda_device_by_uuid

namespace mxlv {

// Išėjimo kodai (architektūra §4.1)
enum Exit : int { kExitOk = 0, kExitUsage = 1, kExitFatal = 2, kExitVerify = 3, kExitCapacity = 4, kExitConflict = 5 };

struct Injections {
    std::uint32_t corrupt_every = 0;
    std::uint32_t corrupt_header_every = 0;
    std::uint32_t skip_every = 0;
    std::uint32_t wrong_index_every = 0;
    std::uint32_t stale_every = 0;
    std::uint32_t commit_before_sync_every = 0;
    std::uint32_t late_write_ms = 5;
    std::uint32_t dup_every = 0;
    bool any() const {
        return corrupt_every || corrupt_header_every || skip_every || wrong_index_every || stale_every ||
               commit_before_sync_every || dup_every;
    }
};

struct Options {
    std::string domain;
    std::string storage = "cuda-linear";  // arba "host"
    std::string device_uuid;               // tuščia → įrenginys 0
    std::uint32_t width = 1920, height = 1080;
    std::int64_t rate_num = 50, rate_den = 1;
    std::uint32_t flows = 1;
    std::uint32_t flow_base = 0;
    std::uint64_t run_seed = 0x4C52544D584C4756ull;  // "LRTMXLGV"
    double duration_s = 20;
    double warmup_s = 2;
    std::string out = ".";
    std::uint32_t read_timeout_ms = 100;
    std::uint32_t stall_timeout_ms = 2000;
    std::uint32_t reopen_timeout_ms = 10000;
    std::uint32_t wait_flow_ms = 10000;
    std::uint32_t host_crosscheck_every = 0;
    double min_coverage = 0.95;  // reader: įvertintų indeksų dalis lange; mažiau → pajėgumo klaida
    bool gc_before_create = false;
    Injections inj;
};

std::string flow_id(std::uint32_t n);
std::uint64_t flow_seed(std::uint64_t run_seed, std::uint32_t n);
std::string flow_def_json(const Options& o, std::uint32_t n);
std::uint32_t v210_grain_bytes(std::uint32_t width, std::uint32_t height);
std::string writer_options(const Options& o, const std::string& uuid);

// Atomiškai skaitomi skaitikliai (rašo viena flow gija, skaito ataskaitų gija).
struct Counters {
    std::atomic<std::uint64_t> head_index{0}, cur_index{0};
    std::atomic<std::uint64_t> grains_ok{0}, grains_invalid{0}, gap_grains{0}, too_late_events{0},
        too_early_timeouts{0}, stalls{0}, lapped_discarded{0}, reopens{0};
    // VERIFIKACIJOS klasė
    std::atomic<std::uint64_t> index_mismatch{0}, slot_mismatch{0}, size_mismatch{0}, hdr_fail{0},
        content_index_mismatch{0}, mismatch_grains{0}, mismatch_words{0}, checksum_fail{0}, dup{0},
        host_crosscheck_fail{0};
    std::atomic<std::uint64_t> seq_break{0}, cuda_errors{0}, host_crosschecks{0}, coverage_low{0};
    // writer'io
    std::atomic<std::uint64_t> commits{0}, writer_late_skips{0}, clock_backward{0}, injections{0};

    std::uint64_t verification_errors() const {
        return index_mismatch + slot_mismatch + size_mismatch + hdr_fail + content_index_mismatch + mismatch_grains +
               checksum_fail + dup + host_crosscheck_fail;
    }
    // Viskas, kas reiškia „indeksas nepatikrintas": gap (įsk. INVALID), stall, lapped, per maža aprėptis.
    std::uint64_t capacity_errors() const { return gap_grains + stalls + lapped_discarded + coverage_low; }
};

// Latencijos: per-sekundės ir viso periodo histogramos.
struct Latencies {
    std::mutex m;
    Histogram sec_c2v, sec_e2e, sec_gpu, sec_jitter;
    Histogram tot_c2v, tot_e2e, tot_gpu, tot_wake, tot_jitter, tot_lag;
    std::uint64_t stale_max_ns = 0;
};

// Klaidos įrašas (indeksas + kategorijų bitai) savitikrai ir diagnostikai.
enum ErrBit : std::uint32_t {
    kErrIndex = 1u << 0, kErrSlot = 1u << 1, kErrSize = 1u << 2, kErrHdr = 1u << 3, kErrContentIndex = 1u << 4,
    kErrMismatch = 1u << 5, kErrChecksum = 1u << 6, kErrDup = 1u << 7, kErrCrosscheck = 1u << 8,
    kErrInvalidFlag = 1u << 9, kErrGap = 1u << 10, kErrLapped = 1u << 11
};
constexpr std::uint32_t kVerifyBits = kErrIndex | kErrSlot | kErrSize | kErrHdr | kErrContentIndex | kErrMismatch |
                                      kErrChecksum | kErrDup | kErrCrosscheck;
struct ErrorRecord {
    std::uint64_t index;
    std::uint32_t bits;
    std::uint32_t first_bad_word;
};

struct FlowState {
    std::uint32_t n = 0;
    std::string id;
    std::uint64_t seed = 0;
    Counters c;
    Latencies lat;
    std::mutex rec_m;
    std::vector<ErrorRecord> errors;           // reader: aptiktos klaidos (iki kMaxRecords)
    std::vector<std::uint64_t> injected;       // writer: injekcijų indeksai
    std::vector<std::string> injected_kind;
    std::uint64_t first_index = ~0ull, last_index = 0;  // reader: pirmas/paskutinis įvertintas indeksas
    std::string layout_storage;
    std::uint32_t slot_count = 0, grain_size = 0;
    std::string fatal;
    int fatal_code = 0;  // pirmos fatalios klaidos kodas
    int exit_code = 0;
    double coverage = 0;  // reader: įvertinti / visi indeksai lange
    // writer: commit'intų indeksų intervalai [a, b] — leidžia įrodyti, kad INVALID grain'as tikrai nebuvo commit'intas
    std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
    static constexpr std::size_t kMaxRecords = 200000;
    void record(std::uint64_t index, std::uint32_t bits, std::uint32_t first_bad = 0xFFFFFFFFu) {
        std::lock_guard g(rec_m);
        if (errors.size() < kMaxRecords) errors.push_back({index, bits, first_bad});
    }
    void add_committed(std::uint64_t index) {
        std::lock_guard g(rec_m);
        if (!committed.empty() && committed.back().second + 1 == index) committed.back().second = index;
        else committed.emplace_back(index, index);
    }
};

std::uint64_t now_tai_ns();
std::uint64_t now_mono_ns();

// 1 Hz ataskaitų gija: flow-<n>-<role>.csv ir proc-<role>.csv.
class Reporter {
public:
    Reporter(const Options& o, std::string role, std::vector<FlowState*> flows, int cuda_device);
    void run(const std::atomic<bool>& stop);
    void write_summary(const std::string& version_json, int exit_code);

private:
    void tick(bool final);
    const Options& o_;
    std::string role_;
    std::vector<FlowState*> flows_;
    int dev_;
    std::vector<FILE*> flow_csv_;
    FILE* proc_csv_ = nullptr;
    std::uint64_t start_mono_ = 0;
};

std::string versions_json();
void json_escape_into(std::string& out, const std::string& s);

}  // namespace mxlv
