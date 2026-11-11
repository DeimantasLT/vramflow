// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bendros GPU media funkcijų dalys: opcijos, skaitikliai (rašo tik tick gija), latencijų histogramos, JSON santrauka.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/histogram.hpp"
#include "func/thread_pool.hpp"

namespace mxlv::func {

struct Opts {
    std::string domain;
    // Variantas (architektūra 2026-11-11-gpu-vs-cpu-ir-upstream.md §1.1):
    //  G = impl gpu + storage cuda-linear; H = impl gpu + storage host; C = impl cpu (storage visada host, be CUDA).
    std::string impl = "gpu";
    std::string storage = "cuda-linear";
    int threads = 0;  // CPU variantui: darbininkų gijų skaičius (0 = min(flows, hardware_concurrency))
    std::uint32_t flows = 1;
    std::uint32_t flow_base = 0;
    std::uint64_t run_seed = 0x4C52544D584C4756ull;
    double duration_s = 30;
    double warmup_s = 5;
    std::string out = ".";
    std::string device_uuid;
    // gen
    std::uint32_t complexity = 2;
    bool audio = false;
    bool video = true;
    // logo
    std::string logo = "builtin";
    int logo_x = 1488, logo_y = 64;
    std::uint32_t inject_every = 0;  // logo: vieno bito klaida išėjime kas M laukų (savitikrai)
    std::uint32_t deadline_ms = 30;  // įėjimų laukimo riba po lauko laiko (WDDM/Windows sutrikimų atsarga; ring 10 laukų)
    std::int64_t read_offset_us = 500;
    int in_role = 1;
    // verify
    std::string mode = "logo";
    std::uint32_t every = 25;
    std::uint32_t lag_fields = 4;
    // enc
    std::string encoder = "nvenc";
    std::string preset = "p4";
    std::string tune = "ll";
    std::string x264_preset = "veryfast";
    int x264_threads = 2;
    std::uint32_t vbitrate = 7'500'000, abitrate = 192'000, muxrate = 8'500'000;
    int port_base = 20000;
    std::string tap_dir;
    int tap_flow = -1;
    std::uint32_t tap_every = 25;
    // dec
    int playout_frames = 6;  // atkūrimo buferio gylis (kadrai)
    std::string ref_crc;
};

// Vieno flow'o skaitikliai. Vardai — ataskaitoje.
struct FlowStats {
    std::map<std::string, std::uint64_t> c;
    void inc(const char* k, std::uint64_t v = 1) { c[k] += v; }
};

struct Stats {
    std::vector<FlowStats> flows;
    std::map<std::string, std::uint64_t> proc;  // proceso lygio
    std::map<std::string, Histogram> hist;      // ns
    std::uint64_t first_index = ~0ull, last_index = 0;
    std::string fatal;
    int exit_code = 0;
    void add_hist(const char* k, std::uint64_t ns) { hist[k].add(ns); }
};

std::string storage_options(const Opts& o, int dev);  // writer'io opcijos pagal --storage
std::atomic<bool>& stop_flag();
void install_signals();
int cuda_device(const std::string& uuid);  // -1 nerasta
std::string cuda_options(int dev);         // cuda-linear writer'io opcijos
void write_summary(const Opts& o, const std::string& role, const Stats& s, const std::string& extra_json = "");
double warm_tai_after(double warmup_s);    // TAI ns, nuo kada skaičiuoti latencijas

}  // namespace mxlv::func
