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
#include "func/flows.hpp"
#include "func/thread_pool.hpp"
#include "common/format.hpp"
#include "gfx/raster.hpp"
#include "runtime/domain_lease.hpp"
#include "runtime/input_slot.hpp"
#include "runtime/open_policy.hpp"

namespace mxlv::func {

struct Opts {
    std::string domain;
    // Variantas (architektūra 2026-11-11-gpu-vs-cpu-ir-upstream.md §1.1):
    //  G = impl gpu + storage cuda-linear; H = impl gpu + storage host; C = impl cpu (storage visada host, be CUDA).
    std::string impl = "gpu";
    std::string storage = "cuda-linear";
    int threads = 0;  // CPU variantui: darbininkų gijų skaičius (0 = min(flows, hardware_concurrency))
    std::uint32_t flows = 1;
    // Flow formatas (--format, planas §P2): "1080i50" (numatytasis), "1080p25", "1080p50", "720p50" …; fmt — išskaidytas.
    std::string format = "1080i50";
    vramflow::VideoFormat fmt = vramflow::kFmt1080i50;
    std::uint32_t flow_base = 0;
    std::uint64_t run_seed = 0x4C52544D584C4756ull;
    double duration_s = 30;
    double warmup_s = 5;
    std::string out = ".";
    std::string device_uuid;
    // Paleidimas ir sustabdymas (planas §P6). run_id — šio paleidimo nuoma `<domain>.vramflow/<run_id>.lease` (tuščia —
    // "<subkomanda>-<pid>-<8 hex>"). input_wait_ms: 0 — įėjimų laukti neribotai (numatyta), N > 0 — riba (exit 4).
    // on_format_change: exit (įėjimas grįžo kitu formatu → exit 6) | wait (laukti, kol grįš tuo pačiu formatu).
    std::string run_id;
    std::int64_t input_wait_ms = 0;
    std::string on_format_change = "exit";
    // gen
    std::uint32_t complexity = 2;
    bool audio = false;
    bool video = true;
    // logo
    std::string logo = "builtin";
    int logo_x = -1, logo_y = -1;  // < 0 — numatytoji vieta nuo rastro (viršutinis dešinys kampas; 1080: 1488, 64)
    std::uint32_t inject_every = 0;  // logo: vieno bito klaida išėjime kas M laukų (savitikrai)
    std::uint32_t deadline_ms = 30;  // įėjimų laukimo riba po lauko laiko (WDDM/Windows sutrikimų atsarga; ring 10 laukų)
    std::int64_t read_offset_us = 500;
    int in_role = 1;
    // gfxgen
    std::uint32_t layers = 1;        // K: grafikos sluoksnių (flow'ų) skaičius vienam šaltiniui, <= 8
    std::string layer_geom;          // "logo,full,full,full" (logo 384x216, full 1920x1080, arba WxH); tuščia = k=0 logo, kiti full
    std::string gfx_mode = "pace";   // pace | manual
    // composite
    std::vector<std::string> layer_xy;    // "X,Y" kiekvienam sluoksniui (K kartų)
    std::vector<float> opacity;           // kiekvienam sluoksniui (tuščia = visi 1.0)
    std::string base = "gen";             // gen | none
    std::string out_format = "v210";      // v210 | v210a
    std::vector<std::string> layer_flow;  // "k:UUID" — k-tojo sluoksnio (0-based) flow ID perrašymas
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
    // Testinė (paslėpta) parinktis: nuo `test_drop_after_s` sekundės po paleidimo kas `test_drop_every`-as kadras
    // atmetamas tuo pačiu keliu kaip pilnos eilės atveju (frames_dropped_queue). 0 — išjungta. A/V sinchronizacijos testams.
    std::uint32_t test_drop_every = 0;
    double test_drop_after_s = 0;
    // dec
    int playout_frames = 6;  // atkūrimo buferio gylis (kadrai)
    // interlaced flow'o 4:2:0 → 4:2:2 chroma: auto — pagal kadro AV_FRAME_FLAG_INTERLACED; field / frame — priverstinai.
    // NVENC (šioje GPU be lauko kodavimo) žymi kadrus progresyviais, nors turinys — supinti laukai su lauko chroma:
    // tokiam srautui reikia `field`.
    std::string chroma_siting = "auto";
    std::string ref_crc;
    double min_chroma_db = 0;  // verify --mode dec: chromos PSNR prieš gen etaloną (0 — netikrinama)
};

// keepwarm (keepwarm.cpp): GPU laikoma virš P8 — užpildymas size_mb kas interval_ms; duration_s 0 — iki sustabdymo.
// Numatytosios — iš docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md §6 (64 MB kas 5 ms).
struct KeepWarmOpts {
    std::uint32_t size_mb = 64;
    double interval_ms = 5.0;
    double duration_s = 0;
};
int run_keepwarm(const KeepWarmOpts& k, const std::string& device_uuid);

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

// Sluoksnių argumentai, bendri gfxgen / composite / verify --mode composite (apibrėžta gfxgen.cpp / composite.cpp).
bool parse_layer_geoms(const Opts& o, std::vector<vramflow::gfx::Raster>& out, std::string& err);  // --layer-geom -> K rastrų
bool parse_xy(const std::string& s, int& x, int& y);                                                // "X,Y"

std::string storage_options(const Opts& o, int dev);  // writer'io opcijos pagal --storage
std::atomic<bool>& stop_flag();
void install_signals();
int cuda_device(const std::string& uuid);  // -1 nerasta
std::string cuda_options(int dev);         // cuda-linear writer'io opcijos
void write_summary(const Opts& o, const std::string& role, const Stats& s, const std::string& extra_json = "");
double warm_tai_after(double warmup_s);    // TAI ns, nuo kada skaičiuoti latencijas

// --- Paleidimas ir sustabdymas (planas §P6) ---------------------------------------------------------------------------
// VIENA vieta visoms funkcijoms (ir būsimam bendram tick varikliui): įėjimų laukimas ir re-attach (runtime::InputSlot),
// išėjimų kūrimas su domeno nuoma (runtime::DomainLease). Funkcijų ciklai šios logikos nekopijuoja — tik kviečia.
runtime::OpenPolicy open_policy(const Opts& o);      // --input-wait-ms
runtime::FormatChange format_change(const Opts& o);  // --on-format-change
// Exit kodai: įėjimai neatsirado per --input-wait-ms; įėjimas grįžo kitu formatu ir --on-format-change exit (PF-3).
constexpr int kExitInputs = 4;
constexpr int kExitFormatChange = 6;
// Vidinis kodas: sustabdyta (SIGINT / SIGTERM) belaukiant įėjimų — tvarkinga pabaiga; exit_code() jį paverčia į 0.
constexpr int kStoppedWaiting = -1;
constexpr int exit_code(int rc) { return rc < 0 ? 0 : rc; }

// Įėjimo slot'as pagal funkcijos opcijas. expect == nullptr — laukiamu tampa pirmas prijungtas formatas.
runtime::InputSlot make_input(const Opts& o, const std::string& flow_id, const vramflow::VideoFormat* expect);
std::vector<runtime::InputSlot*> slot_ptrs(std::vector<runtime::InputSlot>& v);
// Pradinis prisijungimas (paleidimo tvarka nesvarbi): laukia visų įėjimų pagal --input-wait-ms. 0 — visi prijungti;
// kStoppedWaiting — sustabdyta belaukiant; kExitInputs — neatsirado per ribą arba netinkamas formatas (fatal užpildytas).
int attach_inputs(const std::vector<runtime::InputSlot*>& in, mxlInstance inst, const Opts& o, const char* who, std::string& fatal);
// Tick'o pradžioje: atjungtų įėjimų prisijungimas iš naujo (neblokuoja; prijungtam — be sisteminių kvietimų). 0 — tęsti;
// kExitFormatChange — įėjimas grįžo kitu formatu / saugykla ir --on-format-change exit (fatal užpildytas).
int poll_inputs(const std::vector<runtime::InputSlot*>& in, mxlInstance inst, std::string& fatal);

// Proceso domeno nuoma (paima main_func.cpp prieš subkomandą; laikoma iki proceso pabaigos).
runtime::DomainLease& domain_lease();
// Išėjimo flow sukurtas — užregistruojamas nuomoje (kitas paleidimas, bandantis jį sukurti, gaus savininko run_id).
void lease_output(const std::string& flow_id);
// Išėjimo flow sukurti nepavyko, nors jis domene yra — kodėl: gyvas kito paleidimo writer'is (su jo run_id; be nuomos —
// „savininkas nežinomas“) arba pasenęs flow be writer'io. Tuščias tekstas — tokio flow nėra (klaida kita). Kviečiama PO to,
// kai nepavykusio kūrimo writer'is atlaisvintas (kitaip jis pats atrodytų kaip gyvas writer'is).
std::string exists_note(mxlInstance inst, const Opts& o, const std::string& flow_id);
// Išėjimo flow: sukuria ir užregistruoja nuomoje. Flow jau turi gyvą writer'į → err papildomas savininko run_id.
bool create_output(mxlInstance inst, VideoOut& out, const std::string& id, const std::string& def, const std::string& options,
                   const vramflow::VideoFormat& f, const Opts& o, std::string& err);
bool create_output(mxlInstance inst, AudioOut& out, const std::string& id, const std::string& def, vramflow::Rational grain_rate,
                   const Opts& o, std::string& err);

}  // namespace mxlv::func
