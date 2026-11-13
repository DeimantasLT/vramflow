// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Bendros GPU media funkcijų dalys: opcijos, skaitikliai (rašo tik tick gija), latencijų histogramos, JSON santrauka.
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/histogram.hpp"
#include "common/second_ring.hpp"
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
    // Trukmė, s (planas D8, K-T8): 0 — veikti iki sustabdymo (tarnyba; numatyta funkcijoms); < 0 — nenurodyta (main_func:
    // verify — 30 s, kitos — 0). > 0 — matavimo režimas.
    double duration_s = -1;
    double warmup_s = 5;
    // Exit kodas (D8): auto — griežtas, kai --duration > 0 (bet kuris praradimas → 4, sugedęs vienetas → 2), švelnus tarnybos
    // režime (tvarkingas sustabdymas → 0; praradimai — tik santraukoje / statistikoje); on / off — priverstinai.
    std::string strict_exit = "auto";
    // Sekundžių eilių (lateness_s, loss_s, commit_lateness_s) gylis: paskutinės N sekundžių (žiedas; tarnybos režime
    // atmintis nebeauga). Numatyta 3600.
    std::uint32_t seconds_kept = 3600;
    std::uint32_t drain_ms = 500;  // sustabdant: darbininkų eilių (enc) išleidimo riba; antras signalas — nutraukia iškart
    int rt_priority = 0;           // SCHED_FIFO variklio gijai (0 — ne); be teisių — įspėjimas, darbas tęsiamas
    std::string cpu_affinity;      // variklio gijos CPU sąrašas „0,2-3“ (tuščia — nekeisti)
    // Gyva stebėsena (planas §P7, K9): JSON eilutė kas stats_interval_s į stats_file (`-` — stderr; tuščia — eilučių nėra).
    // Įvykiai (vienetas sugedo / atsigavo, įėjimų būsenos, išėjimo klaidos) — į stderr visada, iš karto.
    std::string stats_file;
    double stats_interval_s = 1.0;
    std::string out = ".";
    std::string device_uuid;
    // Paleidimas ir sustabdymas (planas §P6). run_id — šio paleidimo nuoma `<domain>.vramflow/<run_id>.lease` (tuščia —
    // "<subkomanda>-<pid>-<8 hex>"). input_wait_ms: 0 — įėjimų laukti neribotai (numatyta), N > 0 — riba (exit 4).
    // on_format_change: exit (įėjimas grįžo kitu formatu → exit 6) | wait (laukti, kol grįš tuo pačiu formatu).
    std::string run_id;
    std::int64_t input_wait_ms = 0;
    std::string on_format_change = "exit";
    // Briaunos laiko sutartis (V24, planas §6 D7): įėjimo laukimas po T(j) — aukštupio žyma commit-bound-ns; flow'ui be žymų
    // (svetimas rašytojas) — --input-expect-ms (< 0 — nenurodyta: video 0, garsas P — realaus laiko rašytojas kadro garsą
    // baigia ties T + P). --input-offset K (≤ 0): skaitomas j = i + K. --in-flow-base: įėjimų flow bazė (−1 — = --flow-base;
    // grandinėms viename domene, pvz. logo → logo). --allow-bad-clock: praleisti TAI poslinkio (37 s) patikrą paleidžiant.
    double input_expect_ms = -1;
    std::int32_t input_offset = 0;
    std::int64_t in_flow_base = -1;
    bool allow_bad_clock = false;
    std::uint32_t in_base() const { return in_flow_base < 0 ? flow_base : static_cast<std::uint32_t>(in_flow_base); }
    // Spragų politika (production-grade §6 K3, planas §6 D4): rašytojas commit'ina KIEKVIENĄ indeksą; be turinio —
    // invalid (MXL_GRAIN_FLAG_INVALID, turinys neišgalvojamas; numatyta) | freeze (paskutinis geras to paties pariteto grain'as
    // ne ilgiau kaip hold_ms, tada juoda) | black (juodas grain'as iškart). leave — tik testams (nieko necommit'inti).
    std::string on_loss = "invalid";
    std::uint32_t hold_ms = 1000;
    // Paleidimas (planas §6 D4, K-T7, V18): immediate — išėjimai sukuriami iškart, įėjimai prisijungia vėliau (iki tol
    // spragos pagal on_loss, gap_startup — ne praradimas); wait-all — laukti visų įėjimų prieš kuriant išėjimus. Parinktis
    // — tik logo (composite išėjimo rastras priklauso nuo sluoksnių įėjimų, enc MXL išėjimų neturi: abu — wait-all).
    std::string start = "immediate";
    // gen
    std::uint32_t complexity = 2;
    bool audio = false;
    bool video = true;
    // gen: grain'o garsas rašomas ne anksčiau kaip T(i) + audio_late_ms (pirmo tick'o atidarymo metu po jo; 0 — iškart po
    // video commit'o, t. y. ~P anksčiau, nei jį baigtų realaus laiko rašytojas). 20 — realaus laiko garso modelis 50 Hz.
    std::uint32_t audio_late_ms = 0;
    // logo
    std::string logo = "builtin";
    int logo_x = -1, logo_y = -1;  // < 0 — numatytoji vieta nuo rastro (viršutinis dešinys kampas; 1080: 1488, 64)
    std::uint32_t inject_every = 0;  // logo: vieno bito klaida išėjime kas M laukų (savitikrai)
    // Paskutinė įėjimų laukimo riba D nuo T(i) (planas §6 D2, K-T2, V20): 0 — auto 3·P/4 (50 Hz — 15 ms, 25 Hz — 30 ms);
    // > 2P − G apkarpoma su įspėjimu. gen: „vėlaus commit'o“ slenkstis (žemupio riba). WSL2 kampanijos nurodo 30 aiškiai.
    std::uint32_t deadline_ms = 0;
    std::int64_t read_offset_us = 500;
    int in_role = 1;
    // gfxgen
    std::uint32_t layers = 1;        // K: grafikos sluoksnių (flow'ų) skaičius vienam šaltiniui, <= 8
    std::string layer_geom;          // "logo,full,full,full" (logo 384x216, full 1920x1080, arba WxH); tuščia = k=0 logo, kiti full
    std::string gfx_mode = "pace";   // pace | manual
    // gfxgen pace (testinė, paslėpta): rašytojai sukuriami ir ritmas paleidžiamas, bet kadras NIEKADA nesubmit'inamas —
    // bibliotekos ritmas commit'ina tik „be kadro“ grain'us (INVALID). composite sluoksnio testams (peržiūra 2026-11-13 p. 1).
    bool test_no_submit = false;
    // composite
    std::vector<std::string> layer_xy;    // "X,Y" kiekvienam sluoksniui (K kartų)
    std::vector<float> opacity;           // kiekvienam sluoksniui (tuščia = visi 1.0)
    std::string base = "gen";             // gen | none
    std::string out_format = "v210";      // v210 | v210a
    std::vector<std::string> layer_flow;  // "k:UUID" — k-tojo sluoksnio (0-based) flow ID perrašymas
    // Sluoksnio grįžimo histerezė (peržiūra D3): išnykęs sluoksnis vėl naudojamas tik po tiek kadrų iš eilės, kai jis paruoštas
    // iki a(i) (1 — iškart nuo kito kadro).
    std::uint32_t layer_return_frames = 12;
    // verify
    std::string mode = "logo";
    std::uint32_t every = 25;
    std::uint32_t lag_fields = 4;
    // verify --mode composite: sluoksnio k nebuvimas indeksuose [I1, I2] — "K:I1-I2[@N]" (kartojama; @N — tik flow'ui N). absent — tame lange
    // laukiamas etalonas BE sluoksnio k; either — priimamas ir su juo, ir be jo (perėjimo langai, kai tiksli riba nežinoma).
    std::vector<std::string> layer_absent, layer_either;
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
    // Testinė (paslėpta) išėjimo klaidų injekcija (FaultyOutput, engine_glue.hpp): open | commit grąžina klaidą nuo
    // `test_out_error_after_s` sekundės, `test_out_error_count` indeksų iš eilės (0 — iki pabaigos). Tuščia — išjungta.
    std::string test_out_error;
    double test_out_error_after_s = 3;
    std::uint32_t test_out_error_count = 0;
    int test_out_error_flow = -1;  // tik šio flow'o (0…N − 1) išėjimas; < 0 — visi
    // Testinė (paslėpta) gedimo vėlinimo injekcija (logo, enc): process() A fazėje užmiega tiek ms, kai tame indekse kurio nors
    // vieneto įėjimo nėra (spraga atidarant — GapBlocked). Izoliacijos vartų „dantų“ meta-testui (tests/functional/isolation.py fiso_teeth):
    // funkcija, kuri dėl sugedusio įėjimo vėluoja ir sveikiems, vartuose turi būti FAIL. 0 — išjungta.
    std::uint32_t test_fault_proc_delay_ms = 0;
    // dec
    int playout_frames = 6;  // atkūrimo buferio gylis (kadrai)
    // dec: įėjimas, gavęs bent vieną paketą, tylus ilgiau nei tiek — sesija nutraukiama ir jungiamasi iš naujo (atidėjimas
    // 1, 2, 4 … ≤ 30 s; laikas skaičiuojamas, įvykis — stderr). Iki pirmo paketo laukiama neribotai (nutraukiama tik stop).
    std::uint32_t input_silence_ms = 2000;
    // interlaced flow'o 4:2:0 → 4:2:2 chroma: auto — pagal kadro AV_FRAME_FLAG_INTERLACED; field / frame — priverstinai.
    // NVENC (šioje GPU be lauko kodavimo) žymi kadrus progresyviais, nors turinys — supinti laukai su lauko chroma:
    // tokiam srautui reikia `field`.
    std::string chroma_siting = "auto";
    std::string ref_crc;
    double min_chroma_db = 0;  // verify --mode dec: chromos PSNR prieš gen etaloną (0 — netikrinama)
};

// keepwarm (keepwarm.cpp): GPU laikoma virš P8 — užpildymas size_mb kas interval_ms; duration_s 0 — iki sustabdymo.
// Numatytosios — iš docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md §6 (64 MB kas 5 ms).
// parent_pid > 0: baigti, kai to proceso nebėra (harness'as nužudytas SIGKILL / timeout / CI) — tikrinama kas užpildymą
// (getppid pasikeitimas, kai tai tiesioginis tėvas; kitaip kill(pid, 0)); papildomai PR_SET_PDEATHSIG (SIGTERM).
struct KeepWarmOpts {
    std::uint32_t size_mb = 64;
    double interval_ms = 5.0;
    double duration_s = 0;
    int parent_pid = 0;
};
int run_keepwarm(const KeepWarmOpts& k, const std::string& device_uuid);

// Vieno flow'o skaitikliai. Vardai — ataskaitoje. Tick'o kelyje be alokacijų: raktas (std::string) sukuriamas tik pirmą kartą
// (std::less<> — paieška pagal const char*, be laikino std::string; anksčiau > 15 simbolių raktas alokuodavo kiekvieną kartą),
// sekundžių eilės — fiksuoto dydžio žiedai (SecondRing, numatyta paskutinės 3600 s; --lateness-seconds).
struct FlowStats {
    std::map<std::string, std::uint64_t, std::less<>> c;
    std::uint64_t& at(const char* k) {
        auto it = c.find(std::string_view(k));
        if (it == c.end()) it = c.emplace(k, 0).first;
        return it->second;
    }
    void inc(const char* k, std::uint64_t v = 1) { at(k) += v; }
    // lateness_max_us: šio flow'o didžiausias commit'o (enc — eilės) vėlavimas nuo T(i), µs, tik po įšilimo (t ≥ warm).
    // Proceso histograma (out_lateness) apima visą tick'ą; šis — kiekvienam flow'ui atskirai (izoliacijos vartai).
    // lateness_s: tas pats maksimumas kiekvienai indekso laiko T(i) sekundei (TAI s → µs) — izoliacijos vartai lygina
    // to paties paleidimo langą prieš gedimą su gedimo langu (savikontrolė, ne atskiras kontrolinis paleidimas).
    // p99_s: to paties vėlavimo p99 kiekvienai sekundei (tikslus iš tos sekundės indeksų; izoliacijos vartų lango p99 —
    // per-sekundinių p99 mediana). Einamoji sekundė — fiksuotame buferyje (be alokacijų); vėlesnis senesnės sekundės indeksas
    // (pernešimas) į p99 neįskaičiuojamas (į maksimumą — taip).
    SecondRing lateness_s, p99_s;
    void note_lateness(std::uint64_t t, std::uint64_t t_index, double warm) {
        if (static_cast<double>(t) < warm || t < t_index) return;
        const std::uint64_t us = (t - t_index) / 1000;
        std::uint64_t& m = at("lateness_max_us");
        m = std::max<std::uint64_t>(m, us);
        const std::uint64_t sec = t_index / 1'000'000'000;
        lateness_s.max(sec, us);
        if (sec != cur_sec_) {
            if (cur_sec_ != kNoSec && sec < cur_sec_) return;
            if (cur_n_ != 0) p99_s.max(cur_sec_, cur_p99());
            cur_sec_ = sec;
            cur_n_ = 0;
        }
        if (cur_n_ < cur_.size()) cur_[cur_n_++] = static_cast<std::uint32_t>(std::min<std::uint64_t>(us, ~std::uint32_t{0}));
    }
    // Einamosios (dar neužbaigtos) sekundės p99: (sekundė, µs); n == 0 — nėra.
    std::uint64_t cur_sec() const { return cur_sec_; }
    std::uint32_t cur_n() const { return cur_n_; }
    std::uint64_t cur_p99() const {
        std::array<std::uint32_t, 512> v = cur_;
        const std::size_t k = static_cast<std::size_t>(0.99 * static_cast<double>(cur_n_ - 1));  // kaip Hist64::quantile
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.begin() + cur_n_);
        return v[k];
    }
    // loss_s: šio flow'o praradimų (spragų, lap'ų, praleidimų) skaičius kiekvienai indekso laiko T(i) sekundei (TAI s) —
    // izoliacijos vartai lygina funkcijos praradimus GEDIMO LANGE su aukštupio biudžetu tame pačiame lange (ne viso
    // gyvenimo skaitiklius); paleidimo spragos į gedimo langą nepatenka.
    SecondRing loss_s;
    void loss(const char* k, std::uint64_t t_index) {
        inc(k);
        loss_s.add(t_index / 1'000'000'000);
    }
    void set_seconds(std::size_t n) {
        lateness_s.set_capacity(n);
        p99_s.set_capacity(n);
        loss_s.set_capacity(n);
    }

private:
    static constexpr std::uint64_t kNoSec = ~std::uint64_t{0};
    std::array<std::uint32_t, 512> cur_{};
    std::uint32_t cur_n_ = 0;
    std::uint64_t cur_sec_ = kNoSec;
public:
};

struct Stats {
    std::vector<FlowStats> flows;
    std::map<std::string, std::uint64_t> proc;  // proceso lygio
    std::map<std::string, Histogram> hist;      // ns
    // gen: commit'o vėlavimo maksimumas kiekvienai indekso laiko T(i) sekundei (TAI s → µs; santraukoje
    // commit_lateness_s) — izoliacijos vartams sveiko aukštupio vėlavimą priskirti gedimo langui.
    SecondRing commit_lateness_s;
    std::uint64_t first_index = ~0ull, last_index = 0;
    std::string fatal;
    int exit_code = 0;
    int strict = -1;        // exit_code_for: 1 — griežtas, 0 — švelnus; < 0 — nerašoma (ne variklio funkcija)
    bool any_loss = false;  // exit_code_for: bent vienas praradimas
    void add_hist(const char* k, std::uint64_t ns) { hist[k].add(ns); }
    // Sekundžių žiedų dydis (--lateness-seconds): kviečiama po flows.resize, prieš run (alokuoja).
    void set_seconds(std::size_t n) {
        for (FlowStats& f : flows) f.set_seconds(n);
        commit_lateness_s.set_capacity(n);
    }
};

// Sluoksnių argumentai, bendri gfxgen / composite / verify --mode composite (apibrėžta gfxgen.cpp / composite.cpp).
bool parse_layer_geoms(const Opts& o, std::vector<vramflow::gfx::Raster>& out, std::string& err);  // --layer-geom -> K rastrų
float gen_audio_sample(std::uint64_t s);  // gen garso mėginys s (48 kHz); bendra su verify --mode audio
bool parse_xy(const std::string& s, int& x, int& y);                                                // "X,Y"

std::string storage_options(const Opts& o, int dev);  // writer'io opcijos pagal --storage
std::atomic<bool>& stop_flag();
// Antras SIGINT / SIGTERM (D8, A25): greitas išėjimas — eilių išleidimas (--drain-ms) nutraukiamas, flow'ai atlaisvinami.
std::atomic<bool>& fast_stop_flag();
void install_signals();
int cuda_device(const std::string& uuid);  // -1 nerasta
std::string cuda_options(int dev);         // cuda-linear writer'io opcijos
void write_summary(const Opts& o, const std::string& role, const Stats& s, const std::string& extra_json = "");
// CPU variantas (--impl cpu): santraukos „versions“ skaičiuojamos fone paleidžiant (libcuda įkėlimas ~85 ms WSL2 — anksčiau
// santraukos rašyme, t. y. sustabdymo kelyje po antro signalo). GPU variantams — santraukos metu (CUDA jau inicializuota;
// fone — negalima: FFmpeg turi pirmas sukurti pirminį kontekstą, enc.cpp).
void prefetch_versions(const Opts& o);
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
