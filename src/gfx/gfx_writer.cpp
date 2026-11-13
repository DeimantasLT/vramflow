// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// libvramflow-gfx (include/vramflow/gfx.h): RGBA/BGRA -> standartinis MXL video/v210a flow, rankinis ir ritmo režimai.
//
// Konvertavimas: cuda-linear saugykla -> CUDA kernel'is (gfx::cuda), host saugykla -> CPU (gfx::cpu); rezultatas
// bitas į bitą lygus gfx::ref etalonui. Ritmo režimas: vfgfx_submit / vfgfx_submit_at iškart konvertuoja kadrą į du
// paruoštus laukus (paritetai 0 ir 1; progresyviam rastrui — vieną grain'ą) vienoje iš paruoštų aibių; ritmo gija kas
// indeksą tik KOPIJUOJA „dabartinės“ aibės atitinkamo pariteto lauką į slot'ą ir commit'ina (be
// pakartotinio konvertavimo). Aibės: 2 (submit) + kQueueCap (submit_at eilė; išskiriamos tingiai, pirmą kartą
// prireikus). Laisva aibė = ne dabartinė ir ne eilėje; konvertuojama tik į laisvą (po sub_m — vienu metu vienas
// konvertuotojas), o dabartinę / eilę po lat_m keičia submit, submit_at ir ritmo gija (aktyvuoja eilės įrašus), todėl
// submit niekada nerašo į aibę, iš kurios ritmo gija skaito.
//
// Bendras ritmo vykdytojas (planas docs/planai/2026-11-12-tick-variklis.md V14, §6 D1): visi rašytojai su tuo pačiu
// (grain'ų dažnis, offset_ns, CUDA įrenginys) dalijasi VIENA ritmo gija — bendro tick variklio (mxlv::engine::TickEngine)
// šaltinio vienetu „pace“. Kiekviename tick'e c (atidaromas T(c) + offset_ns) PaceGroup::tick visiems nariams, kurių
// pace_next <= c, atidaro slot'ą, D2D kopijas deda į VIENĄ grupės stream'ą ir daro VIENĄ sync (host saugykla — memcpy ir
// commit iškart), tada commit'ina. Rašytojas su testų laiko kabliukais (PacingHooks::now / wait) gauna savo variklį
// (savo grupę): now -> variklio Clock (HookClock), wait -> EngineConfig::open_wait. Narystė dinamiška (start / stop
// pacing) ir gyvena grupėje, ne variklyje: variklis tiksi, kol grupė turi narių; paskutinis stop sustabdo giją.
//
// Vėlavimo taisyklė (kaip func::gen), kiekvienam rašytojui atskirai: gija pabudo vėlai -> indeksas, kurio rašytojas laukė
// (pace_next), VIS TIEK įrašomas (vėlyvas commit; skaitytojas sprendžia, ar per vėlu), o po commit'o, jei einamasis indeksas
// aplenkė kitą, jo laukai praleidžiami (late_skips) ir tęsiama nuo einamojo — jokio „catch-up“ srauto (variklio paties
// praleisti tick'ai tik pažadina vėliau). Laikrodis šoko atgal -> variklis miega iki tick'o laiko naujoje laiko juostoje
// (i tik didėja: tas pats indeksas du kartus neįrašomas).
//
// Kviečiančiojo CUDA darbų tvarka (gfx.h „CUDA FRAMES“): cuda_stream -> įvykis kviečiančiojo stream'e +
// cudaStreamWaitEvent mūsų s_conv (laukiama TIK to stream'o ankstesnių darbų, kviečiančiojo stream'as neblokuojamas);
// kitaip cudaDeviceSynchronize (R20), nebent VFGFX_FRAME_CALLER_SYNCED.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/format.hpp"
#include "engine/engine.hpp"
#include "engine/system_clock.hpp"
#include "func/flows.hpp"
#include "gfx/cpu/gfx_cpu.hpp"
#include "gfx/gfx_kernels.hpp"
#include "gfx/gfx_writer_internal.hpp"
#include "gfx/raster.hpp"
#include "mxlio/cuda_uuid.hpp"
#include "mxlio/slots.hpp"
#include "vramflow/gfx.h"

namespace gfx = vramflow::gfx;
namespace func = mxlv::func;
namespace gi = vramflow::gfx::internal;
namespace eng = mxlv::engine;

namespace {

constexpr int kQueueCap = VFGFX_SUBMIT_AT_MIN_QUEUE;  // submit_at eilės talpa (stats.queue_capacity)
constexpr int kSets = 2 + kQueueCap;                  // paruoštų aibių: dabartinė + submit tikslas + eilė
constexpr std::uint32_t kMaxStruct = 4096;
constexpr std::uint32_t kFrameFlags = VFGFX_FRAME_CALLER_SYNCED;
// mxlStatus kodai perduodami nepakeisti; VFGFX_ERR_* prasideda nuo 2000 — sritys nesikerta (gfx.h „ERRORS“).
static_assert(MXL_ERR_UNSUPPORTED_OPERATION < VFGFX_ERR_INVALID_ARG, "mxlStatus kodai persidengia su VFGFX_ERR_*");

// Kviečiančios gijos CUDA įrenginio perjungimas laikinai (atstatomas išeinant). cudaSetDevice kviečiamas VISADA (ne tik
// kai įrenginys kitas) — atsargumo dėlei: jis aiškiai padaro pirminį kontekstą dabartiniu, net jei kviečiančiojo
// cuCtxCreate kontekstas tame pačiame įrenginyje dabartinis. Testu neįrodyta, kad be to būtų klaida (mutantas M27
// išgyveno: cudart ir taip dirba pirminiame kontekste). Kviečiančiojo kontekstą atstato CtxGuard (išorinis apvalkalas).
struct DeviceScope {
    int prev = -1;
    bool active = false;
    explicit DeviceScope(int dev) {
        if (dev < 0) return;
        if (cudaGetDevice(&prev) != cudaSuccess) { cudaGetLastError(); prev = -1; }
        active = cudaSetDevice(dev) == cudaSuccess;
        if (!active) cudaGetLastError();
    }
    ~DeviceScope() {
        if (active && prev >= 0) cudaSetDevice(prev);
    }
};

// Kviečiančiosios gijos dabartinio CUDA (driver) konteksto išsaugojimas ir atstatymas (peržiūros 7 punktas): MXL
// cuda-linear backend'as (cuCtxSetCurrent + cudaSetDevice) ir mūsų privati cudart perjungia giją į pirminį kontekstą.
// Driver funkcijos gaunamos per cudart (cudaGetDriverEntryPointByVersion) — be naujos priklausomybės nuo libcuda.
struct DriverCtx {
    int (*get)(void**) = nullptr;
    int (*set)(void*) = nullptr;
};
const DriverCtx& driver_ctx() {
    static const DriverCtx d = [] {
        DriverCtx r;
        void* g = nullptr;
        void* s = nullptr;
        cudaDriverEntryPointQueryResult qg{}, qs{};
        if (cudaGetDriverEntryPointByVersion("cuCtxGetCurrent", &g, 12000, cudaEnableDefault, &qg) == cudaSuccess &&
            cudaGetDriverEntryPointByVersion("cuCtxSetCurrent", &s, 12000, cudaEnableDefault, &qs) == cudaSuccess &&
            qg == cudaDriverEntryPointSuccess && qs == cudaDriverEntryPointSuccess && g && s) {
            std::memcpy(&r.get, &g, sizeof g);
            std::memcpy(&r.set, &s, sizeof s);
        } else {
            cudaGetLastError();
        }
        return r;
    }();
    return d;
}
struct CtxGuard {
    void* saved = nullptr;
    bool on = false;
    explicit CtxGuard(bool use) {
        if (!use) return;  // host-only keliai CUDA neliečia (ir neturi inicializuoti cudart)
        const DriverCtx& d = driver_ctx();
        on = d.get && d.get(&saved) == 0;
    }
    ~CtxGuard() {
        if (!on) return;
        const DriverCtx& d = driver_ctx();
        void* now = nullptr;
        if (d.get(&now) == 0 && now != saved) d.set(saved);
    }
    CtxGuard(const CtxGuard&) = delete;
    CtxGuard& operator=(const CtxGuard&) = delete;
};

// Įrenginio eilės numeris pagal "deviceUuid" iš writer_options_json; -1 — nenurodyta / nerasta.
int device_from_options(const std::string& opts) {
    std::string u;
    if (!gfx::detail::json_str(opts, "deviceUuid", u) || u.empty()) return -1;
    return mxlv::cuda_device_by_uuid(u);
}

// Kviečiančiosios gijos klaida funkcijoms be writer'io (create, flow_def): vfgfx_last_error(NULL).
thread_local std::string g_create_error;

std::uint32_t struct_size_of(const void* p) {
    std::uint32_t sz = 0;
    std::memcpy(&sz, p, sizeof sz);
    return sz;
}

std::string size_msg(const char* name, std::uint32_t sz, std::size_t min) {
    return std::string(name) + ".struct_size " + std::to_string(sz) + " out of range (allowed " + std::to_string(min) + ".." +
           std::to_string(kMaxStruct) + "; initialise the struct with its _INIT macro)";
}

// Pratęsiamos įvesties struktūros skaitymas (gfx.h „Extensible structs“): struct_size >= min ir <= 4096; žinoma dalis
// kopijuojama į `out` (likusi — nuliai), nežinoma uodega (naujesnė antraštė) privalo būti nulinė. 0 arba kodas + msg.
template <class T>
int read_struct(const void* p, std::size_t min, const char* name, T& out, std::string& msg) {
    const std::uint32_t sz = struct_size_of(p);
    if (sz < min || sz > kMaxStruct) {
        msg = size_msg(name, sz, min);
        return VFGFX_ERR_STRUCT_SIZE;
    }
    std::memset(static_cast<void*>(&out), 0, sizeof out);
    std::memcpy(static_cast<void*>(&out), p, std::min<std::size_t>(sz, sizeof out));
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t k = sizeof out; k < sz; ++k)
        if (b[k]) {
            msg = std::string(name) + ": fields beyond byte " + std::to_string(sizeof out) + " are non-zero (struct_size " +
                  std::to_string(sz) + "); library " VFGFX_VERSION_STRING " does not support them";
            return VFGFX_ERR_UNSUPPORTED;
        }
    return 0;
}

template <class A>
bool all_zero(const A& a) {
    for (const auto v : a)
        if (v) return false;
    return true;
}

bool is_uuid(const char* s) {
    if (std::strlen(s) != 36) return false;
    for (int k = 0; k < 36; ++k) {
        const bool dash = k == 8 || k == 13 || k == 18 || k == 23;
        if (dash ? s[k] != '-' : !std::isxdigit(static_cast<unsigned char>(s[k]))) return false;
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') {
            o.push_back('\\');
            o.push_back(ch);
        } else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", c);
            o += b;
        } else {
            o.push_back(ch);
        }
    }
    return o;
}

// Nepalaikomo kadrų dažnio žinutė (angliškai; format.hpp žinutės lietuviškos — vidinės).
std::string rate_msg(const vramflow::VideoFormat& f) {
    return "unsupported frame rate " + std::to_string(f.frame_rate.num) + "/" + std::to_string(f.frame_rate.den) +
           (f.raster.interlaced ? " for an interlaced flow (supported: 25/1, 30000/1001)"
                                : " for a progressive flow (supported: 25/1, 50/1, 30000/1001, 60000/1001)");
}

struct Pending {  // submit_at eilės įrašas: aibė `set` tampa dabartine nuo grain'o `first`
    std::uint64_t first;
    int set;
};

struct PaceGroup;  // bendras ritmo vykdytojas (žemiau)

}  // namespace

struct vfgfx_writer_s {
    mxlInstance inst = nullptr;
    std::string flow_id;  // vfgfx_get_format
    func::VideoOut out;
    gfx::Raster r{};
    vramflow::VideoFormat fmt{};  // r + kadrų dažnis iš flow aprašo; ritmas — fmt.grain_rate()
    mxlRational rate{50, 1};
    bool cuda_store = false;
    int device = -1;       // cuda-linear saugyklos įrenginys; host saugykloje -1 (jokių CUDA kvietimų)
    int conv_dev = -1;     // host saugykla: įrenginys CUDA kadrų atsisiuntimui (deviceUuid arba -1 = numatytasis); s_conv kuriamas tik tada
    int fields = 1;  // paruoštų grain'ų skaičius vienoje aibėje (2 interlaced, 1 progresyviam)
    cudaStream_t s_conv = nullptr;  // create / write / submit (po conv_m); host saugykloje kuriamas tingiai (convert)
    cudaEvent_t ev_caller = nullptr;  // cuda_stream keliui (po conv_m), kuriamas tingiai
    std::atomic<bool> cuda_used{false};  // writer'is jau naudoja CUDA kviečiančiojo gijoje (CtxGuard reikalingas)

    // Tarpiniai buferiai (kuriami pirmą kartą prireikus)
    void* d_stage = nullptr;  // device RGBA (host kadras -> cuda-linear)
    std::size_t d_stage_bytes = 0;
    std::vector<std::uint8_t> h_stage;  // host RGBA (CUDA kadras -> host saugykla)
    std::uint8_t* prep[kSets][2] = {};  // [aibė][paritetas]; device arba host
    int nsets = 0;                      // išskirtų aibių (2 nuo create; kSets po pirmo submit_at); keičia tik sub_m / create

    std::atomic<std::uint64_t> committed{0};
    std::atomic<std::uint64_t> last_index{VFGFX_INDEX_NONE};  // paskutinio commit'into indekso (abu režimai)
    std::atomic<std::uint64_t> n_written{0}, n_submitted{0}, n_scheduled{0}, n_sched_late{0}, n_sched_replaced{0},
        n_caller_errors{0};
    std::mutex op_m;   // write / start_pacing / stop_pacing serializacija
    std::mutex sub_m;  // submit / submit_at serializacija
    std::mutex conv_m;  // convert(): bendri staging buferiai ir s_conv (write ir submit gali veikti lygiagrečiai)
    std::mutex lat_m;  // cur / has / queue / pace_next + paruoštų laukų skaitymas ritmo gijoje
    int cur = 0;
    bool has = false;
    std::vector<Pending> queue;   // surikiuota pagal first (didėjančiai), first unikalūs
    std::uint64_t pace_next = 0;  // ritmo metu: pirmas indeksas, kurio turinio gija dar nepasirinko

    std::atomic<bool> pacing{false};
    std::shared_ptr<PaceGroup> group;  // ritmo metu: grupė, kurios narys (keičia tik op_m turėtojas)
    bool pace_dead = false;            // ritmo kelyje išimtis: rašytojas nebeaptarnaujamas iki stop (po grupės m)
    std::atomic<std::int64_t> offset_ns{0};
    gi::PacingHooks hooks;
    std::mutex st_m;
    gi::PacingStats stats;

    std::mutex err_m;
    std::string err;   // paskutinė kviečiančiosios (write/submit/...) klaida; išvaloma sėkmingu kvietimu
    std::string perr;  // paskutinė ritmo gijos klaida; išvaloma sėkmingu ritmo commit'u
    std::atomic<bool> has_err{false}, has_perr{false};

    void set_error(std::string m) {
        std::lock_guard<std::mutex> l(err_m);
        err = std::move(m);
        has_err.store(true);
    }
    void clear_error() {
        if (!has_err.load()) return;
        std::lock_guard<std::mutex> l(err_m);
        err.clear();
        has_err.store(false);
    }
    void set_pace_error(std::string m) {
        std::lock_guard<std::mutex> l(err_m);
        perr = std::move(m);
        has_perr.store(true);
    }
    void clear_pace_error() {
        if (!has_perr.load()) return;
        std::lock_guard<std::mutex> l(err_m);
        perr.clear();
        has_perr.store(false);
    }
    int fail(int code, std::string m) {
        set_error(std::move(m));
        n_caller_errors.fetch_add(1);
        return code;
    }
    // CUDA klaida: savo klaidą išvalome iš gijos būsenos (nepalikti senos klaidos kitiems kvietimams, auditas 21).
    int fail_cuda(const char* what, cudaError_t e) {
        cudaGetLastError();
        return fail(VFGFX_ERR_CUDA, std::string(what) + ": " + cudaGetErrorString(e));
    }
    void bump(std::uint64_t gi::PacingStats::*field, std::uint64_t n = 1) {
        std::lock_guard<std::mutex> l(st_m);
        stats.*field += n;
    }

    // Kadro nuskaitymas ir patikra (gfx.h „FRAMES“): 0 arba klaidos kodas; fr — žinoma struktūros dalis.
    int read_frame(const vfgfx_frame* f, vfgfx_frame& fr) {
        if (!f) return fail(VFGFX_ERR_INVALID_ARG, "frame is NULL");
        std::string msg;
        if (const int c = read_struct(f, offsetof(vfgfx_frame, cuda_stream) + sizeof(void*), "vfgfx_frame", fr, msg))
            return fail(c, msg);
        if (fr.flags & ~kFrameFlags) return fail(VFGFX_ERR_UNSUPPORTED, "unknown vfgfx_frame.flags " + std::to_string(fr.flags));
        if (!all_zero(fr.reserved)) return fail(VFGFX_ERR_UNSUPPORTED, "vfgfx_frame.reserved must be zero");
        if (fr.pixel_format != VFGFX_PIXEL_RGBA8 && fr.pixel_format != VFGFX_PIXEL_BGRA8)
            return fail(VFGFX_ERR_UNSUPPORTED, "unsupported pixel_format " + std::to_string(fr.pixel_format) +
                                                   " (supported: RGBA8 = 0, BGRA8 = 1)");
        if (!fr.pixels) return fail(VFGFX_ERR_INVALID_ARG, "vfgfx_frame.pixels is NULL");
        if (fr.alpha != VFGFX_ALPHA_STRAIGHT && fr.alpha != VFGFX_ALPHA_PREMULTIPLIED)
            return fail(VFGFX_ERR_UNSUPPORTED, "unsupported vfgfx_frame.alpha " + std::to_string(fr.alpha));
        if (fr.mem != VFGFX_MEM_HOST && fr.mem != VFGFX_MEM_CUDA) return fail(VFGFX_ERR_UNSUPPORTED, "unsupported vfgfx_frame.mem " + std::to_string(fr.mem));
        if (fr.cuda_stream && fr.mem != VFGFX_MEM_CUDA) return fail(VFGFX_ERR_INVALID_ARG, "cuda_stream is only allowed for VFGFX_MEM_CUDA frames");
        if (fr.cuda_stream && (fr.flags & VFGFX_FRAME_CALLER_SYNCED))
            return fail(VFGFX_ERR_INVALID_ARG, "cuda_stream and VFGFX_FRAME_CALLER_SYNCED are mutually exclusive");
        if (fr.width != static_cast<std::uint32_t>(r.w) || fr.height != static_cast<std::uint32_t>(r.h))
            return fail(VFGFX_ERR_GEOMETRY, "frame " + std::to_string(fr.width) + "x" + std::to_string(fr.height) + " != flow " +
                                                std::to_string(r.w) + "x" + std::to_string(r.h));
        if (fr.stride < fr.width * 4u) return fail(VFGFX_ERR_GEOMETRY, "stride < width*4");
        if (fr.mem == VFGFX_MEM_CUDA && ((fr.stride & 3u) || (reinterpret_cast<std::uintptr_t>(fr.pixels) & 3u)))
            return fail(VFGFX_ERR_GEOMETRY, "CUDA frame pointer and stride must be multiples of 4 bytes");
        if (cuda_store && fr.mem == VFGFX_MEM_HOST && (fr.stride & 3u))
            return fail(VFGFX_ERR_GEOMETRY, "stride must be a multiple of 4 bytes for a cuda-linear flow");
        return 0;
    }

    // Prieš skaitant CUDA kadrą — tvarka su kviečiančiojo darbais (gfx.h „CUDA FRAMES“). own — mūsų stream'as, kuriame
    // toliau skaitomas kadras. Mūsų stream'ai cudaStreamNonBlocking, todėl be šio žingsnio jokios tvarkos nėra.
    int order_after_caller(const vfgfx_frame& fr, cudaStream_t own) {
        if (fr.flags & VFGFX_FRAME_CALLER_SYNCED) return 0;
        if (fr.cuda_stream) {
            // Įvykis + laukimas: tik to stream'o ankstesni darbai, kviečiančiojo stream'as neblokuojamas. Kviečiančiojo
            // (kitos cudart) stream'o rankena — tas pats CUstream pirminiame kontekste, todėl galioja ir mūsų cudart.
            if (!ev_caller) {
                const cudaError_t e = cudaEventCreateWithFlags(&ev_caller, cudaEventDisableTiming);
                if (e != cudaSuccess) {
                    ev_caller = nullptr;
                    return fail_cuda("cudaEventCreate (cuda_stream)", e);
                }
            }
            cudaError_t e = cudaEventRecord(ev_caller, static_cast<cudaStream_t>(fr.cuda_stream));
            if (e == cudaSuccess) e = cudaStreamWaitEvent(own, ev_caller, 0);
            if (e != cudaSuccess) return fail_cuda("event on the caller's cuda_stream", e);
            return 0;
        }
        // R20: visų konteksto darbų laukimas (pirminis kontekstas bendras visiems proceso CUDA runtime'ams).
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) return fail_cuda("cudaDeviceSynchronize before reading the CUDA frame", e);
        return 0;
    }

    // Konvertuoja kadro `n` laukų (paritetai par[k]) į dst[k] (paskirties saugyklos atmintis: device jei cuda_store, kitaip
    // host). Naudoja s_conv. 0 arba klaidos kodas. Kviečiama po op_m arba sub_m; kadras jau patikrintas (read_frame).
    int convert(const vfgfx_frame& f, int n, const int* par, std::uint8_t* const* dst) {
        std::lock_guard<std::mutex> cl(conv_m);
        const gfx::AlphaIn mode = f.alpha == VFGFX_ALPHA_PREMULTIPLIED ? gfx::AlphaIn::Premultiplied : gfx::AlphaIn::Straight;
        const gfx::PixelOrder order = f.pixel_format == VFGFX_PIXEL_BGRA8 ? gfx::PixelOrder::BGRA : gfx::PixelOrder::RGBA;
        const int w = r.w, h = r.h;
        const std::size_t stride = f.stride;
        const std::size_t span = static_cast<std::size_t>(h - 1) * stride + static_cast<std::size_t>(w) * 4;
        DeviceScope dev(device);
        if (cuda_store) {
            if (f.mem == VFGFX_MEM_CUDA) {
                if (const int e = order_after_caller(f, s_conv)) return e;
            }
            const std::uint8_t* src = static_cast<const std::uint8_t*>(f.pixels);
            if (f.mem == VFGFX_MEM_HOST) {
                if (d_stage_bytes < span) {
                    if (d_stage) cudaFree(d_stage);
                    d_stage = nullptr;
                    d_stage_bytes = 0;
                    if (const cudaError_t e = cudaMalloc(&d_stage, span); e != cudaSuccess) return fail_cuda("cudaMalloc RGBA staging", e);
                    d_stage_bytes = span;
                }
                const cudaError_t e = cudaMemcpyAsync(d_stage, f.pixels, span, cudaMemcpyHostToDevice, s_conv);
                if (e != cudaSuccess) return fail_cuda("H2D", e);
                src = static_cast<const std::uint8_t*>(d_stage);
            }
            for (int k = 0; k < n; ++k) {
                // Geometrija jau patikrinta (read_frame), todėl klaida čia — CUDA klaida (auditas 21: ne GEOMETRY).
                const cudaError_t e = gfx::cuda::rgba_to_v210a(src, w, h, stride, mode, r, par[k], dst[k], s_conv, order);
                if (e != cudaSuccess) return fail_cuda("rgba_to_v210a", e);
            }
            const cudaError_t e = cudaStreamSynchronize(s_conv);
            if (e != cudaSuccess) return fail_cuda("cudaStreamSynchronize", e);
            return 0;
        }
        const std::uint8_t* src = static_cast<const std::uint8_t*>(f.pixels);
        if (f.mem == VFGFX_MEM_CUDA) {
            // Host saugyklos rašytuvas CUDA kontekstą kuria TIK čia, kai pirmą kartą ateina CUDA kadras (po conv_m);
            // host kadrų rašytuvas niekada neliečia CUDA (veikia ir be GPU / su CUDA_VISIBLE_DEVICES="").
            DeviceScope cd(conv_dev);
            if (!s_conv) {
                cuda_used.store(true);
                if (const cudaError_t e = cudaStreamCreateWithFlags(&s_conv, cudaStreamNonBlocking); e != cudaSuccess) {
                    s_conv = nullptr;
                    return fail_cuda("cudaStreamCreate (CUDA frame download)", e);
                }
            }
            if (const int we = order_after_caller(f, s_conv)) return we;
            if (h_stage.size() < span) h_stage.resize(span);
            cudaError_t e = cudaMemcpyAsync(h_stage.data(), f.pixels, span, cudaMemcpyDeviceToHost, s_conv);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s_conv);
            if (e != cudaSuccess) return fail_cuda("D2H", e);
            src = h_stage.data();
        }
        for (int k = 0; k < n; ++k)
            if (!gfx::cpu::rgba_to_v210a(src, w, h, stride, mode, r, par[k], dst[k], order))
                return fail(VFGFX_ERR_GEOMETRY, "CPU rgba_to_v210a rejected the frame");
        return 0;
    }

    // pace = true: iškviečia ritmo gija — klaida eina į perr (ne į kviečiančiosios err).
    int commit_open(bool pace = false) {
        const bool forced = pace && hooks.fail_commit && hooks.fail_commit();  // testo kabliukas
        const mxlStatus st = forced ? MXL_ERR_INTERNAL : out.commit();
        if (st != MXL_STATUS_OK) {
            const std::string m = "mxlFlowWriterCommitGrain status " + std::to_string(st);
            if (pace) set_pace_error(m);
            else fail(static_cast<int>(st), m);
            return static_cast<int>(st);
        }
        last_index.store(out.last_committed);
        committed.fetch_add(1);
        return 0;
    }

    ~vfgfx_writer_s();  // visi keliai (įskaitant create išimtis) išvalo viską; release/free_all idempotentiški

    std::uint64_t now() const { return hooks.now ? hooks.now() : func::now_tai(); }
    std::uint64_t current_index() const { return mxlTimestampToIndex(&rate, now()); }
    // now − offset_ns (saugu be ženklo): indekso i commit'o laikas T(i) + offset praėjo <=> T(i) <= shifted_now().
    std::uint64_t shifted_now() const {
        std::uint64_t t = now();
        const std::int64_t off = offset_ns.load();
        if (off >= 0) {
            const std::uint64_t o = static_cast<std::uint64_t>(off);
            t = t > o ? t - o : 0;
        } else {
            t += 0 - static_cast<std::uint64_t>(off);  // beženklė neigimas: saugu ir INT64_MIN
        }
        return t;
    }
    // MXL indeksas (artimiausias) laikui now − offset_ns — vėlavimo / praleidimo taisyklei (kaip func::gen).
    std::uint64_t due_index() const { return mxlTimestampToIndex(&rate, shifted_now()); }
    // Pirmas indeksas, kurio commit'o laikas dar nepraėjo. MXL indeksas apvalinamas iki ARTIMIAUSIO, todėl due_index()
    // gali būti indeksas, kurio laikas dar ateityje (iki pusės periodo) — jo praleisti negalima (peržiūros 2 punktas).
    std::uint64_t first_not_passed() const {
        const std::uint64_t t = shifted_now();
        std::uint64_t i = mxlTimestampToIndex(&rate, t);
        if (mxlIndexToTimestamp(&rate, i) <= t) ++i;
        return i;
    }
    // Indeksas, nuo kurio ritmas prasidėtų dabar: pirmas, kurio commit'o laikas nepraėjo; niekada ne <= jau commit'into.
    std::uint64_t start_index() const {
        std::uint64_t i = first_not_passed();
        const std::uint64_t last = last_index.load();
        if (last != VFGFX_INDEX_NONE) i = std::max(i, last + 1);
        return i;
    }
    // Mažiausias submit_at priimamas indeksas (po lat_m).
    std::uint64_t min_scheduled_index() const { return pacing.load() ? pace_next : start_index(); }
    // Laisva aibė (ne dabartinė ir ne eilėje) arba -1; po lat_m.
    int free_set() const {
        for (int s = 0; s < nsets; ++s) {
            if (has && s == cur) continue;
            if (std::any_of(queue.begin(), queue.end(), [s](const Pending& p) { return p.set == s; })) continue;
            return s;
        }
        return -1;
    }
    // Aibių [nsets, n) išskyrimas (create arba po sub_m). 0 arba klaidos kodas.
    int alloc_sets(int n) {
        DeviceScope ds(device);
        const std::size_t gb = gfx::grain_bytes(r);
        for (; nsets < n; ++nsets) {
            // Visa aibė arba nieko: dalinės nesėkmės atveju jau išskirti laukai atlaisvinami (kitaip kartojant — nuotėkis).
            void* ptr[2] = {nullptr, nullptr};
            for (int p = 0; p < fields; ++p) {
                if (cuda_store) {
                    if (const cudaError_t e = cudaMalloc(&ptr[p], gb); e != cudaSuccess) {
                        for (int q = 0; q < p; ++q) cudaFree(ptr[q]);
                        return fail_cuda("cudaMalloc (prepared field)", e);
                    }
                } else {
                    ptr[p] = std::aligned_alloc(64, (gb + 63) / 64 * 64);
                    if (!ptr[p]) {
                        for (int q = 0; q < p; ++q) std::free(ptr[q]);
                        return fail(VFGFX_ERR_INTERNAL, "out of host memory (prepared field)");
                    }
                }
            }
            for (int p = 0; p < fields; ++p) prep[nsets][p] = static_cast<std::uint8_t*>(ptr[p]);
        }
        return 0;
    }
    // Eilės įrašų, kurių first <= bound, aktyvavimas (po lat_m): naujausias (didžiausias first) tampa dabartiniu turiniu,
    // rodomu nuo grain'o shown; ankstesni — pakeisti; first < shown — vėlus. Ritmo gija: bound = shown = i.
    void activate(std::uint64_t bound, std::uint64_t shown) {
        while (!queue.empty() && queue.front().first <= bound) {
            const Pending p = queue.front();
            queue.erase(queue.begin());
            if (!queue.empty() && queue.front().first <= bound) {
                n_sched_replaced.fetch_add(1);  // jį iškart pakeičia vėlesnis (jo laikas taip pat atėjo)
                continue;
            }
            if (p.first < shown) n_sched_late.fetch_add(1);
            cur = p.set;
            has = true;
        }
    }
    // Praėję įrašai (first < next_pacing_index) sutraukiami į dabartinį turinį ir neužima eilės (po lat_m).
    void collapse_past() {
        const std::uint64_t min = min_scheduled_index();
        if (min > 0) activate(min - 1, min);
    }
    // Visų laukiančių kadrų atšaukimas (po lat_m); jie skaičiuojami kaip niekada netapę dabartiniais.
    void cancel_all() {
        n_sched_replaced.fetch_add(queue.size());
        queue.clear();
    }

    // --- ritmo vykdytojo dalys vienam rašytojui (kviečia tik grupės gija, po grupės m ir šio lat_m) ---
    // Indekso i turinio pasirinkimas ir slot'as: host — kopija ir commit iškart (false); cuda-linear — true ir D2D
    // kopijos aprašas (dst, src, bytes): grupė visas kopijas paleidžia vienu batch'u, vienas sync, tada pace_commit.
    // Klaidos — į perr / errors.
    bool pace_copy(std::uint64_t i, void*& dst, const void*& src, std::size_t& bytes);
    void pace_commit(std::uint64_t i);                       // po kopijos: commit, on_commit kabliukas, skaitikliai
    void pace_copy_failed(const char* what, cudaError_t e);  // kopija / sync nepavyko: slot'as atšaukiamas
    void pace_advance();                                     // vėlavimo taisyklė: praleisti laukai neįrašomi (late_skips)
    void pace_kill(const char* what);                        // išimtis ritmo kelyje: rašytojas nebeaptarnaujamas
    void free_all();
};

bool vfgfx_writer_s::pace_copy(std::uint64_t i, void*& dst_out, const void*& src_out, std::size_t& bytes) {
    activate(i, i);
    pace_next = i + 1;  // nuo šiol submit_at indeksui i — per vėlu (turinys pasirinktas)
    if (!has) {
        // K3 (production-grade §6 K3, planas §6 D4): rašytojas commit'ina KIEKVIENĄ indeksą — be kadro INVALID grain'as
        // (turinys neliečiamas, validSlices = totalSlices); skaitytojas iškart gauna OK + INVALID, ne TOO_EARLY iki kito.
        const mxlStatus st = out.commit_invalid(i);
        if (st != MXL_STATUS_OK) {
            bump(&gi::PacingStats::errors);
            set_pace_error("INVALID grain " + std::to_string(i) + " status " + std::to_string(st));
            return false;
        }
        last_index.store(i);
        bump(&gi::PacingStats::no_frame);
        clear_pace_error();
        if (hooks.on_gap) hooks.on_gap(i);
        return false;
    }
    mxlStatus st = MXL_STATUS_OK;
    std::uint32_t* dst = out.open(i, st);
    if (!dst) {
        bump(&gi::PacingStats::errors);
        set_pace_error("OpenGrainSlot " + std::to_string(i) + " status " + std::to_string(st));
        return false;
    }
    const std::uint8_t* src = prep[cur][r.interlaced ? (i & 1) : 0];
    const std::size_t gb = gfx::grain_bytes(r);
    if (cuda_store) {  // kopija ir commit — grupės batch'e (po vieno sync)
        dst_out = dst;
        src_out = src;
        bytes = gb;
        return true;
    }
    std::memcpy(dst, src, gb);
    pace_commit(i);
    return false;
}

void vfgfx_writer_s::pace_copy_failed(const char* what, cudaError_t e) {
    cudaGetLastError();
    set_pace_error(std::string(what) + ": " + cudaGetErrorString(e));
    out.cancel();
    bump(&gi::PacingStats::errors);
}

void vfgfx_writer_s::pace_commit(std::uint64_t i) {
    if (commit_open(true) != 0) {
        out.cancel();
        bump(&gi::PacingStats::errors);
        return;
    }
    clear_pace_error();
    if (hooks.on_commit) hooks.on_commit(i);
    bump(&gi::PacingStats::committed);
}

void vfgfx_writer_s::pace_advance() {
    const std::uint64_t ni = due_index();
    if (ni > pace_next) {  // pavėlavome: praleisti laukai neįrašomi, tęsiama nuo einamojo (kaip gen)
        bump(&gi::PacingStats::late_skips, ni - pace_next);
        pace_next = ni;  // praleistiems indeksams submit_at — per vėlu
    }
}

void vfgfx_writer_s::pace_kill(const char* what) {
    set_pace_error(std::string("pacing thread ended: ") + (what ? what : "unknown exception"));
    bump(&gi::PacingStats::errors);
    out.cancel();
    pace_dead = true;
}

void vfgfx_writer_s::free_all() {
    DeviceScope dev(device);
    for (auto& set : prep)
        for (auto*& p : set) {
            if (!p) continue;
            if (cuda_store) cudaFree(p);
            else std::free(p);
            p = nullptr;
        }
    nsets = 0;
    if (d_stage) cudaFree(d_stage);
    d_stage = nullptr;
    if (s_conv || ev_caller) {
        DeviceScope cd(cuda_store ? -1 : conv_dev);  // cuda_store: jau esame `dev` įrenginyje
        if (s_conv) cudaStreamDestroy(s_conv);
        if (ev_caller) cudaEventDestroy(ev_caller);
    }
    s_conv = nullptr;
    ev_caller = nullptr;
}

// --- Bendras ritmo vykdytojas (V14) -------------------------------------------------------------------------------------
namespace {

// Rašytojo su testų laiko kabliukais (PacingHooks::now / wait) laikrodis. „Dabar“ — kabliuko laikas (be jo — TAI), bet ne
// mažiau nei paskutinis pasiektas laukimo tikslas: grįžęs iš kabliuko wait(i) (arba miegas iki tikslo) tick'as atidaromas,
// net jei kabliuko „dabar“ stovi vietoje (kaip anksčiau: ritmo gija pasitikėjo kabliuko laukimu). Monotoninis laikas ir
// tikras miegas — SystemClock. Visus metodus kviečia tik grupės (variklio) gija.
class HookClock final : public eng::Clock {
public:
    explicit HookClock(std::function<std::uint64_t()> now) : now_(std::move(now)) {}
    std::uint64_t now_tai() const override { return std::max(now_ ? now_() : sys_.now_tai(), floor_); }
    std::int64_t now_mono_raw() const override { return sys_.now_mono_raw(); }
    void sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) override {
        sys_.sleep_until(tai_ns, stop);
        if (!stop.load()) reached(tai_ns);
    }
    void reached(std::uint64_t tai_ns) { floor_ = std::max(floor_, tai_ns); }

private:
    std::function<std::uint64_t()> now_;
    eng::SystemClock sys_;
    std::uint64_t floor_ = 0;
};

// Viena ritmo gija (grain'ų dažnis, offset_ns, CUDA įrenginys) rašytojams be laiko kabliukų arba vienam rašytojui su jais
// (owner). Nariai keičiami po m; tick'as laiko m visą apdorojimo laiką, todėl po leave_pacing gija nario nebeliečia.
// Užraktų tvarka: registras -> m -> narių lat_m (didėjančia adresų tvarka) -> st_m / err_m.
struct PaceGroup {
    mxlRational rate{50, 1};
    std::int64_t offset = 0;
    int device = -1;                 // cuda-linear narių įrenginys; -1 — host saugykla (gija CUDA neliečia)
    vfgfx_writer owner = nullptr;    // != nullptr — rašytojo su laiko kabliukais savas variklis
    std::function<bool(std::uint64_t, std::int64_t, const std::atomic<bool>&)> wait_hook;
    std::unique_ptr<eng::Clock> clock;
    HookClock* hclock = nullptr;     // == clock, kai owner

    std::mutex m;
    std::vector<vfgfx_writer> members;  // didėjančia adresų tvarka
    std::atomic<bool> stop{false};
    std::atomic<bool> back{false};      // laikrodžio šuolis atgal nuo paskutinio tick'o (clock_backward)
    cudaStream_t stream = nullptr;      // visų narių D2D kopijos; kuria ir naikina grupės gija
    std::thread th;

    // Tick'o darbiniai masyvai (tik grupės gija, po m)
    struct Pend {
        vfgfx_writer w;
        std::uint64_t i;
    };
    std::vector<std::unique_lock<std::mutex>> locks;
    std::vector<Pend> pend;
    std::vector<void*> dsts;
    std::vector<const void*> srcs;
    std::vector<std::size_t> sizes;

    cudaError_t copy_batch();  // visos šio rato D2D kopijos į stream (vienas kvietimas) + vienas sync

    void run();
    void tick(std::uint64_t c);
    void kill_all(const char* what);
};

// Variklio funkcija: vienas šaltinio vienetas be variklio valdomų išėjimų (planas §7 lentelė: „ritmo gija — visų kopijos
// viename stream'e, vienas sync“); rašytojų slot'us atidaro, kopijuoja ir commit'ina PaceGroup::tick.
struct PacerFn final : eng::TickFunction {
    PaceGroup& g;
    explicit PacerFn(PaceGroup& gg) : g(gg) {}
    bool process(const eng::Batch& b, eng::TickContext&) override {
        g.tick(b.index);
        return true;
    }
    // Be wait kabliuko šuolį atgal aptinka variklis (StepDetector: TAI prieš CLOCK_MONOTONIC_RAW, > 5 ms — kaip ClockWatch).
    void on_clock_step(std::int64_t delta_ns, eng::TickContext&) override {
        if (delta_ns < 0 && !g.wait_hook) g.back.store(true);
    }
};

struct PaceRegistry {
    std::mutex m;
    std::vector<std::shared_ptr<PaceGroup>> groups;
};
// Sąmoningai nesunaikinamas: proceso pabaigoje statinių objektų naikinimas negali stabdyti / jungti ritmo gijų.
PaceRegistry& registry() {
    static auto* r = new PaceRegistry;
    return *r;
}

// Išimtis vieno rašytojo kelyje (pvz. on_commit kabliukas) sustabdo tik jį (gfx.h: rašytojai nepriklausomi).
template <class F>
void pace_guard(vfgfx_writer w, F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        try {
            w->pace_kill(e.what());
        } catch (...) {
            w->pace_dead = true;
        }
    } catch (...) {
        try {
            w->pace_kill(nullptr);
        } catch (...) {
            w->pace_dead = true;
        }
    }
}

void PaceGroup::kill_all(const char* what) {
    std::lock_guard<std::mutex> gl(m);
    for (vfgfx_writer w : members) {
        if (w->pace_dead) continue;
        std::lock_guard<std::mutex> l(w->lat_m);
        try {
            w->pace_kill(what);
        } catch (...) {
            w->pace_dead = true;
        }
    }
}

void PaceGroup::tick(std::uint64_t c) {
    std::lock_guard<std::mutex> gl(m);
    const bool stepped = back.exchange(false);
    locks.clear();
    for (vfgfx_writer w : members) locks.emplace_back(w->lat_m);
    if (stepped)
        for (vfgfx_writer w : members)
            if (!w->pace_dead) w->bump(&gi::PacingStats::clock_backward);
    // Ratas: kiekvienas narys, kurio pace_next <= c, rašo vieną indeksą. Antras ratas — tik po vėlyvo pabudimo, kai
    // praleidimas nustūmė pace_next iki c (vėluojantis indeksas įrašytas, tada einamasis — kaip ankstesnė gija).
    for (bool more = true; more;) {
        pend.clear();
        dsts.clear();
        srcs.clear();
        sizes.clear();
        for (vfgfx_writer w : members) {
            if (w->pace_dead || w->pace_next > c) continue;
            const std::uint64_t i = w->pace_next;
            bool queued = false;
            void* dst = nullptr;
            const void* src = nullptr;
            std::size_t bytes = 0;
            pace_guard(w, [&] { queued = w->pace_copy(i, dst, src, bytes); });
            if (queued) {
                pend.push_back(Pend{w, i});
                dsts.push_back(dst);
                srcs.push_back(src);
                sizes.push_back(bytes);
            } else if (!w->pace_dead) {
                pace_guard(w, [&] { w->pace_advance(); });
            }
        }
        if (!pend.empty()) {
            const cudaError_t e = copy_batch();
            for (const Pend& p : pend) {
                pace_guard(p.w, [&] {
                    if (e == cudaSuccess) p.w->pace_commit(p.i);
                    else p.w->pace_copy_failed("D2D", e);
                });
                if (!p.w->pace_dead) pace_guard(p.w, [&] { p.w->pace_advance(); });
            }
        }
        more = false;
        for (vfgfx_writer w : members)
            if (!w->pace_dead && w->pace_next <= c) more = true;
    }
    locks.clear();
}

cudaError_t PaceGroup::copy_batch() {
    // Vienas cudaMemcpyBatchAsync visoms kopijoms: WSL2 kiekvienas cudaMemcpyAsync — atskiras pateikimas (32 atskiros
    // kopijos G variante pakėlė commit'o vėlavimą p50 iki ~5 ms). Jei batch'as nepalaikomas — po vieną.
    cudaError_t e = cudaErrorNotSupported;
    if (dsts.size() > 1) {
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        std::size_t attr_idx = 0;
        e = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, stream);
        if (e != cudaSuccess) cudaGetLastError();
    }
    if (e != cudaSuccess) {
        e = cudaSuccess;
        for (std::size_t k = 0; k < dsts.size() && e == cudaSuccess; ++k)
            e = cudaMemcpyAsync(dsts[k], srcs[k], sizes[k], cudaMemcpyDeviceToDevice, stream);
    }
    const cudaError_t es = cudaStreamSynchronize(stream);  // vienas sync visoms šio rato D2D kopijoms
    return e != cudaSuccess ? e : es;
}

void PaceGroup::run() {
    DeviceScope dev(device);  // gija nauja: numatytasis įrenginys 0; čia įrenginys pastovus iki gijos pabaigos
    try {
        if (device >= 0) {
            const cudaError_t e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (e != cudaSuccess) {
                stream = nullptr;
                cudaGetLastError();
                throw std::runtime_error(std::string("cudaStreamCreate: ") + cudaGetErrorString(e));
            }
        }
        // Tick'as c atidaromas T(c) + offset_ns (expect = offset, ro = 0, viena riba). offset — briaunos poslinkis
        // (expect_ns), ne ro: variklio pirmas tick'as = due_index(dabar − expect), todėl ir |offset| > P / 2 pradeda nuo
        // pirmo nepraėjusio indekso (su ro = +60 ms variklis pradėtų 2 indeksais vėliau). Šaltinio vienetas be išėjimų —
        // variklis jo nelaiko; spragas (INVALID iki pirmo kadro, K3) commit'ina pats rašytojas (pace_copy). Variklio praleisti tick'ai (T11) tik
        // pažadina vėliau: vėlavimo taisyklę kiekvienam rašytojui taiko tick() (pace_advance).
        eng::EngineConfig cfg = eng::EngineConfig::legacy({rate.numerator, rate.denominator}, 0, 0);
        cfg.cuts.expect_ns = offset;
        cfg.start = eng::StartMode::Immediate;
        cfg.warmup_s = 0;
        cfg.out_fail_fatal = 0;
        if (wait_hook)
            cfg.open_wait = [this](eng::Index i, std::int64_t off, const std::atomic<bool>& s) {
                if (wait_hook(i, off, s)) back.store(true);  // testo laiko šaltinis pats praneša apie šuolį atgal
                if (!s.load()) hclock->reached(mxlIndexToTimestamp(&rate, i) + static_cast<std::uint64_t>(off));
                return true;
            };
        eng::StatsHub hub;
        eng::TickEngine e(cfg, *clock, hub, stop);
        eng::UnitSpec u;
        u.name = "pace";
        std::uint32_t id = 0;
        std::string why;
        if (!e.add_unit(u, id, why)) throw std::runtime_error("add_unit: " + why);
        PacerFn fn(*this);
        const eng::RunResult res = e.run(fn);
        if (res.reason == eng::EndReason::Fatal) throw std::runtime_error(res.fatal);
    } catch (const std::exception& ex) {
        kill_all(ex.what());
    } catch (...) {
        kill_all(nullptr);
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

// Įstoja į grupę (rašytojas su laiko kabliukais — į savo naują). Kviečiama po op_m, pacing == true, pace_next nustatytas.
void join_pacing(vfgfx_writer w) {
    const bool own = static_cast<bool>(w->hooks.now) || static_cast<bool>(w->hooks.wait);
    const std::int64_t off = w->offset_ns.load();
    PaceRegistry& reg = registry();
    std::lock_guard<std::mutex> rl(reg.m);
    std::shared_ptr<PaceGroup> g;
    if (!own)
        for (const auto& x : reg.groups)
            if (!x->owner && x->rate.numerator == w->rate.numerator && x->rate.denominator == w->rate.denominator &&
                x->offset == off && x->device == w->device) {
                g = x;
                break;
            }
    const bool fresh = !g;
    if (fresh) {
        g = std::make_shared<PaceGroup>();
        g->rate = w->rate;
        g->offset = off;
        g->device = w->device;
        if (own) {
            g->owner = w;
            g->wait_hook = w->hooks.wait;
            auto hc = std::make_unique<HookClock>(w->hooks.now);
            g->hclock = hc.get();
            g->clock = std::move(hc);
        } else {
            g->clock = std::make_unique<eng::SystemClock>();
        }
    }
    {
        std::lock_guard<std::mutex> gl(g->m);
        g->members.insert(std::lower_bound(g->members.begin(), g->members.end(), w), w);
        w->pace_dead = false;
    }
    w->group = g;
    if (!fresh) return;
    reg.groups.push_back(g);
    try {
        g->th = std::thread([p = g.get()] { p->run(); });
    } catch (...) {
        reg.groups.pop_back();
        w->group.reset();
        throw;
    }
}

// Palieka grupę; paskutinis narys sustabdo ir sujungia jos giją. Grįžus gija šio rašytojo nebeliečia.
void leave_pacing(vfgfx_writer w) {
    std::shared_ptr<PaceGroup> g = std::move(w->group);
    w->group.reset();
    if (!g) return;
    bool last = false;
    {
        PaceRegistry& reg = registry();
        std::lock_guard<std::mutex> rl(reg.m);
        {
            std::lock_guard<std::mutex> gl(g->m);
            g->members.erase(std::remove(g->members.begin(), g->members.end(), w), g->members.end());
            last = g->members.empty();
        }
        if (last) reg.groups.erase(std::remove(reg.groups.begin(), reg.groups.end(), g), reg.groups.end());
    }
    if (last) {
        g->stop.store(true);
        if (g->th.joinable()) g->th.join();
    }
    w->out.cancel();  // kaip anksčiau gijos pabaigoje (atviro slot'o po tick'o nelieka)
}

}  // namespace

vfgfx_writer_s::~vfgfx_writer_s() {
    if (group) leave_pacing(this);
    out.release(inst);
    free_all();
}

template <class F>
static int guarded(vfgfx_writer w, F&& f) noexcept {
    try {
        const int rc = f();
        if (rc == 0) {  // sėkmingas kvietimas išvalo atitinkamą klaidą (gfx.h „ERRORS“)
            if (w) w->clear_error();
            else g_create_error.clear();
        }
        return rc;
    } catch (const std::exception& e) {
        try {
            if (w) w->fail(VFGFX_ERR_INTERNAL, std::string("internal exception: ") + e.what());
            else g_create_error = std::string("internal exception: ") + e.what();
        } catch (...) {
        }
    } catch (...) {
    }
    return VFGFX_ERR_INTERNAL;
}

// --- C ABI -------------------------------------------------------------------------------------------------------------
extern "C" {

static int create_impl(mxlInstance inst, const char* flow_def_json, const char* writer_options_json, vfgfx_writer* out) {
    auto fail = [&](int code, const std::string& m) {
        g_create_error = m;
        return code;
    };
    if (out) *out = nullptr;
    if (!inst || !flow_def_json || !out) return fail(VFGFX_ERR_INVALID_ARG, "inst, flow_def_json and out must not be NULL");
    const std::string def = flow_def_json;
    const std::string opts = writer_options_json ? writer_options_json : "";
    vramflow::VideoFormat fmt{};
    std::string err;
    if (!vramflow::format_from_flow_def(def, fmt, err))
        return fail(VFGFX_ERR_INVALID_ARG, "invalid flow definition (expected MXL video flow JSON with media_type, frame_width, "
                                           "frame_height, interlace_mode progressive|interlaced_tff and grain_rate)");
    const gfx::Raster r = fmt.raster;
    // dažnis — bet kuris palaikomas (format.hpp); rastras — bet koks (grafikos sluoksniai)
    if (!vramflow::rate_supported(r.interlaced, fmt.frame_rate, err)) return fail(VFGFX_ERR_GEOMETRY, rate_msg(fmt));
    if (!r.alpha) return fail(VFGFX_ERR_INVALID_ARG, "flow media_type must be video/v210a");
    if (!gfx::raster_ok(r)) return fail(VFGFX_ERR_GEOMETRY, "invalid raster (width and, for interlaced flows, height must be even)");

    auto w = std::make_unique<vfgfx_writer_s>();
    w->inst = inst;
    w->r = r;
    w->fmt = fmt;
    w->rate = func::mxl_rate(fmt.grain_rate());
    w->fields = r.interlaced ? 2 : 1;
    gfx::detail::json_str(def, "id", w->flow_id);
    // deviceUuid nurodytas, bet tokio įrenginio nėra: aiški klaida, ne tylus kitas įrenginys. Ieškoma vieną kartą per create.
    const int uuid_dev = device_from_options(opts);
    {
        std::string u;
        if (gfx::detail::json_str(opts, "deviceUuid", u) && !u.empty() && uuid_dev < 0)
            return fail(VFGFX_ERR_INVALID_ARG, "deviceUuid " + u + " matches no CUDA device");
    }
    if (!w->out.create(inst, def, opts, err, fmt)) {
        const int code = w->out.create_status != MXL_STATUS_OK ? static_cast<int>(w->out.create_status)
                         : w->out.existed                       ? VFGFX_ERR_STATE
                                                                : VFGFX_ERR_GEOMETRY;
        w->out.release(inst);
        const std::string m = w->out.create_status != MXL_STATUS_OK
                                  ? "mxlCreateFlowWriter failed, status " + std::to_string(w->out.create_status)
                              : w->out.existed ? "flow already exists in this domain"
                                               : "flow storage layout or grain rate does not match the v210a definition";
        return fail(code, m);
    }
    w->cuda_store = mxlv::is_cuda_layout(w->out.layout);
    w->cuda_used.store(w->cuda_store);
    if (!w->cuda_store) w->conv_dev = uuid_dev;
    if (w->cuda_store) {
        // Be deviceUuid — įrenginys 0 (biblioteka turi privatų cudart, kviečiančiojo cudaSetDevice jai nematomas; gfx.h).
        int n = 0;
        if (uuid_dev < 0 && (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)) {
            cudaGetLastError();
            w->out.release(inst);
            return fail(VFGFX_ERR_CUDA, "no CUDA device");
        }
        w->device = uuid_dev >= 0 ? uuid_dev : 0;
        DeviceScope ds(w->device);
        if (cudaStreamCreateWithFlags(&w->s_conv, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            w->free_all();
            w->out.release(inst);
            return fail(VFGFX_ERR_CUDA, "cudaStreamCreate");
        }
    }  // host saugykla: nė vieno CUDA kvietimo (s_conv sukuriamas tingiai convert(), tik CUDA kadrui)
    if (const int c = w->alloc_sets(2)) {  // submit aibės; submit_at eilės aibės — tingiai (pirmas submit_at)
        const std::string m = w->err;
        w->free_all();
        w->out.release(inst);
        return fail(c, m);
    }
    *out = w.release();
    return 0;
}

static int write_impl(vfgfx_writer w, std::uint64_t index, const vfgfx_frame* f) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> l(w->op_m);
    if (w->pacing.load()) return w->fail(VFGFX_ERR_STATE, "vfgfx_write is not allowed while pacing runs");
    vfgfx_frame fr;
    if (const int c = w->read_frame(f, fr)) return c;
    mxlStatus st = MXL_STATUS_OK;
    std::uint32_t* dst = w->out.open(index, st);
    if (!dst) return w->fail(static_cast<int>(st), "OpenGrainSlot " + std::to_string(index) + " status " + std::to_string(st));
    // Interlaced: grain'as = kadro laukas index%2; progresyvus: visas kadras (paritetas ignoruojamas).
    struct CancelGuard {  // išimtis po open: atidarytas grain'as atšaukiamas (po sėkmingo commit'o cancel() nieko nedaro)
        func::VideoOut& o;
        ~CancelGuard() { o.cancel(); }
    } cancel_guard{w->out};
    const int par[1] = {w->r.interlaced ? static_cast<int>(index & 1) : 0};
    std::uint8_t* const d[1] = {reinterpret_cast<std::uint8_t*>(dst)};
    if (const int c = w->convert(fr, 1, par, d)) {
        w->out.cancel();
        return c;
    }
    const int c = w->commit_open();
    if (c) w->out.cancel();
    else w->n_written.fetch_add(1);
    return c;
}

static int start_pacing_impl(vfgfx_writer w, std::int64_t offset_ns) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> l(w->op_m);
    if (w->pacing.load()) return w->fail(VFGFX_ERR_STATE, "pacing is already running");
    w->offset_ns.store(offset_ns);
    {
        std::lock_guard<std::mutex> ll(w->lat_m);  // submit_at mato nuoseklią porą (pacing, pace_next)
        w->pace_next = w->start_index();
        w->pacing.store(true);
        w->collapse_past();  // eilė išlieka per stop / start; praėję įrašai sutraukiami
    }
    try {
        join_pacing(w);
    } catch (...) {
        w->pacing.store(false);
        return w->fail(VFGFX_ERR_STATE, "could not start the pacing thread");
    }
    return 0;
}

// Konvertuoja kadrą į laisvą aibę (po sub_m). Grąžina aibę arba < 0 (klaida jau užfiksuota, kodas -rc).
static int prepare_set(vfgfx_writer w, const vfgfx_frame& fr) {
    int target;
    {
        std::lock_guard<std::mutex> l(w->lat_m);
        target = w->free_set();
    }
    if (target < 0 && w->nsets < kSets) {  // aibės išskiriamos pagal poreikį: viena kiekvienam laukiančiam kadrui
        if (const int c = w->alloc_sets(w->nsets + 1)) return -c;
        target = w->nsets - 1;  // nauja aibė — niekieno nenaudojama (nsets keičia tik sub_m turėtojas)
    }
    if (target < 0) return -w->fail(VFGFX_ERR_INTERNAL, "no free prepared frame set");
    const int par[2] = {0, 1};
    std::uint8_t* const d[2] = {w->prep[target][0], w->prep[target][1]};
    if (const int c = w->convert(fr, w->fields, par, d)) return -c;
    return target;
}

static int submit_impl(vfgfx_writer w, const vfgfx_frame* f) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> sl(w->sub_m);
    vfgfx_frame fr;
    if (const int c = w->read_frame(f, fr)) return c;
    // Laisvą aibę užimti gali tik sub_m turėtojas; ritmo gija skaito tik w->cur (po lat_m).
    const int target = prepare_set(w, fr);
    if (target < 0) return -target;
    std::lock_guard<std::mutex> l(w->lat_m);
    w->cur = target;
    w->has = true;
    w->cancel_all();  // "nuo dabar šis kadras": suplanuoti kadrai atšaukiami
    w->n_submitted.fetch_add(1);
    return 0;
}

static int submit_at_impl(vfgfx_writer w, std::uint64_t first, const vfgfx_frame* f) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> sl(w->sub_m);
    vfgfx_frame fr;
    if (const int c = w->read_frame(f, fr)) return c;
    // Riba į ateitį: VFGFX_SUBMIT_AT_MAX_LEAD_S sekundžių grain'ų (apvalinant aukštyn).
    const auto num = static_cast<std::uint64_t>(w->rate.numerator), den = static_cast<std::uint64_t>(w->rate.denominator);
    const std::uint64_t lead = (static_cast<std::uint64_t>(VFGFX_SUBMIT_AT_MAX_LEAD_S) * num + den - 1) / den;
    auto too_late = [&](std::uint64_t min) {
        return w->fail(VFGFX_ERR_TOO_LATE, "vfgfx_submit_at: first_index " + std::to_string(first) + " < next_pacing_index " +
                                               std::to_string(min));
    };
    if (w->r.interlaced && (first & 1))
        return w->fail(VFGFX_ERR_INVALID_ARG, "vfgfx_submit_at: first_index " + std::to_string(first) +
                                                  " is odd; on an interlaced flow a frame must start on a top field (even index)");
    {
        std::lock_guard<std::mutex> l(w->lat_m);
        w->collapse_past();
        const std::uint64_t min = w->min_scheduled_index();
        if (first < min) return too_late(min);
        const std::uint64_t now = w->current_index();
        if (first > now + lead)
            return w->fail(VFGFX_ERR_TOO_EARLY, "vfgfx_submit_at: first_index " + std::to_string(first) + " > current index " +
                                                    std::to_string(now) + " + " + std::to_string(lead) + " (" +
                                                    std::to_string(VFGFX_SUBMIT_AT_MAX_LEAD_S) + " s)");
        const bool same = std::any_of(w->queue.begin(), w->queue.end(), [&](const Pending& p) { return p.first == first; });
        if (!same && static_cast<int>(w->queue.size()) >= kQueueCap)
            return w->fail(VFGFX_ERR_QUEUE_FULL, "vfgfx_submit_at: queue full (" + std::to_string(kQueueCap) + " pending frames)");
    }
    const int target = prepare_set(w, fr);
    if (target < 0) return -target;
    if (w->hooks.after_convert) w->hooks.after_convert();  // testo kabliukas (lenktynės su ritmo gija)
    std::lock_guard<std::mutex> l(w->lat_m);
    w->collapse_past();
    const std::uint64_t min = w->min_scheduled_index();  // konvertuojant ritmo gija galėjo pasirinkti šio indekso turinį
    if (first < min) return too_late(min);
    auto it = std::lower_bound(w->queue.begin(), w->queue.end(), first, [](const Pending& p, std::uint64_t v) { return p.first < v; });
    if (it != w->queue.end() && it->first == first) {
        it->set = target;  // senoji aibė tampa laisva
        w->n_sched_replaced.fetch_add(1);
    } else {
        w->queue.insert(it, Pending{first, target});
    }
    w->n_scheduled.fetch_add(1);
    return 0;
}

static int stop_pacing_impl(vfgfx_writer w) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> l(w->op_m);
    if (!w->pacing.load()) return 0;
    leave_pacing(w);  // sinchroniškai: grįžus grupės gija šio rašytojo nebeliečia
    w->pacing.store(false);
    return 0;
}

// Skaitymo funkcija: tik grąžinimo kodas, vfgfx_last_error ir caller_errors nekeičiami (stebėjimo gija neturi
// išvalyti / perrašyti kviečiančiosios klaidos, peržiūros 5 punktas).
static int get_stats_impl(vfgfx_writer w, vfgfx_stats* out) {
    if (!w || !out) return VFGFX_ERR_INVALID_ARG;
    // Išvesties struktūra: tikrinamas tik dydis ir flags; nežinoma uodega (naujesnė antraštė) nunulinama.
    const std::uint32_t sz = struct_size_of(out);
    const std::size_t hdr = offsetof(vfgfx_stats, committed), min = hdr + sizeof(std::uint64_t);
    if (sz < min || sz > kMaxStruct) return VFGFX_ERR_STRUCT_SIZE;
    if (out->flags) return VFGFX_ERR_UNSUPPORTED;
    vfgfx_stats s;
    std::memset(static_cast<void*>(&s), 0, sizeof s);
    {
        std::lock_guard<std::mutex> l(w->st_m);
        s.paced = w->stats.committed;
        s.late_skips = w->stats.late_skips;
        s.no_frame = w->stats.no_frame;
        s.clock_backward = w->stats.clock_backward;
        s.pacing_errors = w->stats.errors;
    }
    s.committed = w->committed.load();
    s.written = w->n_written.load();
    s.submitted = w->n_submitted.load();
    s.scheduled = w->n_scheduled.load();
    s.scheduled_late = w->n_sched_late.load();
    s.scheduled_replaced = w->n_sched_replaced.load();
    s.caller_errors = w->n_caller_errors.load();
    s.last_committed_index = w->last_index.load();
    {
        std::lock_guard<std::mutex> l(w->lat_m);
        s.next_pacing_index = w->min_scheduled_index();
        const std::uint64_t min = s.next_pacing_index;  // tik būsimi (praėję — jau dabartinis turinys)
        s.queue_depth = static_cast<std::uint32_t>(
            std::count_if(w->queue.begin(), w->queue.end(), [min](const Pending& p) { return p.first >= min; }));
    }
    s.pacing = w->pacing.load() ? 1u : 0u;
    s.queue_capacity = kQueueCap;
    // Laukai po antraštės (struct_size, flags nekeičiami) iki kviečiančiojo dydžio; didesnės struktūros uodega = 0.
    auto* o = reinterpret_cast<unsigned char*>(out);
    std::memcpy(o + hdr, reinterpret_cast<const unsigned char*>(&s) + hdr, std::min<std::size_t>(sz, sizeof s) - hdr);
    if (sz > sizeof s) std::memset(o + sizeof s, 0, sz - sizeof s);
    return 0;
}

static int flow_def_impl(const vfgfx_flow_params* p, char* buf, std::size_t buf_size, std::size_t* required) {
    auto fail = [&](int code, const std::string& m) {
        g_create_error = m;
        return code;
    };
    if (!p) return fail(VFGFX_ERR_INVALID_ARG, "vfgfx_flow_def: p nullptr");
    vfgfx_flow_params q;
    std::string msg;
    if (const int c = read_struct(p, offsetof(vfgfx_flow_params, frame_rate_den) + sizeof(std::int64_t), "vfgfx_flow_params", q, msg))
        return fail(c, msg);
    if (q.flags) return fail(VFGFX_ERR_UNSUPPORTED, "unknown vfgfx_flow_params.flags " + std::to_string(q.flags));
    if (q.reserved0 || !all_zero(q.reserved)) return fail(VFGFX_ERR_UNSUPPORTED, "vfgfx_flow_params.reserved must be zero");
    if (!q.flow_id || !is_uuid(q.flow_id))
        return fail(VFGFX_ERR_INVALID_ARG, "flow_id must be a UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)");
    const std::string label = json_escape(q.label ? q.label : "vramflow-gfx");
    const std::string hint = json_escape(q.group_hint ? std::string(q.group_hint) : std::string("vramflow-gfx-") + q.flow_id + ":Video");
    if (label.size() > 256 || hint.size() > 256) return fail(VFGFX_ERR_INVALID_ARG, "label or group_hint longer than 256 bytes (JSON-escaped)");
    vramflow::VideoFormat f{};
    std::string err;
    if (q.format) {
        if (q.width || q.height || q.scan || q.frame_rate_num || q.frame_rate_den)
            return fail(VFGFX_ERR_INVALID_ARG, "format and width/height/scan/frame_rate are mutually exclusive");
        if (!vramflow::parse_format(q.format, f, err))
            return fail(VFGFX_ERR_GEOMETRY, std::string("unsupported format name '") + q.format +
                                                "' (supported: 720p25|50|29.97|59.94, 1080i50|59.94, 1080p25|50|29.97|59.94, "
                                                "2160p25|50|29.97|59.94)");
    } else {
        if (q.scan != VFGFX_SCAN_PROGRESSIVE && q.scan != VFGFX_SCAN_INTERLACED_TFF)
            return fail(VFGFX_ERR_UNSUPPORTED, "unsupported vfgfx_flow_params.scan " + std::to_string(q.scan));
        if (q.width > 16384 || q.height > 16384) return fail(VFGFX_ERR_GEOMETRY, "raster larger than 16384");
        f.raster = gfx::Raster{static_cast<int>(q.width), static_cast<int>(q.height), q.scan == VFGFX_SCAN_INTERLACED_TFF, true};
        if (!gfx::raster_ok(f.raster)) return fail(VFGFX_ERR_GEOMETRY, "invalid raster (width and height > 0; width and, for interlaced flows, height even)");
        if (q.frame_rate_num <= 0 || q.frame_rate_den <= 0) return fail(VFGFX_ERR_GEOMETRY, "frame rate must be > 0");
        f.frame_rate = vramflow::Rational{q.frame_rate_num, q.frame_rate_den};
        if (!vramflow::rate_supported(f.raster.interlaced, f.frame_rate, err)) return fail(VFGFX_ERR_GEOMETRY, rate_msg(f));
    }
    f.raster.alpha = true;  // video/v210a
    const std::string json = vramflow::video_flow_def(q.flow_id, label, f, hint);
    if (json.size() >= 2047) return fail(VFGFX_ERR_INTERNAL, "flow definition too long");
    const std::size_t need = json.size() + 1;
    if (required) *required = need;
    if (!buf && buf_size == 0) return 0;
    if (!buf) return fail(VFGFX_ERR_INVALID_ARG, "buf is NULL but buf_size > 0");
    if (buf_size < need) {
        buf[0] = '\0';
        return fail(VFGFX_ERR_BUFFER_TOO_SMALL, "buf_size " + std::to_string(buf_size) + " < " + std::to_string(need));
    }
    std::memcpy(buf, json.c_str(), need);
    return 0;
}

static void destroy_impl(vfgfx_writer w) {
    if (!w) return;
    stop_pacing_impl(w);
    delete w;  // ~vfgfx_writer_s: release(inst) + free_all
}

// --- Eksportuojami apvalkalai: jokia C++ išimtis neperženg C ABI ribos --------------------------------------------------
uint32_t vfgfx_version(void) { return VFGFX_VERSION; }
const char* vfgfx_version_string(void) { return VFGFX_VERSION_STRING; }

int vfgfx_flow_def(const vfgfx_flow_params* p, char* buf, size_t buf_size, size_t* required) {
    return guarded(nullptr, [&] { return flow_def_impl(p, buf, buf_size, required); });
}
// Ar kvietimas gali liesti CUDA kviečiančiojo gijoje (tada saugomas jos kontekstas).
static bool cuda_call(vfgfx_writer w, const vfgfx_frame* f) {
    if (w && w->cuda_used.load()) return true;
    if (!f || struct_size_of(f) < offsetof(vfgfx_frame, mem) + sizeof f->mem) return false;
    return f->mem == VFGFX_MEM_CUDA;
}

int vfgfx_writer_create(mxlInstance inst, const char* flow_def_json, const char* writer_options_json, vfgfx_writer* out) {
    const CtxGuard ctx(writer_options_json && std::strstr(writer_options_json, "cuda-linear"));
    const int rc = guarded(nullptr, [&] { return create_impl(inst, flow_def_json, writer_options_json, out); });
    if (rc != 0 && out) *out = nullptr;  // išimtis po dalinio sukūrimo: nelaikom pusiau sukurto writer'io
    return rc;
}
int vfgfx_write(vfgfx_writer w, uint64_t index, const vfgfx_frame* f) {
    const CtxGuard ctx(cuda_call(w, f));
    return guarded(w, [&] { return write_impl(w, index, f); });
}
int vfgfx_start_pacing(vfgfx_writer w, int64_t offset_ns) {
    return guarded(w, [&] { return start_pacing_impl(w, offset_ns); });
}
int vfgfx_submit(vfgfx_writer w, const vfgfx_frame* f) {
    const CtxGuard ctx(cuda_call(w, f));
    return guarded(w, [&] { return submit_impl(w, f); });
}
int vfgfx_submit_at(vfgfx_writer w, uint64_t first_index, const vfgfx_frame* f) {
    const CtxGuard ctx(cuda_call(w, f));
    return guarded(w, [&] { return submit_at_impl(w, first_index, f); });
}
int vfgfx_stop_pacing(vfgfx_writer w) {
    return guarded(w, [&] { return stop_pacing_impl(w); });
}
int vfgfx_get_stats(vfgfx_writer w, vfgfx_stats* out) {
    try {
        return get_stats_impl(w, out);
    } catch (...) {
        return VFGFX_ERR_INTERNAL;
    }
}
int vfgfx_cancel_scheduled(vfgfx_writer w) {
    return guarded(w, [&] {
        if (!w) return VFGFX_ERR_INVALID_ARG;
        std::lock_guard<std::mutex> l(w->lat_m);
        w->collapse_past();  // praėję įrašai jau yra dabartinis turinys — jie neatšaukiami
        w->cancel_all();
        return 0;
    });
}
uint64_t vfgfx_index_time_ns(vfgfx_writer w, uint64_t index) { return w ? mxlIndexToTimestamp(&w->rate, index) : 0; }
int vfgfx_get_format(vfgfx_writer w, vfgfx_flow_params* out) {
    // Skaitymo funkcija (kaip vfgfx_get_stats): vfgfx_last_error nekeičiamas.
    if (!w || !out) return VFGFX_ERR_INVALID_ARG;
    const std::uint32_t sz = struct_size_of(out);
    const std::size_t hdr = offsetof(vfgfx_flow_params, flow_id), min = offsetof(vfgfx_flow_params, frame_rate_den) + sizeof(std::int64_t);
    if (sz < min || sz > kMaxStruct) return VFGFX_ERR_STRUCT_SIZE;
    if (out->flags) return VFGFX_ERR_UNSUPPORTED;
    vfgfx_flow_params q;
    std::memset(static_cast<void*>(&q), 0, sizeof q);
    q.flow_id = w->flow_id.c_str();
    q.width = static_cast<std::uint32_t>(w->r.w);
    q.height = static_cast<std::uint32_t>(w->r.h);
    q.scan = w->r.interlaced ? VFGFX_SCAN_INTERLACED_TFF : VFGFX_SCAN_PROGRESSIVE;
    q.frame_rate_num = w->fmt.frame_rate.num;
    q.frame_rate_den = w->fmt.frame_rate.den;
    auto* o = reinterpret_cast<unsigned char*>(out);
    std::memcpy(o + hdr, reinterpret_cast<const unsigned char*>(&q) + hdr, std::min<std::size_t>(sz, sizeof q) - hdr);
    if (sz > sizeof q) std::memset(o + sizeof q, 0, sz - sizeof q);
    return 0;
}
uint64_t vfgfx_committed(vfgfx_writer w) { return w ? w->committed.load() : 0; }
uint64_t vfgfx_current_index(vfgfx_writer w) {
    if (!w) return VFGFX_INDEX_NONE;
    try {
        return w->current_index();
    } catch (...) {
        return VFGFX_INDEX_NONE;
    }
}

const char* vfgfx_last_error(vfgfx_writer w) {
    // Rodyklė — thread_local kopija: galioja iki kito vfgfx_last_error / vfgfx_pacing_error iškvietimo TOJE PAČIOJE gijoje.
    static thread_local std::string copy;
    try {
        if (!w) {
            copy = g_create_error;
        } else {
            std::lock_guard<std::mutex> l(w->err_m);
            copy = w->err;
        }
        return copy.c_str();
    } catch (...) {
        return "";
    }
}

const char* vfgfx_pacing_error(vfgfx_writer w) {
    static thread_local std::string copy;
    try {
        if (!w) return "";
        std::lock_guard<std::mutex> l(w->err_m);
        copy = w->perr;
        return copy.c_str();
    } catch (...) {
        return "";
    }
}

void vfgfx_writer_destroy(vfgfx_writer w) {
    const CtxGuard ctx(cuda_call(w, nullptr));
    try {
        destroy_impl(w);
    } catch (...) {
    }
}

}  // extern "C"

namespace vramflow::gfx::internal {

int set_pacing_hooks(vfgfx_writer w, PacingHooks hooks) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> l(w->op_m);
    if (w->pacing.load()) return VFGFX_ERR_STATE;
    w->hooks = std::move(hooks);
    return 0;
}

std::string pacing_error(vfgfx_writer w) {
    std::lock_guard<std::mutex> l(w->err_m);
    return w->perr;
}

int prepared_sets(vfgfx_writer w) { return w->nsets; }

PacingStats pacing_stats(vfgfx_writer w) {
    std::lock_guard<std::mutex> l(w->st_m);
    return w->stats;
}

}  // namespace vramflow::gfx::internal
