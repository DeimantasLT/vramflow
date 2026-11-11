// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// libvramflow-gfx (include/vramflow/gfx.h): RGBA/BGRA -> standartinis MXL video/v210a flow, rankinis ir ritmo režimai.
//
// Konvertavimas: cuda-linear saugykla -> CUDA kernel'is (gfx::cuda), host saugykla -> CPU (gfx::cpu); rezultatas
// bitas į bitą lygus gfx::ref etalonui. Ritmo režimas: vfgfx_submit / vfgfx_submit_at iškart konvertuoja kadrą į du
// paruoštus laukus (paritetai 0 ir 1; progresyviam rastrui — vieną grain'ą) vienoje iš paruoštų aibių; ritmo gija kas
// indeksą (func::wait_until) tik KOPIJUOJA „dabartinės“ aibės atitinkamo pariteto lauką į slot'ą ir commit'ina (be
// pakartotinio konvertavimo). Aibės: 2 (submit) + kQueueCap (submit_at eilė; išskiriamos tingiai, pirmą kartą
// prireikus). Laisva aibė = ne dabartinė ir ne eilėje; konvertuojama tik į laisvą (po sub_m — vienu metu vienas
// konvertuotojas), o dabartinę / eilę po lat_m keičia submit, submit_at ir ritmo gija (aktyvuoja eilės įrašus), todėl
// submit niekada nerašo į aibę, iš kurios ritmo gija skaito.
//
// Vėlavimo taisyklė (kaip func::gen): gija pabudo vėlai -> indeksas, kurio laukia, VIS TIEK įrašomas (vėlyvas commit;
// skaitytojas sprendžia, ar per vėlu), o po commit'o, jei einamasis indeksas aplenkė kitą, jo laukai praleidžiami
// (late_skips) ir tęsiama nuo einamojo — jokio „catch-up“ srauto. Laikrodis šoko atgal -> wait_until palaukia, kol
// laikas pasivys (i tik didėja: tas pats indeksas du kartus neįrašomas, o kitą indeksą wait_until atidėja iki jo laiko).
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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/format.hpp"
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

namespace {

constexpr int kQueueCap = VFGFX_SUBMIT_AT_MIN_QUEUE;  // submit_at eilės talpa (stats.queue_capacity)
constexpr int kSets = 2 + kQueueCap;                  // paruoštų aibių: dabartinė + submit tikslas + eilė
constexpr std::uint32_t kMaxStruct = 4096;
constexpr std::uint32_t kFrameFlags = VFGFX_FRAME_CALLER_SYNCED;

// Kviečiančios gijos CUDA įrenginio perjungimas laikinai (atstatomas išeinant).
struct DeviceScope {
    int prev = -1;
    bool active = false;
    explicit DeviceScope(int dev) {
        if (dev < 0) return;
        if (cudaGetDevice(&prev) != cudaSuccess) { cudaGetLastError(); prev = -1; }
        if (prev != dev) {
            active = cudaSetDevice(dev) == cudaSuccess;
            if (!active) cudaGetLastError();
        }
    }
    ~DeviceScope() {
        if (active && prev >= 0) cudaSetDevice(prev);
    }
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
    return std::string(name) + ".struct_size " + std::to_string(sz) + " (leidžiama " + std::to_string(min) + "…" +
           std::to_string(kMaxStruct) + "; naudok _INIT makrokomandą)";
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
            msg = std::string(name) + ": nežinomi laukai už " + std::to_string(sizeof out) + " B nenuliniai (struct_size " +
                  std::to_string(sz) + "; biblioteka " VFGFX_VERSION_STRING " jų nepalaiko)";
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

struct Pending {  // submit_at eilės įrašas: aibė `set` tampa dabartine nuo grain'o `first`
    std::uint64_t first;
    int set;
};

}  // namespace

struct vfgfx_writer_s {
    mxlInstance inst = nullptr;
    func::VideoOut out;
    gfx::Raster r{};
    vramflow::VideoFormat fmt{};  // r + kadrų dažnis iš flow aprašo; ritmas — fmt.grain_rate()
    mxlRational rate{50, 1};
    bool cuda_store = false;
    int device = -1;       // cuda-linear saugyklos įrenginys; host saugykloje -1 (jokių CUDA kvietimų)
    int conv_dev = -1;     // host saugykla: įrenginys CUDA kadrų atsisiuntimui (deviceUuid arba -1 = numatytasis); s_conv kuriamas tik tada
    int fields = 1;  // paruoštų grain'ų skaičius vienoje aibėje (2 interlaced, 1 progresyviam)
    cudaStream_t s_conv = nullptr;  // create / write / submit (po conv_m); host saugykloje kuriamas tingiai (convert)
    cudaStream_t s_pace = nullptr;  // tik ritmo gija
    cudaEvent_t ev_caller = nullptr;  // cuda_stream keliui (po conv_m), kuriamas tingiai

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
    std::atomic<bool> stop{false};
    std::thread th;
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
        if (!f) return fail(VFGFX_ERR_INVALID_ARG, "kadras nullptr");
        std::string msg;
        if (const int c = read_struct(f, offsetof(vfgfx_frame, cuda_stream) + sizeof(void*), "vfgfx_frame", fr, msg))
            return fail(c, msg);
        if (fr.flags & ~kFrameFlags) return fail(VFGFX_ERR_UNSUPPORTED, "nežinomos vfgfx_frame.flags " + std::to_string(fr.flags));
        if (!all_zero(fr.reserved)) return fail(VFGFX_ERR_UNSUPPORTED, "vfgfx_frame.reserved turi būti nuliai");
        if (fr.pixel_format != VFGFX_PIXEL_RGBA8 && fr.pixel_format != VFGFX_PIXEL_BGRA8)
            return fail(VFGFX_ERR_UNSUPPORTED, "nepalaikomas pixel_format " + std::to_string(fr.pixel_format) +
                                                   " (palaikoma RGBA8 = 0, BGRA8 = 1)");
        if (!fr.pixels) return fail(VFGFX_ERR_INVALID_ARG, "kadro rodyklė (pixels) nullptr");
        if (fr.alpha != VFGFX_ALPHA_STRAIGHT && fr.alpha != VFGFX_ALPHA_PREMULTIPLIED)
            return fail(VFGFX_ERR_INVALID_ARG, "nežinomas vfgfx_alpha");
        if (fr.mem != VFGFX_MEM_HOST && fr.mem != VFGFX_MEM_CUDA) return fail(VFGFX_ERR_INVALID_ARG, "nežinomas vfgfx_mem");
        if (fr.cuda_stream && fr.mem != VFGFX_MEM_CUDA) return fail(VFGFX_ERR_INVALID_ARG, "cuda_stream galimas tik CUDA kadrui");
        if (fr.cuda_stream && (fr.flags & VFGFX_FRAME_CALLER_SYNCED))
            return fail(VFGFX_ERR_INVALID_ARG, "cuda_stream ir VFGFX_FRAME_CALLER_SYNCED kartu — prieštaringa");
        if (fr.width != static_cast<std::uint32_t>(r.w) || fr.height != static_cast<std::uint32_t>(r.h))
            return fail(VFGFX_ERR_GEOMETRY, "kadras " + std::to_string(fr.width) + "x" + std::to_string(fr.height) + " != flow " +
                                                std::to_string(r.w) + "x" + std::to_string(r.h));
        if (fr.stride < fr.width * 4u) return fail(VFGFX_ERR_GEOMETRY, "stride < width*4");
        if (fr.mem == VFGFX_MEM_CUDA && ((fr.stride & 3u) || (reinterpret_cast<std::uintptr_t>(fr.pixels) & 3u)))
            return fail(VFGFX_ERR_GEOMETRY, "CUDA kadro rodyklė ir stride turi būti kartotiniai 4 B");
        if (cuda_store && fr.mem == VFGFX_MEM_HOST && (fr.stride & 3u))
            return fail(VFGFX_ERR_GEOMETRY, "stride turi būti kartotinis 4 B (cuda-linear saugykla)");
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
            if (e != cudaSuccess) return fail_cuda("kviečiančiojo cuda_stream įvykis", e);
            return 0;
        }
        // R20: visų konteksto darbų laukimas (pirminis kontekstas bendras visiems proceso CUDA runtime'ams).
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) return fail_cuda("laukimas CUDA kadro (R20)", e);
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
            if (e != cudaSuccess) return fail_cuda("sinchronizavimas", e);
            return 0;
        }
        const std::uint8_t* src = static_cast<const std::uint8_t*>(f.pixels);
        if (f.mem == VFGFX_MEM_CUDA) {
            // Host saugyklos rašytuvas CUDA kontekstą kuria TIK čia, kai pirmą kartą ateina CUDA kadras (po conv_m);
            // host kadrų rašytuvas niekada neliečia CUDA (veikia ir be GPU / su CUDA_VISIBLE_DEVICES="").
            DeviceScope cd(conv_dev);
            if (!s_conv) {
                if (const cudaError_t e = cudaStreamCreateWithFlags(&s_conv, cudaStreamNonBlocking); e != cudaSuccess) {
                    s_conv = nullptr;
                    return fail_cuda("cudaStreamCreate (CUDA kadro atsisiuntimui)", e);
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
                return fail(VFGFX_ERR_GEOMETRY, "CPU rgba_to_v210a atmetė užduotį");
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

    ~vfgfx_writer_s() {  // visi keliai (įskaitant create išimtis) išvalo viską; release/free_all idempotentiški
        stop.store(true);
        if (th.joinable()) th.join();
        out.release(inst);
        free_all();
    }

    std::uint64_t now() const { return hooks.now ? hooks.now() : func::now_tai(); }
    std::uint64_t current_index() const { return mxlTimestampToIndex(&rate, now()); }
    // Indeksas, kurio commit'o laikas (indekso laikas + offset_ns) jau atėjo: index_of(now − offset_ns), saugu be ženklo.
    std::uint64_t due_index() const {
        std::uint64_t t = now();
        const std::int64_t off = offset_ns.load();
        if (off >= 0) {
            const std::uint64_t o = static_cast<std::uint64_t>(off);
            t = t > o ? t - o : 0;
        } else {
            t += 0 - static_cast<std::uint64_t>(off);  // beženklė neigimas: saugu ir INT64_MIN
        }
        return mxlTimestampToIndex(&rate, t);
    }
    // Indeksas, nuo kurio ritmas prasidėtų dabar: pirmas, kurio laikas dar nepraėjo; niekada ne <= jau commit'into.
    std::uint64_t start_index() const {
        std::uint64_t i = due_index() + 1;
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
        for (; nsets < n; ++nsets)
            for (int p = 0; p < fields; ++p) {
                void* ptr = nullptr;
                if (cuda_store) {
                    if (const cudaError_t e = cudaMalloc(&ptr, gb); e != cudaSuccess) return fail_cuda("cudaMalloc paruoštam laukui", e);
                } else {
                    ptr = std::aligned_alloc(64, (gb + 63) / 64 * 64);
                    if (!ptr) return fail(VFGFX_ERR_INTERNAL, "nepavyko išskirti paruošto lauko (host atmintis)");
                }
                prep[nsets][p] = static_cast<std::uint8_t*>(ptr);
            }
        return 0;
    }
    // Eilės įrašų, kurių first <= i, aktyvavimas (ritmo gija, po lat_m). Keli tuo pačiu metu — laimi didžiausias first.
    void activate_due(std::uint64_t i) {
        while (!queue.empty() && queue.front().first <= i) {
            const Pending p = queue.front();
            queue.erase(queue.begin());
            if (!queue.empty() && queue.front().first <= i) {
                n_sched_replaced.fetch_add(1);  // jį iškart pakeičia vėlesnis (jo laikas taip pat atėjo)
                continue;
            }
            if (p.first < i) n_sched_late.fetch_add(1);
            cur = p.set;
            has = true;
        }
    }

    void pace_loop();
    void pace_loop_body();
    void free_all();
};

void vfgfx_writer_s::pace_loop() {
    try {
        pace_loop_body();
    } catch (const std::exception& e) {
        set_pace_error(std::string("ritmo gija nutrūko: ") + e.what());
        bump(&gi::PacingStats::errors);
    } catch (...) {
        set_pace_error("ritmo gija nutrūko: nežinoma išimtis");
        bump(&gi::PacingStats::errors);
    }
    out.cancel();
}

void vfgfx_writer_s::pace_loop_body() {
    DeviceScope dev(device);  // gija nauja: numatytasis įrenginys 0; čia įrenginys pastovus iki gijos pabaigos
    std::uint64_t i = 0;
    {
        std::lock_guard<std::mutex> l(lat_m);
        i = pace_next;  // start_pacing: start_index() (restartas niekada nekartoja commit'into indekso)
    }
    const std::size_t gb = gfx::grain_bytes(r);
    while (!stop.load()) {
        const bool back = hooks.wait ? hooks.wait(i, offset_ns.load(), stop) : func::wait_until(rate, i, offset_ns.load(), stop);
        if (stop.load()) break;
        if (back) bump(&gi::PacingStats::clock_backward);
        bool wrote = false;
        {
            std::lock_guard<std::mutex> l(lat_m);
            activate_due(i);
            pace_next = i + 1;  // nuo šiol submit_at indeksui i — per vėlu (turinys pasirinktas)
            if (!has) {
                bump(&gi::PacingStats::no_frame);
            } else {
                mxlStatus st = MXL_STATUS_OK;
                std::uint32_t* dst = out.open(i, st);
                if (!dst) {
                    bump(&gi::PacingStats::errors);
                    set_pace_error("OpenGrainSlot " + std::to_string(i) + " status " + std::to_string(st));
                } else {
                    const std::uint8_t* src = prep[cur][r.interlaced ? (i & 1) : 0];
                    bool ok = true;
                    if (cuda_store) {
                        cudaError_t e = cudaMemcpyAsync(dst, src, gb, cudaMemcpyDeviceToDevice, s_pace);
                        if (e == cudaSuccess) e = cudaStreamSynchronize(s_pace);
                        if (e != cudaSuccess) {
                            cudaGetLastError();
                            ok = false;
                            set_pace_error(std::string("D2D: ") + cudaGetErrorString(e));
                        }
                    } else {
                        std::memcpy(dst, src, gb);
                    }
                    if (ok && commit_open(true) == 0) {
                        wrote = true;
                        clear_pace_error();
                        if (hooks.on_commit) hooks.on_commit(i);
                    } else {
                        out.cancel();
                        bump(&gi::PacingStats::errors);
                    }
                }
            }
        }
        if (wrote) bump(&gi::PacingStats::committed);
        ++i;
        const std::uint64_t ni = due_index();
        if (ni > i) {  // pavėlavome: praleisti laukai neįrašomi, tęsiama nuo einamojo (kaip gen)
            bump(&gi::PacingStats::late_skips, ni - i);
            i = ni;
            std::lock_guard<std::mutex> l(lat_m);
            pace_next = i;  // praleistiems indeksams submit_at — per vėlu
        }
    }
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
    if (s_pace) cudaStreamDestroy(s_pace);
    s_conv = s_pace = nullptr;
    ev_caller = nullptr;
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
            if (w) w->fail(VFGFX_ERR_INTERNAL, std::string("vidinė išimtis: ") + e.what());
            else g_create_error = std::string("vidinė išimtis: ") + e.what();
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
    if (!inst || !flow_def_json || !out) return fail(VFGFX_ERR_INVALID_ARG, "inst, flow_def_json ir out turi būti ne nullptr");
    const std::string def = flow_def_json;
    const std::string opts = writer_options_json ? writer_options_json : "";
    vramflow::VideoFormat fmt{};
    std::string err;
    if (!vramflow::format_from_flow_def(def, fmt, err)) return fail(VFGFX_ERR_INVALID_ARG, err);
    const gfx::Raster r = fmt.raster;
    // dažnis — bet kuris palaikomas (format.hpp); rastras — bet koks (grafikos sluoksniai)
    if (!vramflow::rate_supported(r.interlaced, fmt.frame_rate, err)) return fail(VFGFX_ERR_GEOMETRY, err);
    if (!r.alpha) return fail(VFGFX_ERR_INVALID_ARG, "flow turi būti video/v210a");
    if (!gfx::raster_ok(r)) return fail(VFGFX_ERR_GEOMETRY, "netinkamas rastras (plotis ir interlaced aukštis lyginiai)");

    auto w = std::make_unique<vfgfx_writer_s>();
    w->inst = inst;
    w->r = r;
    w->fmt = fmt;
    w->rate = func::mxl_rate(fmt.grain_rate());
    w->fields = r.interlaced ? 2 : 1;
    // deviceUuid nurodytas, bet tokio įrenginio nėra: aiški klaida, ne tylus kitas įrenginys. Ieškoma vieną kartą per create.
    const int uuid_dev = device_from_options(opts);
    {
        std::string u;
        if (gfx::detail::json_str(opts, "deviceUuid", u) && !u.empty() && uuid_dev < 0)
            return fail(VFGFX_ERR_INVALID_ARG, "deviceUuid " + u + " neatitinka jokio CUDA įrenginio");
    }
    if (!w->out.create(inst, def, opts, err, fmt)) {
        const int code = w->out.create_status != MXL_STATUS_OK ? static_cast<int>(w->out.create_status)
                         : w->out.existed                       ? VFGFX_ERR_STATE
                                                                : VFGFX_ERR_GEOMETRY;
        w->out.release(inst);
        return fail(code, err);
    }
    w->cuda_store = mxlv::is_cuda_layout(w->out.layout);
    if (!w->cuda_store) w->conv_dev = uuid_dev;
    if (w->cuda_store) {
        // Be deviceUuid — įrenginys 0 (biblioteka turi privatų cudart, kviečiančiojo cudaSetDevice jai nematomas; gfx.h).
        int n = 0;
        if (uuid_dev < 0 && (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)) {
            cudaGetLastError();
            w->out.release(inst);
            return fail(VFGFX_ERR_CUDA, "nėra CUDA įrenginio");
        }
        w->device = uuid_dev >= 0 ? uuid_dev : 0;
        DeviceScope ds(w->device);
        if (cudaStreamCreateWithFlags(&w->s_conv, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&w->s_pace, cudaStreamNonBlocking) != cudaSuccess) {
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
    if (w->pacing.load()) return w->fail(VFGFX_ERR_STATE, "vfgfx_write negalimas, kol veikia ritmo režimas");
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
    if (w->pacing.load()) return w->fail(VFGFX_ERR_STATE, "ritmo režimas jau paleistas");
    w->offset_ns.store(offset_ns);
    w->stop.store(false);
    {
        std::lock_guard<std::mutex> ll(w->lat_m);  // submit_at mato nuoseklią porą (pacing, pace_next)
        w->pace_next = w->start_index();
        w->pacing.store(true);
    }
    try {
        w->th = std::thread([w] { w->pace_loop(); });
    } catch (...) {
        w->pacing.store(false);
        return w->fail(VFGFX_ERR_STATE, "nepavyko paleisti ritmo gijos");
    }
    return 0;
}

// Konvertuoja kadrą į laisvą aibę (po sub_m). Grąžina aibę arba < 0 (klaida jau užfiksuota, kodas -rc).
static int prepare_set(vfgfx_writer w, const vfgfx_frame& fr) {
    int target;
    {
        std::lock_guard<std::mutex> l(w->lat_m);
        target = w->free_set();  // cur + eilė <= kSets - 1 (eilė <= kQueueCap), todėl laisva aibė visada yra
    }
    if (target < 0) return -w->fail(VFGFX_ERR_INTERNAL, "nėra laisvos paruoštų laukų aibės");
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
        return w->fail(VFGFX_ERR_TOO_LATE, "submit_at indeksas " + std::to_string(first) + " < " + std::to_string(min) +
                                               " (mažiausias priimamas — stats.next_pacing_index)");
    };
    {
        std::lock_guard<std::mutex> l(w->lat_m);
        const std::uint64_t min = w->min_scheduled_index();
        if (first < min) return too_late(min);
        const std::uint64_t now = w->current_index();
        if (first > now + lead)
            return w->fail(VFGFX_ERR_TOO_EARLY, "submit_at indeksas " + std::to_string(first) + " > dabartinis " +
                                                    std::to_string(now) + " + " + std::to_string(lead) + " (" +
                                                    std::to_string(VFGFX_SUBMIT_AT_MAX_LEAD_S) + " s)");
        const bool same = std::any_of(w->queue.begin(), w->queue.end(), [&](const Pending& p) { return p.first == first; });
        if (!same && static_cast<int>(w->queue.size()) >= kQueueCap)
            return w->fail(VFGFX_ERR_QUEUE_FULL, "submit_at eilė pilna (" + std::to_string(kQueueCap) + " laukiantys kadrai)");
    }
    if (w->nsets < kSets) {
        if (const int c = w->alloc_sets(kSets)) return c;
    }
    const int target = prepare_set(w, fr);
    if (target < 0) return -target;
    std::lock_guard<std::mutex> l(w->lat_m);
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
    w->stop.store(true);
    if (w->th.joinable()) w->th.join();
    w->pacing.store(false);
    return 0;
}

static int get_stats_impl(vfgfx_writer w, vfgfx_stats* out) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    if (!out) return w->fail(VFGFX_ERR_INVALID_ARG, "vfgfx_get_stats: out nullptr");
    // Išvesties struktūra: tikrinamas tik dydis ir flags; nežinoma uodega (naujesnė antraštė) nunulinama.
    const std::uint32_t sz = struct_size_of(out);
    const std::size_t hdr = offsetof(vfgfx_stats, committed), min = hdr + sizeof(std::uint64_t);
    if (sz < min || sz > kMaxStruct) return w->fail(VFGFX_ERR_STRUCT_SIZE, size_msg("vfgfx_stats", sz, min));
    if (out->flags) return w->fail(VFGFX_ERR_UNSUPPORTED, "nežinomos vfgfx_stats.flags " + std::to_string(out->flags));
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
        s.queue_depth = static_cast<std::uint32_t>(w->queue.size());
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
    if (q.flags) return fail(VFGFX_ERR_UNSUPPORTED, "nežinomos vfgfx_flow_params.flags " + std::to_string(q.flags));
    if (q.reserved0 || !all_zero(q.reserved)) return fail(VFGFX_ERR_UNSUPPORTED, "vfgfx_flow_params.reserved turi būti nuliai");
    if (!q.flow_id || !is_uuid(q.flow_id))
        return fail(VFGFX_ERR_INVALID_ARG, "flow_id privalomas UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)");
    const std::string label = json_escape(q.label ? q.label : "vramflow-gfx");
    const std::string hint = json_escape(q.group_hint ? std::string(q.group_hint) : std::string("vramflow-gfx-") + q.flow_id + ":Video");
    if (label.size() > 256 || hint.size() > 256) return fail(VFGFX_ERR_INVALID_ARG, "label / group_hint ilgesnis nei 256 B (JSON)");
    vramflow::VideoFormat f{};
    std::string err;
    if (q.format) {
        if (q.width || q.height || q.scan || q.frame_rate_num || q.frame_rate_den)
            return fail(VFGFX_ERR_INVALID_ARG, "format ir width/height/scan/frame_rate kartu — prieštaringa");
        if (!vramflow::parse_format(q.format, f, err)) return fail(VFGFX_ERR_GEOMETRY, err);
    } else {
        if (q.scan != VFGFX_SCAN_PROGRESSIVE && q.scan != VFGFX_SCAN_INTERLACED_TFF)
            return fail(VFGFX_ERR_INVALID_ARG, "nežinomas vfgfx_scan " + std::to_string(q.scan));
        if (q.width > 16384 || q.height > 16384) return fail(VFGFX_ERR_GEOMETRY, "rastras didesnis nei 16384");
        f.raster = gfx::Raster{static_cast<int>(q.width), static_cast<int>(q.height), q.scan == VFGFX_SCAN_INTERLACED_TFF, true};
        if (!gfx::raster_ok(f.raster)) return fail(VFGFX_ERR_GEOMETRY, "netinkamas rastras (plotis ir interlaced aukštis lyginiai, > 0)");
        if (q.frame_rate_num <= 0 || q.frame_rate_den <= 0) return fail(VFGFX_ERR_GEOMETRY, "kadrų dažnis turi būti > 0");
        f.frame_rate = vramflow::Rational{q.frame_rate_num, q.frame_rate_den};
        if (!vramflow::rate_supported(f.raster.interlaced, f.frame_rate, err)) return fail(VFGFX_ERR_GEOMETRY, err);
    }
    f.raster.alpha = true;  // video/v210a
    const std::string json = vramflow::video_flow_def(q.flow_id, label, f, hint);
    if (json.size() >= 2047) return fail(VFGFX_ERR_INTERNAL, "flow aprašas per ilgas");
    const std::size_t need = json.size() + 1;
    if (required) *required = need;
    if (!buf && buf_size == 0) return 0;
    if (!buf) return fail(VFGFX_ERR_INVALID_ARG, "buf nullptr, bet buf_size > 0");
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
int vfgfx_writer_create(mxlInstance inst, const char* flow_def_json, const char* writer_options_json, vfgfx_writer* out) {
    const int rc = guarded(nullptr, [&] { return create_impl(inst, flow_def_json, writer_options_json, out); });
    if (rc != 0 && out) *out = nullptr;  // išimtis po dalinio sukūrimo: nelaikom pusiau sukurto writer'io
    return rc;
}
int vfgfx_write(vfgfx_writer w, uint64_t index, const vfgfx_frame* f) {
    return guarded(w, [&] { return write_impl(w, index, f); });
}
int vfgfx_start_pacing(vfgfx_writer w, int64_t offset_ns) {
    return guarded(w, [&] { return start_pacing_impl(w, offset_ns); });
}
int vfgfx_submit(vfgfx_writer w, const vfgfx_frame* f) {
    return guarded(w, [&] { return submit_impl(w, f); });
}
int vfgfx_submit_at(vfgfx_writer w, uint64_t first_index, const vfgfx_frame* f) {
    return guarded(w, [&] { return submit_at_impl(w, first_index, f); });
}
int vfgfx_stop_pacing(vfgfx_writer w) {
    return guarded(w, [&] { return stop_pacing_impl(w); });
}
int vfgfx_get_stats(vfgfx_writer w, vfgfx_stats* out) {
    return guarded(w, [&] { return get_stats_impl(w, out); });
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

PacingStats pacing_stats(vfgfx_writer w) {
    std::lock_guard<std::mutex> l(w->st_m);
    return w->stats;
}

}  // namespace vramflow::gfx::internal
