// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// libvramflow-gfx (include/vramflow/gfx.h): RGBA -> standartinis MXL video/v210a flow, rankinis ir ritmo režimai.
//
// Konvertavimas: cuda-linear saugykla -> CUDA kernel'is (gfx::cuda), host saugykla -> CPU (gfx::cpu); rezultatas
// bitas į bitą lygus gfx::ref etalonui. Ritmo režimas: vfgfx_submit iškart konvertuoja kadrą į du paruoštus laukus
// (paritetai 0 ir 1; progresyviam rastrui — vieną grain'ą) vienoje iš dviejų paruoštų aibių ir po mutex'u pakeičia
// „paskutinį“; ritmo gija kas indeksą (func::wait_until) tik KOPIJUOJA atitinkamo pariteto lauką į slot'ą ir commit'ina
// (be pakartotinio konvertavimo). Du aibės + kopijavimas po tuo pačiu mutex'u užtikrina, kad submit nerašo į aibę, iš
// kurios ritmo gija skaito.
//
// Vėlavimo taisyklė (kaip func::gen): gija pabudo vėlai -> indeksas, kurio laukia, VIS TIEK įrašomas (vėlyvas commit;
// skaitytojas sprendžia, ar per vėlu), o po commit'o, jei einamasis indeksas aplenkė kitą, jo laukai praleidžiami
// (late_skips) ir tęsiama nuo einamojo — jokio „catch-up“ srauto. Laikrodis šoko atgal -> wait_until palaukia, kol
// laikas pasivys (i tik didėja: tas pats indeksas du kartus neįrašomas, o kitą indeksą wait_until atidėja iki jo laiko).
#include <cuda_runtime_api.h>

#include <algorithm>
#include <exception>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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

thread_local std::string g_create_error;

}  // namespace

struct vfgfx_writer_s {
    mxlInstance inst = nullptr;
    func::VideoOut out;
    gfx::Raster r{};
    bool cuda_store = false;
    int device = -1;       // cuda-linear saugyklos įrenginys; host saugykloje -1 (jokių CUDA kvietimų)
    int conv_dev = -1;     // host saugykla: įrenginys CUDA kadrų atsisiuntimui (deviceUuid arba -1 = numatytasis); s_conv kuriamas tik tada
    int fields = 1;  // paruoštų grain'ų skaičius vienoje aibėje (2 interlaced, 1 progresyviam)
    cudaStream_t s_conv = nullptr;  // create / write / submit (po conv_m); host saugykloje kuriamas tingiai (convert)
    cudaStream_t s_pace = nullptr;  // tik ritmo gija

    // Tarpiniai buferiai (kuriami pirmą kartą prireikus)
    void* d_stage = nullptr;  // device RGBA (host kadras -> cuda-linear)
    std::size_t d_stage_bytes = 0;
    std::vector<std::uint8_t> h_stage;  // host RGBA (CUDA kadras -> host saugykla)
    std::uint8_t* prep[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};  // [aibė][paritetas]; device arba host

    std::atomic<std::uint64_t> committed{0};
    std::atomic<bool> caller_synced{false};  // R20: true — kviečiantysis garantuoja, kad CUDA kadras jau įrašytas
    std::mutex op_m;   // write / start_pacing / stop_pacing serializacija
    std::mutex sub_m;  // submit serializacija
    std::mutex conv_m;  // convert(): bendri staging buferiai ir s_conv (write ir submit gali veikti lygiagrečiai)
    std::mutex lat_m;  // cur / has + paruoštų laukų skaitymas ritmo gijoje
    int cur = 0;
    bool has = false;

    std::atomic<bool> pacing{false};
    std::atomic<bool> stop{false};
    std::thread th;
    std::int64_t offset_ns = 0;
    gi::PacingHooks hooks;
    std::mutex st_m;
    gi::PacingStats stats;

    std::mutex err_m;
    std::string err;   // paskutinė kviečiančios gijos (create/write/submit/...) klaida
    std::string perr;  // paskutinė ritmo gijos klaida (atskirai, kad nesimaišytų su kviečiančiosios)

    void set_error(std::string m) {
        std::lock_guard<std::mutex> l(err_m);
        err = std::move(m);
    }
    void set_pace_error(std::string m) {
        std::lock_guard<std::mutex> l(err_m);
        perr = std::move(m);
    }
    int fail(int code, std::string m) {
        set_error(std::move(m));
        return code;
    }
    void bump(std::uint64_t gi::PacingStats::*field, std::uint64_t n = 1) {
        std::lock_guard<std::mutex> l(st_m);
        stats.*field += n;
    }

    // Kadro patikra: 0 arba klaidos kodas.
    int check_frame(const vfgfx_frame* f) {
        if (!f || !f->rgba) return fail(VFGFX_ERR_INVALID_ARG, "kadras arba kadro rodyklė nullptr");
        if (f->alpha != VFGFX_ALPHA_STRAIGHT && f->alpha != VFGFX_ALPHA_PREMULTIPLIED)
            return fail(VFGFX_ERR_INVALID_ARG, "nežinomas vfgfx_alpha");
        if (f->mem != VFGFX_MEM_HOST && f->mem != VFGFX_MEM_CUDA) return fail(VFGFX_ERR_INVALID_ARG, "nežinomas vfgfx_mem");
        if (f->width != static_cast<std::uint32_t>(r.w) || f->height != static_cast<std::uint32_t>(r.h))
            return fail(VFGFX_ERR_GEOMETRY, "kadras " + std::to_string(f->width) + "x" + std::to_string(f->height) + " != flow " +
                                                std::to_string(r.w) + "x" + std::to_string(r.h));
        if (f->stride < f->width * 4u) return fail(VFGFX_ERR_GEOMETRY, "stride < width*4");
        if (f->mem == VFGFX_MEM_CUDA && ((f->stride & 3u) || (reinterpret_cast<std::uintptr_t>(f->rgba) & 3u)))
            return fail(VFGFX_ERR_GEOMETRY, "CUDA kadro rodyklė ir stride turi būti kartotiniai 4 B");
        return 0;
    }

    // R20: prieš skaitant CUDA kadrą — palaukti visų konteksto darbų (ir kviečiančiojo stream'ų: pirminis kontekstas
    // bendras visiems proceso CUDA runtime'ams), nebent kviečiantysis pasakė, kad jau sinchronizavo. Mūsų s_conv yra
    // cudaStreamNonBlocking, todėl be šio laukimo jis nesilaiko jokios tvarkos su kviečiančiojo darbais.
    int wait_caller_frame() {
        if (caller_synced.load()) return 0;
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) return fail(VFGFX_ERR_CUDA, std::string("laukimas CUDA kadro (R20): ") + cudaGetErrorString(e));
        return 0;
    }

    // Konvertuoja kadro `n` laukų (paritetai par[k]) į dst[k] (paskirties saugyklos atmintis: device jei cuda_store, kitaip
    // host). Naudoja s_conv. 0 arba klaidos kodas. Kviečiama po op_m arba sub_m.
    int convert(const vfgfx_frame* f, int n, const int* par, std::uint8_t* const* dst) {
        std::lock_guard<std::mutex> cl(conv_m);
        const gfx::AlphaIn mode = f->alpha == VFGFX_ALPHA_PREMULTIPLIED ? gfx::AlphaIn::Premultiplied : gfx::AlphaIn::Straight;
        const int w = r.w, h = r.h;
        const std::size_t stride = f->stride;
        const std::size_t span = static_cast<std::size_t>(h - 1) * stride + static_cast<std::size_t>(w) * 4;
        DeviceScope dev(device);
        if (cuda_store) {
            if (f->mem == VFGFX_MEM_CUDA) {
                const int e = wait_caller_frame();
                if (e) return e;
            }
            const std::uint8_t* src = static_cast<const std::uint8_t*>(f->rgba);
            if (f->mem == VFGFX_MEM_HOST) {
                if ((stride & 3u)) return fail(VFGFX_ERR_GEOMETRY, "stride turi būti kartotinis 4 B (cuda-linear saugykla)");
                if (d_stage_bytes < span) {
                    if (d_stage) cudaFree(d_stage);
                    d_stage = nullptr;
                    d_stage_bytes = 0;
                    if (cudaMalloc(&d_stage, span) != cudaSuccess) {
                        cudaGetLastError();
                        return fail(VFGFX_ERR_CUDA, "cudaMalloc RGBA staging");
                    }
                    d_stage_bytes = span;
                }
                const cudaError_t e = cudaMemcpyAsync(d_stage, f->rgba, span, cudaMemcpyHostToDevice, s_conv);
                if (e != cudaSuccess) return fail(VFGFX_ERR_CUDA, std::string("H2D: ") + cudaGetErrorString(e));
                src = static_cast<const std::uint8_t*>(d_stage);
            }
            for (int k = 0; k < n; ++k) {
                const cudaError_t e = gfx::cuda::rgba_to_v210a(src, w, h, stride, mode, r, par[k], dst[k], s_conv);
                if (e != cudaSuccess) return fail(e == cudaErrorInvalidValue ? VFGFX_ERR_GEOMETRY : VFGFX_ERR_CUDA,
                                                  std::string("rgba_to_v210a: ") + cudaGetErrorString(e));
            }
            const cudaError_t e = cudaStreamSynchronize(s_conv);
            if (e != cudaSuccess) return fail(VFGFX_ERR_CUDA, std::string("sinchronizavimas: ") + cudaGetErrorString(e));
            return 0;
        }
        const std::uint8_t* src = static_cast<const std::uint8_t*>(f->rgba);
        if (f->mem == VFGFX_MEM_CUDA) {
            // Host saugyklos rašytuvas CUDA kontekstą kuria TIK čia, kai pirmą kartą ateina CUDA kadras (po conv_m);
            // host kadrų rašytuvas niekada neliečia CUDA (veikia ir be GPU / su CUDA_VISIBLE_DEVICES="").
            DeviceScope cd(conv_dev);
            if (!s_conv && cudaStreamCreateWithFlags(&s_conv, cudaStreamNonBlocking) != cudaSuccess) {
                cudaGetLastError();
                s_conv = nullptr;
                return fail(VFGFX_ERR_CUDA, "cudaStreamCreate (CUDA kadro atsisiuntimui)");
            }
            if (const int we = wait_caller_frame()) return we;
            if (h_stage.size() < span) h_stage.resize(span);
            cudaError_t e = cudaMemcpyAsync(h_stage.data(), f->rgba, span, cudaMemcpyDeviceToHost, s_conv);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s_conv);
            if (e != cudaSuccess) return fail(VFGFX_ERR_CUDA, std::string("D2H: ") + cudaGetErrorString(e));
            src = h_stage.data();
        }
        for (int k = 0; k < n; ++k)
            if (!gfx::cpu::rgba_to_v210a(src, w, h, stride, mode, r, par[k], dst[k]))
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
            else set_error(m);
            return static_cast<int>(st);
        }
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
    // Indeksas, kurio commit'o laikas (indekso laikas + offset_ns) jau atėjo: index_of(now − offset_ns), saugu be ženklo.
    std::uint64_t due_index() const {
        std::uint64_t t = now();
        if (offset_ns >= 0) {
            const std::uint64_t o = static_cast<std::uint64_t>(offset_ns);
            t = t > o ? t - o : 0;
        } else {
            t += 0 - static_cast<std::uint64_t>(offset_ns);  // beženklė neigimas: saugu ir INT64_MIN
        }
        return mxlTimestampToIndex(&func::kFieldRate, t);
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
    std::uint64_t i = due_index() + 1;
    if (committed.load() > 0) i = std::max(i, out.last_committed + 1);  // restartas: niekada nekartoti commit'into indekso
    const std::size_t gb = gfx::grain_bytes(r);
    while (!stop.load()) {
        const bool back = hooks.wait ? hooks.wait(i, offset_ns, stop) : func::wait_until(i, offset_ns, stop);
        if (stop.load()) break;
        if (back) bump(&gi::PacingStats::clock_backward);
        bool wrote = false;
        {
            std::lock_guard<std::mutex> l(lat_m);
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
                            ok = false;
                            set_pace_error(std::string("D2D: ") + cudaGetErrorString(e));
                        }
                    } else {
                        std::memcpy(dst, src, gb);
                    }
                    if (ok && commit_open(true) == 0) {
                        wrote = true;
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
    if (d_stage) cudaFree(d_stage);
    d_stage = nullptr;
    if (s_conv) {
        DeviceScope cd(cuda_store ? -1 : conv_dev);  // cuda_store: jau esame `dev` įrenginyje
        cudaStreamDestroy(s_conv);
    }
    if (s_pace) cudaStreamDestroy(s_pace);
    s_conv = s_pace = nullptr;
}

template <class F>
static int guarded(vfgfx_writer w, F&& f) noexcept {
    try {
        return f();
    } catch (const std::exception& e) {
        try {
            if (w) w->set_error(std::string("vidinė išimtis: ") + e.what());
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
    gfx::Raster r{};
    std::string err;
    if (!gfx::raster_from_flow_def(def, r, err)) return fail(VFGFX_ERR_INVALID_ARG, err);
    if (!r.alpha) return fail(VFGFX_ERR_INVALID_ARG, "flow turi būti video/v210a");
    if (!gfx::raster_ok(r)) return fail(VFGFX_ERR_GEOMETRY, "netinkamas rastras (plotis ir interlaced aukštis lyginiai)");

    auto w = std::make_unique<vfgfx_writer_s>();
    w->inst = inst;
    w->r = r;
    w->fields = r.interlaced ? 2 : 1;
    // deviceUuid nurodytas, bet tokio įrenginio nėra: aiški klaida, ne tylus kitas įrenginys. Ieškoma vieną kartą per create.
    const int uuid_dev = device_from_options(opts);
    {
        std::string u;
        if (gfx::detail::json_str(opts, "deviceUuid", u) && !u.empty() && uuid_dev < 0)
            return fail(VFGFX_ERR_INVALID_ARG, "deviceUuid " + u + " neatitinka jokio CUDA įrenginio");
    }
    if (!w->out.create(inst, def, opts, err, r)) {
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
    }
    {
        DeviceScope ds(w->device);
        if (w->cuda_store) {
            if (cudaStreamCreateWithFlags(&w->s_conv, cudaStreamNonBlocking) != cudaSuccess ||
                cudaStreamCreateWithFlags(&w->s_pace, cudaStreamNonBlocking) != cudaSuccess) {
                cudaGetLastError();
                w->free_all();
                w->out.release(inst);
                return fail(VFGFX_ERR_CUDA, "cudaStreamCreate");
            }
        }  // host saugykla: nė vieno CUDA kvietimo (s_conv sukuriamas tingiai convert(), tik CUDA kadrui)
        const std::size_t gb = gfx::grain_bytes(r);
        for (int s = 0; s < 2; ++s)
            for (int p = 0; p < w->fields; ++p) {
                void* ptr = nullptr;
                if (w->cuda_store) {
                    if (cudaMalloc(&ptr, gb) != cudaSuccess) {
                        cudaGetLastError();
                        w->free_all();
                        w->out.release(inst);
                        return fail(VFGFX_ERR_CUDA, "cudaMalloc paruoštam laukui");
                    }
                } else {
                    ptr = std::aligned_alloc(64, (gb + 63) / 64 * 64);
                    if (!ptr) {
                        w->free_all();
                        w->out.release(inst);
                        return fail(VFGFX_ERR_INTERNAL, "nepavyko išskirti paruošto lauko (host atmintis)");
                    }
                }
                w->prep[s][p] = static_cast<std::uint8_t*>(ptr);
            }
    }
    *out = w.release();
    return 0;
}

static int write_impl(vfgfx_writer w, std::uint64_t index, const vfgfx_frame* f) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> l(w->op_m);
    if (w->pacing.load()) return w->fail(VFGFX_ERR_STATE, "vfgfx_write negalimas, kol veikia ritmo režimas");
    if (const int c = w->check_frame(f)) return c;
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
    if (const int c = w->convert(f, 1, par, d)) {
        w->out.cancel();
        return c;
    }
    const int c = w->commit_open();
    if (c) w->out.cancel();
    return c;
}

static int start_pacing_impl(vfgfx_writer w, std::int64_t offset_ns) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> l(w->op_m);
    if (w->pacing.load()) return w->fail(VFGFX_ERR_STATE, "ritmo režimas jau paleistas");
    w->offset_ns = offset_ns;
    w->stop.store(false);
    w->pacing.store(true);
    try {
        w->th = std::thread([w] { w->pace_loop(); });
    } catch (...) {
        w->pacing.store(false);
        return w->fail(VFGFX_ERR_STATE, "nepavyko paleisti ritmo gijos");
    }
    return 0;
}

static int submit_impl(vfgfx_writer w, const vfgfx_frame* f) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> sl(w->sub_m);
    if (const int c = w->check_frame(f)) return c;
    // Konvertuojame į aibę, kuri NĖRA „dabartinė“: ritmo gija skaito tik w->cur (po lat_m), o cur keičia tik šita funkcija.
    int target;
    {
        std::lock_guard<std::mutex> l(w->lat_m);
        target = w->has ? 1 - w->cur : w->cur;
    }
    const int par[2] = {0, 1};
    std::uint8_t* const d[2] = {w->prep[target][0], w->prep[target][1]};
    if (const int c = w->convert(f, w->fields, par, d)) return c;
    std::lock_guard<std::mutex> l(w->lat_m);
    w->cur = target;
    w->has = true;
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


static void destroy_impl(vfgfx_writer w, mxlInstance inst) {
    if (!w) return;
    stop_pacing_impl(w);
    if (inst) w->inst = inst;
    delete w;  // ~vfgfx_writer_s: release(inst) + free_all
}

// --- Eksportuojami apvalkalai: jokia C++ išimtis neperženg C ABI ribos --------------------------------------------------
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
int vfgfx_stop_pacing(vfgfx_writer w) {
    return guarded(w, [&] { return stop_pacing_impl(w); });
}
uint64_t vfgfx_committed(vfgfx_writer w) { return w ? w->committed.load() : 0; }
int vfgfx_set_caller_synced(vfgfx_writer w, int synced) {
    if (!w) return VFGFX_ERR_INVALID_ARG;
    w->caller_synced.store(synced != 0);
    return 0;
}

const char* vfgfx_last_error(vfgfx_writer w) {
    // Rodyklė — thread_local kopija: galioja iki kito vfgfx_last_error iškvietimo TOJE PAČIOJE gijoje.
    static thread_local std::string copy;
    try {
        if (!w) {
            copy = g_create_error;
        } else {
            std::lock_guard<std::mutex> l(w->err_m);
            copy = !w->err.empty() ? w->err : w->perr;  // kviečiančiosios klaida turi pirmenybę; kitaip ritmo gijos
        }
        return copy.c_str();
    } catch (...) {
        return "";
    }
}

void vfgfx_writer_destroy(vfgfx_writer w, mxlInstance inst) {
    try {
        destroy_impl(w, inst);
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
