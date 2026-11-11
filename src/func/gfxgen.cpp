// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// gfxgen — sintetinis grafikos šaltinis: N šaltinių × K sluoksnių = N·K standartinių interlaced v210a flow'ų
// (flow_id(kRoleGfx, n·8 + k)). RGBA turinį kuria gfx_content (deterministinis, tikrinamas verify), į MXL rašo TIK per viešą
// libvramflow-gfx (include/vramflow/gfx.h) — tai ir įrodo, kad biblioteka pakankama.
//
// Režimai:
//  pace   — kiekvienam flow'ui kadras 0 submit'inamas VIENĄ kartą, toliau kas lauką jį commit'ina bibliotekos ritmo gija
//           (ši funkcija tik laukia --duration). Kaina: viena konversija submit metu, ne per lauką.
//           Sėkmės taisyklė: ≥ 1 s bėgime kiekvienas flow turi commit'inti ≥ 90 % tikėtinų laukų (0 — visada klaida), kitaip
//           klaida + exit 3 su vfgfx_last_error tekstu stderr.
//  manual — kadras = index/2 kiekvienam indeksui (abu laukai gauna tą patį kadrą, paritetą parenka biblioteka); ši funkcija
//           pati laikosi ritmo (wait_until kaip gen) ir kviečia vfgfx_write kiekvienam laukui. Kito kadro turinys sukuriamas
//           iš anksto (tarp lyginio ir nelyginio lauko), kad kūrimas nepatektų į lyginio lauko commit'ą.
//
// Kadro atmintis (efektyvumas, žr. gfx.h): cuda-linear saugyklai (impl gpu) kadras kartą įkeliamas į GPU (H2D) ir į biblioteką
// perduodamas kaip VFGFX_MEM_CUDA — manual režime tas pats įkeltas kadras naudojamas abiem laukams, todėl H2D mokama kartą
// per kadrą, ne per lauką. Įkėlimas — upload_h2d (grįžta tik baigus DMA: biblioteka kadrą skaito savo srautu, cuda_upload.hpp).
// host saugyklai (cpu arba gpu+host) kadras perduodamas kaip VFGFX_MEM_HOST (konvertuoja CPU).
// Sluoksnio geometrija (--layer-geom): logo = 384×216, full = --format kadras (numatytasis 1920×1080) arba WxH; neišvardintiems
// k: k = 0 logo, kiti full. Skenavimas ir kadrų dažnis — iš --format (bet kuris palaikomas).
// Nelyginis k siunčiamas kaip premultiplied, lyginis — straight (kaip gfx_content).
#include <cuda_runtime_api.h>

#include <vramflow/gfx.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "func/cuda_upload.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "gfx/gfx_content.hpp"

namespace mxlv::func {
namespace {

using vramflow::gfx::Raster;


}  // namespace

// Bendra su verify (--mode composite): sluoksnių geometrija turi sutapti su gfxgen.
bool parse_layer_geoms(const Opts& o, std::vector<Raster>& out, std::string& err) {
    std::vector<std::string> toks;
    std::stringstream ss(o.layer_geom);
    for (std::string t; std::getline(ss, t, ',');) toks.push_back(t);
    if (toks.size() > o.layers) { err = "--layer-geom turi daugiau elementų nei --layers"; return false; }
    for (std::uint32_t k = 0; k < o.layers; ++k) {
        const std::string t = k < toks.size() ? toks[k] : (k == 0 ? "logo" : "full");
        const bool inter = o.fmt.raster.interlaced;
        const Raster full{o.fmt.raster.w, o.fmt.raster.h, inter, true};
        Raster r = full;
        int w = 0, h = 0;
        if (t == "logo") r = Raster{384, 216, inter, true};
        else if (t == "full") r = full;
        else if (std::sscanf(t.c_str(), "%dx%d", &w, &h) == 2 && w >= 256 && h > 0) r = Raster{w, h, inter, true};
        else { err = "netinkama geometrija '" + t + "' (logo | full | WxH, W >= 256)"; return false; }
        if (!vramflow::gfx::raster_ok(r)) { err = "netinkamas rastras '" + t + "' (lyginis plotis ir aukštis)"; return false; }
        out.push_back(r);
    }
    return true;
}

namespace {

// Vienas flow (n, k): rašytojas ir dvi kadrų vietos [0] = dabartinė, [1] = kita (manual prefetch).
struct Src {
    std::uint32_t n = 0, k = 0;
    Raster r{};
    vfgfx_writer w = nullptr;
    std::vector<std::uint8_t> h[2];
    void* d[2] = {nullptr, nullptr};  // device kopija (tik cuda-linear)
    std::uint64_t frame[2] = {~0ull, ~0ull};
    int rc = 0;  // paskutinis vfgfx_write rezultatas (lygiagrečiam rašymui)

    std::size_t bytes() const { return static_cast<std::size_t>(r.w) * static_cast<std::size_t>(r.h) * 4; }
    vfgfx_frame view(int slot, bool cuda) const {
        vfgfx_frame f{};
        f.rgba = cuda ? d[slot] : static_cast<const void*>(h[slot].data());
        f.width = static_cast<std::uint32_t>(r.w);
        f.height = static_cast<std::uint32_t>(r.h);
        f.stride = static_cast<std::uint32_t>(r.w) * 4;
        f.alpha = (k & 1u) ? VFGFX_ALPHA_PREMULTIPLIED : VFGFX_ALPHA_STRAIGHT;
        f.mem = cuda ? VFGFX_MEM_CUDA : VFGFX_MEM_HOST;
        return f;
    }
};

}  // namespace

int run_gfxgen(const Opts& o) {
    Stats st;
    const std::uint32_t K = o.layers, N = o.flows, total = N * K;
    st.flows.resize(total);
    auto& stop = stop_flag();
    auto finish = [&](int rc, const std::string& extra = {}) {
        st.exit_code = rc;
        write_summary(o, "gfxgen", st, extra);
        std::fprintf(stderr, "[gfxgen] baigta, exit %d %s\n", rc, st.fatal.c_str());
        return rc;
    };
    std::vector<Raster> geom;
    std::string err;
    if (!parse_layer_geoms(o, geom, err)) { st.fatal = err; return finish(2); }
    const bool manual = o.gfx_mode == "manual";
    const bool inter = o.fmt.raster.interlaced;
    const mxlRational rate = mxl_rate(o.fmt.grain_rate());
    const bool upload = o.impl != "cpu" && o.storage == "cuda-linear";  // kadras į GPU kartą, perduodamas kaip MEM_CUDA
    int dev = -1;
    if (upload) {
        dev = cuda_device(o.device_uuid);
        if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) { st.fatal = "CUDA įrenginys"; return finish(2); }
        cudaFree(nullptr);
    }
    std::string geom_json = "\"geom\":[";
    for (std::uint32_t k = 0; k < K; ++k)
        geom_json += (k ? "," : "") + std::string("\"") + std::to_string(geom[k].w) + "x" + std::to_string(geom[k].h) + "\"";
    geom_json += "],\"layers\":" + std::to_string(K) + ",\"mode\":\"" + o.gfx_mode + "\",\"impl\":\"" + o.impl + "\",\"storage\":\"" +
                 o.storage + "\"";

    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; return finish(2, geom_json); }
    mxlGarbageCollectFlows(inst);

    std::vector<Src> src(total);
    const std::string opts = upload ? storage_options(o, dev) : std::string{};
    int rc = 0;
    for (std::uint32_t n = 0; n < N && !rc; ++n)
        for (std::uint32_t k = 0; k < K && !rc; ++k) {
            Src& s = src[n * K + k];
            s.n = o.flow_base + n;  // flow ID ir turinys priklauso nuo flow_base + n (kaip gen)
            s.k = k;
            s.r = geom[k];
            const int c = vfgfx_writer_create(inst, gfx_def(s.n, k, vramflow::VideoFormat{s.r, o.fmt.frame_rate}, o.run_seed).c_str(),
                                              opts.empty() ? nullptr : opts.c_str(), &s.w);
            if (c) { st.fatal = "writer " + std::to_string(n) + "/" + std::to_string(k) + ": " + std::to_string(c) + " " + vfgfx_last_error(nullptr); rc = 2; }
        }

    const int slots = manual ? 2 : 1;
    for (auto& s : src) {
        if (rc) break;
        for (int i = 0; i < slots; ++i) {
            s.h[i].resize(s.bytes());
            if (upload && cudaMalloc(&s.d[i], s.bytes()) != cudaSuccess) { st.fatal = "cudaMalloc"; rc = 2; break; }
        }
    }
    const int threads = o.threads > 0 ? o.threads : static_cast<int>(std::min<unsigned>(std::max(1u, total), std::max(1u, std::thread::hardware_concurrency())));
    ThreadPool pool(threads);

    // Kadro `fr` turinys į vietą `slot` visiems flow'ams (CPU gijos), tada H2D (pagrindinė gija — CUDA kontekstas).
    auto prepare = [&](int slot, std::uint64_t fr) -> bool {
        pool.run(static_cast<int>(src.size()), [&](int i) {
            Src& s = src[i];
            vramflow::gfx::gfx_content(o.run_seed, s.n, s.k, fr, s.r.w, s.r.h, s.h[slot].data());
            s.frame[slot] = fr;
        });
        if (upload)
            for (auto& s : src)
                if (upload_h2d(s.d[slot], s.h[slot].data(), s.bytes()) != cudaSuccess) return false;
        return true;
    };
    auto count_err = [&](std::size_t i, int c) {
        st.flows[i].inc("errors");
        if (st.flows[i].c["errors"] == 1)
            std::fprintf(stderr, "[gfxgen] flow %u/%u: klaida %d: %s\n", src[i].n, src[i].k, c, vfgfx_last_error(src[i].w));
    };

    ClockWatch cw;
    cw.stepped();
    if (!rc && !manual) {
        // --- pace: kadras 0 vieną kartą, tada bibliotekos ritmo gija ---
        for (std::size_t i = 0; i < src.size() && !rc; ++i) {
            Src& s = src[i];
            vramflow::gfx::gfx_content(o.run_seed, s.n, s.k, 0, s.r.w, s.r.h, s.h[0].data());
            if (upload && upload_h2d(s.d[0], s.h[0].data(), s.bytes()) != cudaSuccess) { st.fatal = "cudaMemcpy"; rc = 2; break; }
            const vfgfx_frame f = s.view(0, upload);
            const int c = vfgfx_submit(s.w, &f);
            if (c) { st.fatal = "submit: " + std::to_string(c) + " " + vfgfx_last_error(s.w); rc = 2; break; }
            st.flows[i].inc("submits");
            // biblioteka kadro neišsaugo (gfx.h) — buferius galima atlaisvinti iškart
            std::vector<std::uint8_t>().swap(s.h[0]);
            if (s.d[0]) { cudaFree(s.d[0]); s.d[0] = nullptr; }
        }
        std::uint64_t idx0 = 0;
        const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
        if (!rc) {
            idx0 = mxlGetCurrentIndex(&rate) + 1;
            for (auto& s : src) {
                const int c = vfgfx_start_pacing(s.w, 0);
                if (c) { st.fatal = "start_pacing: " + std::to_string(c) + " " + vfgfx_last_error(s.w); rc = 2; break; }
            }
        }
        while (!rc && !stop.load() && now_tai() < end_tai) mxlSleepForNs(20'000'000ull);
        const std::uint64_t idx1 = mxlGetCurrentIndex(&rate);
        for (std::size_t i = 0; i < src.size(); ++i) {
            const int c = vfgfx_stop_pacing(src[i].w);
            if (c) count_err(i, c);
        }
        if (idx0) {
            st.first_index = idx0;
            st.last_index = idx1;
        }
        const std::uint64_t expected = idx0 && idx1 >= idx0 ? idx1 - idx0 + 1 : 0;
        // Ritmo gijos sėkmės taisyklė (bibliotekos stop_pacing klaidų negrąžina): ≥ 1 s trukmės bėgime flow turi commit'inti
        // bent 90 % tikėtinų laukų, o 0 commit'ų — visada klaida. Pažeidimas → klaidų skaitiklis (→ rc 3) + vfgfx_last_error tekstas.
        const bool judge = !stop.load() && o.duration_s >= 1.0 && expected > 0;
        for (std::size_t i = 0; i < src.size(); ++i) {
            const std::uint64_t cm = vfgfx_committed(src[i].w);
            st.flows[i].inc("commits", cm);
            st.flows[i].inc("expected_fields", expected);
            st.flows[i].inc("missed", expected > cm ? expected - cm : 0);
            if (judge && (cm == 0 || cm * 10 < expected * 9)) {
                std::fprintf(stderr, "[gfxgen] flow %u/%u: ritmo gija commit'ino %llu iš %llu laukų: %s\n", src[i].n, src[i].k,
                             static_cast<unsigned long long>(cm), static_cast<unsigned long long>(expected), vfgfx_last_error(src[i].w));
                st.flows[i].inc("errors");
            }
        }
        cw.stepped();
        st.proc["clock_steps"] = cw.steps;
    } else if (!rc) {
        // --- manual: pats laikausi ritmo, vfgfx_write kiekvienam laukui ---
        const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
        const double warm = warm_tai_after(o.warmup_s);
        std::uint64_t i = mxlGetCurrentIndex(&rate) + 1;
        const bool par_write = !upload;  // host saugykla: rašytojai nepriklausomi, konvertuoja CPU — lygiagrečiai
        while (!rc && !stop.load() && now_tai() < end_tai) {
            const std::uint64_t fr = inter ? i / 2 : i;  // interlaced: abu laukai — tas pats kadras
            if (src[0].frame[0] != fr) {
                if (src[0].frame[1] == fr) {  // iš anksto paruoštas kadras
                    for (auto& s : src) {
                        std::swap(s.h[0], s.h[1]);
                        std::swap(s.d[0], s.d[1]);
                        std::swap(s.frame[0], s.frame[1]);
                    }
                } else if (!prepare(0, fr)) { st.fatal = "cudaMemcpy"; rc = 2; break; }
            }
            if (wait_until(rate, i, 0, stop)) st.proc["clock_backward"]++;
            if (stop.load()) break;
            const std::uint64_t t_wake = now_tai();
            auto one = [&](int j) {
                const vfgfx_frame f = src[j].view(0, upload);
                src[j].rc = vfgfx_write(src[j].w, i, &f);
            };
            if (par_write) pool.run(static_cast<int>(src.size()), one);
            else for (int j = 0; j < static_cast<int>(src.size()); ++j) one(j);
            for (std::size_t j = 0; j < src.size(); ++j) {
                if (src[j].rc) count_err(j, src[j].rc);
                else st.flows[j].inc("writes");
            }
            const std::uint64_t t_done = now_tai();
            if (t_done >= warm) {
                st.add_hist("write_ms", t_done - t_wake);
                st.add_hist("commit_lateness", t_done - mxlIndexToTimestamp(&rate, i));
            }
            if (st.first_index == ~0ull) st.first_index = i;
            st.last_index = i;
            const bool wrote_even = (i & 1) == 0;
            ++i;
            // interlaced: laisvas ~20 ms tarpas iki nelyginio lauko -> kito kadro prefetch; progresyvus: kiekvienas grain'as — naujas
            // kadras, prefetch po kiekvieno rašymo
            if (inter ? wrote_even : true) {
                const std::uint64_t t0 = now_tai();
                if (!prepare(1, inter ? i / 2 + 1 : i)) { st.fatal = "cudaMemcpy"; rc = 2; break; }
                if (t_done >= warm) st.add_hist("prep_ms", now_tai() - t0);
            }
            const bool clk = cw.stepped();
            const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
            if (now_idx > i) {
                st.proc[clk ? "clock_skips" : (t_done < warm ? "warmup_skips" : "late_skips")] += now_idx - i;
                i = now_idx;
            }
        }
        st.proc["clock_steps"] = cw.steps;
        for (std::size_t j = 0; j < src.size(); ++j) st.flows[j].inc("commits", vfgfx_committed(src[j].w));
    }

    std::uint64_t errs = 0;
    for (const auto& f : st.flows) {
        auto it = f.c.find("errors");
        if (it != f.c.end()) errs += it->second;
    }
    for (auto& f : st.flows) f.inc("errors", 0);  // raktas visada yra summary
    if (!rc && errs) { st.fatal = std::to_string(errs) + " klaidų (žr. stderr)"; rc = 3; }
    for (auto& s : src) {
        if (s.w) vfgfx_writer_destroy(s.w, inst);
        for (void*& d : s.d)
            if (d) { cudaFree(d); d = nullptr; }
    }
    mxlDestroyInstance(inst);
    return finish(rc, geom_json);
}

}  // namespace mxlv::func
