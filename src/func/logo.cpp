// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// logo — F-LOGO (architektūra GPU funkcijų §3, §5.2): N × MXL įėjimų (--format, numatytasis 1080i50) → logotipas
// įkepamas GPU (vienas batch kernel'is visiems N per grain'ą: laukas arba kadras) → N × MXL išėjimų. Visas kelias VRAM;
// CPU tik valdo.
// Niekas nedingsta tyliai: kiekvienas indeksas kiekvienam flow'ui yra fields_ok arba late/invalid/lapped/too_late.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#include "common/index_math.hpp"
#include "func/cuda_upload.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/media_kernels.hpp"
#include "media/cpu/cpu_media.hpp"
#include "media/media_ref.hpp"

#include <thread>

namespace mxlv::func {

// Logotipo plokštumos device atmintyje.
struct DeviceLogo {
    media::LogoDesc d{};
    std::vector<void*> bufs;
    bool upload(const media::ref::Logo& l, int x, int y) {
        auto up = [&](const void* src, std::size_t bytes) -> void* {
            void* p = nullptr;
            if (cudaMalloc(&p, bytes) != cudaSuccess || upload_h2d(p, src, bytes) != cudaSuccess) return nullptr;
            bufs.push_back(p);
            return p;
        };
        d.y10 = static_cast<const std::uint16_t*>(up(l.y10.data(), l.y10.size() * 2));
        d.a8 = static_cast<const std::uint8_t*>(up(l.a8.data(), l.a8.size()));
        d.cb10 = static_cast<const std::uint16_t*>(up(l.cb10.data(), l.cb10.size() * 2));
        d.cr10 = static_cast<const std::uint16_t*>(up(l.cr10.data(), l.cr10.size() * 2));
        d.ac8 = static_cast<const std::uint8_t*>(up(l.ac8.data(), l.ac8.size()));
        d.lw = l.w;
        d.lh = l.h;
        d.x = x;
        d.y = y;
        return d.y10 && d.a8 && d.cb10 && d.cr10 && d.ac8;
    }
    ~DeviceLogo() {
        for (void* p : bufs) cudaFree(p);
    }
};

bool load_logo(const Opts& o, media::ref::Logo& logo, std::string& err) {
    if (o.logo == "builtin") {
        logo = media::ref::builtin_logo();
    } else {
        std::vector<std::uint8_t> rgba;
        int w = 0, h = 0;
        err = media::ref::load_png_rgba(o.logo, rgba, w, h);
        if (!err.empty()) return false;
        if (!media::ref::logo_from_rgba(rgba.data(), w, h, logo)) { err = "logotipo plotis turi būti lyginis"; return false; }
    }
    const int fw = o.fmt.raster.w, fh = o.fmt.raster.h;
    if (o.logo_x % 2 || o.logo_x < 0 || o.logo_y < 0 || o.logo_x + logo.w > fw || o.logo_y + logo.h > fh) {
        err = "logotipas " + std::to_string(logo.w) + "x" + std::to_string(logo.h) + " ties " + std::to_string(o.logo_x) + "," +
              std::to_string(o.logo_y) + " netelpa į " + vramflow::format_name(o.fmt) + " kadrą " + std::to_string(fw) + "x" +
              std::to_string(fh) + " arba X nelyginis";
        return false;
    }
    return true;
}

int run_logo(const Opts& o) {
    Stats st;
    st.flows.resize(o.flows);
    auto& stop = stop_flag();
    auto finish = [&](int rc) {
        st.exit_code = rc;
        write_summary(o, "logo", st);
        std::fprintf(stderr, "[logo] baigta, exit %d %s\n", rc, st.fatal.c_str());
        return rc;
    };
    // Variantai: G (gpu + cuda-linear), H (gpu + host: H2D → kernel → D2H), C (cpu + host, CUDA nekuriamas).
    // Tick logika (laikas, deadline, skaitikliai, lap patikra) bendra — skiriasi tik apdorojimas.
    const bool cpu = o.impl == "cpu";
    const bool host_gpu = !cpu && o.storage == "host";
    int dev = -1;
    if (!cpu) {
        dev = cuda_device(o.device_uuid);
        if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) { st.fatal = "CUDA įrenginys"; return finish(2); }
        cudaFree(nullptr);
    }
    media::ref::Logo logo;
    std::string err;
    if (!load_logo(o, logo, err)) { st.fatal = err; return finish(1); }
    DeviceLogo dlogo;
    if (!cpu && !dlogo.upload(logo, o.logo_x, o.logo_y)) { st.fatal = "logotipo įkėlimas į GPU"; return finish(2); }
    media::cpu::LogoPlanes hlogo{logo.y10.data(), logo.a8.data(), logo.cb10.data(), logo.cr10.data(), logo.ac8.data(),
                                 logo.w, logo.h, o.logo_x, o.logo_y};
    ThreadPool pool(cpu ? (o.threads > 0 ? o.threads : static_cast<int>(std::min<unsigned>(o.flows, std::thread::hardware_concurrency()))) : 1);

    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; return finish(2); }
    std::vector<VideoIn> in(o.flows);
    std::vector<VideoOut> out(o.flows);
    int rc = 0;
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n)
        if (!in[n].open(inst, flow_id(o.in_role, o.flow_base + n), 15000, stop, err, &o.fmt)) { st.fatal = "in " + std::to_string(n) + ": " + err; rc = 4; }
    const std::string opts = cpu ? std::string{} : storage_options(o, dev);
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n)
        if (!out[n].create(inst, video_def(kRoleLogo, o.flow_base + n, o.run_seed, o.fmt), opts, err, o.fmt)) { st.fatal = "out " + std::to_string(n) + ": " + err; rc = 2; }
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n)
        if (in[n].cuda == (cpu || host_gpu)) { st.fatal = "įėjimo storage neatitinka --impl/--storage"; rc = 1; }

    const mxlRational rate = mxl_rate(o.fmt.grain_rate());
    const media::Dim dim{o.fmt.raster.w, o.fmt.raster.h};
    const bool inter = o.fmt.raster.interlaced;
    const std::size_t gbytes = media::grain_bytes(dim, inter);
    // savitikros injekcijos žodis: logotipo viduje (kadro eilutė logo_y + 8, grupė logo_x / 6 + 1); 1080i50 — kaip anksčiau
    const std::size_t inject_word = static_cast<std::size_t>(inter ? o.logo_y / 2 + 4 : o.logo_y + 8) * media::v210_words_per_line(dim.w) +
                                    static_cast<std::size_t>(o.logo_x / 6) * 4 + 4;
    cudaStream_t stream{};
    if (!cpu) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // H: device scratch įėjimui ir išėjimui kiekvienam flow'ui
    std::vector<std::uint32_t*> s_in, s_out;
    if (host_gpu)
        for (std::uint32_t n = 0; n < o.flows && !rc; ++n) {
            std::uint32_t *a = nullptr, *b = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&a), gbytes) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void**>(&b), gbytes) != cudaSuccess) { st.fatal = "scratch"; rc = 2; }
            s_in.push_back(a);
            s_out.push_back(b);
        }
    std::vector<media::LogoJob> host_jobs;
    std::vector<media::LogoJob> jobs;
    std::vector<std::uint32_t> job_flow;
    std::vector<int> state(o.flows);  // 0 laukia, 1 paruoštas, 2 baigtas (be išėjimo)
    std::vector<const std::uint32_t*> src(o.flows);
    const std::uint32_t grain_count = o.flows ? in[0].layout.slotCount : 10;
    const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    const double warm = warm_tai_after(o.warmup_s);
    std::uint64_t i = mxlGetCurrentIndex(&rate) + 1;
    std::uint64_t injected = 0;
    ClockWatch cw;
    cw.stepped();

    while (!rc && !stop.load() && now_tai() < end_tai) {
        if (wait_until(rate, i, o.read_offset_us * 1000, stop)) st.proc["clock_backward"]++;
        if (stop.load()) break;
        const std::uint64_t t_wake = now_tai();
        const std::uint64_t deadline = mxlIndexToTimestamp(&rate, i) + std::uint64_t{o.deadline_ms} * 1'000'000ull;
        std::fill(state.begin(), state.end(), 0);
        std::uint32_t pending = o.flows;
        // 3: visi įėjimai — neblokuojančiai; laukiantys kartojami kas 0,5 ms iki deadline
        for (;;) {
            for (std::uint32_t n = 0; n < o.flows; ++n) {
                if (state[n]) continue;
                mxlGrainInfo info{};
                const std::uint32_t* p = nullptr;
                const mxlStatus s = in[n].get(i, 0, info, p);
                if (s == MXL_ERR_OUT_OF_RANGE_TOO_EARLY) continue;
                --pending;
                if (s == MXL_STATUS_OK && (info.flags & MXL_GRAIN_FLAG_INVALID)) { st.flows[n].inc("in_invalid"); state[n] = 2; }
                else if (s == MXL_STATUS_OK) { src[n] = p; state[n] = 1; }
                else if (s == MXL_ERR_OUT_OF_RANGE_TOO_LATE) { st.flows[n].inc("too_late"); state[n] = 2; }
                else { st.flows[n].inc("read_errors"); st.flows[n].inc(("read_status_" + std::to_string(s)).c_str()); state[n] = 2; }
            }
            if (!pending || now_tai() >= deadline || stop.load()) break;
            mxlSleepForNs(500'000);
        }
        for (std::uint32_t n = 0; n < o.flows; ++n)
            if (!state[n]) st.flows[n].inc("late_fields");  // nesulaukta iki deadline → išėjimas necommit'inamas (INVALID)
        const std::uint64_t t_ready = now_tai();
        // 4: išėjimų slot'ai paruoštiems
        jobs.clear();
        job_flow.clear();
        for (std::uint32_t n = 0; n < o.flows; ++n) {
            if (state[n] != 1) continue;
            mxlStatus s{};
            std::uint32_t* d = out[n].open(i, s);
            if (!d) {
                if (s == MXL_ERR_INVALID_ARG && i <= out[n].last_committed) { st.proc["clock_backward"]++; continue; }
                st.fatal = "OpenGrainSlot status " + std::to_string(s);
                rc = 2;
                break;
            }
            jobs.push_back({src[n], d});
            job_flow.push_back(n);
        }
        if (rc) break;
        // 5: apdorojimas — G: VIENAS batch paleidimas; H: H2D → batch → D2H; C: gijų baseinas
        cudaError_t ce = cudaSuccess;
        const int field = inter ? static_cast<int>(i & 1) : -1;  // −1 — progresyvus kadras
        if (cpu) {
            pool.run(static_cast<int>(jobs.size()), [&](int k) { media::cpu::logo_field(jobs[k].src, jobs[k].dst, hlogo, dim, field); });
            if (o.inject_every && i % o.inject_every == 0 && !jobs.empty()) {
                jobs[0].dst[inject_word] ^= 1u << 3;
                st.proc["injected"]++;
            }
        } else if (host_gpu) {
            host_jobs = jobs;
            for (std::size_t k = 0; k < jobs.size() && !ce; ++k) {
                ce = cudaMemcpyAsync(s_in[job_flow[k]], jobs[k].src, gbytes, cudaMemcpyHostToDevice, stream);
                jobs[k] = {s_in[job_flow[k]], s_out[job_flow[k]]};
            }
            if (!ce && !jobs.empty()) ce = media::logo_fields(jobs.data(), static_cast<int>(jobs.size()), dlogo.d, dim, field, stream);
            for (std::size_t k = 0; k < jobs.size() && !ce; ++k)
                ce = cudaMemcpyAsync(host_jobs[k].dst, jobs[k].dst, gbytes, cudaMemcpyDeviceToHost, stream);
            jobs = host_jobs;  // injekcija ir tolesni žingsniai — host rodyklės
            if (!ce) ce = cudaStreamSynchronize(stream);
            if (!ce && o.inject_every && i % o.inject_every == 0 && !jobs.empty()) {
                jobs[0].dst[inject_word] ^= 1u << 3;
                st.proc["injected"]++;
            }
        } else if (!jobs.empty()) {
            ce = media::logo_fields(jobs.data(), static_cast<int>(jobs.size()), dlogo.d, dim, field, stream);
        }
        if (!cpu && !host_gpu && !ce && o.inject_every && i % o.inject_every == 0 && !jobs.empty()) {
            // savitikra: vienas bitas pirmo flow'o išėjime logotipo srityje (verify privalo aptikti)
            std::uint32_t w = 0;
            std::uint32_t* p = jobs[0].dst + inject_word;
            ce = cudaMemcpyAsync(&w, p, 4, cudaMemcpyDeviceToHost, stream);
            if (!ce) ce = cudaStreamSynchronize(stream);
            w ^= 1u << 3;
            if (!ce) ce = cudaMemcpyAsync(p, &w, 4, cudaMemcpyHostToDevice, stream);
            ++injected;
            st.proc["injected"]++;
        }
        if (!cpu && !ce) ce = cudaStreamSynchronize(stream);
        if (ce) { st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce); rc = 2; break; }
        // 6: lap patikra įėjimams, 7: commit
        const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
        for (std::size_t k = 0; k < jobs.size(); ++k) {
            const std::uint32_t n = job_flow[k];
            const std::uint64_t h = in[n].head();
            if (lapped(std::max(h == MXL_UNDEFINED_INDEX ? 0 : h, now_idx), i, grain_count)) {
                out[n].cancel();
                st.flows[n].inc("lapped");
                continue;
            }
            if (out[n].commit() != MXL_STATUS_OK) { st.fatal = "CommitGrain"; rc = 2; break; }
            st.flows[n].inc("fields_ok");
        }
        const std::uint64_t t_commit = now_tai();
        if (t_commit >= warm) {
            st.add_hist("proc_ms", t_commit - t_ready);  // paruošta → commit (kernel + sync + commit)
            st.add_hist("wait_ms", t_ready - t_wake);    // laukimas įėjimų
            st.add_hist("out_lateness", t_commit - mxlIndexToTimestamp(&rate, i));
        }
        if (st.first_index == ~0ull) st.first_index = i;
        st.last_index = i;
        ++i;
        const bool clk = cw.stepped();
        const std::uint64_t now2 = mxlGetCurrentIndex(&rate);
        if (now2 > i) {
            for (auto& f : st.flows) f.inc(clk ? "clock_skips" : (t_commit < warm ? "warmup_skips" : "late_skips"), now2 - i);
            i = now2;
        }
    }
    if (!cpu) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        for (auto* p : s_in) cudaFree(p);
        for (auto* p : s_out) cudaFree(p);
    }
    for (auto& v : out) v.release(inst);
    for (auto& v : in) v.release(inst);
    mxlDestroyInstance(inst);
    // praradimai = pajėgumo klaida (exit 4), jei nebuvo fatalios
    if (!rc)
        for (auto& f : st.flows)
            for (const char* k : {"late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips"})
                if (f.c.count(k) && f.c[k]) rc = 4;
    return finish(rc);
}

}  // namespace mxlv::func
