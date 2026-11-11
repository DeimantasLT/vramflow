// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// gen — apkrovos generatorius (architektūra GPU funkcijų §5.1): N × v210 grain'ų (--format: interlaced — laukai,
// progresyvus — kadrai; cuda-linear) vienu batch kernel'iu per grain'ą + pasirinktinai N × 48 kHz stereo audio (1 kHz
// −20 dBFS, kas sekundę 10 ms −6 dBFS pyptelėjimas; 50 Hz grain'ams sutampa su vaizdo sync blyksniu index mod 50 == 0).
// Garso mėginiai grain'ui — [sample_begin(i), sample_begin(i + 1)) (sveikaisiais skaičiais, 59,94 — 800 / 801).
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/cpu/cpu_media.hpp"
#include "media/media_kernels.hpp"

#include <algorithm>
#include <thread>

namespace mxlv::func {

int run_gen(const Opts& o) {
    Stats st;
    st.flows.resize(o.flows);
    auto& stop = stop_flag();
    const bool cpu = o.impl == "cpu";                 // C: CPU turinys į host flow, CUDA nekuriamas
    const bool host_gpu = !cpu && o.storage == "host";  // H: GPU turinys → D2H į host flow
    int dev = -1;
    if (!cpu) {
        dev = cuda_device(o.device_uuid);
        if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) { st.fatal = "CUDA įrenginys"; st.exit_code = 2; write_summary(o, "gen", st); return 2; }
        cudaFree(nullptr);
    }
    ThreadPool pool(cpu ? (o.threads > 0 ? o.threads : static_cast<int>(std::min<unsigned>(o.flows, std::thread::hardware_concurrency()))) : 1);
    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; st.exit_code = 2; write_summary(o, "gen", st); return 2; }
    mxlGarbageCollectFlows(inst);

    std::vector<VideoOut> vout(o.video ? o.flows : 0);
    std::vector<AudioOut> aout(o.audio ? o.flows : 0);
    std::string err;
    const std::string opts = cpu ? std::string{} : storage_options(o, dev);
    const vramflow::Rational grain = o.fmt.grain_rate();
    const mxlRational rate = mxl_rate(grain);
    const media::Dim dim{o.fmt.raster.w, o.fmt.raster.h};
    const bool inter = o.fmt.raster.interlaced;
    const std::size_t gbytes = media::grain_bytes(dim, inter);
    int rc = 0;
    // Išėjimai registruojami domeno nuomoje; jau egzistuojantis (gyvas writer'is po GC) — klaida su savininko run_id (§P6).
    for (std::uint32_t n = 0; n < vout.size() && !rc; ++n) {
        const std::string id = flow_id(kRoleGen, o.flow_base + n);
        if (!vout[n].create(inst, video_def(kRoleGen, o.flow_base + n, o.run_seed, o.fmt), opts, err, o.fmt)) {
            st.fatal = "video " + std::to_string(n) + ": " + err +
                       (vout[n].existed || vout[n].create_status == MXL_ERR_CONFLICT ? owner_note(o, id) : std::string{});
            rc = 2;
        } else {
            lease_output(id);
        }
    }
    for (std::uint32_t n = 0; n < aout.size() && !rc; ++n) {
        const std::string id = flow_id(kRoleAudio, o.flow_base + n);
        if (!aout[n].create(inst, audio_def(o.flow_base + n, o.run_seed), err, grain)) {
            st.fatal = "audio " + std::to_string(n) + ": " + err + (aout[n].existed ? owner_note(o, id) : std::string{});
            rc = 2;
        } else {
            lease_output(id);
        }
    }

    cudaStream_t stream{};
    if (!cpu) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    std::vector<media::GenJob> jobs(vout.size());
    std::vector<std::uint32_t*> host_dst(vout.size(), nullptr);  // H/C: host slot'ų rodyklės
    std::vector<std::uint32_t*> scratch;                          // H: device buferiai
    if (host_gpu)
        for (std::size_t n = 0; n < vout.size(); ++n) {
            std::uint32_t* p = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&p), gbytes) != cudaSuccess) { st.fatal = "scratch"; rc = 2; break; }
            scratch.push_back(p);
        }
    const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    const double warm = warm_tai_after(o.warmup_s);
    std::uint64_t i = mxlGetCurrentIndex(&rate) + 1;
    const double two_pi_f = 2.0 * 3.14159265358979323846 * 1000.0 / 48000.0;
    ClockWatch cw;
    cw.stepped();

    while (!rc && !stop.load() && now_tai() < end_tai) {
        if (wait_until(rate, i, 0, stop)) st.proc["clock_backward"]++;
        if (stop.load()) break;
        const std::uint64_t t_wake = now_tai();
        bool ok = true;
        for (std::uint32_t n = 0; n < vout.size(); ++n) {
            mxlStatus s{};
            jobs[n].dst = vout[n].open(i, s);
            jobs[n].flow = o.flow_base + n;
            if (!jobs[n].dst) {
                if (s == MXL_ERR_INVALID_ARG && i <= vout[n].last_committed) { st.proc["clock_backward"]++; ok = false; break; }
                st.fatal = "OpenGrainSlot status " + std::to_string(s);
                rc = 2;
                ok = false;
                break;
            }
        }
        if (!ok) {
            for (auto& v : vout) v.cancel();
            if (!rc) { i = vout.empty() ? i + 1 : vout[0].last_committed + 1; continue; }
            break;
        }
        cudaError_t ce = cudaSuccess;
        if (cpu) {
            media::GenParams base{o.run_seed, 0, o.complexity};
            base.rate_num = static_cast<std::uint32_t>(grain.num);
            base.rate_den = static_cast<std::uint32_t>(grain.den);
            pool.run(static_cast<int>(jobs.size()), [&](int n) {
                media::GenParams g = base;
                g.flow = jobs[n].flow;
                media::cpu::gen_field(g, i, dim, inter, jobs[n].dst);
            });
        } else if (host_gpu) {
            for (std::size_t n = 0; n < jobs.size(); ++n) {
                host_dst[n] = jobs[n].dst;
                jobs[n].dst = scratch[n];
            }
            if (!jobs.empty()) ce = media::gen_fields(jobs.data(), static_cast<int>(jobs.size()), o.run_seed, o.complexity, i, dim, inter,
                                                        static_cast<std::uint32_t>(grain.num), static_cast<std::uint32_t>(grain.den), stream);
            for (std::size_t n = 0; n < jobs.size() && !ce; ++n)
                ce = cudaMemcpyAsync(host_dst[n], scratch[n], gbytes, cudaMemcpyDeviceToHost, stream);
            if (!ce) ce = cudaStreamSynchronize(stream);
        } else {
            if (!jobs.empty()) ce = media::gen_fields(jobs.data(), static_cast<int>(jobs.size()), o.run_seed, o.complexity, i, dim, inter,
                                                        static_cast<std::uint32_t>(grain.num), static_cast<std::uint32_t>(grain.den), stream);
            if (!ce) ce = cudaStreamSynchronize(stream);
        }
        if (ce) { st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce); rc = 2; break; }
        for (std::uint32_t n = 0; n < vout.size(); ++n) {
            if (vout[n].commit() != MXL_STATUS_OK) { st.fatal = "CommitGrain"; rc = 2; break; }
            st.flows[n].inc("fields_written");
        }
        // audio: grain'as i apima mėginius [sample_begin(i), sample_begin(i + 1)) (1080i50: [i·960, (i+1)·960))
        const std::uint64_t a_end = vramflow::sample_begin(grain, i + 1);
        const auto a_count = static_cast<std::uint32_t>(a_end - vramflow::sample_begin(grain, i));
        for (std::uint32_t n = 0; n < aout.size(); ++n) {
            const mxlStatus as = aout[n].write(a_end, a_count, [&](int, std::uint64_t s) {
                const double amp = (s % 48000) < 480 ? 0.5011872 : 0.1;  // −6 dBFS pyptelėjimas, kitaip −20 dBFS
                return static_cast<float>(amp * std::sin(two_pi_f * static_cast<double>(s % 48000)));
            });
            if (as == MXL_STATUS_OK) st.flows[n].inc("audio_blocks");
            else st.flows[n].inc("audio_write_errors");
        }
        const std::uint64_t t_done = now_tai();
        if (t_done >= warm) {
            st.add_hist("gen_ms", t_done - t_wake);
            st.add_hist("commit_lateness", t_done - mxlIndexToTimestamp(&rate, i));
        }
        if (st.first_index == ~0ull) st.first_index = i;
        st.last_index = i;
        ++i;
        const bool clk = cw.stepped();
        const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
        if (now_idx > i) {
            st.proc[clk ? "clock_skips" : (t_done < warm ? "warmup_skips" : "late_skips")] += now_idx - i;
            i = now_idx;
        }
    }
    st.proc["clock_steps"] = cw.steps;
    if (!cpu) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        for (auto* p : scratch) cudaFree(p);
    }
    for (auto& v : vout) v.release(inst);
    for (auto& a : aout) a.release(inst);
    mxlDestroyInstance(inst);
    st.exit_code = rc;
    write_summary(o, "gen", st);
    std::fprintf(stderr, "[gen] baigta, exit %d %s\n", rc, st.fatal.c_str());
    return rc;
}

}  // namespace mxlv::func
