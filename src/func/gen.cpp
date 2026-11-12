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

#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "func/engine_glue.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/cpu/cpu_media.hpp"
#include "media/media_kernels.hpp"

#include <algorithm>
#include <thread>

namespace mxlv::func {

namespace {
// gen kaip TickFunction (planas §6 D1: šaltinio vienetai). process — turinys visiems paruoštiems vienetams vienu batch'u
// (G: vienas kernel'is; H: kernel'is → D2H; C: gijų baseinas); commit'ina variklis.
struct GenFn final : engine::TickFunction {
    const Opts& o;
    Stats& st;
    bool cpu, host_gpu;
    ThreadPool& pool;
    std::vector<media::GenJob>& jobs;
    std::vector<std::uint32_t*>& host_dst;
    std::vector<std::uint32_t*>& scratch;
    cudaStream_t stream;
    std::vector<AudioOut>& aout;
    vramflow::Rational grain;
    media::Dim dim;
    bool inter;
    std::size_t gbytes;
    double warm;
    std::uint64_t first = ~0ull, last = 0;
    std::uint64_t t_wake = 0;
    std::vector<std::uint32_t> flow_of;  // batch'o eilė → flow numeris
    // --audio-late-ms: grain'o garsas rašomas ne iškart po video commit'o, o pirmo tick'o atidarymo metu, kai praėjo
    // T(i) + audio_late_ms (realaus laiko garso rašytojo modelis: grain'o garsas baigiamas jo periodo pabaigoje, T(i) + P).
    struct PendingAudio {
        std::uint32_t unit;
        std::uint64_t index;
    };
    std::vector<PendingAudio> pending;

    GenFn(const Opts& o_, Stats& st_, bool cpu_, bool host_gpu_, ThreadPool& pool_, std::vector<media::GenJob>& jobs_,
          std::vector<std::uint32_t*>& host_dst_, std::vector<std::uint32_t*>& scratch_, cudaStream_t stream_,
          std::vector<AudioOut>& aout_, vramflow::Rational grain_, media::Dim dim_, bool inter_, std::size_t gbytes_, double warm_)
        : o(o_), st(st_), cpu(cpu_), host_gpu(host_gpu_), pool(pool_), jobs(jobs_), host_dst(host_dst_), scratch(scratch_),
          stream(stream_), aout(aout_), grain(grain_), dim(dim_), inter(inter_), gbytes(gbytes_), warm(warm_) {
        flow_of.reserve(o.flows);
        pending.reserve(static_cast<std::size_t>(o.flows) * 8);
    }

    void begin_tick(engine::Index, engine::TickContext&) override {
        t_wake = now_tai();
        if (pending.empty()) return;
        const mxlRational r = mxl_rate(grain);
        std::size_t keep = 0;
        for (const PendingAudio& p : pending) {
            if (mxlIndexToTimestamp(&r, p.index) + std::uint64_t{o.audio_late_ms} * 1'000'000 <= t_wake) write_audio(p.unit, p.index);
            else pending[keep++] = p;
        }
        pending.resize(keep);
    }

    // audio: grain'as i apima mėginius [sample_begin(i), sample_begin(i + 1)) (1080i50: [i·960, (i+1)·960))
    void write_audio(std::uint32_t n, std::uint64_t i) {
        const std::uint64_t a_end = vramflow::sample_begin(grain, i + 1);
        const auto a_count = static_cast<std::uint32_t>(a_end - vramflow::sample_begin(grain, i));
        constexpr double two_pi_f = 2.0 * 3.14159265358979323846 * 1000.0 / 48000.0;
        const mxlStatus as = aout[n].write(a_end, a_count, [&](int, std::uint64_t s) {
            const double amp = (s % 48000) < 480 ? 0.5011872 : 0.1;  // −6 dBFS pyptelėjimas, kitaip −20 dBFS
            return static_cast<float>(amp * std::sin(two_pi_f * static_cast<double>(s % 48000)));
        });
        if (as == MXL_STATUS_OK) st.flows[n].inc("audio_blocks");
        else st.flows[n].inc("audio_write_errors");
    }

    // Variklio praleisti indeksai (T11): eilutė su TAI laiku (tik pirmam vienetui — visi praleidžia tuos pačius) —
    // izoliacijos testai aukštupio praradimus priskiria tik gedimo langui (kaip ir vėlius commit'us).
    void on_skip(std::uint32_t unit, engine::Index from, engine::Index to, engine::TickContext&) override {
        if (unit != 0) return;
        std::fprintf(stderr, "[gen] praleista: indeksai %llu…%llu (%llu), TAI %llu\n", static_cast<unsigned long long>(from),
                     static_cast<unsigned long long>(to - 1), static_cast<unsigned long long>(to - from),
                     static_cast<unsigned long long>(now_tai()));
    }

    bool process(const engine::Batch& b, engine::TickContext& ctx) override {
        const std::uint64_t i = b.index;
        jobs.clear();
        flow_of.clear();
        for (const engine::UnitTick& ut : b.units) {
            if (ut.out.empty()) continue;  // tik garsas
            jobs.push_back({static_cast<std::uint32_t*>(ut.out[0]), o.flow_base + ut.unit});
            flow_of.push_back(ut.unit);
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
                jobs[n].dst = scratch[flow_of[n]];
            }
            if (!jobs.empty())
                ce = media::gen_fields(jobs.data(), static_cast<int>(jobs.size()), o.run_seed, o.complexity, i, dim, inter,
                                       static_cast<std::uint32_t>(grain.num), static_cast<std::uint32_t>(grain.den), stream);
            for (std::size_t n = 0; n < jobs.size() && !ce; ++n)
                ce = cudaMemcpyAsync(host_dst[n], jobs[n].dst, gbytes, cudaMemcpyDeviceToHost, stream);
            if (!ce) ce = cudaStreamSynchronize(stream);
        } else {
            if (!jobs.empty())
                ce = media::gen_fields(jobs.data(), static_cast<int>(jobs.size()), o.run_seed, o.complexity, i, dim, inter,
                                       static_cast<std::uint32_t>(grain.num), static_cast<std::uint32_t>(grain.den), stream);
            if (!ce) ce = cudaStreamSynchronize(stream);
        }
        if (ce) {
            st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce);
            ctx.fatal(2, st.fatal);
            return false;
        }
        return true;
    }

    void end_unit(const engine::UnitTick& ut, engine::Outcome oc, engine::TickContext&) override {
        if (oc == engine::Outcome::Skipped) {  // AlreadyCommitted — laikrodis atgal (M11)
            st.proc["clock_backward"]++;
            return;
        }
        if (oc == engine::Outcome::OutputError) {  // open / commit klaida (ilgalaikė — variklis baigia fatalu, exit 2)
            st.flows[ut.unit].inc("output_errors");
            return;
        }
        if (oc != engine::Outcome::Ready) return;
        const std::uint32_t n = ut.unit;
        if (o.video) st.flows[n].inc("fields_written");
        if (n >= aout.size()) return;
        if (o.audio_late_ms) pending.push_back({n, ut.index});
        else write_audio(n, ut.index);
    }

    void end_tick(engine::Index i, engine::TickContext&) override {
        const std::uint64_t t_done = now_tai();
        const mxlRational r = mxl_rate(grain);
        const std::uint64_t late = t_done - mxlIndexToTimestamp(&r, i);
        if (static_cast<double>(t_done) >= warm) {
            st.add_hist("gen_ms", t_done - t_wake);
            st.add_hist("commit_lateness", late);
            std::uint64_t& m = st.commit_lateness_s[mxlIndexToTimestamp(&r, i) / 1'000'000'000];
            m = std::max<std::uint64_t>(m, late / 1000);
        }
        // Vėlesnis nei --deadline-ms commit'as: vartotojai su tokia riba šio indekso nesulaukia. Skaitiklis (visam
        // paleidimui, ne tik po warmup) ir eilutė su TAI laiku — izoliacijos testai priskiria praradimus aukštupiui.
        if (late > std::uint64_t{o.deadline_ms} * 1'000'000) {
            st.proc["late_commits"]++;
            std::fprintf(stderr, "[gen] vėlus commit: indeksas %llu, %.3f ms po T(i), TAI %llu\n", static_cast<unsigned long long>(i),
                         static_cast<double>(late) / 1e6, static_cast<unsigned long long>(t_done));
        }
        if (first == ~0ull) first = i;
        last = i;
    }
};
}  // namespace

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
    const media::Dim dim{o.fmt.raster.w, o.fmt.raster.h};
    const bool inter = o.fmt.raster.interlaced;
    const std::size_t gbytes = media::grain_bytes(dim, inter);
    int rc = 0;
    // Išėjimai registruojami domeno nuomoje; jau egzistuojantis (gyvas writer'is po GC) — klaida su savininko run_id (§P6).
    for (std::uint32_t n = 0; n < vout.size() && !rc; ++n)
        if (!create_output(inst, vout[n], flow_id(kRoleGen, o.flow_base + n), video_def(kRoleGen, o.flow_base + n, o.run_seed, o.fmt), opts,
                           o.fmt, o, err)) { st.fatal = "video " + std::to_string(n) + ": " + err; rc = 2; }
    for (std::uint32_t n = 0; n < aout.size() && !rc; ++n)
        if (!create_output(inst, aout[n], flow_id(kRoleAudio, o.flow_base + n), audio_def(o.flow_base + n, o.run_seed), grain, o, err)) {
            st.fatal = "audio " + std::to_string(n) + ": " + err;
            rc = 2;
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
    // Bendras tick variklis (planas V8): N šaltinio vienetų (be įėjimų), išėjimas — video flow'as (jei yra), garsas —
    // end_unit. Laikas — legacy (gen niekada nelaukia įėjimų; ro = 0 — commit'as ties T(i), kaip prieš migraciją).
    std::vector<engine::MxlVideoOutput> ports;
    ports.reserve(vout.size());
    for (auto& v : vout) ports.emplace_back(v, engine::host_copy(), nullptr, nullptr);
    set_release(ports, vout, inst);  // sugedęs vienetas — jo video flow'as atlaisvinamas (K-T11)
    std::vector<FaultyOutput> faulty;
    std::vector<engine::OutputPort*> optr;
    for (auto& p : ports) optr.push_back(&p);
    optr = output_ports(optr, o, faulty);
    engine::SystemClock clk;
    engine::StatsHub hub;
    engine::TickEngine eng(engine_config(o, EngineTiming::Legacy, 0), clk, hub, stop);
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n) {
        engine::UnitSpec u;
        u.name = flow_id(kRoleGen, o.flow_base + n);
        if (o.video) u.out.push_back(optr[n]);
        std::uint32_t id = 0;
        if (!eng.add_unit(u, id, err)) { st.fatal = "add_unit: " + err; rc = 2; }
    }
    GenFn fn(o, st, cpu, host_gpu, pool, jobs, host_dst, scratch, stream, aout, grain, dim, inter, gbytes, warm_tai_after(o.warmup_s));
    bool loss = false;
    std::uint32_t failed = 0;
    if (!rc) {
        const engine::RunResult res = eng.run(fn);
        loss = res.any_loss;  // praleisti indeksai, išėjimo klaidos — gen taip pat (anksčiau gen exit jų nerodė)
        failed = res.failed_units;
        if (res.reason == engine::EndReason::Fatal) {
            if (st.fatal.empty()) st.fatal = res.fatal;
            rc = res.exit_code ? res.exit_code : 2;
        }
        to_summary(hub, eng, res, o.flows, SkipScope::Process, st);
        if (fn.first != ~0ull) {  // first / last — iš įvykdytų tick'ų (kaip anksčiau)
            st.first_index = fn.first;
            st.last_index = fn.last;
        }
    }
    if (!cpu) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        for (auto* p : scratch) cudaFree(p);
    }
    for (auto& v : vout) v.release(inst);
    for (auto& a : aout) a.release(inst);
    mxlDestroyInstance(inst);
    rc = exit_code_for(o, rc, loss, failed, st);  // --strict-exit: kaip kitos funkcijos
    st.exit_code = rc;
    write_summary(o, "gen", st);
    std::fprintf(stderr, "[gen] baigta, exit %d %s\n", rc, st.fatal.c_str());
    return rc;
}

}  // namespace mxlv::func
