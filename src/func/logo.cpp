// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// logo — F-LOGO (architektūra GPU funkcijų §3, §5.2): N × MXL įėjimų (--format, numatytasis 1080i50) → logotipas
// įkepamas GPU (vienas batch kernel'is visiems N per grain'ą: laukas arba kadras) → N × MXL išėjimų. Visas kelias VRAM;
// CPU tik valdo.
// Niekas nedingsta tyliai: kiekvienas indeksas kiekvienam flow'ui yra fields_ok arba late/invalid/lapped/too_late.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "common/index_math.hpp"
#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "func/cuda_upload.hpp"
#include "func/engine_glue.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/media_kernels.hpp"
#include "media/cpu/cpu_media.hpp"
#include "media/media_ref.hpp"
#include "runtime/input_slot.hpp"

#include <thread>

namespace mxlv::func {

// Variklio ribos logo funkcijai (V10): izoliacija — Wr = P/4, M = 3, Late histerezė 8, futex laukimas; D = --deadline-ms.
constexpr EngineTiming kLogoTiming = EngineTiming::Isolation;

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

// Numatytoji logotipo vieta (--logo-x / --logo-y < 0): viršutinis dešinys kampas, 48 px paraštė dešinėje ir 64 px viršuje
// (1080: 1488, 64 — kaip anksčiau), skaičiuojama nuo --format rastro ir logotipo dydžio.
bool load_logo(Opts& o, media::ref::Logo& logo, std::string& err) {
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
    if (o.logo_x < 0) o.logo_x = (fw - logo.w - 48) & ~1;
    if (o.logo_y < 0) o.logo_y = 64;
    if (o.logo_x % 2 || o.logo_x < 0 || o.logo_y < 0 || o.logo_x + logo.w > fw || o.logo_y + logo.h > fh) {
        err = "logotipas " + std::to_string(logo.w) + "x" + std::to_string(logo.h) + " ties " + std::to_string(o.logo_x) + "," +
              std::to_string(o.logo_y) + " netelpa į " + vramflow::format_name(o.fmt) + " kadrą " + std::to_string(fw) + "x" +
              std::to_string(fh) + " arba X nelyginis";
        return false;
    }
    return true;
}

namespace {
// logo kaip TickFunction (planas §6 D1, V9): N transformacijos vienetų (1 įėjimas → 1 išėjimas). process — visi tuo metu
// paruošti vienetai vienu batch'u (G: vienas kernel'is; H: H2D → kernel'is → D2H; C: gijų baseinas); lap patikrą,
// commit'ą ir spragas daro variklis. Santraukos raktai — tie patys kaip prieš migraciją.
struct LogoFn final : engine::TickFunction {
    const Opts& o;
    Stats& st;
    const engine::StatsHub& hub;
    const engine::TickEngine& eng;
    bool cpu, host_gpu, inter;
    media::Dim dim;
    std::size_t gbytes, inject_word;
    ThreadPool& pool;
    DeviceLogo& dlogo;
    const media::cpu::LogoPlanes& hlogo;
    cudaStream_t stream;
    std::vector<std::uint32_t*>& s_in;
    std::vector<std::uint32_t*>& s_out;
    double warm;
    std::vector<media::LogoJob> jobs, host_jobs;
    std::vector<std::uint32_t> job_flow;
    std::vector<engine::InCounters> seen;  // įėjimų skaitikliai po paskutinio vieneto sprendimo (spragos priežasčiai)
    TickTimes times;  // proc_ms / wait_ms / held_ms / out_lateness — atskirai kiekvienam indeksui (pernešimas)
    std::uint64_t injected_at = ~0ull;
    std::uint64_t missing_at = ~0ull;  // paskutinis indeksas su GapBlocked vienetu (--test-fault-proc-delay-ms)

    LogoFn(const Opts& o_, Stats& st_, const engine::StatsHub& hub_, const engine::TickEngine& eng_, bool cpu_, bool host_gpu_,
           bool inter_, media::Dim dim_, std::size_t gbytes_, std::size_t inject_word_, ThreadPool& pool_, DeviceLogo& dlogo_,
           const media::cpu::LogoPlanes& hlogo_, cudaStream_t stream_, std::vector<std::uint32_t*>& s_in_,
           std::vector<std::uint32_t*>& s_out_, double warm_)
        : o(o_), st(st_), hub(hub_), eng(eng_), cpu(cpu_), host_gpu(host_gpu_), inter(inter_), dim(dim_), gbytes(gbytes_),
          inject_word(inject_word_), pool(pool_), dlogo(dlogo_), hlogo(hlogo_), stream(stream_), s_in(s_in_), s_out(s_out_),
          warm(warm_), seen(o_.flows) {
        jobs.reserve(o.flows);
        host_jobs.reserve(o.flows);
        job_flow.reserve(o.flows);
    }

    void begin_tick(engine::Index i, engine::TickContext&) override {
        times.wake(i, now_tai());
    }

    bool process(const engine::Batch& b, engine::TickContext& ctx) override {
        const std::uint64_t i = b.index;
        times.phase_begin(i, now_tai());
        jobs.clear();
        job_flow.clear();
        for (const engine::UnitTick& ut : b.units) {
            jobs.push_back({static_cast<const std::uint32_t*>(ut.in[0].ptr), static_cast<std::uint32_t*>(ut.out[0])});
            job_flow.push_back(ut.unit);
        }
        // Testinė injekcija (--test-fault-proc-delay-ms): šio indekso A fazėje, kai kurio nors vieneto įėjimo nėra (spraga
        // atidarant — GapBlocked: rašytojas dingo / užstrigo) — vėlinimas apdorojime (blogos funkcijos modelis: sveiki vėluoja
        // dėl sugedusio įėjimo). Vien vėluojantis įėjimas (B fazė) jos nesukelia. Izoliacijos vartai tai privalo aptikti.
        if (o.test_fault_proc_delay_ms && b.phase == 0 && missing_at == i)
            std::this_thread::sleep_for(std::chrono::milliseconds(o.test_fault_proc_delay_ms));
        const bool inject = o.inject_every && i % o.inject_every == 0 && !jobs.empty() && injected_at != i;
        if (inject) injected_at = i;
        cudaError_t ce = cudaSuccess;
        const int field = inter ? static_cast<int>(i & 1) : -1;  // −1 — progresyvus kadras
        if (cpu) {
            pool.run(static_cast<int>(jobs.size()), [&](int k) { media::cpu::logo_field(jobs[k].src, jobs[k].dst, hlogo, dim, field); });
            if (inject) {
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
            jobs = host_jobs;  // injekcija — host rodyklės
            if (!ce) ce = cudaStreamSynchronize(stream);
            if (!ce && inject) {
                jobs[0].dst[inject_word] ^= 1u << 3;
                st.proc["injected"]++;
            }
        } else if (!jobs.empty()) {
            ce = media::logo_fields(jobs.data(), static_cast<int>(jobs.size()), dlogo.d, dim, field, stream);
            if (!ce && inject) {
                // savitikra: vienas bitas pirmo batch'o flow'o išėjime logotipo srityje (verify privalo aptikti)
                std::uint32_t w = 0;
                std::uint32_t* p = jobs[0].dst + inject_word;
                ce = cudaMemcpyAsync(&w, p, 4, cudaMemcpyDeviceToHost, stream);
                if (!ce) ce = cudaStreamSynchronize(stream);
                w ^= 1u << 3;
                if (!ce) ce = cudaMemcpyAsync(p, &w, 4, cudaMemcpyHostToDevice, stream);
                st.proc["injected"]++;
            }
        }
        if (!cpu && !ce) ce = cudaStreamSynchronize(stream);
        if (ce) {
            st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce);
            ctx.fatal(2, st.fatal);
            return false;
        }
        return true;
    }

    // Variklio praleisti indeksai (T11): praradimai (late_skips; skaitiklį rašo to_summary) — į loss_s sekundėms.
    void on_skip(std::uint32_t unit, engine::Index from, engine::Index to, engine::TickContext& ctx) override {
        if (!ctx.skip_is_loss()) return;
        const mxlRational r = mxl_rate(o.fmt.grain_rate());
        for (engine::Index i = from; i < to; ++i) st.flows[unit].loss_s.add(mxlIndexToTimestamp(&r, i) / 1'000'000'000);
    }

    void end_unit(const engine::UnitTick& ut, engine::Outcome oc, engine::TickContext&) override {
        const std::uint32_t n = ut.unit;
        FlowStats& f = st.flows[n];
        const engine::InCounters& c = hub.in(eng.in_id(n, 0));
        engine::InCounters& s = seen[n];
        const mxlRational r = mxl_rate(o.fmt.grain_rate());
        const std::uint64_t ti = mxlIndexToTimestamp(&r, ut.index);
        switch (oc) {
            case engine::Outcome::Ready: {
                f.inc("fields_ok");
                const std::uint64_t t_commit = now_tai();
                times.unit_done(ut.index, t_commit, true);
                f.note_lateness(t_commit, ti, warm);
                break;
            }
            case engine::Outcome::Lapped:
                f.loss("lapped", ti);
                times.unit_done(ut.index, now_tai(), false);
                break;
            case engine::Outcome::GapBlocked:  // gamintojo nėra / užstrigęs
                f.loss("missing_fields", ti);
                missing_at = ut.index;
                break;
            case engine::Outcome::GapClock: f.inc("clock_skips"); break;      // laikrodžio žingsnis — aplinka
            case engine::Outcome::Skipped: st.proc["clock_backward"]++; break;  // AlreadyCommitted — laikrodis atgal (M11)
            case engine::Outcome::OutputError:  // open / commit klaida (ilgalaikė — fatalu)
                f.inc("output_errors");
                times.unit_done(ut.index, now_tai(), false);
                break;
            case engine::Outcome::GapNoContent: break;
            case engine::Outcome::GapLate:
                // Kodėl: įėjimo skaitiklių pokytis nuo paskutinio šio vieneto sprendimo
                if (c.invalid > s.invalid) f.loss("in_invalid", ti);
                else if (c.too_late > s.too_late) f.loss("too_late", ti);
                else if (c.read_errors > s.read_errors) f.loss("read_errors", ti);
                else if (c.detaches > s.detaches) {
                    f.inc("detached");
                    f.loss("missing_fields", ti);
                } else f.loss("late_fields", ti);  // nesulaukta iki ribos → išėjimas necommit'inamas
                break;
        }
        s = c;
        if (c.stalls) f.c["stalls"] = c.stalls;
    }

    void end_tick(engine::Index i, engine::TickContext&) override {
        const TickTimes::Result t = times.end(i, now_tai());
        if (static_cast<double>(t.end) >= warm) {
            st.add_hist("proc_ms", t.proc);  // tik apdorojimas: kernel + sync + commit (Σ fazių), be laukimo tarp jų
            st.add_hist("wait_ms", t.wait);  // laukimas įėjimų iki pirmo apdorojimo
            if (t.phases > 1) st.add_hist("held_ms", t.held);  // B / C fazė: laukta vėluojančių (ne apdorojimas)
            const mxlRational r = mxl_rate(o.fmt.grain_rate());
            st.add_hist("out_lateness", t.end - mxlIndexToTimestamp(&r, i));
        }
    }
};
}  // namespace

int run_logo(const Opts& o_in) {
    Opts o = o_in;  // load_logo gali nustatyti numatytąją logotipo vietą
    Stats st;
    st.flows.resize(o.flows);
    st.set_seconds(o.seconds_kept);  // sekundžių žiedai — dabar (ne tick'e)
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
    // Įėjimai (planas §P6): paleidimo tvarka nesvarbi — laukiama pagal --input-wait-ms (numatyta neribotai); dingęs
    // gamintojas → InputSlot Detached (indeksai skaičiuojami missing_fields), grįžęs → re-attach (reattached).
    std::vector<runtime::InputSlot> in;
    in.reserve(o.flows);
    for (std::uint32_t n = 0; n < o.flows; ++n) in.push_back(make_input(o, flow_id(o.in_role, o.flow_base + n), &o.fmt));
    const std::vector<runtime::InputSlot*> in_ptr = slot_ptrs(in);
    if (const int a = attach_inputs(in_ptr, inst, o, "logo", st.fatal)) {  // neatsirado per ribą (exit 4) arba sustabdyta (exit 0)
        for (auto& s : in) s.release();
        mxlDestroyInstance(inst);
        return finish(exit_code(a));
    }
    // Nužudyto ankstesnio logo pasenę išėjimai (be writer'io) pašalinami — kitaip perkrautas logo jų nesukurtų.
    mxlGarbageCollectFlows(inst);
    std::vector<VideoOut> out(o.flows);
    int rc = 0;
    const std::string opts = cpu ? std::string{} : storage_options(o, dev);
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n)
        if (!create_output(inst, out[n], flow_id(kRoleLogo, o.flow_base + n), video_def(kRoleLogo, o.flow_base + n, o.run_seed, o.fmt), opts,
                           o.fmt, o, err)) { st.fatal = "out " + std::to_string(n) + ": " + err; rc = 2; }
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n)
        if (in[n].in().cuda == (cpu || host_gpu)) { st.fatal = "įėjimo storage neatitinka --impl/--storage"; rc = 1; }

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
    // Bendras tick variklis (planas V9): įėjimai — MxlVideoInput ant InputSlot, išėjimai — MxlVideoOutput ant VideoOut.
    std::vector<engine::MxlVideoInput> ports_in;
    std::vector<engine::MxlVideoOutput> ports_out;
    ports_in.reserve(o.flows);
    ports_out.reserve(o.flows);
    for (std::uint32_t n = 0; n < o.flows; ++n) {
        ports_in.emplace_back(in[n], inst);
        ports_out.emplace_back(out[n], engine::host_copy(), nullptr, nullptr);
    }
    // Sugedęs vienetas — jo išėjimas atlaisvinamas (K-T11); atkūrimas (D8) — flow'as sukuriamas iš naujo atkūrimo gijoje.
    set_release(ports_out, out, inst, [&](std::size_t n) {
        if (!cpu) cudaSetDevice(dev);
        std::string e;
        const bool ok = create_output(inst, out[n], flow_id(kRoleLogo, o.flow_base + n), video_def(kRoleLogo, o.flow_base + n, o.run_seed, o.fmt),
                                      opts, o.fmt, o, e);
        if (!ok) log_line("logo", "išėjimo " + std::to_string(n) + " atkurti nepavyko: " + e);
        return ok;
    });
    engine::SystemClock clk;
    EngineStats es(o, "logo");
    engine::StatsHub& hub = es.hub;
    engine::TickEngine eng(engine_config(o, kLogoTiming, o.read_offset_us * 1000), clk, hub, stop);
    std::vector<FaultyOutput> faulty;
    std::vector<engine::OutputPort*> optr;
    for (auto& p : ports_out) optr.push_back(&p);
    optr = output_ports(optr, o, faulty);
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n) {
        engine::UnitSpec u;
        u.name = flow_id(kRoleLogo, o.flow_base + n);
        u.in.push_back(engine::InputSpec{&ports_in[n], {}, 1, engine::Need::Required, engine::InMode::Exact});
        u.out.push_back(optr[n]);
        std::uint32_t id = 0;
        if (!eng.add_unit(u, id, err)) { st.fatal = "add_unit: " + err; rc = 2; }
    }
    LogoFn fn(o, st, hub, eng, cpu, host_gpu, inter, dim, gbytes, inject_word, pool, dlogo, hlogo, stream, s_in, s_out,
              warm_tai_after(o.warmup_s));
    bool engine_loss = false;
    std::uint32_t failed = 0;
    if (!rc) {
        const engine::RunResult res = eng.run(fn);
        es.finish();  // paskutinė statistikos eilutė (Σ eilučių == santrauka)
        engine_loss = res.any_loss;
        failed = res.failed_units;
        if (res.reason == engine::EndReason::Fatal) {
            if (st.fatal.empty()) st.fatal = res.fatal;
            rc = res.exit_code ? res.exit_code : 2;
        } else if (res.reason == engine::EndReason::FormatChange) {
            st.fatal = res.fatal;
            rc = kExitFormatChange;
        }
        to_summary(hub, eng, res, o.flows, SkipScope::PerFlow, st);
    }
    if (!cpu) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        for (auto* p : s_in) cudaFree(p);
        for (auto* p : s_out) cudaFree(p);
    }
    for (auto& v : out) v.release(inst);
    for (std::uint32_t n = 0; n < o.flows; ++n) {
        st.flows[n].c["reattached"] = in[n].reattached();
        in[n].release();
    }
    mxlDestroyInstance(inst);
    // praradimai (pajėgumo / tiekimo; missing_fields — įėjimo spraga) ir variklio (RunResult::any_loss, ir išėjimo klaidos):
    // griežtame režime exit 4, tarnyboje — tik santraukoje (exit_code_for, --strict-exit)
    bool loss = engine_loss;
    for (auto& f : st.flows)
        for (const char* k : {"late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips", "missing_fields",
                              "output_errors"})
            if (f.c.count(k) && f.c[k]) loss = true;
    return finish(exit_code_for(o, rc, loss, failed, st));
}

}  // namespace mxlv::func
