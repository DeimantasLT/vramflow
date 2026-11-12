// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// composite — kompozitorius: kiekvienam išėjimui n bazė (v210 iš `gen`, arba jokios) + K grafikos sluoksnių (v210a iš `gfxgen` ar
// kito composite) → v210 / v210a MXL išėjimas. Bendras tick variklis (V11, izoliacijos ribos kaip logo): viena gija valdo
// visus MXL handle'us; paruošti išėjimai apdorojami vienu CPU / CUDA batch'u fazei, užstrigęs įėjimas kitų nelaiko.
//
// Sprendimai:
//  - Geometrija statinė: rastrai perskaitomi iš flow aprašų, kiekvienas darbas tikrinamas bendru gfx::composite_job_valid
//    PRADŽIOJE (nelyginis y interlaced rastre, x, rastrų neatitikimas, `--base none` su v210 ...) → startas nepavyksta su aiškia
//    klaida (exit 1). Vykdymo metu išėjimą iš batch'o pašalina tik trūkstamas / vėlyvas / INVALID įėjimas.
//  - Storage: VISI įėjimai turi būti tokioje pačioje saugykloje kaip --storage (C, H: host; G: cuda-linear), kaip `logo`.
//    Mišrios saugyklos (pvz. bazė host, sluoksniai cuda) nepalaikomos: startas nepavyksta (exit 1). H variantas kiekvieną
//    įėjimą (host) pakelia į device scratch, apdoroja ir rezultatą nukopijuoja į host išėjimo slot'ą.
//  - Bazė `none` (tik v210a išėjimas): išėjimo rastras = --format rastras (numatytasis 1920×1080 interlaced); kitaip išėjimo
//    rastras = bazės (w, h, interlace), o bazė turi atitikti --format. Visų įėjimų grain'o ritmas turi sutapti su --format.
//  - `--layer-flow K:UUID`: K — sluoksnio indeksas nuo 0 (0…layers-1); tas pats flow VISIEMS išėjimams n (grandinei N=1).
//  - [PF-3, §P6] įėjimai — runtime::InputSlot: paleidžiant laukiama pagal --input-wait-ms (numatyta neribotai); gamintojas
//    dingo (MXL_ERR_FLOW_INVALID arba nebėra gyvo writer'io) → slot'as atjungiamas ir prijungiamas iš naujo neblokuojančiai
//    (bandymas ne dažniau kaip kas 100 ms, tarp tick'ų; skaitiklis `reopened`). Išėjimas, kurio įėjimas trūksta, tuo metu
//    necommit'inamas (`missing_fields`), bet funkcija tęsia. Grįžęs įėjimas privalo turėti tą patį formatą ir saugyklą; jei ne —
//    --on-format-change exit (numatyta): `reopen_mismatch`, exit 6 su abiem formatais; wait: laukiama, kol grįš tinkamas.
//  - Skaitikliai (kiekvienam išėjimui): fields_ok, late_fields (išėjimas necommit'intas — įėjimas nesulaukė iki termino; kartą
//    per išėjimą), too_late / in_invalid / read_errors / flow_invalid (įėjimų įvykiai), missing_fields, reopened, lapped.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "common/index_math.hpp"
#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "func/engine_glue.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "gfx/cpu/gfx_cpu.hpp"
#include "gfx/gfx_kernels.hpp"
#include "gfx/raster.hpp"
#include "runtime/input_slot.hpp"

namespace mxlv::func {
namespace {

namespace gfx = vramflow::gfx;
using gfx::CompositeJob;
using gfx::Layer;
using gfx::Raster;

}  // namespace

// Bendra su verify (--mode composite).
bool parse_xy(const std::string& s, int& x, int& y) {
    int n = 0;
    return std::sscanf(s.c_str(), "%d,%d%n", &x, &y, &n) == 2 && static_cast<std::size_t>(n) == s.size();
}

namespace {

// "k:UUID" → ids[k]
bool parse_layer_flows(const Opts& o, std::vector<std::string>& ids, std::string& err) {
    ids.assign(o.layers, std::string{});
    for (const std::string& s : o.layer_flow) {
        const std::size_t c = s.find(':');
        const std::string ks = c == std::string::npos ? std::string{} : s.substr(0, c);  // vardinis lokalus: strtol `end` rodo į jo buferį
        char* end = nullptr;
        const long k = c == std::string::npos || c == 0 ? -1 : std::strtol(ks.c_str(), &end, 10);
        if (c == std::string::npos || c == 0 || *end != '\0' || k < 0 || k >= static_cast<long>(o.layers) || c + 1 >= s.size()) {
            err = "--layer-flow '" + s + "': tikimasi K:UUID, K = 0…" + std::to_string(o.layers - 1);
            return false;
        }
        if (!ids[static_cast<std::size_t>(k)].empty()) {
            err = "--layer-flow: sluoksnis " + std::to_string(k) + " nurodytas daugiau nei kartą";
            return false;
        }
        ids[static_cast<std::size_t>(k)] = s.substr(c + 1);
    }
    return true;
}

}  // namespace

namespace {
// composite kaip TickFunction (planas V11): N transformacijos vienetų ([bazė] + K sluoksnių → 1 išėjimas). process — visi
// tuo metu paruošti vienetai vienu batch'u (G: composite_batch kMaxBatch dalimis; H: H2D → batch → D2H; C: CPU batch);
// lap patikrą, commit'ą ir spragas daro variklis. Santraukos raktai — tie patys kaip prieš migraciją.
struct CompositeFn final : engine::TickFunction {
    const Opts& o;
    Stats& st;
    const engine::StatsHub& hub;
    const engine::TickEngine& eng;
    std::vector<runtime::InputSlot>& in;
    bool cpu, host_gpu, has_base;
    std::uint32_t K, per;
    const std::vector<int>& lx;
    const std::vector<int>& ly;
    const std::vector<float>& lop;
    std::vector<Layer>& lay;
    const std::vector<Raster>& out_r;
    std::vector<std::uint8_t*>& s_in;
    std::vector<std::uint8_t*>& s_out;
    ThreadPool& pool;
    cudaStream_t stream;
    double warm;
    std::vector<CompositeJob> jobs;
    std::vector<std::uint32_t> job_out;
    std::vector<std::uint8_t*> host_dst;
    std::vector<engine::InCounters> seen;  // įėjimų skaitikliai po paskutinio vieneto sprendimo (spragos priežasčiai)
    std::vector<std::uint64_t> seen_reatt;
    std::uint64_t t_wake = 0, t_ready = 0, t_commit = 0;
    bool processed = false;

    CompositeFn(const Opts& o_, Stats& st_, const engine::StatsHub& hub_, const engine::TickEngine& eng_, std::vector<runtime::InputSlot>& in_,
                bool cpu_, bool host_gpu_, bool has_base_, std::uint32_t K_, std::uint32_t per_, const std::vector<int>& lx_,
                const std::vector<int>& ly_, const std::vector<float>& lop_, std::vector<Layer>& lay_, const std::vector<Raster>& out_r_,
                std::vector<std::uint8_t*>& s_in_, std::vector<std::uint8_t*>& s_out_, ThreadPool& pool_, cudaStream_t stream_, double warm_)
        : o(o_), st(st_), hub(hub_), eng(eng_), in(in_), cpu(cpu_), host_gpu(host_gpu_), has_base(has_base_), K(K_), per(per_), lx(lx_),
          ly(ly_), lop(lop_), lay(lay_), out_r(out_r_), s_in(s_in_), s_out(s_out_), pool(pool_), stream(stream_), warm(warm_),
          seen(in_.size()), seen_reatt(in_.size(), 0) {
        jobs.reserve(o.flows);
        job_out.reserve(o.flows);
        host_dst.reserve(o.flows);
    }

    void begin_tick(engine::Index i, engine::TickContext&) override {
        t_wake = now_tai();
        processed = false;
        t_commit = 0;
        const mxlRational r = mxl_rate(o.fmt.grain_rate());
        if (t_wake < mxlIndexToTimestamp(&r, i)) st.proc["early_wakes"]++;  // tick'as niekada nepabunda prieš grain'o laiką
        for (std::size_t a = 0; a < in.size(); ++a) {
            if (in[a].state() == runtime::InputSlot::State::Mismatch) st.flows[a / per].inc("reopen_mismatch");
            if (in[a].reattached() != seen_reatt[a]) {
                seen_reatt[a] = in[a].reattached();
                st.flows[a / per].inc("reopened");
            }
        }
    }

    bool process(const engine::Batch& b, engine::TickContext& ctx) override {
        const std::uint64_t i = b.index;
        if (!processed) t_ready = now_tai();
        processed = true;
        jobs.clear();
        job_out.clear();
        host_dst.clear();
        cudaError_t ce = cudaSuccess;
        if (!cpu) (void)cudaGetLastError();  // composite_batch naudoja PeekAtLastError — išvalom ankstesnę
        for (const engine::UnitTick& ut : b.units) {
            const std::uint32_t n = ut.unit;
            host_dst.push_back(static_cast<std::uint8_t*>(ut.out[0]));
            job_out.push_back(n);
            for (std::uint32_t k = 0; k < K && !ce; ++k) {
                const std::size_t a = n * per + (has_base ? 1 : 0) + k;
                const auto* src = static_cast<const std::uint8_t*>(ut.in[(has_base ? 1 : 0) + k].ptr);
                if (host_gpu) {
                    ce = cudaMemcpyAsync(s_in[a], src, gfx::grain_bytes(in[a].format().raster), cudaMemcpyHostToDevice, stream);
                    src = s_in[a];
                }
                lay[n * K + k] = Layer{src, in[a].format().raster, lx[k], ly[k], lop[k]};
            }
            const std::uint8_t* base = nullptr;
            if (has_base) {
                const std::size_t a = n * per;
                base = static_cast<const std::uint8_t*>(ut.in[0].ptr);
                if (host_gpu && !ce) {
                    ce = cudaMemcpyAsync(s_in[a], base, gfx::grain_bytes(in[a].format().raster), cudaMemcpyHostToDevice, stream);
                    base = s_in[a];
                }
            }
            jobs.push_back(CompositeJob{base, has_base ? in[n * per].format().raster : out_r[n], &lay[n * K], static_cast<int>(K), out_r[n],
                                        host_gpu ? s_out[n] : host_dst.back()});
        }
        const int parity = static_cast<int>(i & 1);
        if (cpu) {
            if (!jobs.empty() && !gfx::cpu::composite_batch(jobs.data(), static_cast<int>(jobs.size()), parity, &pool)) {
                st.fatal = "cpu::composite_batch atmetė užduotis";
                ctx.fatal(2, st.fatal);
                return false;
            }
        } else {
            for (std::size_t off = 0; off < jobs.size() && !ce; off += gfx::cuda::kMaxBatch)
                ce = gfx::cuda::composite_batch(jobs.data() + off, static_cast<int>(std::min<std::size_t>(gfx::cuda::kMaxBatch, jobs.size() - off)),
                                                parity, stream);
            if (host_gpu)
                for (std::size_t j = 0; j < jobs.size() && !ce; ++j)
                    ce = cudaMemcpyAsync(host_dst[j], s_out[job_out[j]], gfx::grain_bytes(out_r[job_out[j]]), cudaMemcpyDeviceToHost, stream);
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
        const std::uint32_t n = ut.unit;
        FlowStats& f = st.flows[n];
        // Įėjimų skaitiklių pokytis (bet kuris vieneto įėjimas) — spragos priežasčiai
        bool inv = false, tl = false, re = false, det = false;
        for (std::uint32_t s = 0; s < per; ++s) {
            const engine::InCounters& c = hub.in(eng.in_id(n, s));
            engine::InCounters& p = seen[n * per + s];
            inv = inv || c.invalid > p.invalid;
            tl = tl || c.too_late > p.too_late;
            re = re || c.read_errors > p.read_errors;
            det = det || c.detaches > p.detaches;
            p = c;
            if (c.stalls && c.stalls > f.c["stalls"]) f.c["stalls"] = c.stalls;
        }
        switch (oc) {
            case engine::Outcome::Ready: {
                f.inc("fields_ok");
                t_commit = now_tai();
                const mxlRational r = mxl_rate(o.fmt.grain_rate());
                f.note_lateness(t_commit, mxlIndexToTimestamp(&r, ut.index), warm);
                break;
            }
            case engine::Outcome::Lapped: f.inc("lapped"); break;
            case engine::Outcome::GapBlocked: f.inc("missing_fields"); break;
            case engine::Outcome::GapClock: f.inc("clock_skips"); break;
            case engine::Outcome::Skipped: st.proc["clock_backward"]++; break;
            case engine::Outcome::GapNoContent: break;
            case engine::Outcome::GapLate:
                if (inv) f.inc("in_invalid");
                else if (tl) f.inc("too_late");
                else if (re) f.inc("read_errors");
                else if (det) f.inc("flow_invalid");  // [PF-3] gamintojas dingo / perkurtas → slot'as atjungtas
                else {
                    f.inc("late_fields");  // kuris nors įėjimas nesulaukė iki ribos → necommit'inama
                    if (f.c["late_fields"] <= 3)
                        std::fprintf(stderr, "[composite] išėjimas %u: indeksas %llu necommit'intas (įėjimas nesulaukė iki ribos)\n", n,
                                     static_cast<unsigned long long>(ut.index));
                }
                break;
        }
    }

    void end_tick(engine::Index i, engine::TickContext&) override {
        const std::uint64_t t_end = t_commit ? t_commit : now_tai();
        if (static_cast<double>(t_end) >= warm) {
            const std::uint64_t ready = processed ? t_ready : t_end;
            st.add_hist("proc_ms", t_end > ready ? t_end - ready : 0);
            st.add_hist("wait_ms", ready > t_wake ? ready - t_wake : 0);
            const mxlRational r = mxl_rate(o.fmt.grain_rate());
            st.add_hist("out_lateness", t_end - mxlIndexToTimestamp(&r, i));
        }
    }
};
}  // namespace

int run_composite(const Opts& o) {
    Stats st;
    st.flows.resize(o.flows);
    auto& stop = stop_flag();
    const std::uint32_t N = o.flows, K = o.layers;
    const bool has_base = o.base == "gen";
    const std::uint32_t per = K + (has_base ? 1 : 0);  // įėjimų vienam išėjimui: [bazė], sluoksniai 0…K-1
    std::string extra = "\"composite\":{\"layers\":" + std::to_string(K) + ",\"base\":\"" + o.base + "\",\"out_format\":\"" +
                        o.out_format + "\",\"impl\":\"" + o.impl + "\",\"storage\":\"" + o.storage + "\"}";
    auto finish = [&](int rc) {
        st.exit_code = rc;
        write_summary(o, "composite", st, extra);
        std::fprintf(stderr, "[composite] baigta, exit %d %s\n", rc, st.fatal.c_str());
        return rc;
    };
    // --- CLI patikra (nereikia MXL) ---
    const bool out_alpha = o.out_format == "v210a";
    if (!has_base && !out_alpha) { st.fatal = "--base none leidžiamas tik su --out-format v210a (R12)"; return finish(1); }
    if (o.layer_xy.size() != K) { st.fatal = "--layer-xy turi būti nurodytas lygiai --layers kartų"; return finish(1); }
    if (!o.opacity.empty() && o.opacity.size() != K) { st.fatal = "--opacity: arba nė vieno, arba --layers kartų"; return finish(1); }
    std::vector<int> lx(K), ly(K);
    std::vector<float> lop(K, 1.0f);
    for (std::uint32_t k = 0; k < K; ++k) {
        if (!parse_xy(o.layer_xy[k], lx[k], ly[k])) { st.fatal = "--layer-xy '" + o.layer_xy[k] + "': tikimasi X,Y"; return finish(1); }
        if (!o.opacity.empty()) lop[k] = o.opacity[k];
        if (!(lop[k] >= 0.0f && lop[k] <= 1.0f)) { st.fatal = "--opacity turi būti 0…1"; return finish(1); }
    }
    std::vector<std::string> layer_ids;
    std::string err;
    if (!parse_layer_flows(o, layer_ids, err)) { st.fatal = err; return finish(1); }

    // Variantai: G (gpu + cuda-linear), H (gpu + host: H2D → batch → D2H), C (cpu + host, CUDA nekuriamas).
    const bool cpu = o.impl == "cpu";
    const bool host_gpu = !cpu && o.storage == "host";
    int dev = -1;
    if (!cpu) {
        dev = cuda_device(o.device_uuid);
        if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) { st.fatal = "CUDA įrenginys"; return finish(2); }
        cudaFree(nullptr);
    }
    ThreadPool pool(cpu ? (o.threads > 0 ? o.threads : static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())))) : 1);

    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; return finish(2); }
    mxlGarbageCollectFlows(inst);
    // Įėjimai per InputSlot (planas §P6, buvusi [PF-3] logika): laukiama pagal --input-wait-ms; dingęs gamintojas → slot'as
    // atjungiamas, grįžęs → prijungiamas iš naujo (`reopened`). Laukiamas formatas ir saugykla — pirmo prijungimo.
    std::vector<runtime::InputSlot> in;
    in.reserve(static_cast<std::size_t>(N) * per);
    std::vector<VideoOut> out(N);
    auto idx_base = [&](std::uint32_t n) { return n * per; };
    auto idx_layer = [&](std::uint32_t n, std::uint32_t k) { return n * per + (has_base ? 1 : 0) + k; };
    auto in_flow_id = [&](std::uint32_t n, std::uint32_t slot) {  // slot 0 = bazė (jei yra), toliau sluoksniai
        const std::uint32_t k = slot - (has_base ? 1 : 0);
        if (has_base && slot == 0) return flow_id(o.in_role, o.flow_base + n);
        return layer_ids[k].empty() ? flow_id(kRoleGfx, (o.flow_base + n) * 8 + k) : layer_ids[k];
    };
    int rc = 0;
    for (std::uint32_t n = 0; n < N; ++n)
        for (std::uint32_t s = 0; s < per; ++s) in.push_back(make_input(o, in_flow_id(n, s), nullptr));
    const std::vector<runtime::InputSlot*> in_ptr = slot_ptrs(in);
    rc = attach_inputs(in_ptr, inst, o, "composite", st.fatal);  // 4 — neatsirado per ribą; kStoppedWaiting — sustabdyta (exit 0)
    for (std::uint32_t n = 0; n < N && !rc; ++n)
        for (std::uint32_t s = 0; s < per && !rc; ++s)
            if (!vramflow::same_rate(in[n * per + s].format().grain_rate(), o.fmt.grain_rate())) {
                const vramflow::Rational g = in[n * per + s].format().grain_rate(), w = o.fmt.grain_rate();
                st.fatal = "in " + std::to_string(n) + "/" + std::to_string(s) + ": grain'o ritmas " + std::to_string(g.num) + "/" +
                           std::to_string(g.den) + " != " + std::to_string(w.num) + "/" + std::to_string(w.den) + " (--format " +
                           vramflow::format_name(o.fmt) + ")";
                rc = 1;
            } else if (has_base && s == 0 && !(in[n * per].format().raster == o.fmt.raster)) {
                st.fatal = "bazė " + std::to_string(n) + ": formatas " + vramflow::format_name(in[n * per].format()) + " != --format " +
                           vramflow::format_name(o.fmt);
                rc = 1;
            }
    for (std::uint32_t n = 0; n < N && !rc; ++n)
        for (std::uint32_t s = 0; s < per && !rc; ++s)
            if (in[n * per + s].in().cuda == (cpu || host_gpu)) { st.fatal = "įėjimo storage neatitinka --impl/--storage"; rc = 1; }

    // --- statinė geometrija: rastrai, išėjimo rastras, visi darbai per bendrą validatorių ---
    std::vector<Raster> out_r(N);
    std::vector<Layer> lay(static_cast<std::size_t>(N) * K);
    static const std::uint8_t dummy[8]{};
    for (std::uint32_t n = 0; n < N && !rc; ++n) {
        const Raster base_r = has_base ? in[idx_base(n)].format().raster : Raster{o.fmt.raster.w, o.fmt.raster.h, o.fmt.raster.interlaced, false};
        out_r[n] = Raster{base_r.w, base_r.h, base_r.interlaced, out_alpha};
        for (std::uint32_t k = 0; k < K; ++k) lay[n * K + k] = Layer{dummy, in[idx_layer(n, k)].format().raster, lx[k], ly[k], lop[k]};
        const CompositeJob j{has_base ? dummy : nullptr, base_r, &lay[n * K], static_cast<int>(K), out_r[n], const_cast<std::uint8_t*>(dummy)};
        std::string why;
        for (int parity = 0; parity < 2 && !rc; ++parity)
            if (!gfx::composite_job_valid(j, parity, &why)) { st.fatal = "išėjimas " + std::to_string(n) + ": " + why; rc = 1; }
    }
    const std::string opts = cpu ? std::string{} : storage_options(o, dev);
    for (std::uint32_t n = 0; n < N && !rc; ++n) {
        const vramflow::VideoFormat of{out_r[n], o.fmt.frame_rate};
        if (!create_output(inst, out[n], flow_id(kRoleComp, o.flow_base + n), comp_def(o.flow_base + n, of, o.run_seed), opts, of, o, err)) {
            st.fatal = "out " + std::to_string(n) + ": " + err;
            rc = 2;
        }
    }
    // Įėjimai jau gamina (head žinomas): InputSlot prisijungia tik tada — pirmi tick'ai neskaičiuojami kaip vėlyvi.

    cudaStream_t stream{};
    if (!cpu && !rc && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { st.fatal = "cudaStreamCreate"; rc = 2; }
    // H: device scratch kiekvienam įėjimui ir išėjimui
    std::vector<std::uint8_t*> s_in(in.size(), nullptr), s_out(N, nullptr);
    if (host_gpu)
        for (std::size_t a = 0; a < in.size() && !rc; ++a)
            if (cudaMalloc(reinterpret_cast<void**>(&s_in[a]), gfx::grain_bytes(in[a].format().raster)) != cudaSuccess) { st.fatal = "scratch"; rc = 2; }
    if (host_gpu)
        for (std::uint32_t n = 0; n < N && !rc; ++n)
            if (cudaMalloc(reinterpret_cast<void**>(&s_out[n]), gfx::grain_bytes(out_r[n])) != cudaSuccess) { st.fatal = "scratch"; rc = 2; }

    // Bendras tick variklis (planas V11): N vienetų po per = [bazė] + K įėjimų → 1 išėjimas; izoliacijos ribos kaip logo (V10).
    std::vector<engine::MxlVideoInput> ports_in;
    std::vector<engine::MxlVideoOutput> ports_out;
    ports_in.reserve(in.size());
    ports_out.reserve(N);
    for (auto& slot : in) ports_in.emplace_back(slot, inst);
    for (auto& v : out) ports_out.emplace_back(v, engine::host_copy(), nullptr, nullptr);
    engine::SystemClock clk;
    engine::StatsHub hub;
    engine::TickEngine eng(engine_config(o, EngineTiming::Isolation, o.read_offset_us * 1000), clk, hub, stop);
    for (std::uint32_t n = 0; n < N && !rc; ++n) {
        engine::UnitSpec u;
        u.name = flow_id(kRoleComp, o.flow_base + n);
        for (std::uint32_t s = 0; s < per; ++s)
            u.in.push_back(engine::InputSpec{&ports_in[n * per + s], {}, 1, engine::Need::Required, engine::InMode::Exact});
        u.out.push_back(&ports_out[n]);
        std::uint32_t id = 0;
        if (!eng.add_unit(u, id, err)) { st.fatal = "add_unit: " + err; rc = 2; }
    }
    CompositeFn fn(o, st, hub, eng, in, cpu, host_gpu, has_base, K, per, lx, ly, lop, lay, out_r, s_in, s_out, pool, stream,
                   warm_tai_after(o.warmup_s));
    if (!rc) {
        const engine::RunResult res = eng.run(fn);
        if (res.reason == engine::EndReason::Fatal) {
            if (st.fatal.empty()) st.fatal = res.fatal;
            rc = res.exit_code ? res.exit_code : 2;
        } else if (res.reason == engine::EndReason::FormatChange) {
            st.fatal = res.fatal;
            rc = kExitFormatChange;
        }
        to_summary(hub, eng, res, N, SkipScope::PerFlow, st);
    }
    if (!cpu && stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    for (auto* p : s_in) if (p) cudaFree(p);
    for (auto* p : s_out) if (p) cudaFree(p);
    for (auto& v : out) v.release(inst);
    for (auto& v : in) v.release();
    mxlDestroyInstance(inst);
    // praradimai = pajėgumo / tiekimo klaida (exit 4), jei nebuvo fatalios
    if (!rc)
        for (auto& f : st.flows)
            for (const char* k : {"late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips", "missing_fields", "flow_invalid"})
                if (f.c.count(k) && f.c[k]) rc = 4;
    return finish(exit_code(rc));  // kStoppedWaiting — sustabdyta belaukiant įėjimų (tvarkinga pabaiga, exit 0)
}

}  // namespace mxlv::func
