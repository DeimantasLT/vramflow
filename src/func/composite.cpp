// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// composite — kompozitorius: kiekvienam išėjimui n bazė (v210 iš `gen`, arba jokios) + K grafikos sluoksnių (v210a iš `gfxgen` ar
// kito composite) → v210 / v210a MXL išėjimas. Tick variklis kaip `logo`: viena gija valdo visus MXL handle'us; visi įėjimai
// skaitomi neblokuojančiai (kartojama kas 0,5 ms iki termino); vienas CPU batch / CUDA batch laukui visiems N.
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
//  - [PF-3] įėjimas, kurio skaitymas grąžina MXL_ERR_FLOW_INVALID (writer'is perkurtas), uždaromas ir atidaromas iš naujo
//    neblokuojančiai (po vieną bandymą kas 100 ms, tarp tick'ų; skaitiklis `reopened`). Išėjimas, kurio įėjimas trūksta, tuo metu
//    necommit'inamas (`missing_fields`), bet funkcija tęsia. Atidarytas iš naujo įėjimas privalo turėti tą patį rastrą ir
//    saugyklą; jei ne — jis lieka uždarytas visam laikui (skaitiklis `reopen_mismatch`), o jo išėjimas nebecommit'inamas.
//  - Skaitikliai (kiekvienam išėjimui): fields_ok, late_fields (išėjimas necommit'intas — įėjimas nesulaukė iki termino; kartą
//    per išėjimą), too_late / in_invalid / read_errors / flow_invalid (įėjimų įvykiai), missing_fields, reopened, lapped.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "common/index_math.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "gfx/cpu/gfx_cpu.hpp"
#include "gfx/gfx_kernels.hpp"
#include "gfx/raster.hpp"

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
    std::vector<VideoIn> in(N * per);
    std::vector<VideoOut> out(N);
    auto idx_base = [&](std::uint32_t n) { return n * per; };
    auto idx_layer = [&](std::uint32_t n, std::uint32_t k) { return n * per + (has_base ? 1 : 0) + k; };
    auto in_flow_id = [&](std::uint32_t n, std::uint32_t slot) {  // slot 0 = bazė (jei yra), toliau sluoksniai
        const std::uint32_t k = slot - (has_base ? 1 : 0);
        if (has_base && slot == 0) return flow_id(o.in_role, o.flow_base + n);
        return layer_ids[k].empty() ? flow_id(kRoleGfx, (o.flow_base + n) * 8 + k) : layer_ids[k];
    };
    int rc = 0;
    for (std::uint32_t n = 0; n < N && !rc; ++n)
        for (std::uint32_t s = 0; s < per && !rc; ++s)
            if (!in[n * per + s].open(inst, in_flow_id(n, s), 15000, stop, err, static_cast<const vramflow::VideoFormat*>(nullptr))) {
                st.fatal = "in " + std::to_string(n) + "/" + std::to_string(s) + ": " + err;
                rc = 4;
            } else if (!vramflow::same_rate(in[n * per + s].fmt.grain_rate(), o.fmt.grain_rate())) {
                const vramflow::Rational g = in[n * per + s].fmt.grain_rate(), w = o.fmt.grain_rate();
                st.fatal = "in " + std::to_string(n) + "/" + std::to_string(s) + ": grain'o ritmas " + std::to_string(g.num) + "/" +
                           std::to_string(g.den) + " != " + std::to_string(w.num) + "/" + std::to_string(w.den) + " (--format " +
                           vramflow::format_name(o.fmt) + ")";
                rc = 1;
            } else if (has_base && s == 0 && !(in[n * per].fmt.raster == o.fmt.raster)) {
                st.fatal = "bazė " + std::to_string(n) + ": formatas " + vramflow::format_name(in[n * per].fmt) + " != --format " +
                           vramflow::format_name(o.fmt);
                rc = 1;
            }
    for (std::uint32_t n = 0; n < N && !rc; ++n)
        for (std::uint32_t s = 0; s < per && !rc; ++s)
            if (in[n * per + s].cuda == (cpu || host_gpu)) { st.fatal = "įėjimo storage neatitinka --impl/--storage"; rc = 1; }

    // --- statinė geometrija: rastrai, išėjimo rastras, visi darbai per bendrą validatorių ---
    std::vector<Raster> out_r(N);
    std::vector<Layer> lay(static_cast<std::size_t>(N) * K);
    static const std::uint8_t dummy[8]{};
    for (std::uint32_t n = 0; n < N && !rc; ++n) {
        const Raster base_r = has_base ? in[idx_base(n)].raster : Raster{o.fmt.raster.w, o.fmt.raster.h, o.fmt.raster.interlaced, false};
        out_r[n] = Raster{base_r.w, base_r.h, base_r.interlaced, out_alpha};
        for (std::uint32_t k = 0; k < K; ++k) lay[n * K + k] = Layer{dummy, in[idx_layer(n, k)].raster, lx[k], ly[k], lop[k]};
        const CompositeJob j{has_base ? dummy : nullptr, base_r, &lay[n * K], static_cast<int>(K), out_r[n], const_cast<std::uint8_t*>(dummy)};
        std::string why;
        for (int parity = 0; parity < 2 && !rc; ++parity)
            if (!gfx::composite_job_valid(j, parity, &why)) { st.fatal = "išėjimas " + std::to_string(n) + ": " + why; rc = 1; }
    }
    const std::string opts = cpu ? std::string{} : storage_options(o, dev);
    for (std::uint32_t n = 0; n < N && !rc; ++n)
        if (!out[n].create(inst, comp_def(o.flow_base + n, vramflow::VideoFormat{out_r[n], o.fmt.frame_rate}, o.run_seed), opts, err,
                           vramflow::VideoFormat{out_r[n], o.fmt.frame_rate})) {
            st.fatal = "out " + std::to_string(n) + ": " + err;
            rc = 2;
        }
    // Įėjimai turi jau gaminti (head žinomas) — tada pirmi tick'ai neskaičiuojami kaip vėlyvi.
    for (std::size_t a = 0; a < in.size() && !rc; ++a) {
        const auto until = now_tai() + 15'000'000'000ull;
        for (;;) {
            const std::uint64_t h = in[a].head();
            if (h != MXL_UNDEFINED_INDEX && h != 0) break;
            if (stop.load() || now_tai() > until) { st.fatal = "įėjimas " + in[a].id + " negamina grain'ų"; rc = 4; break; }
            mxlSleepForNs(20'000'000);
        }
    }

    cudaStream_t stream{};
    if (!cpu && !rc && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { st.fatal = "cudaStreamCreate"; rc = 2; }
    // H: device scratch kiekvienam įėjimui ir išėjimui
    std::vector<std::uint8_t*> s_in(in.size(), nullptr), s_out(N, nullptr);
    if (host_gpu)
        for (std::size_t a = 0; a < in.size() && !rc; ++a)
            if (cudaMalloc(reinterpret_cast<void**>(&s_in[a]), gfx::grain_bytes(in[a].raster)) != cudaSuccess) { st.fatal = "scratch"; rc = 2; }
    if (host_gpu)
        for (std::uint32_t n = 0; n < N && !rc; ++n)
            if (cudaMalloc(reinterpret_cast<void**>(&s_out[n]), gfx::grain_bytes(out_r[n])) != cudaSuccess) { st.fatal = "scratch"; rc = 2; }

    // Būsena per įėjimą: 0 laukia, 1 paruoštas, 2 baigtas be duomenų. closed — reader'is užmuštas / uždarytas (reopen), lost —
    // neatitinka rastro po reopen (visam laikui).
    std::vector<int> state(in.size());
    std::vector<char> closed(in.size(), 0), lost(in.size(), 0);
    std::vector<std::uint64_t> next_reopen(in.size(), 0);
    std::vector<const std::uint32_t*> ptr(in.size(), nullptr);
    std::vector<Raster> want_r(in.size());
    std::vector<char> want_cuda(in.size());
    for (std::size_t a = 0; a < in.size(); ++a) { want_r[a] = in[a].raster; want_cuda[a] = in[a].cuda; }
    std::vector<char> dead(N), ready(N);
    std::vector<CompositeJob> jobs;
    std::vector<std::uint32_t> job_out;
    std::vector<std::uint8_t*> host_dst;

    const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    const double warm = warm_tai_after(o.warmup_s);
    const mxlRational rate = mxl_rate(o.fmt.grain_rate());
    std::uint64_t i = mxlGetCurrentIndex(&rate) + 1;
    ClockWatch cw;
    cw.stepped();

    while (!rc && !stop.load() && now_tai() < end_tai) {
        // [PF-3] uždarytų įėjimų atidarymas iš naujo — neblokuojantis, tarp tick'ų (laisvas laikas prieš wait_until)
        for (std::size_t a = 0; a < in.size(); ++a) {
            if (!closed[a] || lost[a] || now_tai() < next_reopen[a]) continue;
            next_reopen[a] = now_tai() + 100'000'000ull;
            VideoIn fresh;
            std::string e;
            if (fresh.open(inst, in[a].id, 0, stop, e, static_cast<const vramflow::VideoFormat*>(nullptr))) {
                if (fresh.raster == want_r[a] && fresh.cuda == static_cast<bool>(want_cuda[a]) &&
                    vramflow::same_rate(fresh.fmt.grain_rate(), o.fmt.grain_rate())) {
                    in[a] = fresh;
                    closed[a] = 0;
                    st.flows[a / per].inc("reopened");
                    continue;
                }
                lost[a] = 1;
                st.flows[a / per].inc("reopen_mismatch");
                std::fprintf(stderr, "[composite] įėjimas %s po reopen pakeitė rastrą / saugyklą — lieka uždarytas\n", in[a].id.c_str());
            }
            fresh.release(inst);
        }
        if (wait_until(rate, i, o.read_offset_us * 1000, stop)) st.proc["clock_backward"]++;
        if (stop.load()) break;
        const std::uint64_t t_wake = now_tai();
        // tick'as niekada nepabunda prieš savo grain'o laiką (kitas laikrodis / dažnis — klaida; funkcinis testas tikrina 0)
        if (t_wake < mxlIndexToTimestamp(&rate, i)) st.proc["early_wakes"]++;
        const std::uint64_t deadline = mxlIndexToTimestamp(&rate, i) + std::uint64_t{o.deadline_ms} * 1'000'000ull;
        std::fill(state.begin(), state.end(), 0);
        std::fill(dead.begin(), dead.end(), 0);
        for (std::uint32_t n = 0; n < N; ++n)
            for (std::uint32_t s = 0; s < per; ++s)
                if (closed[n * per + s]) {
                    if (!dead[n]) st.flows[n].inc("missing_fields");
                    dead[n] = 1;
                }
        // 3: visi įėjimai neblokuojančiai; laukiantys kartojami kas 0,5 ms iki deadline
        for (;;) {
            bool waiting = false;
            for (std::uint32_t n = 0; n < N; ++n) {
                if (dead[n]) continue;
                for (std::uint32_t s = 0; s < per && !dead[n]; ++s) {
                    const std::size_t a = n * per + s;
                    if (state[a]) continue;
                    mxlGrainInfo info{};
                    const std::uint32_t* p = nullptr;
                    const mxlStatus r = in[a].get(i, 0, info, p);
                    if (r == MXL_ERR_OUT_OF_RANGE_TOO_EARLY) { waiting = true; continue; }
                    if (r == MXL_STATUS_OK && !(info.flags & MXL_GRAIN_FLAG_INVALID)) { ptr[a] = p; state[a] = 1; continue; }
                    state[a] = 2;
                    dead[n] = 1;
                    if (r == MXL_STATUS_OK) st.flows[n].inc("in_invalid");
                    else if (r == MXL_ERR_OUT_OF_RANGE_TOO_LATE) st.flows[n].inc("too_late");
                    else if (r == MXL_ERR_FLOW_INVALID) {  // [PF-3] writer'is perkurtas → reader'is atidaromas iš naujo
                        in[a].release(inst);
                        closed[a] = 1;
                        next_reopen[a] = 0;
                        st.flows[n].inc("flow_invalid");
                    } else {
                        st.flows[n].inc("read_errors");
                        st.flows[n].inc(("read_status_" + std::to_string(r)).c_str());
                    }
                }
            }
            if (!waiting || now_tai() >= deadline || stop.load()) break;
            mxlSleepForNs(500'000);
        }
        for (std::uint32_t n = 0; n < N; ++n) {
            ready[n] = 0;
            if (dead[n]) continue;
            bool all = true;
            for (std::uint32_t s = 0; s < per; ++s) all = all && state[n * per + s] == 1;
            if (all) ready[n] = 1;
            else {
                st.flows[n].inc("late_fields");  // kuris nors įėjimas nesulaukė iki deadline → necommit'inama
                if (st.flows[n].c["late_fields"] <= 3) {  // pirmi įvykiai — diagnostikai į stderr
                    std::string who;
                    for (std::uint32_t s = 0; s < per; ++s)
                        if (state[n * per + s] != 1)
                            who += " " + in[n * per + s].id.substr(24) + "(head-i=" +
                                   std::to_string(static_cast<long long>(in[n * per + s].head()) - static_cast<long long>(i)) + ")";
                    std::fprintf(stderr, "[composite] išėjimas %u: indeksas %llu (tick %lld, pabudimas +%.2f ms) necommit'intas, nesulaukta:%s\n", n,
                                 static_cast<unsigned long long>(i), static_cast<long long>(i - (st.first_index == ~0ull ? i : st.first_index)),
                                 (static_cast<double>(t_wake) - static_cast<double>(mxlIndexToTimestamp(&rate, i))) / 1e6, who.c_str());
                }
            }
        }
        const std::uint64_t t_ready = now_tai();
        // 4: išėjimų slot'ai paruoštiems; darbų sąrašas
        jobs.clear();
        job_out.clear();
        host_dst.clear();
        for (std::uint32_t n = 0; n < N; ++n) {
            if (!ready[n]) continue;
            mxlStatus s{};
            std::uint32_t* d = out[n].open(i, s);
            if (!d) {
                if (s == MXL_ERR_INVALID_ARG && i <= out[n].last_committed) { st.proc["clock_backward"]++; ready[n] = 0; continue; }
                st.fatal = "OpenGrainSlot status " + std::to_string(s);
                rc = 2;
                break;
            }
            host_dst.push_back(reinterpret_cast<std::uint8_t*>(d));
            job_out.push_back(n);
        }
        if (rc) break;
        // 5: apdorojimas — G: tiesiai device rodyklės; H: H2D → batch → D2H; C: CPU batch
        cudaError_t ce = cudaSuccess;
        const int parity = static_cast<int>(i & 1);
        if (!cpu) (void)cudaGetLastError();  // sąmoningai išvalom ankstesnę (jau apdorotą) klaidą — composite_batch naudoja PeekAtLastError
        for (std::size_t j = 0; j < job_out.size() && !ce; ++j) {
            const std::uint32_t n = job_out[j];
            for (std::uint32_t k = 0; k < K; ++k) {
                const std::size_t a = idx_layer(n, k);
                const std::uint8_t* src = reinterpret_cast<const std::uint8_t*>(ptr[a]);
                if (host_gpu) {
                    if (!ce) ce = cudaMemcpyAsync(s_in[a], src, gfx::grain_bytes(in[a].raster), cudaMemcpyHostToDevice, stream);
                    src = s_in[a];
                }
                lay[n * K + k] = Layer{src, in[a].raster, lx[k], ly[k], lop[k]};
            }
            const std::uint8_t* base = nullptr;
            if (has_base) {
                const std::size_t a = idx_base(n);
                base = reinterpret_cast<const std::uint8_t*>(ptr[a]);
                if (host_gpu && !ce) {
                    ce = cudaMemcpyAsync(s_in[a], base, gfx::grain_bytes(in[a].raster), cudaMemcpyHostToDevice, stream);
                    base = s_in[a];
                }
            }
            jobs.push_back(CompositeJob{base, has_base ? in[idx_base(n)].raster : out_r[n], &lay[n * K], static_cast<int>(K), out_r[n],
                                        host_gpu ? s_out[n] : host_dst[j]});
        }
        if (cpu) {
            if (!jobs.empty() && !gfx::cpu::composite_batch(jobs.data(), static_cast<int>(jobs.size()), parity, &pool)) {
                st.fatal = "cpu::composite_batch atmetė užduotis";
                rc = 2;
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
        if (ce) { st.fatal = std::string("CUDA: ") + cudaGetErrorString(ce); rc = 2; }
        if (rc) break;
        // 6: lap patikra visiems įėjimams, 7: commit
        const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
        for (const std::uint32_t n : job_out) {
            bool lap = false;
            for (std::uint32_t s = 0; s < per; ++s) {
                const std::size_t a = n * per + s;
                const std::uint64_t h = in[a].head();
                if (lapped(std::max(h == MXL_UNDEFINED_INDEX ? 0 : h, now_idx), i, in[a].layout.slotCount)) lap = true;
            }
            if (lap) {
                out[n].cancel();
                st.flows[n].inc("lapped");
                continue;
            }
            if (out[n].commit() != MXL_STATUS_OK) { st.fatal = "CommitGrain"; rc = 2; break; }
            st.flows[n].inc("fields_ok");
        }
        const std::uint64_t t_commit = now_tai();
        if (t_commit >= warm) {
            st.add_hist("proc_ms", t_commit - t_ready);
            st.add_hist("wait_ms", t_ready - t_wake);
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
    if (!cpu && stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    for (auto* p : s_in) if (p) cudaFree(p);
    for (auto* p : s_out) if (p) cudaFree(p);
    for (auto& v : out) v.release(inst);
    for (auto& v : in) v.release(inst);
    mxlDestroyInstance(inst);
    // praradimai = pajėgumo / tiekimo klaida (exit 4), jei nebuvo fatalios
    if (!rc)
        for (auto& f : st.flows)
            for (const char* k : {"late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips", "missing_fields", "flow_invalid"})
                if (f.c.count(k) && f.c[k]) rc = 4;
    return finish(rc);
}

}  // namespace mxlv::func
