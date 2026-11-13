// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// composite — kompozitorius: kiekvienam išėjimui n bazė (v210 iš `gen`, arba jokios) + K grafikos sluoksnių (v210a iš `gfxgen` ar
// kito composite) → v210 / v210a MXL išėjimas. Bendras tick variklis (V11, izoliacijos ribos kaip logo): viena gija valdo
// visus MXL handle'us; paruošti išėjimai apdorojami vienu CPU / CUDA batch'u fazei, užstrigęs įėjimas kitų nelaiko.
//
// Sprendimai:
//  - Geometrija statinė: rastrai perskaitomi iš flow aprašų, tikrinama bendru gfx::composite_job_valid. Bazė ir išėjimas —
//    PRADŽIOJE (rastrų neatitikimas, `--base none` su v210 ...) → startas nepavyksta su aiškia klaida (exit 1). Sluoksnis —
//    pirmą kartą prisijungus (nelyginis y interlaced rastre, x, interlace, ritmas, saugykla): netinkamas sluoksnis atmetamas
//    (įvykis warning; jis lieka skaidrus), programa tęsiama. Vykdymo metu išėjimą iš batch'o pašalina tik trūkstama / vėlyva /
//    INVALID bazė.
//  - Storage: VISI įėjimai turi būti tokioje pačioje saugykloje kaip --storage (C, H: host; G: cuda-linear), kaip `logo`.
//    Mišri saugykla: bazė — startas nepavyksta (exit 1); sluoksnis — atmetamas (skaidrus). H variantas kiekvieną įėjimą (host)
//    pakelia į device scratch, apdoroja ir rezultatą nukopijuoja į host išėjimo slot'ą.
//  - Bazė `none` (tik v210a išėjimas): išėjimo rastras = --format rastras (numatytasis 1920×1080 interlaced); kitaip išėjimo
//    rastras = bazės (w, h, interlace), o bazė turi atitikti --format. Visų įėjimų grain'o ritmas turi sutapti su --format.
//  - `--layer-flow K:UUID`: K — sluoksnio indeksas nuo 0 (0…layers-1); tas pats flow VISIEMS išėjimams n (grandinei N=1).
//  - [PF-3, §P6] įėjimai — runtime::InputSlot. BAZĖ būtina (Required): paleidžiant laukiama pagal --input-wait-ms
//    (numatyta neribotai); gamintojas dingo (MXL_ERR_FLOW_INVALID arba nebėra gyvo writer'io) → slot'as atjungiamas ir
//    prijungiamas iš naujo neblokuojančiai (bandymas ne dažniau kaip kas 100 ms, tarp tick'ų; skaitiklis `reopened`). Išėjimas,
//    kurio bazės trūksta, gauna spragą pagal --on-loss (`missing_fields`), bet funkcija tęsia. Grįžęs įėjimas privalo turėti tą
//    patį formatą ir saugyklą; jei ne — --on-format-change exit (numatyta): `reopen_mismatch`, exit 6 su abiem formatais; wait:
//    laukiama, kol grįš tinkamas.
//  - SLUOKSNIAI neprivalomi (Need::Wanted, peržiūra 2026-11-13 p. 1): sugedęs grafikos sluoksnis niekada nenuima programos.
//    Sluoksnis, kuris INVALID, dar neprisijungęs, užstrigęs, atjungtas, vėluoja iki vieneto ribos arba atmestas (geometrija /
//    saugykla / ritmas netinka — tikrinama pirmą kartą prisijungus), tam indeksui laikomas visiškai skaidriu: kompozicija daroma
//    be jo (bazė praeina bitas į bitą — lygiai etalonas be to sluoksnio), skaičiuojama `layer_absent` (ir `layer_absent_<k>`),
//    būsenos pokytis — operatoriaus įvykis (layer_absent / layer_present), išėjimo praradimu tai nelaikoma. Grįžęs sluoksnis
//    naudojamas nuo kito indekso (interlaced — nuo kadro ribos, lyginio indekso). `--base none`: visiems sluoksniams nesant
//    išėjimas — skaidri juoda (key 0), ne spraga. Paleidimas laukia tik bazių (sluoksniai prisijungia variklio cikle).
//  - Skaitikliai (kiekvienam išėjimui): fields_ok, late_fields (išėjimas necommit'intas — bazė nesulaukė iki termino; kartą
//    per išėjimą), too_late / in_invalid / read_errors / flow_invalid (bazės įvykiai), missing_fields, reopened, lapped,
//    layer_absent.
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
// lap patikrą, commit'ą ir spragas daro variklis. Santraukos raktai — tie patys kaip prieš migraciją (+ layer_absent).
struct CompositeFn final : engine::TickFunction {
    // Sluoksnio patikra (pirmą kartą prisijungus): Unknown — dar neprisijungęs; Ok — naudojamas; Rejected — netinkamas (skaidrus).
    enum class Check : std::uint8_t { Unknown, Ok, Rejected };
    const Opts& o;
    Stats& st;
    engine::StatsHub& hub;
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
    std::vector<engine::InCounters> seen;  // bazės skaitikliai po paskutinio vieneto sprendimo (spragos priežasčiai)
    std::vector<std::uint64_t> seen_reatt;
    std::vector<Check> check;             // sluoksnio patikra; indeksas — įėjimo a
    std::vector<std::uint8_t> present;    // sluoksnis naudojamas (1) / skaidrus (0); pradžioje 1 — pirmas trūkumas praneša
    std::vector<std::string> absent_key;  // "layer_absent_<k>"
    std::vector<std::string> seen_reason;  // sluoksnio slot'o paskutinė atmetimo priežastis (formato pokyčio įvykiui)
    TickTimes times;  // proc_ms / wait_ms / held_ms / out_lateness — atskirai kiekvienam indeksui (pernešimas)

    CompositeFn(const Opts& o_, Stats& st_, engine::StatsHub& hub_, const engine::TickEngine& eng_, std::vector<runtime::InputSlot>& in_,
                bool cpu_, bool host_gpu_, bool has_base_, std::uint32_t K_, std::uint32_t per_, const std::vector<int>& lx_,
                const std::vector<int>& ly_, const std::vector<float>& lop_, std::vector<Layer>& lay_, const std::vector<Raster>& out_r_,
                std::vector<std::uint8_t*>& s_in_, std::vector<std::uint8_t*>& s_out_, ThreadPool& pool_, cudaStream_t stream_, double warm_)
        : o(o_), st(st_), hub(hub_), eng(eng_), in(in_), cpu(cpu_), host_gpu(host_gpu_), has_base(has_base_), K(K_), per(per_), lx(lx_),
          ly(ly_), lop(lop_), lay(lay_), out_r(out_r_), s_in(s_in_), s_out(s_out_), pool(pool_), stream(stream_), warm(warm_),
          seen(in_.size()), seen_reatt(in_.size(), 0), check(in_.size(), Check::Unknown), present(in_.size(), 1),
          seen_reason(in_.size()) {
        jobs.reserve(o.flows);
        job_out.reserve(o.flows);
        host_dst.reserve(o.flows);
        for (std::uint32_t k = 0; k < K; ++k) absent_key.push_back("layer_absent_" + std::to_string(k));
    }

    std::uint32_t layer_slot(std::uint32_t k) const { return (has_base ? 1 : 0) + k; }
    std::size_t layer_in(std::uint32_t n, std::uint32_t k) const { return static_cast<std::size_t>(n) * per + layer_slot(k); }

    // Pirmą kartą prisijungusio sluoksnio patikra: ritmas, saugykla, geometrija (abu paritetai) prieš šio išėjimo rastrą.
    // Netinkamas — atmetamas (skaidrus visam paleidimui; grįžimas kitu formatu — --on-format-change, InputSlot). Retas kelias
    // (prisijungimas): H scratch išskiriamas čia.
    void validate_layer(std::uint32_t n, std::uint32_t k) {
        const std::size_t a = layer_in(n, k);
        const vramflow::VideoFormat& f = in[a].format();
        std::string why;
        if (!vramflow::same_rate(f.grain_rate(), o.fmt.grain_rate())) {
            why = "grain'o ritmas " + std::to_string(f.grain_rate().num) + "/" + std::to_string(f.grain_rate().den) + " != --format " +
                  vramflow::format_name(o.fmt);
        } else if (in[a].in().cuda == (cpu || host_gpu)) {
            why = "saugykla neatitinka --impl/--storage";
        } else {
            static const std::uint8_t dummy[8]{};
            const Layer L{dummy, f.raster, lx[k], ly[k], lop[k]};
            const Raster base_r = has_base ? in[static_cast<std::size_t>(n) * per].format().raster
                                           : Raster{out_r[n].w, out_r[n].h, out_r[n].interlaced, false};
            const CompositeJob j{has_base ? dummy : nullptr, base_r, &L, 1, out_r[n], const_cast<std::uint8_t*>(dummy)};
            for (int parity = 0; parity < 2 && why.empty(); ++parity) gfx::composite_job_valid(j, parity, &why);
        }
        if (why.empty() && host_gpu && !s_in[a] &&
            cudaMalloc(reinterpret_cast<void**>(&s_in[a]), gfx::grain_bytes(f.raster)) != cudaSuccess) {
            (void)cudaGetLastError();
            s_in[a] = nullptr;
            why = "H scratch (cudaMalloc) nepavyko";
        }
        if (why.empty()) {
            check[a] = Check::Ok;
            return;
        }
        check[a] = Check::Rejected;
        const std::string t = "sluoksnis " + std::to_string(k) + " atmestas (" + vramflow::format_name(f) + "): " + why + "; jis skaidrus";
        log_line("composite", "išėjimas " + std::to_string(n) + ": " + t);
        hub.event(engine::EventKind::Warning, eng.in_id(n, layer_slot(k)), now_tai(), engine::kNoIndex, 0, t.c_str());
    }

    void begin_tick(engine::Index i, engine::TickContext&) override {
        times.wake(i, now_tai());
        const mxlRational r = mxl_rate(o.fmt.grain_rate());
        if (now_tai() < mxlIndexToTimestamp(&r, i)) st.proc["early_wakes"]++;  // tick'as niekada nepabunda prieš grain'o laiką
        for (std::size_t a = 0; a < in.size(); ++a) {
            if (in[a].reattached() != seen_reatt[a]) {
                seen_reatt[a] = in[a].reattached();
                st.flows[a / per].inc("reopened");
            }
        }
        for (std::uint32_t n = 0; n < o.flows; ++n)
            for (std::uint32_t k = 0; k < K; ++k) {
                const std::size_t a = layer_in(n, k);
                if (check[a] == Check::Unknown && in[a].state() == runtime::InputSlot::State::Attached) validate_layer(n, k);
                // Sluoksnis grįžo kitu formatu (C1): slot'as jo neprijungia (politika wait — ne exit 6, kaip bazei), sluoksnis lieka
                // skaidrus (layer_absent); vienas operatoriaus įvykis kiekvienam naujam atmetimui (alokacija — tik pokyčio metu)
                if (in[a].state() != runtime::InputSlot::State::Attached && !in[a].reason().empty() && in[a].reason() != seen_reason[a]) {
                    seen_reason[a] = in[a].reason();
                    st.flows[n].inc("layer_format_mismatch");
                    const std::string t = "sluoksnis " + std::to_string(k) + ": " + seen_reason[a] + " — skaidrus, laukiama tinkamo formato";
                    log_line("composite", "išėjimas " + std::to_string(n) + ": " + t);
                    hub.event(engine::EventKind::Warning, eng.in_id(n, layer_slot(k)), now_tai(), i, static_cast<std::int64_t>(k), t.c_str());
                }
            }
    }

    // Ar sluoksnis k naudojamas išėjimo n indekse i (ref — variklio grain'as). Trūkstamas — skaidrus: skaitiklis ir įvykis
    // būsenos pokyčiui; grįžęs — interlaced rastre nuo lyginio indekso (kadro riba), kitaip iškart.
    bool use_layer(std::uint32_t n, std::uint32_t k, engine::Index i, const engine::GrainRef& ref) {
        const std::size_t a = layer_in(n, k);
        const bool invalid = (ref.flags & MXL_GRAIN_FLAG_INVALID) != 0;
        const bool avail = check[a] == Check::Ok && ref.ptr != nullptr && !invalid;
        if (avail && !present[a] && (!out_r[n].interlaced || (i & 1) == 0)) {
            present[a] = 1;
            hub.event(engine::EventKind::LayerPresent, eng.in_id(n, layer_slot(k)), now_tai(), i, static_cast<std::int64_t>(k),
                      "sluoksnis vėl naudojamas");
        } else if (!avail && present[a]) {
            present[a] = 0;
            const char* why = check[a] == Check::Rejected ? "sluoksnis atmestas — skaidrus"
                              : check[a] == Check::Unknown ? "sluoksnis neprisijungęs — skaidrus"
                              : invalid                    ? "sluoksnio grain'as INVALID — skaidrus"
                                                           : "sluoksnio grain'o nėra iki vieneto ribos — skaidrus";
            hub.event(engine::EventKind::LayerAbsent, eng.in_id(n, layer_slot(k)), now_tai(), i, static_cast<std::int64_t>(k), why);
        }
        if (avail && present[a]) return true;
        FlowStats& f = st.flows[n];
        f.inc("layer_absent");
        f.inc(absent_key[k].c_str());
        return false;
    }

    bool process(const engine::Batch& b, engine::TickContext& ctx) override {
        const std::uint64_t i = b.index;
        times.phase_begin(i, now_tai());
        jobs.clear();
        job_out.clear();
        host_dst.clear();
        cudaError_t ce = cudaSuccess;
        if (!cpu) (void)cudaGetLastError();  // composite_batch naudoja PeekAtLastError — išvalom ankstesnę
        for (const engine::UnitTick& ut : b.units) {
            const std::uint32_t n = ut.unit;
            host_dst.push_back(static_cast<std::uint8_t*>(ut.out[0]));
            job_out.push_back(n);
            int used = 0;  // naudojamų sluoksnių; trūkstami praleidžiami (skaidrūs), tvarka išlaikoma
            for (std::uint32_t k = 0; k < K && !ce; ++k) {
                const engine::GrainRef& ref = ut.in[layer_slot(k)];
                if (!use_layer(n, k, i, ref)) continue;
                const std::size_t a = layer_in(n, k);
                const auto* src = static_cast<const std::uint8_t*>(ref.ptr);
                if (host_gpu) {
                    ce = cudaMemcpyAsync(s_in[a], src, gfx::grain_bytes(in[a].format().raster), cudaMemcpyHostToDevice, stream);
                    src = s_in[a];
                }
                lay[static_cast<std::size_t>(n) * K + static_cast<std::size_t>(used++)] = Layer{src, in[a].format().raster, lx[k], ly[k], lop[k]};
            }
            const std::uint8_t* base = nullptr;
            if (has_base) {
                const std::size_t a = static_cast<std::size_t>(n) * per;
                base = static_cast<const std::uint8_t*>(ut.in[0].ptr);
                if (host_gpu && !ce) {
                    ce = cudaMemcpyAsync(s_in[a], base, gfx::grain_bytes(in[a].format().raster), cudaMemcpyHostToDevice, stream);
                    base = s_in[a];
                }
            }
            jobs.push_back(CompositeJob{base, has_base ? in[static_cast<std::size_t>(n) * per].format().raster : out_r[n],
                                        &lay[static_cast<std::size_t>(n) * K], used, out_r[n], host_gpu ? s_out[n] : host_dst.back()});
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

    // Variklio praleisti indeksai (T11): praradimai (late_skips; skaitiklį rašo to_summary) — į loss_s sekundėms.
    void on_skip(std::uint32_t unit, engine::Index from, engine::Index to, engine::TickContext& ctx) override {
        if (!ctx.skip_is_loss()) return;
        const mxlRational r = mxl_rate(o.fmt.grain_rate());
        for (engine::Index i = from; i < to; ++i) st.flows[unit].loss_s.add(mxlIndexToTimestamp(&r, i) / 1'000'000'000);
    }

    void end_unit(const engine::UnitTick& ut, engine::Outcome oc, engine::TickContext&) override {
        const std::uint32_t n = ut.unit;
        FlowStats& f = st.flows[n];
        // Bazės (vienintelio būtino įėjimo) skaitiklių pokytis — spragos priežasčiai. Sluoksniai išėjimo spragos nesukelia
        // (trūkstamas — skaidrus, layer_absent), todėl jų įvykiai čia neskaičiuojami.
        bool inv = false, tl = false, re = false, det = false;
        if (has_base) {
            const engine::InCounters& c = hub.in(eng.in_id(n, 0));
            engine::InCounters& p = seen[static_cast<std::size_t>(n) * per];
            inv = c.invalid > p.invalid;
            tl = c.too_late > p.too_late;
            re = c.read_errors > p.read_errors;
            det = c.detaches > p.detaches;
            p = c;
            if (c.stalls && c.stalls > f.c["stalls"]) f.c["stalls"] = c.stalls;
        }
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
            case engine::Outcome::GapBlocked: f.loss("missing_fields", ti); break;
            case engine::Outcome::GapClock: f.inc("clock_skips"); break;
            case engine::Outcome::Skipped: st.proc["clock_backward"]++; break;  // AlreadyCommitted — laikrodis atgal (M11)
            case engine::Outcome::OutputError:  // open / commit klaida (ilgalaikė — fatalu)
                f.inc("output_errors");
                times.unit_done(ut.index, now_tai(), false);
                break;
            case engine::Outcome::GapNoContent: break;
            case engine::Outcome::GapStartup: break;  // bazių laukiama prieš variklį (wait-all) — neturėtų būti
            case engine::Outcome::GapUpstream: break;  // bazė — INVALID (aukštupio spraga): gap_upstream, ne praradimas
            case engine::Outcome::GapAlign: break;     // interlaced freeze / black: iki kadro ribos — gap_align, ne praradimas
            case engine::Outcome::GapLate:
                if (inv) f.loss("in_invalid", ti);
                else if (tl) f.loss("too_late", ti);
                else if (re) f.loss("read_errors", ti);
                else if (det) f.loss("flow_invalid", ti);  // [PF-3] gamintojas dingo / perkurtas → slot'as atjungtas
                else {
                    f.loss("late_fields", ti);  // bazė nesulaukė iki ribos → necommit'inama
                    if (f.c["late_fields"] <= 3)
                        std::fprintf(stderr, "[composite] išėjimas %u: indeksas %llu necommit'intas (bazė nesulaukė iki ribos)\n", n,
                                     static_cast<unsigned long long>(ut.index));
                }
                break;
        }
    }

    void end_tick(engine::Index i, engine::TickContext&) override {
        const TickTimes::Result t = times.end(i, now_tai());
        if (static_cast<double>(t.end) >= warm) {
            st.add_hist("proc_ms", t.proc);  // tik apdorojimas (Σ fazių), be laukimo tarp jų
            st.add_hist("wait_ms", t.wait);
            if (t.phases > 1) st.add_hist("held_ms", t.held);  // B / C fazė: laukta vėluojančių
            const mxlRational r = mxl_rate(o.fmt.grain_rate());
            st.add_hist("out_lateness", t.end - mxlIndexToTimestamp(&r, i));
        }
    }
};
}  // namespace

int run_composite(const Opts& o) {
    Stats st;
    st.flows.resize(o.flows);
    st.set_seconds(o.seconds_kept);  // sekundžių žiedai — dabar (ne tick'e)
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
    // Įėjimai per InputSlot (planas §P6, buvusi [PF-3] logika): bazių laukiama pagal --input-wait-ms; dingęs gamintojas → slot'as
    // atjungiamas, grįžęs → prijungiamas iš naujo (`reopened`). Laukiamas formatas ir saugykla — pirmo prijungimo. Sluoksnių
    // nelaukiama (neprivalomi): juos prijungia variklio ciklas, patikra — CompositeFn::validate_layer.
    std::vector<runtime::InputSlot> in;
    in.reserve(static_cast<std::size_t>(N) * per);
    std::vector<VideoOut> out(N);
    auto idx_base = [&](std::uint32_t n) { return n * per; };
    auto in_flow_id = [&](std::uint32_t n, std::uint32_t slot) {  // slot 0 = bazė (jei yra), toliau sluoksniai
        const std::uint32_t k = slot - (has_base ? 1 : 0);
        if (has_base && slot == 0) return flow_id(o.in_role, o.in_base() + n);
        return layer_ids[k].empty() ? flow_id(kRoleGfx, (o.flow_base + n) * 8 + k) : layer_ids[k];
    };
    int rc = 0;
    for (std::uint32_t n = 0; n < N; ++n)
        for (std::uint32_t s = 0; s < per; ++s)
            // bazė — --on-format-change (exit 6 numatyta); sluoksnis grįžęs kitu formatu — tik nebenaudojamas (laukiama, skaidrus)
            in.push_back(has_base && s == 0 ? make_input(o, in_flow_id(n, s), nullptr)
                                            : runtime::InputSlot(in_flow_id(n, s), nullptr, runtime::FormatChange::Wait));
    std::vector<runtime::InputSlot*> base_ptr;
    if (has_base)
        for (std::uint32_t n = 0; n < N; ++n) base_ptr.push_back(&in[idx_base(n)]);
    if (!base_ptr.empty()) rc = attach_inputs(base_ptr, inst, o, "composite", st.fatal);  // 4 — neatsirado per ribą; kStoppedWaiting — sustabdyta
    for (std::uint32_t n = 0; n < N && !rc && has_base; ++n) {
        const runtime::InputSlot& b = in[idx_base(n)];
        if (!vramflow::same_rate(b.format().grain_rate(), o.fmt.grain_rate())) {
            const vramflow::Rational g = b.format().grain_rate(), w = o.fmt.grain_rate();
            st.fatal = "bazė " + std::to_string(n) + ": grain'o ritmas " + std::to_string(g.num) + "/" + std::to_string(g.den) + " != " +
                       std::to_string(w.num) + "/" + std::to_string(w.den) + " (--format " + vramflow::format_name(o.fmt) + ")";
            rc = 1;
        } else if (!(b.format().raster == o.fmt.raster)) {
            st.fatal = "bazė " + std::to_string(n) + ": formatas " + vramflow::format_name(b.format()) + " != --format " +
                       vramflow::format_name(o.fmt);
            rc = 1;
        } else if (b.in().cuda == (cpu || host_gpu)) {
            st.fatal = "bazės storage neatitinka --impl/--storage";
            rc = 1;
        }
    }

    // --- statinė geometrija: bazė, išėjimo rastras ir sluoksnių vieta per bendrą validatorių (sluoksnio rastras — prisijungus) ---
    std::vector<Raster> out_r(N);
    std::vector<Layer> lay(static_cast<std::size_t>(N) * K);
    static const std::uint8_t dummy[8]{};
    for (std::uint32_t n = 0; n < N && !rc; ++n) {
        const Raster base_r = has_base ? in[idx_base(n)].format().raster : Raster{o.fmt.raster.w, o.fmt.raster.h, o.fmt.raster.interlaced, false};
        out_r[n] = Raster{base_r.w, base_r.h, base_r.interlaced, out_alpha};
        // Sluoksnių vieta (--layer-xy, --opacity) nuo sluoksnio flow'o nepriklauso — tikrinama dabar (exit 1), su išėjimo rastro
        // formos pakaitalu; sluoksnio rastras (interlace, dydis) — prisijungus (validate_layer)
        const Raster stand_in{out_r[n].w, out_r[n].h, out_r[n].interlaced, true};
        for (std::uint32_t k = 0; k < K; ++k) lay[n * K + k] = Layer{dummy, stand_in, lx[k], ly[k], lop[k]};
        const CompositeJob j{has_base ? dummy : nullptr, base_r, &lay[n * K], static_cast<int>(K), out_r[n], const_cast<std::uint8_t*>(dummy)};
        std::string why;
        for (int parity = 0; parity < 2 && !rc; ++parity)
            if (!gfx::composite_job_valid(j, parity, &why)) { st.fatal = "išėjimas " + std::to_string(n) + ": " + why; rc = 1; }
    }
    // Briaunos laiko sutartis (V24): fazė iš bazių ir sluoksnių žymų (sluoksniai be žymų / dar neatsiradę — --input-expect-ms
    // arba 0); skelbiama riba — išėjimų flow aprašuose
    // Startas — wait-all (V18): išėjimo rastras = bazės, todėl bazių laukta prieš kuriant išėjimus; sluoksniai (Wanted) varikliui
    // nebūtini — wait_all jų nelaukia
    engine::EngineConfig ecfg = engine_config(o, EngineTiming::Isolation, o.read_offset_us * 1000);
    ecfg.start = engine::StartMode::WaitAll;
    std::vector<InEdge> edges;
    for (std::uint32_t n = 0; n < N; ++n)
        for (std::uint32_t s = 0; s < per; ++s) edges.push_back(in_edge(inst, in_flow_id(n, s), o, 0));
    EdgePlan plan;
    if (!rc && !plan_edges(o, ecfg, edges, false, "composite", st, plan, err)) {
        st.fatal = err;
        rc = 2;
    }
    const std::string opts = cpu ? std::string{} : storage_options(o, dev);
    for (std::uint32_t n = 0; n < N && !rc; ++n) {
        const vramflow::VideoFormat of{out_r[n], o.fmt.frame_rate};
        if (!create_output(inst, out[n], flow_id(kRoleComp, o.flow_base + n), comp_def(o.flow_base + n, of, o.run_seed, plan.tags), opts, of,
                           o, err)) {
            st.fatal = "out " + std::to_string(n) + ": " + err;
            rc = 2;
        }
    }
    // Bazės jau gamina (head žinomas): InputSlot prisijungia tik tada — pirmi tick'ai neskaičiuojami kaip vėlyvi.

    cudaStream_t stream{};
    if (!cpu && !rc && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { st.fatal = "cudaStreamCreate"; rc = 2; }
    // H: device scratch kiekvienai bazei ir išėjimui (sluoksniui — prisijungus, validate_layer)
    std::vector<std::uint8_t*> s_in(in.size(), nullptr), s_out(N, nullptr);
    if (host_gpu && has_base)
        for (std::uint32_t n = 0; n < N && !rc; ++n)
            if (cudaMalloc(reinterpret_cast<void**>(&s_in[idx_base(n)]), gfx::grain_bytes(in[idx_base(n)].format().raster)) != cudaSuccess) {
                st.fatal = "scratch";
                rc = 2;
            }
    if (host_gpu)
        for (std::uint32_t n = 0; n < N && !rc; ++n)
            if (cudaMalloc(reinterpret_cast<void**>(&s_out[n]), gfx::grain_bytes(out_r[n])) != cudaSuccess) { st.fatal = "scratch"; rc = 2; }

    // Bendras tick variklis (planas V11): N vienetų po per = [bazė] + K įėjimų → 1 išėjimas; izoliacijos ribos kaip logo (V10).
    std::vector<engine::MxlVideoInput> ports_in;
    std::vector<engine::MxlVideoOutput> ports_out;
    ports_in.reserve(in.size());
    ports_out.reserve(N);
    for (auto& slot : in) ports_in.emplace_back(slot, inst);
    // K3 spragos (--on-loss): freeze / black kopija — host memcpy arba D2D (G); juodi grain'ai (kiekvienam rastrui) — dabar
    GapFill gap_fill(gap_policy(o.on_loss), !cpu && !host_gpu);
    for (auto& v : out)
        if (!rc && !gap_fill.prepare(v.raster, err)) { st.fatal = err; rc = 2; }
    for (auto& v : out) ports_out.emplace_back(v, gap_fill.copy(), gap_fill.flush(), gap_fill.black(v.raster));
    // Sugedęs vienetas — jo išėjimas atlaisvinamas (K-T11); atkūrimas (D8) — flow'as sukuriamas iš naujo atkūrimo gijoje.
    set_release(ports_out, out, inst, [&](std::size_t n) {
        if (!cpu) cudaSetDevice(dev);
        const vramflow::VideoFormat of{out_r[n], o.fmt.frame_rate};
        std::string e;
        const bool ok = create_output(inst, out[n], flow_id(kRoleComp, o.flow_base + n), comp_def(o.flow_base + n, of, o.run_seed, plan.tags),
                                      opts, of, o, e);
        if (!ok) log_line("composite", "išėjimo " + std::to_string(n) + " atkurti nepavyko: " + e);
        return ok;
    });
    engine::SystemClock clk;
    EngineStats es(o, "composite");
    engine::StatsHub& hub = es.hub;
    st.proc["deadline_us"] = static_cast<std::uint64_t>(ecfg.cuts.deadline_ns / 1000);  // efektyvi riba D (V20)
    engine::TickEngine eng(ecfg, clk, hub, stop);
    for (std::uint32_t n = 0; n < N && !rc; ++n) {
        engine::UnitSpec u;
        u.name = flow_id(kRoleComp, o.flow_base + n);
        // bazė — Required (jos nėra → spraga pagal --on-loss); sluoksniai — Wanted: laukiami iki vieneto ribos, tada be jų (skaidrūs)
        for (std::uint32_t s = 0; s < per; ++s)
            u.in.push_back(engine::InputSpec{&ports_in[n * per + s], edges[n * per + s].t, 1,
                                             has_base && s == 0 ? engine::Need::Required : engine::Need::Wanted, engine::InMode::Exact});
        u.out.push_back(&ports_out[n]);
        std::uint32_t id = 0;
        if (!eng.add_unit(u, id, err)) { st.fatal = "add_unit: " + err; rc = 2; }
        else if (const std::string bad = check_advertised(eng, id, plan); !bad.empty()) { st.fatal = bad; rc = 2; }
    }
    CompositeFn fn(o, st, hub, eng, in, cpu, host_gpu, has_base, K, per, lx, ly, lop, lay, out_r, s_in, s_out, pool, stream,
                   warm_tai_after(o.warmup_s));
    bool engine_loss = false;
    std::uint32_t failed = 0;
    if (!rc) {
        const engine::RunResult res = eng.run(fn);
        es.finish();  // paskutinė statistikos eilutė (Σ eilučių == santrauka)
        engine_loss = res.any_loss;
        failed = res.failed_units;
        rc = engine_rc(res, st);  // Fatal → kodas, FormatChange → 6, InputsTimeout → 4
        if (res.reason == engine::EndReason::FormatChange) {
            // variklis baigė iškart (attach_poll → mismatch), begin_tick jau nebekviečiamas: kuris įėjimas grįžo kitu formatu
            for (std::size_t a = 0; a < in.size(); ++a)
                if (in[a].state() == runtime::InputSlot::State::Mismatch) st.flows[a / per].inc("reopen_mismatch");
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
    // praradimai (pajėgumo / tiekimo) ir variklio (RunResult::any_loss): griežtame režime exit 4, tarnyboje — tik santraukoje
    bool loss = engine_loss;
    for (auto& f : st.flows)
        for (const char* k : {"late_fields", "lapped", "too_late", "in_invalid", "read_errors", "late_skips", "missing_fields", "flow_invalid"})
            if (f.c.count(k) && f.c[k]) loss = true;
    return finish(exit_code_for(o, rc, loss, failed, st));  // kStoppedWaiting — sustabdyta belaukiant įėjimų (exit 0)
}

}  // namespace mxlv::func
