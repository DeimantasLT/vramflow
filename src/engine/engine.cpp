// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Tick ciklas: taisyklės T1–T13 (planas §6 D2), ašinis laukimas (§6 D3), spragos (§6 D4), laikrodžio žingsniai (§6 D5).
//
// Vieneto etapai: WaitA (Quorum — laukia A paleidimo), WaitB (Straggler arba neparuoštas po A), WaitC (pernešta per
// kito tick'o atidarymą). Vienam indeksui ne daugiau kaip po vieną A, B ir C paleidimą (≤ 3 process kvietimai).
//
// Patikslinimai, kurių plano tekstas tiesiogiai nenurodo (ataskaitoje — „nuokrypiai“):
//  * Jei tick'e nėra nė vieno Quorum vieneto (pvz. pirmas tick'as — visi naujai prijungti, Late), A aibė = visi
//    tick'o vienetai: A paleidžiama, kai paruošti visi, bet ne vėliau kaip A(i). Kitaip „tuščias kvorumas“ A
//    paleistų tick'o atidarymo metu be nieko, ir Late žymė (nuimama tik A paleidimo metu) niekada nenukristų.
//  * Ašis renkama iš anksčiausios ribos grupės; lygiai ribai WaitA vienetai pirmesni už Straggler'ius, kad
//    neparuoštas Late įėjimas negalėtų sulaikyti A paleidimo iki pat A ribos.
//  * T11 tikrinama atidarant tick'ą (po miego iki t_open) — taip pavėlavęs pabudimas ir šuolis pirmyn miegant
//    skaičiuojami tame pačiame žingsnyje.
//  * held_ns = A paleidimo laikas − momentas, kai paskutinis į batch'ą patekęs vienetas tapo paruoštas.
#include "engine/engine.hpp"

#include <algorithm>
#include <climits>

namespace mxlv::engine {

namespace {
constexpr std::int64_t kRetryNs = 100'000'000;  // prisijungimo ir gyvumo patikra Stalled būsenoje
constexpr std::int64_t kSliceNs = 1'000'000;    // laukimo gabalas, kai laukia du skirtingi indeksai
constexpr std::int64_t kPollNs = 500'000;       // WaitMode::Poll žingsnis (legacy)

enum class Stage : std::uint8_t { None, WaitA, WaitB, WaitC };

// k ∈ [lo, hi), k ≡ residue (mod stride)
std::uint64_t count_in(Index lo, Index hi, std::uint32_t stride, std::uint32_t residue) {
    auto f = [&](Index n) -> std::uint64_t { return n > residue ? (n - residue - 1) / stride + 1 : 0; };
    return hi > lo ? f(hi) - f(lo) : 0;
}
}  // namespace

const char* to_string(Outcome o) {
    switch (o) {
        case Outcome::Ready: return "Ready";
        case Outcome::GapBlocked: return "GapBlocked";
        case Outcome::GapLate: return "GapLate";
        case Outcome::GapNoContent: return "GapNoContent";
        case Outcome::GapClock: return "GapClock";
        case Outcome::Lapped: return "Lapped";
        case Outcome::Skipped: return "Skipped";
    }
    return "?";
}

EngineConfig EngineConfig::legacy(timing::Rational rate, std::int64_t read_offset_ns, std::int64_t deadline_ns) {
    EngineConfig c;
    c.cuts = legacy_cuts(rate, read_offset_ns, deadline_ns);
    c.stall_after = 0;
    c.on_loss = GapPolicy::Leave;
    c.start = StartMode::WaitAll;
    c.wait = WaitMode::Poll;
    return c;
}

struct TickEngine::Impl {
    struct In {
        InputSpec spec;
        InputState st;
        std::uint32_t unit = 0, hub = 0, g = 0;  // g — poslinkis vieneto GrainRef'uose
        std::int64_t ewma = 0;
        bool ewma_set = false, ever = false;
        std::int64_t next_attach = INT64_MIN, next_alive = INT64_MIN;  // monotoninis laikas
        bool done = false, ok = false;  // einamojo vieneto indekso būsena
    };
    struct Out {
        OutputPort* port;
        GapState gs;
        std::uint32_t hub;
        bool flush = false;
    };
    struct Unit {
        std::uint32_t stride = 1, residue = 0;
        std::uint32_t in0 = 0, nin = 0, out0 = 0, nout = 0, g0 = 0, ng = 0;
        Index cur = kNoIndex, join = kNoIndex;
        Stage stage = Stage::None;
        bool failed = false, no_content = false;
        std::int64_t ready_at = 0;
    };
    struct Rec {
        Index idx = kNoIndex;
        Cuts cuts{};
        bool a_done = false, b_done = false, c_done = false, ended = true, opening = false;
        std::uint32_t unresolved = 0;
        std::uint8_t calls = 0;
        std::int64_t held = 0;
    };

    Impl(const EngineConfig& c, Clock& k, StatsHub& h, const std::atomic<bool>& s)
        : cfg(c), clk(k), hub(h), stop(s), ctx(this) {}

    EngineConfig cfg;
    Clock& clk;
    StatsHub& hub;
    const std::atomic<bool>& stop;
    TickContext ctx;
    TickFunction* fn = nullptr;

    std::vector<In> ins;
    std::vector<Out> outs;
    std::vector<Unit> units;
    std::vector<GrainRef> grefs;
    std::vector<void*> slots;  // vienas kiekvienam išėjimui (indeksas == outs)
    std::vector<UnitTick> batch;
    Rec recs[2];

    StepDetector sd;
    bool step_in_tick = false;
    std::int64_t last_step_mono = INT64_MIN;
    std::uint32_t pending = 0;
    Index ctx_index = kNoIndex;
    std::int64_t ctx_held = 0;
    bool ended = false, any_failed = false, inputs_checked = false;
    RunResult res;
    std::int64_t start_tai = 0, start_mono = 0;
    std::string why_;  // mismatch tekstui (alokuoja tik klaidos kelyje)

    // --- pagalbinės ---------------------------------------------------------------------------------------------
    std::int64_t now() const { return static_cast<std::int64_t>(clk.now_tai()); }
    std::int64_t T(Index c) const { return timing::time_of(c, cfg.cuts.rate); }
    Rec& rec(Index c) { return recs[c & 1]; }
    std::uint32_t uid(const Unit& u) const { return static_cast<std::uint32_t>(&u - units.data()); }
    Index jidx(const Unit& u, const In& in) const {
        return static_cast<Index>(static_cast<std::int64_t>(u.cur) + in.spec.timing.index_offset);
    }
    bool waitable(const In& in) const { return !in.done && in.spec.need != Need::Optional && in.st.live(); }

    void end_fatal(EndReason r, int code, std::string why) {
        if (ended) return;
        ended = true;
        res.reason = r;
        res.exit_code = code;
        res.fatal = std::move(why);
    }

    bool ready(const Unit& u, bool final_cut) const {
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            const In& in = ins[u.in0 + k];
            if (in.ok || in.spec.need == Need::Optional) continue;
            if (in.spec.need == Need::Wanted && (in.done || final_cut)) continue;
            return false;
        }
        return true;
    }

    std::int64_t deadline(const Unit& u) {
        const Rec& r = rec(u.cur);
        if (!r.a_done) return r.cuts.a;
        if (u.stage == Stage::WaitC) return r.cuts.c;
        return r.cuts.has_b ? r.cuts.b : r.cuts.a;
    }

    // --- įėjimų skaitymas ------------------------------------------------------------------------------------------
    void mark_ok(Unit& u, In& in, Index j) {
        in.done = in.ok = true;
        u.ready_at = std::max(u.ready_at, now());
        const auto lat = static_cast<std::int64_t>(in.spec.port->last_write_tai()) - T(j);
        in.ewma = in.ewma_set ? in.ewma + (lat - in.ewma) / 4 : lat;
        in.ewma_set = true;
    }

    void detach(In& in) { in.spec.port->detach(); }

    void apply_fail(Unit& u, In& in, Probe p) {
        InCounters& c = hub.in(in.hub);
        switch (p) {
            case Probe::Invalid: in.st.on_invalid_grain(c); break;
            case Probe::TooLate: in.st.on_too_late(c); break;
            case Probe::Gone:
                in.st.on_gone(c);
                detach(in);
                break;
            case Probe::Error: in.st.on_read_error(c); break;
            case Probe::Ready:
            case Probe::NotYet: return;
        }
        in.done = true;
        in.ok = false;
        if (in.spec.need == Need::Required && u.stage != Stage::None) gap_unit(u, Outcome::GapLate, 0);
    }

    // Neblokuojantis bandymas: head() (be sisteminio kvietimo) ir, jei pasiekė j, probe kiekvienam grain'ui.
    void try_resolve(Unit& u, In& in) {
        const Index h = in.spec.port->head();
        if (h == kNoIndex) return;
        Index jt = jidx(u, in);
        if (in.spec.mode == InMode::Latest && h < jt) jt = h;
        if (h < jt) return;
        const std::uint32_t g = in.spec.grains;
        for (std::uint32_t n = 0; n < g; ++n) {
            GrainRef& ref = grefs[u.g0 + in.g + n];
            const Index jg = jt - (g - 1) + n;
            const Probe p = in.spec.port->probe(jg, ref);
            if (p == Probe::NotYet) return;
            if (p != Probe::Ready) {
                apply_fail(u, in, p);
                return;
            }
        }
        mark_ok(u, in, jt);
    }

    void poll_all() {
        for (Unit& u : units) {
            if (u.stage == Stage::None) continue;
            for (std::uint32_t k = 0; k < u.nin && u.stage != Stage::None; ++k) {
                In& in = ins[u.in0 + k];
                if (!in.done && in.st.live()) try_resolve(u, in);
            }
        }
    }

    // --- vieneto gyvavimas indekse --------------------------------------------------------------------------------
    void maybe_end(Rec& r) {
        if (r.ended || r.opening || r.unresolved != 0 || ended) return;
        r.ended = true;
        ctx_index = r.idx;
        fn->end_tick(r.idx, ctx);
    }

    void resolve_unit(Unit& u) {
        Rec& r = rec(u.cur);
        if (r.idx == u.cur && r.unresolved > 0) --r.unresolved;
        u.stage = Stage::None;
        --pending;
        if (u.join != kNoIndex) {
            const Index j = u.join;
            u.join = kNoIndex;
            Rec& rj = rec(j);
            if (rj.idx == j && !u.failed) start_unit(u, j, rj);
        }
        maybe_end(r);
    }

    void gap_unit(Unit& u, Outcome why, std::uint8_t phase) {
        const Index c = u.cur;
        bool startup = false;
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            const In& in = ins[u.in0 + k];
            if (in.spec.need == Need::Required && !in.ever) startup = true;
        }
        for (std::uint32_t m = 0; m < u.nout; ++m) {
            Out& o = outs[u.out0 + m];
            OutCounters& oc = hub.out(o.hub);
            Index from = kNoIndex;
            const GapAction a = o.gs.decide(c, T(c), o.port->interlaced(), o.port->slot_count(), from);
            if (a == GapAction::None) {
                ++(startup ? oc.gap_startup : oc.gap_left);
                o.gs.on_gap(c, a);
                continue;
            }
            if (!o.port->fill_gap(c, a, from)) {
                ++oc.commit_errors;
                continue;
            }
            o.gs.on_gap(c, a);
            o.flush = true;
            if (startup) ++oc.gap_startup;
            else if (a == GapAction::Invalid) ++oc.gap_invalid;
            else if (a == GapAction::Repeat) ++oc.gap_frozen;
            else ++oc.gap_black;
        }
        if (why == Outcome::GapClock) ++hub.engine().step_gaps;
        const UnitTick ut{uid(u), c, phase, {}, {}};
        ctx_index = c;
        fn->end_unit(ut, why, ctx);
        resolve_unit(u);
    }

    // Paskutinė riba praėjo: neparuoštiems įėjimams miss_run++ (T7 / T8), vienetas — spraga.
    void finalize_miss(Unit& u, std::uint8_t phase) {
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            In& in = ins[u.in0 + k];
            if (in.ok || in.done || in.spec.need == Need::Optional || !in.st.live()) continue;
            if (in.st.on_missed(hub.in(in.hub))) in.st.note_head(in.spec.port->head());
        }
        gap_unit(u, Outcome::GapLate, phase);
    }

    void start_unit(Unit& u, Index c, Rec& r) {
        u.cur = c;
        u.stage = Stage::WaitA;
        ++pending;
        ++r.unresolved;
        u.ready_at = 0;
        for (std::uint32_t n = 0; n < u.ng; ++n) grefs[u.g0 + n] = GrainRef{};
        bool blocked = false, on_time = true;
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            In& in = ins[u.in0 + k];
            in.done = in.ok = false;
            if (!in.st.live()) {
                if (in.spec.need == Need::Required) blocked = true;
                else in.done = true;  // Wanted / Optional be rašytojo — tęsiama be jo
                continue;
            }
            if (in.st.late() && in.spec.need != Need::Optional) on_time = false;
        }
        if (u.nin == 0) {
            if (u.no_content) return gap_unit(u, Outcome::GapNoContent, 0);
            u.ready_at = now();
        }
        if (blocked) return gap_unit(u, Outcome::GapBlocked, 0);  // T3 / T4: spraga atidarymo metu
        const std::int64_t n = now();
        if (!r.a_done) {
            u.stage = on_time ? Stage::WaitA : Stage::WaitB;
        } else if (r.cuts.has_b && !r.b_done && n < r.cuts.b) {
            u.stage = Stage::WaitB;  // prisijungė po pernešimo (T2)
        } else if (r.cuts.has_c && !r.c_done && n < r.cuts.c) {
            u.stage = Stage::WaitC;
        } else {
            ++hub.engine().carry_overrun;  // T2: i ribos jau praėjo
            gap_unit(u, Outcome::GapLate, 0);
        }
    }

    // --- paleidimai ----------------------------------------------------------------------------------------------
    template <class Pred>
    void launch(Rec& r, std::uint8_t phase, Pred pred) {
        batch.clear();
        const Index idx = r.idx;
        std::int64_t latest_ready = INT64_MIN;
        for (Unit& u : units) {
            if (u.stage == Stage::None || u.cur != idx || !pred(u)) continue;
            bool hold = false;  // K-T5: interlaced freeze / black — turinys grįžta tik kadro riba
            for (std::uint32_t m = 0; m < u.nout; ++m)
                if (!outs[u.out0 + m].gs.may_resume(idx, outs[u.out0 + m].port->interlaced())) hold = true;
            if (hold) {
                count_inputs(u, phase);
                gap_unit(u, Outcome::GapBlocked, phase);
                continue;
            }
            bool opened = true;
            for (std::uint32_t m = 0; m < u.nout; ++m) {
                void* s = nullptr;
                const OpenResult o = outs[u.out0 + m].port->open(idx, s);
                if (o == OpenResult::Ok) {
                    slots[u.out0 + m] = s;
                    continue;
                }
                for (std::uint32_t p = 0; p < m; ++p) outs[u.out0 + p].port->cancel();
                if (o == OpenResult::Error) ++hub.out(outs[u.out0 + m].hub).commit_errors;
                opened = false;
                break;
            }
            if (!opened) {
                const UnitTick ut{uid(u), idx, phase, {}, {}};
                fn->end_unit(ut, Outcome::Skipped, ctx);
                resolve_unit(u);
                continue;
            }
            latest_ready = std::max(latest_ready, u.ready_at);
            batch.push_back(UnitTick{uid(u), idx, phase, std::span<const GrainRef>(grefs.data() + u.g0, u.ng),
                                     std::span<void* const>(slots.data() + u.out0, u.nout)});
        }
        if (batch.empty()) return;
        const std::int64_t n = now();
        if (phase == 0) {
            r.held = latest_ready == INT64_MIN ? 0 : std::max<std::int64_t>(0, n - latest_ready);
            hub.engine().held_ns_max = std::max(hub.engine().held_ns_max, static_cast<std::uint64_t>(r.held));
        }
        ctx_index = idx;
        ctx_held = r.held;
        const bool ok = fn->process(Batch{idx, phase, std::span<const UnitTick>(batch.data(), batch.size())}, ctx);
        ++r.calls;
        ++hub.engine().process_calls;
        if (!ok || ended) {
            for (const UnitTick& ut : batch) {
                Unit& u = units[ut.unit];
                for (std::uint32_t m = 0; m < u.nout; ++m) outs[u.out0 + m].port->cancel();
                u.stage = Stage::None;
                --pending;
            }
            end_fatal(EndReason::Fatal, 2, "TickFunction::process grąžino false");
            return;
        }
        const Index now_idx = due_index(clk.now_tai(), cfg.cuts);
        for (const UnitTick& ut : batch) {
            Unit& u = units[ut.unit];
            count_inputs(u, phase);
            bool lap = false;  // T12
            for (std::uint32_t k = 0; k < u.nin && !lap; ++k) {
                const In& in = ins[u.in0 + k];
                if (!in.ok) continue;
                const Index j = jidx(u, in);
                for (std::uint32_t g = 0; g < in.spec.grains; ++g)
                    if (in.spec.port->lapped(j - (in.spec.grains - 1) + g, now_idx)) lap = true;
            }
            Outcome oc = Outcome::Ready;
            for (std::uint32_t m = 0; m < u.nout; ++m) {
                Out& o = outs[u.out0 + m];
                OutCounters& c = hub.out(o.hub);
                if (lap) {
                    o.port->cancel();
                    ++c.lapped;
                    oc = Outcome::Lapped;
                } else if (o.port->commit()) {
                    ++(phase == 0 ? c.ok : c.late);
                    o.gs.on_content(idx, T(idx));
                } else {
                    ++c.commit_errors;
                }
            }
            fn->end_unit(ut, oc, ctx);
            resolve_unit(u);
        }
        if (phase == 1 && n > cuts_for(idx + 1, cfg.cuts).open) ++hub.engine().guard_overrun;
    }

    void count_inputs(Unit& u, std::uint8_t phase) {
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            In& in = ins[u.in0 + k];
            InCounters& c = hub.in(in.hub);
            if (in.ok) {
                if (phase == 0) in.st.on_ready_at_a(c);
                else in.st.on_ready_late(c);
            } else if (!in.done && in.spec.need == Need::Wanted && in.st.live()) {
                if (in.st.on_missed(c)) in.st.note_head(in.spec.port->head());
            }
        }
    }

    void launch_a(Rec& r) {
        r.a_done = true;
        const bool final_cut = !r.cuts.has_b;
        const Index idx = r.idx;
        launch(r, 0, [&](const Unit& u) {
            return (u.stage == Stage::WaitA || u.stage == Stage::WaitB) && ready(u, final_cut);
        });
        for (Unit& u : units) {
            if (u.stage == Stage::None || u.cur != idx) continue;
            if (final_cut) {
                finalize_miss(u, 0);
                continue;
            }
            if (u.stage != Stage::WaitA) continue;
            bool dead = false;  // T6
            for (std::uint32_t k = 0; k < u.nin; ++k) {
                In& in = ins[u.in0 + k];
                if (in.ok || in.done || !in.st.live() || in.st.late() || in.spec.need == Need::Optional) continue;
                const bool alive = in.spec.port->writer_alive();
                in.st.on_unready_at_a(alive, hub.in(in.hub));
                if (!alive) {
                    detach(in);
                    if (in.spec.need == Need::Required) dead = true;
                    else in.done = true;
                }
            }
            if (dead) gap_unit(u, Outcome::GapLate, 0);
            else u.stage = Stage::WaitB;
        }
    }

    template <class F>
    void for_stage(Index idx, Stage s, F f) {
        for (Unit& u : units)
            if (u.stage == s && u.cur == idx) f(u);
    }

    void fire_due() {
        for (Rec& r : recs) {
            if (r.idx == kNoIndex || ended) continue;
            const std::int64_t n = now();
            if (!r.a_done) {
                bool any_quorum = false, wait = false;
                for_stage(r.idx, Stage::WaitA, [&](const Unit& u) {
                    any_quorum = true;
                    if (!ready(u, false)) wait = true;
                });
                if (!any_quorum)
                    for_stage(r.idx, Stage::WaitB, [&](const Unit& u) {
                        if (!ready(u, false)) wait = true;
                    });
                if (!wait || n >= r.cuts.a) launch_a(r);
            }
            if (ended) return;
            if (r.a_done && r.cuts.has_b && !r.b_done) {
                bool any = false, all = true;
                for_stage(r.idx, Stage::WaitB, [&](const Unit& u) {
                    any = true;
                    if (!ready(u, false)) all = false;
                });
                if (any && all) {  // T7: visi likę paruošti
                    r.b_done = true;
                    launch(r, 1, [](const Unit& u) { return u.stage == Stage::WaitB; });
                } else if (n >= r.cuts.b) {
                    r.b_done = true;
                    launch(r, 1, [&](const Unit& u) { return u.stage == Stage::WaitB && ready(u, !r.cuts.has_c); });
                    for_stage(r.idx, Stage::WaitB, [&](Unit& u) {
                        if (r.cuts.has_c) u.stage = Stage::WaitC;  // T8: pernešimas
                        else finalize_miss(u, 1);
                    });
                }
            }
            if (ended) return;
            if (r.cuts.has_c && r.b_done && !r.c_done) {
                bool any = false, all = true;
                for_stage(r.idx, Stage::WaitC, [&](const Unit& u) {
                    any = true;
                    if (!ready(u, false)) all = false;
                });
                if (any && all) {
                    r.c_done = true;
                    launch(r, 2, [](const Unit& u) { return u.stage == Stage::WaitC; });
                } else if (n >= r.cuts.c) {
                    r.c_done = true;
                    launch(r, 2, [&](const Unit& u) { return u.stage == Stage::WaitC && ready(u, true); });
                    for_stage(r.idx, Stage::WaitC, [&](Unit& u) { finalize_miss(u, 2); });
                }
            }
        }
    }

    void flush() {
        for (Out& o : outs)
            if (o.flush) {
                o.flush = false;
                if (!o.port->flush_gaps()) ++hub.out(o.hub).commit_errors;
            }
    }

    // --- laikrodis ------------------------------------------------------------------------------------------------
    void check_step() {
        const std::int64_t d = sd.check(clk);
        if (d == 0) return;
        EngineCounters& e = hub.engine();
        ++e.clock_steps;
        if (d < 0) ++e.clock_backward;
        step_in_tick = true;
        last_step_mono = clk.now_mono_raw();
        for (Unit& u : units) {  // atviri tick'ai uždaromi: priežastis — laikrodis
            u.join = kNoIndex;
            if (u.stage != Stage::None) gap_unit(u, Outcome::GapClock, 0);
        }
        fn->on_clock_step(d, ctx);
    }

    // --- laukimas (D3) --------------------------------------------------------------------------------------------
    void pump(std::int64_t until, bool until_idle) {
        for (;;) {
            poll_all();
            fire_due();
            flush();
            if (ended || stop.load(std::memory_order_relaxed)) return;
            if (until_idle && pending == 0) return;
            const std::int64_t n = now();
            if (n >= until) return;
            std::int64_t ev = until;
            Index seen = kNoIndex;
            bool multi = false;
            std::int64_t best_dl = INT64_MAX;
            int best_rank = 2;
            for (Unit& u : units) {
                if (u.stage == Stage::None) continue;
                const std::int64_t dl = deadline(u);
                ev = std::min(ev, dl);
                if (seen == kNoIndex) seen = u.cur;
                else if (u.cur != seen) multi = true;
                bool w = false;
                for (std::uint32_t k = 0; k < u.nin; ++k) w = w || waitable(ins[u.in0 + k]);
                const int rank = u.stage == Stage::WaitA ? 0 : 1;
                if (w && (dl < best_dl || (dl == best_dl && rank < best_rank))) {
                    best_dl = dl;
                    best_rank = rank;
                }
            }
            std::int64_t pivot = -1;
            for (Unit& u : units) {  // ašis: didžiausias ewma_late; lygybė — didžiausias įėjimo numeris
                if (u.stage == Stage::None || deadline(u) != best_dl) continue;
                if ((u.stage == Stage::WaitA ? 0 : 1) != best_rank) continue;
                for (std::uint32_t k = 0; k < u.nin; ++k) {
                    const auto gk = static_cast<std::int64_t>(u.in0 + k);
                    if (!waitable(ins[gk])) continue;
                    if (pivot < 0 || ins[gk].ewma >= ins[pivot].ewma) pivot = gk;
                }
            }
            if (ev <= n) ev = n + 1;  // apsauga nuo sukimosi (neturėtų įvykti)
            if (pivot >= 0 && cfg.wait == WaitMode::Pivot) {
                In& in = ins[pivot];
                Unit& u = units[in.unit];
                std::int64_t to = ev - n;
                if (multi) to = std::min(to, kSliceNs);
                GrainRef g;
                const Probe p = in.spec.port->wait(jidx(u, in), static_cast<std::uint64_t>(to), g);
                if (stop.load(std::memory_order_relaxed)) return;  // T13 — prieš bet kokį sprendimą
                check_step();
                if (u.stage != Stage::None && !in.done) {
                    if (p == Probe::Ready) try_resolve(u, in);
                    else if (p != Probe::NotYet) apply_fail(u, in, p);
                }
            } else {
                std::int64_t t = ev;
                if (cfg.wait == WaitMode::Poll && pivot >= 0) t = std::min(t, n + kPollNs);
                clk.sleep_until(static_cast<std::uint64_t>(t), stop);
                if (stop.load(std::memory_order_relaxed)) return;
                check_step();
            }
        }
    }

    // --- tick'o atidarymas ----------------------------------------------------------------------------------------
    void maintain_inputs() {
        const std::int64_t mono = clk.now_mono_raw();
        for (In& in : ins) {
            InCounters& c = hub.in(in.hub);
            const InState s = in.st.state();
            if (s == InState::Waiting || s == InState::Detached) {
                if (mono < in.next_attach) continue;
                in.next_attach = mono + kRetryNs;
                if (!in.spec.port->attach_poll(clk.now_tai())) continue;
                if (in.spec.port->mismatch(why_)) {
                    in.st.on_mismatch();
                    end_fatal(EndReason::FormatChange, 6, why_);
                    return;
                }
                in.st.on_attach(c);
                in.ever = true;
            } else if (s == InState::Stalled) {
                if (in.st.on_stalled_head(in.spec.port->head())) continue;  // T10
                if (mono < in.next_alive) continue;
                in.next_alive = mono + kRetryNs;
                if (!in.spec.port->writer_alive()) {
                    in.st.on_stalled_dead(c);
                    detach(in);
                }
            }
        }
        if (cfg.input_wait_ns > 0 && !inputs_checked && mono - start_mono >= cfg.input_wait_ns) {
            inputs_checked = true;
            for (const In& in : ins)
                if (in.spec.need == Need::Required && !in.ever) {
                    end_fatal(EndReason::InputsTimeout, 4, "būtinas įėjimas neprisijungė per --input-wait-ms");
                    return;
                }
        }
    }

    void open_tick(Index c) {
        Rec& r = rec(c);
        if (r.idx != kNoIndex && r.idx != c) {  // senas indeksas (≤ c − 2) — negali likti neišspręstas
            for (Unit& u : units) {
                if (u.stage != Stage::None && u.cur == r.idx) {
                    u.join = kNoIndex;
                    gap_unit(u, Outcome::GapLate, 2);
                }
            }
        }
        r = Rec{};
        r.idx = c;
        r.cuts = cuts_for(c, cfg.cuts);
        r.ended = false;
        r.opening = true;
        ++hub.engine().ticks;
        if (res.first_index == kNoIndex) res.first_index = c;
        res.last_index = c;
        ctx_index = c;
        ctx_held = 0;
        maintain_inputs();
        if (ended) return;
        for (Unit& u : units) u.no_content = false;
        fn->begin_tick(c, ctx);
        if (ended) return;
        for (Unit& u : units) {
            if (u.failed || c % u.stride != u.residue) continue;
            if (u.stage != Stage::None) {  // T2: dar laukia ankstesnio (pernešto) indekso
                u.join = c;
                continue;
            }
            start_unit(u, c, r);
        }
        r.opening = false;
        maybe_end(r);
    }

    Index align_up(Index x) const {
        const Index a = std::max<std::uint32_t>(1, cfg.skip_align);
        return (x + a - 1) / a * a;
    }

    void skip(Index from, Index to) {  // T11: praleisti [from, to) — neužpildomi (K-T6)
        const bool warm = static_cast<double>(now() - start_tai) < cfg.warmup_s * 1e9;
        for (Unit& u : units) {
            if (u.failed || u.nout == 0) continue;
            const std::uint64_t n = count_in(from, to, u.stride, u.residue);
            for (std::uint32_t m = 0; m < u.nout; ++m) {
                OutCounters& c = hub.out(outs[u.out0 + m].hub);
                if (step_in_tick) c.clock_skips += n;
                else if (warm) c.warmup_skips += n;
                else c.late_skips += n;
            }
        }
    }

    bool wait_all() {  // StartMode::WaitAll — nepradedama, kol neprisijungė visi būtini įėjimai
        for (;;) {
            if (stop.load()) return false;
            maintain_inputs();
            if (ended) return false;
            bool all = true;
            for (const In& in : ins)
                if (in.spec.need == Need::Required && !in.ever) all = false;
            if (all) return true;
            clk.sleep_until(clk.now_tai() + static_cast<std::uint64_t>(kRetryNs), stop);
        }
    }

    RunResult run(TickFunction& f) {
        fn = &f;
        hub.prepare();
        batch.reserve(units.size());
        sd.check(clk);
        start_tai = now();
        start_mono = clk.now_mono_raw();
        const auto dur_ns = static_cast<std::int64_t>(cfg.duration_s * 1e9);
        if (cfg.start == StartMode::WaitAll && !wait_all()) {
            if (!ended) res.reason = EndReason::Stopped;
            fn->finish(ctx);
            return res;
        }
        Index next = due_index(clk.now_tai(), cfg.cuts);
        while (cuts_for(next, cfg.cuts).open < now()) ++next;
        next = align_up(next);
        while (!ended) {
            if (stop.load()) {
                res.reason = EndReason::Stopped;
                break;
            }
            const Cuts nc = cuts_for(next, cfg.cuts);
            if (dur_ns > 0 && nc.open >= start_tai + dur_ns) {
                pump(INT64_MAX, true);
                if (!ended) res.reason = EndReason::Duration;
                break;
            }
            if (cfg.open_wait && pending == 0) {
                if (!cfg.open_wait(next, cfg.cuts.expect_ns + cfg.cuts.read_offset_ns, stop)) continue;
            } else {
                pump(nc.open, false);
            }
            if (ended) break;
            if (stop.load()) {
                res.reason = EndReason::Stopped;
                break;
            }
            check_step();
            if (now() < nc.open) continue;  // laikrodis atgal — miegama iki t_open naujoje laiko juostoje
            const Index due = due_index(clk.now_tai(), cfg.cuts);
            if (due > next) {
                const Index to = align_up(due);
                skip(next, to);
                next = to;
                if (now() < cuts_for(next, cfg.cuts).open) continue;
            }
            step_in_tick = false;
            open_tick(next);
            ++next;
        }
        // T13: atidarytų grain'ų už launch() ribų nėra; neišspręsti vienetai — ne praradimas
        for (Unit& u : units) {
            u.stage = Stage::None;
            u.join = kNoIndex;
        }
        pending = 0;
        fn->finish(ctx);
        if (any_failed && res.exit_code == 0) res.exit_code = 2;
        for (std::size_t k = 0; k < hub.inputs(); ++k)
            if (hub.in(static_cast<std::uint32_t>(k)).lost != 0) res.any_loss = true;
        for (std::size_t k = 0; k < hub.outputs(); ++k) {
            const OutCounters& c = hub.out(static_cast<std::uint32_t>(k));
            if (c.gap_invalid + c.gap_frozen + c.gap_black + c.gap_left + c.lapped + c.late_skips + c.clock_skips != 0)
                res.any_loss = true;
        }
        return res;
    }
};

// --- TickContext ---------------------------------------------------------------------------------------------------
Clock& TickContext::clock() { return static_cast<TickEngine::Impl*>(e_)->clk; }
Index TickContext::index() const { return static_cast<const TickEngine::Impl*>(e_)->ctx_index; }
void TickContext::no_content(std::uint32_t unit) { static_cast<TickEngine::Impl*>(e_)->units[unit].no_content = true; }
void TickContext::fail_unit(std::uint32_t unit, std::string) {
    auto* d = static_cast<TickEngine::Impl*>(e_);
    d->units[unit].failed = true;
    d->any_failed = true;
}
void TickContext::fatal(int exit_code, std::string why) {
    static_cast<TickEngine::Impl*>(e_)->end_fatal(EndReason::Fatal, exit_code, std::move(why));
}
void TickContext::count(std::uint32_t unit, CounterId id, std::uint64_t n) {
    static_cast<TickEngine::Impl*>(e_)->hub.count(unit, id, n);
}
void TickContext::hist(HistId id, std::uint64_t ns) { static_cast<TickEngine::Impl*>(e_)->hub.hist(id, ns); }
bool TickContext::clock_event_recent(std::uint64_t window_ns) const {
    const auto* d = static_cast<const TickEngine::Impl*>(e_);
    return d->last_step_mono != INT64_MIN &&
           d->clk.now_mono_raw() - d->last_step_mono <= static_cast<std::int64_t>(window_ns);
}
std::int64_t TickContext::held_ns() const { return static_cast<const TickEngine::Impl*>(e_)->ctx_held; }

// --- TickEngine ----------------------------------------------------------------------------------------------------
TickEngine::TickEngine(const EngineConfig& c, Clock& k, StatsHub& h, const std::atomic<bool>& s)
    : d_(std::make_unique<Impl>(c, k, h, s)) {}
TickEngine::~TickEngine() = default;

bool TickEngine::add_unit(const UnitSpec& s, std::uint32_t& unit, std::string& why) {
    Impl& d = *d_;
    if (d.cfg.admit && !d.cfg.admit(s, why)) return false;
    if (s.stride == 0 || s.residue >= s.stride) {
        why = "stride / residue";
        return false;
    }
    for (const InputSpec& i : s.in)
        if (i.port == nullptr || i.grains == 0 || i.timing.index_offset > 0) {
            why = "įėjimo aprašas";
            return false;
        }
    for (const OutputPort* o : s.out)
        if (o == nullptr) {
            why = "išėjimas nullptr";
            return false;
        }
    Impl::Unit u;
    u.stride = s.stride;
    u.residue = s.residue;
    const auto id = static_cast<std::uint32_t>(d.units.size());
    u.in0 = static_cast<std::uint32_t>(d.ins.size());
    u.nin = static_cast<std::uint32_t>(s.in.size());
    for (std::size_t k = 0; k < s.in.size(); ++k) {
        Impl::In in{s.in[k], InputState(d.cfg.stall_after)};
        in.unit = id;
        in.hub = d.hub.add_input(s.name + "/in" + std::to_string(k));
        in.g = u.ng;
        u.ng += s.in[k].grains;
        d.ins.push_back(in);
    }
    u.g0 = static_cast<std::uint32_t>(d.grefs.size());
    d.grefs.resize(d.grefs.size() + u.ng);
    u.out0 = static_cast<std::uint32_t>(d.outs.size());
    u.nout = static_cast<std::uint32_t>(s.out.size());
    for (std::size_t m = 0; m < s.out.size(); ++m) {
        GapState g;
        g.policy = d.cfg.on_loss;
        g.hold_ns = d.cfg.hold_ns;
        d.outs.push_back(Impl::Out{s.out[m], g, d.hub.add_output(s.name + "/out" + std::to_string(m))});
    }
    d.slots.resize(d.outs.size(), nullptr);
    d.hub.add_unit(s.name);
    d.units.push_back(u);
    unit = id;
    return true;
}

RunResult TickEngine::run(TickFunction& f) { return d_->run(f); }

std::uint32_t TickEngine::in_id(std::uint32_t unit, std::uint32_t input) const {
    return d_->ins[d_->units[unit].in0 + input].hub;
}
std::uint32_t TickEngine::out_id(std::uint32_t unit, std::uint32_t output) const {
    return d_->outs[d_->units[unit].out0 + output].hub;
}
InState TickEngine::in_state(std::uint32_t unit, std::uint32_t input) const {
    return d_->ins[d_->units[unit].in0 + input].st.state();
}
bool TickEngine::in_late(std::uint32_t unit, std::uint32_t input) const {
    return d_->ins[d_->units[unit].in0 + input].st.late();
}

}  // namespace mxlv::engine
