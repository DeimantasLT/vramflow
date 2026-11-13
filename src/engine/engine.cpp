// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Tick ciklas: taisyklės T1–T13 (planas §6 D2), ašinis laukimas (§6 D3), spragos (§6 D4), laikrodžio žingsniai (§6 D5).
//
// Vieneto etapai indekse c:
//   WaitA — Quorum (visi įėjimai Live be Late): A laukia jo iki a(c);
//   WaitB — „šviežias“ Straggler (neparuoštų būtinų įėjimų miss_run == 0): B laukia jo iki b(c);
//   WaitC — pernešta per kito tick'o atidarymą (C), arba „pakibęs“ Straggler (miss_run ≥ 1), kurio nelaukiama:
//           paleidžiamas, jei paruoštas B paleidimo metu ar vėliau iki savo ribos (has_c ? c : b), kitaip — spraga.
// Vienam indeksui ≤ 3 process kvietimai (A, B, C) ir retas ketvirtas — paskutinis batch'as ties c(i) (žr. C taisyklę).
//
// Taisyklės, kurių plano tekstas nepasako tiesiogiai (sprendimai po peržiūros, 2026-11-12):
//  * „Laiku“ — įėjimo grain'as paruoštas ne vėliau kaip a(c); tai lemia įėjimo ok / late. Late nuimama po
//    late_clear_run (8) laiku iš eilės; kol Late — A jo nelaukia, bet paima, jei paruoštas paleidimo metu.
//  * Jei tick'e nėra Quorum vienetų, A laukia šviežių Straggler'ių (iki a(c)); jei tick'e kol kas tik
//    prisijungsiantys po pernešimo vienetai — A nepaleidžiama tuščia, laukiama jų arba a(c).
//  * Prisijungiantys po pernešimo — tick'o nariai: end_tick(c) tik jiems išsisprendus.
//  * Paleidimų tvarka: pirmiausia naujausio tick'o A, tik tada senesnio B / C — sveiki vienetai nelaukia
//    pernešto vieneto apdorojimo.
//  * Ašis: pirmiausia iš naujausio tick'o A aibės, kitaip — anksčiausios ribos grupė.
//  * T11: praleidžiami tik indeksai, kurių kito tick'o atidarymas jau praėjo (pavėlavimas < P nieko neišmeta).
//  * Paruošimo laikas = min(dabar, last_write_tai) — ne stebėjimo momentas (ašis gali būti kitas įėjimas).
//  * held_ns = A paleidimas − momentas, kai paskutinis į batch'ą patekęs vienetas tapo paruoštas.
//  * Wanted įėjimas (enc garsas) laukiamas kaip Required: ta pati Late histerezė ir vieneto klasė (Quorum / Straggler),
//    B / C fazės, ašis. Skirtumas tik vieneto paskutinėje riboje (has_c ? c : b): nesulaukus Wanted, vienetas apdorojamas be
//    jo (ptr == nullptr; enc — tyla), ne spraga; Wanted be rašytojo (ne Live / negyvas ties A / Gone / klaida) — tęsiama be jo.
//    Nuolat vėluojantis Wanted į B / C perkelia tik savo vienetą; vienas vėlus grain'as kainuoja tik tą indeksą (+ vienkartinis
//    kvorumo laikymas iki a(c), kaip ir Required).
//  * B taisyklė: kai bent vienas Straggler'is paruoštas (anksčiausias — t_r), B laukia tik tų neparuoštų šviežių
//    Straggler'ių, kurių laukiamas paruošimo laikas T(j) + ewma ≤ t_r + Wr (nežinomas — laukiama), ir ne ilgiau kaip
//    min(b, t_r + Wr); likę — C fazėje iki savo ribos. Paruoštas Straggler'is laikomas ne ilgiau kaip Wr.
//  * C taisyklė (kaip B): kai bent vienas WaitC vienetas paruoštas (anksčiausias — t_r), C laukia neparuoštų WaitC vienetų ir
//    prisijungsiančių po pernešimo ne ilgiau kaip min(c, t_r + Wr); paruoštas WaitC vienetas laikomas ne ilgiau kaip Wr
//    (anksčiau — iki pačios C ribos dėl bet kurio neparuošto ar prisijungiančio). Kol indeksui liko process kvietimų
//    (≤ 3 iš viso), likę WaitC / prisijungę vėliau paleidžiami kitu C kvietimu. Kai kvietimų nebeliko (A + B + ankstyvas
//    C), vėliau paruošti ar prisijungę iki c(i) vienetai NEIŠMETAMI: jie apdorojami paskutiniu batch'u ties c(i)
//    (4-as kvietimas, skaitiklis cut_batches). Niekas, kas paruošta iki savo vieneto paskutinės ribos, nėra spraga.
//    Legacy (viena riba) prisijungę po A — C fazėje iki tos pačios ribos.
//  * Kiekvienas dalyvaujantis indeksas baigiasi: commit / spraga pagal politiką / skaitiklis su priežastimi
//    (*_skips, gap_left, commit_errors). Lap'as — atšaukimas + spraga pagal politiką (lapped++).
#include "engine/engine.hpp"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

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
        case Outcome::GapStartup: return "GapStartup";
        case Outcome::GapClock: return "GapClock";
        case Outcome::Lapped: return "Lapped";
        case Outcome::Skipped: return "Skipped";
        case Outcome::OutputError: return "OutputError";
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
        InState seen = InState::Waiting;  // paskutinė pranešta būsena (įvykiai operatoriui)
        bool done = false, ok = false;  // einamojo vieneto indekso būsena
        std::int64_t ready_at = 0;
        Index observe = kNoIndex;  // vienetas buvo Blocked dėl kito įėjimo — šio grain'as patikrinamas kitame tick'e
    };
    struct Out {
        OutputPort* port;
        GapState gs;
        std::uint32_t hub;
        std::string name;            // pranešimui apie ilgalaikę klaidą (alokuojama tik add_unit)
        std::uint32_t unit = 0;
        bool flush = false;
        std::uint32_t fail_run = 0;  // open / commit klaidų iš eilės
    };
    struct Unit {
        std::uint32_t stride = 1, residue = 0;
        std::uint32_t in0 = 0, nin = 0, out0 = 0, nout = 0, g0 = 0, ng = 0, hub = 0;
        Index cur = kNoIndex, join = kNoIndex;
        Stage stage = Stage::None;
        bool failed = false, no_content = false, ever_failed = false;
        // busy — vienetas šiuo metu batch'e (process) arba jo end_unit kvietime: fail_unit tada jo NEužbaigia (užbaigs
        // kviečiantysis — lygiai vieną kartą); slots_open — jo išėjimų slot'ai atidaryti (fail_unit juos atšaukia prieš release).
        bool busy = false, slots_open = false;
        std::int64_t ready_at = 0;
        // Atkūrimas (D8): kitas bandymas (monotoninis laikas), dabartinis atidėjimas, paskutinio atsigavimo momentas.
        std::int64_t retry_at = 0, recovered_at = INT64_MIN;
        double backoff_s = 0;
        Index fail_index = kNoIndex;  // pirmo gedimo indeksas (tick'o, kuriame sugedo)
    };
    struct Rec {
        Index idx = kNoIndex;
        Cuts cuts{};
        bool a_done = false, b_done = false, c_done = false, ended = true, opening = false;
        std::uint32_t unresolved = 0, joiners = 0;
        std::uint8_t calls = 0;
        std::int64_t held = 0;
        std::int64_t b_hold = INT64_MAX;  // B paleidimo riba, kai jau yra paruoštų Straggler'ių (min(b, t_r + Wr))
        std::int64_t c_hold = INT64_MAX;  // C paleidimo riba, kai jau yra paruoštų WaitC vienetų (min(c, t_r + Wr))
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
    // Atkūrimo gija (retry_async): vieneto būsena rst[k] — 0 laukia, 1 paprašyta, 2 vykdoma, 3 pavyko, 4 nepavyko.
    enum : int { kRetryIdle = 0, kRetryAsked = 1, kRetryRunning = 2, kRetryOk = 3, kRetryFail = 4 };
    std::unique_ptr<std::atomic<int>[]> rst;
    std::thread rworker;
    std::mutex rm;
    std::condition_variable rcv;
    bool rquit = false;
    bool step_in_tick = false;
    bool skip_loss = false;  // on_skip metu: praleidimas — praradimas (TickContext::skip_is_loss)
    std::int64_t last_step_mono = INT64_MIN;
    std::uint32_t pending = 0;
    std::uint64_t changes = 0;
    Index ctx_index = kNoIndex, newest = kNoIndex;
    std::int64_t ctx_held = 0;
    bool ended = false, inputs_checked = false;
    std::uint32_t failed_n = 0;
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
    // Laukiamas įėjimas: Required ir Wanted (Wanted — iki vieneto paskutinės ribos), kol Live ir neišspręstas.
    static bool waitable(const In& in) { return !in.done && in.spec.need != Need::Optional && in.st.live(); }

    // Įvykis operatoriui (be alokacijų: tekstas — steko buferis).
    void emit(EventKind k, std::uint32_t id, Index idx, std::int64_t value, const char* text) {
        hub.event(k, id, clk.now_tai(), idx, value, text);
    }

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
            if (in.spec.need == Need::Wanted && (in.done || final_cut)) continue;  // paskutinė riba — be jo
            return false;
        }
        return true;
    }
    // Šviežias — nė vienas neparuoštas laukiamas (Required / Wanted) įėjimas dar nepraleido paskutinės ribos (miss_run == 0).
    bool fresh(const Unit& u) const {
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            const In& in = ins[u.in0 + k];
            if (!in.ok && !in.done && in.spec.need != Need::Optional && in.st.miss_run() >= 1) return false;
        }
        return true;
    }

    static std::int64_t c_deadline(const Rec& r) { return r.cuts.has_c ? r.cuts.c : (r.cuts.has_b ? r.cuts.b : r.cuts.a); }
    std::int64_t deadline(const Unit& u) {
        const Rec& r = rec(u.cur);
        if (!r.a_done) return r.cuts.a;
        if (u.stage == Stage::WaitC) return std::min(c_deadline(r), r.c_hold);
        return r.cuts.has_b ? std::min(r.cuts.b, r.b_hold) : r.cuts.a;
    }

    // Laukiamas vieneto paruošimo laikas: vėliausias T(j) + ewma tarp neparuoštų laukiamų (Required / Wanted) įėjimų;
    // INT64_MIN — nežinoma.
    std::int64_t expected(const Unit& u) const {
        std::int64_t e = INT64_MIN;
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            const In& in = ins[u.in0 + k];
            if (in.ok || in.done || in.spec.need == Need::Optional) continue;
            if (!in.ewma_set) return INT64_MIN;
            e = std::max(e, T(jidx(u, in)) + in.ewma);
        }
        return e;
    }

    // --- įėjimų skaitymas ------------------------------------------------------------------------------------------
    void mark_ok(Unit& u, In& in, Index j) {
        in.done = in.ok = true;
        const std::int64_t n = now();
        const auto lw = static_cast<std::int64_t>(in.spec.port->last_write_tai());
        // Grain'as j commit'intas ne vėliau kaip head'as (last_write); stebėjimo momentas gali būti vėlesnis (ašis).
        in.ready_at = lw > 0 ? std::min(n, lw) : n;
        u.ready_at = std::max(u.ready_at, in.ready_at);
        const std::int64_t lat = (lw > 0 ? lw : n) - T(j);
        in.ewma = in.ewma_set ? in.ewma + (lat - in.ewma) / 4 : lat;
        in.ewma_set = true;
        hub.in_arrival(in.hub).add(static_cast<std::uint64_t>(std::max<std::int64_t>(0, lat)));
        ++changes;
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
        ++changes;
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
        publish_stats();
    }

    // D6: tick'o pabaigoje, po commit'ų ir už laukimų ribų; kopija į paruoštą buferį, I/O — kitoje gijoje.
    void publish_stats() {
        const std::uint64_t t = clk.now_tai();
        if (hub.due(t)) {
            for (const In& in : ins) hub.set_in_state(in.hub, in.st.state());
            for (const Unit& u : units) {
                hub.set_unit_failed(u.hub, u.failed);
                for (std::uint32_t m = 0; m < u.nout; ++m) hub.set_out_enabled(outs[u.out0 + m].hub, !u.failed);
            }
        }
        hub.end_tick(t);
    }

    void resolve_unit(Unit& u) {
        Rec& r = rec(u.cur);
        if (r.idx == u.cur && r.unresolved > 0) --r.unresolved;
        u.stage = Stage::None;
        --pending;
        ++changes;
        if (u.join != kNoIndex) {
            const Index j = u.join;
            u.join = kNoIndex;
            Rec& rj = rec(j);
            if (rj.idx == j) {
                if (rj.joiners > 0) --rj.joiners;
                if (!u.failed && !ended) {
                    start_unit(u, j, rj, true);
                } else {
                    if (rj.unresolved > 0) --rj.unresolved;
                    maybe_end(rj);
                }
            }
        }
        maybe_end(r);
    }

    // Išėjimų spraga indeksui c pagal politiką. lapped — atšaukto (lap) grain'o vieta.
    void gap_outputs(Unit& u, Index c, bool startup, bool lapped) {
        for (std::uint32_t m = 0; m < u.nout; ++m) {
            Out& o = outs[u.out0 + m];
            OutCounters& oc = hub.out(o.hub);
            if (lapped) ++oc.lapped;
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
            if (lapped) continue;
            if (startup) ++oc.gap_startup;
            else if (a == GapAction::Invalid) ++oc.gap_invalid;
            else if (a == GapAction::Repeat) ++oc.gap_frozen;
            else ++oc.gap_black;
        }
    }

    void gap_unit(Unit& u, Outcome why, std::uint8_t phase) {
        const Index c = u.cur;
        bool startup = false;
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            const In& in = ins[u.in0 + k];
            if (in.spec.need == Need::Required && !in.ever) startup = true;
        }
        gap_outputs(u, c, startup, false);
        UnitCounters& uc = hub.unit(u.hub);
        ++(startup ? uc.startup : uc.gaps);  // paleidimo spragos — ne praradimas
        if (!startup && why == Outcome::GapClock) ++uc.gaps_clock;
        if (!startup && why == Outcome::GapNoContent) ++uc.gaps_no_content;
        if (why == Outcome::GapBlocked)  // gyvi broliai — stebimi kitame tick'e (skaitikliai ir būsena, ne turinys)
            for (std::uint32_t k = 0; k < u.nin; ++k)
                if (ins[u.in0 + k].st.live() && ins[u.in0 + k].spec.need != Need::Optional) ins[u.in0 + k].observe = jidx(u, ins[u.in0 + k]);
        if (why == Outcome::GapClock) ++hub.engine().step_gaps;
        const UnitTick ut{uid(u), c, phase, {}, {}};
        ctx_index = c;
        u.busy = true;
        fn->end_unit(ut, startup ? Outcome::GapStartup : why, ctx);  // paleidimo spraga — atskiras rezultatas (V18)
        u.busy = false;
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

    void start_unit(Unit& u, Index c, Rec& r, bool joiner) {
        u.cur = c;
        u.stage = Stage::WaitA;
        ++pending;
        ++changes;
        if (!joiner) ++r.unresolved;  // prisijungiantis jau įskaičiuotas atidarant
        u.ready_at = 0;
        for (std::uint32_t n = 0; n < u.ng; ++n) grefs[u.g0 + n] = GrainRef{};
        bool blocked = false, on_time = true;
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            In& in = ins[u.in0 + k];
            in.done = in.ok = false;
            in.ready_at = 0;
            if (!in.st.live()) {
                if (in.spec.need == Need::Required) blocked = true;
                else in.done = true;  // Wanted / Optional be rašytojo — tęsiama be jo
                continue;
            }
            if (in.st.late() && in.spec.need != Need::Optional) on_time = false;  // klasę lemia Required ir Wanted
        }
        if (u.nin == 0) {
            if (u.no_content) return gap_unit(u, Outcome::GapNoContent, 0);
            u.ready_at = now();
        }
        if (blocked) return gap_unit(u, Outcome::GapBlocked, 0);  // T3 / T4: spraga atidarymo metu
        const std::int64_t n = now();
        if (!r.a_done) {
            u.stage = on_time ? Stage::WaitA : Stage::WaitB;
        } else if (r.cuts.has_b && !r.b_done && n < r.cuts.b && fresh(u)) {
            u.stage = Stage::WaitB;  // prisijungė po pernešimo (T2)
        } else if (n < c_deadline(r)) {  // C (ir po paskutinio C kvietimo — paskutiniu batch'u ties c); legacy — ta pati riba
            u.stage = Stage::WaitC;
        } else {
            ++hub.engine().carry_overrun;  // T2: c ribos jau praėjo
            gap_unit(u, Outcome::GapLate, 0);
        }
    }

    // --- paleidimai ----------------------------------------------------------------------------------------------
    Out* broken = nullptr;  // launch(): išėjimas, pasiekęs out_fail_fatal klaidų iš eilės (vienetas sugenda po end_unit)

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
                count_inputs(u, r);
                gap_unit(u, Outcome::GapBlocked, phase);
                continue;
            }
            OpenResult opened = OpenResult::Ok;
            for (std::uint32_t m = 0; m < u.nout; ++m) {
                void* s = nullptr;
                Out& o = outs[u.out0 + m];
                opened = o.port->open(idx, s);
                if (opened == OpenResult::Ok) {
                    slots[u.out0 + m] = s;
                    continue;
                }
                for (std::uint32_t p = 0; p < m; ++p) outs[u.out0 + p].port->cancel();
                if (opened == OpenResult::AlreadyCommitted) {
                    ++hub.out(o.hub).already_committed;  // laikrodis atgal (M11): indeksas jau parašytas
                } else {
                    ++hub.out(o.hub).commit_errors;
                    if (out_failed(o)) broken = &o;
                }
                break;
            }
            if (opened != OpenResult::Ok) {
                UnitCounters& uc = hub.unit(u.hub);
                ++uc.gaps;
                if (opened == OpenResult::AlreadyCommitted) ++uc.gaps_clock;
                const UnitTick ut{uid(u), idx, phase, {}, {}};
                u.busy = true;
                fn->end_unit(ut, opened == OpenResult::AlreadyCommitted ? Outcome::Skipped : Outcome::OutputError, ctx);
                u.busy = false;
                resolve_unit(u);
                if (broken != nullptr) fail_broken(u, *broken, "open");
                broken = nullptr;
                continue;
            }
            latest_ready = std::max(latest_ready, u.ready_at);
            u.busy = true;
            u.slots_open = u.nout != 0;
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
        ctx_held = phase == 0 ? r.held : 0;
        const bool ok = fn->process(Batch{idx, phase, std::span<const UnitTick>(batch.data(), batch.size())}, ctx);
        ++r.calls;
        ++hub.engine().process_calls;
        ++changes;
        if (!ok || ended) {
            for (const UnitTick& ut : batch) {
                Unit& u = units[ut.unit];
                u.busy = false;
                if (u.stage == Stage::None) continue;  // jau užbaigtas (fail_unit iš process: išėjimai atšaukti)
                if (u.slots_open)
                    for (std::uint32_t m = 0; m < u.nout; ++m) outs[u.out0 + m].port->cancel();
                u.slots_open = false;
                u.stage = Stage::None;
                --pending;
            }
            end_fatal(EndReason::Fatal, 2, "TickFunction::process grąžino false");
            return;
        }
        // Wanted įėjimai, nesulaukti paleidimo metu (paskutinė riba): vienas neblokuojantis bandymas po apdorojimo (dar prieš
        // apskaitą ir end_unit). Grain'as, atėjęs kol funkcija apdorojo, pasiekia end_unit per tą patį UnitTick::in ir
        // skaičiuojamas ok / late, ne lost. Laukimo nėra.
        for (const UnitTick& ut : batch) {
            Unit& u = units[ut.unit];
            for (std::uint32_t k = 0; k < u.nin; ++k) {
                In& in = ins[u.in0 + k];
                if (in.spec.need == Need::Wanted && !in.done && in.st.live()) try_resolve(u, in);
            }
        }
        const Index now_idx = due_index(clk.now_tai(), cfg.cuts);
        const std::int64_t done_at = now();
        for (const UnitTick& ut : batch) {
            Unit& u = units[ut.unit];
            count_inputs(u, r);
            if (u.failed) {  // fail_unit iš process(): išėjimai jau atšaukti ir atlaisvinti — tik užbaigiamas (vieną kartą)
                ++hub.unit(u.hub).gaps;
                fn->end_unit(ut, Outcome::OutputError, ctx);
                u.busy = false;
                resolve_unit(u);
                continue;
            }
            bool lap = false;  // T12
            for (std::uint32_t k = 0; k < u.nin && !lap; ++k) {
                const In& in = ins[u.in0 + k];
                if (!in.ok) continue;
                const Index j = jidx(u, in);
                for (std::uint32_t g = 0; g < in.spec.grains; ++g)
                    if (in.spec.port->lapped(j - (in.spec.grains - 1) + g, now_idx)) lap = true;
            }
            UnitCounters& uc = hub.unit(u.hub);
            Outcome oc = Outcome::Ready;
            if (lap) {  // atšaukiama ir užpildoma pagal politiką — indeksas neliks be įrašo
                for (std::uint32_t m = 0; m < u.nout; ++m) outs[u.out0 + m].port->cancel();
                u.slots_open = false;
                gap_outputs(u, idx, false, true);
                ++uc.lapped;
                oc = Outcome::Lapped;
            } else {
                bool all_ok = true;
                broken = nullptr;
                for (std::uint32_t m = 0; m < u.nout; ++m) {
                    Out& o = outs[u.out0 + m];
                    OutCounters& c = hub.out(o.hub);
                    if (o.port->commit()) {
                        ++(phase == 0 ? c.ok : c.late);
                        o.gs.on_content(idx, T(idx));
                        if (o.fail_run != 0) emit(EventKind::OutputRecovered, o.hub, idx, 0, "išėjimas vėl commit'ina");
                        o.fail_run = 0;
                        hub.out_commit(o.hub).add(static_cast<std::uint64_t>(std::max<std::int64_t>(0, done_at - T(idx))));
                    } else {
                        ++c.commit_errors;  // praradimas (any_loss); indeksas lieka su priežastimi
                        if (out_failed(o) && broken == nullptr) broken = &o;
                        all_ok = false;
                    }
                }
                u.slots_open = false;
                if (all_ok) {
                    ++(phase == 0 ? uc.ok : uc.late);
                } else {
                    ++uc.gaps;
                    oc = Outcome::OutputError;
                }
            }
            fn->end_unit(ut, oc, ctx);  // busy: fail_unit iš end_unit vieneto neužbaigia antrą kartą
            u.busy = false;
            resolve_unit(u);
            if (broken != nullptr) fail_broken(u, *broken, "commit");
            broken = nullptr;
        }
        // Po apdorojimo: B fazė baigėsi po kito tick'o atidarymo
        if (phase == 1 && done_at > cuts_for(idx + 1, cfg.cuts).open) ++hub.engine().guard_overrun;
    }

    // Išėjimo open / commit klaida: true — cfg.out_fail_fatal kartų iš eilės (ilgalaikė; vienetas sugenda — fail_broken).
    bool out_failed(Out& o) {
        if (++o.fail_run == 1) emit(EventKind::OutputError, o.hub, ctx_index, 0, "išėjimo open / commit klaida");
        return cfg.out_fail_fatal != 0 && o.fail_run >= cfg.out_fail_fatal && !ended;
    }
    void fail_broken(Unit& u, const Out& o, const char* what) {
        fail_unit(u, "išėjimas " + o.name + ": " + std::to_string(o.fail_run) + " " + what + " klaidos iš eilės (open / commit)");
    }

    // K17 / K-T11: vienetas sustabdomas VIENAS — jo išėjimai atlaisvinami (žemupys mato flow dingusį), kiti tęsia. Kai sugenda
    // visi vienetai — fatalu (exit 2) su paskutinio priežastimi. Vykdomas vienetas (stage != None) baigiamas OutputError.
    void fail_unit(Unit& u, std::string why) {
        if (u.failed) return;
        u.failed = u.ever_failed = true;
        if (u.fail_index == kNoIndex) u.fail_index = ctx_index;
        ++failed_n;
        ++hub.unit(u.hub).failures;
        if (res.failed_why.empty()) res.failed_why = why;
        // Atkūrimo atidėjimas: pirmas — retry_min_s; sugedęs netrukus (< 60 s) po atsigavimo — dvigubai ilgesnis nei buvęs.
        const std::int64_t mono = clk.now_mono_raw();
        if (cfg.retry_min_s > 0) {
            const bool flap = u.recovered_at != INT64_MIN && mono - u.recovered_at < 60'000'000'000LL && u.backoff_s > 0;
            u.backoff_s = flap ? std::min(cfg.retry_max_s, 2 * u.backoff_s) : cfg.retry_min_s;
            u.retry_at = mono + static_cast<std::int64_t>(u.backoff_s * 1e9);
        }
        char t[160];
        std::snprintf(t, sizeof t, "%s; bandymas po %.0f s", why.c_str(), cfg.retry_min_s > 0 ? u.backoff_s : 0.0);
        emit(EventKind::UnitFailed, u.hub, u.cur, static_cast<std::int64_t>(u.backoff_s * 1000), t);
        if (u.slots_open) {  // iš process(): atidaryti slot'ai atšaukiami — rašytojas neatlaisvinamas su atviru grain'u
            for (std::uint32_t m = 0; m < u.nout; ++m) outs[u.out0 + m].port->cancel();
            u.slots_open = false;
        }
        for (std::uint32_t m = 0; m < u.nout; ++m) outs[u.out0 + m].port->release();
        // Vykdomas vienetas baigiamas OutputError — nebent jis dabar batch'e ar savo end_unit kvietime (busy): tada jį užbaigia
        // kviečiantysis, lygiai vieną kartą (anksčiau — du kartai: pending persivyniodavo, --duration išleidimas pakibdavo).
        if (u.stage != Stage::None && !u.busy) {
            ++hub.unit(u.hub).gaps;
            const UnitTick ut{uid(u), u.cur, 0, {}, {}};
            u.busy = true;
            fn->end_unit(ut, Outcome::OutputError, ctx);
            u.busy = false;
            resolve_unit(u);
        }
        if (failed_n == units.size()) end_fatal(EndReason::Fatal, 2, "visi vienetai sugedo; " + why);
    }

    // --- atkūrimas (D8) -------------------------------------------------------------------------------------------
    bool reopen_unit(const Unit& u) {
        for (std::uint32_t m = 0; m < u.nout; ++m)
            if (!outs[u.out0 + m].port->reopen()) return false;
        return true;
    }
    void retry_worker() {
        std::unique_lock<std::mutex> lk(rm);
        while (!rquit) {
            rcv.wait_for(lk, std::chrono::milliseconds(50));
            for (std::size_t k = 0; k < units.size() && !rquit; ++k) {
                int want = kRetryAsked;
                if (!rst[k].compare_exchange_strong(want, kRetryRunning)) continue;
                lk.unlock();
                const bool ok = reopen_unit(units[k]);
                rst[k].store(ok ? kRetryOk : kRetryFail, std::memory_order_release);
                lk.lock();
            }
        }
    }
    // Tick'o atidarymas: sugedusiems vienetams — atkūrimo bandymas (laikas atėjo), rezultatas (atsigavo / atidėjimas ×2).
    void service_retries() {
        if (cfg.retry_min_s <= 0 || failed_n == 0) return;
        const std::int64_t mono = clk.now_mono_raw();
        for (std::size_t k = 0; k < units.size(); ++k) {
            Unit& u = units[k];
            if (!u.failed) continue;
            int s = rst[k].load(std::memory_order_acquire);
            if (s == kRetryIdle && mono >= u.retry_at) {
                if (cfg.retry_async) {
                    // gija — tik prireikus (pirmas gedimas): įprastame darbe papildomų gijų nėra (bibliotekos ritmo vykdytojas —
                    // viena gija visiems rašytojams); kūrimas alokuoja — gedimo kelyje, ne kiekviename tick'e
                    if (!rworker.joinable()) rworker = std::thread([this] { retry_worker(); });
                    rst[k].store(kRetryAsked, std::memory_order_release);
                    rcv.notify_one();
                    continue;
                }
                s = reopen_unit(u) ? kRetryOk : kRetryFail;
            }
            if (s == kRetryOk) {
                u.failed = false;
                --failed_n;
                u.recovered_at = mono;
                ++hub.unit(u.hub).recoveries;
                ++res.unit_recoveries;
                for (std::uint32_t m = 0; m < u.nout; ++m) {
                    Out& o = outs[u.out0 + m];
                    o.fail_run = 0;
                    o.flush = false;
                    const GapState fresh_gs{};
                    o.gs = fresh_gs;  // ankstesnis turinys atkurtame ring'e nebegalioja (freeze — nuo naujo)
                    o.gs.policy = cfg.on_loss;
                    o.gs.hold_ns = cfg.hold_ns;
                }
                emit(EventKind::UnitRecovered, u.hub, kNoIndex, 0, "išėjimas sukurtas iš naujo, vienetas vėl dirba");
                rst[k].store(kRetryIdle, std::memory_order_release);
            } else if (s == kRetryFail) {
                ++hub.unit(u.hub).retries;
                ++res.unit_retries;
                u.backoff_s = std::min(cfg.retry_max_s, 2 * std::max(u.backoff_s, cfg.retry_min_s));
                u.retry_at = mono + static_cast<std::int64_t>(u.backoff_s * 1e9);
                char t[96];
                std::snprintf(t, sizeof t, "atkurti nepavyko; kitas bandymas po %.0f s", u.backoff_s);
                emit(EventKind::UnitRetryFailed, u.hub, kNoIndex, static_cast<std::int64_t>(u.backoff_s * 1000), t);
                rst[k].store(kRetryIdle, std::memory_order_release);
            }
        }
    }

    // Įėjimų būsenų pokyčiai operatoriui (kartą per tick'ą; pokytis įvyksta bet kur variklyje — pranešama ≤ 1 periodu).
    void report_inputs() {
        for (In& in : ins) {
            const InState s = in.st.state();
            if (s == in.seen) continue;
            const InState was = in.seen;
            in.seen = s;
            switch (s) {
                case InState::Live:
                    if (was == InState::Stalled) emit(EventKind::InputResumed, in.hub, kNoIndex, 0, "rašytojas vėl rašo");
                    else if (was == InState::Detached) emit(EventKind::InputReattached, in.hub, kNoIndex, 0, "prisijungta iš naujo");
                    else emit(EventKind::InputAttached, in.hub, kNoIndex, 0, "prisijungta");
                    break;
                case InState::Stalled: emit(EventKind::InputStalled, in.hub, kNoIndex, 0, "užstrigo: grain'ų nėra (rašytojas gyvas)"); break;
                case InState::Detached: emit(EventKind::InputDetached, in.hub, kNoIndex, 0, "atsijungta: rašytojo nėra arba flow perkurtas"); break;
                case InState::Mismatch: emit(EventKind::FormatChange, in.hub, kNoIndex, 0, "grįžo kitu formatu"); break;
                case InState::Waiting:
                case InState::Disabled: break;
            }
        }
    }

    void count_inputs(Unit& u, const Rec& r) {
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            In& in = ins[u.in0 + k];
            InCounters& c = hub.in(in.hub);
            if (in.ok) {
                in.st.on_ready(c, in.ready_at <= r.cuts.a);
            } else if (!in.done && in.spec.need == Need::Wanted && in.st.live()) {
                if (in.st.on_missed(c)) in.st.note_head(in.spec.port->head());
            }
        }
    }

    template <class F>
    void for_rec(const Rec& r, F f) {
        for (Unit& u : units)
            if (u.stage != Stage::None && u.cur == r.idx) f(u);
    }

    // T6: įėjimas, kurio A laukė (Quorum arba, be kvorumo, šviežias Straggler), ties A neparuoštas.
    bool check_alive_at_a(Unit& u) {
        bool dead = false;
        for (std::uint32_t k = 0; k < u.nin; ++k) {
            In& in = ins[u.in0 + k];
            if (in.ok || in.done || !in.st.live() || in.spec.need == Need::Optional) continue;
            const bool alive = in.spec.port->writer_alive();
            in.st.on_unready_at_a(alive, hub.in(in.hub));
            if (!alive) {
                detach(in);
                if (in.spec.need == Need::Required) dead = true;
                else in.done = true;
            }
        }
        return dead;
    }

    void fire_a(Rec& r) {
        if (r.a_done || ended) return;
        bool any_q = false, any = false, wait = false;
        for_rec(r, [&](const Unit& u) {
            any = true;
            if (u.stage == Stage::WaitA) {
                any_q = true;
                if (!ready(u, false)) wait = true;
            }
        });
        const bool no_quorum = !any_q;
        // Be kvorumo laukiama tik šviežių Straggler'ių, kurių laukiamas paruošimo laikas ≤ a(i) (nežinomas — laukiama), kaip
        // B taisyklėje. Nuolat vėluojantis vienetas (pvz. vieno srauto garsas ties T + 20 ms) kitų, po vieno bendro vėlaus
        // kadro tapusių Straggler'iais, nebelaiko iki a(i), kol jų Late nusiims (T2: +1,2 ms 8 kadrus iš eilės).
        if (no_quorum)
            for_rec(r, [&](const Unit& u) {
                if (u.stage == Stage::WaitB && fresh(u) && !ready(u, false) && expected(u) <= r.cuts.a) wait = true;
            });
        if (!any && r.joiners > 0) wait = true;  // tik prisijungsiantys — A nepaleidžiama tuščia
        if (wait && now() < r.cuts.a) return;
        r.a_done = true;
        const bool final_cut = !r.cuts.has_b;
        launch(r, 0, [&](const Unit& u) { return ready(u, final_cut); });
        for (Unit& u : units) {
            if (u.stage == Stage::None || u.cur != r.idx) continue;
            const bool gated = u.stage == Stage::WaitA || (no_quorum && u.stage == Stage::WaitB && fresh(u));
            if (gated && check_alive_at_a(u)) {
                gap_unit(u, Outcome::GapLate, 0);
                continue;
            }
            if (final_cut) finalize_miss(u, 0);
            else u.stage = fresh(u) ? Stage::WaitB : Stage::WaitC;
        }
        if (final_cut) r.b_done = true;  // be B: prisijungę po A — C fazėje (iki tos pačios ribos)
    }

    void fire_b(Rec& r) {
        if (!r.a_done || !r.cuts.has_b || r.b_done || ended) return;
        bool any = false, pending = false;
        std::int64_t t_r = INT64_MAX;  // anksčiausiai paruošto Straggler'io paruošimo laikas
        for_rec(r, [&](const Unit& u) {
            if (u.stage != Stage::WaitB) return;
            any = true;
            if (ready(u, false)) t_r = std::min(t_r, u.ready_at);
            else pending = true;
        });
        if (!any) {  // šviežių nėra — B tuščia; pakibę / pernešti (WaitC) tvarkomi C
            r.b_done = true;
            return;
        }
        const std::int64_t n = now();
        const bool at_cut = n >= r.cuts.b;
        bool go = at_cut || !pending;
        if (!go && t_r != INT64_MAX) {
            r.b_hold = std::min(r.cuts.b, t_r + cfg.cuts.ready_wait_ns);
            bool wait = false;
            for_rec(r, [&](const Unit& u) {
                if (u.stage == Stage::WaitB && !ready(u, false) && expected(u) <= r.b_hold) wait = true;
            });
            go = !wait || n >= r.b_hold;
        }
        if (!go) return;
        r.b_done = true;
        const bool final_b = at_cut && !r.cuts.has_c;
        launch(r, 1, [&](const Unit& u) { return ready(u, final_b); });
        for (Unit& u : units) {
            if (u.stage == Stage::None || u.cur != r.idx) continue;
            if (!at_cut) {  // B paleista anksčiau — nesulaukti iki savo ribos C fazėje (has_c ? c : b)
                if (u.stage == Stage::WaitB) u.stage = Stage::WaitC;
            } else if (r.cuts.has_c) {
                if (u.stage == Stage::WaitB) u.stage = Stage::WaitC;  // T8: pernešimas
            } else {
                finalize_miss(u, 1);
            }
        }
    }

    void fire_c(Rec& r) {
        if (!r.b_done || ended) return;
        bool any = false, all = true;
        std::int64_t t_r = INT64_MAX;  // anksčiausiai paruošto WaitC vieneto paruošimo laikas
        for_rec(r, [&](const Unit& u) {
            if (u.stage != Stage::WaitC) return;
            any = true;
            if (ready(u, false)) t_r = std::min(t_r, u.ready_at);
            else all = false;
        });
        if (!any) return;
        const std::int64_t n = now();
        const bool at_cut = n >= c_deadline(r);
        if (!r.c_done && !at_cut) {
            if (t_r == INT64_MAX) return;  // nė vieno paruošto — laukiama iki C ribos
            if (!all || r.joiners > 0) {   // laukiama tik tų, kurie tikėtini per Wr nuo pirmo paruošto (kaip B)
                r.c_hold = std::min(c_deadline(r), t_r + cfg.cuts.ready_wait_ns);
                bool wait = r.joiners > 0;  // prisijungiančių paruošimo laikas nežinomas — laukiama iki c_hold
                for_rec(r, [&](const Unit& u) {
                    if (u.stage == Stage::WaitC && !ready(u, false) && expected(u) <= r.c_hold) wait = true;
                });
                if (wait && n < r.c_hold) return;
            }
        }
        if (!r.c_done) {
            r.c_hold = INT64_MAX;
            launch(r, 2, [&](const Unit& u) { return u.stage == Stage::WaitC && ready(u, at_cut); });
            if (at_cut || r.calls >= 3) r.c_done = true;  // ≤ 3 process kvietimai indeksui iki c(i)
        } else if (at_cut) {
            // Paskutinis batch'as ties c(i): po trečio kvietimo paruošti / prisijungę iki savo ribos — ne spraga (4-as kvietimas)
            const std::uint8_t before = r.calls;
            launch(r, 2, [&](const Unit& u) { return u.stage == Stage::WaitC && ready(u, true); });
            if (r.calls != before) ++hub.engine().cut_batches;
        }
        if (!at_cut) return;
        for (Unit& u : units)
            if (u.stage == Stage::WaitC && u.cur == r.idx) finalize_miss(u, 2);
    }

    void fire_due() {
        Rec* cur = nullptr;
        Rec* old = nullptr;
        for (Rec& r : recs) {
            if (r.idx == kNoIndex) continue;
            if (r.idx == newest) cur = &r;
            else old = &r;
        }
        if (cur != nullptr) fire_a(*cur);  // sveikas naujausio tick'o A — pirmas
        if (old != nullptr) {
            fire_a(*old);
            fire_b(*old);
            fire_c(*old);
        }
        if (cur != nullptr) {
            fire_b(*cur);
            fire_c(*cur);
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
        if (d < 0) {
            ++e.clock_backward;
            e.clock_back_ns += static_cast<std::uint64_t>(-d);
        }
        step_in_tick = true;
        last_step_mono = clk.now_mono_raw();
        char t[64];
        std::snprintf(t, sizeof t, "laikrodžio žingsnis %+.3f ms", static_cast<double>(d) / 1e6);
        emit(EventKind::ClockStep, 0, kNoIndex, d, t);
        // Atviri tick'ai uždaromi: priežastis — laikrodis. Antras ratas — prisijungę po pernešimo (jų indeksas
        // irgi turi baigtis spraga, ne tyla).
        for (int pass = 0; pass < 2; ++pass)
            for (Unit& u : units)
                if (u.stage != Stage::None) gap_unit(u, Outcome::GapClock, 0);
        fn->on_clock_step(d, ctx);
    }

    // --- laukimas (D3) --------------------------------------------------------------------------------------------
    void pump(std::int64_t until, bool until_idle, Index open_idx) {
        for (;;) {
            std::uint64_t c0 = 0;
            do {
                c0 = changes;
                poll_all();
                fire_due();
            } while (changes != c0 && !ended);
            flush();
            if (ended || stop.load(std::memory_order_relaxed)) return;
            if (until_idle && pending == 0) return;
            const std::int64_t n = now();
            if (n >= until) return;
            std::int64_t ev = until;
            Index seen = kNoIndex;
            bool multi = false;
            for (Unit& u : units) {
                if (u.stage == Stage::None) continue;
                ev = std::min(ev, deadline(u));
                if (seen == kNoIndex) seen = u.cur;
                else if (u.cur != seen) multi = true;
            }
            const std::int64_t pivot = choose_pivot();
            if (ev <= n) {  // saugiklis: nėra būsimo įvykio — miegama 100 µs, ne sukamasi (skaičiuojama; neturėtų įvykti)
                ++hub.engine().spin_guard;
                ev = n + 100'000;
            }
            if (pivot >= 0 && cfg.wait == WaitMode::Pivot) {
                In& in = ins[static_cast<std::size_t>(pivot)];
                Unit& u = units[in.unit];
                std::int64_t to = ev - n;
                if (multi || pivot_slice) to = std::min(to, kSliceNs);
                GrainRef g;
                const Probe p = in.spec.port->wait(jidx(u, in), static_cast<std::uint64_t>(to), g);
                if (stop.load(std::memory_order_relaxed)) return;  // T13 — prieš bet kokį sprendimą
                check_step();
                if (u.stage != Stage::None && !in.done) {
                    if (p == Probe::Ready) try_resolve(u, in);
                    else if (p != Probe::NotYet) apply_fail(u, in, p);
                }
                continue;
            }
            std::int64_t t = ev;
            if (cfg.wait == WaitMode::Poll && pivot >= 0) t = std::min(t, n + kPollNs);
            if (open_idx != kNoIndex && t == until && cfg.open_wait) {
                // Tick'o atidarymas — tik per kabliuką (bibliotekos PacingHooks::wait)
                cfg.open_wait(open_idx, cfg.cuts.expect_ns + cfg.cuts.read_offset_ns, stop);
                if (!stop.load(std::memory_order_relaxed) && now() < until) {  // kabliukas grįžo per anksti — sutarties pažeidimas
                    ++hub.engine().open_wait_early;
                    clk.sleep_until(static_cast<std::uint64_t>(until), stop);
                }
            } else {
                clk.sleep_until(static_cast<std::uint64_t>(t), stop);
            }
            if (stop.load(std::memory_order_relaxed)) return;
            check_step();
        }
    }

    // Ašis: naujausio tick'o A aibės (Quorum; be kvorumo — šviežių Straggler'ių) įėjimas su didžiausiu ewma_late
    // (lygybė — didžiausias įėjimo numeris). Jei A aibės nėra — anksčiausios ribos grupė.
    std::int64_t choose_pivot() {
        std::int64_t pivot = -1;
        pivot_slice = false;
        auto consider = [&](const Unit& u) {
            for (std::uint32_t k = 0; k < u.nin; ++k) {
                const auto gk = static_cast<std::int64_t>(u.in0 + k);
                if (!waitable(ins[static_cast<std::size_t>(gk)])) continue;
                if (pivot < 0 || ins[static_cast<std::size_t>(gk)].ewma >= ins[static_cast<std::size_t>(pivot)].ewma)
                    pivot = gk;
            }
        };
        if (newest != kNoIndex) {
            Rec& r = rec(newest);
            if (r.idx == newest && !r.a_done) {
                bool any_q = false;
                for_rec(r, [&](const Unit& u) { any_q = any_q || u.stage == Stage::WaitA; });
                // Be kvorumo — tik tie švieži Straggler'iai, kurie gali paleisti A (laukiami iki a(i), kaip fire_a): kitaip
                // ašis būtų nuolat vėluojantis įėjimas (didžiausias ewma), variklis lauktų jo iki a(i) ir laiku paruoštų
                // brolių nepastebėtų (įėjimas be rašymo laiko — garsas — tada atrodo vėlus, Late nenusiima; T2).
                for_rec(r, [&](const Unit& u) {
                    if (any_q ? u.stage == Stage::WaitA
                              : (u.stage == Stage::WaitB && fresh(u) && (ready(u, false) || expected(u) <= r.cuts.a)))
                        consider(u);
                });
                if (pivot >= 0) return pivot;
            }
        }
        std::int64_t best = INT64_MAX;
        for (Unit& u : units) {
            if (u.stage == Stage::None) continue;
            bool w = false;
            for (std::uint32_t k = 0; k < u.nin; ++k) w = w || waitable(ins[u.in0 + k]);
            if (w) best = std::min(best, deadline(u));
        }
        // B / C grupė: ašis — anksčiausiai laukiamas įėjimas (pirmas paruoštas vienetas paleidžia Wr laikmatį). Pirmenybė:
        // švieži vienetai (pakibęs — miss_run ≥ 1 — tikėtina užstrigęs), tada tie, kurių laukiamas laikas dar ateityje. Kai
        // laukiamų daugiau nei vienas — laukiama 1 ms gabalais, kad kitų atvykimas būtų pastebėtas laiku (C: ir kai ašis —
        // užstrigęs brolis).
        std::uint32_t n_b = 0;
        int best_rank = INT_MAX;
        std::int64_t best_exp = INT64_MAX;
        const std::int64_t tnow = now();
        for (Unit& u : units) {
            if ((u.stage != Stage::WaitB && u.stage != Stage::WaitC) || deadline(u) != best) continue;
            const bool fr = fresh(u);
            for (std::uint32_t k = 0; k < u.nin; ++k) {
                const In& in = ins[u.in0 + k];
                if (!waitable(in)) continue;
                ++n_b;
                const std::int64_t e = in.ewma_set ? T(jidx(u, in)) + in.ewma : INT64_MIN;
                const int rank = (fr ? 0 : 2) + (e >= tnow ? 0 : 1);
                if (pivot < 0 || rank < best_rank || (rank == best_rank && e <= best_exp)) {
                    best_rank = rank;
                    best_exp = e;
                    pivot = static_cast<std::int64_t>(u.in0 + k);
                }
            }
        }
        pivot_slice = n_b > 1;
        if (pivot >= 0) return pivot;
        for (Unit& u : units)
            if (u.stage != Stage::None && deadline(u) == best) consider(u);
        return pivot;
    }
    bool pivot_slice = false;

    // --- tick'o atidarymas ----------------------------------------------------------------------------------------
    void maintain_inputs() {
        const std::int64_t mono = clk.now_mono_raw();
        for (In& in : ins) {
            InCounters& c = hub.in(in.hub);
            const InState s = in.st.state();
            if (s == InState::Waiting || s == InState::Detached) {
                if (mono < in.next_attach) continue;
                in.next_attach = mono + kRetryNs;
                const bool attached = in.spec.port->attach_poll(clk.now_tai());
                // Grįžo kitu formatu (--on-format-change exit): prievadas neprisijungia, bet praneša mismatch — patikrinama ir
                // tada (anksčiau — tik po sėkmingo prisijungimo, todėl exit 6 niekada neįvykdavo, laukta be galo).
                if (in.spec.port->mismatch(why_)) {
                    in.st.on_mismatch();
                    end_fatal(EndReason::FormatChange, 6, why_);
                    return;
                }
                if (!attached) continue;
                in.st.on_attach(c);
                in.ever = true;
            } else if (s == InState::Live && in.observe != kNoIndex) {
                // Vienetas buvo Blocked dėl kito įėjimo: šio (gyvo) grain'as tik suskaičiuojamas (ok / late / invalid)
                const Index j = in.observe;
                in.observe = kNoIndex;
                GrainRef g;
                const Probe p = in.spec.port->probe(j, g);
                if (p == Probe::Ready) {
                    const Index h = in.spec.port->head();
                    const auto lw = static_cast<std::int64_t>(in.spec.port->last_write_tai());
                    const std::int64_t ready_at = (h == j && lw > 0) ? lw : T(j);  // h > j — atėjo anksčiau už head'ą
                    in.st.on_ready(c, ready_at <= cuts_for(j, cfg.cuts).a);
                } else if (p == Probe::Invalid) {
                    in.st.on_invalid_grain(c);
                } else if (p == Probe::Gone) {  // flow atkurtas / ištrintas — kaip ir kitur: Detached, prisijungs iš naujo
                    in.st.on_gone(c);
                    detach(in);
                }
            } else if (s == InState::Stalled) {
                const Index h = in.spec.port->head();
                if (in.st.on_stalled_head(h)) continue;  // T10
                GrainRef g;  // pigus probe be laukimo: atkurtas flow (FLOW_INVALID) → Detached → prisijungs iš naujo
                if (in.spec.port->probe(h == kNoIndex ? 0 : h, g) == Probe::Gone) {
                    in.st.on_gone(c);
                    detach(in);
                    continue;
                }
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
            for (int pass = 0; pass < 2; ++pass)
                for (Unit& u : units)
                    if (u.stage != Stage::None && u.cur == r.idx) gap_unit(u, Outcome::GapLate, 2);
        }
        r = Rec{};
        r.idx = c;
        r.cuts = cuts_for(c, cfg.cuts);
        r.ended = false;
        r.opening = true;
        newest = c;
        ++hub.engine().ticks;
        if (res.first_index == kNoIndex) res.first_index = c;
        res.last_index = c;
        ctx_index = c;
        ctx_held = 0;
        maintain_inputs();
        report_inputs();
        if (ended) return;
        service_retries();
        for (Unit& u : units) u.no_content = false;
        fn->begin_tick(c, ctx);
        if (ended) return;
        for (Unit& u : units) {
            if (u.failed || c % u.stride != u.residue) continue;
            if (u.stage != Stage::None) {  // T2: dar laukia ankstesnio (pernešto) indekso — tick'o narys
                u.join = c;
                ++r.joiners;
                ++r.unresolved;
                continue;
            }
            start_unit(u, c, r, false);
        }
        r.opening = false;
        maybe_end(r);
    }

    Index align_up(Index x) const {
        const Index a = std::max<std::uint32_t>(1, cfg.skip_align);
        return (x + a - 1) / a * a;
    }

    // T11: paskutinis indeksas, kurio t_open ≤ dabar.
    Index last_opened_by(std::int64_t n) const {
        Index c = due_index(static_cast<std::uint64_t>(n), cfg.cuts);
        while (cuts_for(c + 1, cfg.cuts).open <= n) ++c;
        while (c > 0 && cuts_for(c, cfg.cuts).open > n) --c;
        return c;
    }

    void skip(Index from, Index to) {  // T11: praleisti [from, to) — neužpildomi (K-T6), bet suskaičiuoti
        const bool warm = static_cast<double>(clk.now_mono_raw() - start_mono) < cfg.warmup_s * 1e9;
        for (Unit& u : units) {
            if (u.failed) continue;
            const std::uint64_t n = count_in(from, to, u.stride, u.residue);
            skip_loss = !step_in_tick && !warm;
            if (n) fn->on_skip(uid(u), from, to, ctx);
            UnitCounters& uc = hub.unit(u.hub);
            (step_in_tick ? uc.clock_skips : warm ? uc.warmup_skips : uc.late_skips) += n;
            for (std::uint32_t m = 0; m < u.nout; ++m) {
                OutCounters& c = hub.out(outs[u.out0 + m].hub);
                (step_in_tick ? c.clock_skips : warm ? c.warmup_skips : c.late_skips) += n;
            }
        }
    }

    bool wait_all() {  // StartMode::WaitAll — nepradedama, kol neprisijungė visi būtini įėjimai
        for (;;) {
            if (stop.load()) return false;
            maintain_inputs();
            report_inputs();
            if (ended) return false;
            bool all = true;
            for (const In& in : ins)
                if (in.spec.need == Need::Required && !in.ever) all = false;
            if (all) return true;
            clk.sleep_until(clk.now_tai() + static_cast<std::uint64_t>(kRetryNs), stop);
            check_step();  // atskaita atnaujinama — lėtas dreifas laukiant nėra žingsnis
        }
    }

    // --rt-priority / --cpu-affinity: tik ši (variklio) gija; jau paleistos gijos (statistika, darbininkai) nepaveldi.
    void apply_rt() {
        if (!cfg.cpu_affinity.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int c : cfg.cpu_affinity)
                if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
            if (const int e = pthread_setaffinity_np(pthread_self(), sizeof set, &set)) {
                res.warn += std::string("--cpu-affinity: ") + std::strerror(e) + "; ";
                emit(EventKind::Warning, 0, kNoIndex, e, ("--cpu-affinity: " + std::string(std::strerror(e))).c_str());
            }
        }
        if (cfg.rt_priority > 0) {
            sched_param sp{};
            sp.sched_priority = cfg.rt_priority;
            if (const int e = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp)) {
                res.warn += std::string("--rt-priority: ") + std::strerror(e) + " (reikia CAP_SYS_NICE / RLIMIT_RTPRIO); ";
                emit(EventKind::Warning, 0, kNoIndex, e,
                     ("--rt-priority: " + std::string(std::strerror(e)) + " (reikia CAP_SYS_NICE / RLIMIT_RTPRIO)").c_str());
            }
        }
    }

    void stop_worker() {
        if (!rworker.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(rm);
            rquit = true;
        }
        rcv.notify_one();
        rworker.join();
    }

    RunResult run(TickFunction& f) {
        fn = &f;
        if (cfg.cuts.rate.num <= 0 || cfg.cuts.rate.den <= 0) {  // dažnis 0 — ribų nėra (kitaip ciklas be galo)
            end_fatal(EndReason::Fatal, 1, "netinkamas grain'ų dažnis");
            return res;
        }
        hub.prepare();
        rst = std::make_unique<std::atomic<int>[]>(units.size());
        for (std::size_t k = 0; k < units.size(); ++k) rst[k].store(kRetryIdle);
        apply_rt();
        batch.reserve(units.size());
        sd.check(clk);
        start_mono = clk.now_mono_raw();
        if (cfg.start == StartMode::WaitAll && !wait_all()) {
            if (!ended) res.reason = EndReason::Stopped;
            fn->finish(ctx);
            stop_worker();
            return res;
        }
        step_in_tick = false;
        start_tai = now();
        start_mono = clk.now_mono_raw();
        const auto dur_ns = static_cast<std::int64_t>(cfg.duration_s * 1e9);
        Index next = due_index(clk.now_tai(), cfg.cuts);
        while (cuts_for(next, cfg.cuts).open < now()) ++next;
        next = align_up(next);
        while (!ended) {
            if (stop.load()) {
                res.reason = EndReason::Stopped;
                break;
            }
            const Cuts nc = cuts_for(next, cfg.cuts);
            if (dur_ns > 0) {  // trukmė — monotoniniu laiku (TAI žingsniai jos nekeičia)
                const std::int64_t to_open = std::max<std::int64_t>(0, nc.open - now());
                if (clk.now_mono_raw() - start_mono + to_open >= dur_ns) {
                    pump(INT64_MAX, true, kNoIndex);
                    if (!ended) res.reason = EndReason::Duration;
                    break;
                }
            }
            pump(nc.open, false, next);
            if (ended) break;
            if (stop.load()) {
                res.reason = EndReason::Stopped;
                break;
            }
            check_step();
            if (now() < nc.open) continue;  // laikrodis atgal — miegama iki t_open naujoje laiko juostoje
            const Index last = last_opened_by(now());
            if (last > next) {
                const Index to = align_up(last);
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
        stop_worker();
        res.failed_units = failed_n;
        for (const Unit& u : units) res.unit_failures += hub.unit(u.hub).failures;
        // Praradimai — vieneto lygiu (kiekvienas indeksas baigiasi lygiai vienu rezultatu); išėjimų spragų skaitikliai —
        // tų pačių vieneto spragų pasekmė. Laikrodžio įvykiai ir šaltinio „nėra turinio“ — ne praradimas.
        // Įėjimo `lost` — praradimas tik būtinam (Required) įėjimui. Wanted / Optional (composite sluoksnis, enc garsas)
        // trūkumas — to įėjimo būsena: vienetas be jo commit'ina pilną indeksą, o funkcija skaičiuoja savo pasekmę
        // (layer_absent, audio_underrun_frames).
        for (const In& in : ins)
            if (in.spec.need == Need::Required && hub.in(in.hub).lost != 0) res.any_loss = true;
        for (std::size_t k = 0; k < hub.outputs(); ++k)
            if (hub.out(static_cast<std::uint32_t>(k)).commit_errors != 0) res.any_loss = true;
        for (std::size_t k = 0; k < hub.units(); ++k) {
            const UnitCounters& c = hub.unit(static_cast<std::uint32_t>(k));
            if (c.gaps - c.gaps_clock - c.gaps_no_content + c.lapped + c.late_skips != 0) res.any_loss = true;
        }
        return res;
    }
};

// --- TickContext ---------------------------------------------------------------------------------------------------
Clock& TickContext::clock() { return static_cast<TickEngine::Impl*>(e_)->clk; }
Index TickContext::index() const { return static_cast<const TickEngine::Impl*>(e_)->ctx_index; }
void TickContext::no_content(std::uint32_t unit) { static_cast<TickEngine::Impl*>(e_)->units[unit].no_content = true; }
void TickContext::fail_unit(std::uint32_t unit, std::string why) {
    auto* d = static_cast<TickEngine::Impl*>(e_);
    d->fail_unit(d->units[unit], std::move(why));
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
bool TickContext::skip_is_loss() const { return static_cast<const TickEngine::Impl*>(e_)->skip_loss; }

// --- TickEngine ----------------------------------------------------------------------------------------------------
TickEngine::TickEngine(const EngineConfig& c, Clock& k, StatsHub& h, const std::atomic<bool>& s)
    : d_(std::make_unique<Impl>(c, k, h, s)) {}
TickEngine::~TickEngine() = default;

bool TickEngine::add_unit(const UnitSpec& s, std::uint32_t& unit, std::string& why) {
    Impl& d = *d_;
    if (d.cfg.cuts.rate.num <= 0 || d.cfg.cuts.rate.den <= 0) {
        why = "netinkamas grain'ų dažnis";
        return false;
    }
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
    u.hub = d.hub.add_unit(s.name);  // pirma vienetas — jo įėjimai ir išėjimai priskiriami jam (StatsHub)
    u.in0 = static_cast<std::uint32_t>(d.ins.size());
    u.nin = static_cast<std::uint32_t>(s.in.size());
    for (std::size_t k = 0; k < s.in.size(); ++k) {
        Impl::In in{s.in[k], InputState(d.cfg.stall_after, d.cfg.late_clear_run)};
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
        const std::string oname = s.name + "/out" + std::to_string(m);
        d.outs.push_back(Impl::Out{s.out[m], g, d.hub.add_output(oname), oname, id});
    }
    d.slots.resize(d.outs.size(), nullptr);
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
std::uint32_t TickEngine::unit_id(std::uint32_t unit) const { return d_->units[unit].hub; }
InState TickEngine::in_state(std::uint32_t unit, std::uint32_t input) const {
    return d_->ins[d_->units[unit].in0 + input].st.state();
}
bool TickEngine::unit_failed(std::uint32_t unit) const { return d_->units[unit].failed; }
bool TickEngine::unit_ever_failed(std::uint32_t unit) const { return d_->units[unit].ever_failed; }
Index TickEngine::unit_fail_index(std::uint32_t unit) const { return d_->units[unit].fail_index; }
bool TickEngine::in_late(std::uint32_t unit, std::uint32_t input) const {
    return d_->ins[d_->units[unit].in0 + input].st.late();
}

}  // namespace mxlv::engine
