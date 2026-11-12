// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variklio testų pakaitalai (planas §9 V4): ScriptedInput (rašytojas pagal scenarijų virtualiu laiku), RecordingOutput,
// RecordingFunction. Viskas vienoje gijoje; vektoriai rezervuojami iš anksto (alokacijų testas V6).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "engine/clock.hpp"
#include "engine/engine.hpp"
#include "engine/ports.hpp"
#include "timing/index.hpp"

namespace mxlv::engine::test {

inline constexpr std::int64_t kNever = std::numeric_limits<std::int64_t>::max();
inline constexpr timing::Rational k50{50, 1};

inline std::int64_t T(Index i, timing::Rational r = k50) { return timing::time_of(i, r); }

// Grain'as j commit'inamas ties T(j) + arrival(j) (TAI); kNever — niekada. Rašytojas seka TAI, todėl laikrodžio
// žingsniai jį paveikia taip pat kaip variklį.
class ScriptedInput final : public InputPort {
public:
    ScriptedInput(VirtualClock& c, timing::Rational rate = k50) : clk_(c), rate_(rate) {}

    std::function<std::int64_t(Index)> arrival = [](Index) { return std::int64_t{1'200'000}; };
    std::function<bool(std::uint64_t now)> alive = [](std::uint64_t) { return true; };
    std::function<bool(std::uint64_t now)> attachable = [](std::uint64_t) { return true; };
    std::function<bool(Index)> invalid;    // OK + INVALID
    std::function<bool(Index)> lapped_at;  // lap patikra
    std::function<void(Index j, std::uint64_t timeout)> on_wait;
    std::function<bool(std::uint64_t now)> gone;  // flow atkurtas / ištrintas → probe grąžina Gone
    bool mismatch_flag = false;
    std::uint32_t slots = 10;

    std::uint64_t waits = 0, probes = 0, alive_calls = 0, attach_calls = 0;

    bool attach_poll(std::uint64_t now) override {
        ++attach_calls;
        return attachable(now);
    }
    bool mismatch(std::string& why) const override {
        if (mismatch_flag) why = "formatas kitas";
        return mismatch_flag;
    }
    Index head() override {
        const std::uint64_t now = clk_.now_tai();
        Index j = timing::index_of(static_cast<std::int64_t>(now), rate_) + 2;  // arrival ≥ 0 — vėlesnių dar nėra
        for (int k = 0; k < 1000 && (head_ == kNoIndex || j > head_); ++k, --j) {
            if (arrived(j, now)) {
                head_ = j;
                break;
            }
            if (j == 0) break;
        }
        return head_;
    }
    std::uint64_t last_write_tai() override {
        const Index h = head();
        if (h == kNoIndex) return 0;
        return static_cast<std::uint64_t>(T(h, rate_) + arrival(h));
    }
    Probe probe(Index j, GrainRef& out) override {
        ++probes;
        if (gone && gone(clk_.now_tai())) return Probe::Gone;
        const Index h = head();
        if (h == kNoIndex || j > h) return Probe::NotYet;
        if (j + slots < h + 2) return Probe::TooLate;
        out.ptr = &token_;
        out.flags = 0;
        if (invalid && invalid(j)) {
            out.flags = 1;
            return Probe::Invalid;
        }
        return Probe::Ready;
    }
    Probe wait(Index j, std::uint64_t timeout_ns, GrainRef& out) override {
        ++waits;
        if (on_wait) on_wait(j, timeout_ns);
        const Probe p = probe(j, out);
        if (p != Probe::NotYet) return p;
        const std::uint64_t now = clk_.now_tai();
        const std::uint64_t deadline = now + timeout_ns;
        const std::int64_t a = arrival(j);
        if (a != kNever) {
            const auto ta = static_cast<std::uint64_t>(T(j, rate_) + a);
            if (ta <= deadline) {
                if (ta > now) clk_.advance(ta - now);
                return probe(j, out);
            }
        }
        clk_.advance(deadline - now);
        return probe(j, out);
    }
    bool writer_alive() override {
        ++alive_calls;
        return alive(clk_.now_tai());
    }
    bool lapped(Index j, Index) override { return lapped_at && lapped_at(j); }
    void detach() override {}
    void retarget(const FlowRef&) override {}

private:
    bool arrived(Index j, std::uint64_t now) {
        const std::int64_t a = arrival(j);
        return a != kNever && T(j, rate_) + a <= static_cast<std::int64_t>(now);
    }
    VirtualClock& clk_;
    timing::Rational rate_;
    Index head_ = kNoIndex;
    int token_ = 0;
};

class RecordingOutput final : public OutputPort {
public:
    enum class Kind : std::uint8_t { Commit, Gap, Cancel };
    struct Event {
        Kind kind;
        Index index;
        GapAction action;
        Index from;
        std::uint64_t tai;
    };
    explicit RecordingOutput(VirtualClock& c, bool interlaced = false, std::uint32_t slots = 10)
        : clk_(c), interlaced_(interlaced), slots_(slots) {
        events.reserve(1 << 14);
    }

    std::vector<Event> events;
    std::uint64_t opens = 0, commits = 0, cancels = 0, flushes = 0;
    bool open_now = false;

    std::function<bool(Index)> fail_open;   // open klaida šiam indeksui (OpenResult::Error)
    std::function<bool(Index)> already_at;  // open → AlreadyCommitted šiam indeksui (laikrodis atgal, M11)
    OpenResult open(Index i, void*& slot) override {
        if (last_ != kNoIndex && i <= last_) return OpenResult::AlreadyCommitted;
        if (already_at && already_at(i)) return OpenResult::AlreadyCommitted;
        if (fail_open && fail_open(i)) return OpenResult::Error;
        ++opens;
        open_now = true;
        open_i_ = i;
        slot = buf_;
        return OpenResult::Ok;
    }
    std::function<bool(Index)> fail_commit;  // commit klaida šiam indeksui (grain'as neįrašomas)
    std::uint64_t failed = 0;
    bool commit() override {
        if (fail_commit && fail_commit(open_i_)) {
            ++failed;
            open_now = false;
            return false;
        }
        events.push_back({Kind::Commit, open_i_, GapAction::None, kNoIndex, clk_.now_tai()});
        ++commits;
        last_ = open_i_;
        open_now = false;
        return true;
    }
    void cancel() override {
        events.push_back({Kind::Cancel, open_i_, GapAction::None, kNoIndex, clk_.now_tai()});
        ++cancels;
        open_now = false;
    }
    bool fill_gap(Index i, GapAction a, Index from) override {
        if (last_ != kNoIndex && i <= last_) return false;
        events.push_back({Kind::Gap, i, a, from, clk_.now_tai()});
        last_ = i;
        return true;
    }
    bool flush_gaps() override {
        ++flushes;
        return true;
    }
    bool interlaced() const override { return interlaced_; }
    std::uint32_t slot_count() const override { return slots_; }

    const Event* find(Index i) const {
        for (const Event& e : events)
            if (e.index == i && e.kind != Kind::Cancel) return &e;
        return nullptr;
    }
    // Commit'inti (turinys arba spraga) indeksai griežtai didėja — invariantas visuose testuose.
    bool strictly_increasing() const {
        Index prev = kNoIndex;
        for (const Event& e : events) {
            if (e.kind == Kind::Cancel) continue;
            if (prev != kNoIndex && e.index <= prev) return false;
            prev = e.index;
        }
        return true;
    }

private:
    VirtualClock& clk_;
    bool interlaced_;
    std::uint32_t slots_;
    Index last_ = kNoIndex, open_i_ = kNoIndex;
    char buf_[64]{};
};

class RecordingFunction final : public TickFunction {
public:
    struct Call {
        Index index;
        std::uint8_t phase;
        std::uint64_t tai;
        std::int64_t held_ns;
        std::vector<std::uint32_t> units;
    };
    struct End {
        std::uint32_t unit;
        Index index;
        Outcome outcome;
        std::uint64_t tai;
    };
    explicit RecordingFunction(VirtualClock& c) : clk_(c) {
        calls.reserve(1 << 12);
        ends.reserve(1 << 16);
        skips.reserve(1 << 10);
    }

    struct Skip {
        std::uint32_t unit;
        Index from, to;
    };
    std::vector<Call> calls;
    std::vector<End> ends;
    std::vector<Skip> skips;
    std::vector<std::int64_t> steps;
    std::function<void(Index, TickContext&)> on_begin_tick, on_end_tick;
    std::function<void(const Batch&, TickContext&)> on_process;

    void begin_tick(Index i, TickContext& ctx) override {
        if (on_begin_tick) on_begin_tick(i, ctx);
    }
    bool process(const Batch& b, TickContext& ctx) override {
        Call c{b.index, b.phase, clk_.now_tai(), ctx.held_ns(), {}};
        for (const UnitTick& u : b.units) c.units.push_back(u.unit);
        calls.push_back(std::move(c));
        if (on_process) on_process(b, ctx);
        return true;
    }
    void end_unit(const UnitTick& u, Outcome o, TickContext&) override {
        ends.push_back({u.unit, u.index, o, clk_.now_tai()});
    }
    void end_tick(Index i, TickContext& ctx) override {
        if (on_end_tick) on_end_tick(i, ctx);
    }
    void on_clock_step(std::int64_t d, TickContext&) override { steps.push_back(d); }
    void on_skip(std::uint32_t unit, Index from, Index to, TickContext&) override { skips.push_back({unit, from, to}); }

    std::vector<const Call*> calls_for(Index i) const {
        std::vector<const Call*> r;
        for (const Call& c : calls)
            if (c.index == i) r.push_back(&c);
        return r;
    }
    // ≤ 4 process kvietimai indeksui — invariantas (A, B, C ir retas paskutinis batch'as ties c(i); anksčiau ≤ 3, tada
    // po trečio kvietimo paruoštas vienetas buvo išmetamas).
    bool at_most_four_per_index() const {
        std::vector<Index> idx;
        for (const Call& c : calls) idx.push_back(c.index);
        std::sort(idx.begin(), idx.end());
        for (std::size_t k = 0; k + 4 < idx.size(); ++k)
            if (idx[k] == idx[k + 4]) return false;
        return true;
    }

private:
    VirtualClock& clk_;
};

// Per-indeksinė apskaita (kiekviename teste): kiekvienas vieneto dalyvaujantis indeksas [first_index, last_index] baigiasi
// LYGIAI vienu end_unit arba variklio praleidimu (on_skip); be įrašo gali likti tik ≤ 2 paskutiniai (vykdomi sustabdant).
// Išėjimui: Ready ↔ Commit; spraga / Lapped ↔ Gap įrašas, nebent politika leave (be įrašo); Skipped / OutputError — be įrašo; kiekvienas
// Commit / Gap įrašas turi atitinkamą end_unit.
inline void check_run(const RecordingFunction& fn, const RunResult& res, std::uint32_t unit, const RecordingOutput* o,
                      std::uint32_t stride = 1, std::uint32_t residue = 0, bool leave = false) {
    INFO("vienetas " << unit);
    if (res.first_index == kNoIndex) return;
    std::map<Index, Outcome> ends;
    bool dup = false;
    for (const auto& e : fn.ends)
        if (e.unit == unit && !ends.emplace(e.index, e.outcome).second) dup = true;
    CHECK_FALSE(dup);
    std::set<Index> skipped;
    for (const auto& k : fn.skips)
        if (k.unit == unit)
            for (Index i = k.from; i < k.to; ++i)
                if (i % stride == residue && !skipped.insert(i).second) dup = true;
    CHECK_FALSE(dup);
    Index resolved = kNoIndex;
    for (const auto& [i, oc] : ends) resolved = resolved == kNoIndex ? i : std::max(resolved, i);
    for (Index i : skipped) resolved = resolved == kNoIndex ? i : std::max(resolved, i);
    std::uint64_t holes = 0, both = 0, tail = 0;
    for (Index i = res.first_index; i <= res.last_index; ++i) {
        if (i % stride != residue) continue;
        const bool e = ends.count(i) != 0, k = skipped.count(i) != 0;
        if (e && k) ++both;
        if (!e && !k) ++((resolved == kNoIndex || i > resolved) ? tail : holes);
    }
    INFO("indeksai " << res.first_index << "…" << res.last_index);
    CHECK(holes == 0);
    CHECK(both == 0);
    CHECK(tail <= 2);
    if (o == nullptr) return;
    std::map<Index, RecordingOutput::Kind> ev;
    for (const auto& e : o->events)
        if (e.kind != RecordingOutput::Kind::Cancel) ev[e.index] = e.kind;
    std::uint64_t bad = 0;
    for (const auto& [i, oc] : ends) {
        const auto it = ev.find(i);
        if (oc == Outcome::Ready) bad += it == ev.end() || it->second != RecordingOutput::Kind::Commit;
        else if (oc == Outcome::Skipped || oc == Outcome::OutputError) bad += it != ev.end();
        else bad += leave ? it != ev.end() : (it == ev.end() || it->second != RecordingOutput::Kind::Gap);
    }
    for (const auto& [i, kind] : ev) bad += ends.count(i) == 0;
    CHECK(bad == 0);
}

// N vienetų po vieną įėjimą ir vieną išėjimą (logo forma); laikrodis pradeda ties T(start).
struct Rig {
    explicit Rig(std::uint32_t n, EngineConfig c = test_config(), Index start = 1000)
        : clk(static_cast<std::uint64_t>(T(start, c.cuts.rate))), fn(clk), cfg(std::move(c)) {
        for (std::uint32_t u = 0; u < n; ++u) {
            ins.push_back(std::make_unique<ScriptedInput>(clk, cfg.cuts.rate));
            outs.push_back(std::make_unique<RecordingOutput>(clk));
        }
    }
    static EngineConfig test_config() {
        EngineConfig c;
        c.cuts = default_cuts(k50);
        c.warmup_s = 0;
        return c;
    }
    TickEngine& build() {
        eng = std::make_unique<TickEngine>(cfg, clk, hub, stop);
        for (std::uint32_t u = 0; u < ins.size(); ++u) {
            UnitSpec s;
            s.name = "u" + std::to_string(u);
            s.in.push_back(InputSpec{ins[u].get(), {}, 1, Need::Required, InMode::Exact});
            s.out.push_back(outs[u].get());
            std::uint32_t id = 0;
            std::string why;
            if (!eng->add_unit(s, id, why)) throw std::runtime_error(why);
        }
        return *eng;
    }
    // Sustabdyti po tick'o `last` (end_tick).
    void stop_after(Index last) {
        fn.on_end_tick = [this, last](Index i, TickContext&) {
            if (i >= last) stop = true;
        };
    }
    RunResult run() {
        ran = true;
        result = eng->run(fn);
        return result;
    }
    // Invariantai kiekviename Rig teste (tikrinami automatiškai sunaikinant).
    void check_invariants() const {
        for (std::uint32_t u = 0; u < outs.size(); ++u) {
            const RecordingOutput& o = *outs[u];
            CHECK(o.strictly_increasing());
            CHECK_FALSE(o.open_now);
            CHECK(o.opens == o.commits + o.cancels + o.failed);
            check_run(fn, result, u, &o, 1, 0, cfg.on_loss == GapPolicy::Leave);
        }
        CHECK(fn.at_most_four_per_index());
    }
    ~Rig() {
        if (ran) check_invariants();
    }
    const OutCounters& out(std::uint32_t u) const { return hub.out(eng->out_id(u, 0)); }
    const InCounters& in(std::uint32_t u) const { return hub.in(eng->in_id(u, 0)); }

    VirtualClock clk;
    StatsHub hub;
    std::atomic<bool> stop{false};
    RecordingFunction fn;
    EngineConfig cfg;
    std::vector<std::unique_ptr<ScriptedInput>> ins;
    std::vector<std::unique_ptr<RecordingOutput>> outs;
    std::unique_ptr<TickEngine> eng;
    bool ran = false;
    RunResult result;
};

}  // namespace mxlv::engine::test
