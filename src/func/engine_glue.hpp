// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Funkcijų ↔ bendro tick variklio jungtis (planas docs/planai/2026-11-12-tick-variklis.md §7 „mxlv::func“, V8+):
// parinktys → EngineConfig (viena vieta) ir variklio skaitikliai → santraukos raktai (tie patys kaip prieš migraciją).
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine/engine.hpp"
#include "engine/mxl_ports.hpp"
#include "engine/stats.hpp"
#include "func/func_common.hpp"
#include "func/stats_lines.hpp"

namespace mxlv::func {

// Variklio ribos funkcijai:
//   Legacy    — šiandieninis ciklas: viena riba ties T + --deadline-ms, apklausa 0,5 ms, be pertraukiklio, wait-all;
//   Isolation — planas V10: Wr = P/4, M = 3, Late histerezė 8, futex (ašinis) laukimas; D = --deadline-ms (apkarpoma iki
//               2P − G), wait-all (startas keičiamas V18).
// Abiem: spragų politika — --on-loss / --hold-ms (V17, K3; numatyta invalid).
enum class EngineTiming { Legacy, Isolation };

// read_offset_ns — ro (logo: --read-offset-us; gen: 0). Trukmė, warmup, --rt-priority, --cpu-affinity, spragų politika
// (--on-loss, --hold-ms; K3) — iš parinkčių, abiem ribų režimais.
engine::EngineConfig engine_config(const Opts& o, EngineTiming t, std::int64_t read_offset_ns);

// Efektyvi --deadline-ms riba ns (V20, K-T2): 0 — auto 3·P/4 pagal flow grain'ų dažnį (--format); kitaip — nurodyta.
std::int64_t deadline_ns(const Opts& o);
// enc (peržiūra 2026-11-13 p. 4, iki briaunų laiko sutarties): kadro garsas baigiamas ties T(j) + P, todėl numatytoji enc riba
// — 3·P/2 (50 Hz — 30 ms, 25p — 60 ms; > 2P − G apkarpoma), aiški --deadline-ms — nepakitusi; garso (Wanted) įėjimas laukiamas
// ties T + P (enc_audio_expect_ns).
std::int64_t enc_deadline_ns(const Opts& o);
std::int64_t enc_audio_expect_ns(const Opts& o);

// --on-loss reikšmė → politika (invalid | freeze | black | leave; nežinoma — invalid, CLI ją jau atmeta).
engine::GapPolicy gap_policy(const std::string& s);

// K3 spragų užpildymas išėjimams (planas §6 D4): Repeat / Black kopija pagal išėjimo saugyklą — host: memcpy; cuda-linear:
// D2D į savo stream'ą ir vienas sync prieš commit (MxlVideoOutput flush). Juodi grain'ai paruošiami iš anksto (prepare —
// prieš variklio run(), tick'e nealokuojama) visoms politikoms, išskyrus leave: invalid politikai juodas grain'as — INVALID
// grain'o turinys (kartą slot'ui per gedimą, VideoOut::paint_black), kad žymės nepaisantis skaitytojas nematytų pasenusio
// kadro; kaina — vienas grain'as kiekvienam rastrui (black_bytes).
// Naudojimas: prepare(v.raster) → ports.emplace_back(v, gf.copy(), gf.flush(), gf.black(v.raster)).
// cuda — išėjimai cuda-linear; tada konstruktorius kviečiamas po cudaSetDevice (stream'as ir buferiai — tame įrenginyje).
class GapFill {
public:
    GapFill(engine::GapPolicy p, bool cuda);
    ~GapFill();
    GapFill(const GapFill&) = delete;
    GapFill& operator=(const GapFill&) = delete;

    bool prepare(const vramflow::gfx::Raster& r, std::string& err);  // false — CUDA klaida (err)
    engine::GrainCopy copy() const;
    std::function<bool()> flush() const;                      // host — tuščias (kopija sinchroninė)
    const void* black(const vramflow::gfx::Raster& r) const;  // nullptr — politikai juodo nereikia / neparuošta
    std::uint64_t black_bytes() const { return black_bytes_; }  // paruoštų juodų grain'ų atmintis (VRAM, kai cuda-linear)

private:
    struct Black {
        vramflow::gfx::Raster r;
        void* p = nullptr;
        std::vector<std::uint8_t> host;
    };
    engine::GapPolicy policy_;
    bool cuda_;
    void* stream_ = nullptr;  // cudaStream_t (antraštė be CUDA)
    std::vector<Black> black_;
    std::uint64_t black_bytes_ = 0;
};

// „0,2-3“ → {0, 2, 3}; false — netinkamas sąrašas (tuščias — tinkamas, nieko).
bool parse_cpu_list(const std::string& s, std::vector<int>& out);

// Exit kodas (planas D8, --strict-exit). rc — jau nustatytas fatalus kodas (1, 2, 5, 6, kExitInputs, kStoppedWaiting) arba 0;
// loss — bent vienas praradimas; failed_units — vienetai, sugedę sustabdymo metu (neatsigavę; visi — variklis jau grąžino
// Fatal → rc 2).
//   abiem režimais: vienetas vis dar sugedęs → 2 (tarnyboje sugedęs flow'as nėra „tvarkinga pabaiga“);
//   griežtas (on; auto ir --duration > 0): praradimas → 4, kitaip 0;
//   švelnus (off; auto ir --duration 0): 0 (praradimai — santraukoje: any_loss, skaitikliai, gyva statistika).
// Įrašo st.strict ir st.any_loss (santraukos laukai strict_exit, any_loss).
bool strict_exit(const Opts& o);
int exit_code_for(const Opts& o, int rc, bool loss, std::uint32_t failed_units, Stats& st);

// Sugedusio vieneto išėjimų atlaisvinimas (K-T11): MxlVideoOutput::release_fn → VideoOut::release(inst); atkūrimas (D8):
// reopen(n) — flow'as n sukuriamas iš naujo (kviečia variklio atkūrimo gija; tuščias — neatkuriama).
void set_release(std::vector<engine::MxlVideoOutput>& ports, std::vector<VideoOut>& outs, mxlInstance inst,
                 std::function<bool(std::size_t)> reopen = {});

// Variklio statistika funkcijai: StatsLines (--stats-file, įvykiai į stderr) + StatsHub. finish() — po run(): paskutinė
// (nepilno intervalo) eilutė ir publikavimo gijos sustabdymas.
struct EngineStats {
    EngineStats(const Opts& o, const char* role);
    void finish();
    StatsLines lines;
    engine::StatsHub hub;
};

// Variklio pabaigos priežastis → funkcijos rc (Fatal — kodas, FormatChange — 6, InputsTimeout — 4; Stopped / Duration — 0).
int engine_rc(const engine::RunResult& r, Stats& st);

// Kur santraukoje rašyti variklio praleidimus: proceso lygiu (gen) ar kiekvienam flow'ui (logo, composite…).
enum class SkipScope { Process, PerFlow, PerFlowFrames };  // PerFlowFrames — enc: *_skips_frames

// Variklio rezultatas ir skaitikliai → Stats (first / last_index, clock_steps, clock_backward, *_skips). Raktai, kurių
// reikšmė 0, nerašomi (kaip anksčiau — „įvykio“ skaitikliai atsiranda tik padidinti); clock_steps — tik Process (gen
// santraukoje jis visada buvo, kitų funkcijų — ne).
void to_summary(const engine::StatsHub& hub, const engine::TickEngine& eng, const engine::RunResult& r, std::uint32_t units,
                SkipScope scope, Stats& st);

// Tick'o laikai funkcijų histogramoms — ATSKIRAI kiekvienam iš dviejų galimai persidengiančių indeksų (pernešimas:
// i − 1 baigiasi po i atidarymo; slot = index & 1), kad vieno indekso būsena nepasiimtų kito laikų.
//   proc  — tik apdorojimas: Σ per fazes (process pradžia → paskutinis tos fazės vieneto commit / lap / klaida);
//   wait  — tick'o atidarymas → pirmo apdorojimo pradžia (įėjimų laukimas);
//   held  — laikas tarp fazių (paruošti / pavėluoję vienetai laukė B / C fazės): (paskutinis commit − pirma pradžia) − proc;
//   end   — paskutinis commit (be commit'ų — dabar).
class TickTimes {
public:
    struct Result {
        bool processed = false;
        std::uint64_t proc = 0, wait = 0, held = 0, end = 0, first = 0;
        std::uint32_t phases = 0;
    };
    void wake(engine::Index i, std::uint64_t now) { s_[i & 1] = S{now}; }
    void phase_begin(engine::Index i, std::uint64_t now) {
        S& t = s_[i & 1];
        close(t);
        if (t.phases++ == 0) t.first = now;
        t.phase0 = t.last = now;
        t.open = true;
    }
    // Vieneto pabaiga po šios fazės apdorojimo (Ready / Lapped / OutputError); committed — tikras commit.
    void unit_done(engine::Index i, std::uint64_t now, bool committed) {
        S& t = s_[i & 1];
        if (t.open) t.last = now;
        if (committed) t.commit = now;
    }
    std::uint64_t first(engine::Index i) const { return s_[i & 1].first; }
    Result end(engine::Index i, std::uint64_t now) {
        S& t = s_[i & 1];
        close(t);
        Result r;
        r.processed = t.phases > 0;
        r.phases = t.phases;
        r.end = t.commit ? t.commit : now;
        r.first = r.processed ? t.first : r.end;
        r.proc = t.proc;
        r.wait = r.first > t.wake ? r.first - t.wake : 0;
        const std::uint64_t span = r.end > r.first ? r.end - r.first : 0;
        r.held = span > t.proc ? span - t.proc : 0;
        return r;
    }

private:
    struct S {
        std::uint64_t wake = 0, first = 0, phase0 = 0, last = 0, commit = 0, proc = 0;
        std::uint32_t phases = 0;
        bool open = false;
    };
    static void close(S& t) {
        if (!t.open) return;
        t.proc += t.last > t.phase0 ? t.last - t.phase0 : 0;
        t.open = false;
    }
    S s_[2];
};

// Testinė išėjimo klaidų injekcija (--test-out-error open|commit, paslėpta): apgaubia tikrą išėjimą; nuo pirmo open po
// `after_s` (TAI) `count` indeksų iš eilės (0 — visam laikui) open grąžina Error arba commit — false (slot'as atšaukiamas).
// Tikrina, kad išėjimo klaida nėra tyli (skaitiklis output_errors, ilgalaikė — exit 2 su pranešimu).
class FaultyOutput final : public engine::OutputPort {
public:
    FaultyOutput(engine::OutputPort& inner, const Opts& o);
    engine::OpenResult open(engine::Index i, void*& slot) override;
    bool commit() override;
    void cancel() override { in_.cancel(); }
    bool fill_gap(engine::Index i, engine::GapAction a, engine::Index from) override { return in_.fill_gap(i, a, from); }
    bool flush_gaps() override { return in_.flush_gaps(); }
    bool interlaced() const override { return in_.interlaced(); }
    std::uint32_t slot_count() const override { return in_.slot_count(); }
    void release() override { in_.release(); }
    bool reopen() override { return in_.reopen(); }

private:
    bool active(engine::Index i);
    engine::OutputPort& in_;
    bool open_mode_;
    std::uint64_t from_tai_;
    std::uint32_t count_;
    engine::Index first_ = engine::kNoIndex, cur_ = engine::kNoIndex;
};

// Išėjimų prievadai varikliui: be --test-out-error — patys prievadai, su ja — FaultyOutput apvalkalai (laikomi `keep`;
// --test-out-error-flow N — tik flow'o N išėjimas).
std::vector<engine::OutputPort*> output_ports(std::vector<engine::OutputPort*> ports, const Opts& o,
                                              std::vector<FaultyOutput>& keep);

}  // namespace mxlv::func
