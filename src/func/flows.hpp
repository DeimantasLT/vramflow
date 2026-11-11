// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// MXL flow pagalbininkai GPU media funkcijoms (architektūra GPU funkcijų §2): v210 video (cuda-linear arba host) bet
// kuriuo palaikomu formatu (common/format.hpp, planas §P2; interlaced — laukas = grain, lyginis indeksas = viršutinis
// laukas; progresyvus — kadras = grain) ir 48 kHz stereo float32 audio (host atmintis). Numatytasis formatas — 1080i50.
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "common/format.hpp"
#include "gfx/raster.hpp"
#include "runtime/open_policy.hpp"

namespace mxlv::func {

// Flow rolės identifikatoriuose: 4d584c47-00RR-4000-8000-%012x
enum Role : int { kRoleGen = 1, kRoleLogo = 2, kRoleAudio = 3, kRoleDec = 4, kRoleGfx = 5, kRoleComp = 6 };

std::string flow_id(int role, std::uint32_t n);
std::string video_def(int role, std::uint32_t n, std::uint64_t run_seed, const vramflow::VideoFormat& f = vramflow::kFmt1080i50);
// Standartinis v210a (arba v210, jei !r.alpha) flow aprašas grafikos šaltiniui: flow_id(kRoleGfx, n * 8 + k), k < 8.
// Interlaced: grain = laukas, grain_rate 25/1 (MXL kadrų ritmas; grain'ų ritmas 50/1, kaip video_def); progresyvus: 50/1.
std::string gfx_def(std::uint32_t n, std::uint32_t k, const vramflow::gfx::Raster& r, std::uint64_t run_seed = 0);
// Kompozitoriaus išėjimo flow (v210 arba v210a pagal r.alpha): flow_id(kRoleComp, n); grain_rate kaip gfx_def.
std::string comp_def(std::uint32_t n, const vramflow::gfx::Raster& r, std::uint64_t run_seed = 0);
// Tas pats su aiškiu kadrų dažniu (f.raster — sluoksnio / išėjimo rastras, f.frame_rate — flow'o kadrų dažnis).
std::string gfx_def(std::uint32_t n, std::uint32_t k, const vramflow::VideoFormat& f, std::uint64_t run_seed);
std::string comp_def(std::uint32_t n, const vramflow::VideoFormat& f, std::uint64_t run_seed);
std::string audio_def(std::uint32_t n, std::uint64_t run_seed);

// 1080i50 grain'o (lauko) ritmas ir garso mėginiai laukui — numatytojo formato konstantos (seni kvietėjai ir testai).
// Formatui neutralus kodas naudoja VideoFormat::grain_rate() ir vramflow::sample_begin.
constexpr mxlRational kFieldRate{50, 1};
constexpr std::uint32_t kAudioPerField = 960;  // 48000 / 50

constexpr mxlRational mxl_rate(vramflow::Rational r) { return mxlRational{r.num, r.den}; }

// Laukia iki indekso laiko + offset pagal grain'o dažnį `rate` (niekada ne anksčiau; laikrodžio šuolis atgal → laukiam).
// true — buvo šuolis (laukti liko daugiau nei 2 grain'ai).
bool wait_until(mxlRational rate, std::uint64_t index, std::int64_t offset_ns, const std::atomic<bool>& stop);
// Tas pats 1080i50 ritmu (kFieldRate).
bool wait_until(std::uint64_t index, std::int64_t offset_ns, const std::atomic<bool>& stop);
std::uint64_t now_tai();

// Laikrodžio šuolių detektorius: (TAI − monotoninis) pokytis tarp tick'ų > 5 ms = šuolis (WSL2 sinchronizuoja laiką
// su Windows). Tick'e su šuoliu praleisti laukai užskaitomi kaip clock_skips (aplinka), ne late_skips (pajėgumas).
struct ClockWatch {
    std::int64_t last = INT64_MIN;
    std::uint64_t steps = 0;
    std::uint64_t back_ns = 0;  // atgalinių šuolių suma
    // true — nuo paskutinio kvietimo buvo šuolis
    bool stepped();
};

// Plokštumų išdėstymas prieš rastrą: planeCount, fill poslinkis 0, key iškart po fill (poslinkis = fill_bytes), dydžiai ir
// pitch = eilučių ilgiai. Tuščias tekstas — tinka; kitaip klaidos aprašas. Naudoja VideoOut::create ir VideoIn::open.
std::string check_layout(const mxlGrainStorageLayout& l, const vramflow::gfx::Raster& r);

// --- Video išėjimas ---------------------------------------------------------------------------------------------------
struct VideoOut {
    mxlFlowWriter w = nullptr;
    mxlGrainStorageLayout layout{};
    std::vector<mxlGrainStorage> slots;
    std::uint64_t last_committed = 0;
    mxlGrainInfo info{};
    std::uint32_t cur_slot = 0;
    bool open_grain = false;
    std::string id;
    vramflow::gfx::Raster raster = vramflow::gfx::kHd1080i;
    vramflow::VideoFormat fmt = vramflow::kFmt1080i50;  // raster == fmt.raster
    bool existed = false;                      // create: flow jau egzistavo (created == false)
    mxlStatus create_status = MXL_STATUS_OK;  // paskutinis mxlCreateFlowWriter statusas (create klaidai diagnozuoti)
    // Sukuria flow ir patikrina rastro geometriją bei grainRate == 50/1 (visada, R9) (numatytoji — 1080i50 v210).
    // Klaida → tekstas į err, false.
    bool create(mxlInstance inst, const std::string& def, const std::string& options, std::string& err,
                const vramflow::gfx::Raster& r = vramflow::gfx::kHd1080i);
    // Formatui neutralus: tikrinama rastro geometrija ir grainRate == f.grain_rate() (klaidoje — abu dažniai).
    bool create(mxlInstance inst, const std::string& def, const std::string& options, std::string& err,
                const vramflow::VideoFormat& f);
    // Atidaro grain'ą; grąžina device rodyklę arba nullptr (st — MXL statusas).
    std::uint32_t* open(std::uint64_t index, mxlStatus& st);
    mxlStatus commit();
    void cancel();
    void release(mxlInstance inst);
};

// --- Video įėjimas ----------------------------------------------------------------------------------------------------
struct VideoIn {
    mxlFlowReader r = nullptr;
    mxlGrainStorageLayout layout{};
    std::vector<mxlGrainStorage> slots;
    mxlFlowConfigInfo cfg{};
    std::string id;
    bool cuda = false;
    vramflow::gfx::Raster raster = vramflow::gfx::kHd1080i;  // expect arba perskaitytas iš flow aprašo
    vramflow::VideoFormat fmt = vramflow::kFmt1080i50;      // formatui neutralus open: perskaitytas iš flow aprašo
    // Laukia flow iki wait_ms. expect != nullptr → tikrinama prieš tą rastrą (nesutapimas = klaida);
    // nullptr → rastras perskaitomas iš flow aprašo į `raster`. grainRate turi būti 50/1 visada (R9). Klaida → err.
    bool open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop, std::string& err,
              const vramflow::gfx::Raster* expect = nullptr);
    // Formatui neutralus: formatas visada perskaitomas iš flow aprašo į `fmt` (ir `raster`), grainRate turi atitikti
    // fmt.grain_rate(); expect != nullptr — formatas (rastras ir dažnis) turi sutapti, klaidoje — abu dažniai.
    bool open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop, std::string& err,
              const vramflow::VideoFormat* expect);
    // Laukimo politika (planas §P6): p.wait_ms < 0 — neribotai (kas p.log_every_ms eilutė į stderr), 0 — vienas bandymas,
    // > 0 — riba (klaidoje — riba ir `--input-wait-ms`). Formatas tikrinamas kaip formatui neutraliame open.
    bool open(mxlInstance inst, const std::string& flow, const runtime::OpenPolicy& p, const std::atomic<bool>& stop, std::string& err,
              const vramflow::VideoFormat* expect);
    // Neblokuojantis (timeout_ns = 0) arba blokuojantis grain'o gavimas; ptr — slot'o adresas (device arba host).
    mxlStatus get(std::uint64_t index, std::uint64_t timeout_ns, mxlGrainInfo& info, const std::uint32_t*& ptr);
    std::uint64_t head();  // MXL_UNDEFINED_INDEX, jei nežinoma
    void release(mxlInstance inst);

private:
    // Laukia flow ir sukuria reader'į (cfg, layout, slots); formatas netikrinamas.
    bool open_reader(mxlInstance inst, const std::string& flow, const runtime::OpenPolicy& p, const std::atomic<bool>& stop, std::string& err);
};

// --- Audio (host) -----------------------------------------------------------------------------------------------------
struct AudioOut {
    mxlFlowWriter w = nullptr;
    std::uint64_t last_end = 0;
    // grain_rate — video grain'o ritmas: rašymo riba turi talpinti ilgiausią grain'ą (vramflow::max_samples_per_grain).
    bool create(mxlInstance inst, const std::string& def, std::string& err, vramflow::Rational grain_rate = {50, 1});
    // Rašo count mėginių, kurių paskutinis (išskirtinis) indeksas = end. fill(ch, sample_index) → float.
    template <class F>
    mxlStatus write(std::uint64_t end, std::uint32_t count, F&& fill);
    void release(mxlInstance inst);
};

struct AudioIn {
    mxlFlowReader r = nullptr;
    bool open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop, std::string& err);
    bool open(mxlInstance inst, const std::string& flow, const runtime::OpenPolicy& p, const std::atomic<bool>& stop, std::string& err);
    // Nukopijuoja [end-count, end) į l ir r (count float kiekvienam).
    mxlStatus read(std::uint64_t end, std::uint32_t count, float* l, float* r);
    void release(mxlInstance inst);
};

template <class F>
mxlStatus AudioOut::write(std::uint64_t end, std::uint32_t count, F&& fill) {
    mxlMutableWrappedMultiBufferSlice sl{};
    const mxlStatus st = mxlFlowWriterOpenSamples(w, end, count, &sl);
    if (st != MXL_STATUS_OK) return st;
    std::uint64_t s = end - count;
    for (int f = 0; f < 2; ++f) {
        const auto& frag = sl.base.fragments[f];
        const std::size_t n = frag.size / sizeof(float);
        for (std::size_t ch = 0; ch < sl.count; ++ch) {
            auto* p = reinterpret_cast<float*>(static_cast<std::uint8_t*>(frag.pointer) + ch * sl.stride);
            for (std::size_t k = 0; k < n; ++k) p[k] = fill(static_cast<int>(ch), s + k);
        }
        s += n;
    }
    const mxlStatus c = mxlFlowWriterCommitSamples(w);
    if (c == MXL_STATUS_OK) last_end = end;
    return c;
}

}  // namespace mxlv::func
