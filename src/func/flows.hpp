// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// MXL flow pagalbininkai GPU media funkcijoms (architektūra GPU funkcijų §2): 1080i50 v210 video (cuda-linear,
// laukas = grain, lyginis indeksas = viršutinis laukas) ir 48 kHz stereo float32 audio (host atmintis).
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace mxlv::func {

// Flow rolės identifikatoriuose: 4d584c47-00RR-4000-8000-%012x
enum Role : int { kRoleGen = 1, kRoleLogo = 2, kRoleAudio = 3, kRoleDec = 4 };

std::string flow_id(int role, std::uint32_t n);
std::string video_def(int role, std::uint32_t n, std::uint64_t run_seed);
std::string audio_def(std::uint32_t n, std::uint64_t run_seed);

constexpr mxlRational kFieldRate{50, 1};
constexpr std::uint32_t kAudioPerField = 960;  // 48000 / 50

// Laukia iki indekso laiko + offset (niekada ne anksčiau; laikrodžio šuolis atgal → laukiam). true — buvo šuolis.
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
    // Sukuria flow ir patikrina 1080i50 geometriją. Klaida → tekstas į err, false.
    bool create(mxlInstance inst, const std::string& def, const std::string& options, std::string& err);
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
    // Laukia flow iki wait_ms. Klaida → err.
    bool open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop, std::string& err);
    // Neblokuojantis (timeout_ns = 0) arba blokuojantis grain'o gavimas; ptr — slot'o adresas (device arba host).
    mxlStatus get(std::uint64_t index, std::uint64_t timeout_ns, mxlGrainInfo& info, const std::uint32_t*& ptr);
    std::uint64_t head();  // MXL_UNDEFINED_INDEX, jei nežinoma
    void release(mxlInstance inst);
};

// --- Audio (host) -----------------------------------------------------------------------------------------------------
struct AudioOut {
    mxlFlowWriter w = nullptr;
    std::uint64_t last_end = 0;
    bool create(mxlInstance inst, const std::string& def, std::string& err);
    // Rašo count mėginių, kurių paskutinis (išskirtinis) indeksas = end. fill(ch, sample_index) → float.
    template <class F>
    mxlStatus write(std::uint64_t end, std::uint32_t count, F&& fill);
    void release(mxlInstance inst);
};

struct AudioIn {
    mxlFlowReader r = nullptr;
    bool open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop, std::string& err);
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
