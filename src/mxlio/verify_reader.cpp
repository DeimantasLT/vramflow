// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Reader'is (architektūra §4.5–4.6): kiekvieno grain'o pilna patikra GPU, indekso/slot'o/antraštės patikros,
// lap patikra po verify, CPU kryžminė patikra kas K-tą grain'ą, atsilikimo politika pagal MXL statusą.
//
// Aprėpties principas: kiekvienas indeksas tarp pirmo ir paskutinio įvertinto yra ARBA patikrintas
// (ok / verifikacijos klaida), ARBA užskaitytas kaip pajėgumo įvykis (gap / invalid / lapped).
// Nė vienas indeksas negali dingti tyliai; reader'is tai tikrina pabaigoje (coverage).
#include <cuda_runtime_api.h>

#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "common/header.hpp"
#include "common/index_math.hpp"
#include "common/pattern.hpp"
#include "common/pattern_ref.hpp"
#include "gpu/kernels.hpp"
#include "mxlio/common_io.hpp"
#include "mxlio/slots.hpp"
#include "mxlio/verify.hpp"

namespace mxlv {
namespace {

// Užfiksuoja PIRMĄ fatalią klaidą. Galutinį exit kodą skaičiuoja finalize_exit() — verifikacijos klaidos
// visada svarbesnės, todėl vėlesnė fatali klaida jų neužmaskuoja.
int fatal(FlowState& s, const std::string& what, int code = kExitFatal) {
    if (s.fatal_code == 0) {
        s.fatal = what;
        s.fatal_code = code;
    }
    std::fprintf(stderr, "[reader %s] %s\n", s.id.c_str(), what.c_str());
    return code;
}

struct Session {
    mxlFlowReader r = nullptr;
    mxlFlowConfigInfo cfg{};
    mxlGrainStorageLayout layout{};
    std::vector<mxlGrainStorage> slots;
    bool cuda = false;
};

// RD2–RD6. Grąžina 0 arba klaidos kodą.
int open_session(mxlInstance inst, const std::string& id, FlowState& s, Session& ss, std::uint32_t wait_ms,
                 const std::atomic<bool>& stop) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
    mxlStatus st = MXL_ERR_UNKNOWN;
    for (;;) {
        st = mxlCreateFlowReader(inst, id.c_str(), "", &ss.r);
        if (st == MXL_STATUS_OK) {
            if (mxlFlowReaderGetConfigInfo(ss.r, &ss.cfg) == MXL_STATUS_OK &&
                mxlFlowReaderGetStorageLayout(ss.r, &ss.layout) == MXL_STATUS_OK) {
                ss.slots.assign(ss.layout.slotCount, mxlGrainStorage{});
                st = mxlFlowReaderMapSlots(ss.r, ss.layout.slotCount, ss.slots.data());
                if (st == MXL_STATUS_OK && mxlFlowReaderGetStorageLayout(ss.r, &ss.layout) == MXL_STATUS_OK) break;
            }
            mxlReleaseFlowReader(inst, ss.r);
            ss.r = nullptr;
            // NOT_FOUND map'e = niekas neklauso payload.sock (writer'is dar/jau nebėra) → kartoti
            if (st != MXL_ERR_NOT_FOUND && st != MXL_STATUS_OK) return fatal(s, "reader map/layout status " + std::to_string(st));
        } else if (st != MXL_ERR_FLOW_NOT_FOUND) {
            return fatal(s, "mxlCreateFlowReader status " + std::to_string(st));
        }
        if (stop.load() || std::chrono::steady_clock::now() > deadline)
            return fatal(s, "flow neatsirado/neatsidarė per " + std::to_string(wait_ms) + " ms (status " + std::to_string(st) + ")",
                         kExitCapacity);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ss.cuda = is_cuda_layout(ss.layout);
    if (!ss.cuda && !is_host_layout(ss.layout)) return fatal(s, "nežinomas storage " + storage_name(ss.layout));
    if (ss.cuda && mxlCudaLinearLayout(ss.layout).deviceIndex < 0) return fatal(s, "GPU su flow UUID šiam procesui nematomas");
    if (ss.layout.planeCount != 1 || ss.layout.slotCount < 2) return fatal(s, "netikėtas layout", kExitVerify);
    s.layout_storage = storage_name(ss.layout);
    s.slot_count = ss.layout.slotCount;
    s.grain_size = static_cast<std::uint32_t>(ss.layout.planes[0].size);
    s.n = s.grain_size / 4;
    return 0;
}

void close_session(mxlInstance inst, Session& ss) {
    if (ss.r) mxlReleaseFlowReader(inst, ss.r);
    ss = Session{};
}

// GetRuntimeInfo su statuso patikra. headIndex == UNDEFINED po pirmo commit'o laikomas klaida.
bool runtime(Session& ss, mxlFlowRuntimeInfo& rt) {
    return mxlFlowReaderGetRuntimeInfo(ss.r, &rt) == MXL_STATUS_OK;
}

// Visi reader'io resursai vienoje vietoje — atlaisvinami bet kuriuo išėjimo keliu.
struct Resources {
    mxlInstance inst = nullptr;
    Session ss;
    cudaStream_t stream{};
    gpu::VerifyResult* d_res = nullptr;
    gpu::VerifyResult* h_res = nullptr;
    bool h_res_pinned = false;
    std::uint32_t* scratch = nullptr;  // tik host storage (H2D tikslas)
    std::uint32_t scratch_bytes = 0;
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (d_res) cudaFree(d_res);
        if (scratch) cudaFree(scratch);
        if (h_res) {
            if (h_res_pinned) cudaFreeHost(h_res);
            else delete h_res;
        }
        if (stream) cudaStreamDestroy(stream);
        if (inst) {
            close_session(inst, ss);
            mxlDestroyInstance(inst);
        }
    }
    bool ensure_scratch(std::uint32_t bytes) {
        if (ss.cuda || bytes <= scratch_bytes) return true;
        if (scratch) cudaFree(scratch);
        scratch = nullptr;
        scratch_bytes = 0;
        if (cudaMalloc(reinterpret_cast<void**>(&scratch), bytes) != cudaSuccess) return false;
        scratch_bytes = bytes;
        return true;
    }
};

int finalize_exit(FlowState& s, double min_coverage) {
    // Aprėptis: įvertinti indeksai (ok + verifikacijos klaidos) / visi indeksai lange
    const std::uint64_t span = s.first_index == ~0ull ? 0 : s.last_index - s.first_index + 1;
    std::uint64_t verify_grains = 0;
    {
        std::lock_guard g(s.rec_m);
        for (const auto& e : s.errors)
            if (e.bits & kVerifyBits) ++verify_grains;
    }
    const std::uint64_t evaluated = s.c.grains_ok + verify_grains;
    s.coverage = span ? static_cast<double>(evaluated) / static_cast<double>(span) : 0.0;
    if (span == 0 || s.coverage < min_coverage) s.c.coverage_low = 1;

    if (s.c.verification_errors()) s.exit_code = kExitVerify;
    else if (s.fatal_code) s.exit_code = s.fatal_code;
    else if (s.c.capacity_errors()) s.exit_code = kExitCapacity;
    else s.exit_code = kExitOk;
    return s.exit_code;
}

}  // namespace

int run_reader_flow(const Options& o, std::uint32_t n, FlowState& s, int cuda_dev, const std::atomic<bool>& stop,
                    std::uint64_t end_tai) {
    (void)n;
    Resources res;
    if (cudaSetDevice(cuda_dev) != cudaSuccess) {
        fatal(s, "cudaSetDevice");
        return finalize_exit(s, o.min_coverage);
    }
    res.inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!res.inst) {
        fatal(s, "mxlCreateInstance grąžino NULL");
        return finalize_exit(s, o.min_coverage);
    }
    Session& ss = res.ss;
    if (open_session(res.inst, s.id, s, ss, o.wait_flow_ms, stop)) return finalize_exit(s, o.min_coverage);
    if (ss.cuda && cudaSetDevice(mxlCudaLinearLayout(ss.layout).deviceIndex) != cudaSuccess) {
        fatal(s, "cudaSetDevice(layout)");
        return finalize_exit(s, o.min_coverage);
    }
    if (cudaStreamCreateWithFlags(&res.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&res.d_res), sizeof(gpu::VerifyResult)) != cudaSuccess ||
        !res.ensure_scratch(s.grain_size)) {
        fatal(s, "CUDA resursai");
        return finalize_exit(s, o.min_coverage);
    }
    if (cudaMallocHost(reinterpret_cast<void**>(&res.h_res), sizeof(gpu::VerifyResult)) == cudaSuccess) {
        res.h_res_pinned = true;
    } else {
        (void)cudaGetLastError();
        res.h_res = new gpu::VerifyResult{};
    }
    gpu::VerifyResult* const h_res = res.h_res;
    std::vector<std::uint32_t> host_copy;

    const mxlRational rate = ss.cfg.common.grainRate;
    std::uint32_t grain_count = ss.layout.slotCount;
    const std::uint64_t timeout_ns = std::uint64_t{o.read_timeout_ms} * 1'000'000ull;
    const std::uint64_t warm_tai = now_tai_ns() + static_cast<std::uint64_t>(o.warmup_s * 1e9);

    mxlFlowRuntimeInfo rt{};
    if (!runtime(ss, rt)) {
        fatal(s, "GetRuntimeInfo");
        return finalize_exit(s, o.min_coverage);
    }
    std::uint64_t index = (rt.headIndex != MXL_UNDEFINED_INDEX && rt.headIndex != 0) ? rt.headIndex : mxlGetCurrentIndex(&rate);
    std::uint64_t prev_seq = ~0ull, prev_verified = 0;
    bool have_prev = false;
    auto wait_start = std::chrono::steady_clock::now();

    // Indeksų intervalas [from, to) užskaitomas kaip nepatikrintas (pajėgumo įvykis), jei jis lange.
    auto count_gap = [&](std::uint64_t from, std::uint64_t to) {
        if (to <= from) return;
        if (s.first_index == ~0ull) s.first_index = from;
        s.c.gap_grains += to - from;
        s.record(from, kErrGap);
        s.last_index = std::max(s.last_index, to - 1);
    };

    while (!stop.load() && now_tai_ns() < end_tai) {
        mxlGrainInfo info{};
        std::uint32_t slot = 0;
        mxlStatus st = mxlFlowReaderGetGrainSlot(ss.r, index, MXL_GRAIN_VALID_SLICES_ALL, timeout_ns, &info, &slot);
        const std::uint64_t t_wake = now_tai_ns();

        if (st == MXL_ERR_OUT_OF_RANGE_TOO_EARLY) {
            ++s.c.too_early_timeouts;
            if (std::chrono::steady_clock::now() - wait_start > std::chrono::milliseconds(o.stall_timeout_ms)) {
                ++s.c.stalls;
                if (runtime(ss, rt) && rt.headIndex != MXL_UNDEFINED_INDEX && rt.headIndex > index) {
                    count_gap(index, rt.headIndex);
                    index = rt.headIndex;
                }
                wait_start = std::chrono::steady_clock::now();
            }
            continue;
        }
        if (st == MXL_ERR_OUT_OF_RANGE_TOO_LATE) {
            ++s.c.too_late_events;
            if (!runtime(ss, rt) || rt.headIndex == MXL_UNDEFINED_INDEX) { fatal(s, "GetRuntimeInfo po TOO_LATE"); break; }
            // Pradėti nuo seniausio dar skaitomo (head - (grainCount-2)), kad prarastume kuo mažiau
            const std::uint64_t oldest = rt.headIndex >= grain_count - 2 ? rt.headIndex - (grain_count - 2) : rt.headIndex;
            const std::uint64_t target = std::max(oldest, index + 1);
            count_gap(index, target);
            index = target;
            wait_start = std::chrono::steady_clock::now();
            continue;
        }
        if (st == MXL_ERR_FLOW_INVALID) {
            ++s.c.reopens;
            cudaStreamSynchronize(res.stream);
            close_session(res.inst, ss);
            if (open_session(res.inst, s.id, s, ss, o.reopen_timeout_ms, stop)) break;
            grain_count = ss.layout.slotCount;  // naujas flow gali turėti kitą geometriją
            if (!res.ensure_scratch(s.grain_size)) { fatal(s, "CUDA scratch po reopen"); break; }
            if (!runtime(ss, rt)) { fatal(s, "GetRuntimeInfo po reopen"); break; }
            const std::uint64_t new_index = rt.headIndex != MXL_UNDEFINED_INDEX ? rt.headIndex : mxlGetCurrentIndex(&rate);
            if (s.first_index != ~0ull && new_index > index) count_gap(index, new_index);  // praleisti per reopen
            index = new_index;
            have_prev = false;
            wait_start = std::chrono::steady_clock::now();
            continue;
        }
        if (st != MXL_STATUS_OK) { fatal(s, "GetGrainSlot status " + std::to_string(st)); break; }
        wait_start = std::chrono::steady_clock::now();

        if (s.first_index == ~0ull) s.first_index = index;
        s.last_index = std::max(s.last_index, index);
        s.c.cur_index = index;

        if (info.flags & MXL_GRAIN_FLAG_INVALID) {
            // INVALID gali reikšti (a) writer'is indekso nerašė (praleido) arba (b) writer'is jau atidarė tą patį
            // slot'ą vėlesniam indeksui (index + grainCount) — tai lapping, ne duomenų klaida. Atskiriam pagal
            // max(head, laiko indeksas); atsilikimą įrašom (first_bad_word lauke) diagnostikai.
            mxlFlowRuntimeInfo rti{};
            const std::uint64_t h = runtime(ss, rti) && rti.headIndex != MXL_UNDEFINED_INDEX ? rti.headIndex : 0;
            const std::uint64_t bound = std::max(h, mxlGetCurrentIndex(&rate));
            const std::uint32_t lag32 = bound > index ? static_cast<std::uint32_t>(std::min<std::uint64_t>(bound - index, 0xFFFFFFFEu)) : 0;
            ++s.c.grains_invalid;
            if (lapped(bound, index, grain_count) || bound >= index + grain_count - 1) {
                ++s.c.lapped_discarded;
                s.record(index, kErrInvalidFlag | kErrLapped, lag32);
            } else {
                ++s.c.gap_grains;
                s.record(index, kErrInvalidFlag | kErrGap, lag32);
            }
            ++index;
            continue;
        }

        std::uint32_t bits = 0;
        if (info.index != index) { ++s.c.index_mismatch; bits |= kErrIndex; }
        if (slot >= grain_count || slot != slot_of(index, grain_count)) { ++s.c.slot_mismatch; bits |= kErrSlot; }
        if (info.grainSize != s.grain_size || info.validSlices != info.totalSlices) { ++s.c.size_mismatch; bits |= kErrSize; }
        if (slot >= grain_count) {
            s.record(index, bits);
            ++index;
            continue;
        }

        // RD9: tikslus commit laikas, jei head == index (latencijos imtis; kitaip — atsilikimas, matuojamas atskirai)
        if (!runtime(ss, rt)) { fatal(s, "GetRuntimeInfo"); break; }
        const bool exact_commit = rt.headIndex == index;
        const std::uint64_t t_commit = rt.lastWriteTime;
        const std::uint64_t lag = rt.headIndex != MXL_UNDEFINED_INDEX && rt.headIndex > index ? rt.headIndex - index : 0;

        // RD10: patikra GPU
        const void* sp = slot_ptr(ss.layout, ss.slots[slot]);
        const std::uint32_t* dev = static_cast<const std::uint32_t*>(sp);
        cudaError_t ce = cudaSuccess;
        if (!ss.cuda) {
            ce = cudaMemcpyAsync(res.scratch, sp, s.grain_size, cudaMemcpyHostToDevice, res.stream);
            dev = res.scratch;
        }
        if (!ce) ce = gpu::verify(dev, s.seed, index, s.n, res.d_res, res.stream);
        if (!ce) ce = cudaMemcpyAsync(h_res, res.d_res, sizeof(gpu::VerifyResult), cudaMemcpyDeviceToHost, res.stream);
        if (!ce) ce = cudaStreamSynchronize(res.stream);
        if (ce != cudaSuccess) {
            ++s.c.cuda_errors;
            fatal(s, std::string("CUDA: ") + cudaGetErrorString(ce));
            break;
        }
        const std::uint64_t t_ver = now_tai_ns();  // GPU patikra baigta (be CPU kryžminės patikros)

        // CPU kryžminė patikra: NEPRIKLAUSOMA skaliarinė rašto implementacija (pattern_ref.hpp) ir fold64
        bool cross_fail = false;
        if (o.host_crosscheck_every && (index % o.host_crosscheck_every) == 0) {
            host_copy.resize(s.n);
            if (ss.cuda) ce = cudaMemcpy(host_copy.data(), sp, s.grain_size, cudaMemcpyDeviceToHost);
            else std::memcpy(host_copy.data(), sp, s.grain_size);
            if (ce != cudaSuccess) {
                ++s.c.cuda_errors;
                fatal(s, std::string("CUDA kryžminės patikros D2H: ") + cudaGetErrorString(ce));
                break;
            }
            std::uint64_t bad = 0;
            for (std::uint32_t i = kHeaderWords; i < s.n; ++i) bad += host_copy[i] != ref::word(s.seed, index, i);
            const std::uint64_t fold = ref::fold64_of(host_copy.data(), s.n);
            ++s.c.host_crosschecks;
            cross_fail = (bad != h_res->mismatch_words) || (fold != h_res->actual_fold64);
        }

        // RD11: lap patikra. Writer'is slot'ą perrašo ATIDARĘS index+grainCount, kai head dar to nerodo,
        // bet niekada nerašo indekso, vėlesnio už dabartinį laiko indeksą — todėl imam max(head, laikas).
        if (!runtime(ss, rt)) { fatal(s, "GetRuntimeInfo (lap)"); break; }
        const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
        const std::uint64_t head_bound = std::max(rt.headIndex == MXL_UNDEFINED_INDEX ? 0 : rt.headIndex, now_idx);
        if (lapped(head_bound, index, grain_count)) {
            ++s.c.lapped_discarded;
            s.record(index, kErrLapped);
            ++index;
            continue;
        }

        // RD12: vertinimas
        const GrainHeader h = header_from_bytes(h_res->header);
        if (validate(h, s.seed, s.n) != HdrError::Ok) { ++s.c.hdr_fail; bits |= kErrHdr; }
        else {
            if (h.grain_index != index) { ++s.c.content_index_mismatch; bits |= kErrContentIndex; }
            if (h_res->actual_fold64 != h.payload_fold64) { ++s.c.checksum_fail; bits |= kErrChecksum; }
            if (have_prev && (h.seq == prev_seq || h.grain_index <= prev_verified)) { ++s.c.dup; bits |= kErrDup; }
            if (have_prev && h.seq > prev_seq && h.seq - prev_seq != index - prev_verified) ++s.c.seq_break;
            prev_seq = h.seq;
            prev_verified = h.grain_index;
            have_prev = true;
        }
        if (h_res->mismatch_words) {
            ++s.c.mismatch_grains;
            s.c.mismatch_words += h_res->mismatch_words;
            bits |= kErrMismatch;
        }
        if (cross_fail) { ++s.c.host_crosscheck_fail; bits |= kErrCrosscheck; }
        if (bits) s.record(index, bits, h_res->first_bad_word);
        else ++s.c.grains_ok;

        if (t_ver >= warm_tai) {
            std::lock_guard g(s.lat.m);
            if (exact_commit && t_ver > t_commit) {
                s.lat.sec_c2v.add(t_ver - t_commit);
                s.lat.tot_c2v.add(t_ver - t_commit);
                if (t_wake > t_commit) s.lat.tot_wake.add(t_wake - t_commit);
            }
            s.lat.tot_lag.add(lag);  // kiek grain'ų reader'is atsilieka (atskleidžia c2v imties šališkumą)
            if (!(bits & kErrHdr) && t_ver > h.origin_tai_ns) {
                s.lat.sec_e2e.add(t_ver - h.origin_tai_ns);
                s.lat.tot_e2e.add(t_ver - h.origin_tai_ns);
            }
            s.lat.sec_gpu.add(t_ver - t_wake);
            s.lat.tot_gpu.add(t_ver - t_wake);
            const std::uint64_t ideal = mxlIndexToTimestamp(&rate, index);
            if (t_ver > ideal && t_ver - ideal > s.lat.stale_max_ns) s.lat.stale_max_ns = t_ver - ideal;
        }
        s.c.head_index = rt.headIndex;
        ++index;
    }
    return finalize_exit(s, o.min_coverage);
}

}  // namespace mxlv
