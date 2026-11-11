// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Writer'is (architektūra §4.4): deterministinis turinys GPU, commit tik po stream sync, gedimų injekcijos.
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>

#include <cstdio>
#include <string_view>
#include <vector>

#include "common/header.hpp"
#include "common/pattern.hpp"
#include "gpu/kernels.hpp"
#include "mxlio/common_io.hpp"
#include "mxlio/slots.hpp"
#include "mxlio/verify.hpp"

namespace mxlv {
namespace {

bool every(std::uint32_t n, std::uint64_t k) { return n != 0 && k > 0 && k % n == 0; }

int fatal(FlowState& s, const std::string& what, int code = kExitFatal) {
    if (s.fatal_code == 0) {
        s.fatal = what;
        s.fatal_code = code;
    }
    s.exit_code = s.fatal_code;
    std::fprintf(stderr, "[writer %s] %s\n", s.id.c_str(), what.c_str());
    return s.exit_code;
}

// Visi writer'io resursai — atlaisvinami bet kuriuo išėjimo keliu (writer handle'o nuotėkis laikytų flow gyvą).
struct Resources {
    mxlInstance inst = nullptr;
    mxlFlowWriter w = nullptr;
    cudaStream_t stream{};
    std::uint32_t* scratch = nullptr;
    std::uint64_t* d_fold = nullptr;
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (d_fold) cudaFree(d_fold);
        if (scratch) cudaFree(scratch);
        if (stream) cudaStreamDestroy(stream);
        if (w) mxlReleaseFlowWriter(inst, w);
        if (inst) mxlDestroyInstance(inst);
    }
};

// Laukia iki indekso laiko dalimis po ≤ 50 ms. Grąžina true, jei laukti teko ilgiau nei 2 kadrus
// (laikrodis šoko atgal — WSL2 sinchronizuoja laiką su Windows).
bool wait_until_index(std::uint64_t index, const mxlRational& rate, const std::atomic<bool>& stop) {
    std::uint64_t wait = mxlGetNsUntilIndex(index, &rate);
    const auto period_ns = static_cast<std::uint64_t>(1e9 * static_cast<double>(rate.denominator) / static_cast<double>(rate.numerator));
    const bool stepped = wait > 2 * period_ns;
    while (wait > 0 && !stop.load()) {
        mxlSleepForNs(wait < 50'000'000ull ? wait : 50'000'000ull);
        wait = mxlGetNsUntilIndex(index, &rate);
    }
    return stepped;
}

}  // namespace

int run_writer_flow(const Options& o, std::uint32_t n, FlowState& s, int cuda_dev, const std::atomic<bool>& stop,
                    std::uint64_t end_tai) {
    Resources r;
    if (cudaSetDevice(cuda_dev) != cudaSuccess) return fatal(s, "cudaSetDevice");
    r.inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!r.inst) return fatal(s, "mxlCreateInstance grąžino NULL");
    if (o.gc_before_create && mxlGarbageCollectFlows(r.inst) != MXL_STATUS_OK)
        std::fprintf(stderr, "[writer] mxlGarbageCollectFlows nepavyko (tęsiu)\n");

    const std::string def = flow_def_json(o, n);
    const std::string opts = writer_options(o, cuda_uuid_string(cuda_dev));
    mxlFlowConfigInfo cfg{};
    bool created = false;
    mxlStatus st = mxlCreateFlowWriter(r.inst, def.c_str(), opts.empty() ? nullptr : opts.c_str(), &r.w, &cfg, &created);
    if (st != MXL_STATUS_OK) {
        r.w = nullptr;
        if (st == MXL_ERR_CONFLICT) return fatal(s, "mxlCreateFlowWriter: MXL_ERR_CONFLICT", kExitConflict);
        return fatal(s, "mxlCreateFlowWriter status " + std::to_string(st) +
                            (st == MXL_ERR_UNSUPPORTED_OPERATION ? " (plugin nerastas arba FD eksportas nepalaikomas)" : ""));
    }
    if (!created) return fatal(s, "flow jau egzistavo (created=false) — paleisk su --gc-before-create");

    mxlGrainStorageLayout layout{};
    if ((st = mxlFlowWriterGetStorageLayout(r.w, &layout)) != MXL_STATUS_OK) return fatal(s, "GetStorageLayout " + std::to_string(st));
    std::vector<mxlGrainStorage> slots(layout.slotCount);
    if ((st = mxlFlowWriterMapSlots(r.w, layout.slotCount, slots.data())) != MXL_STATUS_OK)
        return fatal(s, "MapSlots " + std::to_string(st));
    const bool cuda = is_cuda_layout(layout);
    if (!cuda && !is_host_layout(layout)) return fatal(s, "nežinomas storage tipas " + storage_name(layout));
    if (layout.planeCount != 1 || layout.slotCount != cfg.discrete.grainCount)
        return fatal(s, "netikėtas layout (planeCount/slotCount)", kExitVerify);

    const std::uint32_t bytes = static_cast<std::uint32_t>(layout.planes[0].size);
    s.n = bytes / 4;
    s.slot_count = layout.slotCount;
    s.grain_size = bytes;
    s.layout_storage = storage_name(layout);
    if (bytes != v210_grain_bytes(o.width, o.height)) return fatal(s, "grain dydis " + std::to_string(bytes) + " != v210", kExitVerify);

    // Layout sanity (W7): rodyklės ne NULL ir (cuda) device atmintyje
    for (std::uint32_t k = 0; k < layout.slotCount; ++k) {
        void* p = slot_ptr(layout, slots[k]);
        if (!p) return fatal(s, "slot rodyklė NULL", kExitVerify);
        if (cuda) {
            cudaPointerAttributes a{};
            if (cudaPointerGetAttributes(&a, p) != cudaSuccess || a.type != cudaMemoryTypeDevice)
                return fatal(s, "slot ne device atmintyje", kExitVerify);
        }
    }

    if (cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&r.d_fold), 8) != cudaSuccess ||
        (!cuda && cudaMalloc(reinterpret_cast<void**>(&r.scratch), bytes) != cudaSuccess))
        return fatal(s, "CUDA resursai");
    cudaStream_t stream = r.stream;
    std::uint64_t* d_fold = r.d_fold;
    std::uint32_t* scratch = r.scratch;

    const mxlRational rate = cfg.common.grainRate;
    std::uint64_t index = mxlGetCurrentIndex(&rate);
    if (index == MXL_UNDEFINED_INDEX) return fatal(s, "mxlGetCurrentIndex");
    std::uint64_t k = 0, seq = 0, last_committed = 0;
    const std::uint32_t pid = static_cast<std::uint32_t>(getpid());
    const Injections& inj = o.inj;
    const std::uint64_t warm_tai = now_tai_ns() + static_cast<std::uint64_t>(o.warmup_s * 1e9);
    auto is = [](const char* kind, std::string_view v) { return kind && v == kind; };

    while (!stop.load() && now_tai_ns() < end_tai) {
        mxlGrainInfo info{};
        std::uint32_t slot = 0;
        const bool skip_now = every(inj.skip_every, k + 1);
        const std::uint64_t target = skip_now ? index + 1 : index;  // skip: indeksas praleidžiamas visai
        st = mxlFlowWriterOpenGrainSlot(r.w, target, &info, &slot);
        if (st == MXL_ERR_INVALID_ARG && target <= last_committed) {
            // Laikrodis šoko atgal: tęsiam nuo last+1, bet tik sulaukę jo laiko; k nedidinam (injekcijų grafikas nepasislenka)
            ++s.c.clock_backward;
            index = last_committed + 1;
            wait_until_index(index, rate, stop);
            continue;
        }
        if (st != MXL_STATUS_OK) { fatal(s, "OpenGrainSlot " + std::to_string(st)); break; }
        ++k;
        if (skip_now) {
            std::lock_guard g(s.rec_m);
            s.injected.push_back(index);
            s.injected_kind.push_back("skip");
            ++s.c.injections;
        }
        index = target;
        if (slot >= layout.slotCount || info.index != index || info.grainSize != bytes) {
            mxlFlowWriterCancelGrain(r.w);
            fatal(s, "OpenGrainSlot grąžino neatitinkantį slot/index/size", kExitVerify);
            break;
        }

        const char* kind = nullptr;
        if (every(inj.corrupt_every, k)) kind = "corrupt";
        else if (every(inj.corrupt_header_every, k)) kind = "corrupt_header";
        else if (every(inj.wrong_index_every, k)) kind = "wrong_index";
        else if (every(inj.stale_every, k)) kind = "stale";
        else if (every(inj.commit_before_sync_every, k)) kind = "commit_before_sync";
        else if (every(inj.dup_every, k)) kind = "dup";

        auto* dst = static_cast<std::uint32_t*>(slot_ptr(layout, slots[slot]));
        std::uint32_t* gpu_dst = cuda ? dst : scratch;
        const std::uint64_t content_index = is(kind, "wrong_index") ? index - 1 : index;
        GrainHeader h{};
        h.magic = kMagic;
        h.version = kHeaderVersion;
        h.header_size = sizeof(GrainHeader);
        h.flow_seed = s.seed;
        h.grain_index = content_index;
        h.payload_words = s.n;
        h.writer_pid = pid;
        h.seq = (is(kind, "dup") && seq > 0) ? seq - 1 : seq;

        const std::uint64_t t0 = now_tai_ns();
        h.origin_tai_ns = t0;
        cudaError_t ce = cudaSuccess;
        bool committed_early = false;
        if (is(kind, "stale")) {
            // nieko nerašom — slot'e lieka turinys iš index - slotCount
        } else if (is(kind, "commit_before_sync")) {
            ce = cudaMemsetAsync(d_fold, 0, 8, stream);
            if (!ce) ce = gpu::fill(gpu_dst, s.seed, content_index, s.n, 0, s.n / 2, d_fold, stream);
            if (!ce) ce = cudaStreamSynchronize(stream);
            if (!cuda && !ce) ce = cudaMemcpy(dst, scratch, bytes, cudaMemcpyDeviceToHost);
            info.validSlices = info.totalSlices;
            if (!ce) {
                if ((st = mxlFlowWriterCommitGrain(r.w, &info)) != MXL_STATUS_OK) { fatal(s, "CommitGrain " + std::to_string(st)); break; }
                committed_early = true;
                s.add_committed(index);
                usleep(inj.late_write_ms * 1000);
                ce = gpu::fill(gpu_dst, s.seed, content_index, s.n, s.n / 2, s.n, d_fold, stream);
                if (!ce) ce = gpu::finalize_header(gpu_dst, h, d_fold, stream);
                if (!ce) ce = cudaStreamSynchronize(stream);
                if (!cuda && !ce) ce = cudaMemcpy(dst, scratch, bytes, cudaMemcpyDeviceToHost);
            }
        } else {
            ce = cudaMemsetAsync(d_fold, 0, 8, stream);
            if (!ce) ce = gpu::fill(gpu_dst, s.seed, content_index, s.n, 0, s.n, d_fold, stream);
            if (!ce) ce = gpu::finalize_header(gpu_dst, h, d_fold, stream);
            if (!ce && is(kind, "corrupt")) {
                const auto pos = static_cast<std::uint32_t>(kHeaderWords + (index * 2654435761ull) % (s.n - kHeaderWords));
                ce = gpu::corrupt(gpu_dst, pos, 1u << (index % 30), stream);
            }
            if (!ce && is(kind, "corrupt_header")) ce = gpu::corrupt(gpu_dst, 2, 1u, stream);
            if (!ce) ce = cudaStreamSynchronize(stream);
            if (!cuda && !ce) ce = cudaMemcpy(dst, scratch, bytes, cudaMemcpyDeviceToHost);
        }
        if (ce != cudaSuccess) {
            ++s.c.cuda_errors;
            if (!committed_early) mxlFlowWriterCancelGrain(r.w);
            fatal(s, std::string("CUDA: ") + cudaGetErrorString(ce));
            break;
        }
        const std::uint64_t t1 = now_tai_ns();
        if (!committed_early) {
            info.validSlices = info.totalSlices;
            if ((st = mxlFlowWriterCommitGrain(r.w, &info)) != MXL_STATUS_OK) { fatal(s, "CommitGrain " + std::to_string(st)); break; }
            s.add_committed(index);
        }
        const std::uint64_t t_commit = now_tai_ns();
        if (kind) {
            std::lock_guard g(s.rec_m);
            s.injected.push_back(index);
            s.injected_kind.push_back(kind);
            ++s.c.injections;
        }
        ++s.c.commits;
        s.c.head_index = index;
        s.c.cur_index = index;
        last_committed = index;
        ++seq;
        if (t_commit >= warm_tai) {
            const std::uint64_t ideal = mxlIndexToTimestamp(&rate, index);
            std::lock_guard g(s.lat.m);
            s.lat.sec_gpu.add(t1 - t0);
            s.lat.tot_gpu.add(t1 - t0);
            const std::uint64_t jit = t_commit > ideal ? t_commit - ideal : ideal - t_commit;
            s.lat.sec_jitter.add(jit);
            s.lat.tot_jitter.add(jit);
        }

        ++index;
        const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
        if (now_idx > index) {
            s.c.writer_late_skips += now_idx - index;
            index = now_idx;
        }
        // NIEKADA nerašom anksčiau už indekso laiką (kitaip writer'is aplenktų realų laiką po laikrodžio šuolio)
        if (wait_until_index(index, rate, stop)) ++s.c.clock_backward;
    }
    if (!s.exit_code) s.exit_code = s.fatal_code;
    return s.exit_code;
}

}  // namespace mxlv
