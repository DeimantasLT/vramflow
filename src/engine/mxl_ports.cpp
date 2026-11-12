// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/mxl_ports.hpp"

#include <mxl/flow.h>
#include <mxl/time.h>

#include <algorithm>
#include <cstring>
#include <ctime>

#include "common/index_math.hpp"
#include "mxlio/slots.hpp"

namespace mxlv::engine {

// --- SystemClock ----------------------------------------------------------------------------------------------------
std::uint64_t SystemClock::now_tai() const { return mxlGetTime(); }

std::int64_t SystemClock::now_mono_raw() const {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

void SystemClock::sleep_until(std::uint64_t tai_ns, const std::atomic<bool>& stop) {
    // Absoliutus miegas (M14) ne ilgesniais kaip 50 ms gabalais: sustabdymas ir laikrodžio žingsnis pastebimi greitai.
    for (;;) {
        const std::uint64_t n = mxlGetTime();
        if (stop.load(std::memory_order_relaxed) || n >= tai_ns) return;
        mxlSleepUntil(std::min<std::uint64_t>(tai_ns, n + 50'000'000));
    }
}

// --- MxlVideoInput --------------------------------------------------------------------------------------------------
bool MxlVideoInput::attach_poll(std::uint64_t now_tai) {
    return slot_.poll(inst_, now_tai) == runtime::InputSlot::State::Attached;
}

bool MxlVideoInput::mismatch(std::string& why) const {
    if (slot_.state() != runtime::InputSlot::State::Mismatch) return false;
    why = slot_.reason();
    return true;
}

Index MxlVideoInput::head() {
    const std::uint64_t h = slot_.head();  // mxlFlowReaderGetRuntimeInfo — bendra atmintis (M8)
    return h == MXL_UNDEFINED_INDEX ? kNoIndex : h;
}

std::uint64_t MxlVideoInput::last_write_tai() {
    if (slot_.state() != runtime::InputSlot::State::Attached) return 0;
    mxlFlowRuntimeInfo rt{};
    if (mxlFlowReaderGetRuntimeInfo(slot_.in().r, &rt) != MXL_STATUS_OK) return 0;
    return rt.lastWriteTime;
}

Probe MxlVideoInput::get(Index j, std::uint64_t timeout_ns, GrainRef& out) {
    mxlGrainInfo info{};
    const std::uint32_t* ptr = nullptr;
    const mxlStatus st = slot_.get(j, timeout_ns, info, ptr);
    out = GrainRef{};
    switch (st) {
        case MXL_STATUS_OK:
            out.ptr = ptr;
            out.flags = info.flags;
            return (info.flags & MXL_GRAIN_FLAG_INVALID) != 0 ? Probe::Invalid : Probe::Ready;
        case MXL_ERR_OUT_OF_RANGE_TOO_EARLY: return Probe::NotYet;  // M1: nesulaukus — TOO_EARLY, ne TIMEOUT
        case MXL_ERR_OUT_OF_RANGE_TOO_LATE: return Probe::TooLate;
        case MXL_ERR_FLOW_INVALID:
        case MXL_ERR_FLOW_NOT_FOUND: return Probe::Gone;
        default: return Probe::Error;
    }
}

Probe MxlVideoInput::probe(Index j, GrainRef& out) { return get(j, 0, out); }

Probe MxlVideoInput::wait(Index j, std::uint64_t timeout_ns, GrainRef& out) { return get(j, timeout_ns, out); }

bool MxlVideoInput::writer_alive() { return slot_.writer_alive(); }

bool MxlVideoInput::lapped(Index j, Index now_idx) {
    if (slot_.state() != runtime::InputSlot::State::Attached) return false;
    const Index h = head();
    const Index after = h == kNoIndex ? now_idx : std::max(h, now_idx);
    return mxlv::lapped(after, j, slot_.in().layout.slotCount);
}

void MxlVideoInput::detach() { slot_.detach_now(); }

// --- MxlVideoOutput -------------------------------------------------------------------------------------------------
GrainCopy host_copy() {
    return [](void* dst, const void* src, std::size_t n) {
        std::memcpy(dst, src, n);
        return true;
    };
}

std::vector<std::uint8_t> black_grain(const vramflow::gfx::Raster& r) {
    namespace gfx = vramflow::gfx;
    std::vector<std::uint8_t> g(gfx::grain_bytes(r), 0);  // key plokštuma (v210a) — 0
    // v210: 4 žodžiai = 6 pikseliai: (Cb Y Cr) (Y Cb Y) (Cr Y Cb) (Y Cr Y); eilutės ilgis — 128 B kartotinis.
    constexpr std::uint32_t kC = 512, kY = 64;
    constexpr std::uint32_t kCyc = kC | (kY << 10) | (kC << 20);
    constexpr std::uint32_t kYcy = kY | (kC << 10) | (kY << 20);
    const std::size_t words = gfx::fill_bytes(r) / 4;
    for (std::size_t w = 0; w < words; ++w) {
        const std::uint32_t v = (w % 2 == 0) ? kCyc : kYcy;
        std::memcpy(g.data() + 4 * w, &v, 4);
    }
    return g;
}

OpenResult MxlVideoOutput::open(Index i, void*& slot) {
    mxlStatus st{};
    std::uint32_t* p = out_.open(i, st);
    if (p == nullptr) return st == MXL_ERR_INVALID_ARG ? OpenResult::AlreadyCommitted : OpenResult::Error;  // M11
    slot = p;
    return OpenResult::Ok;
}

bool MxlVideoOutput::commit() { return out_.commit() == MXL_STATUS_OK; }

void MxlVideoOutput::cancel() { out_.cancel(); }

bool MxlVideoOutput::copy_commit(Index i, const void* src) {
    void* dst = nullptr;
    if (open(i, dst) != OpenResult::Ok) return false;
    if (!copy_(dst, src, vramflow::gfx::grain_bytes(out_.raster)) || (flush_ && !flush_())) {
        out_.cancel();
        return false;
    }
    return commit();
}

bool MxlVideoOutput::fill_gap(Index i, GapAction a, Index repeat_from) {
    switch (a) {
        case GapAction::None: return true;
        case GapAction::Invalid: return out_.commit_invalid(i) == MXL_STATUS_OK;
        case GapAction::Black: return black_ != nullptr && copy_commit(i, black_);
        case GapAction::Repeat: {
            // Šaltinis — ankstesnis to paties ring'o grain'as; jis neperrašytas, kol i − from < slotCount (GapState
            // užtikrina ≤ slotCount − 2).
            const std::uint32_t n = out_.layout.slotCount;
            if (repeat_from == kNoIndex || repeat_from >= i || i - repeat_from >= n) return false;
            const void* src = slot_ptr(out_.layout, out_.slots[slot_of(repeat_from, n)]);
            return copy_commit(i, src);
        }
    }
    return false;
}

}  // namespace mxlv::engine
