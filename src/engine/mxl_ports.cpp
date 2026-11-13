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

// --- MxlAudioInput --------------------------------------------------------------------------------------------------
MxlAudioInput::MxlAudioInput(mxlInstance inst, std::string flow_id, AudioFrames frames, mxlFlowReader adopt)
    : inst_(inst), id_(std::move(flow_id)), fr_(frames), rd_(adopt) {
    // didžiausias kadras mėginiais (29,97: 1602) — buferiai paruošiami iš anksto (tick'e nealokuojama)
    const std::uint32_t cap = vramflow::max_samples_per_grain(fr_.frame_rate) + 1;
    lb_.resize(cap);
    rb_.resize(cap);
}

MxlAudioInput::MxlAudioInput(MxlAudioInput&& o) noexcept
    : inst_(o.inst_), id_(std::move(o.id_)), fr_(o.fr_), rd_(o.rd_), lb_(std::move(o.lb_)), rb_(std::move(o.rb_)), last_(o.last_),
      n_(o.n_), attaches_(o.attaches_) {
    o.rd_ = nullptr;
}

MxlAudioInput::~MxlAudioInput() { detach(); }

bool MxlAudioInput::attach_poll(std::uint64_t) {
    if (rd_ != nullptr) return true;
    bool active = false;
    if (mxlIsFlowActive(inst_, id_.c_str(), &active) != MXL_STATUS_OK || !active) return false;  // tik gyvas rašytojas
    if (mxlCreateFlowReader(inst_, id_.c_str(), "", &rd_) != MXL_STATUS_OK) {
        rd_ = nullptr;
        return false;
    }
    ++attaches_;
    return true;
}

Index MxlAudioInput::head() {
    if (rd_ == nullptr) return kNoIndex;
    mxlFlowRuntimeInfo rt{};
    if (mxlFlowReaderGetRuntimeInfo(rd_, &rt) != MXL_STATUS_OK || rt.headIndex == MXL_UNDEFINED_INDEX) return kNoIndex;
    return fr_.last_ready(rt.headIndex);
}

// 0 — nežinoma: variklis paruošimo laiku laiko stebėjimo momentą. MXL tęstinio flow'o rašytojas (PosixContinuousFlowWriter,
// MXL 1.1 / PR #653) runtime.lastWriteTime NEATNAUJINA — lieka flow sukūrimo laikas. Jį grąžinus, kiekvienas garso grain'as
// atrodė atėjęs „laiku“ (≤ a(i)): Late histerezė nuolat vėluojančiam garsui kas 8 kadrus nusiimdavo, ir A vėl laukdavo iki
// a(i) visų srautų (favsync_late_audio[G]: kitų srautų vėlavimo mediana 6,6 ms vietoj 2,2 ms); ewma — neigiamas.
std::uint64_t MxlAudioInput::last_write_tai() { return 0; }

Probe MxlAudioInput::get(Index j, std::uint64_t timeout_ns, GrainRef& out) {
    out = GrainRef{};
    if (rd_ == nullptr) return Probe::Gone;
    const std::uint64_t end = fr_.end(j);
    const std::uint32_t count = fr_.count(j);
    if (count > lb_.size()) return Probe::Error;
    mxlWrappedMultiBufferSlice sl{};
    const mxlStatus st = timeout_ns == 0 ? mxlFlowReaderGetSamplesNonBlocking(rd_, end, count, &sl)
                                         : mxlFlowReaderGetSamples(rd_, end, count, timeout_ns, &sl);
    switch (st) {
        case MXL_STATUS_OK: break;
        case MXL_ERR_OUT_OF_RANGE_TOO_EARLY:
        case MXL_ERR_TIMEOUT: return Probe::NotYet;
        case MXL_ERR_OUT_OF_RANGE_TOO_LATE: return Probe::TooLate;
        case MXL_ERR_FLOW_INVALID:
        case MXL_ERR_FLOW_NOT_FOUND: return Probe::Gone;
        default: return Probe::Error;
    }
    if (sl.count < 2) return Probe::Error;  // stereo (gen audio_def)
    std::size_t off = 0;
    for (int f = 0; f < 2; ++f) {  // žiedo dvi dalys (fragmentai)
        const auto& frag = sl.base.fragments[f];
        const std::size_t n = frag.size / sizeof(float);
        if (off + n > count) return Probe::Error;
        if (n) {
            std::memcpy(lb_.data() + off, frag.pointer, n * sizeof(float));
            std::memcpy(rb_.data() + off, static_cast<const std::uint8_t*>(frag.pointer) + sl.stride, n * sizeof(float));
        }
        off += n;
    }
    if (off != count) return Probe::Error;
    last_ = j;
    n_ = count;
    out.ptr = lb_.data();
    return Probe::Ready;
}

Probe MxlAudioInput::probe(Index j, GrainRef& out) { return get(j, 0, out); }

Probe MxlAudioInput::wait(Index j, std::uint64_t timeout_ns, GrainRef& out) { return get(j, std::max<std::uint64_t>(1, timeout_ns), out); }

bool MxlAudioInput::writer_alive() {
    bool active = false;
    return mxlIsFlowActive(inst_, id_.c_str(), &active) == MXL_STATUS_OK && active;
}

void MxlAudioInput::detach() {
    if (rd_ != nullptr) mxlReleaseFlowReader(inst_, rd_);
    rd_ = nullptr;
}

// --- MxlVideoOutput -------------------------------------------------------------------------------------------------
GrainCopy host_copy() {
    return [](void* dst, const void* src, std::size_t n) {
        std::memcpy(dst, src, n);
        return true;
    };
}

std::vector<std::uint8_t> black_grain(const vramflow::gfx::Raster& r) { return func::black_grain(r); }

// INVALID_ARG — AlreadyCommitted (laikrodis atgal, M11) TIK indeksui ≤ paskutinio šio prievado įrašyto; kitaip tai išėjimo
// klaida (skaičiuojama; ilgalaikė — vienetas sugenda), ne amžinas tylus „jau commit'intas“.
OpenResult MxlVideoOutput::open(Index i, void*& slot) {
    mxlStatus st{};
    std::uint32_t* p = out_.open(i, st);
    if (p == nullptr) {
        if (st == MXL_ERR_INVALID_ARG && last_ != kNoIndex && i <= last_) return OpenResult::AlreadyCommitted;  // M11
        if (st == MXL_ERR_INVALID_ARG) ++invalid_arg_errors;
        return OpenResult::Error;
    }
    slot = p;
    return OpenResult::Ok;
}

bool MxlVideoOutput::commit() {
    if (out_.commit() != MXL_STATUS_OK) return false;
    last_ = out_.last_committed;
    return true;
}

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
    const bool ok = fill(i, a, repeat_from);
    if (ok && a != GapAction::None) last_ = last_ == kNoIndex ? i : std::max(last_, i);
    return ok;
}

bool MxlVideoOutput::fill(Index i, GapAction a, Index repeat_from) {
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
