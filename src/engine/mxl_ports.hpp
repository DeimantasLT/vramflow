// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variklio MXL prievadai (planas §7 mxl_ports.hpp, V7) — vienintelė vieta, kur variklis kalba su MXL.
//   SystemClock    — engine/system_clock.hpp (čia įtraukiama);
//   MxlVideoInput  — ant runtime::InputSlot: head() be sisteminio kvietimo (M8), probe = get(timeout 0),
//                    wait = blokuojantis get ant flow'o futex (M1, M2), writer_alive = mxlIsFlowActive (M9);
//   MxlVideoOutput — ant func::VideoOut: commit, K3 spragos (Invalid — commit_invalid; Repeat / Black — kopija per
//                    GrainCopy į naujai atidarytą slot'ą, tada commit).
// Visus metodus kviečia tik variklio gija.
#include <mxl/mxl.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine/audio_frames.hpp"
#include "engine/clock.hpp"
#include "engine/ports.hpp"
#include "engine/system_clock.hpp"
#include "func/flows.hpp"
#include "gfx/raster.hpp"
#include "runtime/input_slot.hpp"

namespace mxlv::engine {

class MxlVideoInput final : public InputPort {
public:
    MxlVideoInput(runtime::InputSlot& slot, mxlInstance inst) : slot_(slot), inst_(inst) {}

    bool attach_poll(std::uint64_t now_tai) override;
    bool mismatch(std::string& why) const override;
    Index head() override;
    std::uint64_t last_write_tai() override;
    Probe probe(Index j, GrainRef& out) override;
    Probe wait(Index j, std::uint64_t timeout_ns, GrainRef& out) override;
    bool writer_alive() override;
    bool lapped(Index j, Index now_idx) override;
    void detach() override;
    void retarget(const FlowRef&) override {}  // V22 (valdymo eilė) — kol kas nepalaikoma

private:
    Probe get(Index j, std::uint64_t timeout_ns, GrainRef& out);
    runtime::InputSlot& slot_;
    mxlInstance inst_;
};

// Garso (tęstinio flow'o, 2 kanalai float) įėjimas (planas §5 MxlAudioInput): vieneto indekso j kadro mėginiai
// (AudioFrames) nukopijuojami į prievado buferius left() / right() (galioja iki kito probe / wait); GrainRef.ptr — left().
// head() — paskutinis kadras, kurio garsas visas (runtime info — bendra atmintis, be sisteminio kvietimo); wait —
// blokuojantis mxlFlowReaderGetSamples (futex). Prisijungimas — tik prie flow'o su gyvu rašytoju (mxlIsFlowActive); be
// reader'io probe — Gone (variklis atjungia ir vėl bando kas 100 ms). Lap'o nėra (mėginiai nukopijuoti probe metu).
class MxlAudioInput final : public InputPort {
public:
    MxlAudioInput(mxlInstance inst, std::string flow_id, AudioFrames frames, mxlFlowReader adopt = nullptr);
    ~MxlAudioInput() override;
    MxlAudioInput(const MxlAudioInput&) = delete;
    MxlAudioInput& operator=(const MxlAudioInput&) = delete;
    MxlAudioInput(MxlAudioInput&&) noexcept;
    MxlAudioInput& operator=(MxlAudioInput&&) = delete;

    bool attach_poll(std::uint64_t now_tai) override;
    bool mismatch(std::string&) const override { return false; }
    Index head() override;
    std::uint64_t last_write_tai() override;  // visada 0: tęstinio flow'o lastWriteTime MXL neatnaujina (žr. .cpp)
    Probe probe(Index j, GrainRef& out) override;
    Probe wait(Index j, std::uint64_t timeout_ns, GrainRef& out) override;
    bool writer_alive() override;
    bool lapped(Index, Index) override { return false; }
    void detach() override;
    void retarget(const FlowRef&) override {}

    // Paskutinio sėkmingo probe / wait indeksas ir jo mėginiai (count() kiekvienam kanalui).
    Index last() const { return last_; }
    const float* left() const { return lb_.data(); }
    const float* right() const { return rb_.data(); }
    std::uint32_t count() const { return n_; }
    std::uint64_t attaches() const { return attaches_; }
    bool attached() const { return rd_ != nullptr; }

private:
    Probe get(Index j, std::uint64_t timeout_ns, GrainRef& out);
    mxlInstance inst_;
    std::string id_;
    AudioFrames fr_;
    mxlFlowReader rd_ = nullptr;
    std::vector<float> lb_, rb_;
    Index last_ = kNoIndex;
    std::uint32_t n_ = 0;
    std::uint64_t attaches_ = 0;
};

// Kopija į grain'o slot'ą: host — memcpy; cuda-linear — D2D į funkcijos stream'ą (tada flush — to stream'o sync).
using GrainCopy = std::function<bool(void* dst, const void* src, std::size_t bytes)>;
GrainCopy host_copy();

// Juodas grain'as (host atmintyje): v210 Y = 64, Cb = Cr = 512 visose eilutėse (ir eilutės užpildo dalyje);
// v210a key plokštuma — 0 (skaidru). Dydis — gfx::grain_bytes(r).
std::vector<std::uint8_t> black_grain(const vramflow::gfx::Raster& r);

class MxlVideoOutput final : public OutputPort {
public:
    // flush — kviečiamas po kiekvienos Repeat / Black kopijos prieš commit (nullptr — sinchroninė kopija);
    // black — black_grain() buferis (nullptr — Black nepalaikomas, fill_gap grąžina false).
    MxlVideoOutput(func::VideoOut& out, GrainCopy copy, std::function<bool()> flush, const void* black)
        : out_(out), copy_(std::move(copy)), flush_(std::move(flush)), black_(black) {}

    OpenResult open(Index i, void*& slot) override;
    bool commit() override;
    void cancel() override;
    bool fill_gap(Index i, GapAction a, Index repeat_from) override;
    bool flush_gaps() override { return true; }  // kopijos jau sinchronizuotos fill_gap metu
    bool interlaced() const override { return out_.raster.interlaced; }
    std::uint32_t slot_count() const override { return out_.layout.slotCount; }
    void release() override {
        if (release_fn) release_fn();
    }

    // Funkcija nustato: išėjimo flow atlaisvinimas (VideoOut::release su savo mxlInstance) sugedus vienetui.
    std::function<void()> release_fn;
    // open → INVALID_ARG indeksui, didesniam už paskutinį šio prievado įrašytą (ne laikrodis atgal): skaičius (santraukai).
    std::uint64_t invalid_arg_errors = 0;

private:
    bool copy_commit(Index i, const void* src);
    bool fill(Index i, GapAction a, Index repeat_from);
    Index last_ = kNoIndex;  // paskutinis šio prievado commit'intas (turinys arba spraga) indeksas
    func::VideoOut& out_;
    GrainCopy copy_;
    std::function<bool()> flush_;
    const void* black_;
};

}  // namespace mxlv::engine
