// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/flows.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <thread>

#include "media/v210.hpp"
#include "mxlio/slots.hpp"

namespace mxlv::func {

std::string flow_id(int role, std::uint32_t n) {
    char b[64];
    std::snprintf(b, sizeof b, "4d584c47-00%02d-4000-8000-%012x", role, n);
    return b;
}

std::string video_def(int role, std::uint32_t n, std::uint64_t run_seed) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow-func role %d flow %u","label":"vramflow-func %d/%u",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-func-%016llx-%d:Video %u"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210",)"
                  R"("grain_rate":{"numerator":25,"denominator":1},"frame_width":1920,"frame_height":1080,)"
                  R"("interlace_mode":"interlaced_tff","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":1920,"height":1080,"bit_depth":10},)"
                  R"({"name":"Cb","width":960,"height":1080,"bit_depth":10},)"
                  R"({"name":"Cr","width":960,"height":1080,"bit_depth":10}]})",
                  flow_id(role, n).c_str(), role, n, role, n, static_cast<unsigned long long>(run_seed), role, n);
    return b;
}

std::string audio_def(std::uint32_t n, std::uint64_t run_seed) {
    char b[1024];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow-func audio %u","label":"vramflow-func audio %u",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-func-%016llx-1:Audio %u"]},)"
                  R"("format":"urn:x-nmos:format:audio","media_type":"audio/float32","sample_rate":{"numerator":48000},)"
                  R"("channel_count":2,"bit_depth":32,"parents":[]})",
                  flow_id(kRoleAudio, n).c_str(), n, n, static_cast<unsigned long long>(run_seed), n);
    return b;
}

std::uint64_t now_tai() { return mxlGetTime(); }

bool ClockWatch::stepped() {
    // Lyginam su CLOCK_MONOTONIC_RAW: steady_clock (CLOCK_MONOTONIC) slew'inamas kartu su realtime, todėl jo atžvilgiu
    // slew'o nesimato (išmatuota: chrony vijosi +4,4 % — steady_clock jo „nematė").
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    const std::int64_t raw = static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000ll + ts.tv_nsec;
    const std::int64_t off = static_cast<std::int64_t>(now_tai()) - raw;
    const bool step = last != INT64_MIN && (off - last > 5'000'000 || last - off > 5'000'000);
    if (step && off < last) back_ns += static_cast<std::uint64_t>(last - off);
    last = off;
    if (step) ++steps;
    return step;
}

bool wait_until(std::uint64_t index, std::int64_t offset_ns, const std::atomic<bool>& stop) {
    const std::uint64_t target = mxlIndexToTimestamp(&kFieldRate, index) + offset_ns;
    std::uint64_t now = now_tai();
    const bool stepped = target > now && target - now > 40'000'000ull;  // > 2 laukai — laikrodis šoko atgal
    while (now < target && !stop.load()) {
        const std::uint64_t d = target - now;
        mxlSleepForNs(d < 50'000'000ull ? d : 50'000'000ull);
        now = now_tai();
    }
    return stepped;
}

// --- VideoOut ---------------------------------------------------------------------------------------------------------
namespace {
// Plokštumų skaičius ir dydžiai prieš rastrą (fill [+ key]); tuščias tekstas — tinka.
std::string check_layout(const mxlGrainStorageLayout& l, const vramflow::gfx::Raster& r) {
    const std::uint32_t want_planes = r.alpha ? 2 : 1;
    if (l.planeCount != want_planes) return "plokštumų " + std::to_string(l.planeCount) + " != " + std::to_string(want_planes);
    if (l.planes[0].size != vramflow::gfx::fill_bytes(r))
        return "fill dydis " + std::to_string(l.planes[0].size) + " != " + std::to_string(vramflow::gfx::fill_bytes(r));
    if (r.alpha && l.planes[1].size != vramflow::gfx::key_bytes(r))
        return "key dydis " + std::to_string(l.planes[1].size) + " != " + std::to_string(vramflow::gfx::key_bytes(r));
    return {};
}
}  // namespace

bool VideoOut::create(mxlInstance inst, const std::string& def, const std::string& options, std::string& err,
                      const vramflow::gfx::Raster& r) {
    raster = r;
    mxlFlowConfigInfo cfg{};
    bool created = false;
    mxlStatus st = mxlCreateFlowWriter(inst, def.c_str(), options.empty() ? nullptr : options.c_str(), &w, &cfg, &created);
    if (st != MXL_STATUS_OK) {
        w = nullptr;
        err = "mxlCreateFlowWriter status " + std::to_string(st) + (st == MXL_ERR_CONFLICT ? " (CONFLICT)" : "");
        return false;
    }
    if (!created) { err = "flow jau egzistavo (created=false)"; return false; }
    if (r.interlaced && (cfg.common.grainRate.numerator != 50 || cfg.common.grainRate.denominator != 1)) {
        err = "grainRate ne 50/1 (" + std::to_string(cfg.common.grainRate.numerator) + "/" +
              std::to_string(cfg.common.grainRate.denominator) + ")";
        return false;
    }
    if ((st = mxlFlowWriterGetStorageLayout(w, &layout)) != MXL_STATUS_OK) { err = "GetStorageLayout " + std::to_string(st); return false; }
    slots.assign(layout.slotCount, mxlGrainStorage{});
    if ((st = mxlFlowWriterMapSlots(w, layout.slotCount, slots.data())) != MXL_STATUS_OK) { err = "MapSlots " + std::to_string(st); return false; }
    mxlFlowWriterGetStorageLayout(w, &layout);
    if (const std::string bad = check_layout(layout, r); !bad.empty()) { err = "lauko " + bad; return false; }
    return true;
}

std::uint32_t* VideoOut::open(std::uint64_t index, mxlStatus& st) {
    info = mxlGrainInfo{};
    st = mxlFlowWriterOpenGrainSlot(w, index, &info, &cur_slot);
    if (st != MXL_STATUS_OK) return nullptr;
    if (cur_slot >= layout.slotCount || info.index != index || info.grainSize != vramflow::gfx::grain_bytes(raster)) {
        mxlFlowWriterCancelGrain(w);
        st = MXL_ERR_INTERNAL;
        return nullptr;
    }
    open_grain = true;
    return static_cast<std::uint32_t*>(slot_ptr(layout, slots[cur_slot]));
}

mxlStatus VideoOut::commit() {
    info.validSlices = info.totalSlices;
    const mxlStatus st = mxlFlowWriterCommitGrain(w, &info);
    open_grain = false;
    if (st == MXL_STATUS_OK) last_committed = info.index;
    return st;
}

void VideoOut::cancel() {
    if (open_grain) mxlFlowWriterCancelGrain(w);
    open_grain = false;
}

void VideoOut::release(mxlInstance inst) {
    cancel();
    if (w) mxlReleaseFlowWriter(inst, w);
    w = nullptr;
}

// --- VideoIn ----------------------------------------------------------------------------------------------------------
bool VideoIn::open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop,
                   std::string& err, const vramflow::gfx::Raster* expect) {
    id = flow;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
    for (;;) {
        mxlStatus st = mxlCreateFlowReader(inst, flow.c_str(), "", &r);
        if (st == MXL_STATUS_OK) {
            if (mxlFlowReaderGetConfigInfo(r, &cfg) == MXL_STATUS_OK && mxlFlowReaderGetStorageLayout(r, &layout) == MXL_STATUS_OK) {
                slots.assign(layout.slotCount, mxlGrainStorage{});
                st = mxlFlowReaderMapSlots(r, layout.slotCount, slots.data());
                if (st == MXL_STATUS_OK && mxlFlowReaderGetStorageLayout(r, &layout) == MXL_STATUS_OK) break;
            }
            mxlReleaseFlowReader(inst, r);
            r = nullptr;
            if (st != MXL_ERR_NOT_FOUND && st != MXL_STATUS_OK) { err = "reader map status " + std::to_string(st); return false; }
        } else if (st != MXL_ERR_FLOW_NOT_FOUND) {
            err = "mxlCreateFlowReader status " + std::to_string(st);
            return false;
        }
        if (stop.load() || std::chrono::steady_clock::now() > deadline) { err = "flow " + flow + " neatsirado"; return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    cuda = is_cuda_layout(layout);
    if (expect) {
        raster = *expect;
    } else {
        size_t n = 0;
        mxlGetFlowDef(inst, flow.c_str(), nullptr, &n);  // grąžina INVALID_ARG ir reikiamą dydį
        std::string def(n ? n : 1, ' ');
        const mxlStatus ds = n ? mxlGetFlowDef(inst, flow.c_str(), def.data(), &n) : MXL_ERR_INTERNAL;
        if (ds != MXL_STATUS_OK) { err = "mxlGetFlowDef status " + std::to_string(ds); return false; }
        def.resize(std::strlen(def.c_str()));
        if (!vramflow::gfx::raster_from_flow_def(def, raster, err)) return false;
    }
    if (const std::string bad = check_layout(layout, raster); !bad.empty()) { err = "įėjimo " + bad; return false; }
    return true;
}

mxlStatus VideoIn::get(std::uint64_t index, std::uint64_t timeout_ns, mxlGrainInfo& info, const std::uint32_t*& ptr) {
    std::uint32_t slot = 0;
    const mxlStatus st = mxlFlowReaderGetGrainSlot(r, index, MXL_GRAIN_VALID_SLICES_ALL, timeout_ns, &info, &slot);
    ptr = nullptr;
    if (st == MXL_STATUS_OK) {
        if (slot >= layout.slotCount || info.index != index) return MXL_ERR_INTERNAL;
        ptr = static_cast<const std::uint32_t*>(slot_ptr(layout, slots[slot]));
    }
    return st;
}

std::uint64_t VideoIn::head() {
    mxlFlowRuntimeInfo rt{};
    if (mxlFlowReaderGetRuntimeInfo(r, &rt) != MXL_STATUS_OK) return MXL_UNDEFINED_INDEX;
    return rt.headIndex;
}

void VideoIn::release(mxlInstance inst) {
    if (r) mxlReleaseFlowReader(inst, r);
    r = nullptr;
}

// --- Audio ------------------------------------------------------------------------------------------------------------
bool AudioOut::create(mxlInstance inst, const std::string& def, std::string& err) {
    mxlFlowConfigInfo cfg{};
    bool created = false;
    const mxlStatus st = mxlCreateFlowWriter(inst, def.c_str(), nullptr, &w, &cfg, &created);
    if (st != MXL_STATUS_OK) { w = nullptr; err = "audio mxlCreateFlowWriter status " + std::to_string(st); return false; }
    if (!created) { err = "audio flow jau egzistavo"; return false; }
    if (cfg.continuous.channelCount != 2) { err = "audio kanalų ne 2"; return false; }
    std::size_t maxw = 0;
    if (mxlFlowWriterGetMaxWriteLengthSamples(w, &maxw) != MXL_STATUS_OK || maxw < kAudioPerField) {
        err = "audio max write " + std::to_string(maxw) + " < 960";
        return false;
    }
    return true;
}

void AudioOut::release(mxlInstance inst) {
    if (w) mxlReleaseFlowWriter(inst, w);
    w = nullptr;
}

bool AudioIn::open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop,
                   std::string& err) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
    for (;;) {
        const mxlStatus st = mxlCreateFlowReader(inst, flow.c_str(), "", &r);
        if (st == MXL_STATUS_OK) return true;
        if (st != MXL_ERR_FLOW_NOT_FOUND) { err = "audio reader status " + std::to_string(st); return false; }
        if (stop.load() || std::chrono::steady_clock::now() > deadline) { err = "audio flow " + flow + " neatsirado"; return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

mxlStatus AudioIn::read(std::uint64_t end, std::uint32_t count, float* l, float* rr) {
    mxlWrappedMultiBufferSlice sl{};
    const mxlStatus st = mxlFlowReaderGetSamplesNonBlocking(r, end, count, &sl);
    if (st != MXL_STATUS_OK) return st;
    std::size_t off = 0;
    for (int f = 0; f < 2; ++f) {
        const auto& frag = sl.base.fragments[f];
        const std::size_t n = frag.size / sizeof(float);
        if (sl.count >= 2 && n) {
            std::memcpy(l + off, frag.pointer, n * sizeof(float));
            std::memcpy(rr + off, static_cast<const std::uint8_t*>(frag.pointer) + sl.stride, n * sizeof(float));
        }
        off += n;
    }
    return off == count ? MXL_STATUS_OK : MXL_ERR_INTERNAL;
}

void AudioIn::release(mxlInstance inst) {
    if (r) mxlReleaseFlowReader(inst, r);
    r = nullptr;
}

}  // namespace mxlv::func
