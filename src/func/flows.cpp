// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/flows.hpp"

#include <algorithm>
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

std::string video_def(int role, std::uint32_t n, std::uint64_t run_seed, const vramflow::VideoFormat& f) {
    char label[128], hint[160];
    std::snprintf(label, sizeof label, "vramflow-func %d/%u", role, n);
    std::snprintf(hint, sizeof hint, "vramflow-func-%016llx-%d:Video %u", static_cast<unsigned long long>(run_seed), role, n);
    vramflow::VideoFormat v = f;
    v.raster.alpha = false;
    return vramflow::video_flow_def(flow_id(role, n), label, v, hint);
}

namespace {
// Bendras v210 / v210a flow aprašas (grafikos šaltinis ir kompozitoriaus išėjimas): id, tekstai ir rastras.
std::string raster_flow_def(const std::string& id, const char* what, std::uint32_t n, std::uint32_t k, int role,
                            const vramflow::gfx::Raster& r, vramflow::Rational rate, std::uint64_t run_seed) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"vramflow-func %s %u/%u","label":"vramflow-func %s %u/%u",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-func-%016llx-%d:%s %u/%u"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"%s",)"
                  R"("grain_rate":{"numerator":%lld,"denominator":%lld},"frame_width":%d,"frame_height":%d,)"
                  R"("interlace_mode":"%s","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cb","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cr","width":%d,"height":%d,"bit_depth":10}]})",
                  id.c_str(), what, n, k, what, n, k, static_cast<unsigned long long>(run_seed), role,
                  what[0] == 'g' ? "Gfx" : "Comp", n, k, r.alpha ? "video/v210a" : "video/v210", static_cast<long long>(rate.num), static_cast<long long>(rate.den), r.w, r.h,
                  r.interlaced ? "interlaced_tff" : "progressive", r.w, r.h, r.w / 2, r.h, r.w / 2, r.h);
    return b;
}
// Seni kvietėjai: interlaced 25/1 (grain'ų ritmas 50/1), progresyvus 50/1.
constexpr vramflow::Rational legacy_rate(const vramflow::gfx::Raster& r) { return {r.interlaced ? 25 : 50, 1}; }
}  // namespace

std::string gfx_def(std::uint32_t n, std::uint32_t k, const vramflow::gfx::Raster& r, std::uint64_t run_seed) {
    return raster_flow_def(flow_id(kRoleGfx, n * 8 + k), "gfx", n, k, kRoleGfx, r, legacy_rate(r), run_seed);
}

std::string comp_def(std::uint32_t n, const vramflow::gfx::Raster& r, std::uint64_t run_seed) {
    return raster_flow_def(flow_id(kRoleComp, n), "comp", n, 0, kRoleComp, r, legacy_rate(r), run_seed);
}

std::string gfx_def(std::uint32_t n, std::uint32_t k, const vramflow::VideoFormat& f, std::uint64_t run_seed) {
    return raster_flow_def(flow_id(kRoleGfx, n * 8 + k), "gfx", n, k, kRoleGfx, f.raster, f.frame_rate, run_seed);
}

std::string comp_def(std::uint32_t n, const vramflow::VideoFormat& f, std::uint64_t run_seed) {
    return raster_flow_def(flow_id(kRoleComp, n), "comp", n, 0, kRoleComp, f.raster, f.frame_rate, run_seed);
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

bool wait_until(mxlRational rate, std::uint64_t index, std::int64_t offset_ns, const std::atomic<bool>& stop) {
    const std::uint64_t target = mxlIndexToTimestamp(&rate, index) + offset_ns;
    std::uint64_t now = now_tai();
    // > 2 grain'ai iki tikslo — laikrodis šoko atgal (1080i50: 40 ms, kaip anksčiau)
    const std::uint64_t two = 2 * static_cast<std::uint64_t>(mxlv::timing::time_of(1, {rate.numerator, rate.denominator}));
    const bool stepped = target > now && target - now > two;
    // Absoliutus miegas (A27, M14) ne ilgesniais kaip 50 ms gabalais — kaip engine::SystemClock: santykinis miegas
    // kiekviename gabale pridėdavo kvietimo vėlavimą prie tikslo.
    while (now < target && !stop.load()) {
        mxlSleepUntil(std::min<std::uint64_t>(target, now + 50'000'000ull));
        now = now_tai();
    }
    return stepped;
}

// --- VideoOut ---------------------------------------------------------------------------------------------------------
namespace {
// MXL grain'o ritmas turi atitikti formato grain_rate() (1080i50 ir seni kvietėjai — 50/1, R9). Tuščias tekstas — tinka;
// klaidoje — abu dažniai ir formato vardas.
std::string check_rate(const mxlRational& g, const vramflow::VideoFormat& f) {
    const vramflow::Rational want = f.grain_rate();
    if (vramflow::same_rate({g.numerator, g.denominator}, want)) return {};
    return "grainRate " + std::to_string(g.numerator) + "/" + std::to_string(g.denominator) + " != " + std::to_string(want.num) +
           "/" + std::to_string(want.den) + " (formatas " + vramflow::format_name(f) + ")";
}
// Seni (Raster) kvietėjai: grain'o ritmas visada 50/1 (R9).
vramflow::VideoFormat legacy_format(const vramflow::gfx::Raster& r) { return {r, {r.interlaced ? 25 : 50, 1}}; }
}  // namespace

std::string check_layout(const mxlGrainStorageLayout& l, const vramflow::gfx::Raster& r) {
    namespace gfx = vramflow::gfx;
    const std::uint32_t want_planes = r.alpha ? 2 : 1;
    if (l.planeCount != want_planes) return "plokštumų " + std::to_string(l.planeCount) + " != " + std::to_string(want_planes);
    // Visi variantai (etalonas, CPU, CUDA) adresuoja: fill = grain pradžia, key = grain + fill_bytes, eilutės be tarpų.
    if (l.planes[0].offset != 0) return "fill poslinkis " + std::to_string(l.planes[0].offset) + " != 0";
    if (l.planes[0].size != gfx::fill_bytes(r))
        return "fill dydis " + std::to_string(l.planes[0].size) + " != " + std::to_string(gfx::fill_bytes(r));
    if (l.planes[0].pitch != gfx::fill_line_bytes(r.w))
        return "fill pitch " + std::to_string(l.planes[0].pitch) + " != " + std::to_string(gfx::fill_line_bytes(r.w));
    if (r.alpha) {
        if (l.planes[1].offset != gfx::fill_bytes(r))
            return "key poslinkis " + std::to_string(l.planes[1].offset) + " != " + std::to_string(gfx::fill_bytes(r)) +
                   " (key turi eiti iškart po fill)";
        if (l.planes[1].size != gfx::key_bytes(r))
            return "key dydis " + std::to_string(l.planes[1].size) + " != " + std::to_string(gfx::key_bytes(r));
        if (l.planes[1].pitch != gfx::key_line_bytes(r.w))
            return "key pitch " + std::to_string(l.planes[1].pitch) + " != " + std::to_string(gfx::key_line_bytes(r.w));
    }
    return {};
}

bool VideoOut::create(mxlInstance inst, const std::string& def, const std::string& options, std::string& err,
                      const vramflow::gfx::Raster& r) {
    return create(inst, def, options, err, legacy_format(r));
}

bool VideoOut::create(mxlInstance inst, const std::string& def, const std::string& options, std::string& err,
                      const vramflow::VideoFormat& f) {
    const vramflow::gfx::Raster& r = f.raster;
    raster = r;
    fmt = f;
    mxlFlowConfigInfo cfg{};
    bool created = false;
    mxlStatus st = mxlCreateFlowWriter(inst, def.c_str(), options.empty() ? nullptr : options.c_str(), &w, &cfg, &created);
    create_status = st;
    if (st != MXL_STATUS_OK) {
        w = nullptr;
        err = "mxlCreateFlowWriter status " + std::to_string(st) + (st == MXL_ERR_CONFLICT ? " (CONFLICT)" : "");
        return false;
    }
    if (!created) { existed = true; err = "flow jau egzistavo (created=false)"; return false; }
    if (const std::string bad = check_rate(cfg.common.grainRate, f); !bad.empty()) { err = bad; return false; }
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

mxlStatus VideoOut::commit_invalid(std::uint64_t index) {
    if (open_grain) return MXL_ERR_INVALID_ARG;
    info = mxlGrainInfo{};
    mxlStatus st = mxlFlowWriterOpenGrainSlot(w, index, &info, &cur_slot);
    if (st != MXL_STATUS_OK) return st;
    if (info.index != index) {
        mxlFlowWriterCancelGrain(w);
        return MXL_ERR_INTERNAL;
    }
    info.flags |= MXL_GRAIN_FLAG_INVALID;
    info.validSlices = info.totalSlices;
    st = mxlFlowWriterCommitGrain(w, &info);
    if (st == MXL_STATUS_OK) last_committed = index;
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
namespace {
std::string read_flow_def(mxlInstance inst, const std::string& flow, std::string& err) {
    size_t n = 0;
    mxlGetFlowDef(inst, flow.c_str(), nullptr, &n);  // grąžina INVALID_ARG ir reikiamą dydį
    std::string def(n ? n : 1, '\0');
    const mxlStatus ds = n ? mxlGetFlowDef(inst, flow.c_str(), def.data(), &n) : MXL_ERR_INTERNAL;
    if (ds != MXL_STATUS_OK) { err = "mxlGetFlowDef status " + std::to_string(ds); return {}; }
    def.resize(std::strlen(def.c_str()));
    return def;
}

// Laukimo laikrodis pagal OpenPolicy (planas §P6). keep_waiting: true — pamiegota ir galima bandyti dar kartą; false — riba
// arba sustabdymas (err užpildytas). Miegama ne ilgiau kaip 50 ms — sustabdymas nutraukia per ≤ 100 ms.
struct Waiter {
    const runtime::OpenPolicy& p;
    const std::string& flow;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point next_log = t0 + std::chrono::milliseconds(p.log_every_ms);
    bool keep_waiting(const std::atomic<bool>& stop, std::string& err, const char* what) {
        using namespace std::chrono;
        if (stop.load()) { err = what + flow + " neatsirado (sustabdyta)"; return false; }
        if (p.wait_ms == 0) { err = what + flow + " nerastas (--input-wait-ms: vienas bandymas)"; return false; }
        const auto now = steady_clock::now();
        const auto limit = t0 + milliseconds(p.wait_ms);
        if (p.wait_ms > 0 && now >= limit) {
            err = what + flow + " neatsirado per " + std::to_string(p.wait_ms) + " ms (--input-wait-ms " + std::to_string(p.wait_ms) +
                  "; 0 — laukti neribotai)";
            return false;
        }
        if (p.log_every_ms && now >= next_log) {
            std::fprintf(stderr, "[vramflow] laukiu %s%s, %lld s\n", what, flow.c_str(),
                         static_cast<long long>(duration_cast<seconds>(now - t0).count()));
            next_log += milliseconds(p.log_every_ms);
        }
        auto d = milliseconds(50);
        if (p.wait_ms > 0) d = std::min(d, duration_cast<milliseconds>(limit - now) + milliseconds(1));
        std::this_thread::sleep_for(d);
        return true;
    }
};
}  // namespace

bool VideoIn::open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop,
                   std::string& err, const vramflow::VideoFormat* expect) {
    return open(inst, flow, runtime::OpenPolicy{wait_ms, 0}, stop, err, expect);
}

bool VideoIn::open(mxlInstance inst, const std::string& flow, const runtime::OpenPolicy& p, const std::atomic<bool>& stop,
                   std::string& err, const vramflow::VideoFormat* expect) {
    if (!open_reader(inst, flow, p, stop, err)) return false;
    const std::string def = read_flow_def(inst, flow, err);
    if (def.empty()) return false;
    if (!vramflow::format_from_flow_def(def, fmt, err)) return false;
    raster = fmt.raster;
    if (expect && !(expect->raster == fmt.raster)) {
        err = "įėjimo formatas " + vramflow::format_name(fmt) + " != laukiamas " + vramflow::format_name(*expect);
        return false;
    }
    if (const std::string bad = check_rate(cfg.common.grainRate, expect ? *expect : fmt); !bad.empty()) { err = "įėjimo " + bad; return false; }
    cuda = is_cuda_layout(layout);
    if (const std::string bad = check_layout(layout, raster); !bad.empty()) { err = "įėjimo " + bad; return false; }
    return true;
}

bool VideoIn::open(mxlInstance inst, const std::string& flow, std::uint32_t wait_ms, const std::atomic<bool>& stop,
                   std::string& err, const vramflow::gfx::Raster* expect) {
    if (!open_reader(inst, flow, runtime::OpenPolicy{wait_ms, 0}, stop, err)) return false;
    const vramflow::VideoFormat legacy{vramflow::gfx::kHd1080i, {25, 1}};
    if (const std::string bad = check_rate(cfg.common.grainRate, legacy); !bad.empty()) { err = "įėjimo " + bad; return false; }
    cuda = is_cuda_layout(layout);
    if (expect) {
        raster = *expect;
    } else {
        const std::string def = read_flow_def(inst, flow, err);
        if (def.empty()) return false;
        if (!vramflow::gfx::raster_from_flow_def(def, raster, err)) return false;
    }
    fmt = legacy_format(raster);
    if (const std::string bad = check_layout(layout, raster); !bad.empty()) { err = "įėjimo " + bad; return false; }
    return true;
}

bool VideoIn::open_reader(mxlInstance inst, const std::string& flow, const runtime::OpenPolicy& p, const std::atomic<bool>& stop,
                          std::string& err) {
    id = flow;
    Waiter wt{p, flow};
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
        if (!wt.keep_waiting(stop, err, "flow ")) return false;
    }
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
bool AudioOut::create(mxlInstance inst, const std::string& def, std::string& err, vramflow::Rational grain_rate) {
    mxlFlowConfigInfo cfg{};
    bool created = false;
    const mxlStatus st = mxlCreateFlowWriter(inst, def.c_str(), nullptr, &w, &cfg, &created);
    if (st != MXL_STATUS_OK) { w = nullptr; err = "audio mxlCreateFlowWriter status " + std::to_string(st); return false; }
    if (!created) { existed = true; err = "audio flow jau egzistavo (created=false)"; return false; }
    if (cfg.continuous.channelCount != 2) { err = "audio kanalų ne 2"; return false; }
    std::size_t maxw = 0;
    const std::uint32_t need = vramflow::max_samples_per_grain(grain_rate);
    if (need == 0) { err = "audio: netinkamas grain'o dažnis"; return false; }
    if (mxlFlowWriterGetMaxWriteLengthSamples(w, &maxw) != MXL_STATUS_OK || maxw < need) {
        err = "audio max write " + std::to_string(maxw) + " < " + std::to_string(need);
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
    return open(inst, flow, runtime::OpenPolicy{wait_ms, 0}, stop, err);
}

bool AudioIn::open(mxlInstance inst, const std::string& flow, const runtime::OpenPolicy& p, const std::atomic<bool>& stop,
                   std::string& err) {
    Waiter wt{p, flow};
    for (;;) {
        const mxlStatus st = mxlCreateFlowReader(inst, flow.c_str(), "", &r);
        if (st == MXL_STATUS_OK) return true;
        r = nullptr;
        if (st != MXL_ERR_FLOW_NOT_FOUND) { err = "audio reader status " + std::to_string(st); return false; }
        if (!wt.keep_waiting(stop, err, "audio flow ")) return false;
    }
}

mxlStatus AudioIn::read(std::uint64_t end, std::uint32_t count, float* l, float* rr) {
    if (!r) return MXL_ERR_FLOW_NOT_FOUND;  // atjungtas (gamintojas dingo, laukiama re-attach)
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
