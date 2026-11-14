// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// verify --mode ingest — tiesioginis MXL skaitytojas ingest flow'ams (07 video, 08 garsas), sintetiniam A/V klipui
// (scripts/linux/62-make-av-clip.sh). Kodėl ne enc → TS → av_offset.py: matuojamas pats ingest (be koderio, AAC užlaidos ir
// antro dekoderio), mėginio tikslumu (1 / 48 000 s), ir tais pačiais MXL indeksais, kuriuos mato žemupys.
//
// Kiekvienam grain'ui i (po --warmup, atsiliekant --lag-fields):
//   video — barkodas (pirmos 12 grain'o eilučių, D2H): šaltinio grain'o indeksas m; atvaizdis d = i − m (pastovus, išskyrus
//           slydimus ±1 kadras ir kilpos sandūras — map_changes), žymių tolydumas (m(i) − m(i − 1) == 1; marker_jumps),
//           paritetas (interlaced: m ir i to paties pariteto); kadras F ≡ 0 (mod 25) (F = m / 2 interlaced, = m progresyviam) —
//           video impulsas, laukiamas garso mėginys sample_begin(i);
//   garsas — [sample_begin(i), sample_begin(i + 1)): impulsai (|x| > 0,5, grupės maksimumas), tylos tarpai (≥ 1 ms |x| < 10⁻⁴,
//           kai tonas jau buvo) ir trūkiai (1 kHz tono antrosios diferencijos liekana > 0,004, ne arčiau kaip 3000 mėginių nuo
//           impulso);
//   A/V — kiekvienam garso impulsui artimiausias video impulsas (|Δ| < 0,5 s): poslinkis = (impulsas − laukiamas) / 48 kHz.
// --ref (ref_crc_<fmt>.json): kas --every grain'o CRC32C prieš programinio dekoderio etaloną (bitų tikslumas, kaip --mode dec).
// --dump-dir / --dump-count: pirmi galiojantys grain'ai į v210 failus (chromos palyginimui su FFmpeg).
#include <cuda_runtime_api.h>
#include <mxl/flow.h>
#include <mxl/mxl.h>
#include <mxl/time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common/crc32c.hpp"
#include "common/index_math.hpp"
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/media_ref.hpp"
#include "runtime/input_slot.hpp"

namespace mxlv::func {

namespace {
constexpr double kTwoCos = 1.9828897227476208;  // 2 cos(2π · 1000 / 48000)

struct AudioScan {
    bool seen_signal = false;
    std::uint64_t zero_run = 0;
    std::int64_t since_gap = 0;  // ne tylos mėginių nuo pradžios / paskutinio tylos tarpo
    bool have1 = false, have2 = false;
    float x1 = 0, x2 = 0;
    bool in_pulse = false;
    float best = 0;
    std::int64_t best_pos = 0, last_hot = 0, last_disc = -100000;
    std::vector<std::int64_t> pulses, disc, zero;
    std::int64_t next_pos = -1;  // kito laukiamo mėginio pozicija; kitokia — verify pats praleido blokų (atsiliko)
    void feed(std::int64_t p, float x) {
        if (p != next_pos) {  // ne tolydu (verify_skipped): tolydumo būsena iš naujo — tai ne garso trūkis
            have1 = have2 = false;
            zero_run = 0;
            since_gap = 0;
        }
        next_pos = p + 1;
        const float a = std::fabs(x);
        if (a > 0.5f) {
            if (!in_pulse || a > best) best_pos = p;
            if (!in_pulse) best = a;
            best = std::max(best, a);
            in_pulse = true;
            last_hot = p;
        } else if (in_pulse && p - last_hot > 200) {
            pulses.push_back(best_pos);
            in_pulse = false;
        }
        if (a < 1e-4f) {
            ++zero_run;
        } else {
            if (zero_run >= 48) {
                if (seen_signal && zero.size() < 10000) zero.push_back(p - static_cast<std::int64_t>(zero_run));
                since_gap = 0;  // tylos tarpas jau užskaitytas — jo kraštas nėra atskiras trūkis
            }
            zero_run = 0;
            seen_signal = true;
            ++since_gap;
        }
        // Trūkis tikrinamas tik ištisiniame signale (≥ 10 ms po pradžios ar po tylos tarpo): tylos → tono kraštas (garso pradžia
        // flow'e) kitaip būtų skaičiuojamas kaip trūkis
        if (have2 && since_gap > 480) {
            const double r = static_cast<double>(x) - kTwoCos * x1 + x2;
            if (std::fabs(r) > 0.004 && p - last_disc > 100 && disc.size() < 10000) {
                disc.push_back(p);
                last_disc = p;
            }
        }
        x2 = x1;
        x1 = x;
        have2 = have1;
        have1 = true;
    }
};

struct FlowScan {
    std::int64_t last_i = -1;
    std::uint32_t last_m = 0;
    bool have_d = false;
    std::int64_t d = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> map_changes, jumps, vpulses, pairs;  // (i, reikšmė)
    std::uint32_t dumped = 0;
    AudioScan a;
};

std::int64_t wrap24(std::int64_t v) {
    v &= 0xFFFFFF;
    return v >= 0x800000 ? v - 0x1000000 : v;
}

std::string pairs_json(const std::vector<std::pair<std::int64_t, std::int64_t>>& v, std::size_t cap) {
    std::string j = "[";
    for (std::size_t k = 0; k < v.size() && k < cap; ++k) j += (k ? "," : "") + std::string("[") + std::to_string(v[k].first) + "," + std::to_string(v[k].second) + "]";
    return j + "]";
}
std::string list_json(const std::vector<std::int64_t>& v, std::size_t cap) {
    std::string j = "[";
    for (std::size_t k = 0; k < v.size() && k < cap; ++k) j += (k ? "," : "") + std::to_string(v[k]);
    return j + "]";
}
}  // namespace

int verify_ingest(const Opts& o, Stats& st, std::string& extra) {
    auto& stop = stop_flag();
    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) {
        st.fatal = "mxlCreateInstance";
        return 2;
    }
    const std::uint32_t N = o.flows;
    // etalonas: barkodas → CRC32C (programinis dekoderis + nepriklausomas split)
    std::map<std::uint32_t, std::uint32_t> ref_crc;
    if (!o.ref_crc.empty()) {
        FILE* f = std::fopen(o.ref_crc.c_str(), "r");
        if (!f) {
            st.fatal = "nėra --ref failo";
            mxlDestroyInstance(inst);
            return 1;
        }
        unsigned mk = 0, crc = 0;
        int ch = 0;
        while ((ch = std::fgetc(f)) != EOF)
            if (ch == '[' && std::fscanf(f, "%u,%u]", &mk, &crc) == 2) ref_crc[mk] = crc;
        std::fclose(f);
    }
    std::vector<runtime::InputSlot> vin;
    std::vector<AudioIn> ain(o.audio ? N : 0);
    vin.reserve(N);
    for (std::uint32_t n = 0; n < N && o.video; ++n) vin.push_back(make_input(o, flow_id(kRoleIngest, o.flow_base + n), nullptr));
    std::vector<runtime::InputSlot*> all;
    for (auto& s : vin) all.push_back(&s);
    auto release_all = [&] {
        for (auto* s : all) s->release();
        for (auto& a : ain) a.release(inst);
        mxlDestroyInstance(inst);
    };
    if (const int a = attach_inputs(all, inst, o, "verify ingest", st.fatal)) {
        release_all();
        return exit_code(a);
    }
    std::string err;
    for (std::uint32_t n = 0; n < ain.size(); ++n)
        if (!ain[n].open(inst, flow_id(kRoleIngestAudio, o.flow_base + n), open_policy(o), stop, err)) {
            st.fatal = err;
            release_all();
            return 4;
        }
    const vramflow::VideoFormat fmt = vin.empty() ? vramflow::kFmt1080i50 : vin[0].format();
    for (auto& s : vin)
        if (!(s.format() == fmt)) {
            st.fatal = "verify ingest: visi video flow'ai turi būti to paties formato (" + vramflow::format_name(fmt) + ")";
            release_all();
            return 1;
        }
    const vramflow::Rational g = fmt.grain_rate();
    const mxlRational rate = mxl_rate(g);
    const bool inter = fmt.raster.interlaced;
    const media::Dim dim{fmt.raster.w, fmt.raster.h};
    const std::size_t gwords = media::grain_words(dim, inter);
    const int words = media::v210_words_per_line(dim.w);
    const std::size_t head_words = static_cast<std::size_t>(words) * 12;
    const std::uint32_t gc = vin.empty() ? 16 : vin[0].in().layout.slotCount;
    const std::uint64_t lag = std::min<std::uint64_t>(o.lag_fields, gc > 3 ? gc - 3 : 1);
    std::vector<std::uint32_t> head(head_words), full(gwords);
    const std::uint32_t cap = vramflow::max_samples_per_grain(g) + 1;
    std::vector<float> l(cap), r(cap);
    std::vector<FlowScan> fs(N);
    const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    std::uint64_t i = mxlGetCurrentIndex(&rate) + static_cast<std::uint64_t>(o.warmup_s * static_cast<double>(rate.numerator) / rate.denominator);
    const std::uint64_t i0 = i;
    while (!stop.load() && now_tai() < end_tai) {
        wait_until(rate, i + lag, 0, stop);
        if (stop.load()) break;
        if (const int rc = poll_inputs(all, inst, st.fatal); rc != 0) {
            release_all();
            return rc;
        }
        for (std::uint32_t n = 0; n < N; ++n) {
            FlowStats& f = st.flows[n];
            FlowScan& sc = fs[n];
            if (n < vin.size()) {
                runtime::InputSlot& v = vin[n];
                mxlGrainInfo info{};
                const std::uint32_t* p = nullptr;
                const mxlStatus s = v.state() == runtime::InputSlot::State::Attached ? v.get(i, 0, info, p) : MXL_ERR_FLOW_NOT_FOUND;
                if (s == MXL_ERR_OUT_OF_RANGE_TOO_LATE) f.inc("too_late");
                else if (s != MXL_STATUS_OK) f.inc("missing");
                else if (info.flags & MXL_GRAIN_FLAG_INVALID) f.inc("invalid");
                else {
                    const bool cuda = v.in().cuda;
                    bool ok = cuda ? cudaMemcpy(head.data(), p, head_words * 4, cudaMemcpyDeviceToHost) == cudaSuccess
                                   : (std::memcpy(head.data(), p, head_words * 4), true);
                    const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
                    const std::uint64_t h = v.head();
                    if (ok && lapped(std::max(h == MXL_UNDEFINED_INDEX ? 0 : h, now_idx), i, gc)) ok = false, f.inc("sample_lapped");
                    if (!ok) {
                        if (!cuda) f.inc("read_errors");
                    } else {
                        f.inc("present");
                        f.inc("samples");
                        const std::uint32_t mk = media::ref::decode_marker(head.data(), words);
                        const std::uint32_t m = mk & 0xFFFFFFu;
                        if (inter && ((mk >> 24) & 1u) != (i & 1u)) f.inc("parity_mismatch");
                        const std::int64_t d = wrap24(static_cast<std::int64_t>(i) - static_cast<std::int64_t>(m));
                        if (!sc.have_d || d != sc.d) {
                            if (sc.have_d) {
                                f.inc("map_changes");
                                if (sc.map_changes.size() < 2000) sc.map_changes.emplace_back(static_cast<std::int64_t>(i), d - sc.d);
                            }
                            sc.have_d = true;
                            sc.d = d;
                        }
                        if (sc.last_i == static_cast<std::int64_t>(i) - 1) {
                            const std::int64_t step = wrap24(static_cast<std::int64_t>(m) - static_cast<std::int64_t>(sc.last_m));
                            if (step != 1) {
                                f.inc("marker_jumps");
                                if (sc.jumps.size() < 2000) sc.jumps.emplace_back(static_cast<std::int64_t>(i), step);
                            }
                        }
                        sc.last_i = static_cast<std::int64_t>(i);
                        sc.last_m = m;
                        const std::uint32_t F = inter ? m / 2 : m;
                        if (F % 25 == 0 && (!inter || (m & 1u) == 0) && sc.vpulses.size() < 100000)
                            sc.vpulses.emplace_back(static_cast<std::int64_t>(i), static_cast<std::int64_t>(vramflow::sample_begin(g, i)));
                        const bool want_full = (!ref_crc.empty() && (i + n) % o.every == 0) || (!o.dump_dir.empty() && sc.dumped < o.dump_count);
                        if (want_full) {
                            const bool fok = cuda ? cudaMemcpy(full.data(), p, gwords * 4, cudaMemcpyDeviceToHost) == cudaSuccess
                                                  : (std::memcpy(full.data(), p, gwords * 4), true);
                            if (fok && !ref_crc.empty() && (i + n) % o.every == 0) {
                                auto it = ref_crc.find(mk);
                                if (it == ref_crc.end()) f.inc("ref_marker_unknown");
                                else if (crc32c(reinterpret_cast<const std::uint8_t*>(full.data()), gwords * 4) != it->second) f.inc("dec_mismatch");
                                else f.inc("dec_bitexact");
                            }
                            if (fok && !o.dump_dir.empty() && sc.dumped < o.dump_count) {
                                const std::string path = o.dump_dir + "/ingest_" + std::to_string(n) + "_m" + std::to_string(m) + ".v210";
                                if (FILE* df = std::fopen(path.c_str(), "wb")) {
                                    std::fwrite(full.data(), 4, gwords, df);
                                    std::fclose(df);
                                    ++sc.dumped;
                                    f.inc("dumped");
                                }
                            }
                        }
                    }
                }
            }
            if (n < ain.size()) {
                const std::uint64_t s0 = vramflow::sample_begin(g, i), s1 = vramflow::sample_begin(g, i + 1);
                const auto count = static_cast<std::uint32_t>(s1 - s0);
                const mxlStatus as = ain[n].read(s1, count, l.data(), r.data());
                if (as == MXL_ERR_OUT_OF_RANGE_TOO_EARLY) f.inc("audio_missing");
                else if (as == MXL_ERR_OUT_OF_RANGE_TOO_LATE) f.inc("audio_too_late");
                else if (as != MXL_STATUS_OK) f.inc("audio_read_errors");
                else {
                    f.inc("audio_blocks");
                    if (o.video == false) f.inc("samples");
                    for (std::uint32_t k = 0; k < count; ++k) sc.a.feed(static_cast<std::int64_t>(s0 + k), l[k]);
                }
            }
        }
        if (st.first_index == ~0ull) st.first_index = i;
        st.last_index = i;
        ++i;
        const std::uint64_t now_idx = mxlGetCurrentIndex(&rate);
        if (now_idx > i + gc - 3 && now_idx > i + lag) {
            for (auto& f : st.flows) f.inc("verify_skipped", now_idx - (i + lag));
            i = now_idx - lag;
        }
    }
    // A/V poros, trūkių filtras (ne prie impulsų), santrauka
    extra = "\"ingest\":[";
    for (std::uint32_t n = 0; n < N; ++n) {
        FlowScan& sc = fs[n];
        FlowStats& f = st.flows[n];
        std::vector<std::int64_t> disc;
        for (const std::int64_t p : sc.a.disc) {
            bool near = false;
            for (const std::int64_t q : sc.a.pulses)
                if (std::llabs(p - q) <= 3000) near = true;
            if (!near) disc.push_back(p);
        }
        std::int64_t worst = 0, sum = 0;
        for (const std::int64_t p : sc.a.pulses) {
            const std::pair<std::int64_t, std::int64_t>* best = nullptr;
            for (const auto& v : sc.vpulses)
                if (!best || std::llabs(p - v.second) < std::llabs(p - best->second)) best = &v;
            if (!best || std::llabs(p - best->second) >= 24000) {
                f.inc("audio_pulse_unpaired");
                continue;
            }
            const std::int64_t off_us = (p - best->second) * 1'000'000 / 48000;
            sc.pairs.emplace_back(best->first, off_us);
            worst = std::max<std::int64_t>(worst, std::llabs(off_us));
            sum += off_us;
        }
        f.c["video_pulses"] = sc.vpulses.size();
        f.c["audio_pulses"] = sc.a.pulses.size();
        f.c["av_pairs"] = sc.pairs.size();
        f.c["av_max_abs_us"] = static_cast<std::uint64_t>(worst);
        f.c["audio_zero_runs"] = sc.a.zero.size();
        f.c["audio_disc"] = disc.size();
        extra += std::string(n ? "," : "") + "{\"flow\":" + std::to_string(n) + ",\"first_index\":" + std::to_string(i0) +
                 ",\"av_mean_us\":" + std::to_string(sc.pairs.empty() ? 0 : sum / static_cast<std::int64_t>(sc.pairs.size())) +
                 ",\"pairs\":" + pairs_json(sc.pairs, 4000) + ",\"map_changes\":" + pairs_json(sc.map_changes, 400) +
                 ",\"jumps\":" + pairs_json(sc.jumps, 400) + ",\"disc\":" + list_json(disc, 200) + ",\"zero\":" + list_json(sc.a.zero, 200) + "}";
        if (n < vin.size()) f.c["reattached"] = vin[n].reattached();
    }
    extra += "]";
    release_all();
    return 0;
}

}  // namespace mxlv::func
