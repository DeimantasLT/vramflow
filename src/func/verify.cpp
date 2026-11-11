// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// verify — išėjimo flow'ų patikra atskirame procese (architektūra GPU funkcijų §5.5, §7.1):
//  * kiekvienas indeksas kiekvienam flow'ui: ok / invalid / too_late / missing (aprėptis — niekas nedingsta tyliai);
//  * kas K-tą lauką (išdėstyta per flow'us): D2H ir palyginimas su NEPRIKLAUSOMU CPU etalonu, 0 tolerancija:
//      mode=logo — ref::logo_field(įėjimas) == išėjimas; už logotipo stačiakampio išėjimas == įėjimas bitas į bitą;
//      mode=gen  — ref::gen_field == laukas;
//    abiem — barkodas (indeksas, paritetas, flow) == laukiamas.
// Lap patikra po D2H: jei writer'is galėjo perrašyti slot'ą kopijavimo metu, imtis atmetama (sample_lapped).
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/crc32c.hpp"
#include "common/index_math.hpp"

#include <map>
#include "func/flows.hpp"
#include "func/func_common.hpp"
#include "media/media_ref.hpp"

namespace mxlv::func {

bool load_logo(const Opts& o, media::ref::Logo& logo, std::string& err);

namespace {
bool copy_field(const VideoIn& v, const std::uint32_t* p, std::vector<std::uint32_t>& host) {
    host.resize(media::kFieldWords);
    if (v.cuda) return cudaMemcpy(host.data(), p, media::kFieldBytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    std::memcpy(host.data(), p, media::kFieldBytes);
    return true;
}
}  // namespace

int run_verify(const Opts& o) {
    Stats st;
    st.flows.resize(o.flows);
    auto& stop = stop_flag();
    auto finish = [&](int rc) {
        st.exit_code = rc;
        write_summary(o, "verify-" + o.mode, st);
        std::fprintf(stderr, "[verify %s] baigta, exit %d %s\n", o.mode.c_str(), rc, st.fatal.c_str());
        return rc;
    };
    // CPU varianto (host flow'ų) patikra CUDA nekuria — kitaip verify proceso GPU kontekstas iškreiptų C matavimą
    if (o.impl != "cpu") {
        const int dev = cuda_device(o.device_uuid);
        if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) { st.fatal = "CUDA įrenginys"; return finish(2); }
    }
    const bool logo_mode = o.mode == "logo";
    const bool dec_mode = o.mode == "dec";
    if (!logo_mode && !dec_mode && o.mode != "gen") { st.fatal = "nežinomas --mode " + o.mode; return finish(1); }
    // dec: etalonas barkodas → laukų CRC32C iš programinio dekoderio (ref-crc); be --ref tikrinamas tik barkodas
    std::map<std::uint32_t, std::uint32_t> ref_crc;
    if (dec_mode && !o.ref_crc.empty()) {
        FILE* f = std::fopen(o.ref_crc.c_str(), "r");
        if (!f) { st.fatal = "nėra --ref failo"; return finish(1); }
        unsigned mk = 0, crc = 0;
        int ch = 0;
        while ((ch = std::fgetc(f)) != EOF)
            if (ch == '[' && std::fscanf(f, "%u,%u]", &mk, &crc) == 2) ref_crc[mk] = crc;
        std::fclose(f);
        if (ref_crc.empty()) { st.fatal = "tuščias --ref"; return finish(1); }
    }
    std::vector<std::int64_t> seq_offset(o.flows, INT64_MIN);
    // klipo ilgis laukais ir pirmo lauko barkodo indeksas (iš etalono); be etalono — 250 kadrų klipas
    const std::int64_t kLoopFields = ref_crc.empty() ? 500 : static_cast<std::int64_t>(ref_crc.size());
    std::int64_t ref_k0 = INT64_MAX;
    for (const auto& [mk, crc] : ref_crc) ref_k0 = std::min<std::int64_t>(ref_k0, mk & 0xFFFFFFu);
    media::ref::Logo logo;
    std::string err;
    if (logo_mode && !load_logo(o, logo, err)) { st.fatal = err; return finish(1); }

    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; return finish(2); }
    const int out_role = logo_mode ? kRoleLogo : (dec_mode ? kRoleDec : kRoleGen);
    std::vector<VideoIn> out(o.flows), in(logo_mode ? o.flows : 0);
    int rc = 0;
    for (std::uint32_t n = 0; n < o.flows && !rc; ++n)
        if (!out[n].open(inst, flow_id(out_role, o.flow_base + n), 20000, stop, err, &vramflow::gfx::kHd1080i)) { st.fatal = err; rc = 4; }
    for (std::uint32_t n = 0; n < in.size() && !rc; ++n)
        if (!in[n].open(inst, flow_id(o.in_role, o.flow_base + n), 20000, stop, err, &vramflow::gfx::kHd1080i)) { st.fatal = err; rc = 4; }
    if (rc) { mxlDestroyInstance(inst); return finish(rc); }

    const std::uint32_t gc = out[0].layout.slotCount;
    std::vector<std::uint32_t> ho, hi, ref(media::kFieldWords);
    const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    const std::uint64_t warm_idx = mxlGetCurrentIndex(&kFieldRate) + static_cast<std::uint64_t>(o.warmup_s * 50);
    std::uint64_t i = warm_idx;
    // logotipo stačiakampis grupėmis (už jo išėjimas turi sutapti su įėjimu)
    const int g0 = o.logo_x / 6, g1 = (o.logo_x + logo.w + 5) / 6;

    while (!stop.load() && now_tai() < end_tai) {
        // laukiam, kol laukas i bus seniai commit'intas (lag), bet dar ring'e (lag < gc-2)
        // verify sąmoningai atsilieka lag_fields laukų, todėl ilgas laukimas čia nėra laikrodžio šuolis
        wait_until(i + o.lag_fields, 0, stop);
        if (stop.load()) break;
        for (std::uint32_t n = 0; n < o.flows; ++n) {
            mxlGrainInfo info{};
            const std::uint32_t* p = nullptr;
            const mxlStatus s = out[n].get(i, 0, info, p);
            auto& f = st.flows[n];
            if (s == MXL_ERR_OUT_OF_RANGE_TOO_EARLY) { f.inc("missing"); continue; }
            if (s == MXL_ERR_OUT_OF_RANGE_TOO_LATE) { f.inc("too_late"); continue; }
            if (s != MXL_STATUS_OK) { f.inc("read_errors"); continue; }
            if (info.flags & MXL_GRAIN_FLAG_INVALID) { f.inc("invalid"); continue; }
            f.inc("present");
            if ((i + n) % o.every != 0) continue;
            // imtis
            if (!copy_field(out[n], p, ho)) { f.inc("cuda_errors"); continue; }
            const std::uint32_t* ip = nullptr;
            if (logo_mode) {
                mxlGrainInfo ii{};
                if (in[n].get(i, 0, ii, ip) != MXL_STATUS_OK || (ii.flags & MXL_GRAIN_FLAG_INVALID) || !copy_field(in[n], ip, hi)) {
                    f.inc("sample_in_unavailable");
                    continue;
                }
            }
            const std::uint64_t now_idx = mxlGetCurrentIndex(&kFieldRate);
            const std::uint64_t ho_head = out[n].head();
            const std::uint64_t hi_head = logo_mode ? in[n].head() : 0;
            if (lapped(std::max(ho_head == MXL_UNDEFINED_INDEX ? 0 : ho_head, now_idx), i, gc) ||
                (logo_mode && lapped(std::max(hi_head == MXL_UNDEFINED_INDEX ? 0 : hi_head, now_idx), i, gc))) {
                f.inc("sample_lapped");
                continue;
            }
            if (dec_mode) {
                // Po kodavimo/dekodavimo tikslaus turinio neturime — tikrinam barkodą (paritetas, tolydi seka) ir,
                // jei yra etalonas, CRC32C bitų tikslumu prieš programinį dekoderį.
                const std::uint32_t mk = media::ref::decode_marker(ho.data());
                f.inc("samples");
                if (((mk >> 24) & 1u) != (i & 1u)) f.inc("parity_mismatch");
                const std::int64_t off = (static_cast<std::int64_t>(mk & 0xFFFFFFu) - static_cast<std::int64_t>(i % (1u << 24)) +
                                          (kLoopFields << 30)) % kLoopFields;
                // Poslinkio pokytis (paritetas teisingas) = kadras praleistas/pakartotas atkūrime — pajėgumo įvykio
                // pasekmė (jis jau užskaitytas dec pusėje), todėl informacinis. Lauko tvarkos klaida — parity_mismatch.
                if (seq_offset[n] == INT64_MIN) seq_offset[n] = off;
                else if (off != seq_offset[n]) { f.inc("seq_jump"); seq_offset[n] = off; }
                if (!ref_crc.empty()) {
                    // klipo kilpos sandūra (siuntėjo -stream_loop): ±50 laukų nuo klipo pradžios/pabaigos vertinama atskirai
                    const std::int64_t idx = static_cast<std::int64_t>(mk & 0xFFFFFFu) - ref_k0;
                    const bool near_loop = idx < 50 || idx >= static_cast<std::int64_t>(ref_crc.size()) - 50;
                    auto it = ref_crc.find(mk);
                    if (it == ref_crc.end()) f.inc("ref_marker_unknown");
                    else if (crc32c(reinterpret_cast<const std::uint8_t*>(ho.data()), media::kFieldBytes) != it->second) {
                        f.inc(near_loop ? "dec_mismatch_near_loop" : "dec_mismatch");
                        // diagnostika: pirmi 24 neatitinkantys laukai į failą (klipo indeksas, MXL indeksas)
                        static int dumped = 0;
                        if (dumped < 24) {
                            ++dumped;
                            const std::string p = o.out + "/mismatch_clip" + std::to_string(idx) + "_i" + std::to_string(i) + ".v210";
                            if (FILE* df = std::fopen(p.c_str(), "wb")) {
                                std::fwrite(ho.data(), 1, media::kFieldBytes, df);
                                std::fclose(df);
                            }
                        }
                    }
                    else f.inc(near_loop ? "dec_bitexact_near_loop" : "dec_bitexact");
                }
                continue;
            }
            if (logo_mode) media::ref::logo_field(hi.data(), ref.data(), logo, o.logo_x, o.logo_y, static_cast<int>(i & 1));
            else media::ref::gen_field({o.run_seed, o.flow_base + n, o.complexity}, i, ref.data());
            std::uint64_t bad = 0, outside = 0;
            for (std::uint32_t w = 0; w < media::kFieldWords; ++w) {
                if (ho[w] != ref[w]) ++bad;
                if (logo_mode) {
                    const int j = static_cast<int>(w / media::kWordsPerLine);
                    const int g = static_cast<int>((w % media::kWordsPerLine) / 4);
                    const int fy = 2 * j + static_cast<int>(i & 1);
                    const bool inside = fy >= o.logo_y && fy < o.logo_y + logo.h && g >= g0 && g < g1;
                    if (!inside && ho[w] != hi[w]) ++outside;
                }
            }
            f.inc("samples");
            if (bad) { f.inc("mismatch_fields"); f.inc("mismatch_words", bad); }
            if (outside) { f.inc("outside_diff_fields"); f.inc("outside_diff_words", outside); }
            const std::uint32_t mk = media::ref::decode_marker(ho.data());
            if (mk != media::marker_word(i, o.flow_base + n)) f.inc("marker_bad");
        }
        if (st.first_index == ~0ull) st.first_index = i;
        st.last_index = i;
        ++i;
        const std::uint64_t now_idx = mxlGetCurrentIndex(&kFieldRate);
        if (now_idx > i + gc - 3) {  // atsilikom per daug — praleidžiam (užskaitoma)
            for (auto& f : st.flows) f.inc("verify_skipped", now_idx - (i + o.lag_fields));
            i = now_idx - o.lag_fields;
        }
    }
    for (auto& v : out) v.release(inst);
    for (auto& v : in) v.release(inst);
    mxlDestroyInstance(inst);
    std::uint64_t verify_err = 0, capacity = 0, samples = 0;
    for (auto& f : st.flows) {
        verify_err += f.c["mismatch_fields"] + f.c["outside_diff_fields"] + f.c["marker_bad"] + f.c["cuda_errors"] +
                      f.c["parity_mismatch"] + f.c["dec_mismatch"] + f.c["dec_mismatch_near_loop"] + f.c["ref_marker_unknown"];
        capacity += f.c["missing"] + f.c["too_late"] + f.c["invalid"] + f.c["read_errors"] + f.c["verify_skipped"];
        samples += f.c["samples"];
    }
    st.proc["verification_errors"] = verify_err;
    st.proc["capacity_errors"] = capacity;
    st.proc["samples_total"] = samples;
    if (samples == 0) st.proc["no_samples"] = 1;
    return finish(verify_err ? 3 : (capacity || samples == 0) ? 4 : 0);
}

}  // namespace mxlv::func
