// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// verify — išėjimo flow'ų patikra atskirame procese (architektūra GPU funkcijų §5.5, §7.1):
//  * kiekvienas indeksas kiekvienam flow'ui: ok / invalid / too_late / missing (aprėptis — niekas nedingsta tyliai);
//  * kas K-tą lauką (išdėstyta per flow'us): D2H ir palyginimas su NEPRIKLAUSOMU CPU etalonu, 0 tolerancija:
//      mode=logo — ref::logo_field(įėjimas) == išėjimas; už logotipo stačiakampio išėjimas == įėjimas bitas į bitą;
//      mode=gen  — ref::gen_field == laukas;
//      mode=composite — išėjimo laukas i == ref::composite(gen_field bazė, gfx_content → rgba_to_v210a sluoksniai), CRC32C;
//                       prielaida: sluoksniai iš gfxgen (--layer-geom, --gfx-mode, --layer-xy, --opacity, --base, --out-format
//                       kaip composite / gfxgen); grandinės (--layer-flow) nepalaikomos;
//    abiem — barkodas (indeksas, paritetas, flow) == laukiamas.
// Lap patikra po D2H: jei writer'is galėjo perrašyti slot'ą kopijavimo metu, imtis atmetama (sample_lapped).
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/crc32c.hpp"
#include "common/index_math.hpp"
#include "gfx/gfx_content.hpp"
#include "gfx/gfx_ref.hpp"

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

namespace {
// Skaitikliai -> exit kodas ir proc santrauka (bendra visiems režimams).
int summarize(Stats& st) {
    std::uint64_t verify_err = 0, capacity = 0, samples = 0;
    for (auto& f : st.flows) {
        verify_err += f.c["mismatch_fields"] + f.c["outside_diff_fields"] + f.c["marker_bad"] + f.c["cuda_errors"] +
                      f.c["parity_mismatch"] + f.c["dec_mismatch"] + f.c["dec_mismatch_near_loop"] + f.c["ref_marker_unknown"] +
                      f.c["expect_failed"];
        capacity += f.c["missing"] + f.c["too_late"] + f.c["invalid"] + f.c["read_errors"] + f.c["verify_skipped"];
        samples += f.c["samples"];
    }
    st.proc["verification_errors"] = verify_err;
    st.proc["capacity_errors"] = capacity;
    st.proc["samples_total"] = samples;
    if (samples == 0) st.proc["no_samples"] = 1;
    return verify_err ? 3 : (capacity || samples == 0) ? 4 : 0;
}

// --mode composite. Laukiamas išėjimas perskaičiuojamas CPU etalonu: bazė ref::gen_field (flow_base + n), sluoksnis k =
// gfx_content(run_seed, flow_base + n, k, kadras) -> ref::rgba_to_v210a (nelyginis k - premultiplied, kaip gfxgen) ->
// ref::composite. gfxgen kadras: pace - 0; manual - i / 2. Konvertuoti sluoksnių laukai kešuojami pagal (n, k, paritetas,
// kadras), RGBA kadras - pagal (n, k, kadras), todėl pace režimu po pirmo patikrinimo likusieji kainuoja tik gen_field +
// ref::composite. Etalono skaičiavimas (dešimtys ms) vyksta darbinėse gijose (flow n -> gija n % W), kad skaitymo gija nenukristų
// už ring'o: ji tik nukopijuoja faktinį lauką (D2H / memcpy), patikrina lap ir įdeda į eilę. Pilna eilė -> "check_dropped"
// (informacinis: tikrinta mažiau, bet klaidos nėra). Grąžina nenulinį kodą, jei paleisti nepavyko (st.fatal užpildytas).
int verify_composite(const Opts& o, Stats& st) {
    namespace gfx = vramflow::gfx;
    const std::uint32_t N = o.flows, K = o.layers;
    const bool has_base = o.base == "gen", manual = o.gfx_mode == "manual", out_alpha = o.out_format == "v210a";
    if (!has_base && !out_alpha) { st.fatal = "--base none leidžiamas tik su --out-format v210a (R12)"; return 1; }
    if (o.layer_xy.size() != K) { st.fatal = "--layer-xy turi būti nurodytas lygiai --layers kartų"; return 1; }
    if (!o.opacity.empty() && o.opacity.size() != K) { st.fatal = "--opacity: arba nė vieno, arba --layers kartų"; return 1; }
    std::vector<int> lx(K), ly(K);
    std::vector<float> lop(K, 1.0f);
    for (std::uint32_t k = 0; k < K; ++k) {
        if (!parse_xy(o.layer_xy[k], lx[k], ly[k])) { st.fatal = "--layer-xy '" + o.layer_xy[k] + "': tikimasi X,Y"; return 1; }
        if (!o.opacity.empty()) lop[k] = o.opacity[k];
    }
    std::vector<gfx::Raster> geom;
    std::string err;
    if (!parse_layer_geoms(o, geom, err)) { st.fatal = err; return 1; }
    const gfx::Raster base_r = gfx::kHd1080i;
    const gfx::Raster out_r{base_r.w, base_r.h, base_r.interlaced, out_alpha};
    const std::size_t out_bytes = gfx::grain_bytes(out_r);
    auto& stop = stop_flag();

    mxlInstance inst = mxlCreateInstance(o.domain.c_str(), "");
    if (!inst) { st.fatal = "mxlCreateInstance"; return 2; }
    std::vector<VideoIn> out(N);
    int rc = 0;
    for (std::uint32_t n = 0; n < N && !rc; ++n)
        if (!out[n].open(inst, flow_id(kRoleComp, o.flow_base + n), 20000, stop, err, &out_r)) { st.fatal = err; rc = 4; }
    if (rc) { mxlDestroyInstance(inst); return rc; }

    struct LayerCache {
        std::uint64_t rgba_frame = ~0ull, grain_frame[2] = {~0ull, ~0ull};
        std::vector<std::uint8_t> rgba, grain[2];
    };
    struct Job {
        std::uint32_t n;
        std::uint64_t i;
        std::vector<std::uint8_t> act;
    };
    // Darbinė gija: savo flow'ų (n % W == w) kešas, eilė ir lokalūs skaitikliai (sujungiami po join).
    struct Worker {
        std::mutex m;
        std::condition_variable cv;
        std::deque<Job> q;
        bool done = false;
        std::thread t;
        std::vector<FlowStats> fs;  // pagal n
        Histogram expect;
    };
    const std::uint32_t W = std::max<std::uint32_t>(1, std::min<std::uint32_t>(N, o.threads > 0 ? static_cast<std::uint32_t>(o.threads) : 4));
    constexpr std::size_t kMaxQueue = 6;
    std::vector<std::unique_ptr<Worker>> wk;
    std::vector<LayerCache> cache(static_cast<std::size_t>(N) * K);
    std::atomic<int> dumped{0};

    auto check = [&](Worker& w, Job& job, std::vector<std::uint32_t>& base, std::vector<std::uint8_t>& exp, std::vector<gfx::Layer>& lay) {
        FlowStats& f = w.fs[job.n];
        const int parity = static_cast<int>(job.i & 1);
        const std::uint64_t t0 = now_tai();
        const std::uint64_t fr = manual ? job.i / 2 : 0;
        if (has_base) media::ref::gen_field({o.run_seed, o.flow_base + job.n, o.complexity}, job.i, base.data());
        bool ok = true;
        for (std::uint32_t k = 0; k < K && ok; ++k) {
            LayerCache& c = cache[static_cast<std::size_t>(job.n) * K + k];
            const gfx::Raster& g = geom[k];
            if (c.grain_frame[parity] != fr) {
                if (c.rgba_frame != fr) {
                    c.rgba.resize(static_cast<std::size_t>(g.w) * static_cast<std::size_t>(g.h) * 4);
                    gfx::gfx_content(o.run_seed, o.flow_base + job.n, k, fr, g.w, g.h, c.rgba.data());
                    c.rgba_frame = fr;
                }
                c.grain[parity].resize(gfx::grain_bytes(g));
                ok = gfx::ref::rgba_to_v210a(c.rgba.data(), g.w, g.h, static_cast<std::size_t>(g.w) * 4,
                                             (k & 1u) ? gfx::AlphaIn::Premultiplied : gfx::AlphaIn::Straight, g, parity,
                                             c.grain[parity].data());
                c.grain_frame[parity] = ok ? fr : ~0ull;
            }
            lay[k] = gfx::Layer{c.grain[parity].data(), g, lx[k], ly[k], lop[k]};
        }
        ok = ok && gfx::ref::composite(has_base ? reinterpret_cast<const std::uint8_t*>(base.data()) : nullptr, base_r, lay.data(),
                                       static_cast<int>(K), out_r, parity, exp.data());
        if (!ok) {  // parametrai nesutampa su etalono sutartimi - patikra neįmanoma; tai klaida, ne tylus praleidimas
            f.inc("expect_failed");
            return;
        }
        w.expect.add(now_tai() - t0);
        f.inc("samples");
        if (crc32c(exp.data(), out_bytes) == crc32c(job.act.data(), out_bytes)) return;
        std::uint64_t bad = 0;
        for (std::size_t b = 0; b + 4 <= out_bytes; b += 4)
            if (std::memcmp(exp.data() + b, job.act.data() + b, 4) != 0) ++bad;
        f.inc("mismatch_fields");
        f.inc("mismatch_words", bad);
        if (dumped.fetch_add(1) < 8) {  // diagnostika: faktinis ir laukiamas grain'as
            const std::string stem = o.out + "/mismatch_n" + std::to_string(job.n) + "_i" + std::to_string(job.i);
            for (int x = 0; x < 2; ++x)
                if (FILE* df = std::fopen((stem + (x ? ".expected." : ".actual.") + (out_alpha ? "v210a" : "v210")).c_str(), "wb")) {
                    std::fwrite(x ? exp.data() : job.act.data(), 1, out_bytes, df);
                    std::fclose(df);
                }
        }
    };
    for (std::uint32_t w = 0; w < W; ++w) {
        wk.push_back(std::make_unique<Worker>());
        wk[w]->fs.resize(N);
    }
    for (std::uint32_t w = 0; w < W; ++w)
        wk[w]->t = std::thread([&, w] {
            Worker& me = *wk[w];
            std::vector<std::uint32_t> base(media::kFieldWords);
            std::vector<std::uint8_t> exp(out_bytes);
            std::vector<gfx::Layer> lay(K);
            for (;;) {
                Job job;
                {
                    std::unique_lock lk(me.m);
                    me.cv.wait(lk, [&] { return me.done || !me.q.empty(); });
                    if (me.q.empty()) return;  // done ir eilė ištuštinta
                    job = std::move(me.q.front());
                    me.q.pop_front();
                }
                check(me, job, base, exp, lay);
            }
        });

    const std::uint32_t gc = out[0].layout.slotCount;
    const std::uint64_t end_tai = now_tai() + static_cast<std::uint64_t>(o.duration_s * 1e9);
    std::uint64_t i = mxlGetCurrentIndex(&kFieldRate) + static_cast<std::uint64_t>(o.warmup_s * 50);
    std::vector<std::uint8_t> act(out_bytes);
    while (!stop.load() && now_tai() < end_tai) {
        wait_until(i + o.lag_fields, 0, stop);  // verify sąmoningai atsilieka lag_fields laukų
        if (stop.load()) break;
        for (std::uint32_t n = 0; n < N; ++n) {
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
            Worker& w = *wk[n % W];
            {
                std::lock_guard lk(w.m);
                if (w.q.size() >= kMaxQueue) { f.inc("check_dropped"); continue; }
            }
            if (out[n].cuda) {
                if (cudaMemcpy(act.data(), p, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { f.inc("cuda_errors"); continue; }
            } else {
                std::memcpy(act.data(), p, out_bytes);
            }
            const std::uint64_t head = out[n].head();
            if (lapped(std::max(head == MXL_UNDEFINED_INDEX ? 0 : head, mxlGetCurrentIndex(&kFieldRate)), i, gc)) {
                f.inc("sample_lapped");
                continue;
            }
            {
                std::lock_guard lk(w.m);
                w.q.push_back(Job{n, i, act});
            }
            w.cv.notify_one();
        }
        if (st.first_index == ~0ull) st.first_index = i;
        st.last_index = i;
        ++i;
        const std::uint64_t now_idx = mxlGetCurrentIndex(&kFieldRate);
        if (now_idx > i + gc - 3) {  // atsilikom per daug - praleidžiam (užskaitoma)
            for (auto& f : st.flows) f.inc("verify_skipped", now_idx - (i + o.lag_fields));
            i = now_idx - o.lag_fields;
        }
    }
    for (auto& w : wk) {
        {
            std::lock_guard lk(w->m);
            w->done = true;
        }
        w->cv.notify_one();
    }
    for (auto& w : wk) {
        w->t.join();
        st.hist["expect_ms"].merge(w->expect);
        for (std::uint32_t n = 0; n < N; ++n)
            for (const auto& [key, v] : w->fs[n].c) st.flows[n].c[key] += v;
    }
    for (auto& v : out) v.release(inst);
    mxlDestroyInstance(inst);
    return 0;
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
    if (o.mode == "composite") {
        const int rc = verify_composite(o, st);
        return finish(rc ? rc : summarize(st));
    }
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
    return finish(summarize(st));
}

}  // namespace mxlv::func
