// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// vramflow-func — GPU media funkcijos ant MXL (PR #653 cuda-linear): gen, logo, verify, enc, dec.
// Architektūra: docs/architektura/2026-11-10-gpu-media-funkcijos.md
#include <CLI/CLI.hpp>

#include <cstdio>

#include "func/func_common.hpp"

namespace mxlv::func {
int run_gen(const Opts&);
int run_gfxgen(const Opts&);
int run_logo(const Opts&);
int run_composite(const Opts&);
int run_verify(const Opts&);
int run_enc(const Opts&);
int run_dec(const Opts&);
int run_clip(const Opts&, std::uint32_t frames, std::uint64_t first_field);
int run_ref_crc(const Opts&, const std::string& path);
int run_split_raw(const Opts&, const std::string& in, const std::string& out, bool frame_chroma);
}  // namespace mxlv::func

using namespace mxlv::func;

namespace {
void common(CLI::App* c, Opts& o) {
    c->add_option("--domain", o.domain)->required();
    c->add_option("--flows", o.flows)->check(CLI::Range(1u, 128u));
    c->add_option("--flow-base", o.flow_base);
    c->add_option("--run-seed", o.run_seed);
    c->add_option("--duration", o.duration_s,
                  "trukmė, s (numatyta 30). gen, logo, composite, enc, dec: 0 — veikti iki sustabdymo (SIGINT / SIGTERM; "
                  "tarnybos režimas, K-T8); verify, gfxgen: 0 — baigti iškart");
    c->add_option("--warmup", o.warmup_s);
    c->add_option("--out", o.out);
    c->add_option("--device-uuid", o.device_uuid);
    c->add_option("--complexity", o.complexity)->check(CLI::Range(0u, 3u));
    c->add_option("--logo", o.logo, "builtin | kelias į PNG");
    c->add_option("--logo-x", o.logo_x);
    c->add_option("--logo-y", o.logo_y);
    c->add_option("--in-role", o.in_role);
    c->add_option("--impl", o.impl, "gpu | cpu (cpu ⇒ host storage, CUDA nekuriamas)")->check(CLI::IsMember({"gpu", "cpu"}));
    c->add_option("--storage", o.storage, "cuda-linear | host")->check(CLI::IsMember({"cuda-linear", "host"}));
    c->add_option("--threads", o.threads, "CPU varianto gijos (0 = automatiškai)");
    c->add_option("--format", o.format, "flow formatas: 1080i50 (numatytasis), 1080p25, 1080p50, 720p50, 1080i59.94, 2160p50 …");
    c->add_option("--run-id", o.run_id, "paleidimo ID domeno nuomoje <domain>.vramflow/ (numatyta <subkomanda>-<pid>-<8 hex>)");
    // testinės parinktys (paslėptos --help): išėjimo klaidų injekcija (logo, gen; tests/functional startstop fout_error)
    c->add_option("--test-out-error", o.test_out_error)->check(CLI::IsMember({"open", "commit"}))->group("");
    c->add_option("--test-out-error-after", o.test_out_error_after_s)->group("");
    c->add_option("--test-out-error-count", o.test_out_error_count)->group("");
}
// Įėjimus skaitančios funkcijos (logo, composite, verify, enc): laukimas ir formato pasikeitimas po re-attach (planas §P6).
void inputs(CLI::App* c, Opts& o) {
    c->add_option("--input-wait-ms", o.input_wait_ms, "kiek laukti įėjimų paleidžiant: 0 — neribotai (numatyta), N — riba ms (tada exit 4)")
        ->check(CLI::NonNegativeNumber);
    c->add_option("--on-format-change", o.on_format_change, "įėjimas grįžo kitu formatu: exit (exit 6, numatyta) | wait")
        ->check(CLI::IsMember({"exit", "wait"}));
}
}  // namespace

int main(int argc, char** argv) {
    install_signals();
    CLI::App app{"vramflow-func — GPU media funkcijos ant MXL"};
    app.require_subcommand(1);
    Opts o;
    auto* g = app.add_subcommand("gen", "N × v210 (--format, numatytasis 1080i50) (+ audio) generatorius");
    common(g, o);
    g->add_option("--audio", o.audio);
    g->add_option("--video", o.video);
    g->add_option("--audio-late-ms", o.audio_late_ms,
                  "garsas rašomas ne anksčiau kaip T(i) + N ms (realaus laiko garso rašytojo modelis; 0 — iškart po video)");
    auto* gg = app.add_subcommand("gfxgen", "N × K sintetinių v210a grafikos šaltinių (tik libvramflow-gfx)");
    common(gg, o);
    gg->add_option("--layers", o.layers, "sluoksnių (flow'ų) skaičius šaltiniui, 1…8")->check(CLI::Range(1u, 8u));
    gg->add_option("--layer-geom", o.layer_geom, "pvz. logo,full,full,full (logo 384x216, full 1920x1080 arba WxH); numatytoji: k=0 logo, kiti full");
    gg->add_option("--mode", o.gfx_mode, "pace (statinis kadras 0, bibliotekos ritmas) | manual (kadras = index/2)")
        ->check(CLI::IsMember({"pace", "manual"}));
    auto* l = app.add_subcommand("logo", "logotipo įkepimas MXL → MXL GPU");
    common(l, o);
    inputs(l, o);
    l->add_option("--inject-every", o.inject_every);
    l->add_option("--deadline-ms", o.deadline_ms);
    l->add_option("--read-offset-us", o.read_offset_us);
    l->add_option("--test-fault-proc-delay-ms", o.test_fault_proc_delay_ms)->group("");  // izoliacijos vartų meta-testui
    auto* cp = app.add_subcommand("composite", "bazė + K grafikos sluoksnių → v210 / v210a (MXL → MXL, C/H/G)");
    common(cp, o);
    inputs(cp, o);
    cp->add_option("--layers", o.layers, "grafikos sluoksnių skaičius išėjimui, 1…8")->check(CLI::Range(1u, 8u));
    cp->add_option("--layer-xy", o.layer_xy, "X,Y (kadro koordinatės) kiekvienam sluoksniui, kartoti K kartų; X lyginis, interlaced Y lyginis");
    cp->add_option("--opacity", o.opacity, "sluoksnio nepermatomumas 0…1 (K kartų; numatytasis 1)");
    cp->add_option("--base", o.base, "gen (bazė flow_id(--in-role, n)) | none (tik su --out-format v210a)")->check(CLI::IsMember({"gen", "none"}));
    cp->add_option("--out-format", o.out_format, "v210 | v210a")->check(CLI::IsMember({"v210", "v210a"}));
    cp->add_option("--layer-flow", o.layer_flow, "K:UUID — k-tojo (0…K-1) sluoksnio flow ID (visiems išėjimams), pvz. kito composite išėjimas");
    cp->add_option("--deadline-ms", o.deadline_ms);
    cp->add_option("--read-offset-us", o.read_offset_us);
    auto* v = app.add_subcommand("verify", "išėjimo flow'ų patikra su CPU etalonu");
    common(v, o);
    inputs(v, o);
    v->add_option("--mode", o.mode, "logo | gen | dec | composite");
    // --mode composite: tie patys parametrai kaip gfxgen / composite (verify atkuria laukiamą išėjimą CPU etalonu)
    v->add_option("--layers", o.layers, "composite: sluoksnių skaičius išėjimui, 1…8")->check(CLI::Range(1u, 8u));
    v->add_option("--layer-geom", o.layer_geom, "composite: kaip gfxgen (logo,full,… arba WxH); turi sutapti su gfxgen");
    v->add_option("--gfx-mode", o.gfx_mode, "composite: gfxgen režimas pace | manual")->check(CLI::IsMember({"pace", "manual"}));
    v->add_option("--layer-xy", o.layer_xy, "composite: kaip composite (K kartų)");
    v->add_option("--opacity", o.opacity, "composite: kaip composite (K kartų)");
    v->add_option("--base", o.base, "composite: gen | none")->check(CLI::IsMember({"gen", "none"}));
    v->add_option("--out-format", o.out_format, "composite: v210 | v210a")->check(CLI::IsMember({"v210", "v210a"}));
    v->add_option("--every", o.every);
    v->add_option("--lag-fields", o.lag_fields);
    v->add_option("--ref", o.ref_crc);
    v->add_option("--min-chroma-db", o.min_chroma_db, "dec: Cb/Cr PSNR prieš gen etaloną (šaltinio indeksas iš barkodo) >= riba; 0 — netikrinama");
    auto* e = app.add_subcommand("enc", "MXL → H.264 + AAC MPEG-TS per UDP");
    common(e, o);
    inputs(e, o);
    e->add_option("--encoder", o.encoder, "nvenc | x264");
    e->add_option("--deadline-ms", o.deadline_ms);
    e->add_option("--read-offset-us", o.read_offset_us);
    e->add_option("--preset", o.preset);
    e->add_option("--tune", o.tune);
    e->add_option("--x264-preset", o.x264_preset);
    e->add_option("--x264-threads", o.x264_threads);
    e->add_option("--vbitrate", o.vbitrate);
    e->add_option("--abitrate", o.abitrate);
    e->add_option("--muxrate", o.muxrate);
    e->add_option("--port-base", o.port_base);
    e->add_option("--tap-dir", o.tap_dir);
    e->add_option("--tap-flow", o.tap_flow);
    e->add_option("--tap-every", o.tap_every);
    e->add_option("--audio", o.audio);
    // testinės parinktys (paslėptos --help): deterministiniai kadrų atmetimai (tests/functional/avsync.py)
    e->add_option("--test-drop-every", o.test_drop_every)->group("");
    e->add_option("--test-drop-after", o.test_drop_after_s)->group("");
    auto* d = app.add_subcommand("dec", "H.264 TS per UDP → NVDEC → MXL");
    common(d, o);
    d->add_option("--port-base", o.port_base);
    d->add_option("--playout-frames", o.playout_frames);
    d->add_option("--chroma-siting", o.chroma_siting, "interlaced flow'ui: auto (pagal kadro žymę) | field | frame")
        ->check(CLI::IsMember({"auto", "field", "frame"}));
    std::uint32_t clip_frames = 250;
    std::uint64_t clip_first = 1000;
    auto* cl = app.add_subcommand("clip", "bandomojo klipo raw NV12 kadrai į stdout (CPU etalonas)");
    cl->add_option("--frames", clip_frames);
    cl->add_option("--first-field", clip_first, "pirmo lauko indeksas (lyginis)");
    cl->add_option("--complexity", o.complexity);
    cl->add_option("--run-seed", o.run_seed);
    cl->add_option("--flow-base", o.flow_base);
    cl->add_option("--format", o.format, "formatas (progresyviam — kadrai iš kadro grain'ų)");
    std::string conv_chroma = "frame";
    std::string conv_in, conv_out;
    auto* cv = app.add_subcommand("split-raw", "raw NV12 kadrai → v210 kadrai etalonu (chromos PSNR įrankiui)");
    cv->add_option("--in", conv_in)->required();
    cv->add_option("--out", conv_out)->required();
    cv->add_option("--format", o.format, "kadro matmenys (pvz. 1080p50)");
    cv->add_option("--chroma", conv_chroma, "frame (kadro chroma) | field (lauko chroma: kadras kaip du laukai)")
        ->check(CLI::IsMember({"frame", "field"}));
    std::string ref_path;
    auto* rc = app.add_subcommand("ref-crc", "raw NV12 → laukų CRC32C etalonas (JSON)");
    rc->add_option("nv12", ref_path)->required();
    rc->add_option("--format", o.format, "formatas (progresyviam — kadro grain'ų CRC)");
    KeepWarmOpts kw;
    auto* k = app.add_subcommand("keepwarm",
                                 "laiko GPU virš tuščiosios eigos P8 lengva periodine apkrova (nešiojamiems / dev kompiuteriams, kol yra "
                                 "gyvas eteris; docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md)");
    k->add_option("--size-mb", kw.size_mb, "užpildomo VRAM bloko dydis, MB")->check(CLI::Range(1u, 4096u));
    k->add_option("--interval-ms", kw.interval_ms, "užpildymo periodas, ms")->check(CLI::Range(0.1, 10000.0));
    k->add_option("--duration", kw.duration_s, "trukmė, s; 0 — iki sustabdymo (SIGINT / SIGTERM; numatyta)")->check(CLI::NonNegativeNumber);
    k->add_option("--device-uuid", o.device_uuid);
    CLI11_PARSE(app, argc, argv);
    if (app.got_subcommand("keepwarm")) return run_keepwarm(kw, o.device_uuid);  // ne MXL funkcija: be domeno ir nuomos
    {
        std::string why;
        if (!vramflow::parse_format(o.format, o.fmt, why) || !vramflow::format_supported(o.fmt, why)) {
            std::fprintf(stderr, "--format: %s\n", why.c_str());
            return 1;
        }
    }
    if (o.impl == "cpu") o.storage = "host";  // CPU variantas visada su host flow'ais
    if (app.got_subcommand("clip")) return run_clip(o, clip_frames, o.fmt.raster.interlaced ? clip_first & ~1ull : clip_first);
    if (app.got_subcommand("ref-crc")) return run_ref_crc(o, ref_path);
    if (app.got_subcommand("split-raw")) return run_split_raw(o, conv_in, conv_out, conv_chroma == "frame");
    // Domeno nuoma (planas §P6, K8): laikoma iki proceso pabaigos; harness'as domeno netrina, kol ji gyva.
    {
        const std::string sub = app.get_subcommands().front()->get_name();
        // cuda-linear: per ilgas domeno kelias → reader'iai neprisijungtų prie payload.sock ir lauktų be galo — klaida iškart
        if (o.storage == "cuda-linear")
            if (const std::string bad = mxlv::runtime::cuda_domain_path_error(o.domain); !bad.empty()) {
                std::fprintf(stderr, "[%s] --domain: %s\n", sub.c_str(), bad.c_str());
                return 1;
            }
        if (o.run_id.empty()) o.run_id = mxlv::runtime::make_run_id(sub);
        std::string err;
        if (!mxlv::runtime::DomainLease::acquire(o.domain, o.run_id, domain_lease(), err)) {
            std::fprintf(stderr, "[%s] domeno nuoma: %s\n", sub.c_str(), err.c_str());
            return 2;
        }
    }
    if (app.got_subcommand("gen")) return run_gen(o);
    if (app.got_subcommand("gfxgen")) return run_gfxgen(o);
    if (app.got_subcommand("logo")) return run_logo(o);
    if (app.got_subcommand("composite")) return run_composite(o);
    if (app.got_subcommand("verify")) return run_verify(o);
    if (app.got_subcommand("enc")) return run_enc(o);
    if (app.got_subcommand("dec")) return run_dec(o);
    return 1;
}
