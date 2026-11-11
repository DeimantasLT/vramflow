// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// vramflow-func — GPU media funkcijos ant MXL (PR #653 cuda-linear): gen, logo, verify, enc, dec.
// Architektūra: docs/architektura/2026-11-10-gpu-media-funkcijos.md
#include <CLI/CLI.hpp>

#include "func/func_common.hpp"

namespace mxlv::func {
int run_gen(const Opts&);
int run_logo(const Opts&);
int run_verify(const Opts&);
int run_enc(const Opts&);
int run_dec(const Opts&);
int run_clip(const Opts&, std::uint32_t frames, std::uint64_t first_field);
int run_ref_crc(const std::string& path);
}  // namespace mxlv::func

using namespace mxlv::func;

namespace {
void common(CLI::App* c, Opts& o) {
    c->add_option("--domain", o.domain)->required();
    c->add_option("--flows", o.flows)->check(CLI::Range(1u, 128u));
    c->add_option("--flow-base", o.flow_base);
    c->add_option("--run-seed", o.run_seed);
    c->add_option("--duration", o.duration_s);
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
}
}  // namespace

int main(int argc, char** argv) {
    install_signals();
    CLI::App app{"vramflow-func — GPU media funkcijos ant MXL"};
    app.require_subcommand(1);
    Opts o;
    auto* g = app.add_subcommand("gen", "N × 1080i50 v210 (+ audio) generatorius");
    common(g, o);
    g->add_option("--audio", o.audio);
    g->add_option("--video", o.video);
    auto* l = app.add_subcommand("logo", "logotipo įkepimas MXL → MXL GPU");
    common(l, o);
    l->add_option("--inject-every", o.inject_every);
    l->add_option("--deadline-ms", o.deadline_ms);
    l->add_option("--read-offset-us", o.read_offset_us);
    auto* v = app.add_subcommand("verify", "išėjimo flow'ų patikra su CPU etalonu");
    common(v, o);
    v->add_option("--mode", o.mode, "logo | gen | dec");
    v->add_option("--every", o.every);
    v->add_option("--lag-fields", o.lag_fields);
    v->add_option("--ref", o.ref_crc);
    auto* e = app.add_subcommand("enc", "MXL → H.264 + AAC MPEG-TS per UDP");
    common(e, o);
    e->add_option("--encoder", o.encoder, "nvenc | x264");
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
    auto* d = app.add_subcommand("dec", "H.264 TS per UDP → NVDEC → MXL");
    common(d, o);
    d->add_option("--port-base", o.port_base);
    d->add_option("--playout-frames", o.playout_frames);
    std::uint32_t clip_frames = 250;
    std::uint64_t clip_first = 1000;
    auto* cl = app.add_subcommand("clip", "bandomojo klipo raw NV12 kadrai į stdout (CPU etalonas)");
    cl->add_option("--frames", clip_frames);
    cl->add_option("--first-field", clip_first, "pirmo lauko indeksas (lyginis)");
    cl->add_option("--complexity", o.complexity);
    cl->add_option("--run-seed", o.run_seed);
    cl->add_option("--flow-base", o.flow_base);
    std::string ref_path;
    auto* rc = app.add_subcommand("ref-crc", "raw NV12 → laukų CRC32C etalonas (JSON)");
    rc->add_option("nv12", ref_path)->required();
    CLI11_PARSE(app, argc, argv);
    if (o.impl == "cpu") o.storage = "host";  // CPU variantas visada su host flow'ais
    if (app.got_subcommand("clip")) return run_clip(o, clip_frames, clip_first & ~1ull);
    if (app.got_subcommand("ref-crc")) return run_ref_crc(ref_path);
    if (app.got_subcommand("gen")) return run_gen(o);
    if (app.got_subcommand("logo")) return run_logo(o);
    if (app.got_subcommand("verify")) return run_verify(o);
    if (app.got_subcommand("enc")) return run_enc(o);
    if (app.got_subcommand("dec")) return run_dec(o);
    return 1;
}
