// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// ograf_feed — adapteris: RGBA kadrai (iš renderio vamzdžio arba sintetinio šaltinio) -> libvramflow-gfx ritmo režimas
// (vfgfx_submit) -> standartinis MXL `video/v210a` flow (1920x1080i50, grain = laukas). Naudoja TIK viešą
// <vramflow/gfx.h> ir MXL C API.
//
// Šaltiniai (--source):
//   stdin      — neapdoroti kadrai iš stdin, W*H*4 baitai kiekvienas, --format rgba | bgra (GStreamer
//                `video/x-raw,format=BGRA` -> --format bgra), straight alfa. Pvz. `node render_edge.mjs | ograf_feed ...`.
//                Pasibaigus stdin paskutinis kadras toliau commit'inamas kiekvieną lauką iki --seconds.
//   synthetic  — įmontuotas animuotas šaltinis su minkšta (gradientine, antialiasuota) alfa; renderio nereikia.
//
// Naudojimas: ograf_feed --domain D --flow-id UUID [--source stdin|synthetic] [--format rgba|bgra] [--alpha straight|premultiplied] [--seconds 20]
#include <vramflow/gfx.h>

#include <mxl/mxl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int W = 1920, H = 1080;

std::string flow_def(const std::string& id) {
    char b[2048];
    std::snprintf(b, sizeof b,
                  R"({"id":"%s","description":"OGraf example","label":"OGraf example",)"
                  R"("tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-ograf:Gfx"]},)"
                  R"("format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210a",)"
                  R"("grain_rate":{"numerator":25,"denominator":1},"frame_width":%d,"frame_height":%d,)"
                  R"("interlace_mode":"interlaced_tff","colorspace":"BT709","components":[)"
                  R"({"name":"Y","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cb","width":%d,"height":%d,"bit_depth":10},)"
                  R"({"name":"Cr","width":%d,"height":%d,"bit_depth":10}]})",
                  id.c_str(), W, H, W, H, W / 2, H, W / 2, H);
    return b;
}

// Sintetinis kadras: judanti suapvalinta plokštė (antialiasuoti kraštai), minkštas šešėlis, horizontalus alfa gradientas.
void synth(std::vector<std::uint8_t>& px, int t) {
    std::fill(px.begin(), px.end(), 0);
    const float cx = 700.f + 300.f * std::sin(t * 0.05f), cy = 700.f, hw = 450.f, hh = 110.f, rad = 40.f;
    for (int y = 300; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float dx = std::max(std::fabs(x + .5f - cx) - (hw - rad), 0.f), dy = std::max(std::fabs(y + .5f - cy) - (hh - rad), 0.f);
            const float d = std::sqrt(dx * dx + dy * dy) - rad;  // < 0 = plokštės viduje
            const float body = std::clamp(0.5f - d, 0.f, 1.f);  // antialias ~1 px
            const float shadow = d > 0 ? 0.6f * std::exp(-d / 28.f) : 0.6f;
            const float grad = 0.55f + 0.45f * (1.f - (x - (cx - hw)) / (2 * hw));
            const float a = body * std::clamp(grad, 0.f, 1.f) + (1.f - body) * shadow * (d < 160.f ? 1.f : 0.f);
            if (a <= 0.002f) continue;
            std::uint8_t* p = &px[(static_cast<size_t>(y) * W + x) * 4];
            const bool in = body > 0.f;
            p[0] = in ? 20 : 0; p[1] = in ? 140 : 0; p[2] = in ? 240 : 0;  // straight RGB; šešėlis juodas
            p[3] = static_cast<std::uint8_t>(std::lround(std::clamp(a, 0.f, 1.f) * 255.f));
        }
}
}  // namespace

int main(int argc, char** argv) {
    std::string domain, id, source = "synthetic", format = "rgba", alpha = "straight";
    double seconds = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--domain") domain = v; else if (k == "--flow-id") id = v; else if (k == "--source") source = v;
        else if (k == "--format") format = v; else if (k == "--alpha") alpha = v; else if (k == "--seconds") seconds = std::atof(v.c_str());
        else { std::fprintf(stderr, "nežinomas parametras %s\n", k.c_str()); return 2; }
    }
    if (domain.empty() || id.empty() || (source != "stdin" && source != "synthetic") || (format != "rgba" && format != "bgra") || (alpha != "straight" && alpha != "premultiplied")) {
        std::fprintf(stderr, "naudojimas: ograf_feed --domain D --flow-id UUID [--source stdin|synthetic] [--format rgba|bgra] [--alpha straight|premultiplied] [--seconds N]\n");
        return 2;
    }
    mxlInstance inst = mxlCreateInstance(domain.c_str(), "");
    vfgfx_writer g = nullptr;
    int rc = inst ? vfgfx_writer_create(inst, flow_def(id).c_str(), nullptr, &g) : 1;
    if (rc) { std::fprintf(stderr, "vfgfx_writer_create=%d: %s\n", rc, vfgfx_last_error(nullptr)); return 1; }
    if ((rc = vfgfx_start_pacing(g, 0))) { std::fprintf(stderr, "start_pacing=%d: %s\n", rc, vfgfx_last_error(g)); return 1; }

    std::vector<std::uint8_t> px(static_cast<size_t>(W) * H * 4);
    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    int frames = 0;
    auto submit = [&] {
        if (format == "bgra")
            for (size_t i = 0; i < px.size(); i += 4) std::swap(px[i], px[i + 2]);  // BGRA -> RGBA
        vfgfx_frame f{px.data(), W, H, W * 4, alpha == "straight" ? VFGFX_ALPHA_STRAIGHT : VFGFX_ALPHA_PREMULTIPLIED, VFGFX_MEM_HOST};
        if ((rc = vfgfx_submit(g, &f))) { std::fprintf(stderr, "submit=%d: %s\n", rc, vfgfx_last_error(g)); return false; }
        ++frames;
        return true;
    };
    bool ok = true;
    if (source == "stdin") {
        while (ok && elapsed() < seconds) {
            size_t got = 0;
            while (got < px.size()) {
                const size_t n = std::fread(px.data() + got, 1, px.size() - got, stdin);
                if (n == 0) break;
                got += n;
            }
            if (got < px.size()) break;  // EOF: laikom paskutinį kadrą
            ok = submit();
        }
        while (ok && elapsed() < seconds) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } else {
        for (int t = 0; ok && elapsed() < seconds; ++t) {
            synth(px, t);
            ok = submit();
            const double next = (t + 1) / 25.0;  // 25 kadrų/s
            if (next > elapsed()) std::this_thread::sleep_for(std::chrono::duration<double>(next - elapsed()));
        }
    }
    vfgfx_stop_pacing(g);
    std::printf("ograf_feed: source=%s submitted=%d committed=%llu\n", source.c_str(), frames,
                static_cast<unsigned long long>(vfgfx_committed(g)));
    vfgfx_writer_destroy(g, inst);
    mxlDestroyInstance(inst);
    return ok && frames > 0 ? 0 : 1;
}
