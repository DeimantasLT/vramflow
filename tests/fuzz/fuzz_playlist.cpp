// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Fuzz taikinys: HLS grojaraščio skaitymas ir patikra (mxlv::ingest::parse_playlist, check_playlist, resolve_playlist_url ir
// libav atminties sąmatos — src/ingest/url_policy.cpp). Grojaraštį teikia nuotolinis serveris (planas 2026-11-17 §1.5, S2).
// Invariantai: priimtas grojaraštis neviršija ribų; atmetimas turi priežastį; priežastyje nėra bazinio URL paslapčių.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "ingest/url_policy.hpp"

namespace in = mxlv::ingest;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string text(reinterpret_cast<const char*>(data), size);
    const std::string base = "https://user:Zq9SECRETq9Z@media.example.test/live/index.m3u8?token=Zq9SECRETq9Z";
    const in::PlaylistInfo pi = in::parse_playlist(text);
    const in::HlsLimits lim;
    for (const bool allow_master : {true, false}) {
        std::string why;
        if (in::check_playlist(base, pi, allow_master, lim, why)) {
            if (pi.master && !allow_master) std::abort();
            if (pi.variants.size() + 1 > lim.max_playlists) std::abort();
            if (pi.refs.size() > lim.max_segments) std::abort();
            if (pi.target_s > lim.max_target_s) std::abort();
        } else {
            if (why.empty()) std::abort();
            if (text.find("SECRET") == std::string::npos && why.find("SECRET") != std::string::npos) std::abort();
        }
    }
    (void)in::init_section_bytes(base, pi);
    (void)in::segment_bytes(base, pi);
    for (const std::string& ref : pi.refs) {
        std::string out, why;
        if (in::resolve_playlist_url(base, ref, out, why)) {
            const std::string s = in::url_scheme(out);
            if (s != "http" && s != "https") std::abort();
            if (in::playlist_multiplicity(base, pi, out) < 1) std::abort();
        } else if (why.empty()) {
            std::abort();
        }
    }
    return 0;
}
