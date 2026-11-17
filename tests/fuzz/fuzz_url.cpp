// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Fuzz taikinys: URL politika (mxlv::ingest::source_url_allowed, redact_url, canonical_authority ir kt. —
// src/ingest/url_policy.cpp). URL ateina iš operatoriaus, grojaraščių ir libav žurnalo eilučių (planas 2026-11-17 §1.5, S2).
// Invariantai: leistas šaltinio URL turi tik dokumentuotą schemą; atmetimas turi priežastį; išvalytame URL nėra paslapčių
// (userinfo ir užklausos reikšmių); libav grojaraščio URL telpa į 4095 baitus.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "ingest/url_policy.hpp"

namespace in = mxlv::ingest;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string url(reinterpret_cast<const char*>(data), size);
    std::string why;
    if (in::source_url_allowed(url, why)) {
        const std::string s = in::url_scheme(url);
        if (!s.empty() && s != "udp" && s != "rtp" && s != "srt" && s != "http" && s != "https") std::abort();
    } else if (why.empty()) {
        std::abort();
    }
    (void)in::url_authority(url);
    (void)in::canonical_authority(url);
    (void)in::http_scheme_without_slashes(url);
    (void)in::exact_http_url(url);
    (void)in::redact_urls_in(url);
    if (in::libav_playlist_url(url).size() > 4095) std::abort();
    (void)in::with_srt_passphrase(url, "0123456789abcdef");
    if (in::srt_passphrase_ok(url, why)) (void)in::with_srt_passphrase("srt://host.example.test:9000?mode=caller", url);
    // Paslaptys: userinfo ir užklausos reikšmė su žyme; likusi užklausos dalis — iš fuzzerio (be žymės ir be valdymo simbolių)
    if (url.find("Zq9") == std::string::npos && url.find_first_of(std::string("\0\r\n", 3)) == std::string::npos) {
        const std::string secret = "https://u:Zq9SECRETq9Z@host.example.test/p?k=Zq9SECRETq9Z&" + url;
        if (in::redact_url(secret).find("Zq9SECRETq9Z") != std::string::npos) std::abort();
    }
    return 0;
}
