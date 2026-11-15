// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// URL politika ingest'ui (round 9, A1) — gryna logika be libav (tests/unit/test_url_policy.cpp).
//
//   * operatoriaus šaltinio URL (--input) — patikimas, bet jo schema turi būti dokumentuota: udp, rtp, srt, http, https arba
//     vietinis failas (be schemos); kita (file:, tcp:, concat:, data:, crypto+…, subfile,…) — naudojimo klaida;
//   * kiekvienam libavformat atidarymui (viršutiniam ir įdėtiems: HLS segmentai, RTP → UDP, HTTP → TCP / TLS) — aiškus
//     protocol_whitelist ir format_whitelist pagal viršutinę schemą (zondavimas negali parinkti bet kokio demuxerio);
//   * grojaraščio pateikti URL (variantai, segmentai, URI="…" atributai) — tik http / https po išsprendimo pagal grojaraščio URL;
//   * paslaptys žurnaluose: userinfo ir visos užklausos reikšmės (SRT passphrase=…, žetonai) pakeičiamos „***“.
#include <cstdint>
#include <string>
#include <vector>

namespace mxlv::ingest {

// Schema mažosiomis raidėmis (kaip libavformat url_find_protocol: [A-Za-z0-9+.-]+, po jos ':' arba ','); "" — vietinis kelias.
std::string url_scheme(const std::string& url);

// Operatoriaus šaltinio URL: true — leidžiama; kitaip why — „<schema> protokolas nepalaikomas (leidžiami: …): <URL>“.
bool source_url_allowed(const std::string& url, std::string& why);

// libavformat baltieji sąrašai pagal viršutinio URL schemą (įdėti atidarymai juos paveldi). hls — HTTP(S) šaltinio patikra rado
// M3U8; false (H1) — tik MPEG-TS per HTTP: HLS demuxeris nepasiekiamas, net jei libav paties atidarymas gautų M3U8 (TOCTOU).
std::string protocol_whitelist_for(const std::string& scheme, bool hls = true);
std::string format_whitelist_for(const std::string& scheme, bool hls = true);

// Grojaraščio nuoroda ref, išspręsta pagal grojaraščio URL base (RFC 3986 supaprastintai: absoliuti, "//host…", "/kelias",
// santykinė). true — rezultatas http / https (out); kitaip why — aiški priežastis (schema išvalyta nuo paslapčių).
bool resolve_playlist_url(const std::string& base, const std::string& ref, std::string& out, std::string& why);

// URL be paslapčių: userinfo → „***@“, kiekvienos užklausos poros reikšmė → „***“ (raktas paliekamas), fragmentas → „#***“.
// Vietinis kelias — nepakeistas.
std::string redact_url(const std::string& url);
// Tekste kiekvienas „schema://…“ (iki tarpo ar kabučių) išvalomas redact_url (libav žurnalo eilutėms, klaidų tekstams).
std::string redact_urls_in(const std::string& text);

// HLS grojaraščio (M3U8) santrauka.
struct PlaylistInfo {
    bool m3u = false;     // prasideda #EXTM3U
    bool master = false;  // turi #EXT-X-STREAM-INF (master grojaraštis)
    double target_s = -1;  // #EXT-X-TARGETDURATION; < 0 — nėra
    std::vector<std::string> variants;  // master: variantų ir EXT-X-MEDIA URI (grojaraščiai)
    std::vector<std::string> refs;      // media: segmentų ir URI="…" (raktai, EXT-X-MAP) nuorodos
};
PlaylistInfo parse_playlist(const std::string& text);

// HLS ribos (A1).
struct HlsLimits {
    double max_target_s = 30;          // --hls-max-target-s
    std::size_t max_playlists = 16;    // master + variantai, kuriuos tikrinam
    std::size_t max_bytes = 1u << 20;  // grojaraščio dydis
};

// Garso žiedo dydis baitais (talpa apvalinama aukštyn iki 2 laipsnio, float32, kanalai) — tikrinama prieš --max-audio-ring-mb.
std::uint64_t audio_ring_bytes(int channels, std::uint64_t samples);

}  // namespace mxlv::ingest
