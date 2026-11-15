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
#include <utility>
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
// E: HLS segmentų (įdėto) demuxerio format_whitelist — be hls (segmentas negali būti grojaraštis).
std::string segment_format_whitelist();

// Grojaraščio nuoroda ref, išspręsta pagal grojaraščio URL base (RFC 3986 supaprastintai: absoliuti, "//host…", "/kelias",
// santykinė). true — rezultatas http / https (out); kitaip why — aiški priežastis (schema išvalyta nuo paslapčių).
bool resolve_playlist_url(const std::string& base, const std::string& ref, std::string& out, std::string& why);

// URL be paslapčių: userinfo → „***@“, kiekvienos užklausos poros reikšmė → „***“ (raktas paliekamas), fragmentas → „#***“.
// Vietinis kelias — nepakeistas.
std::string redact_url(const std::string& url);
// Tekste kiekvienas „schema://…“ (iki tarpo ar kabučių) išvalomas redact_url (libav žurnalo eilutėms, klaidų tekstams).
std::string redact_urls_in(const std::string& text);

// L1: slapta reikšmė ne iš argv (/proc/*/cmdline mato visi naudotojai): "env:VARDAS" — aplinkos kintamasis, "file:/kelias" — pirma failo
// eilutė (be CR / LF). Kitaip (pvz. pats slaptažodis) — false, why be reikšmės.
bool secret_from(const std::string& spec, std::string& out, std::string& why);
// SRT slaptažodis: 10–79 simbolių (libsrt), be URL užklausos skyriklių ir tarpų (&, #, ?, %, +, tarpai; libav + verčia tarpu) — kitaip sugadintų URL.
bool srt_passphrase_ok(const std::string& pass, std::string& why);
// srt:// URL be passphrase= — pridedama &passphrase=… (tik proceso atmintyje); jau su passphrase arba ne srt — nekeičiamas.
std::string with_srt_passphrase(const std::string& url, const std::string& pass);

// HLS grojaraščio (M3U8) santrauka.
struct PlaylistInfo {
    bool m3u = false;     // prasideda #EXTM3U
    bool master = false;  // turi #EXT-X-STREAM-INF (master grojaraštis)
    double target_s = -1;  // #EXT-X-TARGETDURATION; < 0 — nėra
    std::vector<std::string> variants;  // master: variantų ir EXT-X-MEDIA URI (grojaraščiai)
    std::vector<std::string> refs;      // media: segmentų ir URI="…" (raktai, EXT-X-MAP) nuorodos
    // F: kiekviena #EXT-X-MAP eilutė — (jos URI, tuo metu galiojęs EXT-X-KEY URI arba "")
    std::vector<std::pair<std::string, std::string>> maps;
};
PlaylistInfo parse_playlist(const std::string& text);

// HLS ribos (A1, H2 / H3).
struct HlsLimits {
    double max_target_s = 30;           // --hls-max-target-s
    std::size_t max_playlists = 16;     // master + variantai, kuriuos tikrinam
    std::size_t max_bytes = 1u << 20;   // grojaraščio dydis (--hls-max-playlist); daugiau — šaltinis atmetamas
    std::size_t max_segments = 10000;   // media grojaraščio nuorodų (segmentų, raktų, EXT-X-MAP) riba
    std::string only_authority;         // --hls-same-host: tik šis canonical_authority (viršutinio grojaraščio location); "" — bet kuris
    // F: libav init sekcijų (EXT-X-MAP) iki sesijos pabaigos neatlaisvina — ≤ max_maps eilučių grojaraštyje ir ≤ max_init_bytes jų
    // sąmatos (init_section_bytes) per sesiją; daugiau — sesija nutraukiama (prisijungus iš naujo libav atmintis atlaisvinta)
    std::size_t max_maps = 4;
    std::size_t max_init_bytes = std::size_t{16} << 20;
};

// F: kiek atminties libav skirs šio grojaraščio init sekcijoms (EXT-X-MAP): kiekvienai 256 + |absoliutus URL| + 1 (+ |rakto URL| + 1).
std::size_t init_section_bytes(const std::string& base, const PlaylistInfo& pi);

// URL host[:port] mažosiomis raidėmis, be userinfo ("" — ne schema://).
std::string url_authority(const std::string& url);
// D: --hls-same-host lyginimo forma "host:prievadas" (http / https / crypto+http(s); kita — ""): host mažosiomis be galinio taško,
// prievadas visada (nėra — 80 / 443, skaitmenys — skaitinė reikšmė, kita — kaip parašyta), [IPv6] su skliaustais.
std::string canonical_authority(const std::string& url);

// H2: VISO grojaraščio patikra (ir patikroje prieš libav, ir sarge kiekvienam libav atsisiųstam M3U8): master leidžiamas tik
// viršutiniam (allow_master), variantų ≤ max_playlists − 1, nuorodų ≤ max_segments, #EXT-X-TARGETDURATION ≤ max_target_s, kiekviena
// nuoroda (išspręsta pagal base) — tik http / https (ir, jei lim.only_authority, tik į tą serverį). false — why (be paslapčių).
bool check_playlist(const std::string& base, const PlaylistInfo& pi, bool allow_master, const HlsLimits& lim, std::string& why);

// Garso žiedo dydis baitais (talpa apvalinama aukštyn iki 2 laipsnio, float32, kanalai) — tikrinama prieš --max-audio-ring-mb.
std::uint64_t audio_ring_bytes(int channels, std::uint64_t samples);

}  // namespace mxlv::ingest
