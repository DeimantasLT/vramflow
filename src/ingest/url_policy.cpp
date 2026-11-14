// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/url_policy.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>

namespace mxlv::ingest {

namespace {

constexpr const char* kSchemeChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";

bool is_http(const std::string& s) { return s == "http" || s == "https"; }

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Eilutės pradžios / pabaigos tarpai ir \r
std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
    return s.substr(a, b - a);
}

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// URI="…" atributo reikšmė (EXT-X-KEY, EXT-X-MAP, EXT-X-MEDIA, EXT-X-I-FRAME-STREAM-INF …); "" — nėra
std::string uri_attr(const std::string& line) {
    const auto p = line.find("URI=\"");
    if (p == std::string::npos) return {};
    const auto e = line.find('"', p + 5);
    return e == std::string::npos ? std::string{} : line.substr(p + 5, e - p - 5);
}

}  // namespace

std::string url_scheme(const std::string& url) {
    const std::size_t n = std::strspn(url.c_str(), kSchemeChars);
    if (n == 0 || n >= url.size() || (url[n] != ':' && url[n] != ',')) return {};
    return lower(url.substr(0, n));
}

bool source_url_allowed(const std::string& url, std::string& why) {
    const std::string s = url_scheme(url);
    if (s.empty() && !url.empty()) return true;  // vietinis failas (kaip iki šiol)
    if (s == "udp" || s == "rtp" || s == "srt" || is_http(s)) return true;
    why = (s.empty() ? std::string("tuščias") : s) + " protokolas nepalaikomas (leidžiami: udp, rtp, srt, http, https, vietinis failas): " +
          redact_url(url);
    return false;
}

std::string protocol_whitelist_for(const std::string& scheme) {
    if (scheme == "udp") return "udp";
    if (scheme == "rtp") return "rtp,udp";
    if (scheme == "srt") return "srt";
    // HLS ir HTTP TS: http(s) virš tcp / tls; crypto — tik AES-128 HLS segmentams (crypto+http…, viduje vėl tik šis sąrašas)
    if (is_http(scheme)) return "http,https,tcp,tls,crypto";
    return "file";
}

std::string format_whitelist_for(const std::string& scheme) {
    if (scheme == "rtp") return "rtp,sdp,mpegts";
    // HLS: grojaraštis ir segmentai (TS, fMP4, ADTS garsas); taip pat tiesioginis MPEG-TS per HTTP
    if (is_http(scheme)) return "hls,mpegts,mov,mp4,m4a,3gp,3g2,mj2,aac";
    return "mpegts";  // udp, srt, vietinis failas
}

bool resolve_playlist_url(const std::string& base, const std::string& ref_in, std::string& out, std::string& why) {
    const std::string ref = trim(ref_in);
    if (ref.empty()) {
        why = "tuščias grojaraščio URL";
        return false;
    }
    std::string r;
    if (!url_scheme(ref).empty()) {
        r = ref;
    } else {
        const std::string bs = url_scheme(base);
        const auto sep = base.find("://");
        if (bs.empty() || sep == std::string::npos) {
            why = "grojaraščio bazė be schemos: " + redact_url(base);
            return false;
        }
        const std::size_t auth0 = sep + 3;
        const std::size_t path0 = std::min(base.find_first_of("/?#", auth0), base.size());
        if (starts(ref, "//")) {
            r = bs + ":" + ref;
        } else if (ref[0] == '/') {
            r = base.substr(0, path0) + ref;
        } else {
            // santykinė: bazės kelias iki paskutinio '/' (be užklausos ir fragmento)
            const std::size_t q = std::min(base.find_first_of("?#", path0), base.size());
            const std::string path = base.substr(path0, q - path0);
            const auto slash = path.rfind('/');
            r = base.substr(0, path0) + (slash == std::string::npos ? std::string("/") : path.substr(0, slash + 1)) + ref;
        }
    }
    const std::string rs = url_scheme(r);
    if (!is_http(rs)) {
        why = "grojaraščio URL schema „" + (rs.empty() ? std::string("vietinis kelias") : rs) + "“ neleidžiama (tik http / https): " + redact_url(r);
        return false;
    }
    out = r;
    return true;
}

std::string redact_url(const std::string& url) {
    const auto sep = url.find("://");
    if (url_scheme(url).empty() || sep == std::string::npos) {
        // be „//“ (pvz. data:, concat:) — viskas po schemos gali būti paslaptis
        const std::string s = url_scheme(url);
        return s.empty() ? url : s + ":***";
    }
    std::string out = url.substr(0, sep + 3);
    const std::size_t auth0 = sep + 3;
    const std::size_t auth1 = std::min(url.find_first_of("/?#", auth0), url.size());
    const std::string auth = url.substr(auth0, auth1 - auth0);
    const auto at = auth.rfind('@');
    out += at == std::string::npos ? auth : "***@" + auth.substr(at + 1);
    const std::size_t q = std::min(url.find_first_of("?#", auth1), url.size());
    out += url.substr(auth1, q - auth1);
    if (q < url.size() && url[q] == '?') {
        const std::size_t f = std::min(url.find('#', q), url.size());
        const std::string query = url.substr(q + 1, f - q - 1);
        out += "?";
        std::size_t p = 0;
        bool first = true;
        while (p <= query.size()) {
            const std::size_t e = std::min(query.find('&', p), query.size());
            const std::string kv = query.substr(p, e - p);
            if (!first) out += "&";
            first = false;
            const auto eq = kv.find('=');
            out += eq == std::string::npos ? (kv.empty() ? std::string{} : std::string("***")) : kv.substr(0, eq + 1) + "***";
            p = e + 1;
        }
        if (f < url.size()) out += "#***";
    } else if (q < url.size()) {
        out += "#***";
    }
    return out;
}

std::string redact_urls_in(const std::string& text) {
    std::string out;
    std::size_t p = 0;
    for (;;) {
        const auto sep = text.find("://", p);
        if (sep == std::string::npos) break;
        std::size_t a = sep;
        while (a > p && std::strchr(kSchemeChars, text[a - 1]) != nullptr) --a;
        std::size_t b = sep + 3;
        while (b < text.size() && !std::isspace(static_cast<unsigned char>(text[b])) && std::strchr("'\"<>`", text[b]) == nullptr) ++b;
        out.append(text, p, a - p);
        out += a == sep ? text.substr(a, b - a) : redact_url(text.substr(a, b - a));
        p = b;
    }
    out.append(text, p, std::string::npos);
    return out;
}

PlaylistInfo parse_playlist(const std::string& text) {
    PlaylistInfo pi;
    std::size_t p = 0;
    bool first = true, want_variant = false;
    while (p < text.size()) {
        const std::size_t e = std::min(text.find('\n', p), text.size());
        std::string line = trim(text.substr(p, e - p));
        p = e + 1;
        if (first) {
            if (starts(line, "\xEF\xBB\xBF")) line = line.substr(3);  // UTF-8 BOM
            if (line.empty()) continue;
            first = false;
            pi.m3u = starts(line, "#EXTM3U");
            if (!pi.m3u) return pi;
            continue;
        }
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (starts(line, "#EXT-X-TARGETDURATION:")) {
                pi.target_s = std::strtod(line.c_str() + 22, nullptr);
            } else if (starts(line, "#EXT-X-STREAM-INF")) {
                pi.master = true;
                want_variant = true;
            } else if (starts(line, "#EXT-X-MEDIA:") || starts(line, "#EXT-X-I-FRAME-STREAM-INF")) {
                pi.master = pi.master || starts(line, "#EXT-X-I-FRAME-STREAM-INF");
                if (const std::string u = uri_attr(line); !u.empty()) pi.variants.push_back(u);
            } else if (const std::string u = uri_attr(line); !u.empty()) {
                pi.refs.push_back(u);  // EXT-X-KEY, EXT-X-MAP, EXT-X-SESSION-KEY, EXT-X-PART …
            }
            continue;
        }
        if (want_variant) {
            pi.variants.push_back(line);
            want_variant = false;
        } else {
            pi.refs.push_back(line);
        }
    }
    if (!pi.variants.empty() && !pi.master) pi.master = true;  // tik EXT-X-MEDIA — vis tiek master lygmuo
    return pi;
}

std::uint64_t audio_ring_bytes(int channels, std::uint64_t samples) {
    std::uint64_t c = 1;
    while (c < samples) c <<= 1;
    return c * static_cast<std::uint64_t>(channels < 0 ? 0 : channels) * sizeof(float);
}

}  // namespace mxlv::ingest
