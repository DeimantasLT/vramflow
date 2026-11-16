// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "ingest/url_policy.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

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

// av_isspace (libavutil/avstring.h): ' ', \t, \n, \v, \f, \r
bool av_space(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

// libav HLS eilutės buferis: char line[MAX_URL_SIZE] (hls.c n6.1:737, MAX_URL_SIZE 4096) → ff_get_line ≤ 4095 baitų
constexpr std::size_t kMaxLine = 4095;

// B: kita eilutė TIKSLIAI kaip libav ff_get_chomp_line (aviobuf.c n6.1:816-839): pabaiga '\n', '\r' arba NUL, CR+LF — vienas
// skyriklis, ilgesnė nei 4095 baitų nukerpama (likusi dalis praleidžiama iki skyriklio), gale nukerpami av_isspace; pradžioje — ne.
bool next_line(const std::string& t, std::size_t& p, std::string& line) {
    if (p >= t.size()) return false;
    std::size_t e = p;
    while (e < t.size() && t[e] != '\n' && t[e] != '\r' && t[e] != '\0') ++e;
    line.assign(t, p, std::min(e - p, kMaxLine));
    if (e < t.size()) e += t[e] == '\r' && e + 1 < t.size() && t[e + 1] == '\n' ? 2 : 1;
    p = e;
    while (!line.empty() && av_space(line.back())) line.pop_back();
    return true;
}

// C: URI atributo reikšmė žymės eilutėje (po pirmo ':') TIKSLIAI kaip libav ff_parse_key_value (utils.c n6.1:504-553) su
// hls.c handle_*_args (raktas lyginamas strncmp(key, "URI=", key_len) — tik lygiai „URI=“): praleidžiami tarpai ir kableliai, raktas —
// iki pirmo '=', reikšmė kabutėse (\x — x pažodžiui, be uždarančios — iki galo) arba be jų iki tarpo / kablelio, ≤ 4095 baitų
// (char uri[MAX_URL_SIZE]); tas pats raktas vėl — paskutinė reikšmė. "" — nėra (libav tuščią URI irgi laiko „nėra“).
// M2: tas pats bet kuriam raktui want ("METHOD=" …), reikšmė nukerpama iki cap baitų (libav dest_len − 1)
std::string tag_attr(const std::string& line, const char* want, std::size_t cap) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) return {};
    const std::size_t n = line.size();
    std::size_t p = colon + 1;
    std::string out;
    for (;;) {
        while (p < n && (av_space(line[p]) || line[p] == ',')) ++p;
        if (p >= n) break;
        const std::size_t k0 = p;
        const auto eq = line.find('=', k0);
        if (eq == std::string::npos) break;
        p = eq + 1;
        const bool hit = line.compare(k0, p - k0, want) == 0;
        std::string v;
        if (p < n && line[p] == '"') {
            ++p;
            while (p < n && line[p] != '"') {
                if (line[p] == '\\') {
                    if (p + 1 >= n) break;
                    v += line[p + 1];
                    p += 2;
                } else {
                    v += line[p++];
                }
            }
            if (p < n && line[p] == '"') ++p;
        } else {
            while (p < n && !av_space(line[p]) && line[p] != ',') v += line[p++];
        }
        if (hit) out = v.substr(0, cap);
    }
    return out;
}

std::string uri_attr(const std::string& line) { return tag_attr(line, "URI=", kMaxLine); }

}  // namespace

std::string url_scheme(const std::string& url) {
    const std::size_t n = std::strspn(url.c_str(), kSchemeChars);
    if (n == 0 || n >= url.size() || (url[n] != ':' && url[n] != ',')) return {};
    return lower(url.substr(0, n));
}

bool source_url_allowed(const std::string& url, std::string& why) {
    const std::string s = url_scheme(url);
    if (s.empty() && !url.empty()) return true;  // vietinis failas (kaip iki šiol)
    if (http_scheme_without_slashes(url)) {
        why = s + " URL be „//“ po schemos neleidžiamas (libav av_url_split host'ą imtų iš kelio): " + redact_url(url);
        return false;
    }
    if (s == "udp" || s == "rtp" || s == "srt" || is_http(s)) return true;
    why = (s.empty() ? std::string("tuščias") : s) + " protokolas nepalaikomas (leidžiami: udp, rtp, srt, http, https, vietinis failas): " +
          redact_url(url);
    return false;
}

std::string protocol_whitelist_for(const std::string& scheme, bool hls) {
    if (scheme == "udp") return "udp";
    if (scheme == "rtp") return "rtp,udp";
    if (scheme == "srt") return "srt";
    // HLS ir HTTP TS: http(s) virš tcp / tls; crypto — tik AES-128 HLS segmentams (crypto+http…, viduje vėl tik šis sąrašas)
    if (is_http(scheme)) return hls ? "http,https,tcp,tls,crypto" : "http,https,tcp,tls";
    return "file";
}

std::string segment_format_whitelist() { return "mpegts,mov,mp4,m4a,3gp,3g2,mj2,aac"; }

std::string format_whitelist_for(const std::string& scheme, bool hls) {
    if (scheme == "rtp") return "rtp,sdp,mpegts";
    // HLS: grojaraštis ir segmentai (TS, fMP4, ADTS garsas); tiesioginis MPEG-TS per HTTP — tik mpegts (H1)
    if (is_http(scheme)) return hls ? "hls," + segment_format_whitelist() : std::string("mpegts");
    return "mpegts";  // udp, srt, vietinis failas
}

namespace {

// URL dalių ribos kaip libav ff_url_decompose (url.c n6.1:91-151): schema iki pirmo iš ":/?#", jei tai ':'; autoritetas — "//…" iki
// "/?#"; kelias iki "?#"; užklausa iki '#'; likusi — fragmentas. Indeksai — kiekvienos dalies PABAIGA.
struct UrlParts {
    std::size_t scheme, auth, path, query;
};

UrlParts decompose(const std::string& u) {
    UrlParts r{};
    std::size_t cur = 0;
    if (const auto p = u.find_first_of(":/?#"); p != std::string::npos && u[p] == ':') cur = p + 1;
    r.scheme = cur;
    if (u.compare(cur, 2, "//") == 0) cur = std::min(u.find_first_of("/?#", cur + 2), u.size());
    r.auth = cur;
    cur = std::min(u.find_first_of("?#", cur), u.size());
    r.path = cur;
    if (cur < u.size() && u[cur] == '?') cur = std::min(u.find('#', cur), u.size());
    r.query = cur;
    return r;
}

// Kelio segmentai [in, end) pridedami prie out, kurio šaknis (po pradinio '/') — root: "." praleidžiamas, ".." nuima paskutinį
// segmentą (ne žemiau šaknies) — kaip libav append_path (url.c:166-193)
void append_path(std::string& out, std::size_t root, const std::string& s, std::size_t in, std::size_t end) {
    if (in < end && s[in] == '/') ++in;
    while (in < end) {
        const std::size_t d = std::min(s.find('/', in), end);
        const std::size_t next = d + (d < end && s[d] == '/' ? 1 : 0);
        if (d - in == 1 && s[in] == '.') {
            // praleidžiama
        } else if (d - in == 2 && s[in] == '.' && s[in + 1] == '.') {
            if (out.size() - root > 1)
                while (out.size() > root) {
                    out.pop_back();
                    if (out.back() == '/') break;
                }
        } else {
            out.append(s, in, next - in);
        }
        in = next;
    }
}

// E: santykinės nuorodos išsprendimas TIKSLIAI kaip libav ff_make_absolute_url2 (url.c n6.1:195-319; HAVE_DOS_PATHS = 0 Linux'e):
// rezultatas turi sutapti iki simbolio su URL, kurį libav HLS perduoda io_open (žinomų grojaraščių sąrašas lyginamas eilutėmis)
std::string absolute_like_libav(const std::string& base, const std::string& rel) {
    const UrlParts b = decompose(base), c = decompose(rel);
    std::size_t keep = 0;
    bool simplify = false;
    // KEEP: komponentai, kurių rel neturi nuo pradžios, imami iš bazės (jei bazė juos turi)
    const auto keep_if = [&](std::size_t c_end, std::size_t b_end) {
        if (c_end != 0 || b_end <= keep) return false;
        keep = b_end;
        return true;
    };
    keep_if(c.scheme, b.scheme);
    if (keep_if(c.auth, b.auth)) simplify = true;
    keep_if(c.path, b.path);
    keep_if(c.query, b.query);
    keep_if(rel.size(), base.size());
    std::string out = base.substr(0, keep) + rel.substr(0, c.auth);
    const bool c_path = c.path > c.auth;
    bool use_base = b.path > b.auth && keep <= b.auth;
    if (c.auth > 0) use_base = false;
    if (c_path && rel[c.auth] == '/') use_base = false;
    std::size_t bpe = b.path;
    if (use_base && c_path)
        while (bpe > b.auth && base[bpe - 1] != '/') --bpe;
    if (keep > b.auth) simplify = false;
    if (c.scheme > 0) simplify = false;
    if (rel.compare(c.scheme, 2, "//") == 0) simplify = true;
    if (!use_base && !c_path) simplify = false;
    if (simplify) {
        out += '/';
        const std::size_t root = out.size();
        if (use_base) append_path(out, root, base, b.auth, bpe);
        if (c_path) append_path(out, root, rel, c.auth, c.path);
    } else {
        if (use_base) out.append(base, b.auth, bpe - b.auth);
        out.append(rel, c.auth, c.path - c.auth);
    }
    out.append(rel, c.path, std::string::npos);
    return out;
}

}  // namespace

bool resolve_playlist_url(const std::string& base, const std::string& ref_in, std::string& out, std::string& why) {
    const std::string ref = trim(ref_in);
    if (ref.empty()) {
        why = "tuščias grojaraščio URL";
        return false;
    }
    if (url_scheme(ref).empty() && (url_scheme(base).empty() || base.find("://") == std::string::npos)) {
        why = "grojaraščio bazė be schemos: " + redact_url(base);
        return false;
    }
    const std::string r = absolute_like_libav(base, ref);
    const std::string rs = url_scheme(r);
    if (!is_http(rs)) {
        why = "grojaraščio URL schema „" + (rs.empty() ? std::string("vietinis kelias") : rs) + "“ neleidžiama (tik http / https): " + redact_url(r);
        return false;
    }
    if (http_scheme_without_slashes(r)) {
        why = "grojaraščio URL „" + rs + ":“ be „//“ neleidžiamas (libav av_url_split host'ą imtų iš kelio): " + redact_url(r);
        return false;
    }
    out = r;
    return true;
}

std::string redact_url(const std::string& url) {
    const std::string s = url_scheme(url);
    // H1: „://“ — TIK iš karto po schemos (ne bet kur kelyje: „http:u:pw@h/x://a/“ paliktų userinfo); kitaip (data:, concat:,
    // „http:h/…“) viskas po schemos gali būti paslaptis
    if (s.empty()) return url;
    if (url.compare(s.size(), 3, "://") != 0) return s + ":***";
    const std::size_t sep = s.size();
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
        // H1: „://“ gali būti URL kelyje („http:u:pw@h/x://a/“) — URL pradžia tada anksčiau: pirma „schema:“ to paties žodžio (iki tarpo
        // ar kabutės) pradžia; visas URL nuo jos išvalomas redact_url (be „://“ iš karto — „schema:***“)
        std::size_t w = a;
        while (w > p && !std::isspace(static_cast<unsigned char>(text[w - 1])) && std::strchr("'\"<>`", text[w - 1]) == nullptr) --w;
        for (std::size_t i = w; i < a; ++i)
            if ((i == w || std::strchr(kSchemeChars, text[i - 1]) == nullptr) && !url_scheme(text.substr(i, sep + 1 - i)).empty()) {
                a = i;
                break;
            }
        std::size_t b = sep + 3;
        if (a > 0 && (text[a - 1] == '\'' || text[a - 1] == '"')) {
            // L1: kabutėse (libav „Opening '<URL>' for reading“) URL gali turėti tarpų — iki uždarančios kabutės ar eilutės galo
            const char qc = text[a - 1];
            while (b < text.size() && text[b] != qc && text[b] != '\n') ++b;
        } else {
            while (b < text.size() && !std::isspace(static_cast<unsigned char>(text[b])) && std::strchr("'\"<>`", text[b]) == nullptr) ++b;
        }
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
    std::string line, key;
    int seg_key = -1;  // M2: galiojančio rakto indeksas pi.keys (−1 — nėra)
    while (next_line(text, p, line)) {
        if (first) {
            // #EXTM3U atpažinimas platesnis nei libav (pirma eilutė == "#EXTM3U", hls.c n6.1:786-787): BOM, tušti priekyje, tarpai —
            // vis tiek M3U8; viskas, ką libav laiko M3U8, laikoma M3U8 ir čia (peržiūra, patikra ir sargas — tas pats skaidymas)
            std::string h = trim(line);
            if (starts(h, "\xEF\xBB\xBF")) h = h.substr(3);  // UTF-8 BOM
            if (h.empty()) continue;
            first = false;
            pi.m3u = starts(h, "#EXTM3U");
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
                // H2: bet kuri #EXT-X-MEDIA (ir be URI) — master lygmuo: libav kiekvieną kaupia c->renditions iki uždarymo
                // (hls.c n6.1:824-828, 523-527, 303-311), todėl perkraunamame ar įdėtame grojaraštyje jų būti negali
                if (starts(line, "#EXT-X-MEDIA:")) ++pi.media;
                pi.master = true;
                if (const std::string u = uri_attr(line); !u.empty()) pi.variants.push_back(u);
            } else {
                const std::string u = uri_attr(line);
                if (!u.empty()) pi.refs.push_back(u);  // EXT-X-KEY, EXT-X-MAP, EXT-X-SESSION-KEY, EXT-X-PART …
                // F: libav rakto URI laiko iki kito EXT-X-KEY (hls.c n6.1:809-823); kiekviena EXT-X-MAP eilutė — nauja init sekcija
                if (starts(line, "#EXT-X-KEY:")) {
                    key = u;
                    // M2: libav rakto URL kopijuoja kiekvienam segmentui tik kai METHOD AES-128 / SAMPLE-AES (char method[11], hls.c
                    // n6.1:393-395, 808-823), kitaip — rakto nėra
                    const std::string m = tag_attr(line, "METHOD=", 10);
                    seg_key = -1;
                    if (m == "AES-128" || m == "SAMPLE-AES") {
                        pi.keys.push_back(u);
                        seg_key = static_cast<int>(pi.keys.size()) - 1;
                    }
                } else if (starts(line, "#EXT-X-MAP:")) pi.maps.emplace_back(u, key);
            }
            continue;
        }
        if (want_variant) {
            pi.variants.push_back(line);
            want_variant = false;
        } else {
            pi.refs.push_back(line);
            pi.segments.emplace_back(line, seg_key);
        }
    }
    if (!pi.variants.empty() && !pi.master) pi.master = true;  // tik EXT-X-MEDIA — vis tiek master lygmuo
    return pi;
}

bool check_playlist(const std::string& base, const PlaylistInfo& pi, bool allow_master, const HlsLimits& lim, std::string& why) {
    if (!pi.m3u) {
        why = "ne M3U8 grojaraštis: " + redact_url(base);
        return false;
    }
    if (pi.master && !allow_master) {
        why = pi.media > 0 ? "#EXT-X-MEDIA perkraunamame ar įdėtame grojaraštyje (master lygmuo; libav kiekvieną eilutę kaupia iki sesijos "
                             "pabaigos): " + redact_url(base)
                           : "įdėtas master grojaraštis (gylis > 1: master → media): " + redact_url(base);
        return false;
    }
    if (pi.media > lim.max_media) {
        why = "grojaraštyje " + std::to_string(pi.media) + " #EXT-X-MEDIA eilučių > riba " + std::to_string(lim.max_media) +
              " (libav jų neatlaisvina iki sesijos pabaigos)";
        return false;
    }
    if (pi.variants.size() + 1 > lim.max_playlists) {
        why = "master grojaraštyje " + std::to_string(pi.variants.size()) + " variantų > " + std::to_string(lim.max_playlists - 1);
        return false;
    }
    if (pi.maps.size() > lim.max_maps) {
        why = "grojaraštyje " + std::to_string(pi.maps.size()) + " #EXT-X-MAP eilučių > riba " + std::to_string(lim.max_maps) +
              " (libav init sekcijų neatlaisvina iki sesijos pabaigos)";
        return false;
    }
    if (pi.refs.size() > lim.max_segments) {
        why = "grojaraštyje " + std::to_string(pi.refs.size()) + " nuorodų (segmentų, raktų) > riba " + std::to_string(lim.max_segments);
        return false;
    }
    if (const std::size_t sb = segment_bytes(base, pi); sb > lim.max_playlist_bytes) {
        why = "libav segmentų sąmata grojaraščiui " + std::to_string(sb >> 10) + " KiB > riba " + std::to_string(lim.max_playlist_bytes >> 10) +
              " KiB (kiekvienam segmentui — URL ir rakto URL kopija)";
        return false;
    }
    if (pi.target_s > lim.max_target_s) {
        char b[160];
        std::snprintf(b, sizeof b, "#EXT-X-TARGETDURATION %.0f s > riba %.0f s (--hls-max-target-s)", pi.target_s, lim.max_target_s);
        why = b;
        return false;
    }
    std::string r;
    for (const auto* v : {&pi.variants, &pi.refs})
        for (const std::string& x : *v) {
            if (!resolve_playlist_url(base, x, r, why)) return false;
            if (!lim.only_authority.empty() && canonical_authority(r) != lim.only_authority) {
                why = "grojaraščio nuoroda į kitą serverį (--hls-same-host: tik " + lim.only_authority + "): " + redact_url(r);
                return false;
            }
        }
    return true;
}

std::size_t init_section_bytes(const std::string& base, const PlaylistInfo& pi) {
    // libav vienai EXT-X-MAP: av_mallocz(struct segment) + dynarray + av_strdup(absoliutus URL) + av_strdup(rakto URL), jei raktas
    // (hls.c n6.1:417-457, 880-891); URL buferis tmp_str[MAX_URL_SIZE] — ≤ 4096 baitų
    const auto len = [&](const std::string& ref) -> std::size_t {
        if (ref.empty()) return 0;
        std::string r, why;
        return std::min<std::size_t>(4096, (resolve_playlist_url(base, ref, r, why) ? r.size() : base.size() + ref.size()) + 1);
    };
    std::size_t n = 0;
    for (const auto& [map, key] : pi.maps) n += 256 + len(map) + len(key);
    return n;
}

bool http_scheme_without_slashes(const std::string& url) {
    const std::string s = url_scheme(url);
    const std::string inner = starts(s, "crypto+") ? s.substr(7) : s;
    return is_http(inner) && url.compare(s.size(), 3, "://") != 0;
}

namespace {

// M2: libav ff_make_absolute_url(tmp_str[MAX_URL_SIZE], base, ref) ilgis su NUL (≤ 4096); ref "" — bazė (url.c n6.1:195-319)
std::size_t abs_url_bytes(const std::string& base, const std::string& ref) {
    std::string r, why;
    const std::size_t n = ref.empty() ? base.size() : resolve_playlist_url(base, ref, r, why) ? r.size() : base.size() + ref.size();
    return std::min<std::size_t>(4096, n + 1);
}

}  // namespace

std::size_t segment_bytes(const std::string& base, const PlaylistInfo& pi) {
    // libav kiekvienam segmentui (hls.c n6.1:931-978): av_malloc(struct segment), av_strdup(rakto URL), jei raktas, av_strdup(URL)
    std::vector<std::size_t> key_bytes(pi.keys.size());
    for (std::size_t k = 0; k < pi.keys.size(); ++k) key_bytes[k] = abs_url_bytes(base, pi.keys[k]);
    std::size_t n = 0;
    for (const auto& [uri, key] : pi.segments)
        n += kSegmentOverhead + abs_url_bytes(base, uri) + (key >= 0 ? key_bytes[static_cast<std::size_t>(key)] : 0);
    return n;
}

std::string url_authority(const std::string& url) {
    // H1: av_url_split (utils.c n6.1:377-389): schema iki pirmo ':' (url_scheme — tas pats ':'), po jo ≤ 2 '/', autoritetas iki "/?#"
    // (ne „po pirmo ://“ — „http:evil/x://allowed/“ libav jungiasi į evil)
    const std::string s = url_scheme(url);
    if (s.empty()) return {};
    std::size_t a0 = s.size() + 1;
    for (int k = 0; k < 2 && a0 < url.size() && url[a0] == '/'; ++k) ++a0;
    const std::size_t a1 = std::min(url.find_first_of("/?#", a0), url.size());
    std::string auth = url.substr(a0, a1 - a0);
    if (const auto at = auth.rfind('@'); at != std::string::npos) auth = auth.substr(at + 1);  // utils.c:395-401
    return lower(auth);
}

std::string canonical_authority(const std::string& url) {
    std::string s = url_scheme(url);
    if (starts(s, "crypto+")) s = s.substr(7);
    if (!is_http(s) || http_scheme_without_slashes(url)) return {};  // H1: be „://“ — niekada nesutampa
    // url_authority: userinfo iki PASKUTINIO '@' (kaip av_url_split, utils.c n6.1:397-401), mažosios raidės
    const std::string auth = url_authority(url);
    std::string host, port;
    bool has_port = false;
    if (!auth.empty() && auth[0] == '[' && auth.find(']') != std::string::npos) {  // [IPv6]:prievadas (utils.c:403-407)
        const auto b = auth.find(']');
        host = auth.substr(0, b + 1);
        has_port = b + 1 < auth.size() && auth[b + 1] == ':';
        if (has_port) port = auth.substr(b + 2);
    } else {  // prievadas po PIRMO ':' (utils.c:408-411)
        const auto c = auth.find(':');
        host = auth.substr(0, c);
        has_port = c != std::string::npos;
        if (has_port) port = auth.substr(c + 1);
    }
    // H1: http host'o buferis hostname[1024] (http.c n6.1:205) — ilgesnį libav nukirptų iki 1023 baitų (kitas host'as) → nesutampa
    if (!host.empty() && host.size() - (host[0] == '[' ? 2 : 0) > 1023) return {};
    if (!host.empty() && host.back() == '.') host.pop_back();  // FQDN šaknies taškas: "h." == "h"
    if (!has_port) return host + (s == "https" ? ":443" : ":80");
    // tik skaitmenys (≤ 5) — skaitinė reikšmė (":0080" == ":80"); kita paliekama kaip yra — nesutaps su leistinu, bus atmesta
    if (!port.empty() && port.size() <= 5 && port.find_first_not_of("0123456789") == std::string::npos)
        return host + ":" + std::to_string(std::stoi(port));
    return host + ":" + port;
}

bool secret_from(const std::string& spec, std::string& out, std::string& why) {
    out.clear();
    if (starts(spec, "env:")) {
        const std::string name = spec.substr(4);
        const char* v = name.empty() ? nullptr : std::getenv(name.c_str());
        if (v == nullptr || *v == 0) {
            why = "aplinkos kintamasis „" + name + "“ nenustatytas arba tuščias";
            return false;
        }
        out = v;
        return true;
    }
    if (starts(spec, "file:")) {
        const std::string path = spec.substr(5);
        std::ifstream f(path, std::ios::binary);
        if (!f || !std::getline(f, out)) {
            why = "slaptažodžio failo „" + path + "“ perskaityti nepavyko";
            return false;
        }
        while (!out.empty() && (out.back() == '\r' || out.back() == '\n')) out.pop_back();
        if (out.empty()) {
            why = "slaptažodžio failas „" + path + "“ tuščias";
            return false;
        }
        return true;
    }
    why = "laukta env:VARDAS arba file:/kelias (slaptažodis komandinėje eilutėje matomas /proc/*/cmdline)";
    return false;
}

bool srt_passphrase_ok(const std::string& pass, std::string& why) {
    if (pass.size() < 10 || pass.size() > 79) {
        why = "SRT slaptažodis turi būti 10–79 simbolių (dabar " + std::to_string(pass.size()) + ")";
        return false;
    }
    // G: libsrt passphrase ima av_find_info_tag (libsrt.c n6.1:548-550), kuris '+' verčia tarpu (parseutils.c n6.1:774-775); '%'
    // nedekoduojamas, bet draudžiamas (vienareikšmiškumui)
    for (const char c : pass)
        if (c == 0 || std::strchr("&#?%+ \t\r\n", c) != nullptr || static_cast<unsigned char>(c) < 0x20) {
            why = "SRT slaptažodyje neleistinas simbolis (&, #, ?, %, +, tarpai — sugadintų URL; libav '+' paverčia tarpu)";
            return false;
        }
    return true;
}

std::string with_srt_passphrase(const std::string& url, const std::string& pass) {
    if (url_scheme(url) != "srt") return url;
    const auto q = url.find('?');
    if (q != std::string::npos) {
        const std::string query = "&" + url.substr(q + 1);
        if (query.find("&passphrase=") != std::string::npos) return url;
        return url + "&passphrase=" + pass;
    }
    return url + "?passphrase=" + pass;
}

std::uint64_t audio_ring_bytes(int channels, std::uint64_t samples) {
    std::uint64_t c = 1;
    while (c < samples) c <<= 1;
    return c * static_cast<std::uint64_t>(channels < 0 ? 0 : channels) * sizeof(float);
}

}  // namespace mxlv::ingest
