// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// URL politika (round 9, A1): šaltinio schemos, libav baltieji sąrašai, grojaraščio URL (tik http / https), paslapčių valymas,
// grojaraščio analizė, garso žiedo dydis.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

#include "ingest/url_policy.hpp"

using namespace mxlv::ingest;

TEST_CASE("url_policy: schema kaip libavformat (':' arba ','), vietinis kelias — be schemos", "[ingest][url]") {
    CHECK(url_scheme("udp://127.0.0.1:20000") == "udp");
    CHECK(url_scheme("SRT://h:1?passphrase=x") == "srt");
    CHECK(url_scheme("crypto+http://h/a") == "crypto+http");
    CHECK(url_scheme("subfile,,0,10,:/etc/passwd") == "subfile");
    CHECK(url_scheme("concat:a.ts|b.ts") == "concat");
    CHECK(url_scheme("/tmp/a.ts").empty());
    CHECK(url_scheme("clip.ts").empty());
    CHECK(url_scheme("").empty());
}

TEST_CASE("url_policy: operatoriaus URL — tik dokumentuotos schemos", "[ingest][url]") {
    std::string why;
    for (const char* ok : {"udp://127.0.0.1:1", "rtp://127.0.0.1:2", "srt://127.0.0.1:3?mode=listener", "http://h/live.m3u8", "https://h/a.m3u8",
                           "/media/clip.ts", "clip.ts"}) {
        INFO(ok);
        CHECK(source_url_allowed(ok, why));
    }
    for (const char* bad : {"file:///etc/passwd", "tcp://127.0.0.1:9", "concat:a|b", "data:text/plain,x", "crypto+http://h/a", "xsrt://127.0.0.1:9000",
                            "subfile,,0,1,:/etc/passwd", "ftp://h/a.ts", "pipe:0"}) {
        INFO(bad);
        why.clear();
        CHECK_FALSE(source_url_allowed(bad, why));
        CHECK(why.find("protokolas nepalaikomas") != std::string::npos);
    }
    // aiškus pranešimas su schema ir (išvalytu) URL — kaip ingest_bad_url[protocol]
    CHECK_FALSE(source_url_allowed("xsrt://127.0.0.1:9000", why));
    CHECK(why.find("xsrt protokolas nepalaikomas") != std::string::npos);
    CHECK(why.find("xsrt://127.0.0.1:9000") != std::string::npos);
    CHECK_FALSE(source_url_allowed("tcp://u:secretpw@h:1?token=abc", why));
    CHECK(why.find("secretpw") == std::string::npos);
    CHECK(why.find("abc") == std::string::npos);
}

TEST_CASE("url_policy: libav baltieji sąrašai pagal schemą", "[ingest][url]") {
    CHECK(protocol_whitelist_for("udp") == "udp");
    CHECK(protocol_whitelist_for("rtp") == "rtp,udp");
    CHECK(protocol_whitelist_for("srt") == "srt");
    CHECK(protocol_whitelist_for("https") == "http,https,tcp,tls,crypto");
    CHECK(protocol_whitelist_for("") == "file");
    for (const char* s : {"udp", "rtp", "srt", "http", "https", ""}) {
        INFO(s);
        const std::string w = protocol_whitelist_for(s);
        CHECK(w.find("file") == (std::string(s).empty() ? 0 : std::string::npos));  // file — tik vietiniam failui
        CHECK(w.find("data") == std::string::npos);
        CHECK(w.find("concat") == std::string::npos);
        CHECK(format_whitelist_for(s).find("concat") == std::string::npos);
    }
    CHECK(format_whitelist_for("udp") == "mpegts");
    CHECK(format_whitelist_for("") == "mpegts");  // vietinis failas — ne HLS (kitaip segmentai būtų file:)
    CHECK(format_whitelist_for("http").find("hls") != std::string::npos);
}

TEST_CASE("url_policy: HTTP šaltinis, kurio patikra rado ne HLS — tik mpegts (H1, TOCTOU)", "[ingest][url]") {
    // patikra mato MPEG-TS, o libav atidarymas gauna M3U8: HLS demuxeris neturi būti pasiekiamas
    for (const char* s : {"http", "https"}) {
        INFO(s);
        CHECK(format_whitelist_for(s, false) == "mpegts");
        CHECK(protocol_whitelist_for(s, false) == "http,https,tcp,tls");  // be crypto (tik AES-128 HLS segmentams)
        CHECK(format_whitelist_for(s, true).find("hls") != std::string::npos);
        CHECK(protocol_whitelist_for(s, true) == "http,https,tcp,tls,crypto");
    }
    CHECK(format_whitelist_for("udp", false) == "mpegts");
    CHECK(protocol_whitelist_for("srt", false) == "srt");
}

TEST_CASE("url_policy: grojaraščio nuorodos išsprendžiamos ir ribojamos http / https", "[ingest][url]") {
    std::string out, why;
    const std::string base = "http://127.0.0.1:8080/live/master.m3u8?tok=1";
    REQUIRE(resolve_playlist_url(base, "v0/index.m3u8", out, why));
    CHECK(out == "http://127.0.0.1:8080/live/v0/index.m3u8");
    REQUIRE(resolve_playlist_url(base, "/abs/seg.ts", out, why));
    CHECK(out == "http://127.0.0.1:8080/abs/seg.ts");
    REQUIRE(resolve_playlist_url(base, "//cdn.example/a.ts", out, why));
    CHECK(out == "http://cdn.example/a.ts");
    REQUIRE(resolve_playlist_url("https://h", "seg.ts", out, why));
    CHECK(out == "https://h/seg.ts");
    REQUIRE(resolve_playlist_url(base, "https://other/x.m3u8", out, why));
    CHECK(out == "https://other/x.m3u8");
    for (const char* bad : {"file:///tmp/canary", "FILE:/etc/passwd", "tcp://127.0.0.1:9999", "data:,x", "crypto+file:///a", "subfile,,0,1,:/a",
                            "concat:a|b", "udp://1.2.3.4:5"}) {
        INFO(bad);
        why.clear();
        CHECK_FALSE(resolve_playlist_url(base, bad, out, why));
        CHECK(why.find("neleidžiama") != std::string::npos);
    }
    CHECK_FALSE(resolve_playlist_url(base, "   ", out, why));
}

TEST_CASE("url_policy: paslaptys išvalomos (userinfo, visos užklausos reikšmės)", "[ingest][url]") {
    CHECK(redact_url("srt://127.0.0.1:9000?mode=listener&passphrase=VerySecret123") == "srt://127.0.0.1:9000?mode=***&passphrase=***");
    CHECK(redact_url("http://user:pw@host:80/a/b.m3u8?token=xyz#frag") == "http://***@host:80/a/b.m3u8?token=***#***");
    CHECK(redact_url("udp://127.0.0.1:20000") == "udp://127.0.0.1:20000");
    CHECK(redact_url("https://h/p?justatoken") == "https://h/p?***");
    CHECK(redact_url("/local/file.ts") == "/local/file.ts");
    CHECK(redact_url("data:text/plain,secret") == "data:***");
    const std::string line = "[hls @ 0x1] Opening 'http://h/seg1.ts?sig=SECRET' for reading; srt://a:1?passphrase=PW2 end";
    const std::string r = redact_urls_in(line);
    CHECK(r.find("SECRET") == std::string::npos);
    CHECK(r.find("PW2") == std::string::npos);
    CHECK(r.find("'http://h/seg1.ts?sig=***'") != std::string::npos);
    CHECK(r.find("srt://a:1?passphrase=*** end") != std::string::npos);
    CHECK(redact_urls_in("be URL") == "be URL");
}

TEST_CASE("url_policy: grojaraščio analizė — master, media, TARGETDURATION, URI atributai", "[ingest][url]") {
    const PlaylistInfo m = parse_playlist(
        "\xEF\xBB\xBF#EXTM3U\r\n#EXT-X-STREAM-INF:BANDWIDTH=1\r\nv0.m3u8\r\n#EXT-X-MEDIA:TYPE=AUDIO,URI=\"a.m3u8\"\n#EXT-X-STREAM-INF:BANDWIDTH=2\nv1.m3u8\n");
    CHECK(m.m3u);
    CHECK(m.master);
    REQUIRE(m.variants.size() == 3);
    CHECK(m.variants[0] == "v0.m3u8");
    CHECK(m.variants[1] == "a.m3u8");
    CHECK(m.variants[2] == "v1.m3u8");
    const PlaylistInfo p = parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-KEY:METHOD=AES-128,URI=\"file:///k\"\n#EXTINF:4,\nseg0.ts\n");
    CHECK(p.m3u);
    CHECK_FALSE(p.master);
    CHECK(p.target_s == 4.0);
    REQUIRE(p.refs.size() == 2);
    CHECK(p.refs[0] == "file:///k");
    CHECK(p.refs[1] == "seg0.ts");
    CHECK_FALSE(parse_playlist("\x47\x40\x00\x10 TS bytes").m3u);
    CHECK(parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:100000\n").target_s == 100000.0);
}

TEST_CASE("url_policy: B — eilutės skaidomos kaip libav ff_get_chomp_line (CR, LF, CRLF, NUL)", "[ingest][url]") {
    // libav (aviobuf.c n6.1:816-839) eilutę baigia '\n', '\r' arba NUL; CR+LF — vienas skyriklis; gale nukerpami tarpai
    const PlaylistInfo cr = parse_playlist("#EXTM3U\r#EXT-X-STREAM-INF:BANDWIDTH=1\rv0.m3u8\r#EXT-X-STREAM-INF:BANDWIDTH=2\rv1.m3u8\r");
    CHECK(cr.m3u);
    CHECK(cr.master);
    REQUIRE(cr.variants.size() == 2);
    CHECK(cr.variants[0] == "v0.m3u8");
    CHECK(cr.variants[1] == "v1.m3u8");
    CHECK(cr.refs.empty());
    static const char kNul[] = "#EXTM3U\0#EXT-X-STREAM-INF:BANDWIDTH=1\0v0.m3u8\0";
    const PlaylistInfo nul = parse_playlist(std::string(kNul, sizeof kNul - 1));
    CHECK(nul.m3u);
    CHECK(nul.master);
    REQUIRE(nul.variants.size() == 1);
    CHECK(nul.variants[0] == "v0.m3u8");
    static const char kMixed[] = "#EXTM3U\r\n#EXT-X-TARGETDURATION:2\r#EXTINF:2,\n\rseg0.ts\0#EXT-X-KEY:METHOD=AES-128,URI=\"k.bin\"\n\r"
                                 "#EXTINF:2,\r\nseg1.ts \t\r#EXT-X-STREAM-INF:BANDWIDTH=1\r\nx.m3u8";
    const PlaylistInfo mx = parse_playlist(std::string(kMixed, sizeof kMixed - 1));
    CHECK(mx.m3u);
    CHECK(mx.master);
    CHECK(mx.target_s == 2.0);
    REQUIRE(mx.refs.size() == 3);
    CHECK(mx.refs[0] == "seg0.ts");
    CHECK(mx.refs[1] == "k.bin");
    CHECK(mx.refs[2] == "seg1.ts");  // gale tarpai nukirpti
    REQUIRE(mx.variants.size() == 1);
    CHECK(mx.variants[0] == "x.m3u8");
    // pradžios tarpai NEnukerpami (kaip libav): eilutė su tarpu priekyje — URI, ne žymė
    const PlaylistInfo lead = parse_playlist("#EXTM3U\n #EXT-X-STREAM-INF:BANDWIDTH=1\nv.m3u8\n");
    CHECK_FALSE(lead.master);
    REQUIRE(lead.refs.size() == 2);
    CHECK(lead.refs[0] == " #EXT-X-STREAM-INF:BANDWIDTH=1");
    // libav eilutės buferis MAX_URL_SIZE = 4096: ilgesnė eilutė nukerpama iki 4095 baitų, likusi dalis praleidžiama iki skyriklio
    const PlaylistInfo lng = parse_playlist("#EXTM3U\n#EXTINF:2,\n" + std::string(5000, 'a') + "\nb.ts\n");
    REQUIRE(lng.refs.size() == 2);
    CHECK(lng.refs[0].size() == 4095);
    CHECK(lng.refs[1] == "b.ts");
    // #EXTM3U atpažinimas (peržiūra ir patikra) — tas pats skaidymas: CR-only ir NUL pirmoje eilutėje
    CHECK(parse_playlist("#EXTM3U\r#EXTINF:2,\rs.ts\r").m3u);
    static const char kNulHead[] = "#EXTM3U\0junk";
    CHECK(parse_playlist(std::string(kNulHead, sizeof kNulHead - 1)).m3u);
    static const char kTs[] = "\x47\x00#EXTM3U\n";
    CHECK_FALSE(parse_playlist(std::string(kTs, sizeof kTs - 1)).m3u);
}

TEST_CASE("url_policy: C — URI atributai kaip libav ff_parse_key_value (kabutėse ir be)", "[ingest][url]") {
    // libav (utils.c n6.1:504-553): reikšmė kabutėse (\\ — kitas simbolis pažodžiui) arba be jų iki tarpo / kablelio; raktas — iki
    // pirmo '=', lyginamas tiksliai; tas pats raktas vėl — paskutinė reikšmė
    const PlaylistInfo b = parse_playlist(
        "#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=http://169.254.169.254/k\n#EXT-X-MAP:URI=init.mp4,BYTERANGE=10@0\n"
        "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=a,URI=a.m3u8\n#EXTINF:2,\ns.ts\n");
    REQUIRE(b.refs.size() == 3);
    CHECK(b.refs[0] == "http://169.254.169.254/k");
    CHECK(b.refs[1] == "init.mp4");
    CHECK(b.refs[2] == "s.ts");
    REQUIRE(b.variants.size() == 1);
    CHECK(b.variants[0] == "a.m3u8");
    const auto one = [](const std::string& tag) {
        const PlaylistInfo p = parse_playlist("#EXTM3U\n" + tag + "\n");
        return p.refs.empty() ? std::string("<nėra>") : p.refs.back();
    };
    CHECK(one("#EXT-X-KEY:METHOD=AES-128,URI=\"k\\\"x\\\\y\"") == "k\"x\\y");  // \" ir \\ kabutėse
    CHECK(one("#EXT-X-KEY:URI=\"a\",URI=\"b\"") == "b");                        // paskutinė reikšmė
    CHECK(one("#EXT-X-MAP: URI=m.mp4 BYTERANGE=1") == "m.mp4");                 // tarpas — skyriklis
    CHECK(one("#EXT-X-MAP:URI=\"unterminated") == "unterminated");              // be uždarančios kabutės — iki galo
    CHECK(one("#EXT-X-KEY:METHOD=AES-128,uri=\"x\"") == "<nėra>");             // raktas jautrus raidžių dydžiui
    CHECK(one("#EXT-X-KEY:METHOD=NONE,FOO BAR,URI=x") == "<nėra>");            // libav raktas „FOO BAR,URI=“ — ne URI
    CHECK(one("#EXT-X-SESSION-KEY:METHOD=AES-128,URI=sk.bin") == "sk.bin");     // libav neatidaro, bet tikrinama (griežčiau)
    // --hls-same-host: be kabučių nurodytas kitas serveris atmetamas
    HlsLimits lim;
    lim.only_authority = "127.0.0.1:8080";
    std::string why;
    CHECK_FALSE(check_playlist("http://127.0.0.1:8080/live/x.m3u8", b, true, lim, why));
    CHECK(why.find("169.254.169.254") != std::string::npos);
}

TEST_CASE("url_policy: D — autoritetas lyginamas normalizuotas (host raidės, numatytas prievadas, galinis taškas)", "[ingest][url]") {
    // libav jungiasi pagal av_url_split (utils.c n6.1:358-411): paskutinis '@', [IPv6], prievadas po pirmo ':'; nėra — 80 / 443
    CHECK(canonical_authority("http://Host.Example.:80/x") == "host.example:80");
    CHECK(canonical_authority("http://h/x") == "h:80");
    CHECK(canonical_authority("https://h/x") == "h:443");
    CHECK(canonical_authority("HTTPS://h:443") == "h:443");
    CHECK(canonical_authority("crypto+https://u:p@h/x.ts") == "h:443");
    CHECK(canonical_authority("http://a@b@H:1/") == "h:1");
    CHECK(canonical_authority("http://[::1]:8080/") == "[::1]:8080");
    CHECK(canonical_authority("http://[::1]/") == "[::1]:80");
    CHECK(canonical_authority("http://h:0080/") == "h:80");
    CHECK(canonical_authority("http://h:80x/") == "h:80x");  // ne skaitmenys — paliekama (nesutaps → atmetama)
    CHECK(canonical_authority("http://h:/") == "h:");
    CHECK(canonical_authority("ftp://h/x").empty());
    CHECK(canonical_authority("/local").empty());
    HlsLimits lim;
    lim.only_authority = canonical_authority("http://h/live/x.m3u8");
    std::string why;
    const std::string base = "http://h/live/x.m3u8";
    CHECK(check_playlist(base, parse_playlist("#EXTM3U\n#EXTINF:2,\nhttp://H.:80/s.ts\n#EXTINF:2,\nhttp://h:80/t.ts\n"), false, lim, why));
    CHECK_FALSE(check_playlist(base, parse_playlist("#EXTM3U\n#EXTINF:2,\nhttps://h/s.ts\n"), false, lim, why));  // 443 ≠ 80
    CHECK(why.find("--hls-same-host") != std::string::npos);
    CHECK_FALSE(check_playlist(base, parse_playlist("#EXTM3U\n#EXTINF:2,\nhttp://h:8080/s.ts\n"), false, lim, why));
}

TEST_CASE("url_policy: E — nuorodos išsprendžiamos kaip libav ff_make_absolute_url (taškų segmentai, užklausa, fragmentas)", "[ingest][url]") {
    // libav (url.c n6.1:166-319): URL su schema ir autoritetu — taškų segmentai suprastinami; tik užklausa / fragmentas — bazės kelias
    // išlieka; grojaraščių URL sąrašas (E) lyginamas su tuo, ką libav perduoda io_open, todėl turi sutapti iki simbolio
    const std::string base = "http://h:8080/a/b/master.m3u8?tok=1";
    const auto res = [&](const char* ref) {
        std::string out, why;
        return resolve_playlist_url(base, ref, out, why) ? out : "<klaida: " + why + ">";
    };
    CHECK(res("../v/x.m3u8") == "http://h:8080/a/v/x.m3u8");
    CHECK(res("./x.m3u8") == "http://h:8080/a/b/x.m3u8");
    CHECK(res("c/./d/../e.ts") == "http://h:8080/a/b/c/e.ts");
    CHECK(res("../../../../x.ts") == "http://h:8080/x.ts");  // virš šaknies — lieka šaknyje
    CHECK(res("/p/../q.ts") == "http://h:8080/q.ts");
    CHECK(res("?q=2") == "http://h:8080/a/b/master.m3u8?q=2");
    CHECK(res("#f") == "http://h:8080/a/b/master.m3u8?tok=1#f");
    CHECK(res("//o/p/../q") == "http://o/q");
    CHECK(res("http://o/a/../b?x=1") == "http://o/b?x=1");  // absoliutus su autoritetu — irgi suprastinamas
    CHECK(res("x.ts?s=1#t") == "http://h:8080/a/b/x.ts?s=1#t");
    std::string out, why;
    REQUIRE(resolve_playlist_url("https://h", "seg.ts", out, why));
    CHECK(out == "https://h/seg.ts");
    REQUIRE(resolve_playlist_url("http://h/dir/", "a/.", out, why));
    CHECK(out == "http://h/dir/a/");
}

TEST_CASE("url_policy: E — segmentų demuxeriui hls neleidžiamas (įdėtas grojaraštis vietoje segmento)", "[ingest][url]") {
    const std::string seg = segment_format_whitelist();
    CHECK(seg.find("hls") == std::string::npos);
    CHECK(seg == "mpegts,mov,mp4,m4a,3gp,3g2,mj2,aac");
    CHECK(format_whitelist_for("http", true) == "hls," + seg);  // viršutinis — hls + tie patys segmentų formatai
}

TEST_CASE("url_policy: F — EXT-X-MAP riba grojaraščiui ir init sekcijų atminties sąmata", "[ingest][url]") {
    // libav (hls.c n6.1:417-457, 860-896): kiekviena #EXT-X-MAP eilutė prideda init sekciją (struct segment + URL + rakto URL), perkrovimas
    // jų neatlaisvina (free_init_section_list — tik uždarant, hls.c:251-269) → riba eilutėms ir sesijos sąmatai
    HlsLimits lim;
    std::string why;
    const std::string base = "http://h/live/x.m3u8";
    std::string four = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n";
    for (int k = 0; k < 4; ++k) four += "#EXT-X-MAP:URI=\"i" + std::to_string(k) + ".mp4\"\n#EXTINF:2,\ns" + std::to_string(k) + ".m4s\n";
    const PlaylistInfo p4 = parse_playlist(four);
    CHECK(p4.maps.size() == 4);
    CHECK(check_playlist(base, p4, false, lim, why));
    const PlaylistInfo p5 = parse_playlist(four + "#EXT-X-MAP:URI=\"i4.mp4\"\n");
    CHECK_FALSE(check_playlist(base, p5, false, lim, why));
    CHECK(why.find("EXT-X-MAP") != std::string::npos);
    // tos pačios eilutės kartojimas — vis tiek nauja libav init sekcija (skaičiuojamos eilutės, ne skirtingi URI)
    std::string fifty = "#EXTM3U\n";
    for (int k = 0; k < 50; ++k) fifty += "#EXT-X-MAP:URI=\"init.mp4\"\n";
    CHECK_FALSE(check_playlist(base, parse_playlist(fifty), false, lim, why));
    // sąmata: kiekvienai MAP — 256 + |absoliutus MAP URL| + 1 (+ |absoliutus rakto URL| + 1, jei prieš ją buvo EXT-X-KEY su URI)
    const PlaylistInfo pk = parse_playlist("#EXTM3U\n#EXT-X-MAP:URI=\"i.mp4\"\n#EXT-X-KEY:METHOD=AES-128,URI=\"k.bin\"\n#EXT-X-MAP:URI=\"j.mp4\"\n");
    REQUIRE(pk.maps.size() == 2);
    const std::size_t i_url = std::string("http://h/live/i.mp4").size(), j_url = std::string("http://h/live/j.mp4").size(),
                      k_url = std::string("http://h/live/k.bin").size();
    CHECK(init_section_bytes(base, pk) == (256 + i_url + 1) + (256 + j_url + 1 + k_url + 1));
    CHECK(init_section_bytes(base, parse_playlist("#EXTM3U\n#EXTINF:2,\ns.ts\n")) == 0);
    CHECK(lim.max_maps == 4);
    CHECK(lim.max_init_bytes == std::size_t{16} << 20);
}

TEST_CASE("url_policy: visas grojaraštis tikrinamas — gylis, ribos, nuorodos (H2 / H3)", "[ingest][url]") {
    HlsLimits lim;
    std::string why;
    const std::string base = "http://127.0.0.1:8080/live/x.m3u8";
    // > 4 KiB komentarų, tada master eilutė (anksčiau sargas matė tik pirmus 4 KiB)
    const std::string padded = "#EXTM3U\n" + std::string(6000, '#') + "\n#EXT-X-STREAM-INF:BANDWIDTH=1\nx.m3u8\n";
    const PlaylistInfo pp = parse_playlist(padded);
    CHECK(pp.master);
    CHECK_FALSE(check_playlist(base, pp, false, lim, why));
    CHECK(why.find("įdėtas master") != std::string::npos);
    CHECK(check_playlist(base, pp, true, lim, why));  // viršutinis master — leidžiamas
    // variantų riba
    std::string many_v = "#EXTM3U\n";
    for (int k = 0; k < 40; ++k) many_v += "#EXT-X-STREAM-INF:BANDWIDTH=1\nv" + std::to_string(k) + ".m3u8\n";
    CHECK_FALSE(check_playlist(base, parse_playlist(many_v), true, lim, why));
    CHECK(why.find("variantų") != std::string::npos);
    // segmentų riba
    std::string many_s = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n";
    for (std::size_t k = 0; k <= lim.max_segments; ++k) many_s += "#EXTINF:2,\ns" + std::to_string(k) + ".ts\n";
    CHECK_FALSE(check_playlist(base, parse_playlist(many_s), false, lim, why));
    CHECK(why.find("nuorodų") != std::string::npos);
    // TARGETDURATION ir nuorodų schema — visame tekste, ne tik pradžioje
    CHECK_FALSE(check_playlist(base, parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:31\n#EXTINF:2,\ns.ts\n"), false, lim, why));
    CHECK(why.find("TARGETDURATION") != std::string::npos);
    const std::string late_file = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n" + std::string(8000, '#') + "\n#EXTINF:2,\nfile:///etc/passwd\n";
    CHECK_FALSE(check_playlist(base, parse_playlist(late_file), false, lim, why));
    CHECK(why.find("neleidžiama") != std::string::npos);
    CHECK(check_playlist(base, parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\ns.ts\n"), false, lim, why));
    CHECK_FALSE(check_playlist(base, parse_playlist("not m3u"), true, lim, why));
}

TEST_CASE("url_policy: garso žiedo dydis baitais (2 laipsnis × kanalai × float)", "[ingest][url]") {
    CHECK(audio_ring_bytes(2, 211200) == 262144ull * 2 * 4);
    CHECK(audio_ring_bytes(16, 93 * 48000) == 8388608ull * 16 * 4);  // HLS 30 s segmentai, 16 kanalų — 512 MB
    CHECK(audio_ring_bytes(1, 1) == 4);
}

TEST_CASE("url_policy: L1 — kabutėse URL su tarpu kelyje išvalomas iki uždarančios kabutės", "[ingest][url]") {
    // libav HLS rašo „Opening '<URL>' for reading“: URL kelyje gali būti tarpas, po jo — užklausa su žetonu
    const std::string r = redact_urls_in("[hls @ 0x1] Opening 'http://h/live seg 1.ts?token=SECRET9' for reading\n");
    CHECK(r.find("SECRET9") == std::string::npos);
    CHECK(r.find("token=***' for reading") != std::string::npos);
    const std::string q = redact_urls_in("url \"srt://h:1?x=1 &passphrase=PW77\" end");
    CHECK(q.find("PW77") == std::string::npos);
    CHECK(q.find("\" end") != std::string::npos);
    // be kabučių — kaip iki šiol (iki tarpo); neuždaryta kabutė — iki eilutės galo
    CHECK(redact_urls_in("a http://h/p?t=1 b") == "a http://h/p?t=*** b");
    CHECK(redact_urls_in("'http://h/p q?t=S1\nnext").find("S1") == std::string::npos);
}

TEST_CASE("url_policy: L1 — SRT slaptažodis iš aplinkos / failo, ne argv", "[ingest][url]") {
    std::string v, why;
    ::setenv("VRAMFLOW_TEST_SRT_PASS", "Abcdefgh12345", 1);
    REQUIRE(secret_from("env:VRAMFLOW_TEST_SRT_PASS", v, why));
    CHECK(v == "Abcdefgh12345");
    ::unsetenv("VRAMFLOW_TEST_SRT_PASS");
    CHECK_FALSE(secret_from("env:VRAMFLOW_TEST_SRT_PASS", v, why));
    CHECK(why.find("VRAMFLOW_TEST_SRT_PASS") != std::string::npos);
    CHECK_FALSE(secret_from("file:/nonexistent/vramflow-pass", v, why));
    CHECK_FALSE(secret_from("Abcdefgh12345", v, why));  // pats slaptažodis argumente — neleidžiama (būtų /proc/*/cmdline)
    CHECK(why.find("Abcdefgh12345") == std::string::npos);
    // srt URL papildomas tik atmintyje; jau su passphrase — nekeičiamas; ne srt — nekeičiamas
    CHECK(with_srt_passphrase("srt://h:1?mode=listener", "Abcdefgh12345") == "srt://h:1?mode=listener&passphrase=Abcdefgh12345");
    CHECK(with_srt_passphrase("srt://h:1", "Abcdefgh12345") == "srt://h:1?passphrase=Abcdefgh12345");
    CHECK(with_srt_passphrase("srt://h:1?passphrase=X1234567890", "Abcdefgh12345") == "srt://h:1?passphrase=X1234567890");
    CHECK(with_srt_passphrase("udp://h:1", "Abcdefgh12345") == "udp://h:1");
    CHECK(srt_passphrase_ok("Abcdefgh12345", why));
    CHECK_FALSE(srt_passphrase_ok("short", why));                    // SRT: 10–79 simbolių
    CHECK_FALSE(srt_passphrase_ok("Abcdefgh&mode=caller", why));     // užklausos skyrikliai sugadintų URL
    CHECK(why.find("Abcdefgh") == std::string::npos);
}

TEST_CASE("url_policy: G — SRT slaptažodyje '+' neleidžiamas (libav jį verčia tarpu)", "[ingest][url]") {
    // libsrt.c n6.1:548-550 passphrase ima av_find_info_tag, kuris '+' pakeičia tarpu (parseutils.c n6.1:774-775); % nedekoduojamas
    std::string why;
    CHECK_FALSE(srt_passphrase_ok("Abcdefgh+12345", why));
    CHECK(why.find('+') != std::string::npos);
    CHECK(why.find("Abcdefgh") == std::string::npos);
    CHECK(srt_passphrase_ok("Abcdefgh-12345_x", why));
}

TEST_CASE("url_policy: --hls-same-host — nuorodos tik į grojaraščio serverį", "[ingest][url]") {
    CHECK(url_authority("http://User:pw@Host.Example:8080/a?b") == "host.example:8080");
    CHECK(url_authority("https://h/x") == "h");
    HlsLimits lim;
    lim.only_authority = "127.0.0.1:8080";
    std::string why;
    const std::string base = "http://127.0.0.1:8080/live/x.m3u8";
    CHECK(check_playlist(base, parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\ns.ts\n#EXTINF:2,\n/abs/t.ts\n"), false, lim, why));
    CHECK_FALSE(check_playlist(base, parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\nhttp://169.254.169.254/latest\n"), false, lim, why));
    CHECK(why.find("--hls-same-host") != std::string::npos);
    CHECK_FALSE(check_playlist(base, parse_playlist("#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"http://127.0.0.1:9/k\"\n"), false, lim, why));
    lim.only_authority.clear();  // numatyta — bet koks http(s) serveris (SSRF dokumentuota)
    CHECK(check_playlist(base, parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\nhttp://169.254.169.254/latest\n"), false, lim, why));
}

TEST_CASE("url_policy: H1 — autoritetas tiksliai kaip libav av_url_split; http(s) be „://“ — atmetama", "[ingest][url]") {
    // av_url_split (utils.c n6.1:377-411): schema iki PIRMO ':', po jo praleidžiami ne daugiau kaip du '/', autoritetas iki "/?#";
    // http jungiasi būtent į šį host'ą (http.c n6.1:213-215), net jei toliau kelyje yra „://kitas“
    CHECK(url_authority("http:evil.com/x://allowed.com/s.ts") == "evil.com");
    CHECK(url_authority("http:10.0.0.5/?u=http://allowed.com/") == "10.0.0.5");
    CHECK(url_authority("http:/evil.com/x://allowed.com/") == "evil.com");
    CHECK(url_authority("http:///x://allowed.com/").empty());
    CHECK(url_authority("crypto+http:evil.com/x://allowed.com/s.ts") == "evil.com");
    CHECK(url_authority("http://u@evil.com@allowed.com/").find("allowed.com") == 0);  // paskutinis '@' (utils.c:397-401)
    // --hls-same-host lyginimo forma: schema be „://“ iš karto — niekada nesutampa (fail closed)
    for (const char* u : {"http:evil.com/x://allowed.com/s.ts", "http:10.0.0.5/?u=http://allowed.com/", "http:allowed.com/s.ts",
                          "https:/allowed.com/s.ts", "crypto+http:evil.com/x://allowed.com/s.ts", "crypto+https:allowed.com/x://allowed.com/"}) {
        INFO(u);
        CHECK(canonical_authority(u).empty());
        CHECK(http_scheme_without_slashes(u));
    }
    CHECK_FALSE(http_scheme_without_slashes("http://allowed.com/x://evil.com/"));
    CHECK_FALSE(http_scheme_without_slashes("crypto+https://allowed.com/s.ts"));
    CHECK_FALSE(http_scheme_without_slashes("data:,x"));
    CHECK(canonical_authority("http://allowed.com/x://evil.com/") == "allowed.com:80");
    CHECK(canonical_authority("http://" + std::string(1024, 'a') + "/").empty());  // libav hostname[1024] — ilgesnis nukerpamas
    // visos trys patikros naudoja check_playlist / canonical_authority: grojaraščio nuorodos atmetamos su --hls-same-host ir be jo
    HlsLimits lim;
    lim.only_authority = "allowed.com:80";
    std::string why;
    const std::string base = "http://allowed.com/live/x.m3u8";
    for (const char* ref : {"http:evil.com/x://allowed.com/s.ts", "http:10.0.0.5/?u=http://allowed.com/", "http:/evil.com/s.ts"}) {
        INFO(ref);
        CHECK_FALSE(check_playlist(base, parse_playlist(std::string("#EXTM3U\n#EXTINF:2,\n") + ref + "\n"), false, lim, why));
        CHECK(why.find("av_url_split") != std::string::npos);
        HlsLimits any;
        CHECK_FALSE(check_playlist(base, parse_playlist(std::string("#EXTM3U\n#EXTINF:2,\n") + ref + "\n"), false, any, why));
        std::string out;
        CHECK_FALSE(resolve_playlist_url(base, ref, out, why));
        CHECK(why.find("neleidžiama") != std::string::npos);
    }
    CHECK_FALSE(check_playlist(base, parse_playlist("#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"http:evil.com/k://allowed.com/k\"\n"), false, lim, why));
    // operatoriaus URL — irgi tik „schema://“
    CHECK_FALSE(source_url_allowed("http:allowed.com/live.m3u8", why));
    CHECK(why.find("av_url_split") != std::string::npos);
    CHECK(source_url_allowed("http://allowed.com/live.m3u8", why));
    // paslaptys: userinfo išvaloma visada (anksčiau „://“ kelyje palikdavo pradžią su slaptažodžiu)
    CHECK(redact_url("http:user:SECRETPW@evil.com/x://allowed.com/") == "http:***");
    CHECK(redact_url("https:/u:SECRETPW@h/p") == "https:***");
    for (const char* line : {"[hls @ 0x1] Opening 'http:user:SECRETPW@evil.com/x://allowed.com/s.ts' for reading",
                             "err http:user:SECRETPW@evil.com/x://allowed.com/s.ts end", "(crypto+http:u:SECRETPW@e/x://a/k)"}) {
        INFO(line);
        const std::string r = redact_urls_in(line);
        CHECK(r.find("SECRETPW") == std::string::npos);
    }
    CHECK(redact_urls_in("a http://h/p?t=1 b") == "a http://h/p?t=*** b");  // įprasti URL — kaip iki šiol
}
