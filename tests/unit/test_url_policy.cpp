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
