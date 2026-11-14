// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// URL politika (round 9, A1): šaltinio schemos, libav baltieji sąrašai, grojaraščio URL (tik http / https), paslapčių valymas,
// grojaraščio analizė, garso žiedo dydis.
#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("url_policy: garso žiedo dydis baitais (2 laipsnis × kanalai × float)", "[ingest][url]") {
    CHECK(audio_ring_bytes(2, 211200) == 262144ull * 2 * 4);
    CHECK(audio_ring_bytes(16, 93 * 48000) == 8388608ull * 16 * 4);  // HLS 30 s segmentai, 16 kanalų — 512 MB
    CHECK(audio_ring_bytes(1, 1) == 4);
}
