// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// TsSender (OK-1, planas operatoriaus-kelias 2 užduotis): muxer'is → ribota eilė → siuntimo gija → avio. Tinklas imituojamas
// savu AVIOContext (rašymo callback'as su gedimais / blokavimu); open — imitacija, kuri gerbia FFmpeg pertraukimo callback'ą.
// Tikrinama: push niekada neblokuoja, pakartotiniai bandymai su atidėjimu, atsijungimas ir vėl prisijungimas, seniausių
// 188 B paketų išmetimas pilnoje eilėje (imtuvas gauna tik sveikus, iš eilės einančius paketus), stop() ≤ 1 s.
extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
}

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "func/ts_sender.hpp"

using namespace std::chrono_literals;
using mxlv::func::SenderStats;
using mxlv::func::TsSender;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::size_t kTs = 188;
constexpr std::size_t kChunk = 7 * kTs;  // 1316 — kaip muxer'io AVIOContext buferis

// Imituotas tinklas: ką „išsiuntė“ rašymo callback'as, kada jis grąžina klaidą arba blokuojasi.
struct FakeNet {
    std::mutex m;
    std::vector<std::uint8_t> got;     // visi priimti baitai (per visus prisijungimus)
    std::atomic<int> opens{0};         // open kvietimų skaičius
    std::atomic<int> fail_opens{0};    // tiek pirmų open grąžina klaidą
    std::atomic<bool> block_open{false};   // open laukia, kol pertraukimo callback'as grąžins 1
    std::atomic<bool> release_open{false};  // block_open: atleisti (open pavyksta)
    std::atomic<long> fail_write_after{-1};  // po tiek priimtų baitų vienas rašymas grąžina klaidą (−1 — niekada)
    std::atomic<bool> block_write{false};    // rašymas laukia pertraukimo
    std::atomic<std::size_t> bad_writes{0};  // rašymo kvietimų, kurių dydis ne 188 kartotinis
    AVIOInterruptCB cb{};                    // paskutinio open pertraukimo callback'as
    std::size_t bytes() {
        std::lock_guard g(m);
        return got.size();
    }
};

#if FF_API_AVIO_WRITE_NONCONST
int fake_write(void* opaque, std::uint8_t* buf, int n) {
#else
int fake_write(void* opaque, const std::uint8_t* buf, int n) {
#endif
    auto* net = static_cast<FakeNet*>(opaque);
    if (static_cast<std::size_t>(n) % kTs != 0) ++net->bad_writes;
    if (net->block_write.load()) {
        while (!(net->cb.callback && net->cb.callback(net->cb.opaque))) std::this_thread::sleep_for(5ms);
        return AVERROR_EXIT;
    }
    std::lock_guard g(net->m);
    const long after = net->fail_write_after.load();
    if (after >= 0 && static_cast<long>(net->got.size()) >= after) {
        net->fail_write_after = -1;
        return AVERROR(EIO);
    }
    net->got.insert(net->got.end(), buf, buf + n);
    return n;
}

TsSender::OpenFn open_fn(FakeNet& net) {
    return [&net](AVIOContext** pb, const AVIOInterruptCB& cb) {
        net.cb = cb;
        ++net.opens;
        if (net.fail_opens.load() > 0) {
            --net.fail_opens;
            return AVERROR(ECONNREFUSED);
        }
        while (net.block_open.load() && !net.release_open.load()) {
            if (cb.callback(cb.opaque)) return AVERROR_EXIT;
            std::this_thread::sleep_for(5ms);
        }
        auto* buf = static_cast<std::uint8_t*>(av_malloc(kChunk));
        *pb = avio_alloc_context(buf, static_cast<int>(kChunk), 1, &net, nullptr, fake_write, nullptr);
        return 0;
    };
}

void close_fn(AVIOContext** pb) {
    if (!*pb) return;
    avio_flush(*pb);
    av_freep(&(*pb)->buffer);
    avio_context_free(pb);
}

// TS paketų blokas: kiekvienas 188 B paketas prasideda 0x47, toliau — eilės numeris (4 B), kad imtuvo pusėje būtų matyti,
// ar paketai sveiki ir iš eilės.
std::vector<std::uint8_t> ts_block(std::uint32_t& seq, std::size_t packets) {
    std::vector<std::uint8_t> b(packets * kTs, 0xff);
    for (std::size_t k = 0; k < packets; ++k) {
        std::uint8_t* p = b.data() + k * kTs;
        p[0] = 0x47;
        std::memcpy(p + 1, &seq, sizeof seq);
        ++seq;
    }
    return b;
}

std::uint32_t seq_at(const std::vector<std::uint8_t>& v, std::size_t packet) {
    std::uint32_t s = 0;
    std::memcpy(&s, v.data() + packet * kTs + 1, sizeof s);
    return s;
}

template <class F>
bool wait_until(F f, std::chrono::milliseconds limit) {
    const auto end = Clock::now() + limit;
    while (Clock::now() < end) {
        if (f()) return true;
        std::this_thread::sleep_for(2ms);
    }
    return f();
}

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

}  // namespace

TEST_CASE("ts_sender: open klaida 3 kartus, paskui pavyksta — reconnects 3, push niekada > 1 ms", "[ts_sender][realtime]") {
    FakeNet net;
    net.fail_opens = 3;
    TsSender s("udp://imitacija:1", open_fn(net), 400 * kChunk, close_fn);
    std::uint32_t seq = 0;
    double max_push_ms = 0;
    const auto t0 = Clock::now();
    // ~4 s pastovus srautas: kas 1 ms vienas 1316 B blokas (≈ 10,5 Mbit/s)
    while (ms_since(t0) < 4500 && !(s.stats().connected && net.bytes() >= 100 * kChunk)) {
        const std::vector<std::uint8_t> b = ts_block(seq, 7);
        const auto p0 = Clock::now();
        s.push(b.data(), b.size());
        max_push_ms = std::max(max_push_ms, ms_since(p0));
        std::this_thread::sleep_for(1ms);
    }
    const double connected_after_ms = ms_since(t0);
    const SenderStats st = s.stats();
    s.stop();
    INFO("prisijungta po " << connected_after_ms << " ms, max push " << max_push_ms << " ms");
    CHECK(net.opens.load() == 4);
    CHECK(st.reconnects == 3);
    CHECK(st.connected);
    CHECK(max_push_ms < 1.0);
    // atidėjimas 0,5 + 1 + 2 s prieš ketvirtą bandymą
    CHECK(connected_after_ms >= 3400);
    CHECK(net.bad_writes.load() == 0);
}

TEST_CASE("ts_sender: rašymo klaida viduryje — atsijungia, vėl prisijungia, viskas apskaityta", "[ts_sender]") {
    FakeNet net;
    net.fail_write_after = static_cast<long>(50 * kChunk);
    TsSender s("srt://imitacija:2?mode=listener", open_fn(net), 4000 * kChunk, close_fn);  // atidėjimo 0,5 s metu nieko neišmeta
    std::uint32_t seq = 0;
    std::size_t pushed = 0;
    bool saw_disconnect = false;
    const auto t0 = Clock::now();
    while (ms_since(t0) < 3000) {
        const std::vector<std::uint8_t> b = ts_block(seq, 7);
        s.push(b.data(), b.size());
        pushed += b.size();
        if (!s.stats().connected && net.opens.load() >= 1 && net.bytes() > 0) saw_disconnect = true;
        if (net.opens.load() >= 2 && s.stats().connected && net.bytes() > 100 * kChunk) break;
        std::this_thread::sleep_for(1ms);
    }
    const SenderStats mid = s.stats();
    CHECK(net.opens.load() == 2);
    CHECK(mid.reconnects == 1);
    CHECK(mid.connected);
    CHECK(saw_disconnect);
    s.stop();
    const SenderStats st = s.stats();
    CHECK(st.connected);  // po stop() — būsena sustabdymo metu
    CHECK(st.bytes_sent == net.bytes());
    CHECK(st.bytes_sent % kTs == 0);
    // kiekvienas įstumtas paketas — arba išsiųstas, arba suskaičiuotas kaip išmestas (nesėkmingas rašymas)
    CHECK(st.bytes_sent + st.packets_dropped * kTs == pushed);
    CHECK(st.packets_dropped == 7);  // vienas nepavykęs 1316 B rašymas
    std::lock_guard g(net.m);
    for (std::size_t k = 0; k < net.got.size() / kTs; ++k) REQUIRE(net.got[k * kTs] == 0x47);
}

TEST_CASE("ts_sender: eilė pilna — seniausi 188 B paketai išmetami, imtuvas gauna sveiką galą", "[ts_sender]") {
    FakeNet net;
    net.block_open = true;  // imtuvas dar neprisijungęs (SRT listener be caller'io)
    constexpr std::size_t kCap = 10 * kChunk;
    TsSender s("srt://imitacija:3?mode=listener", open_fn(net), kCap, close_fn);
    std::uint32_t seq = 0;
    for (int k = 0; k < 100; ++k) {
        // blokai skirtingo dydžio (1–9 paketai), kaip muxer'io flush'ai
        const std::vector<std::uint8_t> b = ts_block(seq, static_cast<std::size_t>(1 + k % 9));
        s.push(b.data(), b.size());
    }
    const std::uint32_t total = seq;
    SenderStats st = s.stats();
    CHECK_FALSE(st.connected);
    CHECK(st.packets_dropped == total - kCap / kTs);
    net.release_open = true;
    REQUIRE(wait_until([&] { return net.bytes() >= kCap; }, 2000ms));
    s.stop();
    st = s.stats();
    CHECK(st.bytes_sent % kTs == 0);
    CHECK(st.bytes_sent == kCap);
    CHECK(net.bad_writes.load() == 0);
    std::lock_guard g(net.m);
    REQUIRE(net.got.size() == kCap);
    // gautas lygiai paskutinių kCap / 188 paketų galas, iš eilės, kiekvienas prasideda 0x47
    for (std::size_t k = 0; k < kCap / kTs; ++k) {
        REQUIRE(net.got[k * kTs] == 0x47);
        REQUIRE(seq_at(net.got, k) == total - kCap / kTs + k);
    }
}

TEST_CASE("ts_sender: stop() per užblokuotą open — ≤ 1 s", "[ts_sender]") {
    FakeNet net;
    net.block_open = true;
    TsSender s("srt://imitacija:4?mode=listener", open_fn(net), 100 * kChunk, close_fn);
    REQUIRE(wait_until([&] { return net.opens.load() == 1; }, 1000ms));
    std::uint32_t seq = 0;
    const std::vector<std::uint8_t> b = ts_block(seq, 7);
    s.push(b.data(), b.size());
    const auto t0 = Clock::now();
    s.stop();
    const double ms = ms_since(t0);
    INFO("stop " << ms << " ms");
    CHECK(ms <= 1000);
    CHECK(net.opens.load() == 1);  // sustabdžius nebandoma jungtis iš naujo
    const SenderStats st = s.stats();
    CHECK_FALSE(st.connected);
    CHECK(st.packets_dropped == 7);  // neišsiųsti eilės paketai suskaičiuoti
}

TEST_CASE("ts_sender: stop() per užblokuotą rašymą — ≤ 1 s", "[ts_sender]") {
    FakeNet net;
    TsSender s("srt://imitacija:5?mode=caller", open_fn(net), 100 * kChunk, close_fn);
    std::uint32_t seq = 0;
    {
        const std::vector<std::uint8_t> b = ts_block(seq, 7);
        s.push(b.data(), b.size());
    }
    REQUIRE(wait_until([&] { return net.bytes() == kChunk; }, 2000ms));
    net.block_write = true;  // pvz. SRT siuntimo buferis pilnas, imtuvas nebeatsako
    for (int k = 0; k < 20; ++k) {
        const std::vector<std::uint8_t> b = ts_block(seq, 7);
        s.push(b.data(), b.size());
    }
    std::this_thread::sleep_for(50ms);
    const auto t0 = Clock::now();
    s.stop();
    const double ms = ms_since(t0);
    INFO("stop " << ms << " ms");
    CHECK(ms <= 1000);
    const SenderStats st = s.stats();
    CHECK(st.bytes_sent == kChunk);
    CHECK(st.bytes_sent + st.packets_dropped * kTs == static_cast<std::uint64_t>(seq) * kTs);
}

TEST_CASE("ts_sender: stop() be klaidų — eilė išsiunčiama iki galo", "[ts_sender]") {
    FakeNet net;
    TsSender s("udp://imitacija:6", open_fn(net), 100 * kChunk, close_fn);
    REQUIRE(wait_until([&] { return s.stats().connected; }, 1000ms));
    std::uint32_t seq = 0;
    for (int k = 0; k < 50; ++k) {
        const std::vector<std::uint8_t> b = ts_block(seq, 7);
        s.push(b.data(), b.size());
    }
    s.stop();
    const SenderStats st = s.stats();
    CHECK(st.packets_dropped == 0);
    CHECK(st.bytes_sent == static_cast<std::uint64_t>(seq) * kTs);
    CHECK(st.reconnects == 0);
}
