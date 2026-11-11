// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// CPU media branduoliai (variantas C) prieš nepriklausomą media_ref etaloną: 0 skirtingų baitų.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstring>
#include <random>
#include <vector>

#include "media/cpu/cpu_media.hpp"
#include "media/media_ref.hpp"

using namespace mxlv::media;

namespace {

std::vector<std::uint32_t> random_field(std::mt19937& rng) {
    // legalus v210: kiekvienas 10 bitų mėginys 4..1019, bitai 30–31 atsitiktiniai (turi būti išsaugoti)
    std::vector<std::uint32_t> f(kFieldWords);
    std::uniform_int_distribution<int> s(4, 1019), hi(0, 3);
    for (auto& w : f)
        w = static_cast<std::uint32_t>(s(rng)) | (static_cast<std::uint32_t>(s(rng)) << 10) |
            (static_cast<std::uint32_t>(s(rng)) << 20) | (static_cast<std::uint32_t>(hi(rng)) << 30);
    return f;
}

cpu::LogoPlanes planes(const ref::Logo& l, int x, int y) {
    cpu::LogoPlanes p{};
    p.y10 = l.y10.data();
    p.a8 = l.a8.data();
    p.cb10 = l.cb10.data();
    p.cr10 = l.cr10.data();
    p.ac8 = l.ac8.data();
    p.lw = l.w;
    p.lh = l.h;
    p.x = x;
    p.y = y;
    return p;
}

std::size_t diff_words(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}

}  // namespace

TEST_CASE("cpu logo == media_ref (pozicijos, paritetai, visos alfa)", "[cpu]") {
    std::mt19937 rng(11);
    const auto logo = ref::builtin_logo();
    const auto [x, y] = GENERATE(std::pair{0, 0}, std::pair{1488, 64}, std::pair{1536, 864}, std::pair{100, 33});
    const int parity = GENERATE(0, 1);
    const auto in = random_field(rng);
    std::vector<std::uint32_t> a(kFieldWords), b(kFieldWords);
    ref::logo_field(in.data(), a.data(), logo, x, y, parity);
    cpu::logo_field(in.data(), b.data(), planes(logo, x, y), parity);
    INFO("x=" << x << " y=" << y << " p=" << parity);
    CHECK(diff_words(a, b) == 0);
    // in-place variantas
    auto c = in;
    cpu::logo_field(c.data(), c.data(), planes(logo, x, y), parity);
    CHECK(diff_words(a, c) == 0);
}

TEST_CASE("cpu weave/split == media_ref", "[cpu]") {
    std::mt19937 rng(12);
    const int pitch = GENERATE(1920, 2048);
    for (int it = 0; it < 3; ++it) {
        const auto top = random_field(rng), bot = random_field(rng);
        std::vector<std::uint8_t> ya(pitch * 1080), uva(pitch * 540), yb(pitch * 1080), uvb(pitch * 540);
        ref::weave_nv12(top.data(), bot.data(), ya.data(), uva.data(), pitch, pitch);
        cpu::weave_nv12(top.data(), bot.data(), yb.data(), uvb.data(), pitch, pitch);
        std::size_t dy = 0, duv = 0;
        for (int r = 0; r < 1080; ++r) dy += std::memcmp(ya.data() + r * pitch, yb.data() + r * pitch, 1920) != 0;
        for (int r = 0; r < 540; ++r) duv += std::memcmp(uva.data() + r * pitch, uvb.data() + r * pitch, 1920) != 0;
        CHECK(dy == 0);
        CHECK(duv == 0);
        for (int p = 0; p < 2; ++p) {
            std::vector<std::uint32_t> fa(kFieldWords), fb(kFieldWords);
            ref::split_nv12(ya.data(), uva.data(), pitch, pitch, p, fa.data());
            cpu::split_nv12(ya.data(), uva.data(), pitch, pitch, p, fb.data());
            CHECK(diff_words(fa, fb) == 0);
        }
    }
}

TEST_CASE("cpu split_yuv420p == media_ref split_nv12 (tie patys chroma mėginiai)", "[cpu]") {
    std::mt19937 rng(13);
    std::uniform_int_distribution<int> b(0, 255);
    const int py = 1920, pc = 960;
    std::vector<std::uint8_t> y(py * 1080), uv(1920 * 540), u(pc * 540), v(pc * 540);
    for (auto& x : y) x = static_cast<std::uint8_t>(b(rng));
    for (int r = 0; r < 540; ++r)
        for (int c = 0; c < 960; ++c) {
            u[r * pc + c] = uv[r * 1920 + 2 * c] = static_cast<std::uint8_t>(b(rng));
            v[r * pc + c] = uv[r * 1920 + 2 * c + 1] = static_cast<std::uint8_t>(b(rng));
        }
    for (int p = 0; p < 2; ++p) {
        std::vector<std::uint32_t> a(kFieldWords), c(kFieldWords);
        ref::split_nv12(y.data(), uv.data(), py, 1920, p, a.data());
        cpu::split_yuv420p(y.data(), u.data(), v.data(), py, pc, p, c.data());
        CHECK(diff_words(a, c) == 0);
    }
}

TEST_CASE("cpu gen == media_ref", "[cpu]") {
    const std::uint32_t cx = GENERATE(0u, 1u, 2u, 3u);
    for (std::uint64_t idx : {1000ull, 1001ull, 89717224826ull}) {
        std::vector<std::uint32_t> a(kFieldWords), b(kFieldWords);
        const GenParams g{0x1234ull, 3, cx};
        ref::gen_field(g, idx, a.data());
        cpu::gen_field(g, idx, b.data());
        CHECK(diff_words(a, b) == 0);
    }
}

// --- P2: formatui neutralūs CPU branduoliai == etalonas ----------------------------------------------------------------
namespace {
struct FCase {
    Dim d;
    int field;
};
const FCase kFCases[] = {{{1920, 1080}, 0}, {{1920, 1080}, 1}, {{1920, 1080}, -1}, {{1280, 720}, -1}, {{3840, 2160}, -1}};
std::vector<std::uint32_t> rnd_words(std::mt19937& rng, std::size_t n) {
    std::vector<std::uint32_t> v(n);
    for (auto& w : v) w = rng();
    return v;
}
std::vector<std::uint8_t> rnd_bytes(std::mt19937& rng, std::size_t n) {
    std::vector<std::uint8_t> v(n);
    for (auto& b : v) b = static_cast<std::uint8_t>(rng());
    return v;
}
}  // namespace

TEST_CASE("media cpu: gen ir logo == ref (visi matmenys, laukas ir kadras)", "[cpu][format]") {
    std::mt19937 rng(41);
    const ref::Logo L = ref::builtin_logo();
    for (const FCase& c : kFCases) {
        const std::size_t n = geom_of(c.d, c.field).grain_words();
        std::vector<std::uint32_t> got(n, 0xFFFFFFFFu), want(n);
        const std::uint64_t index = c.field == 1 ? 51 : 50;
        cpu::gen_field({7, 2, 3}, index, c.d, c.field >= 0, got.data());
        ref::gen_field({7, 2, 3}, index, c.d, c.field >= 0, want.data());
        INFO("w " << c.d.w << " field " << c.field);
        CHECK(diff_words(got, want) == 0);
        const auto bg = rnd_words(rng, n);
        for (const int x : {0, c.d.w - L.w, c.d.w - 100})
            for (const int y : {0, c.d.h - 50}) {
                cpu::logo_field(bg.data(), got.data(), planes(L, x, y), c.d, c.field);
                ref::logo_field(bg.data(), want.data(), L, x, y, c.d, c.field);
                INFO("X " << x << " Y " << y);
                CHECK(diff_words(got, want) == 0);
            }
    }
}

TEST_CASE("media cpu: weave, split (abi chromos), split_yuv420p == ref", "[cpu][format]") {
    std::mt19937 rng(43);
    for (const FCase& c : kFCases) {
        const bool inter = c.field >= 0;
        const std::size_t n = geom_of(c.d, c.field).grain_words();
        for (const int pitch : {c.d.w, c.d.w + 128}) {
            const auto t = rnd_words(rng, n), b = rnd_words(rng, n);
            std::vector<std::uint8_t> gy(static_cast<std::size_t>(pitch) * c.d.h, 3), guv(static_cast<std::size_t>(pitch) * c.d.h / 2, 3);
            auto wy = gy, wuv = guv;
            if (c.field != 1) {
                cpu::weave_nv12(t.data(), b.data(), gy.data(), guv.data(), pitch, pitch, c.d, inter);
                ref::weave_nv12(t.data(), inter ? b.data() : nullptr, wy.data(), wuv.data(), pitch, pitch, c.d, inter);
                INFO("weave w " << c.d.w << " inter " << inter << " pitch " << pitch);
                CHECK(gy == wy);
                CHECK(guv == wuv);
            }
            const auto y = rnd_bytes(rng, static_cast<std::size_t>(pitch) * c.d.h);
            const auto uv = rnd_bytes(rng, static_cast<std::size_t>(pitch) * c.d.h / 2);
            // planar — tie patys chroma mėginiai: u = uv[2i], v = uv[2i+1], pitch_c = pitch / 2
            std::vector<std::uint8_t> u(uv.size() / 2), v(uv.size() / 2);
            for (std::size_t i = 0; i < u.size(); ++i) {
                u[i] = uv[2 * i];
                v[i] = uv[2 * i + 1];
            }
            for (const ChromaMode m : {ChromaMode::FieldSited, ChromaMode::FrameSited}) {
                if (!inter && m == ChromaMode::FieldSited) continue;
                std::vector<std::uint32_t> got(n, 1), want(n, 2), gp(n, 3);
                cpu::split_nv12(y.data(), uv.data(), pitch, pitch, c.d, c.field, m, got.data());
                ref::split_nv12(y.data(), uv.data(), pitch, pitch, c.d, c.field, m, want.data());
                cpu::split_yuv420p(y.data(), u.data(), v.data(), pitch, pitch / 2, c.d, c.field, m, gp.data());
                INFO("split w " << c.d.w << " field " << c.field << " frame " << (m == ChromaMode::FrameSited) << " pitch " << pitch);
                CHECK(diff_words(got, want) == 0);
                CHECK(diff_words(gp, want) == 0);
            }
            if (!inter) {
                std::vector<std::uint32_t> got(n, 1), want(n, 2);
                cpu::split_nv12_progressive(y.data(), uv.data(), pitch, pitch, c.d, got.data());
                ref::split_nv12_progressive(y.data(), uv.data(), pitch, pitch, c.d, want.data());
                CHECK(diff_words(got, want) == 0);
            }
        }
    }
}
