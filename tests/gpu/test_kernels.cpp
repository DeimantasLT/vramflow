// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// GPU testai be MXL (architektūra §5.3): kernel'iai prieš skaliarinę CPU implementaciją.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/header.hpp"
#include "common/pattern.hpp"
#include "gpu/kernels.hpp"

using namespace mxlv;

namespace {

// Catch2 REQUIRE nėra saugus iš kelių gijų, todėl CUDA klaidos — išimtys (pagrindinėje gijoje Catch jas
// pažymi kaip FAIL; darbinės gijos jas sugauna ir skaičiuoja).
#define CUDA_OK(x)                                                                                   \
    do {                                                                                             \
        const cudaError_t _e = (x);                                                                  \
        if (_e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(_e)); \
    } while (0)

struct Buf {
    std::uint32_t* words = nullptr;
    std::uint64_t* fold = nullptr;
    gpu::VerifyResult* res = nullptr;
    cudaStream_t s{};
    std::uint32_t n = 0;
    explicit Buf(std::uint32_t n_) : n(n_) {
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&words), std::size_t{n} * 4));
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&fold), 8));
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&res), sizeof(gpu::VerifyResult)));
        CUDA_OK(cudaStreamCreate(&s));
    }
    ~Buf() {
        cudaFree(words);
        cudaFree(fold);
        cudaFree(res);
        cudaStreamDestroy(s);
    }
    void write(std::uint64_t seed, std::uint64_t index) {
        CUDA_OK(cudaMemsetAsync(fold, 0, 8, s));
        CUDA_OK(gpu::fill(words, seed, index, n, 0, n, fold, s));
        GrainHeader h{};
        h.magic = kMagic;
        h.version = kHeaderVersion;
        h.header_size = 64;
        h.flow_seed = seed;
        h.grain_index = index;
        h.payload_words = n;
        CUDA_OK(gpu::finalize_header(words, h, fold, s));
    }
    gpu::VerifyResult check(std::uint64_t seed, std::uint64_t index) {
        gpu::VerifyResult r{};
        CUDA_OK(gpu::verify(words, seed, index, n, res, s));
        CUDA_OK(cudaMemcpyAsync(&r, res, sizeof r, cudaMemcpyDeviceToHost, s));
        CUDA_OK(cudaStreamSynchronize(s));
        return r;
    }
    std::vector<std::uint32_t> download() {
        std::vector<std::uint32_t> v(n);
        CUDA_OK(cudaMemcpy(v.data(), words, std::size_t{n} * 4, cudaMemcpyDeviceToHost));
        return v;
    }
};

}  // namespace

TEST_CASE("k_fill sutampa su CPU word() ir fold64", "[gpu]") {
    const std::uint32_t n = GENERATE(64u, 1024u, 1'000'003u, 1'382'400u);
    const std::uint64_t seed = 0x0123456789ABCDEFull, index = 88123456789ull;
    Buf b(n);
    b.write(seed, index);
    CUDA_OK(cudaStreamSynchronize(b.s));
    const auto v = b.download();
    std::uint32_t bad = 0;
    for (std::uint32_t i = kHeaderWords; i < n; ++i) bad += v[i] != word(seed, index, i);
    CHECK(bad == 0);
    const auto h = header_from_bytes(v.data());
    CHECK(validate(h, seed, n) == HdrError::Ok);
    CHECK(h.payload_fold64 == fold64_reference(seed, index, n));
    CHECK(h.grain_index == index);
}

TEST_CASE("k_verify: švarus grain'as", "[gpu]") {
    Buf b(1'382'400);
    b.write(5, 77);
    const auto r = b.check(5, 77);
    CHECK(r.mismatch_words == 0);
    CHECK(r.first_bad_word == 0xFFFFFFFFu);
    const auto h = header_from_bytes(r.header);
    CHECK(r.actual_fold64 == h.payload_fold64);
    CHECK(validate(h, 5, 1'382'400) == HdrError::Ok);
}

TEST_CASE("k_verify aptinka vieno bito klaidą bet kurioje vietoje", "[gpu]") {
    const std::uint32_t n = 1'382'400;
    const std::uint32_t pos = GENERATE(16u, 700'001u, 1'382'399u);
    Buf b(n);
    b.write(5, 77);
    CUDA_OK(gpu::corrupt(b.words, pos, 1u << 3, b.s));
    const auto r = b.check(5, 77);
    CHECK(r.mismatch_words == 1);
    CHECK(r.first_bad_word == pos);
    CHECK(r.actual_fold64 != header_from_bytes(r.header).payload_fold64);
}

TEST_CASE("k_verify suskaičiuoja 1000 sugadintų žodžių", "[gpu]") {
    Buf b(1'382'400);
    b.write(5, 77);
    for (std::uint32_t k = 0; k < 1000; ++k) CUDA_OK(gpu::corrupt(b.words, 16 + k * 997, 0x1u, b.s));
    const auto r = b.check(5, 77);
    CHECK(r.mismatch_words == 1000);
    CHECK(r.first_bad_word == 16);
}

TEST_CASE("k_verify: kitas indeksas ar seed → beveik visi žodžiai skiriasi", "[gpu]") {
    const std::uint32_t n = 1'382'400;
    Buf b(n);
    b.write(5, 77);
    CHECK(b.check(5, 76).mismatch_words > (n - 16) * 99ull / 100);
    CHECK(b.check(6, 77).mismatch_words > (n - 16) * 99ull / 100);
}

TEST_CASE("stream izoliacija: 8 gijos × 200 iteracijų", "[gpu]") {
    std::vector<std::thread> ts;
    std::vector<int> errors(8, 0);
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([t, &errors] {
            try {
                Buf b(1'382'400);
                for (int it = 0; it < 200; ++it) {
                    b.write(1000 + t, it);
                    const auto r = b.check(1000 + t, it);
                    errors[t] += r.mismatch_words != 0 || r.actual_fold64 != header_from_bytes(r.header).payload_fold64;
                }
            } catch (const std::exception&) {
                errors[t] += 1000;
            }
        });
    for (auto& t : ts) t.join();
    for (int t = 0; t < 8; ++t) CHECK(errors[t] == 0);
}
