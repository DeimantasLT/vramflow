// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Korpuso failų paleidimas be libFuzzer (planas docs/planai/2026-11-17-production-ready.md §1.5): tas pats
// LLVMFuzzerTestOneInput, kurį fuzzina mxlv-fuzz-<taikinys>, čia paleidžiamas su kiekvienu tests/fuzz/corpus/<taikinys>/
// failu — ctest fuzz_replay_<taikinys> gamybiniame build'e. Fuzzerio radinys → failas į korpusą → lieka testu visam laikui.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "naudojimas: %s <korpuso katalogas>\n", argv[0]);
        return 2;
    }
    std::size_t n = 0;
    for (const auto& e : std::filesystem::directory_iterator(argv[1])) {
        if (!e.is_regular_file()) continue;
        std::ifstream f(e.path(), std::ios::binary);
        const std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(b.data()), b.size());
        ++n;
    }
    std::printf("%zu korpuso failų, be avarijų\n", n);
    return n > 0 ? 0 : 1;  // tuščias korpusas — klaida: testas nieko netikrintų
}
