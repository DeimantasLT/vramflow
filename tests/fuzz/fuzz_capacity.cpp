// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Fuzz taikinys: talpos failo skaitymas (mxlv::runtime::parse_capacity, src/runtime/admission.cpp — ranka rašytas JSON
// skaitytuvas). Failą teikia operatorius, bet jis keliauja tarp mašinų (planas 2026-11-17 §1.5, kriterijus S2).
// Invariantai: priimtos talpos skaičiai baigtiniai ir neneigiami; atmetimas visada turi priežastį.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "runtime/admission.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string json(reinterpret_cast<const char*>(data), size);
    mxlv::runtime::Capacity c;
    std::string err;
    if (!mxlv::runtime::parse_capacity(json, c, err)) {
        if (err.empty()) std::abort();
        return 0;
    }
    if (!std::isfinite(c.nvdec_mpix_s) || c.nvdec_mpix_s < 0) std::abort();
    if (!std::isfinite(c.nvenc_mpix_s) || c.nvenc_mpix_s < 0) std::abort();
    if (c.nvenc_sessions < 0) std::abort();
    for (const auto& kv : c.functions)
        if (kv.second.knee < 0) std::abort();
    (void)mxlv::runtime::capacity_warnings(c, "596.49");
    return 0;
}
