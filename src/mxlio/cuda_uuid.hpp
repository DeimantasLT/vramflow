// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Vienintelė CUDA UUID pagalbinių funkcijų realizacija (vramflow-func, vramflow-verify ir libvramflow-gfx).
// Kanoninė forma — "GPU-" + mažosios hex raidės "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"; paieškoje užklausos raidžių dydis
// normalizuojamas (priimamos ir didžiosios, ir "gpu-"). inline: kiekvienas vartotojas turi savo cudart (bibliotekos cudart privatus).
#include <cuda_runtime_api.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>

namespace mxlv {

inline std::string cuda_uuid_string(int ordinal) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        (void)cudaGetLastError();
        return {};
    }
    static const char* hex = "0123456789abcdef";
    std::string r = "GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) r.push_back('-');
        const auto v = static_cast<std::uint8_t>(p.uuid.bytes[i]);
        r.push_back(hex[v >> 4]);
        r.push_back(hex[v & 15]);
    }
    return r;
}

// Įrenginio eilės numeris pagal UUID (tuščias → 0); -1 jei nėra.
inline int cuda_device_by_uuid(std::string uuid) {
    // "GPU-" priešdėlis kanoninėje formoje didžiosiomis, hex — mažosiomis: normalizuojame tik hex dalį
    for (std::size_t i = 4; i < uuid.size(); ++i) uuid[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(uuid[i])));
    if (uuid.size() >= 4) {
        for (std::size_t i = 0; i < 4; ++i) uuid[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(uuid[i])));
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    for (int i = 0; i < count; ++i)
        if (uuid.empty() || cuda_uuid_string(i) == uuid) return i;
    return -1;
}

}  // namespace mxlv
