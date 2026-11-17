// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "runtime/fail_text.hpp"

namespace mxlv::runtime {
namespace {
constexpr std::uint64_t kMiB = 1ull << 20;
std::string mib_down(std::uint64_t bytes) { return std::to_string(bytes / kMiB); }
std::string mib_up(std::uint64_t bytes) { return std::to_string((bytes + kMiB - 1) / kMiB); }

// libav žurnalo eilutės pranešimas: be „[koderis @ 0x…] “ priešdėlio (rodyklė kiekvieną kartą kita) ir be galinių tarpų.
std::string log_message(const std::string& line) {
    std::size_t b = 0, e = line.size();
    if (!line.empty() && line.front() == '[')
        if (const std::size_t p = line.find("] "); p != std::string::npos) b = p + 2;
    while (e > b && (line[e - 1] == '\n' || line[e - 1] == '\r' || line[e - 1] == ' ' || line[e - 1] == '\t')) --e;
    return line.substr(b, e - b);
}
}  // namespace

std::string storage_fail_text(const StorageFacts& f) {
    if (!f.writable) return "; domeno katalogas nerašomas (EACCES, permission denied)";
    if (f.total == 0) return {};
    std::string s = f.cuda ? "; saugykla cuda-linear: VRAM laisva " : "; saugykla host: domene laisva ";
    s += mib_down(f.free) + " MiB iš " + mib_down(f.total) + " MiB";
    if (f.need == 0) return s;
    s += ", žiedui reikia " + mib_up(f.need) + " MiB";
    if (f.free < f.need) s += f.cuda ? " — VRAM neužtenka" : " — domene nėra vietos (ENOSPC, no space left on device)";
    return s;
}

std::string cuda_alloc_fail_text(const std::string& what, const std::string& cuda_error, std::uint64_t free, std::uint64_t total) {
    std::string s = what + ": CUDA " + cuda_error;
    if (total != 0) s += "; VRAM laisva " + mib_down(free) + " MiB iš " + mib_down(total) + " MiB";
    return s;
}

std::string nvenc_open_fail_text(const std::string& av_error, const std::string& log_line) {
    std::string s = "video avcodec_open2: " + av_error;
    const std::string msg = log_message(log_line);
    if (msg.empty()) return s;
    s += " (" + msg + ")";
    if (msg.find("OpenEncodeSessionEx failed") == std::string::npos) return s;
    s += " — NVENC sesijos negauta";
    if (msg.find("(21)") != std::string::npos)
        s += ": tvarkyklės kodas 21 reiškia, kad viršyta vienalaikių NVENC sesijų riba (šios sistemos licencijos tipas daugiau neleidžia)";
    else if (msg.find("(10)") != std::string::npos)
        s += ": tvarkyklei neužteko atminties sesijai (kodas 10)";
    return s;
}

}  // namespace mxlv::runtime
