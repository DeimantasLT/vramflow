// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <cstdint>

#include "mxlio/common_io.hpp"

namespace mxlv {

// Vieno flow'o writer'io/reader'io ciklas (viena gija vienam handle'ui). Grąžina išėjimo kodą (Exit).
int run_writer_flow(const Options& o, std::uint32_t n, FlowState& s, int cuda_dev, const std::atomic<bool>& stop,
                    std::uint64_t end_tai);
int run_reader_flow(const Options& o, std::uint32_t n, FlowState& s, int cuda_dev, const std::atomic<bool>& stop,
                    std::uint64_t end_tai);

}  // namespace mxlv
