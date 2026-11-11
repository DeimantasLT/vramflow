// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "engine/stats.hpp"

namespace mxlv::engine {

CounterId StatsHub::register_counter(std::string name) {
    counter_names_.push_back(std::move(name));
    return static_cast<CounterId>(counter_names_.size() - 1);
}

HistId StatsHub::register_hist(std::string name) {
    hist_names_.push_back(std::move(name));
    return static_cast<HistId>(hist_names_.size() - 1);
}

std::uint32_t StatsHub::add_unit(std::string name) {
    unit_names_.push_back(std::move(name));
    return static_cast<std::uint32_t>(unit_names_.size() - 1);
}

std::uint32_t StatsHub::add_input(std::string name) {
    in_names_.push_back(std::move(name));
    in_.emplace_back();
    return static_cast<std::uint32_t>(in_.size() - 1);
}

std::uint32_t StatsHub::add_output(std::string name) {
    out_names_.push_back(std::move(name));
    out_.emplace_back();
    return static_cast<std::uint32_t>(out_.size() - 1);
}

void StatsHub::prepare() { fn_.assign(unit_names_.size() * counter_names_.size(), 0); }

}  // namespace mxlv::engine
