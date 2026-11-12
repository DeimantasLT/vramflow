// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Gyvos stebėsenos JSON eilutės (B1; planas §P7 testai, V21): viena eilutė per intervalą, skirtumai (ne sumos), vieneto /
// įėjimo / išėjimo būsenos, įvykio eilutė, exit kodo politika (D8) ir variklio pabaigos priežasties → rc atvaizdis.
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "engine/stats.hpp"
#include "func/engine_glue.hpp"
#include "func/func_common.hpp"
#include "func/stats_lines.hpp"

using namespace mxlv;

namespace {
std::vector<std::string> read_lines(const std::string& p) {
    std::vector<std::string> v;
    std::ifstream f(p);
    for (std::string l; std::getline(f, l);)
        if (!l.empty()) v.push_back(l);
    return v;
}
std::string tmp_path(const char* n) {
    return (std::filesystem::temp_directory_path() / (std::string("vramflow-") + n + "-" + std::to_string(::getpid()) + ".jsonl")).string();
}
}  // namespace

TEST_CASE("stats_lines: viena eilutė per intervalą, skirtumai, būsenos, įvykis", "[func][stats_lines]") {
    const std::string path = tmp_path("lines");
    std::filesystem::remove(path);
    func::Opts o;
    o.stats_file = path;
    o.run_id = "testas-1";
    {
        func::StatsLines sink(o, "logo");
        engine::StatsHub hub(1.0, &sink);
        const std::uint32_t u = hub.add_unit("flow-a");
        const std::uint32_t in = hub.add_input("flow-a/in0");
        const std::uint32_t out = hub.add_output("flow-a/out0");
        const std::uint32_t u2 = hub.add_unit("flow-b");
        hub.add_input("flow-b/in0");
        hub.add_output("flow-b/out0");
        hub.prepare();
        const std::uint64_t t0 = 1'000'000'000'000'000'000ull;
        hub.end_tick(t0);
        // kiekviena eilutė išleidžiama prieš kitą intervalą (kitaip — dropped_lines, skirtumai pereina)
        auto wait_lines = [&](std::uint64_t n) {
            for (int t = 0; t < 400 && sink.lines() < n; ++t) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            REQUIRE(sink.lines() == n);
        };
        for (int k = 1; k <= 3; ++k) {
            hub.unit(u).ok += 50;  // kaupiamieji 50, 100, 150 → eilutėse 50, 50, 50
            hub.in(in).ok += 50;
            hub.out(out).ok += 50;
            hub.out_commit(out).add(2'000'000);
            hub.set_in_state(in, engine::InState::Live);
            if (k == 2) hub.set_unit_failed(u2, true);
            hub.end_tick(t0 + static_cast<std::uint64_t>(k) * 1'000'000'000ull);
            wait_lines(static_cast<std::uint64_t>(k));
        }
        hub.event(engine::EventKind::UnitFailed, u2, t0 + 2'500'000'000ull, 77, 0, "išėjimas flow-b/out0: 3 open klaidos iš eilės");
        hub.unit(u).ok += 7;  // nepilnas paskutinis intervalas
        hub.flush(t0 + 3'400'000'000ull);
        hub.stop();
        CHECK(sink.lines() == 4);
    }
    const std::vector<std::string> lines = read_lines(path);
    std::vector<std::string> stats, events;
    for (const auto& l : lines) (l.find("\"event\":") != std::string::npos ? events : stats).push_back(l);
    REQUIRE(stats.size() == 4);
    REQUIRE(events.size() == 1);
    for (std::size_t k = 0; k < 3; ++k) {
        INFO(stats[k]);
        CHECK(stats[k].rfind("{\"v\":1,", 0) == 0);
        CHECK(stats[k].find("\"role\":\"logo\"") != std::string::npos);
        CHECK(stats[k].find("\"run\":\"testas-1\"") != std::string::npos);
        CHECK(stats[k].find("\"id\":\"flow-a\",\"state\":\"Healthy\",\"failed\":false,\"ok\":50,") != std::string::npos);
        CHECK(stats[k].find("\"id\":\"flow-a/in0\",\"state\":\"Live\",\"health\":\"Healthy\",\"ok\":50") != std::string::npos);
        CHECK(stats[k].find("\"id\":\"flow-a/out0\",\"health\":\"Healthy\",\"ok\":50") != std::string::npos);
        CHECK(stats[k].find("\"lat_max_ms\":2.000") != std::string::npos);
        CHECK(stats[k].find("\"gpu\":") != std::string::npos);
        CHECK(stats[k].back() == '}');
    }
    CHECK(stats[0].find("\"id\":\"flow-b\",\"state\":\"Inactive\",\"failed\":false") != std::string::npos);
    CHECK(stats[1].find("\"id\":\"flow-b\",\"state\":\"Unhealthy\",\"failed\":true") != std::string::npos);
    CHECK(stats[1].find("\"state\":\"PartiallyHealthy\"") != std::string::npos);  // procesas: vienas flow'as sugedęs
    CHECK(stats[3].find("\"ok\":7,") != std::string::npos);  // flush — paskutinė nepilna eilutė
    CHECK(events[0].find("\"event\":\"unit_failed\"") != std::string::npos);
    CHECK(events[0].find("\"id\":\"flow-b\"") != std::string::npos);
    CHECK(events[0].find("\"index\":77") != std::string::npos);
    CHECK(events[0].find("3 open klaidos") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("stats_lines: be --stats-file eilučių nėra (įvykiai — tik stderr)", "[func][stats_lines]") {
    func::Opts o;
    func::StatsLines sink(o, "enc");
    engine::StatsHub hub(1.0, &sink);
    hub.add_unit("u");
    hub.prepare();
    hub.end_tick(1'000'000'000ull);
    hub.end_tick(2'000'000'000ull);
    hub.stop();
    CHECK(sink.lines() == 0);
}

TEST_CASE("exit_code_for: sugedęs vienetas → 2 abiem režimais; praradimas → 4 tik griežtai", "[func][exit]") {
    func::Opts o;
    func::Stats st;
    o.duration_s = 0;  // tarnyba (auto → švelnus)
    CHECK(func::exit_code_for(o, 0, true, 0, st) == 0);
    CHECK(func::exit_code_for(o, 0, false, 1, st) == 2);  // anksčiau — 0
    o.duration_s = 10;  // griežtas
    CHECK(func::exit_code_for(o, 0, true, 0, st) == 4);
    CHECK(func::exit_code_for(o, 0, true, 1, st) == 2);
    CHECK(func::exit_code_for(o, 6, true, 1, st) == 6);
}
