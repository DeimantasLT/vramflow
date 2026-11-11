# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Bendras tick variklis (planas docs/planai/2026-11-12-tick-variklis.md §5): laikrodis, ribos, įėjimų automatas,
# spragų būsena, ciklas, statistika. Biblioteka be MXL, be CUDA, be CLI11 — testuojama virtualiu laiku.
# Nepriklauso nuo MXLV_BUILD_FUNC. MXL linkuojamas tik į testų binarą (due_index prieš mxlTimestampToIndex).
add_library(mxlv_engine STATIC src/engine/clock.cpp src/engine/cuts.cpp src/engine/gap_state.cpp
            src/engine/stats.cpp src/engine/engine.cpp)
vramflow_warnings(mxlv_engine)
target_include_directories(mxlv_engine PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mxlv_engine PUBLIC mxlv_timing Threads::Threads)
set_target_properties(mxlv_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MXLV_BUILD_TESTS)
  add_executable(mxlv-engine-unit tests/engine/test_cuts.cpp tests/engine/test_input_state.cpp
                 tests/engine/test_gap_state.cpp tests/engine/test_engine.cpp
                 tests/engine/test_engine_timing.cpp)
  vramflow_warnings(mxlv-engine-unit)
  target_link_libraries(mxlv-engine-unit PRIVATE mxlv_engine mxl::mxl Catch2::Catch2WithMain)
  set_target_properties(mxlv-engine-unit PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
  catch_discover_tests(mxlv-engine-unit)
endif()
