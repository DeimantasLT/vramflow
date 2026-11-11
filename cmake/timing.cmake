# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Laiko modelis (planas docs/planai/2026-11-11-production-grade.md §P1): PTS → TAI → MXL indeksas.
# Biblioteka be MXL, be libav ir be CUDA; testai naudoja MXL tik index_of / time_of sutapimui su mxlTimestampToIndex.
# Įtraukiama iš šakninio CMakeLists.txt po Catch2 (testų bloko).
add_library(mxlv_timing STATIC src/timing/source_clock.cpp)
vramflow_warnings(mxlv_timing)
target_include_directories(mxlv_timing PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(mxlv_timing PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MXLV_BUILD_TESTS)
  add_executable(mxlv-timing-unit tests/unit/test_timing_index.cpp tests/unit/test_source_clock.cpp
                 tests/unit/test_audio_timeline.cpp)
  vramflow_warnings(mxlv-timing-unit)
  target_link_libraries(mxlv-timing-unit PRIVATE mxlv_timing mxl::mxl Catch2::Catch2WithMain)
  set_target_properties(mxlv-timing-unit PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
  catch_discover_tests(mxlv-timing-unit)
endif()
