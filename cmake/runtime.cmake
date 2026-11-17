# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Tvirtas paleidimas ir sustabdymas (planas §P6): InputSlot (įėjimas atsiranda / dingsta / grįžta), OpenPolicy (laukimas),
# DomainLease (paleidimo nuoma `<domenas>.vramflow/`). Naudoja mxlv_flows (VideoIn).
# Priėmimo kontrolė (planas §P4): talpos failas, sprendimas, rezervacijų knyga nuomos kataloge.
add_library(mxlv_runtime STATIC src/runtime/input_slot.cpp src/runtime/domain_lease.cpp src/runtime/admission.cpp
            src/runtime/safe_files.cpp src/runtime/fail_text.cpp)
vramflow_warnings(mxlv_runtime)
target_include_directories(mxlv_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mxlv_runtime PUBLIC mxlv_flows mxl::mxl)
set_target_properties(mxlv_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MXLV_BUILD_TESTS)
  # Ant tikro MXL, host saugykla (GPU nereikia). Lėtas testas (20 s laukimas) — žymė [slow].
  add_executable(vramflow-runtime-tests tests/gpu/test_runtime.cpp)
  vramflow_warnings(vramflow-runtime-tests)
  target_link_libraries(vramflow-runtime-tests PRIVATE mxlv_runtime Catch2::Catch2WithMain)
  set_target_properties(vramflow-runtime-tests PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
  catch_discover_tests(vramflow-runtime-tests PROPERTIES TIMEOUT 120)
  # Priėmimo kontrolė (planas §P4) — gryna logika ir rezervacijų knyga (flock, fork / kill -9), be MXL ir GPU.
  # Saugus nuomos / knygos failų kūrimas (planas 2026-11-17 §1.2, S4): tests/unit/test_safe_files.cpp
  # Klaidų tekstai „fatal“ laukui (planas 2026-11-17 §2.5, K4 / S9): tests/unit/test_fail_text.cpp
  add_executable(mxlv-admission-unit tests/unit/test_admission.cpp tests/unit/test_safe_files.cpp tests/unit/test_fail_text.cpp
                 src/runtime/admission.cpp src/runtime/domain_lease.cpp src/runtime/safe_files.cpp src/runtime/fail_text.cpp)
  vramflow_warnings(mxlv-admission-unit)
  target_include_directories(mxlv-admission-unit PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(mxlv-admission-unit PRIVATE MXLV_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/unit/data")
  target_link_libraries(mxlv-admission-unit PRIVATE Catch2::Catch2WithMain)
  catch_discover_tests(mxlv-admission-unit)
endif()
