# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Tvirtas paleidimas ir sustabdymas (planas §P6): InputSlot (įėjimas atsiranda / dingsta / grįžta), OpenPolicy (laukimas),
# DomainLease (paleidimo nuoma `<domenas>.vramflow/`). Naudoja mxlv_flows (VideoIn).
add_library(mxlv_runtime STATIC src/runtime/input_slot.cpp src/runtime/domain_lease.cpp)
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
endif()
