# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Variklio MXL prievadai (planas docs/planai/2026-11-12-tick-variklis.md §5, V7): SystemClock, MxlVideoInput (ant
# runtime::InputSlot), MxlVideoOutput (ant func::VideoOut). Vienintelė vieta, kur variklis kalba su MXL; mxlv_engine
# lieka be MXL. Įtraukiama iš func.cmake (reikia mxlv_flows / mxlv_runtime).
add_library(mxlv_engine_mxl STATIC src/engine/mxl_ports.cpp)
vramflow_warnings(mxlv_engine_mxl)
target_link_libraries(mxlv_engine_mxl PUBLIC mxlv_engine mxlv_runtime mxlv_flows mxl::mxl)
set_target_properties(mxlv_engine_mxl PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MXLV_BUILD_TESTS)
  # Tikras MXL, host saugykla (GPU nereikia). Laiko testai serializuojami tarpusavyje ir su kitais realtime testais.
  add_executable(vramflow-engine-mxl-tests tests/engine/test_engine_mxl.cpp)
  vramflow_warnings(vramflow-engine-mxl-tests)
  target_link_libraries(vramflow-engine-mxl-tests PRIVATE mxlv_engine_mxl Catch2::Catch2WithMain)
  set_target_properties(vramflow-engine-mxl-tests PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
  catch_discover_tests(vramflow-engine-mxl-tests PROPERTIES TIMEOUT 60 LABELS realtime RESOURCE_LOCK realtime)
endif()
