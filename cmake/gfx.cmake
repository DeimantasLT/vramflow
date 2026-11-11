# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Alfa grafika ir kompozitorius: bendras rastro aprašas (src/gfx/raster.hpp) ir jo testai.
# Vėlesnės užduotys čia pridės libvramflow-gfx bibliotekos taikinį.
if(MXLV_BUILD_TESTS)
  add_executable(mxlv-gfx-unit tests/unit/test_gfx_raster.cpp)
  target_link_libraries(mxlv-gfx-unit PRIVATE mxlv_common Catch2::Catch2WithMain)
  target_compile_definitions(mxlv-gfx-unit PRIVATE MXLV_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/unit/data")
  catch_discover_tests(mxlv-gfx-unit)
endif()
