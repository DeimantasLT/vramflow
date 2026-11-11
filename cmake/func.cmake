# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# vramflow-func: GPU media funkcijos ant MXL (gen, logo, verify, enc, dec, clip, ref-crc)
# MXL flow pagalbininkai (VideoIn / VideoOut / wait_until ...): atskira biblioteka, kad juos galėtų tikrinti testai (R4).
add_library(mxlv_flows STATIC src/func/flows.cpp)
target_include_directories(mxlv_flows PUBLIC ${CMAKE_SOURCE_DIR}/src ${MXL_SOURCE_DIR}/lib/payload-backends/cuda-linear)
target_link_libraries(mxlv_flows PUBLIC mxl::mxl mxlv_common)
set_target_properties(mxlv_flows PROPERTIES POSITION_INDEPENDENT_CODE ON)

# gfxgen naudoja TIK viešą libvramflow-gfx (include/vramflow/gfx.h): įrodo, kad biblioteka pakankama.
add_executable(vramflow-func
  src/func/main_func.cpp src/func/func_common.cpp
  src/func/gen.cpp src/func/gfxgen.cpp src/func/logo.cpp src/func/composite.cpp src/func/verify.cpp src/func/enc.cpp src/func/dec.cpp
  src/mxlio/common_io.cpp)
target_link_libraries(vramflow-func PRIVATE mxlv_flows mxlv_threadpool mxlv_media mxlv_media_cpu mxlv_common mxl::mxl vramflow-gfx mxlv_gfx_cpu mxlv_gfx_cuda mxlv_gfx_ref CLI11::CLI11 Threads::Threads PkgConfig::LIBAV CUDA::cudart_static)
target_include_directories(vramflow-func PRIVATE ${CMAKE_SOURCE_DIR}/src ${MXL_SOURCE_DIR}/lib/payload-backends/cuda-linear)
set_target_properties(vramflow-func PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")

if(MXLV_BUILD_TESTS)
  # flows.cpp testai ant tikro MXL (host ir cuda-linear): layout, grainRate. Host dalis nereikalauja GPU.
  add_executable(vramflow-flows-tests tests/gpu/test_flows.cpp)
  target_link_libraries(vramflow-flows-tests PRIVATE mxlv_flows CUDA::cudart_static Catch2::Catch2WithMain)
  set_target_properties(vramflow-flows-tests PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
  catch_discover_tests(vramflow-flows-tests)
endif()
