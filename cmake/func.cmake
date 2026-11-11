# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# vramflow-func: GPU media funkcijos ant MXL (gen, logo, verify, enc, dec, clip, ref-crc)
add_executable(vramflow-func
  src/func/main_func.cpp src/func/func_common.cpp src/func/flows.cpp
  src/func/gen.cpp src/func/logo.cpp src/func/verify.cpp src/func/enc.cpp src/func/dec.cpp
  src/mxlio/common_io.cpp)
target_link_libraries(vramflow-func PRIVATE mxlv_media mxlv_media_cpu mxlv_common mxl::mxl CLI11::CLI11 Threads::Threads PkgConfig::LIBAV CUDA::cudart_static)
target_include_directories(vramflow-func PRIVATE ${CMAKE_SOURCE_DIR}/src ${MXL_SOURCE_DIR}/lib/payload-backends/cuda-linear)
set_target_properties(vramflow-func PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
