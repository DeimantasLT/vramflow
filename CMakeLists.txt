# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(vramflow-verify LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# RTX 4070 Laptop = sm_89. Keičiama: -DCMAKE_CUDA_ARCHITECTURES=native|86|89|...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(MXL_ROOT "$ENV{HOME}/src/mxl-pr653/install/Linux-Clang-Release" CACHE PATH "MXL PR #653 instaliacija")
set(MXL_SOURCE_DIR "$ENV{HOME}/src/mxl-pr653" CACHE PATH "MXL šaltinis (CudaLinearStorage.h)")
# Catch2 ir CLI11 imami iš MXL vcpkg (ta pati baseline, nieko papildomai nesiunčiam)
set(MXL_VCPKG_PREFIX "${MXL_SOURCE_DIR}/build/Linux-Clang-Release/vcpkg_installed/x64-linux" CACHE PATH "")
option(MXLV_ENABLE_SANITIZERS "ASan + UBSan host kodui" OFF)
option(MXLV_BUILD_TESTS "Unit ir GPU testai" ON)

list(PREPEND CMAKE_PREFIX_PATH "${MXL_ROOT}" "${MXL_VCPKG_PREFIX}")
find_package(CUDAToolkit REQUIRED)
find_package(mxl CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

if(MXLV_ENABLE_SANITIZERS)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(mxlv_common INTERFACE)
target_include_directories(mxlv_common INTERFACE ${CMAKE_SOURCE_DIR}/src)

add_library(mxlv_gpu STATIC src/gpu/kernels.cu)
target_link_libraries(mxlv_gpu PUBLIC mxlv_common CUDA::cudart_static)
set_target_properties(mxlv_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(mxlv_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

find_package(Threads REQUIRED)
option(MXLV_BUILD_TOOL "vramflow-verify įrankis" ON)
if(MXLV_BUILD_TOOL)
  add_executable(vramflow-verify src/main.cpp src/mxlio/verify_writer.cpp src/mxlio/verify_reader.cpp src/mxlio/common_io.cpp)
  target_link_libraries(vramflow-verify PRIVATE mxlv_gpu mxl::mxl CLI11::CLI11 Threads::Threads)
  target_include_directories(vramflow-verify PRIVATE ${MXL_SOURCE_DIR}/lib/payload-backends/cuda-linear)
  set_target_properties(vramflow-verify PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
endif()

if(MXLV_BUILD_TESTS)
  find_package(Catch2 3 CONFIG REQUIRED)
  enable_testing()
  include(Catch)
  add_executable(mxlv-unit tests/unit/test_common.cpp)
  target_link_libraries(mxlv-unit PRIVATE mxlv_common Catch2::Catch2WithMain)
  catch_discover_tests(mxlv-unit)

  add_executable(mxlv-gpu-tests tests/gpu/test_kernels.cpp)
  target_link_libraries(mxlv-gpu-tests PRIVATE mxlv_gpu Catch2::Catch2WithMain)
  catch_discover_tests(mxlv-gpu-tests)

  # Atitikties testas: v210a (fill + key) ant MXL PR #653 cuda-linear (alfa kompozitoriaus vartai)
  add_executable(vramflow-v210a-gpu-tests tests/gpu/test_v210a_cuda_linear.cpp)
  target_link_libraries(vramflow-v210a-gpu-tests PRIVATE mxl::mxl CUDA::cudart_static Catch2::Catch2WithMain)
  target_include_directories(vramflow-v210a-gpu-tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${MXL_SOURCE_DIR}/lib/payload-backends/cuda-linear)
  set_target_properties(vramflow-v210a-gpu-tests PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
  catch_discover_tests(vramflow-v210a-gpu-tests)
endif()

# GPU media funkcijos (logotipas, kodavimas, dekodavimas) — architektūra docs/architektura/2026-11-10-gpu-media-funkcijos.md
option(MXLV_BUILD_MEDIA "GPU media funkcijos (vramflow-func)" ON)
if(MXLV_BUILD_MEDIA)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
  include(${CMAKE_SOURCE_DIR}/cmake/media.cmake)       # mxlv_media biblioteka + jos testai
  option(MXLV_BUILD_FUNC "vramflow-func vykdomasis" OFF)
  if(MXLV_BUILD_FUNC)
    include(${CMAKE_SOURCE_DIR}/cmake/func.cmake)       # vramflow-func vykdomasis
  endif()
endif()
