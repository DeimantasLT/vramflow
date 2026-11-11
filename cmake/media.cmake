# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# GPU media funkcijų biblioteka (architektūra docs/architektura/2026-11-10-gpu-media-funkcijos.md §6):
# kernel'iai (logotipas, weave/split NV12, generatorius) ir jų nepriklausomi CPU etalonai + PNG įkėlimas per libav.
# Įtraukiama iš šakninio CMakeLists.txt po pkg_check_modules(LIBAV ...).
add_library(mxlv_media STATIC src/media/media_kernels.cu src/media/media_ref.cpp)
target_include_directories(mxlv_media PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mxlv_media PUBLIC mxlv_common CUDA::cudart_static PRIVATE PkgConfig::LIBAV)
set_target_properties(mxlv_media PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(mxlv_media PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)  # media_ref.cpp etalonas be FMA kontrakcijos

# Variantas C: CPU media branduoliai be CUDA (architektūra 2026-11-11-gpu-vs-cpu-ir-upstream.md §1.3).
# -march=x86-64-v3 (AVX2/FMA/BMI2): kompiliatoriaus vektorizacija; vėliavėlės skelbiamos ataskaitoje.
add_library(mxlv_media_cpu STATIC src/media/cpu/cpu_media.cpp)
target_include_directories(mxlv_media_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(mxlv_media_cpu PUBLIC mxlv_common)
target_compile_options(mxlv_media_cpu PRIVATE -O3 -march=x86-64-v3)
set_target_properties(mxlv_media_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MXLV_BUILD_TESTS)
  add_executable(mxlv-media-cpu-unit tests/unit/test_media_cpu.cpp)
  target_link_libraries(mxlv-media-cpu-unit PRIVATE mxlv_media_cpu mxlv_media Catch2::Catch2WithMain)
  catch_discover_tests(mxlv-media-cpu-unit)

  add_executable(mxlv-media-unit tests/unit/test_media.cpp tests/unit/test_media_format.cpp)
  target_link_libraries(mxlv-media-unit PRIVATE mxlv_media Catch2::Catch2WithMain)
  catch_discover_tests(mxlv-media-unit)

  add_executable(mxlv-media-gpu-tests tests/gpu/test_media_kernels.cpp)
  target_link_libraries(mxlv-media-gpu-tests PRIVATE mxlv_media Catch2::Catch2WithMain)
  catch_discover_tests(mxlv-media-gpu-tests)
endif()
