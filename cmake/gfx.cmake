# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Alfa grafika ir kompozitorius: bendras rastro aprašas (src/gfx/raster.hpp) ir jo testai.
# Čia pat: CPU (variantas C) ir CUDA kompozitoriaus branduoliai, libvramflow-gfx (C ABI, vfgfx_*) ir jų testai.
# Nepriklausomas bitų tikslumo etalonas (src/gfx/gfx_ref.*): RGBA -> v210a ir composite. RGB->YCbCr ima iš mxlv_media
# (ref::rgb_to_ycbcr10), todėl reikia MXLV_BUILD_MEDIA.
# Pastovus gijų baseinas (iš func_common išskirtas, R3): be CUDA / MXL, todėl jį gali naudoti ir gfx::cpu, ir vramflow-func.
add_library(mxlv_threadpool STATIC src/func/thread_pool.cpp)
target_include_directories(mxlv_threadpool PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mxlv_threadpool PUBLIC Threads::Threads)
set_target_properties(mxlv_threadpool PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Variantas C: CPU RGBA -> v210a ir composite (be CUDA). -ffp-contract=off: RGB->YCbCr double aritmetika privalo būti
# bitas į bitą kaip etalone (be FMA kontrakcijos, kurią įjungia -march=x86-64-v3).
add_library(mxlv_gfx_cpu STATIC src/gfx/cpu/gfx_cpu.cpp)
target_include_directories(mxlv_gfx_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mxlv_gfx_cpu PUBLIC mxlv_threadpool)
target_compile_options(mxlv_gfx_cpu PRIVATE -O3 -march=x86-64-v3 -ffp-contract=off)
set_target_properties(mxlv_gfx_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MXLV_BUILD_MEDIA)
  add_library(mxlv_gfx_ref STATIC src/gfx/gfx_ref.cpp)
  target_include_directories(mxlv_gfx_ref PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(mxlv_gfx_ref PUBLIC mxlv_common mxlv_media)
  target_compile_options(mxlv_gfx_ref PRIVATE -ffp-contract=off)  # etalonas niekada nenaudoja FMA (clang / -march=native)
  set_target_properties(mxlv_gfx_ref PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(MXLV_BUILD_TESTS)
  set(GFX_UNIT_SRC tests/unit/test_gfx_raster.cpp tests/unit/test_gfx_content.cpp)
  if(MXLV_BUILD_MEDIA)
    list(APPEND GFX_UNIT_SRC tests/unit/test_gfx_ref.cpp tests/unit/test_gfx_cpu.cpp)
  endif()
  add_executable(mxlv-gfx-unit ${GFX_UNIT_SRC})
  target_link_libraries(mxlv-gfx-unit PRIVATE mxlv_common Catch2::Catch2WithMain)
  if(MXLV_BUILD_MEDIA)
    target_link_libraries(mxlv-gfx-unit PRIVATE mxlv_gfx_ref mxlv_gfx_cpu)
  endif()
  target_compile_definitions(mxlv-gfx-unit PRIVATE MXLV_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/unit/data")
  catch_discover_tests(mxlv-gfx-unit)
endif()

# Variantai H / G: CUDA RGBA -> v210a ir batch kompozitorius. RGB->YCbCr double aritmetika naudoja __dmul_rn / __dadd_rn
# intrinsics (nvcc jų nesujungia į FMA nepriklausomai nuo -fmad). Parametrai kaip mxlv_media.
if(MXLV_BUILD_MEDIA)
  add_library(mxlv_gfx_cuda STATIC src/gfx/gfx_kernels.cu)
  target_include_directories(mxlv_gfx_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(mxlv_gfx_cuda PUBLIC CUDA::cudart_static)
  set_target_properties(mxlv_gfx_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(mxlv_gfx_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

  if(MXLV_BUILD_TESTS)
    add_executable(vramflow-gfx-gpu-tests tests/gpu/test_gfx_kernels.cpp)
    target_link_libraries(vramflow-gfx-gpu-tests PRIVATE mxlv_gfx_cuda mxlv_gfx_ref Catch2::Catch2WithMain)
    catch_discover_tests(vramflow-gfx-gpu-tests)
  endif()
endif()

# libvramflow-gfx: viešas C ABI (include/vramflow/gfx.h) — RGBA -> standartinis MXL v210a flow, rankinis ir ritmo režimai.
# Reikia MXL srauto pagalbininkų (mxlv_flows, R4: statinė PIC biblioteka iš flows.cpp) ir abiejų konvertavimo variantų.
# Tas pats gfx_writer.cpp kompiliuojamas du kartus: į .so ir į statinę biblioteką testams (testai naudoja vidinius kabliukus; .so eksportuoja tik vfgfx_*, žr. vramflow-gfx.map).
if(TARGET mxlv_flows AND TARGET mxlv_gfx_cuda)
  include(GNUInstallDirs)
  add_library(mxlv_gfx_writer STATIC src/gfx/gfx_writer.cpp)
  target_include_directories(mxlv_gfx_writer PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
                             ${MXL_SOURCE_DIR}/lib/payload-backends/cuda-linear)
  target_link_libraries(mxlv_gfx_writer PUBLIC mxl::mxl mxlv_flows mxlv_gfx_cpu mxlv_gfx_cuda CUDA::cudart_static Threads::Threads)
  set_target_properties(mxlv_gfx_writer PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(vramflow-gfx SHARED src/gfx/gfx_writer.cpp)
  target_include_directories(vramflow-gfx PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
                             ${MXL_SOURCE_DIR}/lib/payload-backends/cuda-linear)
  # mxl::mxl PUBLIC: antraštė gfx.h įtraukia <mxl/mxl.h>, vartotojas gauna jo include kelią.
  target_link_libraries(vramflow-gfx PUBLIC mxl::mxl PRIVATE mxlv_flows mxlv_gfx_cpu mxlv_gfx_cuda CUDA::cudart_static Threads::Threads)
  target_include_directories(vramflow-gfx INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_options(vramflow-gfx PRIVATE -Wl,--version-script=${CMAKE_SOURCE_DIR}/cmake/vramflow-gfx.map)
  set_target_properties(vramflow-gfx PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib" VERSION 0.1.0 SOVERSION 0)
  # Pastaba: INSTALL_RPATH nenustatytas — įdiegtą .so vartotojas turi rasti libmxl per LD_LIBRARY_PATH / ldconfig.
  install(TARGETS vramflow-gfx LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/vramflow/gfx.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vramflow)

  if(MXLV_BUILD_TESTS)
    # Antraštė turi kompiliuotis kaip C (-std=c11 -pedantic -Werror) ir ją naudojanti programa — susilinkuoti su .so.
    enable_language(C)
    add_executable(vramflow-gfx-c-check tests/gpu/gfx_c_check.c)
    target_link_libraries(vramflow-gfx-c-check PRIVATE vramflow-gfx mxl::mxl)
    set_target_properties(vramflow-gfx-c-check PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
                          BUILD_RPATH "${MXL_ROOT}/lib")
    target_compile_options(vramflow-gfx-c-check PRIVATE -Wall -Wextra -pedantic -Werror)
    add_test(NAME gfx_writer_c_header COMMAND vramflow-gfx-c-check)

    add_executable(vramflow-gfx-writer-tests tests/gpu/test_gfx_writer.cpp)
    target_link_libraries(vramflow-gfx-writer-tests PRIVATE mxlv_gfx_writer mxlv_gfx_ref Catch2::Catch2WithMain)
    set_target_properties(vramflow-gfx-writer-tests PROPERTIES BUILD_RPATH "${MXL_ROOT}/lib")
    catch_discover_tests(vramflow-gfx-writer-tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
      add_test(NAME gfx_writer_ctypes_smoke
               COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/gfx_ctypes_smoke.py $<TARGET_FILE:vramflow-gfx>
                       ${MXL_ROOT}/lib/libmxl.so)
    endif()
  endif()
endif()
