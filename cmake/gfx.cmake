# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Alfa grafika ir kompozitorius: bendras rastro aprašas (src/gfx/raster.hpp) ir jo testai.
# Vėlesnės užduotys čia pridės libvramflow-gfx bibliotekos taikinį.
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
  set_target_properties(mxlv_gfx_ref PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(MXLV_BUILD_TESTS)
  set(GFX_UNIT_SRC tests/unit/test_gfx_raster.cpp)
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
