# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Ingest (planas docs/planai/2026-11-11-production-grade.md §P3): libavformat įėjimai (udp / rtp / srt / hls; K10), demuxeris,
# video (NVDEC / programinis) ir garso (libswresample) dekoderiai. Bendra su dec: pertraukiamas I/O ir sesijų ciklas.
# Vienintelė nauja build priklausomybė — libswresample (ta pati sisteminio FFmpeg versija; K10).
# Įtraukiama iš šakninio CMakeLists.txt po pkg_check_modules(LIBAV ...).
pkg_check_modules(SWR REQUIRED IMPORTED_TARGET libswresample)
add_library(mxlv_ingest STATIC src/ingest/av_input.cpp src/ingest/demux.cpp src/ingest/video_decode.cpp)
vramflow_warnings(mxlv_ingest)
target_include_directories(mxlv_ingest PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mxlv_ingest PUBLIC mxlv_timing mxlv_common PkgConfig::LIBAV PkgConfig::SWR mxl::mxl)

if(MXLV_BUILD_TESTS)
  # Gryna logika (be libav / MXL / GPU): tiekimo vartai, H.264 skenavimas.
  add_executable(mxlv-ingest-unit tests/unit/test_feed_gate.cpp tests/unit/test_h264_scan.cpp)
  vramflow_warnings(mxlv-ingest-unit)
  target_include_directories(mxlv-ingest-unit PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(mxlv-ingest-unit PRIVATE Catch2::Catch2WithMain)
  catch_discover_tests(mxlv-ingest-unit)
endif()
