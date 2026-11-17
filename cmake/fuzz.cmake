# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Fuzz testai parseriams, gaunantiems ne mūsų baitus (planas docs/planai/2026-11-17-production-ready.md §1.5, kriterijus S2):
# flow aprašas, talpos failas, HLS grojaraštis, URL politika. Visi taikiniai — gryna logika (be MXL, CUDA ir libav).
# Iš to paties šaltinio (tests/fuzz/fuzz_<taikinys>.cpp, vienas LLVMFuzzerTestOneInput) daromi du vykdomieji:
#   mxlv-fuzz-replay-<taikinys> — visada (MXLV_BUILD_TESTS): paprastas main paleidžia tests/fuzz/corpus/<taikinys>/ failus;
#       ctest fuzz_replay_<taikinys>. Fuzzerio radinys → failas į korpusą → lieka testu visam laikui.
#   mxlv-fuzz-<taikinys> — tik su MXLV_BUILD_FUZZ=ON: libFuzzer + ASan + UBSan; reikia clang su fuzzer vykdymo biblioteka
#       (pvz. clang++-18) — scripts/linux/56-fuzz.sh.
option(MXLV_BUILD_FUZZ "libFuzzer taikiniai parseriams (reikia clang su fuzzer vykdymo biblioteka)" OFF)

set(MXLV_FUZZ_TARGETS flow_def capacity playlist url)
set(MXLV_FUZZ_SRC_flow_def "")
set(MXLV_FUZZ_SRC_capacity src/runtime/admission.cpp src/runtime/domain_lease.cpp src/runtime/safe_files.cpp)
set(MXLV_FUZZ_SRC_playlist src/ingest/url_policy.cpp)
set(MXLV_FUZZ_SRC_url src/ingest/url_policy.cpp)

foreach(t ${MXLV_FUZZ_TARGETS})
  set(_src tests/fuzz/fuzz_${t}.cpp ${MXLV_FUZZ_SRC_${t}})
  if(MXLV_BUILD_TESTS)
    add_executable(mxlv-fuzz-replay-${t} ${_src} tests/fuzz/replay_main.cpp)
    vramflow_warnings(mxlv-fuzz-replay-${t})
    target_include_directories(mxlv-fuzz-replay-${t} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME fuzz_replay_${t} COMMAND mxlv-fuzz-replay-${t} ${CMAKE_SOURCE_DIR}/tests/fuzz/corpus/${t})
  endif()
  if(MXLV_BUILD_FUZZ)
    add_executable(mxlv-fuzz-${t} ${_src})
    vramflow_warnings(mxlv-fuzz-${t})
    target_include_directories(mxlv-fuzz-${t} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    # -fno-sanitize-recover: UBSan pranešimas nutraukia taikinį (kitaip jis tik išspausdinamas ir paleidimas lieka „žalias“)
    target_compile_options(mxlv-fuzz-${t} PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined
                           -fno-omit-frame-pointer -g)
    target_link_options(mxlv-fuzz-${t} PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined)
  endif()
endforeach()
