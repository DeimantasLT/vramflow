# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Build grūdinimas visiems saviems tikslams (planas docs/planai/2026-11-17-production-ready.md §1.1, kriterijus S6).
# Kviečiama iš vramflow_warnings(), todėl naujas tikslas vėliavas gauna pats; tests/arch/check_hardening.py (ctest
# build_hardening) tikrina kiekvieną savą vertimo vienetą ir kiekvieną savą ELF failą.
#   Kompiliavimas: stack protector, stack clash protection, CET (-fcf-protection), libstdc++ tvirtinimai ir
#                  _FORTIFY_SOURCE=3 — tik optimizuotam build'ui be sanitizerių (su ASan / TSan / libFuzzer nesuderinamas).
#   Susiejimas:    pilnas RELRO (-z relro -z now), nevykdomas stekas. PIE — CMAKE_POSITION_INDEPENDENT_CODE (CMakeLists.txt).
# CUDA failų host daliai tos pačios vėliavos perduodamos per -Xcompiler (nvcc host kompiliatorius — g++).
option(VRAMFLOW_HARDENING "Grūdinimo vėliavos saviems tikslams (stack protector, FORTIFY, RELRO, CET)" ON)

function(vramflow_hardening target)
  if(NOT VRAMFLOW_HARDENING)
    return()
  endif()
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    return()
  endif()
  set(c_flags -fstack-protector-strong -fstack-clash-protection -fcf-protection=full)
  set(cxx_flags ${c_flags} -D_GLIBCXX_ASSERTIONS)
  set(cuda_host_flags "-fstack-protector-strong,-fstack-clash-protection,-fcf-protection=full,-D_GLIBCXX_ASSERTIONS")
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler ${cuda_host_flags}>")
  if(NOT MXLV_ENABLE_SANITIZERS AND NOT MXLV_ENABLE_TSAN AND NOT MXLV_BUILD_FUZZ)
    # -U pirma: Ubuntu g++ ir kai kurie clang paketai _FORTIFY_SOURCE apibrėžia patys (kitaip — „redefined“ su -Werror)
    set(fortify -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3)
    target_compile_options(${target} PRIVATE
      "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<NOT:$<CONFIG:Debug>>>:${fortify}>"
      "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:SHELL:-Xcompiler -U_FORTIFY_SOURCE,-D_FORTIFY_SOURCE=3>")
  endif()
  if(_type STREQUAL "EXECUTABLE" OR _type STREQUAL "SHARED_LIBRARY" OR _type STREQUAL "MODULE_LIBRARY")
    target_link_options(${target} PRIVATE -Wl,-z,relro,-z,now -Wl,-z,noexecstack)
  endif()
endfunction()
