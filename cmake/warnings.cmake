# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Vieninga kompiliatoriaus įspėjimų politika visiems savo tikslams (biblioteka, vykdomasis, testas).
# C++: -Wall -Wextra -Wpedantic. CUDA: -Wall -Wextra perduodami host kompiliatoriui per -Xcompiler; nvcc pats
# dar gauna `--Werror all-warnings`, kai VRAMFLOW_WERROR=ON. Trečiųjų šalių antraštės (MXL, Catch2, CLI11,
# libav, CUDA) turi būti SYSTEM include'ai, kad jų įspėjimai nesimaišytų — žr. vramflow_system_includes().
option(VRAMFLOW_WERROR "Įspėjimai = klaidos (-Werror) savo tikslams" ON)

# -Wpedantic CUDA host daliai NEįjungtas: nvcc sugeneruotas .cudafe1.cpp turi `# N "failas"` eilučių žymas, dėl kurių
# g++ -Wpedantic duoda ~800 įspėjimų "style of line directive is a GCC extension" (nvcc artefaktas, ne mūsų kodas;
# be savo -W vėliavėlės). Mūsų antraštės pedantiškai tikrinamos per .cpp tikslus; kitų -Wpedantic įspėjimų .cu faile nebuvo.
function(vramflow_warnings target)
  set(cxx_flags -Wall -Wextra -Wpedantic)
  set(cuda_host_flags -Wall,-Wextra)
  if(VRAMFLOW_WERROR)
    list(APPEND cxx_flags -Werror)
    set(cuda_host_flags "${cuda_host_flags},-Werror")
  endif()
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    return()
  endif()
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler ${cuda_host_flags}>")
  if(VRAMFLOW_WERROR)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Werror all-warnings>")
  endif()
endfunction()
