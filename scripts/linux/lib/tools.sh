# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Bendros įrankių paieškos funkcijos Linux skriptams (natyvus Linux ir WSL2). Naudojimas: source lib/tools.sh
# Šis failas tik apibrėžia funkcijas; jokių `set` parinkčių nekeičia.

# WSL2 atsarginis nvidia-smi kelias; testams perrašomas VRAMFLOW_WSL_NVIDIA_SMI.
_vramflow_wsl_smi() { printf '%s' "${VRAMFLOW_WSL_NVIDIA_SMI:-/usr/lib/wsl/lib/nvidia-smi}"; }

# pick_compiler — nustato ir eksportuoja CC_BIN, CXX_BIN, CUDA_HOST_CXX.
# Jau nustatytos reikšmės gerbiamos; numatyta clang-19 / clang++-19, jei yra, kitaip cc / c++; CUDA host — g++.
# Grąžina 1 su aiškiu pranešimu, jei pasirinkto kompiliatoriaus nėra PATH.
pick_compiler() {
  if [[ -z "${CC_BIN:-}" ]]; then
    if command -v clang-19 >/dev/null 2>&1; then CC_BIN=clang-19; else CC_BIN=cc; fi
  fi
  if [[ -z "${CXX_BIN:-}" ]]; then
    if command -v clang++-19 >/dev/null 2>&1; then CXX_BIN=clang++-19; else CXX_BIN=c++; fi
  fi
  CUDA_HOST_CXX="${CUDA_HOST_CXX:-g++}"
  local c
  for c in "$CC_BIN" "$CXX_BIN"; do
    if ! command -v "$c" >/dev/null 2>&1; then
      echo "pick_compiler: kompiliatoriaus „$c“ nėra PATH; įdiekite jį arba nustatykite CC_BIN / CXX_BIN" >&2
      return 1
    fi
  done
  export CC_BIN CXX_BIN CUDA_HOST_CXX
}

# find_nvidia_smi — išveda nvidia-smi kelią: pirma `command -v`, WSL2 kelias tik kaip atsarginis.
# Grąžina 1 (nieko neišvedus), jei nėra nei vieno.
find_nvidia_smi() {
  local p
  p="$(command -v nvidia-smi 2>/dev/null)" && [[ -n "$p" ]] && { printf '%s\n' "$p"; return 0; }
  p="$(_vramflow_wsl_smi)"
  [[ -x "$p" ]] && { printf '%s\n' "$p"; return 0; }
  return 1
}
