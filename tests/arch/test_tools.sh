#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Tikrina scripts/linux/lib/tools.sh (pick_compiler, find_nvidia_smi) ir scripts/wsl/ nukreipimus.
# Naudojimas: test_tools.sh [repo_šaknis]. Išėjimo kodas 0 = viskas gerai.
set -uo pipefail
REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TOOLS="$REPO/scripts/linux/lib/tools.sh"
fail=0
check() { # aprašas, tikėtina, gauta
  if [[ "$2" == "$3" ]]; then echo "OK   $1"; else echo "FAIL $1: tikėtasi „$2“, gauta „$3“"; fail=1; fi
}
ENVB="$(command -v env)"; BASH_BIN="$(command -v bash)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
BIN="$TMP/bin"; mkdir -p "$BIN"
for t in cc c++ gcc g++; do printf '#!/bin/sh\nexit 0\n' > "$BIN/$t"; chmod +x "$BIN/$t"; done

# 1) nustatyti kintamieji gerbiami
r="$("$ENVB" -i PATH="$BIN" CC_BIN=gcc CXX_BIN=g++ "$BASH_BIN" -c 'source "$1"; pick_compiler; echo "$CC_BIN $CXX_BIN $CUDA_HOST_CXX"' _ "$TOOLS")"
check "pick_compiler gerbia CC_BIN/CXX_BIN" "gcc g++ g++" "$r"
# 2) be kintamųjų ir be clang-19 — cc / c++
r="$("$ENVB" -i PATH="$BIN" "$BASH_BIN" -c 'source "$1"; pick_compiler; echo "$CC_BIN $CXX_BIN $CUDA_HOST_CXX"' _ "$TOOLS")"
check "pick_compiler be clang-19 -> cc/c++" "cc c++ g++" "$r"
# 3) yra clang-19 — jis pasirenkamas
for t in clang-19 clang++-19; do cp "$BIN/cc" "$BIN/$t"; done
r="$("$ENVB" -i PATH="$BIN" "$BASH_BIN" -c 'source "$1"; pick_compiler; echo "$CC_BIN $CXX_BIN"' _ "$TOOLS")"
check "pick_compiler su clang-19" "clang-19 clang++-19" "$r"
rm -f "$BIN/clang-19" "$BIN/clang++-19"
# 4) nėra jokio kompiliatoriaus — kodas != 0 ir pranešimas
rm -f "$BIN/cc" "$BIN/c++"
msg="$("$ENVB" -i PATH="$BIN" "$BASH_BIN" -c 'source "$1"; pick_compiler' _ "$TOOLS" 2>&1)"; rc=$?
check "pick_compiler be kompiliatoriaus -> kodas 1" "1" "$rc"
[[ "$msg" == *pick_compiler* ]] && echo "OK   pranešimas aiškus" || { echo "FAIL nėra pranešimo"; fail=1; }

# 5) find_nvidia_smi: tuščias PATH ir nėra atsarginio kelio -> 1
out="$("$ENVB" -i PATH= VRAMFLOW_WSL_NVIDIA_SMI="$TMP/nera" "$BASH_BIN" -c 'source "$1"; find_nvidia_smi' _ "$TOOLS")"; rc=$?
check "find_nvidia_smi nieko nerasta -> kodas 1" "1" "$rc"; check "find_nvidia_smi nieko neišveda" "" "$out"
# 6) atsarginis kelias
printf '#!/bin/sh\nexit 0\n' > "$TMP/smi"; chmod +x "$TMP/smi"
out="$("$ENVB" -i PATH= VRAMFLOW_WSL_NVIDIA_SMI="$TMP/smi" "$BASH_BIN" -c 'source "$1"; find_nvidia_smi' _ "$TOOLS")"; rc=$?
check "find_nvidia_smi atsarginis kelias" "0 $TMP/smi" "$rc $out"
# 7) PATH turi pirmumą prieš atsarginį
cp "$TMP/smi" "$BIN/nvidia-smi"
out="$("$ENVB" -i PATH="$BIN" VRAMFLOW_WSL_NVIDIA_SMI="$TMP/smi" "$BASH_BIN" -c 'source "$1"; find_nvidia_smi' _ "$TOOLS")"
check "find_nvidia_smi pirma PATH" "$BIN/nvidia-smi" "$out"

# 8) scripts/wsl/*.sh nukreipimai
shopt -s nullglob
n=0
for stub in "$REPO"/scripts/wsl/*.sh "$REPO"/scripts/wsl/lib/*.sh "$REPO"/scripts/wsl/*.py; do
  rel="${stub#"$REPO"/scripts/wsl/}"
  n=$((n + 1))
  if ! grep -q "linux/$rel" "$stub" || [[ "$(grep -vc '^#' "$stub")" -gt 3 ]]; then echo "FAIL $rel nėra nukreipimas į linux/$rel"; fail=1; continue; fi
  [[ -f "$REPO/scripts/linux/$rel" ]] || { echo "FAIL $rel: nėra tikslo scripts/linux/$rel"; fail=1; continue; }
done
[[ $n -gt 0 ]] || { echo "FAIL nėra nukreipimų scripts/wsl/"; fail=1; }
# kiekvienam scripts/linux failui yra nukreipimas
for f in "$REPO"/scripts/linux/*.sh "$REPO"/scripts/linux/lib/pair.sh "$REPO"/scripts/linux/*.py; do
  rel="${f#"$REPO"/scripts/linux/}"
  [[ -f "$REPO/scripts/wsl/$rel" ]] || { echo "FAIL trūksta nukreipimo scripts/wsl/$rel"; fail=1; }
done
echo "nukreipimų patikrinta: $n"
exit $fail
