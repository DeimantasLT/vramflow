#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Žingsnis 0: sukompiliuoja ir paleidžia cuda_fd_probe. Rezultatas: $EVIDENCE/00-probe/probe.jsonl
source "$(dirname "$0")/env.sh"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${EVIDENCE:-$HOME/evidence/latest}/00-probe"
mkdir -p "$OUT" "$HOME/build/vramflow"
WSL_LIBDIR=(); [[ -d /usr/lib/wsl/lib ]] && WSL_LIBDIR=(-L/usr/lib/wsl/lib)
g++ -std=c++20 -O2 -Wall -Wextra -I"$CUDA_HOME/include" \
    "$REPO/probe/cuda_fd_probe.cpp" -o "$HOME/build/vramflow/cuda_fd_probe" \
    "${WSL_LIBDIR[@]}" -L"$CUDA_HOME/lib64/stubs" -lcuda
set +e
"$HOME/build/vramflow/cuda_fd_probe" | tee "$OUT/probe.jsonl"
code=${PIPESTATUS[0]}
set -e
echo "$code" > "$OUT/exit_code"
echo "PROBE_EXIT=$code"
exit "$code"
