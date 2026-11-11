#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Sukonfigūruoja ir sukompiliuoja MXL PR #653 su CUDA payload backend'u.
source "$(dirname "$0")/env.sh"
cd "$MXL_SRC"
actual="$(git rev-parse HEAD)"
[[ "$actual" == "$MXL_PR_SHA" ]] || { echo "KLAIDA: HEAD $actual != $MXL_PR_SHA"; exit 2; }

overlay_args=()
if [[ -d "$HOME/vcpkg-overlay" ]] && [[ -n "$(ls -A "$HOME/vcpkg-overlay")" ]]; then
  echo "PASTABA: naudojami vcpkg overlay port'ai: $(ls "$HOME/vcpkg-overlay" | tr '\n' ' ')(žr. 05-vcpkg-patch-overlay.sh)"
  overlay_args=(-DVCPKG_OVERLAY_PORTS="$HOME/vcpkg-overlay")
fi

cmake --preset "$MXL_PRESET" "${overlay_args[@]}" \
  -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 \
  -DMXL_ENABLE_CUDA_PAYLOAD=ON -DBUILD_DOCS=OFF \
  -DCUDAToolkit_ROOT="$CUDA_HOME" 2>&1 | tee "$LOG_DIR/configure.log"

cmake --build "$MXL_BUILD" 2>&1 | tee "$LOG_DIR/build.log"
cmake --install "$MXL_BUILD" 2>&1 | tee "$LOG_DIR/install.log"  # → $MXL_SRC/install/$MXL_PRESET (preset installDir)
echo "BUILD_OK"
