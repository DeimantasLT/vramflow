#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Sukompiliuoja vramflow-verify ir jo testus (build katalogas WSL Linux FS, ne /mnt/c).
# Papildomi CMake argumentai perduodami toliau, pvz.: 30-build-verify.sh -DMXLV_ENABLE_SANITIZERS=ON
source "$(dirname "$0")/env.sh"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${VERIFY_BUILD:-$HOME/build/vramflow/release}"
cmake -S "$REPO" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=g++ \
  -DMXL_ROOT="$MXL_SRC/install/$MXL_PRESET" -DMXL_SOURCE_DIR="$MXL_SRC" "$@"
cmake --build "$BUILD"
echo "VERIFY_BUILD_OK $BUILD"
