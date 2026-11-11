# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Bendra aplinka visiems WSL skriptams. Naudojimas: source "$(dirname "$0")/env.sh"
set -euo pipefail
export MXL_SRC="${MXL_SRC:-$HOME/src/mxl-pr653}"
export MXL_PRESET="${MXL_PRESET:-Linux-Clang-Release}"
export MXL_BUILD="$MXL_SRC/build/$MXL_PRESET"
export MXL_PR_SHA="${MXL_PR_SHA:-6dddc91d92f56c3c9d90fa089c95d8f262bc2c92}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export VCPKG_ROOT="${VCPKG_ROOT:-$HOME/vcpkg}"
# Windows PATH WSL'e nereikalingas ir trukdo (tarpai, Windows CUDA)
export PATH="$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
export LOG_DIR="${LOG_DIR:-$HOME/logs}"
mkdir -p "$LOG_DIR"
