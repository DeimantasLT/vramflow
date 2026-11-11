#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Tikrina, kad binare yra SASS šio kompiuterio GPU architektūrai (ne tik kitos architektūros kodas + JIT).
# Naudojimas: check_arch.sh <binaras> <cuobjdump>. Išėjimo kodas 77 = praleista (nėra GPU ar įrankių).
set -euo pipefail
bin="$1"; cuobjdump="$2"
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/linux/lib/tools.sh"
smi="$(find_nvidia_smi || true)"
[[ -n "$smi" && -x "$cuobjdump" ]] || { echo "SKIP: nėra nvidia-smi arba cuobjdump"; exit 77; }
cc="$("$smi" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
[[ -n "$cc" ]] || { echo "SKIP: GPU nerasta"; exit 77; }
elfs="$("$cuobjdump" --list-elf "$bin" 2>/dev/null || true)"
echo "GPU sm_$cc; binare: $(echo "$elfs" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
echo "$elfs" | grep -q "sm_${cc}\b" || { echo "FAIL: $bin neturi sm_$cc kodo"; exit 1; }
echo "OK"
