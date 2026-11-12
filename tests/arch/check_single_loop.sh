#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Vieno ciklo sargas (planas docs/planai/2026-11-12-tick-variklis.md V16, A3): funkcijos ir libvramflow-gfx ritmą gauna
# tik iš bendro tick variklio (src/engine/). Jų šaltiniuose neturi būti savo laukimo ciklo pagalbininkų: nei
# func::wait_until(, nei santykinio miego mxlSleepForNs( (A27 — absoliutus mxlSleepUntil gyvena SystemClock).
# Naudojimas: check_single_loop.sh [repo_šaknis]. Išėjimo kodas 0 = švaru.
set -uo pipefail
REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FILES=(src/func/gen.cpp src/func/logo.cpp src/func/composite.cpp src/func/enc.cpp src/func/dec.cpp src/func/gfxgen.cpp
       src/gfx/gfx_writer.cpp)
fail=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$REPO/$f" ]]; then
    echo "FAIL $f: failo nėra (sargo sąrašas pasenęs?)"
    fail=1
    continue
  fi
  if hits="$(grep -nE 'wait_until\(|mxlSleepForNs\(' "$REPO/$f")"; then
    echo "FAIL $f: antras ritmo ciklas (naudok engine::TickEngine / Clock):"
    echo "$hits" | sed 's/^/       /'
    fail=1
  else
    echo "OK   $f"
  fi
done
exit $fail
