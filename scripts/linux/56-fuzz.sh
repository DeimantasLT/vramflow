#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Fuzz testai parseriams (planas docs/planai/2026-11-17-production-ready.md §1.5, kriterijus S2): flow aprašas, talpos failas,
# HLS grojaraštis, URL politika. libFuzzer + ASan + UBSan; kompiliatorius su fuzzer vykdymo biblioteka — FUZZ_CXX
# (numatyta clang++-18: clang++-19 pakete tos bibliotekos gali nebūti).
#
# Naudojimas: 56-fuzz.sh [išvesties katalogas]      FUZZ_SECONDS=600 (vienam taikiniui), FUZZ_TARGETS="flow_def capacity playlist url"
# Rezultatas: <out>/fuzz.json (taikinys, vykdymų skaičius, radiniai) ir <out>/<taikinys>/{log.txt, crash-*}; exit ≠ 0 — yra radinių.
# Radinį pridėkite į tests/fuzz/corpus/<taikinys>/ — ctest fuzz_replay_<taikinys> jį saugos visam laikui.
source "$(dirname "$0")/env.sh"
set +e; set +o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-${EVIDENCE:-$HOME/evidence/latest}/56-fuzz}"
SECS="${FUZZ_SECONDS:-600}"; TARGETS="${FUZZ_TARGETS:-flow_def capacity playlist url}"
FCXX="${FUZZ_CXX:-clang++-18}"; FCC="${FUZZ_CC:-clang-18}"
BUILD="${FUZZ_BUILD:-$HOME/build/vramflow/fuzz}"
command -v "$FCXX" >/dev/null || { echo "56-fuzz: nėra $FCXX (nustatykite FUZZ_CXX / FUZZ_CC)" >&2; exit 2; }
mkdir -p "$OUT"
cmake -S "$REPO" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER="$FCC" -DCMAKE_CXX_COMPILER="$FCXX" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_CXX" \
  -DMXL_ROOT="$MXL_SRC/install/$MXL_PRESET" -DMXL_SOURCE_DIR="$MXL_SRC" -DMXLV_BUILD_FUZZ=ON > "$OUT/cmake.log" 2>&1 \
  || { echo "56-fuzz: cmake nepavyko ($OUT/cmake.log)" >&2; exit 2; }
tg=(); for t in $TARGETS; do tg+=("mxlv-fuzz-$t"); done
cmake --build "$BUILD" --target "${tg[@]}" > "$OUT/build.log" 2>&1 || { echo "56-fuzz: build nepavyko ($OUT/build.log)" >&2; tail -20 "$OUT/build.log" >&2; exit 2; }
results=(); bad=0
for t in $TARGETS; do
  d="$OUT/$t"; rm -rf "$d"; mkdir -p "$d/corpus"
  cp "$REPO/tests/fuzz/corpus/$t/"* "$d/corpus/" 2>/dev/null
  ( cd "$d" && "$BUILD/mxlv-fuzz-$t" -max_total_time="$SECS" -timeout=5 -rss_limit_mb=2048 -max_len=65536 -print_final_stats=1 corpus ) > "$d/log.txt" 2>&1
  code=$?
  execs=$(grep -oE 'stat::number_of_executed_units: [0-9]+' "$d/log.txt" | awk '{print $2}'); execs=${execs:-0}
  found=$(ls "$d" | grep -cE '^(crash|timeout|oom|leak)-')
  [[ $code != 0 || $found != 0 ]] && bad=1
  results+=("{\"target\":\"$t\",\"seconds\":$SECS,\"execs\":$execs,\"findings\":$found,\"exit\":$code,\"corpus\":$(ls "$d/corpus" | wc -l)}")
  echo "[$([[ $code == 0 && $found == 0 ]] && echo PASS || echo FAIL)] fuzz $t: $execs vykdymų per $SECS s, radinių $found, exit $code"
done
printf '[%s]\n' "$(IFS=,; echo "${results[*]}")" > "$OUT/fuzz.json"
python3 -c "import json;json.load(open('$OUT/fuzz.json'))" && echo "FUZZ_DONE $OUT"
exit $bad
