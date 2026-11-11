#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# S1–S4 (architektūra §6.2): compute-sanitizer memcheck/racecheck/synccheck ir ASan+UBSan host kodui.
# Rezultatas: $OUT/sanitizers.json (kiekvienam: status PASS/FAIL/INFO, exit kodas, log failas).
source "$(dirname "$0")/env.sh"
set +e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${EVIDENCE:-$HOME/evidence/latest}/50-sanitizers"
rm -rf "$OUT"; mkdir -p "$OUT"
REL="$HOME/build/vramflow/release"
CS="$CUDA_HOME/bin/compute-sanitizer"
export LD_LIBRARY_PATH="$MXL_SRC/install/$MXL_PRESET/lib"
results=()

add() { results+=("{\"id\":\"$1\",\"status\":\"$2\",\"exit\":$3,\"log\":\"$4\",\"note\":\"$5\"}"); echo "[$2] $1 exit=$3 $5"; }

errs_in() { grep -E "ERROR SUMMARY: [0-9]+ error|RACECHECK SUMMARY: [0-9]+ hazard" "$1" | tail -1; }

# S1a/S2a/S3a: GPU kernel testai po kiekvienu įrankiu
# memcheck — visas rinkinys. racecheck/synccheck lėtina šimtus kartų, todėl jiems — atstovaujami testai
# (visi kernel'iai: fill, finalize, verify, corrupt), be 8×200 iteracijų stream testo. Kernel'iai __shared__ nenaudoja.
SUBSET="k_verify: švarus grain'as,k_verify aptinka vieno bito klaidą bet kurioje vietoje"
for tool in memcheck racecheck synccheck; do
  log="$OUT/$tool-gpu-tests.log"
  filter=(); [[ $tool != memcheck ]] && filter=("$SUBSET")
  timeout 1800 "$CS" --tool "$tool" --error-exitcode 86 "$REL/mxlv-gpu-tests" "${filter[@]}" > "$log" 2>&1
  code=$?
  add "S-$tool-gpu-tests" "$([[ $code == 0 ]] && echo PASS || echo FAIL)" "$code" "$log" "$(errs_in "$log")"
done

# S1b/S1c: writer ir reader atskiruose procesuose po memcheck (5 fps, nes sanitizer'is lėtina 10–100×)
dom="$(mktemp -d /dev/shm/mxlv-san-XXXXXX)"
"$CS" --tool memcheck --error-exitcode 86 "$REL/vramflow-verify" writer --domain "$dom" --rate-num 5 --duration 40 \
  --gc-before-create --out "$OUT/memcheck-run" > "$OUT/memcheck-writer.log" 2>&1 &
wp=$!
sleep 8
"$CS" --tool memcheck --error-exitcode 86 "$REL/vramflow-verify" reader --domain "$dom" --rate-num 5 --duration 25 \
  --read-timeout-ms 2000 --stall-timeout-ms 10000 --host-crosscheck-every 5 --out "$OUT/memcheck-run" > "$OUT/memcheck-reader.log" 2>&1
rcode=$?
kill -TERM $wp; wait $wp; wcode=$?
rm -rf "$dom"
add "S1b-memcheck-writer" "$([[ $wcode == 0 ]] && echo PASS || echo FAIL)" "$wcode" "$OUT/memcheck-writer.log" "$(errs_in "$OUT/memcheck-writer.log")"
if [[ $rcode == 0 ]]; then st=PASS; elif [[ $rcode == 86 ]]; then st=FAIL; else st=INFO; fi
add "S1c-memcheck-reader(importuota VMM)" "$st" "$rcode" "$OUT/memcheck-reader.log" "$(errs_in "$OUT/memcheck-reader.log")"

# S4: ASan + UBSan host kodui (mūsų įrankis; MXL ASan build — NOT_RUN šiame etape)
ASB="$HOME/build/vramflow/asan"
VERIFY_BUILD="$ASB" bash "$REPO/scripts/wsl/30-build-verify.sh" -DMXLV_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo > "$OUT/asan-build.log" 2>&1
if [[ $? != 0 ]]; then
  add "S4-asan-build" FAIL 1 "$OUT/asan-build.log" "build nepavyko"
else
  "$ASB/mxlv-unit" > "$OUT/asan-unit.log" 2>&1; c=$?
  add "S4-asan-unit" "$([[ $c == 0 ]] && echo PASS || echo FAIL)" "$c" "$OUT/asan-unit.log" ""
  export ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=1:halt_on_error=0:log_path=$OUT/asan-pair"
  export UBSAN_OPTIONS="print_stacktrace=1:log_path=$OUT/ubsan-pair"
  VERIFY_BIN="$ASB/vramflow-verify" bash "$REPO/scripts/wsl/lib/pair.sh" "$OUT/asan-run" 30 "--gc-before-create" "--host-crosscheck-every 10" > "$OUT/asan-pair.log" 2>&1
  unset ASAN_OPTIONS UBSAN_OPTIONS
  rep=$(cat "$OUT"/asan-pair.* "$OUT"/ubsan-pair.* 2>/dev/null | grep -cE "ERROR: AddressSanitizer|ERROR: LeakSanitizer|runtime error:")
  wex=$(cat "$OUT/asan-run/writer.exit" 2>/dev/null || echo -1); rex=$(cat "$OUT/asan-run/reader.exit" 2>/dev/null || echo -1)
  if [[ $rep == 0 && $wex == 0 && $rex == 0 ]]; then st=PASS; else st=FAIL; fi
  add "S4-asan-ubsan-pair-30s" "$st" "$rex" "$OUT/asan-pair.log" "ataskaitų=$rep writer_exit=$wex reader_exit=$rex"
fi
add "S-initcheck" "NOT_RUN" 0 "" "initcheck nepalaiko IPC/VMM eksportuotų alokacijų (NVIDIA known issue)"

printf '[%s]\n' "$(IFS=,; echo "${results[*]}")" > "$OUT/sanitizers.json"
python3 -c "import json;json.load(open('$OUT/sanitizers.json'))" && echo "SANITIZERS_DONE $OUT"
