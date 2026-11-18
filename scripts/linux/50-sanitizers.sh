#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Sanitizeriai (planas docs/planai/2026-11-17-production-ready.md §2.1–2.3, kriterijai K2, S3). Trys etapai:
#   cs    compute-sanitizer GPU branduoliams (gamybinis build'as): memcheck — visi GPU testų binarai ir writer ↔ reader pora;
#         racecheck / synccheck — atstovaujami testai (lėtina šimtus kartų);
#   asan  ASan + UBSan visam host kodui: atskiras build'as, visas ctest, writer ↔ reader pora ir funkciniai scenarijai, tarp
#         jų visi klaidų kelių testai (planas RF-3: sanitizeris randa tik tai, kas įvykdoma);
#   tsan  ThreadSanitizer gijų kodui: atskiras build'as, tik CPU kelias (su NVIDIA tvarkykle TSan nepalaikomas).
# Sanitizerių build'ams — clang++-18: jis turi vykdymo bibliotekas (clang++-19 pakete jų nėra).
#
# Naudojimas: 50-sanitizers.sh [cs | asan | tsan | all] …      (numatyta all)
#   SAN_OUT=<katalogas>                 numatyta $EVIDENCE/50-sanitizers
#   VERIFY_BUILD=<gamybinis build'as>   numatyta ~/build/vramflow/func (reikia -DMXLV_BUILD_FUNC=ON: flow'ų skaitymo testai)
#   SAN_ASAN_BUILD, SAN_TSAN_BUILD      numatyta ~/build/vramflow/asan ir …/tsan;  SAN_SKIP_BUILD=1 — nekompiliuoti iš naujo
#   SAN_CC, SAN_CXX                     numatyta clang-18, clang++-18
#   SAN_FUNC_TESTS                      funkciniai scenarijai po ASan (functional.py --only)
#   SAN_NONROOT_USER=<vartotojas>       papildomai paleisti fault_domain_readonly jo vardu (testui reikia ne root)
#   SAN_LSAN_SUPPRESSIONS=<failas>      LeakSanitizer slopinimo sąrašas — TIK trečiųjų šalių bibliotekoms ir tik autoriui
#                                       patvirtinus; kai jis naudojamas, tai įrašoma į kiekvieno rezultato pastabą
# Vertinimas: FAIL — sanitizerio ataskaita arba avarija (signalas). Laiko tvirtinimai po sanitizeriu nevertinami (lėtėja
# kelis kartus): testų nesėkmės be ataskaitų įrašomos kaip INFO su testų vardais — jas reikia peržiūrėti akimis.
# Rezultatas: <out>/sanitizers.json (id, status, exit, log, note); exit 0 — nė vieno FAIL.
source "$(dirname "$0")/env.sh"
set +e; set +o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${SAN_OUT:-${EVIDENCE:-$HOME/evidence/latest}/50-sanitizers}"
REL="${VERIFY_BUILD:-$HOME/build/vramflow/func}"
ASB="${SAN_ASAN_BUILD:-$HOME/build/vramflow/asan}"; TSB="${SAN_TSAN_BUILD:-$HOME/build/vramflow/tsan}"
SAN_CC="${SAN_CC:-clang-18}"; SAN_CXX="${SAN_CXX:-clang++-18}"
SAN_FUNC_TESTS="${SAN_FUNC_TESTS:-fcomp1,ffmt_logo,ffmt_dec,ffmt_enc,overload_partial,fault_vram_exhausted,fault_alloc_fail,fault_nvenc_sessions,fault_cuda_midrun,fault_shm_full}"
CS="$CUDA_HOME/bin/compute-sanitizer"
CLIPS="${INGEST_CLIPS:-$HOME/media-test/av}"
export LD_LIBRARY_PATH="$MXL_SRC/install/$MXL_PRESET/lib"
SYMB="$(command -v llvm-symbolizer-18 || command -v llvm-symbolizer || true)"
STAGES=("$@"); [[ ${#STAGES[@]} == 0 || " ${STAGES[*]} " == *" all "* ]] && STAGES=(cs asan tsan)
for s in "${STAGES[@]}"; do
  case "$s" in cs|asan|tsan) ;; *) echo "50-sanitizers: nežinomas etapas „$s“ (cs | asan | tsan | all)" >&2; exit 2;; esac
done
mkdir -p "$OUT"
results=(); fails=0
SUPP_NOTE=""; [[ -n ${SAN_LSAN_SUPPRESSIONS:-} ]] && SUPP_NOTE=" [su LSan slopinimo sąrašu $(basename "$SAN_LSAN_SUPPRESSIONS")]"
RX_REPORT="ERROR: AddressSanitizer|ERROR: LeakSanitizer|runtime error:|WARNING: ThreadSanitizer|ERROR: ThreadSanitizer"

add() { # id, statusas (PASS | FAIL | INFO | NOT_RUN), exit kodas, žurnalas, pastaba
  local note="${5//$'\n'/ }"; note="${note//\\/\\\\}"; note="${note//\"/\\\"}"; note="${note//$'\t'/ }"
  results+=("{\"id\":\"$1\",\"status\":\"$2\",\"exit\":$3,\"log\":\"$4\",\"note\":\"$note\"}")
  [[ $2 == FAIL ]] && fails=$((fails + 1))
  echo "[$2] $1 exit=$3 $5"
}
# Sanitizerių ataskaitų skaičius failuose (ASan, LSan, UBSan, TSan) ir jų antraštės pastabai
reports() { cat "$@" 2>/dev/null | grep -cE "$RX_REPORT"; }
report_heads() { cat "$@" 2>/dev/null | grep -E "$RX_REPORT|SUMMARY: " | sort | uniq -c | sort -rn | head -4 | cut -c1-170 | tr '\n' ';'; }
crashed() { [[ $1 -ge 128 ]]; }
# ASan / TSan vertinimas: ataskaitos → FAIL; avarija → FAIL; exit 86 (sanitizerio kodas) → FAIL; kitaip exit 0 → PASS, ne 0 → INFO (pastaboje — kas nepraėjo)
judge() { # id, exit, žurnalas, ataskaitų skaičius, pastaba
  if [[ $4 != 0 ]]; then add "$1" FAIL "$2" "$3" "sanitizerio ataskaitų: $4; $5$SUPP_NOTE"
  elif crashed "$2"; then add "$1" FAIL "$2" "$3" "avarija (signalas); $5$SUPP_NOTE"
  elif [[ $2 == 86 ]]; then add "$1" FAIL "$2" "$3" "sanitizerio exit kodas 86 be perskaitytos ataskaitos; $5$SUPP_NOTE"
  elif [[ $2 == 0 ]]; then add "$1" PASS "$2" "$3" "ataskaitų 0; $5$SUPP_NOTE"
  else add "$1" INFO "$2" "$3" "ataskaitų 0, bet exit $2 (laiko tvirtinimai po sanitizeriu nevertinami): $5$SUPP_NOTE"; fi
}
build_san() { # build katalogas, žurnalas, papildomos cmake vėliavos…
  local b="$1" log="$2"; shift 2
  CC_BIN="$SAN_CC" CXX_BIN="$SAN_CXX" VERIFY_BUILD="$b" bash "$REPO/scripts/linux/30-build-verify.sh" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMXLV_BUILD_FUNC=ON "$@" > "$log" 2>&1
}
ctest_failed() { sed -n '/The following tests FAILED/,$p' "$1" | grep -E '^[[:space:]]+[0-9]+ - ' | sed -E 's/^[[:space:]]+[0-9]+ - //' | cut -c1-90 | head -12 | tr '\n' ';'; }

# ---------------------------------------------------------------------------------------------------------------- cs
errs_in() { grep -E "ERROR SUMMARY: [0-9]+ error|RACECHECK SUMMARY: [0-9]+ hazard" "$1" | tail -1; }
cs_judge() { # id, exit, žurnalas: FAIL — sanitizerio klaidos, nėra SUMMARY eilutės arba avarija
  local sum st; sum="$(errs_in "$3")"
  if [[ $2 == 86 || -z $sum || ( $sum != *": 0 error"* && $sum != *": 0 hazard"* ) ]]; then st=FAIL
  elif crashed "$2"; then st=FAIL
  elif [[ $2 == 0 ]]; then st=PASS
  else st=INFO; sum="$sum; testų nesėkmės be sanitizerio klaidų (lėtėjimas) — peržiūrėti"; fi
  add "$1" "$st" "$2" "$3" "${sum:-nėra SUMMARY eilutės}"
}
stage_cs() {
  local d="$OUT/cs" bin log code tool dom wp rcode wcode
  rm -rf "$d"; mkdir -p "$d"
  [[ -x $CS ]] || { add "cs" FAIL 2 "" "nėra $CS"; return; }
  # memcheck — visi GPU testų binarai: media, gfx, v210a branduoliai, verify branduoliai ir flow'ų skaitymo kelias (S3)
  for bin in mxlv-gpu-tests mxlv-media-gpu-tests vramflow-gfx-gpu-tests vramflow-v210a-gpu-tests vramflow-flows-tests; do
    log="$d/memcheck-$bin.log"
    timeout 3600 "$CS" --tool memcheck --error-exitcode 86 "$REL/$bin" > "$log" 2>&1; code=$?
    cs_judge "cs-memcheck-$bin" "$code" "$log"
  done
  # racecheck / synccheck — atstovaujami testai (branduoliai __shared__ nenaudoja), be ilgų stream testų
  local subset="k_verify: švarus grain'as,k_verify aptinka vieno bito klaidą bet kurioje vietoje"
  for tool in racecheck synccheck; do
    log="$d/$tool-mxlv-gpu-tests.log"
    timeout 1800 "$CS" --tool "$tool" --error-exitcode 86 "$REL/mxlv-gpu-tests" "$subset" > "$log" 2>&1; code=$?
    cs_judge "cs-$tool-mxlv-gpu-tests" "$code" "$log"
  done
  # writer ir reader atskiruose procesuose po memcheck (5 fps: sanitizeris lėtina 10–100×). Istorija 2 s: su numatytąja
  # 200 ms 5 fps flow turėtų 1 grain'o ring'ą, o skaitytojas tokio nepriima („netikėtas layout“). Rašytojas baigia pats
  # (--duration): signalu nutrauktas compute-sanitizer suvestinės neišspausdina (exit 143).
  dom="$(mktemp -d /dev/shm/mxlv-san-XXXXXX)"
  echo '{"urn:x-mxl:option:history_duration/v1.0": 2000000000}' > "$dom/options.json"
  "$CS" --tool memcheck --error-exitcode 86 "$REL/vramflow-verify" writer --domain "$dom" --rate-num 5 --duration 55 \
    --gc-before-create --out "$d/memcheck-run" > "$d/memcheck-writer.log" 2>&1 &
  wp=$!
  sleep 8
  "$CS" --tool memcheck --error-exitcode 86 "$REL/vramflow-verify" reader --domain "$dom" --rate-num 5 --duration 25 \
    --read-timeout-ms 2000 --stall-timeout-ms 10000 --host-crosscheck-every 5 --out "$d/memcheck-run" > "$d/memcheck-reader.log" 2>&1
  rcode=$?
  wait $wp; wcode=$?
  rm -rf "$dom" "$dom.vramflow"
  cs_judge "cs-memcheck-writer" "$wcode" "$d/memcheck-writer.log"
  cs_judge "cs-memcheck-reader" "$rcode" "$d/memcheck-reader.log"
  add "cs-initcheck" NOT_RUN 0 "" "initcheck nepalaiko IPC / VMM eksportuotų alokacijų (NVIDIA known issue)"
}

# -------------------------------------------------------------------------------------------------------------- asan
san_env() { # ataskaitų katalogas → ASAN_OPTIONS / UBSAN_OPTIONS / LSAN_OPTIONS (exit 86 — sanitizerio radinys)
  mkdir -p "$1"
  export ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=1:exitcode=86:log_path=$1/asan"
  export UBSAN_OPTIONS="print_stacktrace=1:exitcode=86:log_path=$1/ubsan"
  export LSAN_OPTIONS="exitcode=86${SAN_LSAN_SUPPRESSIONS:+:suppressions=$SAN_LSAN_SUPPRESSIONS}"
  [[ -n $SYMB ]] && export ASAN_SYMBOLIZER_PATH="$SYMB"
}
stage_asan() {
  local d="$OUT/asan" c rep wex rex sig uh ud
  rm -rf "$d"; mkdir -p "$d"
  if [[ ${SAN_SKIP_BUILD:-0} != 1 ]]; then
    build_san "$ASB" "$d/build.log" -DMXLV_ENABLE_SANITIZERS=ON; c=$?
    add "asan-build" "$([[ $c == 0 ]] && echo PASS || echo FAIL)" "$c" "$d/build.log" "$SAN_CXX, įspėjimų $(grep -c 'warning:' "$d/build.log")"
    [[ $c != 0 ]] && return
  fi
  # 1) visas ctest (vieneto ir GPU testai): kiekvienas testas — atskiras procesas
  san_env "$d/rep-ctest"
  ( cd "$ASB" && ctest ) > "$d/ctest.log" 2>&1; c=$?
  rep=$(reports "$d"/rep-ctest/*)
  judge "asan-ctest" "$c" "$d/ctest.log" "$rep" "$(grep -E 'tests passed|tests failed' "$d/ctest.log" | tail -1); nepraėjo: $(ctest_failed "$d/ctest.log") $(report_heads "$d"/rep-ctest/*)"
  # 2) writer ↔ reader pora 30 s
  san_env "$d/rep-pair"
  VERIFY_BIN="$ASB/vramflow-verify" bash "$REPO/scripts/linux/lib/pair.sh" "$d/pair" 30 "--gc-before-create" "--host-crosscheck-every 10" > "$d/pair.log" 2>&1
  wex=$(cat "$d/pair/writer.exit" 2>/dev/null || echo 1); rex=$(cat "$d/pair/reader.exit" 2>/dev/null || echo 1)
  rep=$(reports "$d"/rep-pair/*)
  judge "asan-pair-30s" "$(( wex != 0 ? wex : rex ))" "$d/pair.log" "$rep" "writer exit $wex, reader exit $rex $(report_heads "$d"/rep-pair/*)"
  # 3) funkciniai scenarijai su ASan binarais (tarp jų klaidų keliai — RF-3)
  san_env "$d/rep-func"
  FUNC_BIN="$ASB/vramflow-func" VERIFY_BIN="$ASB/vramflow-verify" FFMT_FORMATS="${SAN_FFMT_FORMATS:-1080i50}" FFMT_DURATION=10 \
    python3 "$REPO/tests/functional/functional.py" "$d/functional" --only "$SAN_FUNC_TESTS" > "$d/functional.log" 2>&1; c=$?
  rep=$(reports "$d"/rep-func/*)
  sig=$(grep -rlE "Segmentation fault|core dumped|AddressSanitizer|runtime error:" "$d/functional" --include='*.err' 2>/dev/null | wc -l)
  judge "asan-functional" "$c" "$d/functional.log" "$((rep + sig))" "PASS $(grep -cE '^\[PASS' "$d/functional.log"), nepraėjo: $(grep -E '^\[FAIL' "$d/functional.log" | cut -d: -f1 | sed -E 's/^\[FAIL *\] //' | tr '\n' ';') $(report_heads "$d"/rep-func/*)"
  # 4) fault_domain_readonly — tik ne root vartotoju
  if [[ -n ${SAN_NONROOT_USER:-} && $(id -u) == 0 ]] && id -u "$SAN_NONROOT_USER" >/dev/null 2>&1; then
    uh="$(getent passwd "$SAN_NONROOT_USER" | cut -d: -f6)"; ud="$uh/vramflow-san"
    as_user() { setpriv --reuid="$SAN_NONROOT_USER" --regid="$(id -g "$SAN_NONROOT_USER")" --init-groups "$@"; }
    as_user rm -rf "$ud"
    as_user env -i HOME="$uh" USER="$SAN_NONROOT_USER" PATH="$PATH" LANG="${LANG:-C.UTF-8}" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      MXL_LIB="$MXL_SRC/install/$MXL_PRESET/lib" FUNC_BIN="$ASB/vramflow-func" VERIFY_BIN="$ASB/vramflow-verify" \
      ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=1:exitcode=86:log_path=$ud/rep/asan" \
      UBSAN_OPTIONS="print_stacktrace=1:exitcode=86:log_path=$ud/rep/ubsan" LSAN_OPTIONS="$LSAN_OPTIONS" ASAN_SYMBOLIZER_PATH="$SYMB" \
      bash -c 'mkdir -p "$1/rep" && python3 "$2/tests/functional/functional.py" "$1/functional" --only fault_domain_readonly' _ "$ud" "$REPO" \
      > "$d/functional-nonroot.log" 2>&1; c=$?
    rep=$(reports "$ud"/rep/*)
    judge "asan-functional-nonroot" "$c" "$d/functional-nonroot.log" "$rep" "$SAN_NONROOT_USER: $(grep -E '^\[(PASS|FAIL|SKIP)' "$d/functional-nonroot.log" | cut -c1-120 | tr '\n' ';')"
  else
    add "asan-functional-nonroot" NOT_RUN 0 "" "fault_domain_readonly po ASan nepaleistas: nustatykite SAN_NONROOT_USER ir kvieskite kaip root"
  fi
  unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS ASAN_SYMBOLIZER_PATH
}

# -------------------------------------------------------------------------------------------------------------- tsan
tsan_env() { mkdir -p "$1"; export TSAN_OPTIONS="exitcode=86:second_deadlock_stack=1:log_path=$1/tsan${SYMB:+:external_symbolizer_path=$SYMB}"; }
stage_tsan() {
  local d="$OUT/tsan" c rep bin tmpb dom snd port=26790 wrap=()
  rm -rf "$d"; mkdir -p "$d"
  # MXLV_ENABLE_SANITIZERS ir MXLV_ENABLE_TSAN kartu — konfigūravimo klaida (viename binare jų būti negali)
  tmpb="$(mktemp -d)"
  cmake -S "$REPO" -B "$tmpb" -G Ninja -DCMAKE_C_COMPILER="$SAN_CC" -DCMAKE_CXX_COMPILER="$SAN_CXX" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_CXX" \
    -DMXL_ROOT="$MXL_SRC/install/$MXL_PRESET" -DMXL_SOURCE_DIR="$MXL_SRC" -DMXLV_ENABLE_SANITIZERS=ON -DMXLV_ENABLE_TSAN=ON > "$d/conflict.log" 2>&1; c=$?
  rm -rf "$tmpb"
  if [[ $c != 0 ]] && grep -q "MXLV_ENABLE_TSAN" "$d/conflict.log"; then add "tsan-asan-conflict" PASS "$c" "$d/conflict.log" "abi parinktys kartu — konfigūravimo klaida"
  else add "tsan-asan-conflict" FAIL "$c" "$d/conflict.log" "abi parinktys kartu turi būti konfigūravimo klaida su paaiškinimu"; fi
  if [[ ${SAN_SKIP_BUILD:-0} != 1 ]]; then
    build_san "$TSB" "$d/build.log" -DMXLV_ENABLE_TSAN=ON; c=$?
    add "tsan-build" "$([[ $c == 0 ]] && echo PASS || echo FAIL)" "$c" "$d/build.log" "$SAN_CXX, įspėjimų $(grep -c 'warning:' "$d/build.log")"
    [[ $c != 0 ]] && return
  fi
  # Kai kurie branduoliai su didele ASLR entropija TSan binarų nepaleidžia („unexpected memory mapping“) — tada be ASLR
  tsan_env "$d/rep-probe"
  if "$TSB/mxlv-timing-unit" --list-tests 2>&1 | grep -q "unexpected memory mapping"; then wrap=(setarch "$(uname -m)" -R); fi
  # 1) vieneto testai be GPU ir glue testai (be [realtime] — tai laiko tvirtinimai)
  for bin in mxlv-unit mxlv-engine-unit mxlv-gfx-unit mxlv-ingest-unit mxlv-media-cpu-unit mxlv-media-unit mxlv-timing-unit mxlv-admission-unit; do
    tsan_env "$d/rep-$bin"
    "${wrap[@]}" "$TSB/$bin" > "$d/$bin.log" 2>&1; c=$?
    rep=$(reports "$d/rep-$bin"/*)
    judge "tsan-$bin" "$c" "$d/$bin.log" "$rep" "$(tail -3 "$d/$bin.log" | tr '\n' ' ' | cut -c1-160) $(report_heads "$d/rep-$bin"/*)"
  done
  tsan_env "$d/rep-glue"
  "${wrap[@]}" "$TSB/vramflow-glue-tests" "~[realtime]" > "$d/vramflow-glue-tests.log" 2>&1; c=$?
  rep=$(reports "$d"/rep-glue/*)
  judge "tsan-vramflow-glue-tests" "$c" "$d/vramflow-glue-tests.log" "$rep" "$(tail -3 "$d/vramflow-glue-tests.log" | tr '\n' ' ' | cut -c1-160) $(report_heads "$d"/rep-glue/*)"
  # 2) funkcijos be GPU: gen → logo → verify (C variantas, host saugykla), 10 s
  tsan_env "$d/rep-func"
  FUNC_BIN="$TSB/vramflow-func" VERIFY_BIN="$TSB/vramflow-verify" FCOMP_VARIANTS=C FFMT_FORMATS=1080i50 FFMT_DURATION=10 VRAMFLOW_GPU_KEEPWARM=0 \
    "${wrap[@]}" python3 "$REPO/tests/functional/functional.py" "$d/functional" --only ffmt_logo > "$d/functional.log" 2>&1; c=$?
  rep=$(reports "$d"/rep-func/*)
  judge "tsan-func-gen-logo-verify-cpu" "$c" "$d/functional.log" "$rep" "$(grep -E '^\[(PASS|FAIL)' "$d/functional.log" | cut -c1-110 | tr '\n' ';') $(report_heads "$d"/rep-func/*)"
  # 3) ingest su programiniu dekoderiu, 20 s (tinklo gija, dekoderio gija, tick variklis)
  if [[ -f $CLIPS/av_1080i50.ts ]]; then
    dom="$(mktemp -d /dev/shm/vf-san-ing-XXXXXX)"
    tsan_env "$d/rep-ingest"
    python3 "$REPO/tests/perf/ts_burst.py" "$CLIPS/av_1080i50.ts" --port "$port" > "$d/ingest-sender.log" 2>&1 & snd=$!
    "${wrap[@]}" "$TSB/vramflow-func" ingest --domain "$dom" --out "$d/ingest" --input "udp://127.0.0.1:$port" --decoder sw --impl cpu \
      --storage host --duration 20 --admission off --input-wait-s 30 > "$d/ingest.log" 2>&1; c=$?
    kill $snd 2>/dev/null; wait $snd 2>/dev/null
    rm -rf "$dom" "$dom.vramflow"
    rep=$(reports "$d"/rep-ingest/*)
    judge "tsan-func-ingest-sw" "$c" "$d/ingest.log" "$rep" "$(tail -1 "$d/ingest.log" | cut -c1-160) $(report_heads "$d"/rep-ingest/*)"
  else
    add "tsan-func-ingest-sw" NOT_RUN 0 "" "nėra $CLIPS/av_1080i50.ts (scripts/linux/62-make-av-clip.sh)"
  fi
  unset TSAN_OPTIONS
}

for s in "${STAGES[@]}"; do "stage_$s"; done
printf '[%s]\n' "$(IFS=,; echo "${results[*]}")" > "$OUT/sanitizers.json"
python3 -c "import json, sys; json.load(open(sys.argv[1], encoding='utf-8'))" "$OUT/sanitizers.json" || { echo "50-sanitizers: sanitizers.json sugadintas" >&2; exit 2; }
echo "SANITIZERS_DONE $OUT (FAIL: $fails)"
exit $(( fails > 0 ))
