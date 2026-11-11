#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# F1/F2: build artefaktų patikra + upstream ctest (NUOSEKLIAI — testai dalijasi /dev/shm domain'u).
source "$(dirname "$0")/env.sh"
OUT="${EVIDENCE:-$HOME/evidence/latest}/20-upstream"
mkdir -p "$OUT"
cd "$MXL_BUILD"

{
  echo "## Artefaktai"
  plugin="$(find . -name 'libmxl-payload-cuda-linear.so*' -type f | head -1)"
  libmxl="$(find . -name 'libmxl.so*' -type f | head -1)"
  echo "plugin=$plugin"
  echo "libmxl=$libmxl"
  echo "## ldd libmxl | cuda (turi būti tuščia)"
  ldd "$libmxl" | grep -i cuda || echo "(nėra CUDA priklausomybių)"
  echo "## ldd plugin | cudart (turi būti tuščia — statinis runtime)"
  ldd "$plugin" | grep -i cudart || echo "(nėra libcudart.so)"
  echo "## ldd plugin (visas)"
  ldd "$plugin"
} > "$OUT/artifacts.txt" 2>&1
cat "$OUT/artifacts.txt"

ctest -N > "$OUT/ctest-list.txt"
echo "Testų: $(grep -c 'Test *#' "$OUT/ctest-list.txt")"

# root nepaiso failų leidimų, todėl leidimų testai (unwritable/unreadable) kaip root krenta netikrai.
# Testus visada leidžiam kaip neprivilegijuotas TEST_USER.
TEST_USER="${TEST_USER:-mxltest}"
run_as=()
if [[ $EUID -eq 0 ]]; then
  id "$TEST_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$TEST_USER"
  chmod 755 "$HOME"
  chown -R "$TEST_USER:$TEST_USER" "$MXL_SRC" "$OUT"
  run_as=(runuser -u "$TEST_USER" --)
fi
echo "ctest naudotojas: ${run_as[*]:-$(id -un)}" | tee "$OUT/ctest-user.txt"

# "Reader in another container" (examples/cuda-cross-container-test.sh) kviečia `sudo docker info` be -n:
# ne root naudotojui be docker'io sudo laukia slaptažodžio ir testas kabo iki timeout (PR #653 radinys).
# Todėl jį leidžiam atskirai kaip root su trumpu timeout ir užrašom rezultatą (be docker'io → Skipped).
CONTAINER_RE="Reader in another container"
set +e
"${run_as[@]}" ctest --output-on-failure -j1 --timeout 600 -E "$CONTAINER_RE" \
  --output-junit "$OUT/ctest-junit.xml" > "$OUT/ctest.log" 2>&1
code=$?
ctest --output-on-failure -j1 --timeout 60 -R "$CONTAINER_RE" > "$OUT/ctest-container-as-root.log" 2>&1
echo "container testas (root): $(grep -oE 'Passed|\*\*\*Skipped|\*\*\*Failed|\*\*\*Timeout' "$OUT/ctest-container-as-root.log" | head -1)" \
  | tee "$OUT/container.txt"
set -e
tail -25 "$OUT/ctest.log"
# Atskirai suskaičiuojam SKIP (CUDA testai be GPU praleidžiami — tai ne PASS)
grep -E "Skipped|\*\*\*Skipped|Not Run" "$OUT/ctest.log" | head -20 > "$OUT/skipped.txt" || true
echo "SKIPPED_LINES=$(wc -l < "$OUT/skipped.txt")"
echo "CTEST_EXIT=$code"
exit "$code"
