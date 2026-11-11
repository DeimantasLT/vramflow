#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Visa MXL PR #653 GPU kelio patikra viena komanda (architektūra §8).
# Naudojimas: run-all.sh [--soak 2h|24h|skip] [--scale-seconds 120] [--soak-n 16]
# Įrodymai: ~/evidence/<UTC laikas>/ (+ kopija į repo evidence/ pabaigoje), ataskaita report.md.
source "$(dirname "$0")/env.sh"
set +e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SMI="$(find_nvidia_smi)" || { echo "nvidia-smi nerastas" >&2; exit 1; }
SOAK=2h; SCALE_S=120; SOAK_N=16; SCALE_N="1,4,8,16,32"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --soak) SOAK="$2"; shift 2;;
    --scale-seconds) SCALE_S="$2"; shift 2;;
    --soak-n) SOAK_N="$2"; shift 2;;
    --scale-n) SCALE_N="$2"; shift 2;;
    *) echo "nežinomas argumentas $1"; exit 1;;
  esac
done
export EVIDENCE="$HOME/evidence/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EVIDENCE"
ln -sfn "$EVIDENCE" "$HOME/evidence/latest"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$EVIDENCE/run-all.log"; }
stage() { local name="$1"; shift; log "▶ $name"; local t0=$SECONDS; "$@" > "$EVIDENCE/$name.log" 2>&1; local c=$?;
          echo "{\"stage\":\"$name\",\"exit\":$c,\"seconds\":$((SECONDS - t0))}" >> "$EVIDENCE/stages.jsonl"; log "  $name exit=$c ($((SECONDS - t0)) s)"; return $c; }

# Manifestas: versijos ir aplinka
{
  echo "{"
  echo "\"mxl_sha\":\"$(git -C "$MXL_SRC" rev-parse HEAD)\","
  echo "\"verify_sha\":\"$(git -C "$REPO" rev-parse HEAD 2>/dev/null)\","
  echo "\"verify_dirty\":$(git -C "$REPO" status --porcelain 2>/dev/null | grep -q . && echo true || echo false),"
  echo "\"nvcc\":\"$(nvcc --version | tail -1)\","
  echo "\"driver\":\"$("$SMI" --query-gpu=driver_version --format=csv,noheader)\","
  echo "\"gpu\":\"$("$SMI" --query-gpu=name,memory.total --format=csv,noheader)\","
  echo "\"kernel\":\"$(uname -r)\","
  echo "\"os\":\"$(. /etc/os-release; echo "$PRETTY_NAME")\","
  echo "\"cpu\":\"$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\","
  echo "\"nproc\":$(nproc),"
  echo "\"mem_gb\":$(free -g | awk '/Mem:/{print $2}'),"
  echo "\"shm\":\"$(df -h /dev/shm | tail -1 | awk '{print $2}')\","
  echo "\"tai_offset_s\":$(python3 -c 'import time;print(round(time.clock_gettime(time.CLOCK_TAI)-time.time()))'),"
  echo "\"started_utc\":\"$(date -u +%FT%TZ)\""
  echo "}"
} > "$EVIDENCE/manifest.json"

stage 00-probe        bash "$REPO/scripts/linux/00-cuda-fd-probe.sh"
stage 20-upstream     bash "$REPO/scripts/linux/20-upstream-tests.sh"
stage 30-build        bash "$REPO/scripts/linux/30-build-verify.sh"
stage 31-unit         "$HOME/build/vramflow/release/mxlv-unit" -r junit -o "$EVIDENCE/unit-junit.xml"
stage 32-gpu-tests    "$HOME/build/vramflow/release/mxlv-gpu-tests" -r junit -o "$EVIDENCE/gpu-junit.xml"
stage 40-selftest-cuda python3 "$REPO/tests/selftest/selftest.py" "$EVIDENCE/40-selftest" --storage cuda-linear
c1=$?
stage 41-selftest-host python3 "$REPO/tests/selftest/selftest.py" "$EVIDENCE/40-selftest" --storage host
c2=$?
if [[ $c1 != 0 || $c2 != 0 ]]; then
  log "STOP: savitikra nepraėjo — tolesni „0 klaidų“ rezultatai negaliotų (architektūra §5.1)"
  python3 "$REPO/scripts/report.py" "$EVIDENCE"; exit 3
fi
stage 45-functional   python3 "$REPO/tests/functional/functional.py" "$EVIDENCE/45-functional"
stage 50-sanitizers   bash "$REPO/scripts/linux/50-sanitizers.sh"
stage 60-scale-cuda   python3 "$REPO/tests/perf/scale.py" "$EVIDENCE/60-scale" --storage cuda-linear --n "$SCALE_N" --seconds "$SCALE_S"
stage 61-scale-host   python3 "$REPO/tests/perf/scale.py" "$EVIDENCE/60-scale" --storage host --n "$SCALE_N" --seconds "$SCALE_S"
stage 62-repeat       python3 "$REPO/tests/perf/scale.py" "$EVIDENCE/62-repeat" --storage cuda-linear --n "$SOAK_N,$SOAK_N,$SOAK_N" --seconds "$SCALE_S" --label repeat
case "$SOAK" in
  2h)  stage 70-soak python3 "$REPO/tests/perf/scale.py" "$EVIDENCE/70-soak" --storage cuda-linear --n "$SOAK_N" --seconds 7200 --warmup 300 --label soak;;
  24h) stage 70-soak python3 "$REPO/tests/perf/scale.py" "$EVIDENCE/70-soak" --storage cuda-linear --n "$SOAK_N" --seconds 86400 --warmup 300 --label soak;;
  *)   log "soak praleistas (--soak skip)";;
esac
python3 "$REPO/scripts/report.py" "$EVIDENCE"
mkdir -p "$REPO/evidence"
dest="$REPO/evidence/$(basename "$EVIDENCE")"
rsync -a --exclude '*.mxl-flow' "$EVIDENCE/" "$dest/" 2>/dev/null || cp -r "$EVIDENCE" "$dest"
log "BAIGTA. Ataskaita: $dest/report.md"
