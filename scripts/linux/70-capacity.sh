#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# GPU media funkcijų talpos kampanija (architektūra GPU funkcijų §8): logo, NVENC, x264, NVDEC, grandinė.
# Prieš ir po — laikrodžio zondas (80-clock-probe.py): rezultatai galioja tik su tinkamu laikrodžiu (0 žingsnių, ≤ 50 ppm).
# Rezultatai: $OUT/capacity-<func>.json + kiekvienos pakopos katalogas; $OUT/clock-{before,after}.json.
# Pakopų rinkinys: PROFILE=full (numatyta) | quick
source "$(dirname "$0")/env.sh"
set +e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/evidence/capacity-$(date -u +%Y%m%dT%H%M%SZ)}"
S="${SECONDS_PER_LEVEL:-60}"; W="${WARMUP:-10}"; PROFILE="${PROFILE:-full}"
CLIP="${CLIP:-$HOME/media-test/clip.ts}"; REF="${REF:-$HOME/media-test/ref_crc.json}"
mkdir -p "$OUT"
C=(python3 "$REPO/tests/perf/capacity.py" "$OUT" --seconds "$S" --warmup "$W")  # masyvas: kelyje yra tarpas
run() { echo "[$(date -u +%T)] ▶ $*" | tee -a "$OUT/campaign.log"; "${C[@]}" "$@" 2>&1 | tee -a "$OUT/campaign.log" | grep -E "^\[|N\*"; }
python3 "$REPO/scripts/linux/80-clock-probe.py" 30 --json "$OUT/clock-before.json" > /dev/null
echo "laikrodis prieš: $(python3 -c "import json;d=json.load(open('$OUT/clock-before.json'));print(d['realtime_vs_raw_ppm'],'ppm, žingsnių',len(d['steps']))")" | tee -a "$OUT/campaign.log"
if [[ $PROFILE == quick ]]; then
  run --func logo              --n 1,8,32,64 --every 25 --stop-after-fails 9
  run --func enc-nvenc-ll      --n 1,6,12 --stop-after-fails 9
  run --func enc-x264-superfast --n 2,4,6 --stop-after-fails 9
  run --func dec               --n 1,8,16 --every 10 --clip "$CLIP" --ref "$REF" --stop-after-fails 9
  run --func chain             --n 1,4,8 --clip "$CLIP" --stop-after-fails 9
else
  run --func logo              --n 1,2,4,8,16,32,48,64 --every 25
  run --func enc-nvenc-ll      --n 1,2,4,6,8,10,12,13
  run --func enc-nvenc-hq      --n 4,8,12
  run --func enc-x264-veryfast --n 1,2,3,4,5,6,8
  run --func enc-x264-superfast --n 4,6,8,10
  run --func dec               --n 1,2,4,8,12,16,24,32 --every 10 --clip "$CLIP" --ref "$REF"
  run --func chain             --n 1,2,4,6,8,10,12 --clip "$CLIP"
fi
python3 "$REPO/scripts/linux/80-clock-probe.py" 30 --json "$OUT/clock-after.json" > /dev/null
echo "laikrodis po: $(python3 -c "import json;d=json.load(open('$OUT/clock-after.json'));print(d['realtime_vs_raw_ppm'],'ppm, žingsnių',len(d['steps']))")" | tee -a "$OUT/campaign.log"
echo "[$(date -u +%T)] KAMPANIJA BAIGTA $OUT" | tee -a "$OUT/campaign.log"
