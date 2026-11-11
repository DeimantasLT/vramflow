#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# GPU prieš CPU palyginimas (architektūra 2026-11-11-gpu-vs-cpu-ir-upstream.md §1): ta pati funkcija trimis variantais
#   C = CPU + host flow, H = GPU + host flow (MXL v1.1 su GPU), G = GPU + cuda-linear (PR #653).
# Variantai leidžiami pakaitomis kiekvienai N pakopai (C, H, G, C, H, G…), kad laiko efektai pasiskirstytų vienodai.
# Naudojimas: 71-compare.sh <out> [funkcija=logo] [N sąrašas]
source "$(dirname "$0")/env.sh"
set +e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/evidence/compare-$(date -u +%Y%m%dT%H%M%SZ)}"
FUNC="${2:-logo}"; NS="${3:-1,2,4,8,16,32,48,64}"
S="${SECONDS_PER_LEVEL:-60}"; W="${WARMUP:-10}"
# dec/chain: H variantas neįgyvendintas (žr. ataskaitą) — numatytai tik C ir G; klipas be -stream_loop (≥ S+W+20 s)
if [[ $FUNC == dec || $FUNC == chain ]]; then
  DEF_V="cpu;gpu-cuda"; EXTRA="--clip ${CLIP:-$HOME/media-test/clip.ts} --ref ${REF:-$HOME/media-test/ref_crc.json}"
else
  DEF_V="cpu;gpu-host;gpu-cuda"; EXTRA=""
fi
IFS=';' read -ra VL <<< "${VARIANTS:-$DEF_V}"
declare -A VARG=([cpu]="--impl cpu" [gpu-host]="--impl gpu --storage host" [gpu-cuda]="--impl gpu --storage cuda-linear")
mkdir -p "$OUT"
python3 "$REPO/scripts/wsl/80-clock-probe.py" 30 --json "$OUT/clock-before.json" > /dev/null
echo "laikrodis prieš: $(python3 -c "import json;d=json.load(open('$OUT/clock-before.json'));print(d['realtime_vs_raw_ppm'],'ppm, žingsnių',len(d['steps']))")" | tee -a "$OUT/compare.log"
IFS=, read -ra NL <<< "$NS"
for n in "${NL[@]}"; do
  for vk in "${VL[@]}"; do
    v="${VARG[$vk]}"
    # shellcheck disable=SC2086
    python3 "$REPO/tests/perf/capacity.py" "$OUT" --func "$FUNC" --n "$n" --seconds "$S" --warmup "$W" --every 25 \
      --stop-after-fails 9 $v $EXTRA 2>&1 | tee -a "$OUT/compare.log" | grep -E "^\["
    # kiekvienas variantas rašo savo capacity-<func>-<V>.json; pakopos rezultatai kaupiami atskirai
    for f in "$OUT"/capacity-"$FUNC"-?.json; do
      [[ -f $f ]] && python3 -c "
import json,sys
p=sys.argv[1]; acc=p.replace('.json','.all.json')
try: allr=json.load(open(acc))
except Exception: allr=[]
allr+= json.load(open(p)); json.dump(allr, open(acc,'w'), indent=1)" "$f" && rm -f "$f"
    done
  done
done
python3 "$REPO/scripts/wsl/80-clock-probe.py" 30 --json "$OUT/clock-after.json" > /dev/null
echo "laikrodis po: $(python3 -c "import json;d=json.load(open('$OUT/clock-after.json'));print(d['realtime_vs_raw_ppm'],'ppm, žingsnių',len(d['steps']))")" | tee -a "$OUT/compare.log"
echo "PALYGINIMAS BAIGTAS $OUT" | tee -a "$OUT/compare.log"
