#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# GPU prieš CPU palyginimas (architektūra 2026-11-11-gpu-vs-cpu-ir-upstream.md §1): ta pati funkcija trimis variantais
#   C = CPU + host flow, H = GPU + host flow (MXL v1.1 su GPU), G = GPU + cuda-linear (PR #653).
# Variantai leidžiami pakaitomis kiekvienai N pakopai (C, H, G, C, H, G…), kad laiko efektai pasiskirstytų vienodai.
# Naudojimas: 71-compare.sh <out> [funkcija=logo] [N sąrašas]
# composite: LAYERS="1,2,4" (K sąrašas; išorinis ciklas), OUT_FORMAT=v210|v210a, SOURCE=same|fair (C šaltinis),
#   STOP_FAILS (numatyta composite: 2 — variantas K sluoksniams praleidžia didesnes N po tiek PASS_tol FAIL iš eilės;
#   kitoms funkcijoms 0 — nesustoti, kaip anksčiau).
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
if [[ $FUNC == composite ]]; then
  EXTRA="--out-format ${OUT_FORMAT:-v210} --source ${SOURCE:-fair}"
  IFS=',' read -ra KL <<< "${LAYERS:-1,2,4}"
else
  KL=("")
fi
if [[ $FUNC == composite ]]; then STOP_FAILS="${STOP_FAILS:-2}"; else STOP_FAILS="${STOP_FAILS:-0}"; fi
IFS=';' read -ra VL <<< "${VARIANTS:-$DEF_V}"
declare -A VARG=([cpu]="--impl cpu" [gpu-host]="--impl gpu --storage host" [gpu-cuda]="--impl gpu --storage cuda-linear")
declare -A VLET=([cpu]=C [gpu-host]=H [gpu-cuda]=G)
mkdir -p "$OUT"
python3 "$REPO/scripts/wsl/80-clock-probe.py" 30 --json "$OUT/clock-before.json" > /dev/null
echo "laikrodis prieš: $(python3 -c "import json;d=json.load(open('$OUT/clock-before.json'));print(d['realtime_vs_raw_ppm'],'ppm, žingsnių',len(d['steps']))")" | tee -a "$OUT/compare.log"
IFS=, read -ra NL <<< "$NS"
declare -A FAILRUN   # (K,variantas) → iš eilės PASS_tol FAIL
for K in "${KL[@]}"; do
  if [[ -n $K ]]; then TAG="$FUNC-k$K"; KARG="--layers $K"; else TAG="$FUNC"; KARG=""; fi
  for n in "${NL[@]}"; do
    for vk in "${VL[@]}"; do
      v="${VARG[$vk]}"
      key="$K:$vk"
      if [[ $STOP_FAILS -gt 0 && ${FAILRUN[$key]:-0} -ge $STOP_FAILS ]]; then
        echo "[$TAG/${VLET[$vk]} N=$n] praleista: $STOP_FAILS iš eilės PASS_tol FAIL mažesnėse N" | tee -a "$OUT/compare.log"
        echo "{\"tag\":\"$TAG\",\"variant\":\"${VLET[$vk]}\",\"n\":$n,\"reason\":\"stop-after-fails\"}" >> "$OUT/skipped-after-fails.jsonl"
        continue
      fi
      # shellcheck disable=SC2086
      python3 "$REPO/tests/perf/capacity.py" "$OUT" --func "$FUNC" --n "$n" --seconds "$S" --warmup "$W" --every 25 \
        --stop-after-fails 9 $KARG $v $EXTRA 2>&1 | tee -a "$OUT/compare.log" | grep -E "^\["
      # kiekvienas variantas rašo savo capacity-<tag>-<V>.json; pakopos rezultatai kaupiami atskirai
      if ! compgen -G "$OUT/capacity-$TAG-?.json" > /dev/null; then
        echo "KLAIDA: [$TAG/${VLET[$vk]} N=$n] capacity.py nesukūrė rezultato (žr. žurnalą); laikoma FAIL" | tee -a "$OUT/compare.log"
        FAILRUN[$key]=$(( ${FAILRUN[$key]:-0} + 1 ))
      fi
      for f in "$OUT"/capacity-"$TAG"-?.json; do
        [[ -f $f ]] || continue
        # kaupimas: sugadinto .all.json niekada neperrašome, nesėkmės atveju step JSON lieka (garsiai nutraukiame)
        python3 -c "
import json,sys
p=sys.argv[1]; acc=p.replace('.json','.all.json')
try: allr=json.load(open(acc))
except FileNotFoundError: allr=[]
new=json.load(open(p))
if not isinstance(new,list) or not new: raise SystemExit('tuščias/neteisingas pakopos JSON: '+p)
allr+=new; tmp=acc+'.tmp'; json.dump(allr, open(tmp,'w'), indent=1); import os; os.replace(tmp, acc)
r=new[-1]
# 0 — praleista pagal VRAM (nelaikoma FAIL), 1 — PASS_tol, 2 — FAIL
print(0 if r.get('skipped') else (1 if r.get('pass_tol') else 2))" "$f" > "$OUT/.last"
        if [[ $? -ne 0 ]]; then
          echo "KLAIDA: nepavyko sukaupti $f į ${f%.json}.all.json (abu failai palikti); kampanija nutraukta" | tee -a "$OUT/compare.log"
          exit 1
        fi
        rc=$(cat "$OUT/.last")
        if [[ $rc == 2 ]]; then FAILRUN[$key]=$(( ${FAILRUN[$key]:-0} + 1 )); elif [[ $rc == 1 ]]; then FAILRUN[$key]=0; fi
        rm -f "$f"
      done
    done
  done
done
python3 "$REPO/scripts/wsl/80-clock-probe.py" 30 --json "$OUT/clock-after.json" > /dev/null
echo "laikrodis po: $(python3 -c "import json;d=json.load(open('$OUT/clock-after.json'));print(d['realtime_vs_raw_ppm'],'ppm, žingsnių',len(d['steps']))")" | tee -a "$OUT/compare.log"
echo "PALYGINIMAS BAIGTAS $OUT" | tee -a "$OUT/compare.log"
