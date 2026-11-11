#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# OGraf pavyzdys: RGBA šaltinis -> ograf_feed (libvramflow-gfx, ritmo režimas) -> v210a -> composite virš `gen` -> ograf_probe.
# Naudojimas (WSL):  SOURCE=synthetic|edge  bash examples/ograf/run-demo.sh [išvesties katalogas]
#   SOURCE=synthetic (numatytasis) — įmontuotas sintetinis šaltinis, nieko daugiau nereikia;
#   SOURCE=edge                    — OGraf grafika, atvaizduota jau įdiegtu Windows Edge (render_edge.mjs, Node >= 22).
# Aplinka: VF_BUILD (numatyta /root/build/vramflow/func), MXL (numatyta /root/src/mxl-pr653/install/Linux-Clang-Release),
#   NODE (numatyta /mnt/c/nvm4w/nodejs/node.exe), SECONDS_RUN (numatyta 20).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
VF_BUILD="${VF_BUILD:-/root/build/vramflow/func}"; MXL="${MXL:-/root/src/mxl-pr653/install/Linux-Clang-Release}"
NODE="${NODE:-/mnt/c/nvm4w/nodejs/node.exe}"; T="${SECONDS_RUN:-20}"; SOURCE="${SOURCE:-synthetic}"
OUT="${1:-/tmp/vf-ograf}"; D=/dev/shm/vf-ograf; mkdir -p "$OUT/bin"; rm -rf "$D"; mkdir -p "$D"
LAYER=4d584c47-0005-4000-8000-0000000000ff; COMP=4d584c47-0006-4000-8000-000000000001   # alfa išėjimas (composite --flow-base 1)
trap 'rm -rf "$D"' EXIT

for p in feed probe; do
  clang++-19 -std=c++20 -O2 -I "$REPO/include" -I "$MXL/include" "$HERE/ograf_$p.cpp" -o "$OUT/bin/ograf_$p" \
    -L "$VF_BUILD" -lvramflow-gfx -L "$MXL/lib" -lmxl -Wl,-rpath,"$VF_BUILD" -Wl,-rpath,"$MXL/lib" 2>/dev/null \
    || g++ -std=c++20 -O2 -I "$REPO/include" -I "$MXL/include" "$HERE/ograf_$p.cpp" -o "$OUT/bin/ograf_$p" \
         -L "$VF_BUILD" -lvramflow-gfx -L "$MXL/lib" -lmxl -Wl,-rpath,"$VF_BUILD" -Wl,-rpath,"$MXL/lib"
done
export LD_LIBRARY_PATH="$VF_BUILD:$MXL/lib:${LD_LIBRARY_PATH:-}"
F="$VF_BUILD/vramflow-func"; V="--impl cpu --storage host"

"$F" gen --domain "$D" --flows 2 --duration $((T + 6)) $V --out "$OUT/gen" > "$OUT/gen.log" 2>&1 &
if [ "$SOURCE" = edge ]; then
  ( "$NODE" "$(wslpath -w "$HERE/render_edge.mjs")" --seconds $((T - 2)) | "$OUT/bin/ograf_feed" --domain "$D" --flow-id "$LAYER" \
      --source stdin --format rgba --seconds $((T + 2)) ) > "$OUT/feed.log" 2>&1 &
else
  "$OUT/bin/ograf_feed" --domain "$D" --flow-id "$LAYER" --source synthetic --seconds $((T + 2)) > "$OUT/feed.log" 2>&1 &
fi
sleep 3   # gen ir grafikos flow'ai turi atsirasti (edge: Edge paleidimas ~ kelios sekundės, composite jo palauks)
# A: grafika virš gen bazės -> v210 (programos vaizdas; bazė nepermatoma, todėl key čia nebūtų prasmingas)
"$F" composite --domain "$D" --flows 1 --flow-base 0 --layers 1 --layer-xy 0,0 --opacity 1 --layer-flow "0:$LAYER" --base gen    --out-format v210 --duration $T $V --out "$OUT/comp-v210" > "$OUT/comp-v210.log" 2>&1 &
# B: ta pati grafika be bazės -> v210a (alfa praeina per composite; ją tikrina ograf_probe)
"$F" composite --domain "$D" --flows 1 --flow-base 1 --layers 1 --layer-xy 0,0 --opacity 1 --layer-flow "0:$LAYER" --base none    --out-format v210a --duration $T $V --out "$OUT/comp-v210a" > "$OUT/comp-v210a.log" 2>&1 &
sleep 8
"$OUT/bin/ograf_probe" --domain "$D" --flow-id "$COMP" --out "$OUT" --seconds 6 | tee "$OUT/probe.log"
wait || true
echo "--- feed:"; cat "$OUT/feed.log"; for c in comp-v210 comp-v210a; do echo "--- $c:"; head -c 700 "$OUT/$c"/summary-composite.json 2>/dev/null; echo; done
