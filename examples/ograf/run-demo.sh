#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# OGraf pavyzdys: RGBA šaltinis -> ograf_feed (libvramflow-gfx, ritmo režimas) -> v210a -> composite virš `gen` -> ograf_probe.
# Naudojimas (WSL):  SOURCE=synthetic|edge  bash examples/ograf/run-demo.sh [išvesties katalogas]
#   SOURCE=synthetic (numatytasis) — įmontuotas sintetinis šaltinis, nieko daugiau nereikia;
#   SOURCE=edge                    — OGraf grafika, atvaizduota jau įdiegtu Windows Edge (render_edge.mjs, Node >= 22).
# Eiga: gen -> grafikos šaltinis (feed) -> laukiama pirmo grafikos grain'o commit'o -> 2 composite (T s) -> probe ->
#   feed ir gen sustabdomi tik po composite pabaigos (kad grafikos grain'ai būtų iki paskutinio composite lauko).
# Aplinka: VF_BUILD (numatyta /root/build/vramflow/func), MXL (numatyta /root/src/mxl-pr653/install/Linux-Clang-Release),
#   NODE (numatyta /mnt/c/nvm4w/nodejs/node.exe), SECONDS_RUN (numatyta 20), CXX (numatyta clang++-19).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
VF_BUILD="${VF_BUILD:-/root/build/vramflow/func}"; MXL="${MXL:-/root/src/mxl-pr653/install/Linux-Clang-Release}"
NODE="${NODE:-/mnt/c/nvm4w/nodejs/node.exe}"; T="${SECONDS_RUN:-20}"; SOURCE="${SOURCE:-synthetic}"; CXX="${CXX:-clang++-19}"
OUT="${1:-/tmp/vf-ograf}"; D=$(mktemp -d /dev/shm/vf-ograf-XXXXXXXX); mkdir -p "$OUT/bin"; rm -rf "$OUT/feed.ready" "$OUT/stop.fifo"; mkfifo "$OUT/stop.fifo"; sleep 100000 > "$OUT/stop.fifo" & HOLD=$!   # HOLD laiko fifo rašymo galą; jį nužudžius renderis gauna stdin EOF
LAYER=4d584c47-0005-4000-8000-0000000000ff; COMP=4d584c47-0006-4000-8000-000000000001   # alfa išėjimas (composite --flow-base 1)
# Nutraukimo / klaidos atveju: visi fono procesai (gen, feed, renderis, composite) nutraukiami, domenas išvalomas.
stop_bg() {
  kill "$HOLD" 2>/dev/null || true   # fifo EOF: renderis (Edge) pats baigiasi ir išvalo savo profilį   # feed (SIGTERM -> graceful), gen, composite; renderis baigiasi pats (EPIPE, kai feed užsidaro)
  pkill -TERM -f "bin/ograf_feed --domain $D" 2>/dev/null || true
  pkill -TERM -f "vramflow-func (gen|composite) --domain $D" 2>/dev/null || true
  wait 2>/dev/null || true
}
# Unikalus domenas (planas §P6): du demo paleidimai vienas kito netrina; šalinamas kartu su nuomos katalogu <D>.vramflow
cleanup() { trap - EXIT; stop_bg; rm -rf "$D" "$D.vramflow"; }
trap cleanup EXIT; trap 'exit 130' INT TERM

for p in feed probe; do   # kompiliatoriaus klaidos rodomos (set -e nutraukia)
  "$CXX" -std=c++20 -O2 -I "$REPO/include" -I "$MXL/include" "$HERE/ograf_$p.cpp" -o "$OUT/bin/ograf_$p" \
    -L "$VF_BUILD" -lvramflow-gfx -L "$MXL/lib" -lmxl -Wl,-rpath,"$VF_BUILD" -Wl,-rpath,"$MXL/lib"
done
export LD_LIBRARY_PATH="$VF_BUILD:$MXL/lib:${LD_LIBRARY_PATH:-}"
F="$VF_BUILD/vramflow-func"; V="--impl cpu --storage host"

"$F" gen --domain "$D" --flows 2 --duration $((T + 120)) $V --out "$OUT/gen" > "$OUT/gen.log" 2>&1 &
FEED=("$OUT/bin/ograf_feed" --domain "$D" --flow-id "$LAYER" --ready-file "$OUT/feed.ready" --seconds $((T + 110)))
if [ "$SOURCE" = edge ]; then
  { "$NODE" "$(wslpath -w "$HERE/render_edge.mjs")" --seconds $((T + 100)) --stop-on-stdin-eof < "$OUT/stop.fifo" | "${FEED[@]}" --source stdin --format rgba; } > "$OUT/feed.log" 2>&1 &
else
  "${FEED[@]}" --source synthetic > "$OUT/feed.log" 2>&1 &
fi
for i in $(seq 1 600); do [ -e "$OUT/feed.ready" ] && break; sleep 0.1; done   # pirmas grafikos grain'as commit'intas (Edge: kelios s)
[ -e "$OUT/feed.ready" ] || { echo "grafikos šaltinis nepasiruošė per 60 s"; cat "$OUT/feed.log"; exit 1; }
# A: grafika virš gen bazės -> v210 (programos vaizdas; bazė nepermatoma, todėl key čia nebūtų prasmingas)
"$F" composite --domain "$D" --flows 1 --flow-base 0 --layers 1 --layer-xy 0,0 --opacity 1 --layer-flow "0:$LAYER" --base gen \
  --out-format v210 --duration $T $V --out "$OUT/comp-v210" > "$OUT/comp-v210.log" 2>&1 & PA=$!
# B: ta pati grafika be bazės -> v210a (alfa praeina per composite; ją tikrina ograf_probe)
"$F" composite --domain "$D" --flows 1 --flow-base 1 --layers 1 --layer-xy 0,0 --opacity 1 --layer-flow "0:$LAYER" --base none \
  --out-format v210a --duration $T $V --out "$OUT/comp-v210a" > "$OUT/comp-v210a.log" 2>&1 & PB=$!
sleep 6
"$OUT/bin/ograf_probe" --domain "$D" --flow-id "$COMP" --out "$OUT" --seconds 6 | tee "$OUT/probe.log"
RA=0; RB=0; wait $PA || RA=$?; wait $PB || RB=$?
stop_bg   # tik dabar: feed (ir renderis) ir gen
echo "composite exit codes: A(v210)=$RA B(v210a)=$RB"
echo "--- feed:"; cat "$OUT/feed.log"
for c in comp-v210 comp-v210a; do echo "--- $c:"; head -c 700 "$OUT/$c"/summary-composite.json 2>/dev/null; echo; done
[ "$RA" = 0 ] && [ "$RB" = 0 ]
