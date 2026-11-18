#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# verify --mode composite --layer-dropped-ok (planas production-ready §3.1, fsplit tikslus apskaitymas): laukas, kuris nesutampa
# su etalonu, bet TIKSLIAI sutampa su etalonu be kurio nors sluoksnio, skaičiuojamas layer_dropped_fields (ne mismatch) ir
# išvedamas „[verify] layer_dropped: flow N indeksas I sluoksniai 0xM“ — harness'as jį sieja su composite layer_absent ir
# aukštupio vėlavimu. Be parinkties — elgesys nepakitęs (mismatch).
# Scenarijus: gen + composite su 1 sluoksniu, bet BE gfxgen → composite visą laiką be sluoksnio (skaidrus).
#   test_verify_layer_dropped.sh <vramflow-func> <darbo katalogas>
set -u
BIN="$1"; W="$2"
rm -rf "$W"; mkdir -p "$W/d" "$W/g" "$W/c" "$W/v0" "$W/v1"
cpu=(--impl cpu --storage host --admission off)
xy=(--layer-xy 1488,64 --opacity 1.0 --base gen --out-format v210)
"$BIN" gen --domain "$W/d" --out "$W/g" --flows 1 --duration 12 "${cpu[@]}" 2>"$W/g.err" &
pg=$!
sleep 1
"$BIN" composite --domain "$W/d" --out "$W/c" --flows 1 --layers 1 "${xy[@]}" --duration 9 --warmup 1 "${cpu[@]}" 2>"$W/c.err" &
pc=$!
sleep 2
vargs=(--mode composite --flows 1 --layers 1 --layer-geom logo --gfx-mode pace "${xy[@]}" --every 5 --duration 4 --warmup 1 --impl cpu)
"$BIN" verify --domain "$W/d" --out "$W/v0" "${vargs[@]}" 2>"$W/v0.err" &
p0=$!
"$BIN" verify --domain "$W/d" --out "$W/v1" "${vargs[@]}" --layer-dropped-ok 2>"$W/v1.err"
r1=$?
wait $p0; r0=$?
wait $pc $pg
get() { python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["flows"][0].get(sys.argv[2], 0))' "$1" "$2" 2>/dev/null || echo "?"; }
m0=$(get "$W/v0/summary-verify-composite.json" mismatch_fields); s0=$(get "$W/v0/summary-verify-composite.json" samples)
m1=$(get "$W/v1/summary-verify-composite.json" mismatch_fields); s1=$(get "$W/v1/summary-verify-composite.json" samples)
d1=$(get "$W/v1/summary-verify-composite.json" layer_dropped_fields)
l1=$(grep -cE '^\[verify\] layer_dropped: flow 0 indeksas [0-9]+ sluoksniai 0x1$' "$W/v1.err")
echo "be parinkties: exit $r0 samples $s0 mismatch $m0 | su parinktimi: exit $r1 samples $s1 mismatch $m1 layer_dropped $d1 eilučių $l1"
[[ $s0 != "?" && $s0 -gt 0 && $m0 == "$s0" ]] || { echo "FAIL: be parinkties visi laukai turi būti mismatch"; exit 1; }
[[ $s1 != "?" && $s1 -gt 0 && $m1 == 0 && $d1 == "$s1" && $l1 == "$d1" && $r1 == 0 ]] ||
  { echo "FAIL: su --layer-dropped-ok laukta mismatch 0, layer_dropped = samples = eilučių, exit 0"; tail -3 "$W/v1.err"; exit 1; }
rm -rf "$W"
echo PASS
