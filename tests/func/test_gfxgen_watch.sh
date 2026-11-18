#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# gfxgen pace commit'ų stebėtojas (planas production-ready §3.1, fsplit sluoksnio nebuvimo apskaita): indeksas, kurio grain'o
# nėra (arba jis INVALID) ties T(i) + skelbiama commit'o riba, — eilutė „[gfxgen] vėlus commit: flow N/K indeksas I“ (atėjo
# vėliau) arba „[gfxgen] praleista: flow N/K indeksas I“ (neatėjo); santraukoje watch_late / watch_missed.
#   test_gfxgen_watch.sh <vramflow-func> <darbo katalogas>
# Atvejai: (1) --test-no-submit (ritmas be kadro — tik INVALID grain'ai) → kiekvienas stebėtas indeksas „praleista“;
#          (2) įprastas pace → nė vienos eilutės, watch_* nėra.
set -u
BIN="$1"; W="$2"
rm -rf "$W"; mkdir -p "$W/d1" "$W/d2" "$W/o1" "$W/o2"
common=(--flows 2 --layers 1 --layer-geom logo --mode pace --duration 2 --impl cpu --storage host --admission off)
e1="$("$BIN" gfxgen --domain "$W/d1" --out "$W/o1" "${common[@]}" --test-no-submit 2>&1)"
n_miss=$(grep -cE '^\[gfxgen\] praleista: flow [0-9]+/[0-9]+ indeksas [0-9]+$' <<<"$e1")
n_late=$(grep -cE '^\[gfxgen\] vėlus commit:' <<<"$e1")
sum=$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sum(f.get("watch_missed",0) for f in s["flows"]))' "$W/o1/summary-gfxgen.json" 2>/dev/null)
# 2 s × 50 laukų × 2 flow'ai ≈ 200; stebėtojas pradeda nuo antro indekso ir baigia prieš ritmo sustabdymą
[[ $n_miss -ge 150 && $n_late == 0 && "$sum" == "$n_miss" ]] || { echo "FAIL (1): praleista $n_miss, vėlus $n_late, watch_missed ${sum:-?}"; echo "$e1" | tail -5; exit 1; }
e2="$("$BIN" gfxgen --domain "$W/d2" --out "$W/o2" "${common[@]}" 2>&1)"
grep -qE '^\[gfxgen\] (praleista|vėlus commit):' <<<"$e2" && { echo "FAIL (2): įprastame pace yra stebėtojo eilučių"; grep -E 'praleista|vėlus' <<<"$e2" | head -3; exit 1; }
keys=$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sum(("watch_missed" in f) + ("watch_late" in f) for f in s["flows"]))' "$W/o2/summary-gfxgen.json")
[[ $keys == 0 ]] || { echo "FAIL (2): watch_* raktai santraukoje be įvykių"; exit 1; }
rm -rf "$W"
echo "PASS: no-submit — $n_miss praleista eilučių = watch_missed; įprastas pace — tylu"
