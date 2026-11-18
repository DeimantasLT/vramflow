#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# keepwarm --event-ms (planas production-ready §3.1, GPU įvykių liudytojas izoliacijos vartams): kiekvienas užpildymas,
# trukęs ilgiau nei --event-ms, — viena eilutė „[keepwarm] GPU įvykis: TAI <ns> trukmė <ms> ms“ (CLOCK_TAI užpildymo pradžia).
# --event-ms 0 — kiekvienas užpildymas; be parinkties — jokių įvykio eilučių (elgesys nepakitęs). Be GPU — SKIP (77).
#   test_keepwarm_events.sh <vramflow-func>
set -u
BIN="$1"
out="$("$BIN" keepwarm --duration 0.3 --event-ms 0 2>&1)"; rc=$?
if [[ $rc != 0 ]] && grep -q "CUDA įrenginys nerastas\|CUDA:" <<<"$out"; then echo "SKIP: nėra GPU"; exit 77; fi
[[ $rc == 0 ]] || { echo "FAIL: exit $rc: $out"; exit 1; }
n=$(grep -cE '^\[keepwarm\] GPU įvykis: TAI [0-9]{19} trukmė [0-9]+\.[0-9]{3} ms$' <<<"$out")
fills=$(grep -oE 'baigta: [0-9]+ užpildymų' <<<"$out" | grep -oE '[0-9]+')
[[ -n $fills && $n == "$fills" && $n -ge 10 ]] || { echo "FAIL: įvykio eilučių $n, užpildymų ${fills:-?} (laukta lygu, ≥ 10)"; exit 1; }
# TAI — tos pačios eros kaip mxlGetTime (≈ dabar), didėjantis
now=$(python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_TAI))')
first=$(grep -oE 'TAI [0-9]+' <<<"$out" | head -1 | cut -d' ' -f2)
last=$(grep -oE 'TAI [0-9]+' <<<"$out" | tail -1 | cut -d' ' -f2)
(( now - first < 10000000000 && last > first )) || { echo "FAIL: TAI ne CLOCK_TAI arba nedidėja ($first … $last, dabar $now)"; exit 1; }
quiet="$("$BIN" keepwarm --duration 0.2 2>&1)"
grep -q "GPU įvykis" <<<"$quiet" && { echo "FAIL: be --event-ms išvesta įvykio eilutė"; exit 1; }
echo "PASS: $n įvykio eilučių = užpildymų; TAI didėja; be parinkties — tylu"
