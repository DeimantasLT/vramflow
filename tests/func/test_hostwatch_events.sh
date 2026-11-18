#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# hostwatch --event-ms (planas production-ready §3.1, sprendimas (b) išplėstas 2026-11-18): host'o planuoklės / VM pauzių
# liudytojas izoliacijos vartams — ir C variantui (be GPU). Proceso pauzė (čia SIGSTOP 80 ms) — viena eilutė
# „[hostwatch] host įvykis: TAI <ns> trukmė <ms> ms“: TAI — paskutinis pabudimas prieš pauzę, trukmė — tarpas iki kito pabudimo.
# Be --event-ms — jokių įvykio eilučių; pabaigoje — santraukos eilutė.
#   test_hostwatch_events.sh <vramflow-func>
set -u
BIN="$1"
tai() { python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_TAI))'; }
err=$(mktemp); trap 'rm -f "$err"' EXIT
"$BIN" hostwatch --duration 1.5 --event-ms 5 2>"$err" &
pid=$!
sleep 0.5
t0=$(tai)
kill -STOP "$pid"; sleep 0.08; kill -CONT "$pid"
wait "$pid"; rc=$?
out=$(cat "$err")
[[ $rc == 0 ]] || { echo "FAIL: exit $rc: $out"; exit 1; }
grep -qE '^\[hostwatch\] baigta: [0-9]+ pabudimų' <<<"$out" || { echo "FAIL: nėra santraukos: $out"; exit 1; }
ev=$(grep -E '^\[hostwatch\] host įvykis: TAI [0-9]{19} trukmė [0-9]+\.[0-9]{3} ms$' <<<"$out" | awk '{print $5, $7}' |
     awk -v t0="$t0" '$2 >= 70 && $1 >= t0 - 5000000 && $1 <= t0 + 50000000')
[[ $(wc -l <<<"$ev") == 1 && -n $ev ]] || { echo "FAIL: laukta 1 įvykio ≥ 70 ms ties SIGSTOP (t0 $t0): $out"; exit 1; }
quiet="$("$BIN" hostwatch --duration 0.2 2>&1)"
grep -q "host įvykis" <<<"$quiet" && { echo "FAIL: be --event-ms išvesta įvykio eilutė"; exit 1; }
echo "PASS: SIGSTOP 80 ms → įvykis $ev; be parinkties — tylu"
