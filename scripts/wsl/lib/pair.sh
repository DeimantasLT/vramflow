#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Paleidžia vieną writer + reader porą atskiruose procesuose ir parašo trumpą santrauką.
# Naudojimas: pair.sh <out_dir> <reader_seconds> "<writer papildomi arg>" "<reader papildomi arg>"
# Aplinka: VERIFY_BIN, MXL_LIB, DOMAIN (numatyta /dev/shm/mxlv-<atsitiktinis>), HISTORY_MS
set -uo pipefail
OUT="$1"; DUR="$2"; WARGS="${3:-}"; RARGS="${4:-}"
VERIFY_BIN="${VERIFY_BIN:-$HOME/build/vramflow/release/vramflow-verify}"
export LD_LIBRARY_PATH="${MXL_LIB:-$HOME/src/mxl-pr653/install/Linux-Clang-Release/lib}"
DOMAIN="${DOMAIN:-$(mktemp -d /dev/shm/mxlv-XXXXXX)}"
mkdir -p "$OUT" "$DOMAIN"
if [[ -n "${HISTORY_MS:-}" ]]; then
  echo "{\"urn:x-mxl:option:history_duration/v1.0\": $((HISTORY_MS * 1000000))}" > "$DOMAIN/options.json"
fi
# shellcheck disable=SC2086
"$VERIFY_BIN" writer --domain "$DOMAIN" --duration $((DUR + 6)) --out "$OUT" $WARGS 2> "$OUT/writer.err" &
wp=$!
sleep 2
# shellcheck disable=SC2086
"$VERIFY_BIN" reader --domain "$DOMAIN" --duration "$DUR" --out "$OUT" $RARGS 2> "$OUT/reader.err"
rcode=$?
kill -TERM "$wp" 2>/dev/null
wait "$wp"; wcode=$?
echo "$wcode" > "$OUT/writer.exit"; echo "$rcode" > "$OUT/reader.exit"
rm -rf "$DOMAIN"
echo "writer_exit=$wcode reader_exit=$rcode out=$OUT"
