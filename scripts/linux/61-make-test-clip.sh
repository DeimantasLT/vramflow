#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Bandomasis H.264 1080i50 klipas F-DEC testams (architektūra GPU funkcijų §7.3):
# gen turinys (CPU etalonas, su barkodais) → x264 tikras interlaced (tff, MBAFF), uždaras GOP 25, 8 Mbit CBR + AAC.
# Klipas ILGAS (numatyta 90 s), kad matavimo lange nebūtų kilpos sandūros: `ffmpeg -stream_loop -c copy` sandūroje
# praranda pradinį IDR, ir pirmas GOP po sandūros dekoduojamas su klaidų maskavimu (išmatuota: ir FFmpeg CLI, ir NVDEC).
# Etalonas bitų tikslumo patikrai: programinis dekoderis → raw NV12 (per pipe) → laukų CRC32C (ref_crc.json).
source "$(dirname "$0")/env.sh"
OUT="${1:-$HOME/media-test}"
BIN="${FUNC_BIN:-$HOME/build/vramflow/func/vramflow-func}"
FRAMES="${FRAMES:-2250}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$MXL_SRC/install/$MXL_PRESET/lib"
"$BIN" clip --frames "$FRAMES" --first-field 1000 --complexity 2 | \
  ffmpeg -nostdin -loglevel error -y -f rawvideo -pix_fmt nv12 -s 1920x1080 -r 25 -i - \
    -f lavfi -i "sine=frequency=1000:sample_rate=48000:duration=$((FRAMES / 25))" \
    -map 0:v -map 1:a -c:v libx264 -preset medium -flags +ildct+ilme \
    -x264-params "tff=1:keyint=25:min-keyint=25:scenecut=0:open-gop=0:nal-hrd=cbr" \
    -b:v 8M -maxrate 8M -bufsize 8M -bf 2 -field_order tt \
    -c:a aac -b:a 192k -shortest -f mpegts -muxrate 9000000 "$OUT/clip.ts"
ffprobe -v error -select_streams v -show_entries stream=codec_name,width,height,field_order,r_frame_rate -of compact "$OUT/clip.ts"
ffmpeg -nostdin -loglevel error -y -c:v h264 -i "$OUT/clip.ts" -map 0:v -f rawvideo -pix_fmt nv12 - | "$BIN" ref-crc /dev/stdin > "$OUT/ref_crc.json"
echo "CLIP_OK $OUT/clip.ts ($(stat -c %s "$OUT/clip.ts") B), etalonas $(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['fields']))" "$OUT/ref_crc.json") laukų"
