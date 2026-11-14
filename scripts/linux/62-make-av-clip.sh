#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Sintetiniai A/V klipai `ingest` testams (planas docs/planai/2026-11-11-production-grade.md §P3 „Testinė medžiaga“).
#
#   video — `vramflow-func clip` (CPU etalonas su barkodu: grain'o indeksas nuo --first-field 1000 = šaltinio kadro žymė);
#   garsas — 48 kHz stereo: nuolatinis 1 kHz tonas −26 dBFS (0,05; tolydumo patikrai — tyla ar trūkis jame matomi) ir
#            1 mėginio impulsas 0,9 tiksliai kiekvieno kadro F ≡ 0 (mod 25) pradžioje (F — šaltinio kadro numeris: interlaced
#            barkodas / 2, progresyvus — barkodas; kadras 500 / 1000 ≡ 0, todėl impulsai ties mėginiu n ≡ 0 mod 48000);
#   H.264 (x264, uždaras GOP 25, 8 Mbit/s CBR) + AAC 192 kbit/s, MPEG-TS (muxrate 10 Mbit/s).
# Planas siūlė tik impulsą tyloje; tonas pridėtas, nes be jo „be girdimų trūkių“ (K6) neišmatuojama.
#
# Klipai (KATALOGAS, numatyta ~/media-test/av):
#   av_1080i50.ts, av_1080p25.ts   — pagrindiniai (CLIP_SECONDS, numatyta 150 s) + ref_crc_<fmt>.json (programinis dekoderis →
#                                     grain'ų CRC32C bitų tikslumo patikrai)
#   av_1080i50_long.ts               — 300 s 1080i50 (dreifo testai numatytomis parinktimis, be kilpos)
#   av_720p50_short.ts               — 20 s 720p50 (formato pasikeitimui)
#   intra_refresh_1080p25.ts, open_gop_1080p25.ts — 30 s be IDR vidury (B7: įėjimas nuo recovery point / I kadro)
#   fmtchange.ts                     — 15 s 1080i50, tada 720p50 (sujungti TS: PTS šuolis ir SPS pasikeitimas)
#   audio_only_90.ts                 — 90 s tik AAC
#   bad_422.ts, bad_10bit.ts, bad_bff.ts — nepalaikomi (4:2:2, 10 bitų, BFF interlaced), 6 s
#   wrap_1080i50.ts                  — 40 s, PTS persiverčia per 2^33 ~20-ą sekundę (-output_ts_offset)
#   hevc_1080p25.ts                  — 20 s HEVC (jei yra libx265)
# Tik sintetinė medžiaga. Pakartotinis paleidimas esamų failų neperdaro (FORCE=1 — perdaryti).
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="${1:-$HOME/media-test/av}"
BIN="${FUNC_BIN:-$HOME/build/vramflow/func/vramflow-func}"
SECONDS_LONG="${CLIP_SECONDS:-150}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$MXL_SRC/install/$MXL_PRESET/lib"
FF=(ffmpeg -nostdin -hide_banner -loglevel error -y)

audio_src() {  # $1 — trukmė, s
  echo "aevalsrc=0.05*sin(2*PI*1000*t)+0.9*eq(mod(n\,48000)\,0)|0.05*sin(2*PI*1000*t)+0.9*eq(mod(n\,48000)\,0):s=48000:d=$1"
}

# make <failas> <formatas> <sekundės> [papildomi x264 / ffmpeg argumentai…]
make_clip() {
  local f="$1" fmt="$2" sec="$3"; shift 3
  [[ -f "$OUT/$f" && "${FORCE:-0}" != 1 ]] && { echo "yra $OUT/$f"; return; }
  local w=1920 h=1080 fps=25 vflags=() xp="keyint=25:min-keyint=25:scenecut=0:open-gop=0:nal-hrd=cbr" fo=() vf=()
  # interlaced: įėjimo kadrai pažymimi setfield — libx264 pic_struct (laukų tvarkos SEI) ima iš kadro TOP_FIELD_FIRST žymės;
  # be jos rawvideo kadras laikomas BFF ir TFF turinys pažymimas bottom-top (išmatuota: be setfield pic_struct 3 → 4 nuo 16 kadro)
  case "$fmt" in
    1080i50) vflags=(-flags +ildct+ilme); xp="tff=1:$xp"; fo=(-field_order tt); vf=(-vf setfield=tff) ;;
    1080i50bff) vflags=(-flags +ildct+ilme); xp="bff=1:$xp"; fo=(-field_order bb); vf=(-vf setfield=bff); fmt=1080i50 ;;
    1080p25) ;;
    720p50) w=1280; h=720; fps=50; xp="keyint=50:min-keyint=50:scenecut=0:open-gop=0:nal-hrd=cbr" ;;
    *) echo "nežinomas formatas $fmt" >&2; return 1 ;;
  esac
  local frames=$((sec * fps))
  "$BIN" clip --format "$fmt" --frames "$frames" --first-field 1000 --complexity 2 | \
    "${FF[@]}" -f rawvideo -pix_fmt nv12 -s "${w}x${h}" -r "$fps" -i - -f lavfi -i "$(audio_src "$sec")" \
      -map 0:v -map 1:a "${vf[@]}" -c:v libx264 -preset veryfast "${vflags[@]}" -x264-params "$xp" \
      -b:v 8M -maxrate 8M -bufsize 8M -bf 2 "${fo[@]}" "$@" \
      -c:a aac -b:a 192k -shortest -f mpegts -muxrate 10000000 "$OUT/$f"
  echo "sukurta $OUT/$f"
}

make_clip av_1080i50.ts 1080i50 "$SECONDS_LONG"
make_clip av_1080p25.ts 1080p25 "$SECONDS_LONG"
make_clip av_1080i50_long.ts 1080i50 300  # dreifas numatytomis parinktimis (±500 ppm, 240 s be kilpos sandūrų)
make_clip av_720p50_short.ts 720p50 20
# B7: be IDR vidury — periodinis intra atnaujinimas (IDR tik pirmas kadras, recovery point SEI) ir atviras GOP (I kadrai ne IDR)
# (vėlesnis -x264-params pakeičia make_clip numatytąjį)
make_clip intra_refresh_1080p25.ts 1080p25 30 -x264-params "keyint=50:min-keyint=50:scenecut=0:intra-refresh=1:nal-hrd=cbr"
make_clip open_gop_1080p25.ts 1080p25 30 -x264-params "keyint=25:min-keyint=25:scenecut=0:open-gop=1:nal-hrd=cbr"
make_clip fmt_1080i50_short.ts 1080i50 15
make_clip bad_422.ts 1080p25 6 -pix_fmt yuv422p -profile:v high422
make_clip bad_10bit.ts 1080p25 6 -pix_fmt yuv420p10le -profile:v high10
make_clip bad_bff.ts 1080i50bff 6
make_clip wrap_1080i50.ts 1080i50 40 -output_ts_offset 95423
if [[ ! -f "$OUT/fmtchange.ts" || "${FORCE:-0}" == 1 ]]; then
  cat "$OUT/fmt_1080i50_short.ts" "$OUT/av_720p50_short.ts" > "$OUT/fmtchange.ts"
  echo "sukurta $OUT/fmtchange.ts"
fi
if [[ ! -f "$OUT/audio_only_90.ts" || "${FORCE:-0}" == 1 ]]; then
  "${FF[@]}" -f lavfi -i "$(audio_src 90)" -c:a aac -b:a 192k -f mpegts "$OUT/audio_only_90.ts"
  echo "sukurta $OUT/audio_only_90.ts"
fi
if ffmpeg -hide_banner -encoders 2>/dev/null | grep libx265 > /dev/null; then  # be -q: grep perskaito viską (pipefail + SIGPIPE)
  if [[ ! -f "$OUT/hevc_1080p25.ts" || "${FORCE:-0}" == 1 ]]; then
    "$BIN" clip --format 1080p25 --frames 500 --first-field 1000 --complexity 2 | \
      "${FF[@]}" -f rawvideo -pix_fmt nv12 -s 1920x1080 -r 25 -i - -f lavfi -i "$(audio_src 20)" -map 0:v -map 1:a \
        -c:v libx265 -preset veryfast -x265-params "keyint=25:min-keyint=25:scenecut=0:log-level=error" -b:v 8M -pix_fmt yuv420p \
        -c:a aac -b:a 192k -shortest -f mpegts -muxrate 10000000 "$OUT/hevc_1080p25.ts"
    echo "sukurta $OUT/hevc_1080p25.ts"
  fi
fi
# Bitų tikslumo etalonai: programinis dekoderis → raw NV12 → grain'ų CRC32C su barkodu
for fmt in 1080i50 1080p25; do
  if [[ ! -f "$OUT/ref_crc_$fmt.json" || "${FORCE:-0}" == 1 ]]; then
    "${FF[@]}" -c:v h264 -i "$OUT/av_$fmt.ts" -map 0:v -f rawvideo -pix_fmt nv12 - | "$BIN" ref-crc --format "$fmt" /dev/stdin > "$OUT/ref_crc_$fmt.json"
  fi
done
ls -la "$OUT"
echo "AV_CLIPS_OK $OUT"
