#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Paleidimas NE root vartotoju (planas docs/planai/2026-11-17-production-ready.md §1.3, kriterijus S5): tas pats build,
# ctest ir funkcinis rinkinys kito vartotojo vardu. Įrodo, kad nei build'ui, nei funkcijoms root nereikia.
#
# Naudojimas (kviečiama kaip root — vartotojui perjungti):
#   45-nonroot.sh <vartotojas> [build | ctest | functional | all] [functional.py parinktys, pvz. --only fcomp1]   (numatyta all)
# Vartotojo kataloge /home/<vartotojas>/vramflow-nonroot/ yra tik jo paties build ir testų išvestis. Žurnalai — root kataloge
# NONROOT_LOG_DIR (numatyta $EVIDENCE/45-nonroot arba ~/evidence/latest/45-nonroot): root niekada nerašo į vartotojo katalogą
# (ten padėta simbolinė nuoroda leistų vartotojui root teisėmis perrašyti bet kurį failą).
# MXL instaliacija ir bandomoji medžiaga imamos iš šio (root) vartotojo katalogų tik skaitymui: MXL_SRC, MEDIA_DIR.
# Exit ≠ 0 — kuris nors etapas nepavyko.
source "$(dirname "$0")/env.sh"
set +e; set +o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
U="${1:-}"; STAGE="${2:-all}"
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift   # likę argumentai — functional.py parinktys
if [[ -z $U ]] || ! id -u "$U" >/dev/null 2>&1; then
  echo "45-nonroot: nurodykite esamą ne root vartotoją (gauta „$U“)" >&2; exit 2
fi
if [[ $(id -u "$U") == 0 ]]; then echo "45-nonroot: „$U“ yra root" >&2; exit 2; fi
if [[ $(id -u) != 0 ]]; then echo "45-nonroot: kvieskite kaip root (vartotojui perjungti reikia setpriv)" >&2; exit 2; fi
case "$STAGE" in build|ctest|functional|all) ;; *) echo "45-nonroot: etapas „$STAGE“ (build | ctest | functional | all)" >&2; exit 2;; esac
UHOME="$(getent passwd "$U" | cut -d: -f6)"
W="$UHOME/vramflow-nonroot"; B="$W/build"; OUT="$W/out"
MEDIA_DIR="${MEDIA_DIR:-$HOME/media-test}"
LOGDIR="${NONROOT_LOG_DIR:-${EVIDENCE:-$HOME/evidence/latest}/45-nonroot}"
mkdir -p "$LOGDIR" || { echo "45-nonroot: negaliu sukurti $LOGDIR" >&2; exit 2; }
LOG="$LOGDIR/nonroot.log"
# Vartotojo aplinka: savas HOME ir build, o MXL ir medžiaga — iš root katalogų (tik skaitymas)
as_user() {
  setpriv --reuid="$U" --regid="$(id -g "$U")" --init-groups env -i HOME="$UHOME" USER="$U" LOGNAME="$U" \
    PATH="$PATH:/usr/lib/wsl/lib:$MXL_SRC/install/$MXL_PRESET/bin" LANG="${LANG:-C.UTF-8}" \
    MXL_SRC="$MXL_SRC" MXL_PRESET="$MXL_PRESET" CUDA_HOME="$CUDA_HOME" VCPKG_ROOT="$VCPKG_ROOT" LOG_DIR="$W/logs" \
    CC_BIN="${CC_BIN:-}" CXX_BIN="${CXX_BIN:-}" CUDA_HOST_CXX="${CUDA_HOST_CXX:-}" \
    VERIFY_BUILD="$B" VERIFY_BIN="$B/vramflow-verify" FUNC_BIN="$B/vramflow-func" \
    MXL_INFO="$MXL_SRC/install/$MXL_PRESET/bin/mxl-info" MXL_LIB="$MXL_SRC/install/$MXL_PRESET/lib" \
    MXL_BIN="$MXL_SRC/install/$MXL_PRESET/bin" \
    INGEST_CLIPS="$MEDIA_DIR/av" CLIP="$MEDIA_DIR/clip.ts" REF="$MEDIA_DIR/ref_crc.json" \
    FGOLD_CLIP="$MEDIA_DIR/clip.ts" FGOLD_REF="$MEDIA_DIR/ref_crc.json" "$@"
}
as_user mkdir -p "$W" "$W/logs" || { echo "45-nonroot: vartotojas $U negali sukurti $W" >&2; exit 2; }
say() { echo "$*" | tee -a "$LOG"; }
say "== 45-nonroot $(date -u +%FT%TZ): vartotojas $U (uid $(id -u "$U")), etapas $STAGE, HEAD $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
rc=0
if [[ $STAGE == build || $STAGE == all ]]; then
  as_user rm -rf "$B"
  as_user bash "$REPO/scripts/linux/30-build-verify.sh" -DMXLV_BUILD_FUNC=ON > "$LOGDIR/build.log" 2>&1; c=$?
  say "build exit=$c warnings=$(grep -c 'warning:' "$LOGDIR/build.log") savininkas=$(stat -c %U "$B/vramflow-func" 2>/dev/null)"
  [[ $c != 0 ]] && rc=1
fi
if [[ $STAGE == ctest || $STAGE == all ]]; then
  as_user bash -c "cd '$B' && ctest" > "$LOGDIR/ctest.log" 2>&1; c=$?
  say "ctest exit=$c: $(grep -E 'tests passed|tests failed' "$LOGDIR/ctest.log" | tail -1)"
  grep -E '\*\*\*Failed|\*\*\*Timeout|\*\*\*Exception' "$LOGDIR/ctest.log" | cut -c1-200 | tee -a "$LOG"
  [[ $c != 0 ]] && rc=1
fi
if [[ $STAGE == functional || $STAGE == all ]]; then
  as_user rm -rf "$OUT"
  as_user python3 "$REPO/tests/functional/functional.py" "$OUT" "$@" > "$LOGDIR/functional.log" 2>&1; c=$?
  say "functional exit=$c: $(grep -E '^FUNCTIONAL' "$LOGDIR/functional.log" | tail -1 | cut -c1-300)"
  grep -E '^\[FAIL' "$LOGDIR/functional.log" | cut -c1-400 | tee -a "$LOG"
  say "PASS: $(grep -cE '^\[PASS' "$LOGDIR/functional.log")  INFO: $(grep -cE '^\[INFO' "$LOGDIR/functional.log")  SKIP: $(grep -cE '^\[SKIP' "$LOGDIR/functional.log")  (žurnalas — $LOGDIR/functional.log)"
  [[ $c != 0 ]] && rc=1
fi
say "== 45-nonroot baigta, exit $rc"
exit $rc
