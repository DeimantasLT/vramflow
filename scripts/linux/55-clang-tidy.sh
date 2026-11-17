#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Statinė analizė (planas docs/planai/2026-11-17-production-ready.md §2.4, kriterijus K3): clang-tidy pagal repo .clang-tidy
# visiems src/ vertimo vienetams. .cu failai netikrinami (juos kompiliuoja nvcc — clang-tidy jo komandų nesupranta); testai
# (tests/) į šių vartų apimtį neįeina. Reikia compile_commands.json (jį generuoja 30-build-verify.sh).
#
# Naudojimas: 55-clang-tidy.sh [išvesties katalogas]
#   TIDY_BUILD=~/build/vramflow/func   build katalogas su compile_commands.json
#   TIDY_JOBS=<nproc>                  lygiagrečių clang-tidy procesų skaičius
#   TIDY_ONLY=<regex>                  tik failai, kurių kelias repo atžvilgiu atitinka (pvz. '^src/runtime/')
#   TIDY_RUN / TIDY_BIN                run-clang-tidy ir clang-tidy vykdomieji (numatyta …-19)
# Rezultatas: <out>/tidy.log (visa išvestis), <out>/findings.txt (unikalūs radiniai, keliai repo atžvilgiu),
# <out>/tidy.json ({"files", "findings", "errors", "by_check"}); stdout — lentelė pagal tikrinimą.
# Exit 0 — radinių nėra; 1 — yra radinių; 2 — analizė neįvyko arba nepilna (kompiliavimo klaidos).
source "$(dirname "$0")/env.sh"
set +e; set +o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-${EVIDENCE:-$HOME/evidence/latest}/55-clang-tidy}"
BUILD="${TIDY_BUILD:-$HOME/build/vramflow/func}"
JOBS="${TIDY_JOBS:-$(nproc)}"
RUN="${TIDY_RUN:-run-clang-tidy-19}"; TIDY="${TIDY_BIN:-clang-tidy-19}"
command -v "$RUN" >/dev/null || { echo "55-clang-tidy: nėra $RUN (nustatykite TIDY_RUN)" >&2; exit 2; }
command -v "$TIDY" >/dev/null || { echo "55-clang-tidy: nėra $TIDY (nustatykite TIDY_BIN)" >&2; exit 2; }
[[ -f $BUILD/compile_commands.json ]] || { echo "55-clang-tidy: nėra $BUILD/compile_commands.json (pirma 30-build-verify.sh)" >&2; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT/db"

# Duomenų bazė tik mūsų src/ .cpp ir .c vienetams (be nvcc įrašų ir be testų) + tikslus antraščių filtras šiam repo keliui
python3 - "$BUILD/compile_commands.json" "$REPO" "$OUT/db" "${TIDY_ONLY:-}" <<'PY' || { echo "55-clang-tidy: nėra ką tikrinti" >&2; exit 2; }
import json, os, re, sys
src, repo, dbdir, only = sys.argv[1:5]
root = os.path.join(repo, "src") + os.sep
keep = []
for e in json.load(open(src, encoding="utf-8")):
    f = os.path.normpath(os.path.join(e.get("directory", ""), e["file"]))
    if not f.startswith(root) or not f.endswith((".cpp", ".c")):
        continue
    if only and not re.search(only, os.path.relpath(f, repo)):
        continue
    keep.append(e)
json.dump(keep, open(os.path.join(dbdir, "compile_commands.json"), "w", encoding="utf-8"), indent=0)
open(os.path.join(dbdir, "header-filter.txt"), "w", encoding="utf-8").write("^" + re.escape(repo) + "/(src|include)/")
print(f"55-clang-tidy: {len(keep)} vertimo vienetai")
sys.exit(0 if keep else 1)
PY

t0=$SECONDS
"$RUN" -clang-tidy-binary "$TIDY" -p "$OUT/db" -j "$JOBS" -quiet -header-filter "$(cat "$OUT/db/header-filter.txt")" > "$OUT/tidy.log" 2>&1
code=$?

python3 - "$OUT/tidy.log" "$REPO" "$OUT" "$code" "$((SECONDS - t0))" <<'PY'
import collections, json, os, re, sys
log, repo, out, code, secs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rx = re.compile(r"^(/[^:\n]+):(\d+):(\d+): (warning|error): (.*?)(?: \[([^\]]+)\])?$")
seen, errors = {}, []
for line in open(log, encoding="utf-8", errors="replace"):
    m = rx.match(line.rstrip("\n"))
    if not m:
        continue
    path, ln, col, kind, msg, check = m.groups()
    rel = os.path.relpath(path, repo) if path.startswith(repo + os.sep) else path
    # kompiliavimo klaida (ne tikrinimo radinys): analizė šiam vienetui nepilna
    if kind == "error" and (check is None or check.startswith("clang-diagnostic")):
        errors.append(f"{rel}:{ln}:{col} {msg}")
        continue
    check = (check or "be-pavadinimo").replace(",-warnings-as-errors", "")
    seen.setdefault((rel, int(ln), int(col), check), msg)  # ta pati antraštės vieta iš kelių vienetų — vienas radinys
by = collections.Counter(k[3] for k in seen)
files = len(json.load(open(os.path.join(out, "db", "compile_commands.json"), encoding="utf-8")))
with open(os.path.join(out, "findings.txt"), "w", encoding="utf-8") as f:
    for (rel, ln, col, check), msg in sorted(seen.items()):
        f.write(f"{rel}:{ln}:{col} [{check}] {msg}\n")
json.dump({"files": files, "findings": len(seen), "errors": len(errors), "seconds": secs, "exit": code, "by_check": dict(by.most_common())},
          open(os.path.join(out, "tidy.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"clang-tidy: {files} vertimo vienetai, {secs} s, unikalių radinių {len(seen)}, kompiliavimo klaidų {len(errors)}")
for check, n in by.most_common():
    print(f"  {n:5d}  {check}")
for e in sorted(set(errors))[:20]:
    print(f"  KLAIDA {e}")
sys.exit(2 if errors or (code != 0 and not seen) else 1 if seen else 0)
PY
rc=$?
echo "TIDY_DONE $OUT (exit $rc)"
exit $rc
