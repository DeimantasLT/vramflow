#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Apeina GitHub HTTP 429 anoniminiams `https://github.com/<o>/<r>/commit/<sha>.patch` atsisiuntimams vcpkg'e.
# Kiekvienam nurodytam port'ui: nukopijuoja jo versijuotą portfile į overlay, o kiekvieną
# vcpkg_download_distfile(VAR URLS https://github.com/O/R/commit/SHA.patch... FILENAME F SHA512 ...)
# pakeičia set(VAR <overlay>/F), kur F sugeneruotas `git format-patch --full-index` iš oficialaus repo.
# Pataiso turinys tas pats; skiriasi tik GitHub antraštės, todėl vcpkg SHA512 netinka.
# Naudojimas: 05-vcpkg-patch-overlay.sh fmt spdlog ...
source "$(dirname "$0")/env.sh"
OVERLAY="$HOME/vcpkg-overlay"
mkdir -p "$OVERLAY"
for port in "$@"; do
  src="$(grep -rlE "github.com/[^/]+/[^/]+/commit/[0-9a-f]{40}\.patch" \
         "$VCPKG_ROOT/buildtrees/versioning_/versions/$port" --include=portfile.cmake 2>/dev/null | head -1 || true)"
  [[ -n "$src" ]] || { echo "[$port] nėra commit .patch atsisiuntimų versijuotame portfile — praleidžiu"; continue; }
  rm -rf "$OVERLAY/$port" && mkdir -p "$OVERLAY/$port"
  cp -r "$(dirname "$src")/." "$OVERLAY/$port/"
  python3 - "$OVERLAY/$port" <<'PY'
import re, subprocess, sys, tempfile, os, pathlib
d = pathlib.Path(sys.argv[1]); pf = d / "portfile.cmake"; s = pf.read_text()
blk = re.compile(r'vcpkg_download_distfile\(\s*(\w+)\s+URLS\s+(https://github\.com/([^/]+)/([^/]+)/commit/([0-9a-f]{40})\.patch\S*)\s+FILENAME\s+"?([^\s")]+)"?\s+SHA512\s+\w+\s*\)', re.S)
clones = {}
def repl(m):
    var, url, owner, repo, sha, fname = m.groups()
    key = f"{owner}/{repo}"
    if key not in clones:
        t = tempfile.mkdtemp()
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                        f"https://github.com/{key}.git", t], check=True)
        clones[key] = t
    out = subprocess.run(["git", "-C", clones[key], "format-patch", "--full-index", "--stdout", "-1", sha],
                         check=True, capture_output=True).stdout
    (d / fname).write_bytes(out)
    print(f"  {var}: {key}@{sha[:10]} -> {fname} ({len(out)} B)")
    return f'set({var} "${{CMAKE_CURRENT_LIST_DIR}}/{fname}")'
s2, n = blk.subn(repl, s)
pf.write_text(s2)
print(f"  pakeista blokų: {n}")
PY
  echo "[$port] overlay paruoštas"
done
echo "OVERLAY_OK $OVERLAY"
