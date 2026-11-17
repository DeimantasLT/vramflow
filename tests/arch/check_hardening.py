#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Build grūdinimo patikra (planas docs/planai/2026-11-17-production-ready.md §1.1, kriterijus S6, RF-1).

Tikrina VISUS savus vertimo vienetus (compile_commands.json įrašai iš src/ ir tests/) ir VISUS savus ELF failus build
kataloge (vramflow-*, mxlv-*, libvramflow*.so*), ne fiksuotą sąrašą: naujas tikslas be vėliavų šį testą sulaužo.

  Vertimo vienetai: -fstack-protector-strong, -fstack-clash-protection, -fcf-protection=full; C++ ir CUDA host daliai —
                    -D_GLIBCXX_ASSERTIONS; -D_FORTIFY_SOURCE=3 (ne Debug ir ne sanitizerių / fuzz build'e).
  ELF failai:       PIE / DYN, GNU_RELRO, BIND_NOW (pilnas RELRO), nevykdomas stekas; pristatomiems binarams
                    (vramflow-func, vramflow-verify, libvramflow-gfx.so) — __stack_chk_fail.

CET žyma (IBT, SHSTK) — tik informacijai: ji priklauso nuo visų susietų objektų, o CUDA::cudart_static jos neturi.

Naudojimas: check_hardening.py <build katalogas> <šaltinio katalogas>. Exit: 0 — tinka, 1 — netinka, 77 — praleista.
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys

TU_FLAGS = ["-fstack-protector-strong", "-fstack-clash-protection", "-fcf-protection=full"]
CXX_FLAG = "-D_GLIBCXX_ASSERTIONS"
FORTIFY_FLAG = "-D_FORTIFY_SOURCE=3"
ELF_PATTERNS = ["vramflow-*", "mxlv-*", "libvramflow*.so*"]
DELIVERABLES = ("vramflow-func", "vramflow-verify", "libvramflow-gfx.so")
NO_FORTIFY_OPTIONS = ("MXLV_ENABLE_SANITIZERS", "MXLV_ENABLE_TSAN", "MXLV_BUILD_FUZZ")


def cache(build):
    """CMakeCache.txt → {vardas: reikšmė}."""
    out = {}
    p = build / "CMakeCache.txt"
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def is_on(v):
    return str(v).upper() in ("ON", "1", "TRUE", "YES", "Y")


def check_units(build, src, want_fortify):
    """Grąžina (patikrintų skaičius, klaidų sąrašas)."""
    p = build / "compile_commands.json"
    if not p.exists():
        return 0, ["compile_commands.json nėra (CMAKE_EXPORT_COMPILE_COMMANDS)"]
    bad, n = [], 0
    for e in json.loads(p.read_text()):
        f = pathlib.Path(e["file"])
        try:
            rel = f.resolve().relative_to(src.resolve()).as_posix()
        except ValueError:
            continue  # ne mūsų šaltinis (generuotas ar trečiųjų šalių)
        if not rel.startswith(("src/", "tests/")) or f.suffix not in (".c", ".cpp", ".cu"):
            continue
        n += 1
        cmd = e.get("command") or " ".join(e.get("arguments", []))
        need = list(TU_FLAGS)
        if f.suffix != ".c":
            need.append(CXX_FLAG)
        if want_fortify:
            need.append(FORTIFY_FLAG)
        missing = [x for x in need if x not in cmd]
        if missing:
            bad.append(f"{rel}: trūksta {' '.join(missing)}")
    if n == 0:
        bad.append("compile_commands.json neturi nė vieno įrašo iš src/ ar tests/")
    return n, bad


def readelf(args, f):
    return subprocess.run(["readelf", *args, str(f)], capture_output=True, text=True, errors="replace").stdout


def own_elfs(build):
    seen = []
    for pat in ELF_PATTERNS:
        for f in sorted(build.glob(pat)):
            if f.is_symlink() or not f.is_file():
                continue
            with open(f, "rb") as fh:
                if fh.read(4) != b"\x7fELF":
                    continue
            seen.append(f)
    return seen


def check_elfs(build):
    """Grąžina (patikrintų skaičius, klaidų sąrašas, CET informacijos eilutės)."""
    bad, info = [], []
    elfs = own_elfs(build)
    if not elfs:
        bad.append(f"{build}: nerasta nė vieno savo ELF failo")
    for f in elfs:
        n = f.name
        hdr, seg, dyn = readelf(["-h"], f), readelf(["-lW"], f), readelf(["-dW"], f)
        if not re.search(r"Type:\s+DYN", hdr):
            bad.append(f"{n}: ne PIE (Type ne DYN)")
        if "GNU_RELRO" not in seg:
            bad.append(f"{n}: GNU_RELRO nėra")
        if not re.search(r"BIND_NOW|\(FLAGS_1\).*\bNOW\b", dyn):
            bad.append(f"{n}: BIND_NOW nėra (RELRO tik dalinis)")
        stack = [ln for ln in seg.splitlines() if "GNU_STACK" in ln]
        if not stack:
            bad.append(f"{n}: GNU_STACK segmento nėra")
        elif re.search(r"\bRWE\b", stack[0]):
            bad.append(f"{n}: vykdomas stekas")
        if n.startswith(DELIVERABLES) and "__stack_chk_fail" not in readelf(["-sW"], f):
            bad.append(f"{n}: __stack_chk_fail nėra (stack protector)")
        notes = readelf(["-nW"], f)
        info.append(f"{n}: CET {'IBT' if 'IBT' in notes else '-'} {'SHSTK' if 'SHSTK' in notes else '-'}")
    return len(elfs), bad, info


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    build, src = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    c = cache(build)
    if "VRAMFLOW_HARDENING" in c and not is_on(c["VRAMFLOW_HARDENING"]):
        print("SKIP: VRAMFLOW_HARDENING=OFF")
        return 77
    if shutil.which("readelf") is None:
        print("SKIP: nėra readelf")
        return 77
    want_fortify = c.get("CMAKE_BUILD_TYPE", "").lower() != "debug" and not any(is_on(c.get(k, "OFF")) for k in NO_FORTIFY_OPTIONS)
    n_tu, bad_tu = check_units(build, src, want_fortify)
    n_elf, bad_elf, info = check_elfs(build)
    for ln in info:
        print("INFO", ln)
    for ln in (bad_tu + bad_elf)[:60]:
        print("FAIL", ln)
    total = len(bad_tu) + len(bad_elf)
    print(f"vertimo vienetų {n_tu} (netinka {len(bad_tu)}), ELF failų {n_elf} (klaidų {len(bad_elf)}), "
          f"_FORTIFY_SOURCE {'tikrinamas' if want_fortify else 'netikrinamas'}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
