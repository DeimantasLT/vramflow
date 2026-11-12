#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Pristatomos libvramflow-gfx.so ABI patikra (gfx.h „VERSIONING AND ABI“): SONAME = libvramflow-gfx.so.<ABI epocha>
(VFGFX_ABI_VERSION iš antraštės, ne leidimo versija), eksportuojami TIK vfgfx_* simboliai, visi su numatytąja simbolių
versija VFGFX_<epocha> (versijų mazgas iš cmake/vramflow-gfx.map). Naudojimas: gfx_so_abi_check.py <libvramflow-gfx.so>
<include/vramflow/gfx.h>. Sėkmės atveju spausdina `OK`."""
import re
import subprocess
import sys


def main():
    so, header = sys.argv[1], sys.argv[2]
    text = open(header, encoding="utf-8").read()
    epoch = int(re.search(r"^#define VFGFX_ABI_VERSION ([0-9]+)", text, re.M).group(1))
    dyn = subprocess.run(["readelf", "-d", so], capture_output=True, text=True, check=True).stdout
    soname = re.search(r"Library soname: \[(.*)\]", dyn).group(1)
    assert soname == "libvramflow-gfx.so.%d" % epoch, "SONAME %s, tikėtasi libvramflow-gfx.so.%d" % (soname, epoch)
    nm = subprocess.run(["nm", "-D", "--defined-only", "--with-symbol-versions", so], capture_output=True, text=True,
                        check=True).stdout
    node = "VFGFX_%d" % epoch
    funcs = []
    for line in nm.splitlines():
        parts = line.split()
        name = parts[-1]
        if parts[-2] == "A" and name.split("@")[0] == node:  # versijos mazgo apibrėžimo simbolis
            continue
        m = re.fullmatch(r"(vfgfx_\w+)@@(VFGFX_[\w.]+)", name)
        assert m, "netikėtas eksportuojamas simbolis: %s" % line
        assert m.group(2) == node, "simbolis %s ne versijos mazge %s" % (name, node)
        funcs.append(m.group(1))
    assert len(funcs) >= 15, funcs
    print("OK", soname, node, len(funcs), "funkcijos:", " ".join(sorted(funcs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
