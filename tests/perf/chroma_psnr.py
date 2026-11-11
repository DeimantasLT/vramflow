#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Chromos PSNR prieš FFmpeg progresyviam turiniui (planas docs/planai/2026-11-11-production-grade.md §P2).

Sintetinis progresyvus NV12 (FFmpeg `zoneplate` + `vramflow-func clip --format <progresyvus>`) → mūsų 4:2:2 10 bitų
(`vramflow-func split-raw`, nepriklausomas etalonas; CUDA ir CPU jam lygūs bitas į bitą) prieš
`ffmpeg -vf scale=flags=bilinear -pix_fmt yuv422p10le` (swscale tiesinė vertikali interpoliacija — ta pati 0 tipo
padėtis ir formulė kaip mūsų). Palyginimas — FFmpeg `psnr` filtras (pikas 1023).
  * kadro chroma (FrameSited): Y, U ir V — bitas į bitą (PSNR = inf);
  * neigiamas: tas pats per lauko chromą (FieldSited, kadras kaip du laukai) — max(U, V) < --neg-max-db (numatyta 50),
    t. y. testas skiria teisingą ir neteisingą kelią (brief'o pamoka 5);
  * PSNR prieš swscale numatytąjį bicubic spausdinamas tik informacijai (ne kriterijus: kita interpoliacija).
Tik sintetinė medžiaga. Išėjimas: 0 — PASS, 1 — FAIL, 77 — SKIP (nėra ffmpeg / psnr filtro).

Naudojimas: chroma_psnr.py --bin <vramflow-func> [--format 1080p50] [--frames 3] [--work DIR] [--ffmpeg ffmpeg]
"""
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

SIZES = {"720": (1280, 720), "1080": (1920, 1080), "2160": (3840, 2160)}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def psnr(ffmpeg, w, h, ours_v210, ref_yuv):
    """FFmpeg psnr filtras: mūsų v210 (dekoduojamas tiksliai) prieš FFmpeg yuv422p10le. Grąžina {'y','u','v'} dB."""
    r = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-f", "v210", "-video_size", f"{w}x{h}", "-i", str(ours_v210),
                        "-f", "rawvideo", "-pix_fmt", "yuv422p10le", "-video_size", f"{w}x{h}", "-i", str(ref_yuv),
                        "-lavfi", "[0:v]format=yuv422p10le[a];[a][1:v]psnr", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"PSNR y:(\S+) u:(\S+) v:(\S+)", r.stderr)
    if r.returncode != 0 or not m:
        raise RuntimeError("psnr nepavyko: " + r.stderr[-800:])
    return {k: float("inf") if x == "inf" else float(x) for k, x in zip("yuv", m.groups())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="vramflow-func")
    ap.add_argument("--format", default="1080p50")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--neg-max-db", type=float, default=50.0)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--work", default="")
    ap.add_argument("--zoneplate", default="kx2=20:ky2=20:kt2=4:ku=96:kv=160", help="zoneplate parametrai (dažnių juosta)")
    a = ap.parse_args()
    ff = shutil.which(a.ffmpeg) or (a.ffmpeg if pathlib.Path(a.ffmpeg).exists() else None)
    if not ff:
        print("SKIP: nėra ffmpeg")
        return 77
    filters = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " psnr " not in filters or " zoneplate " not in filters:
        print("SKIP: ffmpeg be psnr / zoneplate filtro")
        return 77
    m = re.fullmatch(r"(720|1080|2160)p[0-9.]+", a.format)
    if not m:
        print(f"FAIL: --format {a.format} — tik progresyvūs formatai")
        return 1
    w, h = SIZES[m.group(1)]
    work = pathlib.Path(a.work) if a.work else pathlib.Path(tempfile.mkdtemp(prefix="chroma_psnr_"))
    work.mkdir(parents=True, exist_ok=True)

    # šaltiniai: zonų plokštė (žema / vidutinė dažnių juosta abiem kryptimis, juda laike) ir mūsų generatoriaus klipas
    sources = {}
    zp = work / "zoneplate.nv12"
    run([ff, "-hide_banner", "-y", "-f", "lavfi", "-i", f"zoneplate=s={w}x{h}:r=50:{a.zoneplate}",
         "-frames:v", str(a.frames), "-pix_fmt", "nv12", "-f", "rawvideo", str(zp)])
    sources["zoneplate"] = zp
    clip = work / "clip.nv12"
    with open(clip, "wb") as f:
        subprocess.run([a.bin, "clip", "--format", a.format, "--frames", str(a.frames), "--first-field", "1000", "--complexity", "2"],
                       check=True, stdout=f)
    sources["clip"] = clip

    results, ok = {}, True
    inf = float("inf")
    for name, src in sources.items():
        # etalonas — swscale BILINEAR (tiesinė vertikali interpoliacija, ta pati 0 tipo padėtis kaip mūsų formulė);
        # bicubic (numatytasis swscale) — tik informacijai
        refs = {}
        for flt in ("bilinear", "bicubic"):
            refs[flt] = work / f"{name}.ref.{flt}.yuv"
            run([ff, "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "nv12", "-video_size", f"{w}x{h}", "-i", str(src),
                 "-vf", f"scale=flags={flt}", "-f", "rawvideo", "-pix_fmt", "yuv422p10le", str(refs[flt])])
        res = {}
        for chroma in ("frame", "field"):
            ours = work / f"{name}.{chroma}.v210"
            run([a.bin, "split-raw", "--format", a.format, "--chroma", chroma, "--in", str(src), "--out", str(ours)])
            res[chroma] = psnr(ff, w, h, ours, refs["bilinear"])
            res[chroma + "_bicubic"] = psnr(ff, w, h, ours, refs["bicubic"])
        good = (res["frame"]["y"] == inf and res["frame"]["u"] == inf and res["frame"]["v"] == inf
                and max(res["field"]["u"], res["field"]["v"]) < a.neg_max_db)
        ok = ok and good
        results[name] = res
        fb = res["frame_bicubic"]
        print(f"[{'PASS' if good else 'FAIL'}] {name}: kadro chroma prieš bilinear Y {res['frame']['y']} U {res['frame']['u']} "
              f"V {res['frame']['v']} (reikia: bitas į bitą = inf); lauko chroma U {res['field']['u']:.2f} V {res['field']['v']:.2f} dB "
              f"(riba < {a.neg_max_db}); informacijai prieš bicubic: U {fb['u']:.2f} V {fb['v']:.2f} dB", flush=True)
    (work / "chroma_psnr.json").write_text(json.dumps({"format": a.format, "frames": a.frames, "results": results}, indent=1))
    print(f"CHROMA_PSNR: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
