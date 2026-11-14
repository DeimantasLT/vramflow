#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""A/V poslinkis sintetiniame ingest klipe (scripts/linux/62-make-av-clip.sh): kadro žymė ↔ 1 mėginio impulsas.

Klipas: video — `vramflow-func clip` barkodas (32 blokai po 32 px kadro viršuje; b0..23 — grain'o indeksas); garsas —
1 kHz tonas 0,05 ir impulsas 0,9 kiekvieno šaltinio kadro F ≡ 0 (mod 25) pradžioje (F = barkodas / 2 interlaced, = barkodas
progresyviam). Teisingame sraute impulsas sutampa su to kadro PTS.

Matavimas (sistemos ffmpeg, viena dekodavimo eiga, -copyts):
  video — viršutinės juostos (1024×24) luma kiekvienam kadrui + kadro PTS (showinfo); barkodas — bloko centro vidurkis
          lyginėse eilutėse 8..22 (interlaced — viršutinis laukas, progresyviam — tos pačios barkodo eilutės);
  garsas — mono f32 PCM ir kiekvieno dekoduoto kadro PTS (ashowinfo); impulsas — |x| maksimumas grupėje, kur |x| > 0,5;
  poslinkis = impulso laikas − kadro F ≡ 0 (mod 25) PTS (artimiausias, |Δ| < 0,5 s), ms: > 0 — garsas vėluoja.
Tai PR 1 matavimas: įrodo, kad pats klipas sinchroniškas (AAC pradžios užlaida kompensuota PTS), prieš matuojant ingest.

Naudojimas: av_marker.py KLIPAS.ts [--interlaced auto|0|1] [--json OUT.json]; išėjimas 0 — išmatuota, 2 — nepavyko.
"""
import argparse
import array
import json
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile

W, ROWS = 1024, 24


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(ffprobe, ts):
    r = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,field_order,r_frame_rate",
             "-of", "json", str(ts)])
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    return st.get("width"), st.get("height"), st.get("field_order", "progressive")


def marker(frame):
    """Barkodas iš viršutinės juostos (W × ROWS, 8 bitų luma)."""
    word = 0
    for k in range(32):
        s = 0
        for j in range(8, 23, 2):  # lyginės eilutės 8..22 = viršutinio lauko eilutės 4..11
            row = frame[j * W:(j + 1) * W]
            s += sum(row[32 * k + 12:32 * k + 20])
        if s > 125 * 64:
            word |= 1 << k
    return word


def analyze(ts, interlaced=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    ts = pathlib.Path(ts)
    w, h, fo = probe(ffprobe, ts)
    if interlaced is None:
        interlaced = fo not in (None, "progressive", "unknown")
    with tempfile.TemporaryDirectory(prefix="av_marker_") as td:
        td = pathlib.Path(td)
        vraw, araw = td / "v.gray", td / "a.f32"
        fc = (f"[0:v:0]crop={W}:{ROWS}:0:0,format=gray,showinfo[v];"
              "[0:a:0]pan=mono|c0=c0,aformat=sample_fmts=flt,ashowinfo[a]")
        args = [ffmpeg, "-hide_banner", "-nostats", "-y", "-copyts", "-i", str(ts), "-filter_complex", fc,
                "-map", "[v]", "-f", "rawvideo", str(vraw)]
        has_audio = "audio" in run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(ts)]).stdout
        if has_audio:
            args += ["-map", "[a]", "-f", "f32le", str(araw)]
        else:
            args[args.index("-filter_complex") + 1] = fc.split(";")[0]
        r = run(args)
        vts, ats = [], []
        for line in r.stderr.splitlines():
            m = re.search(r"Parsed_showinfo.*pts_time:(\S+)", line)
            if m:
                vts.append(float(m.group(1)))
            m = re.search(r"Parsed_ashowinfo.*pts_time:(\S+).*?rate:(\d+).*?nb_samples:(\d+)", line)
            if m:
                ats.append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
        vb = vraw.read_bytes() if vraw.exists() else b""
        pcm = array.array("f")
        if araw.exists():
            pcm.frombytes(araw.read_bytes())
    n = min(len(vts), len(vb) // (W * ROWS))
    frames = []
    for k in range(n):
        mk = marker(vb[k * W * ROWS:(k + 1) * W * ROWS])
        idx = mk & 0xFFFFFF
        frames.append((vts[k], idx // 2 if interlaced else idx))
    marks = [t for t, f in frames if f % 25 == 0]
    pulses, pos = [], 0
    for t0, sr, cnt in ats:
        blk = pcm[pos:pos + cnt]
        best = None
        for i, x in enumerate(blk):
            if abs(x) > 0.5 and (best is None or abs(x) > abs(blk[best])):
                best = i
        if best is not None:
            t = t0 + best / sr
            if not pulses or t - pulses[-1] > 0.5:
                pulses.append(t)
        pos += cnt
    pairs = []
    for tp in pulses:
        near = min(marks, key=lambda tm: abs(tm - tp), default=None)
        if near is not None and abs(near - tp) < 0.5:
            pairs.append({"t": round(near, 4), "offset_ms": round((tp - near) * 1000, 3)})
    offs = [p["offset_ms"] for p in pairs]
    seq_jumps = sum(1 for (_, a), (_, b) in zip(frames, frames[1:]) if b != a + 1)
    return {"ok": bool(pairs), "file": str(ts), "width": w, "height": h, "interlaced": interlaced, "frames": n,
            "marker_frames": len(marks), "pulses": len(pulses), "pairs": len(pairs), "seq_jumps": seq_jumps,
            "offset_mean_ms": round(statistics.mean(offs), 3) if offs else None,
            "offset_max_abs_ms": round(max(abs(x) for x in offs), 3) if offs else None,
            "offsets_ms": offs[:400], "ffmpeg_rc": r.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ts")
    ap.add_argument("--interlaced", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--json")
    a = ap.parse_args()
    res = analyze(a.ts, None if a.interlaced == "auto" else a.interlaced == "1")
    txt = json.dumps(res, indent=1)
    if a.json:
        pathlib.Path(a.json).write_text(txt)
    print(json.dumps({k: v for k, v in res.items() if k != "offsets_ms"}))
    sys.exit(0 if res["ok"] else 2)


if __name__ == "__main__":
    main()
