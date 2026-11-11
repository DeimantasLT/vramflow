#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""A/V poslinkio matavimas užkoduotame MPEG-TS (lip-sync): gen sync blyksnis ↔ 1 s pyptelėjimas.

gen turinys (src/media/gen_content.hpp, src/func/gen.cpp): kartą per sekundę grain'as, apimantis sveiką TAI sekundę,
turi baltą stačiakampį x [w·1600/1920, w·1856/1920), lauko eilutės 32..95 (kadro eilutės 64..191; interlaced — tik
viršutinis laukas); garsas — 1 kHz −20 dBFS (0,1), o kiekvienos sekundės pirmos 10 ms (s % 48000 < 480) — −6 dBFS (0,5).
Abu prasideda tame pačiame TAI momente, todėl teisingame sraute poslinkis ≈ 0.

Matavimas (sistemos ffmpeg, viena dekodavimo eiga):
  video — blyksnio srities YAVG kiekvienam kadrui (crop + signalstats + metadata=print), blyksnis = YAVG virš
          (mediana + maksimumas) / 2; blyksnio laikas = kadro PTS;
  audio — mono f32 PCM ir kiekvieno dekoduoto kadro PTS (ashowinfo); pyptelėjimo pradžia = pirmas mėginys |x| > 0,3 po
          ≥ 0,5 s tylos virš ribos (sistemingas paklaidos dydis +0,10 ms: sin fazė asin(0,6)/(2π·1 kHz));
  poslinkis = pyptelėjimas − artimiausias blyksnis (|Δ| < 0,5 s), ms: > 0 — garsas vėluoja, < 0 — skuba.
Papildomai: video PTS spragos (> 1,5 kadro — prarasti kadrai), audio paketų PTS monotoniškumas (ffprobe), audio kadrų
PTS spragos, ffmpeg dekodavimo klaidos.

Pastaba: matavimas periodinis (1 s), todėl poslinkis, lygus sveikam sekundžių skaičiui, nematomas; 1 kadro (40 ms)
nuokrypis — matomas visada.

Naudojimas:
  av_offset.py FAILAS.ts [--json OUT.json]                          — esamo TS analizė
  av_offset.py FAILAS.ts --capture-port 23000 --seconds 20 [--json]  — pirma įrašo UDP TS (ts_sink.Port), tada analizuoja
  av_offset.py --selftest                                           — sintetiniai TS su žinomu poslinkiu ir spraga
Išėjimas: 0 — analizė pavyko (ribų netikrina; tai daro kviečiantis testas); 2 — nepavyko išmatuoti; selftest — 0 / 1.
"""
import argparse
import array
import json
import pathlib
import re
import selectors
import statistics
import subprocess
import sys
import tempfile
import time

AAC_FRAME_MS = 1024 / 48.0  # 21,33 ms — vieno AAC kadro trukmė 48 kHz
BEEP_THR = 0.3
BEEP_QUIET_S = 0.5


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe_video(ffprobe, ts):
    r = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "json", str(ts)])
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    num, den = (st.get("r_frame_rate") or "25/1").split("/")
    return st.get("width"), st.get("height"), int(num) / int(den)


def audio_packet_pts(ffprobe, ts):
    r = run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "packet=pts", "-of", "csv=p=0", str(ts)])
    return [int(x) for x in r.stdout.split() if x.strip().lstrip("-").isdigit()]


ERR_RE = re.compile(r"error|corrupt|invalid|non[- ]monoton|concealing|missing picture|no frame", re.I)


def decode(ffmpeg, ts, w, work):
    """Viena ffmpeg eiga: video YAVG su PTS, audio PCM + kadrų PTS. Grąžina (video[(t, yavg)], audio[(t, sr, n)], pcm, klaidos)."""
    x0, x1 = w * 1600 // 1920, w * 1856 // 1920
    vtxt, araw = work / "video_yavg.txt", work / "audio.f32"
    fc = (f"[0:v:0]crop={x1 - x0 - 16}:{191 - 64 - 16}:{x0 + 8}:{64 + 8},signalstats,"
          f"metadata=mode=print:key=lavfi.signalstats.YAVG:file={vtxt}[v];"
          "[0:a:0]pan=mono|c0=c0,aformat=sample_fmts=flt,ashowinfo[a]")
    r = run([ffmpeg, "-hide_banner", "-nostats", "-y", "-i", str(ts), "-filter_complex", fc,
             "-map", "[v]", "-f", "null", "-", "-map", "[a]", "-f", "f32le", str(araw)])
    video, t = [], None
    for line in vtxt.read_text().splitlines() if vtxt.exists() else []:
        m = re.search(r"pts_time:(\S+)", line)
        if m:
            t = float(m.group(1))
        elif line.startswith("lavfi.signalstats.YAVG=") and t is not None:
            video.append((t, float(line.split("=", 1)[1])))
    audio = []
    for line in r.stderr.splitlines():
        if "ashowinfo" in line:
            m = re.search(r"pts_time:(\S+).*?rate:(\d+).*?nb_samples:(\d+)", line)
            if m:
                audio.append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
    pcm = array.array("f")
    if araw.exists():
        pcm.frombytes(araw.read_bytes())
    errs = [ln for ln in r.stderr.splitlines() if ERR_RE.search(ln) and "ashowinfo" not in ln]
    return video, audio, pcm, errs, r.returncode


def flash_times(video):
    if len(video) < 3:
        return [], None
    ys = [y for _, y in video]
    med, hi = statistics.median(ys), max(ys)
    thr = (med + hi) / 2
    if hi - med < 10:  # blyksnio nėra (ne gen turinys arba per trumpas įrašas)
        return [], thr
    return [t for t, y in video if y > thr], thr


def beep_times(audio, pcm):
    out, pos, last_hot = [], 0, None
    for t0, sr, n in audio:
        blk = pcm[pos:pos + n]
        pos += n
        for i, x in enumerate(blk):
            if x > BEEP_THR or x < -BEEP_THR:
                t = t0 + i / sr
                if last_hot is None or t - last_hot > BEEP_QUIET_S:
                    out.append(t)
                last_hot = t
    return out


def gaps(times, step, tol=1.5):
    """Laikai, po kurių seka spraga > tol·step (pvz. prarasti kadrai)."""
    return [(a, round((b - a) / step) - 1) for a, b in zip(times, times[1:]) if b - a > tol * step]


def analyze(ts, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    ts = pathlib.Path(ts)
    w, h, fps = probe_video(ffprobe, ts)
    if not w:
        return {"ok": False, "why": "nėra video srauto"}
    with tempfile.TemporaryDirectory(prefix="av_offset_") as td:
        video, audio, pcm, errs, rc = decode(ffmpeg, ts, w, pathlib.Path(td))
    flashes, thr = flash_times(video)
    beeps = beep_times(audio, pcm)
    pairs = []
    for tf in flashes:
        near = min(beeps, key=lambda tb: abs(tb - tf), default=None)
        if near is not None and abs(near - tf) < 0.5:
            pairs.append({"t": round(tf, 4), "offset_ms": round((near - tf) * 1000, 3)})
    vt = [t for t, _ in video]
    at = [t for t, _, _ in audio]
    apk = audio_packet_pts(ffprobe, ts)
    # 90 kHz su 33 bitų apsivertimu: atgal — kai skirtumas (mod 2^33) yra „neigiamas“ (> 2^32)
    back = sum(1 for a, b in zip(apk, apk[1:]) if ((b - a) % (1 << 33)) > (1 << 32) or b == a)
    a_step = (audio[0][2] / audio[0][1]) if audio else AAC_FRAME_MS / 1000
    offs = [p["offset_ms"] for p in pairs]
    res = {
        "ok": bool(pairs), "file": str(ts), "width": w, "height": h, "fps": fps, "ffmpeg_rc": rc,
        "video_frames": len(video), "video_first_t": vt[0] if vt else None, "audio_frames": len(audio), "flashes": len(flashes), "beeps": len(beeps),
        "flash_threshold": thr, "pairs": pairs,
        "offset_mean_ms": round(statistics.mean(offs), 3) if offs else None,
        "offset_max_abs_ms": round(max(abs(x) for x in offs), 3) if offs else None,
        "video_gaps": [{"t": round(t, 4), "frames": n} for t, n in gaps(vt, 1 / fps)],
        "audio_gaps": [{"t": round(t, 4), "frames": n} for t, n in gaps(at, a_step)],
        "audio_packets": len(apk), "audio_pts_backward": back,
        "audio_frames_monotonic": all(b > a for a, b in zip(at, at[1:])),
        "decode_errors": len(errs), "decode_error_lines": errs[:20],
    }
    if not pairs:
        res["why"] = f"nėra blyksnio–pyptelėjimo porų (flashes={len(flashes)} beeps={len(beeps)})"
    return res


def window(res, t_from=None, t_to=None):
    """Poslinkių suvestinė laiko lange [t_from, t_to) (sekundės TS laiko ašyje)."""
    offs = [p["offset_ms"] for p in res.get("pairs", [])
            if (t_from is None or p["t"] >= t_from) and (t_to is None or p["t"] < t_to)]
    return {"n": len(offs), "mean_ms": round(statistics.mean(offs), 3) if offs else None,
            "max_abs_ms": round(max(abs(x) for x in offs), 3) if offs else None}


def capture(port, seconds, path):
    """UDP TS įrašas per ts_sink.Port (tie patys sync / CC skaitikliai). Grąžina Port.summary()."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import ts_sink
    p = ts_sink.Port(port, str(path))
    sel = selectors.DefaultSelector()
    sel.register(p.sock, selectors.EVENT_READ, p)
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        for _key, _ in sel.select(timeout=0.2):
            while True:
                try:
                    data = p.sock.recv(65536)
                except BlockingIOError:
                    break
                p.feed(data, ts_sink.tai())
    p.dump.close()
    return p.summary()


def make_synthetic(ffmpeg, path, offset_ms, drop_frame=None, seconds=6):
    """Sintetinis TS gen stiliumi: blyksnis sveikoje sekundėje, pyptelėjimas pastumtas offset_ms; drop_frame — išmestas kadras."""
    d = offset_ms / 1000.0
    vf = ("drawbox=x=1600:y=64:w=256:h=128:color=white:t=fill:enable='lt(mod(t+0.001\\,1)\\,0.02)'"
          + (f",select='not(eq(n\\,{drop_frame}))'" if drop_frame is not None else ""))
    aexpr = (f"if(lt(mod(t-{d}+10\\,1)\\,0.01)\\,0.5\\,0.1)*sin(2*PI*1000*(t-{d}))|"
             f"if(lt(mod(t-{d}+10\\,1)\\,0.01)\\,0.5\\,0.1)*sin(2*PI*1000*(t-{d}))")
    r = run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"color=c=0x606060:s=1920x1080:r=25:d={seconds}",
             "-f", "lavfi", "-i", f"aevalsrc={aexpr}:s=48000:d={seconds}",
             "-vf", vf, "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "ultrafast", "-g", "25",
             "-c:a", "aac", "-b:a", "192k", "-f", "mpegts", str(path)])
    return r.returncode == 0, r.stderr


def selftest(ffmpeg, ffprobe):
    ok_all = True
    with tempfile.TemporaryDirectory(prefix="av_offset_selftest_") as td:
        for want, drop in ((0.0, None), (60.0, None), (-40.0, None), (0.0, 80)):
            ts = pathlib.Path(td) / f"s_{int(want)}_{drop}.ts"
            made, err = make_synthetic(ffmpeg, ts, want, drop)
            if not made:
                print(f"[FAIL] selftest: ffmpeg nesugeneravo ({err.strip()[:200]})")
                return 1
            r = analyze(ts, ffmpeg, ffprobe)
            good = (r["ok"] and len(r["pairs"]) >= 4 and abs(r["offset_mean_ms"] - want) <= 1.5
                    and abs(r["offset_max_abs_ms"] - abs(want)) <= 1.5 and r["audio_pts_backward"] == 0
                    and r["audio_frames_monotonic"] and r["decode_errors"] == 0
                    and (len(r["video_gaps"]) == (1 if drop is not None else 0)))
            if drop is not None and r["video_gaps"]:
                good = good and abs(r["video_gaps"][0]["t"] - r["video_first_t"] - (drop - 1) / 25) < 0.005 and r["video_gaps"][0]["frames"] == 1
            ok_all &= good
            print(f"[{'PASS' if good else 'FAIL'}] laukta {want:+.1f} ms, drop={drop}: vidurkis {r.get('offset_mean_ms')} ms, "
                  f"max |Δ| {r.get('offset_max_abs_ms')} ms, porų {len(r.get('pairs', []))}, video spragos {r.get('video_gaps')}, "
                  f"dekodavimo klaidų {r.get('decode_errors')}")
    print(f"SELFTEST: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ts", nargs="?")
    ap.add_argument("--json", default="")
    ap.add_argument("--capture-port", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.ffmpeg, a.ffprobe))
    if not a.ts:
        ap.error("reikia TS failo")
    sink = capture(a.capture_port, a.seconds, a.ts) if a.capture_port else None
    r = analyze(a.ts, a.ffmpeg, a.ffprobe)
    if sink:
        r["sink"] = sink
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(r, indent=1))
    brief = {k: r.get(k) for k in ("ok", "offset_mean_ms", "offset_max_abs_ms", "video_gaps", "audio_gaps", "audio_pts_backward",
                                    "decode_errors", "why")}
    print(json.dumps(brief, ensure_ascii=False))
    sys.exit(0 if r["ok"] else 2)


if __name__ == "__main__":
    main()
