#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""`vramflow-run` funkciniai testai (reikalavimai docs/reikalavimai/2026-11-19-operatoriaus-kelias.md OK-2; kviečiami iš
functional.py --only). Prižiūrėtojas paleidžiamas BE LD_LIBRARY_PATH (anti-tikslas): build medžio vramflow-func turi RUNPATH.

Išėjimas tikrinamas uždaru ratu: prižiūrėtojo TS → (UDP relė, matuoja pirmą paketą ir tarpus) → antras ingest →
verify --mode ingest (A/V ≤ 1 laukas, kaip fenc_ingest_audio). Medžiaga — tik sintetinė (scripts/linux/62-make-av-clip.sh).

frun_logo_udp   — OK-2.1: 1 UDP įėjimas, logo builtin, 1 UDP išėjimas (127.0.0.2): ≤ 5 s nuo paleidimo iki pirmo TS paketo;
                  A/V ≤ 20 ms; logotipo sritis (luma) prieš CPU etaloną (`vramflow-func clip` šaltinio kadras + builtin logotipo
                  blend, ta pati formulė kaip media::ref) PSNR ≥ LOGO_MIN_DB, o be logotipo — bent LOGO_SEP_DB mažesnis; sritis
                  už logotipo prieš šaltinį ≥ LOGO_MIN_DB; enc be savų praradimų; SIGTERM → exit 0 ≤ 5 s.
frun_multi4     — OK-2.2 ir OK-2.7 k. 2 ([logo], [composite]): 4 įėjimai (siuntėjai nuo 0 / 20 / 40 / 60 s klipo) → 4 išėjimai: kiekviename A/V ≤ 20 ms, ir išėjimas n
                  neša įėjimo n turinį (pirmos išėjimo žymės skiriasi ~1000 laukų iš eilės).
frun_stop       — OK-2.4: SIGTERM → prižiūrėtojas ir visi vaikai baigiasi ≤ 5 s, domenas ir jo nuomos katalogas pašalinti, exit 0.
frun_child_kill — OK-2.5: kill -9 logo → perkrautas, išėjimo spraga (enc stats „gaps“ po nužudymo) ≤ 50 kadrų (2 s) ir > 0
                  (suskaičiuota statistikoje); antras paleidimas: 6 kritimai per 60 s → exit 2, stderr „logo nukrito 6 kartus per
                  60 s“, vaikų nelieka, domenas pašalintas.
frun_source_gap — OK-2.6: SRT šaltinis (ffmpeg caller → ingest listener) dingsta 10 s ir grįžta: išėjimo TS nenutrūksta (tarpas tarp
                  paketų < 1 s; ingest / logo --on-loss freeze — paskutinis kadras ≤ 1 s, tada juoda, garsas — tyla), nė vienas
                  vaikas neperkrautas, po grįžimo A/V ≤ 20 ms kiekvienam impulsui (be išimčių),
                  ingest input_reconnects ≥ 1; spraga matoma: ingest gap_fields / gap_frozen / gap_black / audio_silent_samples.
fenc_startup    — OK-2.1 dalis: gen jau veikia → enc --output: nuo Popen iki pirmo TS paketo ≤ 2 s (be fiksuoto 1,5 s laukimo).
frun_no_source  — fokusas 4: įėjimas be duomenų → po input_wait_s exit 2, žinutė su redaguotu URL (užklausos reikšmė — „***“).
frun_bad_config — OK-2.3: nežinomas laukas / trūkstami outputs / netinkamas JSON → exit 2 be vaikų ir be domeno, žinutė su keliu.
frun_composite  — OK-2.7: composite su libvramflow-gfx sluoksniu (gfxgen „synthetic“ — OGraf atvaizdavimo įrankio WSL nėra,
                  examples/ograf/README.md): ≤ 5 s iki pirmo paketo, A/V ≤ 20 ms, sluoksnis matomas (srities PSNR prieš šaltinį
                  bent LOGO_SEP_DB mažesnis nei už jos), SIGTERM → exit 0 ≤ 5 s.
frun_udp_srt    — OK-1 + OK-2 vienu paleidimu: išėjimai udp://239.1.1.1 (multicast; be maršruto — 127.0.0.2) ir SRT listener su
                  slaptažodžiu iš env: ffprobe abiejuose mato H.264 1920×1080 ir AAC 48 kHz.
"""
import functools
import json
import os
import pathlib
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

import functional as F  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
RUN = REPO / "tools" / "vramflow-run" / "vramflow-run"
ENV_PLAIN = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}  # be LD_LIBRARY_PATH (anti-tikslas)
TESTS = {}
FIRST_PACKET_S = 5.0     # OK-2.1
STOP_S = 5.0             # OK-2.4
RECOVER_FRAMES = 50      # OK-2.5: 2 s × 25 kadrai
AV_MS = 20.0             # 1 laukas
LOGO_MIN_DB = 30.0       # logotipo sritis prieš CPU etaloną (du kodavimai ~8 Mbit/s)
LOGO_SEP_DB = 10.0       # be logotipo (ar sluoksnio) PSNR turi būti bent tiek mažesnis
LOGO_X, LOGO_Y, LOGO_W, LOGO_H = 1488, 64, 384, 216  # builtin logotipo numatytoji vieta 1080 rastrui (logo.cpp)
DUMPS = 8


def tai_ns():
    return time.clock_gettime_ns(time.CLOCK_TAI)


class Relay(threading.Thread):
    """UDP relė: priima prižiūrėtojo išėjimą (host:port), persiunčia į 127.0.0.1:fwd; pirmo paketo laikas, tarpai > 0,2 s."""

    def __init__(self, host, port, fwd):
        super().__init__(daemon=True)
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sk.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sk.bind((host, port))
        self.sk.settimeout(0.1)
        self.fwd = ("127.0.0.1", fwd)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.first = None
        self.last = None
        self.packets = 0
        self.gaps = []  # (monotonic laikas, tarpas s) — tarpai > 0,2 s po pirmo paketo
        self.halt = False
        self.start()

    def run(self):
        while not self.halt:
            try:
                d = self.sk.recv(65536)
            except socket.timeout:
                continue
            now = time.monotonic()
            if self.first is None:
                self.first = now
            elif now - self.last > 0.2:
                self.gaps.append((now, now - self.last))
            self.last = now
            self.packets += 1
            self.out.sendto(d, self.fwd)

    def stop(self):
        self.halt = True
        self.join(2)
        self.sk.close()
        self.out.close()


class Run:
    """vramflow-run kaip procesas: konfigūracija <out>/<tag>/cfg.json, stderr — run.err, vaikų santraukos — out/<vaikas>/."""

    def __init__(self, out, tag, cfg, env_extra=None):
        self.dir = pathlib.Path(out) / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg_path = self.dir / "cfg.json"
        self.cfg_path.write_text(cfg if isinstance(cfg, str) else json.dumps(cfg, indent=1), encoding="utf-8")
        self.err_path = self.dir / "run.err"
        self.outd = self.dir / "out"
        env = dict(ENV_PLAIN, **(env_extra or {}))
        self.t0 = time.monotonic()
        self.p = subprocess.Popen([sys.executable, str(RUN), "--exe", F.FUNC_BIN, "--out", str(self.outd), str(self.cfg_path)],
                                  stdout=subprocess.DEVNULL, stderr=open(self.err_path, "w"), env=env)
        self.rc = None
        self.stop_s = None
        self.kids_at_stop = []
        self._domain = None

    def text(self):
        return self.err_path.read_text(errors="replace") if self.err_path.exists() else ""

    def domain(self):
        if self._domain is None:
            m = re.search(r"domenas (\S+/vf-run-[0-9a-f]{8})", self.text())
            if m:
                self._domain = m.group(1)
                F.DOMAINS.append(pathlib.Path(self._domain))  # jei prižiūrėtojas nepašalintų — pašalins functional.py
        return self._domain

    def children(self):
        try:
            return [int(x) for x in pathlib.Path(f"/proc/{self.p.pid}/task/{self.p.pid}/children").read_text().split()]
        except (OSError, ValueError):
            return []

    def child(self, sub):
        for pid in self.children():
            try:
                argv = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if len(argv) > 1 and argv[1].decode() == sub:
                return pid
        return None

    def wait_child(self, sub, timeout, not_pid=None):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            pid = self.child(sub)
            if pid is not None and pid != not_pid:
                return pid
            if self.p.poll() is not None:
                return None
            time.sleep(0.05)
        return None

    def wait(self, timeout):
        try:
            self.rc = self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.rc = None
        return self.rc

    def stop(self, timeout=30):
        """SIGTERM ir laukti; stop_s — trukmė iki prižiūrėtojo pabaigos, kids_at_stop — vaikų pid sustabdymo metu."""
        if self.p.poll() is None:
            self.kids_at_stop = self.children()
            t = time.monotonic()
            self.p.send_signal(signal.SIGTERM)
            self.wait(timeout)
            self.stop_s = time.monotonic() - t
            if self.rc is None:
                self.p.kill()
                self.rc = self.p.wait()
        elif self.rc is None:
            self.rc = self.p.returncode
        return self.rc

    def summary(self, name, sub):
        f = self.outd / name / f"summary-{sub}.json"
        try:
            return json.loads(f.read_text())
        except (OSError, ValueError):
            return None

    def stats(self, name):
        f = self.outd / name / "stats.jsonl"
        try:
            return [json.loads(x) for x in f.read_text().splitlines() if x.startswith("{")]
        except (OSError, ValueError):
            return []


def alive(pid):
    try:
        st = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return st.rsplit(")", 1)[1].split()[0] != "Z"


def stop_checks(r, problems, want_rc=0):
    """Bendros sustabdymo patikros: exit kodas, ≤ STOP_S, vaikų nelieka, domenas ir nuomos katalogas pašalinti."""
    if r.rc != want_rc:
        problems.append(f"vramflow-run exit {r.rc} (laukta {want_rc})")
    if r.stop_s is not None and r.stop_s > STOP_S:
        problems.append(f"sustabdymas {r.stop_s:.2f} s > {STOP_S} s")
    left = [p for p in r.kids_at_stop if alive(p)]
    if left:
        problems.append(f"po sustabdymo gyvi vaikai {left}")
    d = r.domain()
    if d is None:
        problems.append("stderr nėra domeno kelio")
    elif pathlib.Path(d).exists() or pathlib.Path(os.path.realpath(d) + ".vramflow").exists():
        problems.append(f"domenas {d} (ar jo nuomos katalogas) nepašalintas")


def logo_cfg(inputs, outputs, process=None, **over):
    c = {"version": 1, "inputs": [{"url": u} for u in inputs], "process": process or {"logo": "builtin"},
         "outputs": [{"url": u} for u in outputs]}
    c.update(over)
    return c


# --- logotipo srities etalonas (CPU, ta pati formulė kaip src/media/media_ref.cpp) -------------------------------------

def _rhu(v):
    return int(v + 0.5) if v >= 0 else -int(-v + 0.5)


def builtin_logo_y():
    """media::ref::builtin_logo(): (y10[ly][lx], a8[ly][lx]) 384×216."""
    W, H, B, band = 384, 216, 8, 48
    iw = W - 2 * B
    ys, als = [], []
    for y in range(H):
        ry, ra = [], []
        for x in range(W):
            if B <= y < H - B and B <= x < W - B:
                ix = x - B
                if y < B + band:
                    r = g = b = 255
                else:
                    s = ix * 3 // iw
                    r, g, b = (255 if s == 0 else 0), (255 if s == 1 else 0), (255 if s == 2 else 0)
                a = ix * 255 // (iw - 1)
            else:
                r = g = b = a = 0
            yp = 0.2126 * r / 255 + 0.7152 * g / 255 + 0.0722 * b / 255
            ry.append(min(940, max(64, _rhu(64.0 + 876.0 * yp))))
            ra.append(a)
        ys.append(ry)
        als.append(ra)
    return ys, als


def v210_luma(data, k, x0, w):
    """v210 lauko eilutės k luma [x0, x0 + w) (x0 ir w — 6 kartotiniai)."""
    wpl = len(data) // 540 // 4
    g0, ng = x0 // 6, w // 6
    words = struct.unpack_from(f"<{ng * 4}I", data, (k * wpl + g0 * 4) * 4)
    out = []
    for g in range(ng):
        w0, w1, w2, w3 = words[4 * g:4 * g + 4]
        out += [(w0 >> 10) & 1023, w1 & 1023, (w1 >> 20) & 1023, (w2 >> 10) & 1023, w3 & 1023, (w3 >> 20) & 1023]
    return out


@functools.lru_cache(maxsize=64)
def clip_frame(m):
    """Šaltinio kadras (NV12) su lauku m: `vramflow-func clip` — tas pats generatorius kaip 62-make-av-clip.sh."""
    r = subprocess.run([F.FUNC_BIN, "clip", "--format", "1080i50", "--frames", "1", "--first-field", str(m & ~1), "--complexity", "2"],
                       capture_output=True, env=F.ENV, timeout=60)
    return r.stdout if r.returncode == 0 and len(r.stdout) == 1920 * 1080 * 3 // 2 else None


def region_psnr(dump_dir, n, x0, y0, w, h, logo=None):
    """Flow'o n iškirptų laukų (ingest_<n>_m<žymė>.v210) sritis prieš etaloną: šaltinio kadras (clip) su logo blend (logo —
    (y10, a8)) arba be jo. Grąžina (psnr_dB, laukų skaičius, pikselių skaičius) arba (None, 0, 0)."""
    import math
    se, cnt, fields = 0, 0, 0
    for f in sorted(pathlib.Path(dump_dir).glob(f"ingest_{n}_m*.v210")):
        m = int(re.search(r"_m(\d+)\.v210$", f.name).group(1))
        src = clip_frame(m)
        if src is None:
            continue
        data = f.read_bytes()
        p = m & 1
        fields += 1
        for row in range(y0 + ((y0 + p) & 1), y0 + h, 2):  # kadro eilutės šio lauko paritetu
            k = (row - p) // 2
            out = v210_luma(data, k, x0, w)
            for i, yv in enumerate(out):
                s10 = src[row * 1920 + x0 + i] * 4
                ref = s10
                if logo is not None:
                    ly, lx = row - LOGO_Y, x0 + i - LOGO_X
                    ref = (s10 * (255 - logo[1][ly][lx]) + logo[0][ly][lx] * logo[1][ly][lx] + 127) // 255
                se += (yv - ref) ** 2
                cnt += 1
    if not cnt:
        return None, 0, 0
    mse = se / cnt
    return (99.0 if mse == 0 else round(10 * math.log10(1023 ** 2 / mse), 2)), fields, cnt


# --- bendras uždaro rato atvejis ---------------------------------------------------------------------------------------

class Chain:
    """Siuntėjai → vramflow-run → (relė) → antras ingest (I.run_case). stop() — SIGTERM prižiūrėtojui ir viskas kita."""

    def __init__(self, out, tid, cfg, senders, relays=(), env_extra=None):
        self.relays = [Relay(*r) for r in relays]
        self.snd = [s() for s in senders]
        self.run = Run(out, f"{tid}/run", cfg, env_extra)

    def stop(self):
        self.run.stop()
        for s in self.snd:
            s.stop()
        for r in self.relays:
            r.stop()


def closed_loop(out, tid, cfg, senders, mids, T, relays=(), dumps=DUMPS, env_extra=None):
    """mids — antro ingest įėjimų portai (127.0.0.1). Grąžina (si, sv, ri, chain, dump_dir)."""
    import ingest as I
    dump = pathlib.Path(out) / tid / "dump"
    dump.mkdir(parents=True, exist_ok=True)
    holder = []

    def start():
        holder.append(Chain(out, tid, cfg, senders, relays, env_extra))
        return holder[0]

    si, sv, ri, _err, _snd = I.run_case(out, tid, [f"udp://127.0.0.1:{q}" for q in mids], [start], T,
                                        verify_args=("--dump-dir", dump, "--dump-count", dumps))
    return si, sv, ri, (holder[0] if holder else None), dump


def first_packet(ch, problems):
    r = ch.relays[0] if ch.relays else None
    if r is None or r.first is None:
        problems.append("išėjimo paketų nebuvo")
        return None
    s = round(r.first - ch.run.t0, 2)
    if s > FIRST_PACKET_S:
        problems.append(f"pirmas TS paketas po {s} s > {FIRST_PACKET_S} s")
    return s


def enc_losses(run):
    f = F.flow_ctr(run.summary("enc", "enc"), 0)
    return {k: f[k] for k in F.ENC_LOSS if f.get(k)}


def need_clips(tid):
    import ingest as I
    why = I.ensure_clips()
    return [F.result(tid, "FAIL", why=why)] if why else None


# --- testai ------------------------------------------------------------------------------------------------------------

def frun_logo_udp(out):
    tid = "frun_logo_udp"
    if (r := need_clips(tid)):
        return r
    import ingest as I
    F.gpu_mode("gpu")
    T = 30
    src, p_out, mid = F.free_port(), F.free_port(), F.free_port()
    cfg = logo_cfg([f"udp://127.0.0.1:{src}"], [f"udp://127.0.0.2:{p_out}?pkt_size=1316"])
    snd = [lambda: I.Sender.ts(out, f"{tid}/snd", I.CLIPS / "av_1080i50.ts", src, "--loop")]
    si, sv, ri, ch, dump = closed_loop(out, tid, cfg, snd, [mid], T, relays=[("127.0.0.2", p_out, mid)])
    if ch is None:
        return [F.result(tid, "FAIL", why="grandinė nepaleista")]
    problems, a, ratio = I.base_checks(si, sv, ri, AV_MS, T)
    fp = first_packet(ch, problems)
    stop_checks(ch.run, problems)
    if (el := enc_losses(ch.run)):
        problems.append(f"enc savi praradimai {el}")
    logo = builtin_logo_y()
    p_logo, nf, npx = region_psnr(dump, 0, LOGO_X, LOGO_Y, LOGO_W, LOGO_H, logo)
    p_nologo, _, _ = region_psnr(dump, 0, LOGO_X, LOGO_Y, LOGO_W, LOGO_H)
    p_out_rgn, _, _ = region_psnr(dump, 0, LOGO_X, LOGO_Y + 300, LOGO_W, LOGO_H)
    if p_logo is None or p_logo < LOGO_MIN_DB:
        problems.append(f"logotipo sritis prieš CPU etaloną {p_logo} dB < {LOGO_MIN_DB} ({nf} laukai)")
    if p_logo is not None and (p_nologo is None or p_nologo > p_logo - LOGO_SEP_DB):
        problems.append(f"be logotipo {p_nologo} dB — neatskiria (> {p_logo} − {LOGO_SEP_DB})")
    if p_out_rgn is None or p_out_rgn < LOGO_MIN_DB:
        problems.append(f"sritis už logotipo prieš šaltinį {p_out_rgn} dB < {LOGO_MIN_DB}")
    line = (f"pirmas paketas {fp} s; logotipo PSNR {p_logo} dB (be logotipo {p_nologo}, už jo {p_out_rgn}; {nf} laukai, {npx} px); "
            f"sustabdymas {ch.run.stop_s and round(ch.run.stop_s, 2)} s exit {ch.run.rc} | " + I.summary_line(si, sv, a, ratio))
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") + line,
                     first_packet_s=fp, psnr_logo_db=p_logo, psnr_nologo_db=p_nologo, psnr_outside_db=p_out_rgn,
                     av_max_abs_ms=a["max_abs_ms"], stop_s=ch.run.stop_s)]


MULTI4_PROCESS = {"logo": {"logo": "builtin"},
                  "composite": {"composite": {"layers": [{"source": "synthetic", "geom": "logo", "x": LOGO_X, "y": LOGO_Y}]}}}


def frun_multi4(out):
    """OK-2.2 (logo) ir OK-2.7 k. 2 (composite): tas pats 4 → 4 atvejis abiem apdorojimo variantams."""
    if (r := need_clips("frun_multi4")):
        return r
    return [x for kind in MULTI4_PROCESS for x in multi4_case(out, kind)]


def multi4_case(out, kind):
    tid = f"frun_multi4[{kind}]"
    import ingest as I
    F.gpu_mode("gpu")
    T, N = 30, 4
    srcs = [F.free_port() for _ in range(N)]
    mids = [F.free_port() for _ in range(N)]
    cfg = logo_cfg([f"udp://127.0.0.1:{p}" for p in srcs], [f"udp://127.0.0.1:{q}?pkt_size=1316" for q in mids],
                   process=MULTI4_PROCESS[kind])
    tid_dir = f"frun_multi4-{kind}"
    snd = [(lambda k=k: I.Sender.ts(out, f"{tid_dir}/snd{k}", I.CLIPS / "av_1080i50.ts", srcs[k], "--loop", "--start-s", 20 * k))
           for k in range(N)]
    si, sv, ri, ch, dump = closed_loop(out, tid_dir, cfg, snd, mids, T, dumps=1)
    if ch is None:
        return [F.result(tid, "FAIL", why="grandinė nepaleista")]
    problems, lines = [], []
    for n in range(N):
        pr, a, ratio = I.base_checks(si, sv, ri, AV_MS, T, n=n)
        problems += [f"[{n}] {p}" for p in pr]
        lines.append(f"[{n}] A/V maks. |{a['max_abs_ms']}| ms ({a['pairs']}), video {ratio:.4%}")
    stop_checks(ch.run, problems)
    if (el := enc_losses(ch.run)):
        problems.append(f"enc savi praradimai {el}")
    # įėjimas n → išėjimas n: siuntėjo n medžiaga pradedama nuo 20·n s → pirmoji išėjimo n žymė ≈ m0 + 1000·n
    marks = []
    for n in range(N):
        fs = sorted(pathlib.Path(dump).glob(f"ingest_{n}_m*.v210"))
        marks.append(int(re.search(r"_m(\d+)\.v210$", fs[0].name).group(1)) if fs else None)
    if None in marks or any(abs((marks[n] - marks[0]) - 1000 * n) > 150 for n in range(N)):
        problems.append(f"išėjimų žymės {marks} — ne įėjimų tvarka (laukta ≈ m0 + 1000·n)")
    if kind == "composite" and ("composite paleistas" not in ch.run.text() or "gfx paleistas" not in ch.run.text()):
        problems.append("composite / gfx nepaleisti")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                     f"žymės {marks}; " + "; ".join(lines) + f"; sustabdymas {ch.run.stop_s and round(ch.run.stop_s, 2)} s",
                     marks=marks)]


def started_chain(out, tid, cfg, src_clip_port, relay_to=None, warm_s=6.0):
    """Siuntėjas + vramflow-run (+ relė); laukia pirmo išėjimo paketo ir dar warm_s. Grąžina (run, sender, relay)."""
    import ingest as I
    snd = I.Sender.ts(out, f"{tid}/snd", I.CLIPS / "av_1080i50.ts", src_clip_port, "--loop")
    relay = Relay(*relay_to) if relay_to else None
    run = Run(out, f"{tid}/run", cfg)
    end = time.monotonic() + 30
    while time.monotonic() < end and run.p.poll() is None and (relay is None or relay.first is None):
        time.sleep(0.1)
    time.sleep(warm_s)
    return run, snd, relay


def frun_stop(out):
    tid = "frun_stop"
    if (r := need_clips(tid)):
        return r
    F.gpu_mode("gpu")
    src, p_out = F.free_port(), F.free_port()
    run, snd, relay = started_chain(out, tid, logo_cfg([f"udp://127.0.0.1:{src}"], [f"udp://127.0.0.2:{p_out}"]), src,
                                    ("127.0.0.2", p_out, F.free_port()))
    problems = []
    try:
        kids = {s: run.child(s) for s in ("ingest", "logo", "enc")}
        if None in kids.values() or relay.first is None:
            problems.append(f"grandinė neveikė prieš sustabdymą: vaikai {kids}, paketai {relay.packets}")
        run.stop()
    finally:
        run.stop()
        snd.stop()
        relay.stop()
    stop_checks(run, problems)
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                     f"SIGTERM → exit {run.rc} per {run.stop_s and round(run.stop_s, 2)} s; vaikai {run.kids_at_stop} — "
                     f"gyvų po sustabdymo {[p for p in run.kids_at_stop if alive(p)]}; domenas {run.domain()} pašalintas",
                     stop_s=run.stop_s)]


def frun_child_kill(out):
    tid = "frun_child_kill"
    if (r := need_clips(tid)):
        return r
    F.gpu_mode("gpu")
    res = []
    # A: vienas kill -9 → perkrovimas, išėjimo spraga ≤ 2 s ir matoma enc statistikoje
    src, p_out = F.free_port(), F.free_port()
    cfg = logo_cfg([f"udp://127.0.0.1:{src}"], [f"udp://127.0.0.2:{p_out}"], restart={"max": 5, "window_s": 60})
    run, snd, relay = started_chain(out, f"{tid}-a", cfg, src, ("127.0.0.2", p_out, F.free_port()))
    problems = []
    restart_s, k_tai = None, 0
    try:
        pid = run.child("logo")
        if pid is None:
            problems.append("logo nepaleistas")
        else:
            t_kill, k_tai = time.monotonic(), tai_ns()
            os.kill(pid, signal.SIGKILL)
            new = run.wait_child("logo", 10, not_pid=pid)
            restart_s = round(time.monotonic() - t_kill, 2) if new else None
            if new is None:
                problems.append("logo neperkrautas per 10 s")
            time.sleep(8)
        run.stop()
    finally:
        run.stop()
        snd.stop()
        relay.stop()
    stop_checks(run, problems)
    lines = [x for x in run.stats("enc") if x.get("tai", 0) >= k_tai]
    gaps = sum((x.get("flows") or [{}])[0].get("gaps", 0) for x in lines)
    before = [x for x in run.stats("enc") if pid and x.get("tai", 0) < k_tai]
    gaps_before = sum((x.get("flows") or [{}])[0].get("gaps", 0) for x in before[3:])  # be paleidimo sekundžių
    if not 0 < gaps <= RECOVER_FRAMES:
        problems.append(f"išėjimo spraga po kill -9: {gaps} kadrų (laukta 1…{RECOVER_FRAMES}: atsistato ≤ 2 s ir suskaičiuota)")
    if gaps_before:
        problems.append(f"spragos prieš nužudymą {gaps_before}")
    txt = run.text()
    if "logo nukrito (signalas SIGKILL" not in txt or "perkraunamas (1/5 per 60 s)" not in txt:
        problems.append("stderr nėra perkrovimo eilutės")
    res.append(F.result(f"{tid}[restart]", "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                        f"logo perkrautas po {restart_s} s; enc spragos po nužudymo {gaps} kadrų ({gaps / 25:.2f} s); "
                        f"sustabdymas exit {run.rc} per {run.stop_s and round(run.stop_s, 2)} s", gap_frames=gaps, restart_s=restart_s))
    # B: 6 kritimai per 60 s → exit 2 su priežastimi
    src, p_out = F.free_port(), F.free_port()
    cfg = logo_cfg([f"udp://127.0.0.1:{src}"], [f"udp://127.0.0.2:{p_out}"], restart={"max": 5, "window_s": 60})
    run, snd, relay = started_chain(out, f"{tid}-b", cfg, src, ("127.0.0.2", p_out, F.free_port()), warm_s=2)
    problems = []
    kills = 0
    try:
        kids = run.children()
        pid = run.child("logo")
        t0 = time.monotonic()
        while pid is not None and kills < 6:
            os.kill(pid, signal.SIGKILL)
            kills += 1
            if kills < 6:
                pid = run.wait_child("logo", 10, not_pid=pid)
                time.sleep(1.0)
                kids += [p for p in run.children() if p not in kids]
        run.wait(15)
        t_end = time.monotonic() - t0
        run.kids_at_stop = kids
    finally:
        run.stop()
        snd.stop()
        relay.stop()
    stop_checks(run, problems, want_rc=2)
    txt = run.text()
    if kills != 6:
        problems.append(f"nužudyta {kills} kartų (laukta 6)")
    if "logo nukrito 6 kartus per 60 s" not in txt:
        problems.append("stderr nėra „logo nukrito 6 kartus per 60 s“")
    if txt.count("perkraunamas (") != 5:
        problems.append(f"perkrovimų {txt.count('perkraunamas (')} (laukta 5)")
    res.append(F.result(f"{tid}[limit]", "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                        f"6 kill -9 → exit {run.rc} po {t_end:.1f} s; " +
                        next((x for x in txt.splitlines() if "baigta, exit" in x), "")))
    return res


def frun_source_gap(out):
    tid = "frun_source_gap"
    if (r := need_clips(tid)):
        return r
    import ingest as I
    protos = subprocess.run(["ffmpeg", "-hide_banner", "-protocols"], capture_output=True, text=True).stdout.split()
    if "srt" not in protos:
        return [F.result(tid, "SKIP", why="sisteminis FFmpeg be srt protokolo")]
    F.gpu_mode("gpu")
    UP1, GAP, UP2 = 15, 10, 25
    T = UP1 + GAP + UP2
    port, p_out, mid = F.free_port(), F.free_port(), F.free_port()  # SRT — per UDP
    url = f"srt://127.0.0.1:{port}"
    cfg = logo_cfg([f"{url}?mode=listener&latency=120000"], [f"udp://127.0.0.2:{p_out}?pkt_size=1316"])

    def sender(tag):
        return I.Sender(out, f"{tid}/{tag}", ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
                                              "-i", str(I.CLIPS / "av_1080i50.ts"), "-c", "copy", "-f", "mpegts",
                                              f"{url}?mode=caller&latency=120000"])
    d2 = F.fresh_domain(tid)
    v = F.vcommon("gpu", "cuda-linear")
    ing2 = vf = run = None
    s1 = s2 = None
    relay = Relay("127.0.0.2", p_out, mid)
    problems = []
    try:
        ing2 = F.FProc(out, tid, "ingest", d2, "--input", f"udp://127.0.0.1:{mid}", "--duration", T + 25, "--warmup", 6, *v)
        run = Run(out, f"{tid}/run", cfg)
        s1 = sender("snd1")
        vf = F.FProc(out, tid, "verify", d2, "--mode", "ingest", "--flows", 1, "--duration", T + 8, "--warmup", 6, "--lag-fields", 3,
                     "--audio", 1, "--input-wait-ms", 30000, "--impl", "gpu", "--storage", "cuda-linear")
        time.sleep(5 + UP1)
        s1.stop()
        t_gap = time.monotonic()
        time.sleep(GAP)
        s2 = sender("snd2")
        t_back = time.monotonic()
        vf.wait(timeout=T + 90)
        run.stop()
        ri = ing2.wait(timeout=60)
    finally:
        if run is not None:
            run.stop()
        for pr in (vf, ing2):
            if pr is not None:
                pr.term()
        for s in (s1, s2):
            if s is not None:
                s.stop()
        relay.stop()
    sv = F.vsummary(vf, "verify-ingest")
    stop_checks(run, problems)
    a = I.av_stats(sv)
    # po grįžimo: VISI impulsai po didžiausio tarpo tarp jų (= spraga) — be išimčių (peržiūra: −480 ms impulsas vieną kartą,
    # 5 paleidimuose nepasikartojo, priežastis neįrodyta — jei pasikartos, testas krenta, o ne slepia)
    raw = a["raw"]
    jump = max(range(1, len(raw)), key=lambda k: raw[k][0] - raw[k - 1][0], default=None)
    post = raw[jump:] if jump is not None else []
    after = [p[1] / 1000.0 for p in post]
    if len(after) < (UP2 - 12) * 0.6:
        problems.append(f"po grįžimo A/V porų {len(after)}")
    if not after or max(abs(x) for x in after) > AV_MS:
        problems.append(f"po grįžimo |A/V| {max((abs(x) for x in after), default=None)} ms > {AV_MS}")
    gap_window = [g for t, g in relay.gaps if t_gap - 1 <= t <= t_back + 10]
    if relay.first is None or max(gap_window, default=0) >= 1.0:
        problems.append(f"išėjimas nutrūko: tarpai tarp paketų {gap_window} s")
    txt = run.text()
    if "nukrito" in txt:
        problems.append("vaikas perkrautas: " + next(x for x in txt.splitlines() if "nukrito" in x))
    fi = F.flow_ctr(run.summary("ingest", "ingest"), 0)
    if fi.get("input_reconnects", 0) < 1:
        problems.append(f"ingest input_reconnects {fi.get('input_reconnects')}")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                     f"spraga {GAP} s: didžiausias tarpas tarp išėjimo paketų {max(gap_window, default=0):.3f} s; ingest "
                     f"input_reconnects {fi.get('input_reconnects')} gap_fields {fi.get('gap_fields')}; po grįžimo A/V maks. "
                     f"{max((abs(x) for x in after), default=None)} ms ({len(after)} impulsų: {[round(x, 1) for x in after[:14]]}…); ingest gap_frozen "
                     f"{fi.get('gap_frozen')} gap_black {fi.get('gap_black')} audio_silent_samples {fi.get('audio_silent_samples')}; "
                     f"enc missing_frames {F.flow_ctr(run.summary('enc', 'enc'), 0).get('missing_frames', 0)}; sustabdymas exit {run.rc}",
                     av_after_max_ms=max((abs(x) for x in after), default=None),
                     post_pairs=[(p[0] - post[0][0], p[1]) for p in post] if post else [],
                     jumps_post=[(j[0] - post[0][0], j[1]) for j in a["jumps"] if post and -60 <= j[0] - post[0][0] <= 150])]


def frun_no_source(out):
    tid = "frun_no_source"
    F.gpu_mode("gpu")
    port = F.free_port()
    W = 3
    url = f"udp://127.0.0.1:{port}?buffer_size=4242421"
    run = Run(out, f"{tid}/run", logo_cfg([url], [f"udp://127.0.0.2:{F.free_port()}"], input_wait_s=W))
    problems = []
    t0 = time.monotonic()
    kids = []
    try:
        while run.p.poll() is None and time.monotonic() - t0 < W + 30:
            kids += [p for p in run.children() if p not in kids]
            time.sleep(0.05)
        el = time.monotonic() - t0
    finally:
        run.stop()
    run.kids_at_stop = kids
    stop_checks(run, problems, want_rc=2)
    txt = run.text()
    shown = f"udp://127.0.0.1:{port}?buffer_size=***"
    if not W <= el <= W + 10:
        problems.append(f"exit po {el:.1f} s (laukta {W}…{W + 10} s)")
    if f"input_wait_s = {W} s" not in txt or shown not in txt:
        problems.append("žinutėje nėra input_wait_s ar redaguoto URL")
    if "4242421" in txt:
        problems.append("užklausos reikšmė stderr neredaguota")
    msg = next((x for x in txt.splitlines() if "nepristato" in x), "")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                     f"exit {run.rc} po {el:.1f} s: {msg}")]


def frun_bad_config(out):
    tid = "frun_bad_config"
    res = []
    good = logo_cfg(["udp://127.0.0.1:9"], ["udp://127.0.0.2:9"])
    bad_field = json.loads(json.dumps(good))
    bad_field["inputs"][0] = {"urll": "udp://127.0.0.1:9"}
    no_out = {k: v for k, v in good.items() if k != "outputs"}
    for name, cfg, want in (("unknown", bad_field, "inputs[0].urll: nežinomas laukas"), ("no-outputs", no_out, "outputs: privalomas laukas"),
                            ("json", '{"version": 1,\n "inputs": [\n}', "eilutė 3, stulpelis 1")):
        before = set(pathlib.Path("/dev/shm").glob("vf-run-*"))
        run = Run(out, f"{tid}/{name}", cfg)
        kids = []
        while run.p.poll() is None:
            kids += run.children()
            time.sleep(0.01)
        run.wait(10)
        txt = run.text()
        problems = []
        if run.rc != 2:
            problems.append(f"exit {run.rc}")
        if want not in txt:
            problems.append(f"stderr be „{want}“: {txt.strip()[:200]!r}")
        if kids or "paleistas" in txt:
            problems.append(f"paleisti vaikai {kids}")
        if set(pathlib.Path("/dev/shm").glob("vf-run-*")) - before:
            problems.append("sukurtas domenas")
        res.append(F.result(f"{tid}[{name}]", "FAIL" if problems else "PASS",
                            why="; ".join(problems) + (" | " if problems else "") + f"exit {run.rc}: {txt.strip()[:200]}"))
    return res


def frun_composite(out):
    tid = "frun_composite"
    if (r := need_clips(tid)):
        return r
    import ingest as I
    F.gpu_mode("gpu")
    T = 30
    src, p_out, mid = F.free_port(), F.free_port(), F.free_port()
    proc = {"composite": {"layers": [{"source": "synthetic", "geom": "logo", "x": LOGO_X, "y": LOGO_Y}]}}
    cfg = logo_cfg([f"udp://127.0.0.1:{src}"], [f"udp://127.0.0.2:{p_out}?pkt_size=1316"], process=proc)
    snd = [lambda: I.Sender.ts(out, f"{tid}/snd", I.CLIPS / "av_1080i50.ts", src, "--loop")]
    si, sv, ri, ch, dump = closed_loop(out, tid, cfg, snd, [mid], T, relays=[("127.0.0.2", p_out, mid)])
    if ch is None:
        return [F.result(tid, "FAIL", why="grandinė nepaleista")]
    problems, a, ratio = I.base_checks(si, sv, ri, AV_MS, T)
    fp = first_packet(ch, problems)
    stop_checks(ch.run, problems)
    if (el := enc_losses(ch.run)):
        problems.append(f"enc savi praradimai {el}")
    p_in, nf, _ = region_psnr(dump, 0, LOGO_X, LOGO_Y, LOGO_W, LOGO_H)
    p_outside, _, _ = region_psnr(dump, 0, LOGO_X, LOGO_Y + 300, LOGO_W, LOGO_H)
    if p_outside is None or p_outside < LOGO_MIN_DB:
        problems.append(f"sritis už sluoksnio prieš šaltinį {p_outside} dB < {LOGO_MIN_DB}")
    if p_in is None or p_outside is None or p_in > p_outside - LOGO_SEP_DB:
        problems.append(f"sluoksnio sritis prieš šaltinį {p_in} dB — sluoksnis nematomas (> {p_outside} − {LOGO_SEP_DB})")
    txt = ch.run.text()
    if "composite paleistas" not in txt or "gfx paleistas" not in txt:
        problems.append("composite / gfx nepaleisti")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                     f"pirmas paketas {fp} s; sluoksnio sritis prieš šaltinį {p_in} dB, už jos {p_outside} dB ({nf} laukai); "
                     f"sustabdymas {ch.run.stop_s and round(ch.run.stop_s, 2)} s exit {ch.run.rc} | " + I.summary_line(si, sv, a, ratio),
                     first_packet_s=fp, psnr_layer_db=p_in, psnr_outside_db=p_outside)]


def frun_udp_srt(out):
    tid = "frun_udp_srt"
    if (r := need_clips(tid)):
        return r
    import ingest as I
    F.gpu_mode("gpu")
    mc = subprocess.run(["ip", "route", "get", "239.1.1.1"], capture_output=True, text=True).returncode == 0
    host = "239.1.1.1" if mc else "127.0.0.2"
    srcs = [F.free_port(), F.free_port()]
    p_udp, p_srt = F.free_port(), F.free_port()  # SRT — per UDP
    udp = f"udp://{host}:{p_udp}" + ("?ttl=1" if mc else "")
    srt = f"srt://127.0.0.1:{p_srt}?mode=listener"
    cfg = logo_cfg([f"udp://127.0.0.1:{p}" for p in srcs], [udp, srt])
    cfg["outputs"][1]["srt_passphrase_from"] = f"env:{F.FOUT_PASS_ENV}"
    snd = [I.Sender.ts(out, f"{tid}/snd{k}", I.CLIPS / "av_1080i50.ts", srcs[k], "--loop") for k in range(2)]
    run = Run(out, f"{tid}/run", cfg, {F.FOUT_PASS_ENV: F.FOUT_PASS})
    problems = []
    try:
        with __import__("concurrent.futures").futures.ThreadPoolExecutor(2) as ex:
            fu = ex.submit(F.probe_streams, f"udp://{host}:{p_udp}?timeout=20000000", 40)
            time.sleep(3)  # SRT listener atsiranda, kai enc paleistas
            fs = ex.submit(F.probe_streams, f"srt://127.0.0.1:{p_srt}?mode=caller&passphrase={F.FOUT_PASS}&connect_timeout=15000", 40)
            su, ss = fu.result(), fs.result()
        run.stop()
    finally:
        run.stop()
        for s in snd:
            s.stop()
    stop_checks(run, problems)
    if not F.av_ok(su):
        problems.append(f"ffprobe UDP: {su}")
    if not F.av_ok(ss):
        problems.append(f"ffprobe SRT: {ss}")
    if F.FOUT_PASS in run.text() or F.FOUT_PASS in run.cfg_path.read_text():
        problems.append("slaptažodis žurnale ar konfigūracijoje")
    st = (run.summary("enc", "enc") or {}).get("streams") or [{}, {}]
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                     f"{udp} → {su}; {srt} → {ss}; out_url {[x.get('out_url') for x in st]}, out_bytes "
                     f"{[x.get('out_bytes') for x in st]}; sustabdymas exit {run.rc}", udp=su, srt=ss, multicast=mc)]


ENC_START_S = 2.0  # enc dalis 5 s biudžete (OK-2.1): nuo Popen iki pirmo TS paketo, kai įėjimo flow'ai jau yra


def fenc_startup(out):
    """OK-2.1 paleidimo laikas (enc): gen (video + garsas) jau veikia → enc --output udp: nuo Popen iki pirmo TS paketo ≤ ENC_START_S
    (3 pakartojimai, vertinama maks.). Anksčiau enc po prisijungimo laukdavo fiksuotas 1500 ms („kol koderiai atsidarys“)."""
    tid = "fenc_startup"
    F.gpu_mode("gpu")
    d = F.fresh_domain("fenc-start")
    v = F.vcommon("gpu", "cuda-linear")
    g = F.FProc(out, f"{tid}/gen", "gen", d, "--flows", 1, "--audio", 1, "--duration", 120, *v)
    times, problems = [], []
    try:
        if not F.wait_flows(d, [F.role_flow_id(1, 0), F.role_flow_id(3, 0)]):
            return [F.result(tid, "FAIL", why="gen flow'ai neatsirado")]
        time.sleep(1)
        for k in range(3):
            port = F.free_port()
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.bind(("127.0.0.2", port))
            sk.settimeout(15)
            t0 = time.monotonic()
            e = F.FProc(out, f"{tid}/enc{k}", "enc", d, "--flows", 1, "--audio", 1, "--output", f"udp://127.0.0.2:{port}", *v)
            try:
                sk.recv(2048)
                times.append(round(time.monotonic() - t0, 3))
            except socket.timeout:
                problems.append(f"[{k}] paketų nėra per 15 s")
            finally:
                sk.close()
                e.term()
    finally:
        g.term()
    if times and max(times) > ENC_START_S:
        problems.append(f"enc iki pirmo paketo maks. {max(times)} s > {ENC_START_S} s")
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") +
                     f"enc Popen → pirmas TS paketas: {times} s", times_s=times)]


for _fn in (fenc_startup, frun_logo_udp, frun_multi4, frun_stop, frun_child_kill, frun_source_gap, frun_no_source, frun_bad_config, frun_composite,
            frun_udp_srt):
    TESTS[_fn.__name__] = _fn
