#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""MPEG-TS siuntėjas ingest testams (planas §P3): TS failas → UDP arba RTP (PT 33, MP2T) 127.0.0.1, ritmas pagal PCR.

Režimai (galima derinti):
  realus laikas         — kiekvienas 7 TS paketų datagramas siunčiamas jo PCR laiku (tarp PCR — tiesinė interpoliacija);
  --speed 1.005         — siuntėjo laikrodis greitesnis (pagreitintas dreifas: 1,005 s medijos per 1 s);
  --burst-s 2 --burst-ms 50 — HLS tipo pliūpsniai: kiekvieno 2 s medijos lango paketai išsiunčiami per 50 ms lango PABAIGOJE
                          (kaip segmentas, pasiekiamas tik visas), tarp jų — tyla;
  --stall-every-s 5 --stall-ms 300 — kas 5 s paketai sulaikomi 300 ms ir išsiunčiami iškart (vėluojantys paketai);
  --loop                — failas kartojamas (PTS šoka atgal — kilpos sandūra), --duration S — kiek siųsti (s, sienos laiku);
  --rtp                 — RTP antraštė (PT 33, seka, 90 kHz laikas iš PCR).
Tik sintetinė medžiaga, tik loopback. Išėjimas: 0; statistika (išsiųsta, pliūpsniai, sulaikymai) — stdout JSON.

Naudojimas: ts_burst.py FAILAS.ts --port 20000 [--speed 1.005] [--burst-s 2 --burst-ms 50] [--stall-every-s 5 --stall-ms 300]
            [--loop] [--duration 120] [--rtp] [--start-at TAI_UNIX_S] [--start-s 7.3] [--shift-at-s 20 --shift-ms 600]
  --start-s S           — siųsti nuo medijos sekundės S (prisijungimas vidury srauto: pirmas kadras — ne IDR);
  --shift-at-s T --shift-ms D — nuo sienos sekundės T visi paketai vėluoja D ms visam laikui (latencijos pokytis, perfiksavimas)
  --pcr-continuous      — PCR šuolis > 1 s (sujungti TS) siunčiamas be pauzės / pliūpsnio: sandūra tik PTS / PCR reikšmėse
"""
import argparse
import json
import random
import socket
import struct
import sys
import time

TS = 188
PER_DGRAM = 7
PCR_WRAP = (1 << 33) * 300


def pcr_of(pkt):
    if pkt[0] != 0x47:
        return None
    afc = (pkt[3] >> 4) & 3
    if not (afc & 2) or pkt[4] < 7 or not (pkt[5] & 0x10):
        return None
    b = pkt[6:12]
    base = (b[0] << 25) | (b[1] << 17) | (b[2] << 9) | (b[3] << 1) | (b[4] >> 7)
    ext = ((b[4] & 1) << 8) | b[5]
    return base * 300 + ext


def packet_times(data, continuous=False):
    """Kiekvieno TS paketo laikas, s (nuo pirmo PCR), pagal vieno PID PCR su tiesine interpoliacija. continuous — PCR šuolis
    > 1 s (sujungti TS: sandūra) siuntimo laike nematomas: laikas tęsiasi paskutiniu PCR intervalu."""
    n = len(data) // TS
    pcr_pid = None
    marks = []  # (paketo nr., pcr 27 MHz, ištiesintas)
    last = None
    acc = 0
    for k in range(n):
        p = data[k * TS:(k + 1) * TS]
        v = pcr_of(p)
        if v is None:
            continue
        pid = ((p[1] & 0x1F) << 8) | p[2]
        if pcr_pid is None:
            pcr_pid = pid
        if pid != pcr_pid:
            continue
        if last is not None and v < last and last - v > PCR_WRAP // 2:
            acc += PCR_WRAP
        if continuous and marks and abs(v + acc - marks[-1][1]) > 27_000_000:
            step = marks[-1][1] - marks[-2][1] if len(marks) > 1 else 0
            acc += marks[-1][1] + step - (v + acc)
        last = v
        marks.append((k, v + acc))
    if len(marks) < 2:
        raise SystemExit("TS be PCR")
    t = [0.0] * n
    p0 = marks[0][1]
    for (ka, va), (kb, vb) in zip(marks, marks[1:]):
        for k in range(ka, kb):
            t[k] = (va - p0 + (vb - va) * (k - ka) / (kb - ka)) / 27e6
    for k in range(marks[-1][0], n):  # po paskutinio PCR — paskutinis greitis
        ka, va = marks[-2]
        kb, vb = marks[-1]
        t[k] = (vb - p0 + (vb - va) * (k - kb) / max(1, kb - ka)) / 27e6
    for k in range(0, marks[0][0]):
        t[k] = 0.0
    return t, (marks[-1][1] - p0) / 27e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ts")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--burst-s", type=float, default=0)
    ap.add_argument("--burst-ms", type=float, default=50)
    ap.add_argument("--stall-every-s", type=float, default=0)
    ap.add_argument("--stall-ms", type=float, default=300)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--rtp", action="store_true")
    ap.add_argument("--start-s", type=float, default=0)  # pradėti nuo šios medijos sekundės (prisijungimas vidury srauto)
    ap.add_argument("--shift-at-s", type=float, default=0)  # nuo šios sienos sekundės visi paketai vėluoja --shift-ms visam laikui
    ap.add_argument("--shift-ms", type=float, default=0)
    ap.add_argument("--pcr-continuous", action="store_true")  # sujungtas TS: PCR šuolis siuntimo ritmo nekeičia (sandūra)
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("tik loopback")
    data = open(a.ts, "rb").read()
    data = data[:len(data) // TS * TS]
    times, clip_s = packet_times(data, a.pcr_continuous)
    period = clip_s + (times[1] - times[0] if len(times) > 1 else 0.001) if a.loop else None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
    dest = (a.host, a.port)
    ssrc = random.getrandbits(32)
    seq = random.getrandbits(16)
    sent = bursts = stalls = loops = 0
    n_pk = len(times)
    k = 0
    if a.start_s > 0:  # pirmas siunčiamas paketas — pirmasis, kurio laikas ≥ start_s (TS paketų riba; PES vidurys — demuxeris jį išmes)
        while k < n_pk and times[k] < a.start_s:
            k += 1
        k -= k % PER_DGRAM
    t0 = time.monotonic() + 0.2 - (times[k] if k < n_pk else 0)
    t_end = time.monotonic() + 0.2 + a.duration if a.duration > 0 else None
    shift = 0.0
    base = 0.0  # kilpos laiko poslinkis (medijos s)
    held = []
    next_stall = a.stall_every_s if a.stall_every_s > 0 else None
    stall_until = None
    last_w = None

    def send(buf, media_t):
        nonlocal seq, sent
        if a.rtp:
            ts90 = int(media_t * 90000) & 0xFFFFFFFF
            buf = struct.pack("!BBHII", 0x80, 33, seq & 0xFFFF, ts90, ssrc) + buf
            seq += 1
        try:
            sock.sendto(buf, dest)
        except OSError:
            pass
        sent += 1

    while True:
        if k >= n_pk:
            if not a.loop:
                break
            k = 0
            base += period
            loops += 1
        j = min(k + PER_DGRAM, n_pk)
        media = base + times[k]
        buf = data[k * TS:j * TS]
        k = j
        if a.burst_s > 0:  # lango pabaigoje, išskleista per burst_ms
            w = int(media // a.burst_s)
            frac = (media - w * a.burst_s) / a.burst_s
            when = t0 + ((w + 1) * a.burst_s) / a.speed + frac * a.burst_ms / 1000.0
        else:
            when = t0 + media / a.speed
        if a.shift_ms > 0 and shift == 0.0 and time.monotonic() - (t0 + times[0]) >= a.shift_at_s:
            shift = a.shift_ms / 1000.0  # latencijos pokytis: nuo čia visi paketai vėliau (be pasivijimo)
        when += shift
        if t_end is not None and when > t_end:
            break
        now = time.monotonic()
        if when > now:
            time.sleep(when - now)
        if next_stall is not None:
            wall = time.monotonic() - t0
            if stall_until is None and wall >= next_stall:
                stall_until = wall + a.stall_ms / 1000.0
                stalls += 1
            if stall_until is not None:
                if wall < stall_until:
                    held.append((buf, media))
                    continue
                for hb, hm in held:
                    send(hb, hm)
                held.clear()
                stall_until = None
                next_stall += a.stall_every_s
        send(buf, media)
        if a.burst_s > 0 and w != last_w:
            bursts += 1
            last_w = w
    for hb, hm in held:
        send(hb, hm)
    print(json.dumps({"sent_dgrams": sent, "loops": loops, "stalls": stalls, "bursts": bursts, "clip_s": round(clip_s, 3), "speed": a.speed, "rtp": a.rtp}))


if __name__ == "__main__":
    sys.exit(main())
