#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""MPEG-TS UDP imtuvas (architektūra GPU funkcijų §5.5): N portų, minimalus TS skeneris.

Kiekvienam portui: paketai, baitai, sync klaidos, CC klaidos (kiekvienam PID su payload), PCR intervalų maksimumas,
bitų sparta, video PES PTS → e2e latencija (atvykimo TAI − PTS, PTS = TAI·90 kHz mod 2^33 — enc naudoja TAI indeksus).
Pasirinktinai įrašo vieno srauto TS į failą (--dump k:FAILAS) tolimesnei kokybės ir lauko tvarkos analizei.

Naudojimas: ts_sink.py --port-base 20000 --flows N --seconds S --out summary.json [--dump 0:/tmp/s0.ts]
"""
import argparse
import json
import selectors
import socket
import statistics
import time

VIDEO_PID = 0x100  # libavformat mpegts numatytasis pirmam srautui


def tai():
    return time.clock_gettime(time.CLOCK_TAI)


class Port:
    def __init__(self, port, dump=None):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.rcvbuf = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.dump = open(dump, "wb") if dump else None
        self.datagrams = self.bytes = self.ts_packets = 0
        self.sync_errors = self.cc_errors = 0
        self.cc = {}
        self.pcr_last = None
        self.pcr_intervals = []
        self.lat_ms = []
        self.pes_count = 0
        self.first_t = self.last_t = None

    def feed(self, data, now):
        if self.first_t is None:
            self.first_t = now
        self.last_t = now
        self.datagrams += 1
        self.bytes += len(data)
        if self.dump:
            self.dump.write(data)
        for off in range(0, len(data) - 187, 188):
            p = data[off:off + 188]
            self.ts_packets += 1
            if p[0] != 0x47:
                self.sync_errors += 1
                continue
            pid = ((p[1] & 0x1F) << 8) | p[2]
            if pid == 0x1FFF:
                continue
            afc = (p[3] >> 4) & 3
            cc = p[3] & 15
            has_payload = afc & 1
            if has_payload:
                prev = self.cc.get(pid)
                if prev is not None and cc != ((prev + 1) & 15):
                    self.cc_errors += 1
                self.cc[pid] = cc
            pos = 4
            if afc & 2:
                alen = p[4]
                if alen and p[5] & 0x10 and alen >= 7:  # PCR
                    b = p[6:12]
                    base = (b[0] << 25) | (b[1] << 17) | (b[2] << 9) | (b[3] << 1) | (b[4] >> 7)
                    if self.pcr_last is not None:
                        self.pcr_intervals.append((now - self.pcr_last) * 1000.0)
                    self.pcr_last = now
                pos = 5 + alen
            if pid == VIDEO_PID and has_payload and (p[1] & 0x40) and pos + 14 <= 188:
                q = p[pos:]
                if q[0] == 0 and q[1] == 0 and q[2] == 1 and (q[7] & 0x80):
                    t = q[9:14]
                    pts = ((t[0] >> 1) & 7) << 30 | t[1] << 22 | (t[2] >> 1) << 15 | t[3] << 7 | t[4] >> 1
                    arr90 = int(now * 90000) % (1 << 33)
                    self.lat_ms.append(((arr90 - pts) % (1 << 33)) / 90.0)
                    self.pes_count += 1

    def summary(self):
        dur = (self.last_t - self.first_t) if self.first_t and self.last_t else 0
        lat = sorted(self.lat_ms)
        q = lambda f: lat[min(len(lat) - 1, int(f * (len(lat) - 1)))] if lat else None
        return {
            "port": self.port, "rcvbuf": self.rcvbuf, "datagrams": self.datagrams, "bytes": self.bytes,
            "ts_packets": self.ts_packets, "sync_errors": self.sync_errors, "cc_errors": self.cc_errors,
            "bitrate_mbps": (self.bytes * 8 / dur / 1e6) if dur > 0 else 0,
            "pcr_interval_max_ms": max(self.pcr_intervals) if self.pcr_intervals else None,
            "pcr_interval_p50_ms": statistics.median(self.pcr_intervals) if self.pcr_intervals else None,
            "video_pes": self.pes_count, "e2e_latency_p50_ms": q(0.5), "e2e_latency_p99_ms": q(0.99),
            "e2e_latency_max_ms": lat[-1] if lat else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-base", type=int, default=20000)
    ap.add_argument("--flows", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--warmup", type=float, default=5, help="pirmų sekundžių latencija neskaičiuojama")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default="", help="k:FAILAS")
    a = ap.parse_args()
    dk, dfile = (int(a.dump.split(":")[0]), a.dump.split(":", 1)[1]) if a.dump else (-1, None)
    ports = [Port(a.port_base + k, dfile if k == dk else None) for k in range(a.flows)]
    sel = selectors.DefaultSelector()
    for p in ports:
        sel.register(p.sock, selectors.EVENT_READ, p)
    t_end = time.monotonic() + a.seconds
    t_warm = time.monotonic() + a.warmup
    while time.monotonic() < t_end:
        for key, _ in sel.select(timeout=0.2):
            p = key.data
            while True:
                try:
                    data = p.sock.recv(65536)
                except BlockingIOError:
                    break
                now = tai()
                p.feed(data, now)
                if time.monotonic() < t_warm:
                    p.lat_ms.clear()
                    p.pcr_intervals.clear()
    res = {"flows": [p.summary() for p in ports]}
    res["totals"] = {k: sum(f[k] for f in res["flows"]) for k in ("sync_errors", "cc_errors", "datagrams", "video_pes")}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for p in ports:
        if p.dump:
            p.dump.close()
    print(json.dumps(res["totals"]))


if __name__ == "__main__":
    main()
