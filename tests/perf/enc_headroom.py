#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""enc C (x264) talpos atsarga izoliacijos vartams (T1): ta pati grandinė kaip fiso_enc (gen video + garsas → enc N x264
ultrafast → ts_sink), BE gedimo, N = 2, 4, 6, 8 (ENC_HEADROOM_N). Kiekvienam N — pakartojimai (ENC_HEADROOM_REPS, numatyta 2).

Matuojama (kas 1 s, visą funkcijos paleidimą po įšilimo):
  * sistemos CPU užimtumas, % visų loginių CPU (/proc/stat; mašina — tik šiam darbui);
  * enc proceso CPU, % vieno CPU (utime + stime / laikas; 100 % = viena gija pilnai);
  * enc frame_lateness (santraukos hist: p50, p99, max) ir sveikų srautų per-sekundinių maksimumų (lateness_s) mediana ir
    maksimumas — vartų „kontrolinis langas“ ir jo sklaida;
  * praradimai (late_frames, frames_dropped_queue, late_skips_frames, audio_underrun_frames) ir frames_encoded.
Rezultatas — JSON (argv[1]/enc_headroom.json) ir lentelė stdout. Naudojimas (WSL2, kaip functional.py):
  FUNC_BIN=… PATH=$PATH:… python3 tests/perf/enc_headroom.py /root/build/vramflow/headroom
"""
import json
import os
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "functional"))
import isolation as I  # noqa: E402  tests/functional/isolation.py (run_enc, fault "none")

TICK = os.sysconf("SC_CLK_TCK")


def sys_busy():
    a = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle = a[3] + a[4]
    return sum(a), idle


def enc_pids():
    out = []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if len(cmd) > 1 and cmd[0].endswith(b"vramflow-func") and cmd[1] == b"enc":
            out.append(int(p.name))
    return out


def proc_cpu(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / TICK
    except (OSError, IndexError, ValueError):
        return None


class Sampler(threading.Thread):
    """Kas 1 s: sistemos CPU % ir enc proceso CPU % (nuo enc starto + įšilimo)."""

    def __init__(self, warm_s):
        super().__init__(daemon=True)
        self.warm_s, self.stop, self.sys, self.enc = warm_s, threading.Event(), [], []

    def run(self):
        pid, t_start = None, None
        prev_sys, prev_enc, prev_t = sys_busy(), None, time.monotonic()
        while not self.stop.wait(1.0):
            now = time.monotonic()
            if pid is None:
                ps = enc_pids()
                if ps:
                    pid, t_start = ps[0], now
            cur_sys = sys_busy()
            tot, idle = cur_sys[0] - prev_sys[0], cur_sys[1] - prev_sys[1]
            ce = proc_cpu(pid) if pid else None
            if t_start is not None and now - t_start >= self.warm_s and tot > 0:
                self.sys.append(100.0 * (tot - idle) / tot)
                if ce is not None and prev_enc is not None:
                    self.enc.append(100.0 * (ce - prev_enc) / (now - prev_t))
            prev_sys, prev_enc, prev_t = cur_sys, ce, now


def med(v):
    s = sorted(v)
    return round((s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2, 2) if s else None


def one(out, n, rep):
    I.N = n
    smp = Sampler(I.WARM_S + 1)
    smp.start()
    r = I.run_enc(out, "C", "cpu", "host", "x264", "none", pos="last", suffix=f"-hr{n}-{rep}")
    smp.stop.set()
    smp.join()
    if r.get("err"):
        return {"n": n, "rep": rep, "err": r["err"]}
    lat = r.get("out_lateness") or {}
    per_flow_ctrl = [f.get("lateness_max_us", 0) / 1000.0 for f in r.get("fn_flows") or []]
    # sveikų srautų per-sekundiniai maksimumai po įšilimo (kaip vartų kontrolinis langas)
    t_from = (r.get("start_tai") or 0) + int(I.WARM_S * 1e9)
    sec = [v for m in (r.get("lat_s") or {}).values() for k, v in m.items() if k * 1_000_000_000 >= t_from]
    fh = r["fn"]["healthy"]
    loss = {k: fh.get(k, 0) + r["fn"]["affected"].get(k, 0) for k in I.ENC_LOSS}
    return {"n": n, "rep": rep, "sys_cpu_pct_med": med(smp.sys), "sys_cpu_pct_max": round(max(smp.sys), 1) if smp.sys else None,
            "enc_cpu_pct_med": med(smp.enc), "frame_lateness": lat,
            "sec_max_med_ms": med(sec), "sec_max_max_ms": round(max(sec), 2) if sec else None, "flow_lateness_max_ms": per_flow_ctrl,
            "loss": {k: v for k, v in loss.items() if v}, "frames_encoded_min": min(
                [f.get("frames_encoded", 0) for f in r.get("fn_flows") or []] + [r["fn"]["affected"].get("frames_encoded", 0)]),
            "sink": r.get("sink")}


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in os.environ.get("ENC_HEADROOM_N", "2,4,6,8").split(",")]
    reps = int(os.environ.get("ENC_HEADROOM_REPS", "2"))
    rows = []
    for rep in range(reps):
        for n in ns:
            row = one(str(out), n, rep)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / "enc_headroom.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("| N | sistemos CPU % (med / max) | enc CPU % | frame_lateness p50 / p99 / max ms | sekundžių maks. med / max ms | praradimai |")
    for r in rows:
        if r.get("err"):
            print(f"| {r['n']} | klaida: {r['err']} |")
            continue
        fl = r["frame_lateness"]
        print(f"| {r['n']} | {r['sys_cpu_pct_med']} / {r['sys_cpu_pct_max']} | {r['enc_cpu_pct_med']} | "
              f"{fl.get('p50_ms')} / {fl.get('p99_ms')} / {fl.get('max_ms')} | "
              f"{r['sec_max_med_ms']} / {r['sec_max_max_ms']} | {r['loss'] or 0} |")


if __name__ == "__main__":
    main()
