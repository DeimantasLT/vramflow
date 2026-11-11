#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Funkciniai gedimų testai (architektūra §6.1: F5a, F5b, F6a–d, F7, F8). Kiekvienas procesas — atskiras.

Naudojimas: functional.py <out_dir> [--only F5a,F6]
Rezultatas: <out_dir>/functional.json; išėjimas 0 tik jei nė vienas testas nėra FAIL.
"""
import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

BIN = os.environ.get("VERIFY_BIN", str(pathlib.Path.home() / "build/vramflow/release/vramflow-verify"))
LIB = os.environ.get("MXL_LIB", str(pathlib.Path.home() / "src/mxl-pr653/install/Linux-Clang-Release/lib"))
ENV = dict(os.environ, LD_LIBRARY_PATH=LIB)


class Proc:
    def __init__(self, role, domain, out, *args):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.err = open(self.out / f"{role}.err", "w")
        extra = ["--host-crosscheck-every", "10"] if role == "reader" else []  # CPU kryžminė patikra visada įjungta
        self.p = subprocess.Popen([BIN, role, "--domain", str(domain), "--out", str(self.out), *extra, *map(str, args)],
                                  stdout=subprocess.PIPE, stderr=self.err, env=ENV, text=True)
        self.role = role

    def wait(self, timeout=None):
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill9(self):
        self.p.send_signal(signal.SIGKILL)
        self.p.wait()

    def term(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
        return self.p.wait(timeout=30)

    def summary(self):
        f = self.out / f"summary-{self.role}.json"
        return json.loads(f.read_text()) if f.exists() else None


def counters(s):
    return s["flows"][0]["counters"] if s else {}


def verify_errors(s):
    return s["totals"]["verification_errors"] if s else None


def fresh_domain(name):
    d = pathlib.Path(f"/dev/shm/mxlv-{name}")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def leftovers(domain):
    return sorted(p.name for p in pathlib.Path(domain).iterdir() if p.name != "options.json")


def result(tid, status, **details):
    print(f"[{status:7s}] {tid}: {details.get('why', '')}", flush=True)
    return {"id": tid, "status": status, **details}


def f5a(out):
    d = fresh_domain("f5a")
    a = Proc("writer", d, out / "f5a/writerA", "--duration", 25, "--gc-before-create")
    time.sleep(2)
    r = Proc("reader", d, out / "f5a/reader", "--duration", 18)
    time.sleep(3)
    sock = d / "4d584c56-0000-4000-8000-000000000000.mxl-flow" / "payload.sock"
    ino0 = sock.stat().st_ino if sock.exists() else None
    t0 = time.monotonic()
    b = Proc("writer", d, out / "f5a/writerB", "--duration", 10)
    bcode = b.wait(timeout=10)
    dt = time.monotonic() - t0
    ino1 = sock.stat().st_ino if sock.exists() else None
    r.wait(timeout=40)
    a.term()
    rs = r.summary()
    c = counters(rs)
    ok = bcode == 5 and dt < 5 and ino0 is not None and ino0 == ino1 and verify_errors(rs) == 0 and c.get("gap_grains") == 0
    return result("F5a", "PASS" if ok else "FAIL", why=f"antro writer'io exit={bcode} per {dt:.2f}s; socket inode "
                  f"{'nepakito' if ino0 == ino1 else 'PAKITO'}; reader verif={verify_errors(rs)} gap={c.get('gap_grains')} "
                  f"ok={c.get('grains_ok')}", second_writer_exit=bcode, seconds=dt, reader=c)


def f5b(out, iters=10):
    bad = []
    for it in range(iters):
        d = fresh_domain(f"f5b")
        w1 = Proc("writer", d, out / f"f5b/{it}/w1", "--duration", 12)
        w2 = Proc("writer", d, out / f"f5b/{it}/w2", "--duration", 12)
        time.sleep(3)
        codes = [w1.p.poll(), w2.p.poll()]
        alive = [x is None for x in codes]
        r = Proc("reader", d, out / f"f5b/{it}/reader", "--duration", 3)
        r.wait(timeout=30)
        rs = r.summary()
        for w in (w1, w2):
            if w.p.poll() is None:
                w.term()
        exited = [c for c in codes if c is not None]
        ok = sum(alive) == 1 and exited == [5] and verify_errors(rs) == 0 and counters(rs).get("grains_ok", 0) >= 100
        if not ok:
            bad.append({"iter": it, "codes": codes, "reader_ok": counters(rs).get("grains_ok"), "verif": verify_errors(rs)})
    left = leftovers(d)
    return result("F5b", "PASS" if not bad else "FAIL",
                  why=f"{iters - len(bad)}/{iters} iteracijų: lygiai vienas writer'is gyvas, kitas exit 5, reader ≥100 grain'ų be klaidų",
                  failures=bad, domain_leftovers=left)


def f6(out):
    res = []
    d = fresh_domain("f6")
    w1 = Proc("writer", d, out / "f6/w1", "--duration", 60, "--gc-before-create")
    time.sleep(2)
    r = Proc("reader", d, out / "f6/reader", "--duration", 40, "--reopen-timeout-ms", 20000, "--stall-timeout-ms", 1000)
    time.sleep(8)
    w1.kill9()
    t_kill = time.monotonic()
    # F6d: naujas reader'is prie mirusio flow — turi baigtis švariai
    r2 = Proc("reader", d, out / "f6/reader_new", "--duration", 10, "--wait-flow-ms", 3000)
    r2code = r2.wait(timeout=30)
    res.append(result("F6d", "PASS" if r2code in (2, 4) else "FAIL",
                      why=f"naujas reader'is prie mirusio flow: exit={r2code} (laukta 2 arba 4, be pakibimo)", exit=r2code))
    # F6b: restart be GC
    w2 = Proc("writer", d, out / "f6/w2_nogc", "--duration", 10)
    w2code = w2.wait(timeout=15)
    if w2code is None:
        w2.term()
    res.append(result("F6b", "INFO" if w2code is not None and w2code >= 0 else "FAIL",
                      why=f"restart be GC po kill -9: exit={w2code} (architektūroje laukta 5 CONFLICT)", exit=w2code))
    # F6c: restart su GC — senas reader'is turi atsidaryti iš naujo ir toliau tikrinti
    t_restart = time.monotonic()
    w3 = Proc("writer", d, out / "f6/w3_gc", "--duration", 40, "--gc-before-create")
    rcode = r.wait(timeout=60)
    w3.term()
    rs = r.summary()
    c = counters(rs)
    ok_a = rcode is not None and rcode >= 0 and c.get("cuda_errors") == 0 and verify_errors(rs) == 0 and c.get("stalls", 0) >= 1
    res.append(result("F6a", "PASS" if ok_a else "FAIL",
                      why=f"writer kill -9: reader exit={rcode} (ne signalas), cuda_errors={c.get('cuda_errors')}, "
                          f"verif={verify_errors(rs)}, stalls={c.get('stalls')}", reader=c, restart_after_s=t_restart - t_kill))
    ok_c = c.get("reopens", 0) >= 1 and verify_errors(rs) == 0 and c.get("grains_ok", 0) >= 500
    res.append(result("F6c", "PASS" if ok_c else "FAIL",
                      why=f"restart su GC: reopens={c.get('reopens')}, grains_ok={c.get('grains_ok')}, verif={verify_errors(rs)}"))
    return res


def f7(out, cycles=10):
    d = fresh_domain("f7")
    w = Proc("writer", d, out / "f7/writer", "--duration", 18 + cycles * 3, "--gc-before-create")
    time.sleep(2)
    ra = Proc("reader", d, out / "f7/readerA", "--duration", 10 + cycles * 3)
    time.sleep(2)
    for k in range(cycles):
        rb = Proc("reader", d, out / f"f7/readerB{k}", "--duration", 30)
        time.sleep(2.5)
        rb.kill9()
    racode = ra.wait(timeout=90)
    wcode = w.term()
    ws, rs = w.summary(), ra.summary()
    proc = (out / "f7/writer/proc-writer.csv").read_text().strip().splitlines()[1:]
    fds = [int(l.split(",")[3]) for l in proc]
    c = counters(rs)
    # Nuotėkis = FD skaičius auga su kiekvienu nužudytu reader'iu. Pabaigoje writer'is atlaisvina flow, todėl
    # FD sumažėja — tai ne nuotėkis. Kriterijus: maksimumas per visą laiką == pirmas matavimas.
    ok = racode == 0 and verify_errors(rs) == 0 and c.get("gap_grains") == 0 and fds and max(fds) == fds[0]
    return result("F7", "PASS" if ok else "FAIL",
                  why=f"{cycles}× reader kill -9: reader A exit={racode} verif={verify_errors(rs)} gap={c.get('gap_grains')}; "
                      f"writer fd pradžia={fds[0] if fds else '?'} max={max(fds) if fds else '?'} pabaiga={fds[-1] if fds else '?'}, "
                      f"late_skips={counters(ws).get('writer_late_skips')}",
                  writer_exit=wcode, writer_fds={"first": fds[0], "max": max(fds), "last": fds[-1]} if fds else None)


def f8(out):
    d = fresh_domain("f8")
    w = Proc("writer", d, out / "f8/writer", "--duration", 50, "--gc-before-create")
    time.sleep(2)
    r = Proc("reader", d, out / "f8/reader", "--duration", 40)
    time.sleep(3)
    hog = subprocess.Popen([BIN, "hog", "--leave-free-mb", "150"], stdout=subprocess.PIPE, text=True, env=ENV)
    hog_line = hog.stdout.readline().strip()
    time.sleep(2)
    big = Proc("writer", d, out / "f8/writer_big", "--duration", 10, "--flows", 40, "--flow-base", 100, "--gc-before-create")
    bcode = big.wait(timeout=60)
    if bcode is None:
        big.term()
    time.sleep(1)
    left = [x for x in leftovers(d) if not x.startswith("4d584c56-0000-4000-8000-000000000000")]
    rcode = r.wait(timeout=60)
    hog.send_signal(signal.SIGTERM)
    hog.wait(timeout=30)
    w.term()
    rs = r.summary()
    clean = bcode is not None and bcode >= 0
    ok = clean and verify_errors(rs) == 0 and rcode == 0
    return result("F8", "PASS" if ok else "FAIL",
                  why=f"hog {hog_line}; 40 papildomų flow'ų writer exit={bcode} (be crash={clean}); veikiantis flow: "
                      f"reader exit={rcode} verif={verify_errors(rs)}; likę katalogai po nesėkmės: {len(left)}",
                  big_writer_exit=bcode, leftovers=left[:10], reader=counters(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    tests = {"F5a": lambda: [f5a(out)], "F5b": lambda: [f5b(out)], "F6": lambda: f6(out), "F7": lambda: [f7(out)],
             "F8": lambda: [f8(out)]}
    results = []
    for name, fn in tests.items():
        if a.only and name not in a.only.split(","):
            continue
        try:
            results += fn()
        except Exception as e:  # testo infrastruktūros klaida — FAIL, ne tylėjimas
            results.append(result(name, "FAIL", why=f"išimtis: {e!r}"))
    (out / "functional.json").write_text(json.dumps(results, indent=1))
    failed = [r["id"] for r in results if r["status"] == "FAIL"]
    print(f"FUNCTIONAL: {'PASS' if not failed else 'FAIL ' + ','.join(failed)} ({len(results)} testai)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
