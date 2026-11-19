#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Paleidimo ir atsigavimo laikas (planas production-ready §3.3, G5).

  startup_time.py <out> [--n 1] [--repeats 5] [--impl gpu] [--storage cuda-linear]

Gen (šaltinis) veikia visą laiką. Kiekvienam pakartojimui: logo paleidžiamas su numatytomis parinktimis (+ --stats-file,
--stats-interval 0.05); matuojama nuo Popen iki pirmos --stats-file eilutės su ok > 0 (paleidimas) ir su ok > 0 visuose
flow'uose; tada logo nužudomas kill -9 ir paleidžiamas vėl — tas pats matavimas nuo naujo Popen (atsigavimas po kill -9).
Rezultatas: <out>/startup-time.json (kiekvienas bandymas ir p50 / maks.). H / G variantams fone — keepwarm (kaip capacity.py).
"""
import argparse
import json
import pathlib
import signal
import subprocess
import time

POLL_S = 0.01
TIMEOUT_S = 30.0


def first_ok(lines, all_flows=False):
    """--stats-file eilutės → pirmos periodinės eilutės, kurioje ok > 0 (bet kuriame / visuose flow'uose), seq arba None.
    Įvykių ("event") ir nepilnos (dar rašomos) eilutės praleidžiamos."""
    for ln in lines:
        try:
            j = json.loads(ln)
        except ValueError:
            continue
        flows = j.get("flows") if isinstance(j, dict) else None
        if not flows:
            continue
        oks = [f.get("ok", 0) > 0 for f in flows]
        if (all(oks) if all_flows else any(oks)):
            return j.get("seq")
    return None


def summary(times):
    """Bandymų laikai s (None — neprasidėjo per TIMEOUT_S) → {n, failed, p50_s, max_s}."""
    ok = sorted(t for t in times if t is not None)
    p50 = (ok[(len(ok) - 1) // 2] + ok[len(ok) // 2]) / 2 if ok else None
    return {"n": len(ok), "failed": len(times) - len(ok), "p50_s": None if p50 is None else round(p50, 3),
            "max_s": round(ok[-1], 3) if ok else None}


def wait_first_ok(stats, p, t0):
    """Laukia, kol stats faile atsiras ok > 0 (bet kuris, visi) → (s iki bet kurio, s iki visų); None — neįvyko."""
    t_any = t_all = None
    while time.monotonic() - t0 < TIMEOUT_S and p.poll() is None:
        lines = stats.read_text(errors="replace").splitlines() if stats.exists() else []
        now = time.monotonic() - t0
        if t_any is None and first_ok(lines) is not None:
            t_any = now
        if first_ok(lines, all_flows=True) is not None:
            return t_any, now
        time.sleep(POLL_S)
    return t_any, t_all


def main():
    import capacity as C  # BIN, ENV, wait_flow_defs, role_flow_id — tos pačios paleidimo prielaidos kaip talpos matavime
    import domain
    import keepwarm

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--impl", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--storage", default="cuda-linear", choices=["cuda-linear", "host"])
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = keepwarm.KeepWarm(C.BIN, C.ENV, log_dir=out)
    kw.want(a.impl == "gpu")
    dom = str(domain.make_domain(f"mxlf-startup-{a.n}"))
    common = ["--domain", dom, "--flows", str(a.n), "--impl", a.impl, "--storage", a.storage]
    long_s = str(int(a.repeats * 2 * (TIMEOUT_S + 5) + 30))
    gen = subprocess.Popen([C.BIN, "gen", *common, "--duration", long_s, "--out", str(out)],
                           stdout=subprocess.DEVNULL, stderr=open(out / "gen.err", "w"), env=C.ENV)
    runs = []
    try:
        if not C.wait_flow_defs(dom, [C.role_flow_id(1, i) for i in range(a.n)], 60):
            raise SystemExit("gen flow'ai neatsirado per 60 s")
        time.sleep(1.0)
        for rep in range(a.repeats):
            for kind in ("start", "restart"):
                stats = out / f"stats-{rep}-{kind}.jsonl"
                stats.unlink(missing_ok=True)
                t0 = time.monotonic()
                p = subprocess.Popen([C.BIN, "logo", *common, "--stats-file", str(stats), "--stats-interval", "0.05",
                                      "--duration", long_s, "--out", str(out / f"logo-{rep}-{kind}")],
                                     stdout=subprocess.DEVNULL, stderr=open(out / f"logo-{rep}-{kind}.err", "w"), env=C.ENV)
                t_any, t_all = wait_first_ok(stats, p, t0)
                runs.append({"repeat": rep, "kind": kind, "first_ok_s": t_any, "all_ok_s": t_all, "exit": p.poll()})
                print(f"[{rep} {kind}] pirmas ok {t_any} s, visi {t_all} s")
                p.send_signal(signal.SIGKILL)  # start → kill -9 → restart (atsigavimas); restart → kill -9 → kitas start
                p.wait()
    finally:
        gen.send_signal(signal.SIGTERM)
        try:
            gen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gen.kill()
        kw.stop()
        domain.remove_domain(dom)
    res = {"n": a.n, "impl": a.impl, "storage": a.storage, "keepwarm": a.impl == "gpu", "runs": runs,
           **{f"{kind}_{m}": summary([r[f"{m}_s"] for r in runs if r["kind"] == kind])
              for kind in ("start", "restart") for m in ("first_ok", "all_ok")}}
    (out / "startup-time.json").write_text(json.dumps(res, indent=1))
    for k in ("start_first_ok", "start_all_ok", "restart_first_ok", "restart_all_ok"):
        print(f"{k}: {res[k]}")


if __name__ == "__main__":
    main()
