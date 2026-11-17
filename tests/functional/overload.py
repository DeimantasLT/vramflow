#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Priėmimo kontrolės funkciniai testai (planas docs/planai/2026-11-11-production-grade.md §P4, user story P4).

overload_enforce_synthetic — dirbtinis talpos failas šiai GPU (NVDEC 270 Mpix/s · 0,85 = 229,5: telpa 4 × 1080i50 po 51,84),
                             8 ingest procesai (po 1 UDP šaltinį) paleidžiami iš eilės su --admission enforce: pirmi 4 veikia
                             60 s (exit 0, late_frames == 0, skip_to_key == 0), 5–8 — exit 5, pranešime variklis ir skaičiai
                             (prašoma 51,84, užimta 207,36, leidžiama 229,50 Mpix/s); domene lygiai 8 flow'ai (4 video + 4 garso).
                             US 3 (skip_events — kartu su termino vartais, kitas etapas; čia — esamas skip_to_key).
overload_no_capacity       — enforce be talpos failo → exit 1 ir „talpa neišmatuota, paleiskite 70-capacity“; numatytoji (auto)
                             be failo — warn: paleidžia, lygiai viena įspėjimo eilutė (US 7).
overload_default           — numatytoji --admission auto (planas K13) su talpos failu veikia kaip enforce: dec --flows 3 be
                             --admission parinkties, talpa 1 srautui → 1 priimtas, 2 atmesti, exit 5.
overload_partial           — dec --flows 3, talpa 1 srautui: priimtas 1 (domene 1 flow), 2 atmesti, procesas veikia ir baigiasi
                             exit 5 (iš eilės: „ankstesnis netilpo“ — vieneto teste); enc --flows 3, NVENC sesijų riba 2: trečias atmestas dėl nvenc_sessions (US 2, 8).
Medžiaga — sintetinė (scripts/linux/62-make-av-clip.sh), siuntimas — tik loopback.
"""
import datetime
import json
import pathlib
import re
import subprocess
import sys
import time

import functional as F
import ingest as I

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf"))
import capacity_file  # noqa: E402  tests/perf/capacity_file.py — tas pats formatas, kurį rašo P8
import gpuinfo  # noqa: E402

BASE_PORT = 26300
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def gpu0():
    q = subprocess.run([gpuinfo.nvidia_smi(), "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], capture_output=True,
                       text=True, timeout=30)
    if q.returncode != 0 or not q.stdout.strip():
        return None
    name, uuid, driver = (x.strip() for x in q.stdout.splitlines()[0].split(","))
    return {"name": name, "uuid": uuid, "driver": driver}


def write_capacity(path, gpu, nvdec, nvenc=1000.0, sessions=None, functions=None):
    """Dirbtinis talpos failas (schema 1, patikrintas capacity_file.validate) šiai GPU."""
    d = {"schema": capacity_file.SCHEMA, "gpu": gpu, "format": "1080i50", "mpix_s_per_flow": 51.84,
         "measured_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "engines": {"nvdec": {"mpix_s": nvdec, "bound": "throughput", "from": "dirbtinis"},
                     "nvenc": {"mpix_s": nvenc, "bound": "throughput", "from": "dirbtinis"}},
         "functions": functions or {}}
    if sessions is not None:
        d["engines"]["nvenc"]["sessions"] = sessions
    capacity_file.validate(d)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    return path


def err_text(p):
    return p.err_path.read_text(errors="replace") if p.err_path.exists() else ""


def wait_text(p, needle, timeout):
    """Laukia, kol proceso stderr turės needle arba procesas baigsis. True — rasta."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if needle in err_text(p):
            return True
        if p.p.poll() is not None:
            return needle in err_text(p)
        time.sleep(0.1)
    return False


def domain_flows(d):
    """Flow'ai domene: mxl-info -l (UUID eilutės) ir *.mxl-flow katalogai."""
    r = subprocess.run([F.MXL_INFO, "-d", str(d), "-l"], capture_output=True, text=True, env=F.ENV, timeout=20)
    listed = set(UUID_RE.findall(r.stdout))
    dirs = {p.name[:-len(".mxl-flow")] for p in pathlib.Path(d).glob("*.mxl-flow")}
    return listed, dirs, r.stdout[-600:]


def overload_enforce_synthetic(out):
    tid = "overload_enforce_synthetic"
    if (why := I.ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    gpu = gpu0()
    if gpu is None:
        return [F.result(tid, "SKIP", why="nvidia-smi nerado GPU")]
    cap = write_capacity(pathlib.Path(out) / tid / f"capacity-{gpu['uuid']}.json", gpu, nvdec=270.0)
    d = F.fresh_domain("ovl-enf")
    T, warm = I.dur(60), 6
    procs, snd, problems, rows, listed = [], [], [], [], set()
    try:
        for k in range(8):
            port = BASE_PORT + k
            snd.append(I.Sender.ts(out, f"{tid}/snd{k}", I.CLIPS / "av_1080i50.ts", port))
            p = F.FProc(out, f"{tid}/ing{k}", "ingest", d, "--input", f"udp://127.0.0.1:{port}", "--flow-base", k, "--duration",
                        T + warm + 4, "--warmup", warm, *F.vcommon("gpu", "cuda-linear"), "--device-uuid", gpu["uuid"],
                        "--admission", "enforce", "--capacity-file", cap, "--input-wait-s", 30, "--run-id", f"ovl{k}")
            procs.append(p)
            # iš eilės: kitas paleidžiamas, kai šis jau rezervavo (arba buvo atmestas)
            if not wait_text(p, "priėmimo kontrolė (enforce): priimta", 40):
                problems.append(f"ing{k}: per 40 s nėra priėmimo eilutės (exit {p.p.poll()})")
        for k in range(4, 8):
            rc = procs[k].wait(timeout=30)
            e = err_text(procs[k])
            if rc != 5:
                problems.append(f"ing{k}: exit {rc} (laukta 5)")
            for need in ("atmestas", "nvdec: prašoma 51,84, užimta 207,36, leidžiama 229,50 Mpix/s"):
                if need not in e:
                    problems.append(f"ing{k}: pranešime nėra „{need}“")
            s = procs[k].summary() or {}
            adm = s.get("admission") or {}
            if adm.get("refused") != 1 or adm.get("admitted") != 0 or s.get("exit_code") != 5:
                problems.append(f"ing{k}: santrauka admission={adm} exit_code={s.get('exit_code')}")
        listed, dirs, txt = domain_flows(d)
        if len(listed) != 8 or len(dirs) != 8:
            problems.append(f"domene flow'ų: mxl-info -l {len(listed)}, katalogų {len(dirs)} (laukta 8 = 4 video + 4 garso)")
        for k in range(4):
            rc = procs[k].wait(timeout=T + warm + 90)
            s = procs[k].summary() or {}
            f = (s.get("flows") or [{}])[0]
            adm = s.get("admission") or {}
            rows.append(f"ing{k}: exit {rc}, frames_decoded {f.get('frames_decoded')}, late_frames {f.get('late_frames')}, "
                        f"skip_to_key {f.get('skip_to_key')}, missing {f.get('missing_frames')}")
            if rc != 0:
                problems.append(f"ing{k}: exit {rc} (laukta 0) {s.get('fatal', '')}")
            if f.get("late_frames", -1) != 0 or f.get("skip_to_key", -1) != 0:
                problems.append(f"ing{k}: late_frames {f.get('late_frames')}, skip_to_key {f.get('skip_to_key')} (laukta 0)")
            if (f.get("frames_decoded") or 0) < 25 * T:
                problems.append(f"ing{k}: frames_decoded {f.get('frames_decoded')} < {25 * T}")
            if adm.get("admitted") != 1 or adm.get("refused") != 0:
                problems.append(f"ing{k}: santrauka admission={adm}")
    finally:
        for p in procs:
            p.term()
        for s in snd:
            s.stop()
    return [F.result(tid, "FAIL" if problems else "PASS", why="; ".join(problems) + (" | " if problems else "") + "; ".join(rows),
                     flows_listed=len(listed))]


def overload_no_capacity(out):
    tid = "overload_no_capacity"
    res = []
    d = F.fresh_domain("ovl-nocap")
    v = F.vcommon("gpu", "cuda-linear")
    port = BASE_PORT + 10
    p = F.FProc(out, f"{tid}/enforce", "ingest", d, "--input", f"udp://127.0.0.1:{port}", *v, "--admission", "enforce",
                "--capacity-file", "/nonexistent/vramflow-capacity", "--input-wait-s", 3)
    rc = p.wait(timeout=30)
    if rc is None:
        p.term()
    e = err_text(p)
    need = "talpa neišmatuota, paleiskite 70-capacity"
    ok = rc == 1 and need in e
    res.append(F.result(f"{tid}[enforce]", "PASS" if ok else "FAIL", why=f"exit {rc}, „{need}“ {'yra' if need in e else 'NĖRA'}: {e.strip()[-300:]}"))
    # numatytoji (auto) be failo → warn: paleidžia (zondavimas be duomenų baigiasi exit 2 per --input-wait-s), lygiai viena įspėjimo eilutė
    p = F.FProc(out, f"{tid}/warn", "ingest", d, "--input", f"udp://127.0.0.1:{port}", *v, "--input-wait-s", 2)
    rc = p.wait(timeout=30)
    if rc is None:
        p.term()
    e = err_text(p)
    n = e.count("įspėjimas: priėmimo kontrolė (warn) — talpa neišmatuota")
    ok = n == 1 and rc not in (1, 5) and "šaltiniai" in e  # „[ingest] 1 šaltiniai …“ — funkcija paleista
    res.append(F.result(f"{tid}[warn]", "PASS" if ok else "FAIL", why=f"exit {rc}, įspėjimų eilučių {n} (laukta 1)"))
    return res


def dec_partial(out, tid, sub, gpu, port, admission_args):
    """dec --flows 3, NVDEC talpa 1 srautui (61 · 0,85 = 51,85 ≥ 51,84): 1 priimtas, 2 atmesti, exit 5. admission_args —
    „--admission …“ parinktys (tuščia — numatytoji auto). VRAM — įvertis prieš visą GPU atmintį."""
    v = F.vcommon("gpu", "cuda-linear")
    cap = write_capacity(pathlib.Path(out) / tid / sub / f"capacity-{gpu['uuid']}.json", gpu, nvdec=61.0)
    d = F.fresh_domain(f"ovl-{sub}")
    snd = I.Sender.ts(out, f"{tid}/snd-{sub}", I.CLIPS / "av_1080i50.ts", port)  # dec flow 0 įėjimas (H.264; garsas nenaudojamas)
    try:
        p = F.FProc(out, f"{tid}/{sub}", "dec", d, "--flows", 3, "--port-base", port, "--duration", 6, *v, "--device-uuid",
                    gpu["uuid"], *admission_args, "--capacity-file", cap.parent)
        # flow'ai tikrinami veikiant (baigdamas dec savo flow'us atlaisvina)
        wait_text(p, "priėmimo kontrolė (enforce): priimta", 30)
        end, dirs = time.monotonic() + 5, set()
        while time.monotonic() < end and not dirs:
            time.sleep(0.2)
            _, dirs, _ = domain_flows(d)
        time.sleep(0.5)
        _, dirs, _ = domain_flows(d)
        rc = p.wait(timeout=60)
        if rc is None:
            p.term()
    finally:
        snd.stop()
    e = err_text(p)
    s = p.summary() or {}
    adm = s.get("admission") or {}
    problems = []
    if rc != 5:
        problems.append(f"exit {rc} (laukta 5)")
    if adm.get("admitted") != 1 or adm.get("refused") != 2 or s.get("flows_n") != 1:
        problems.append(f"admission={adm}, flows_n={s.get('flows_n')}")
    if len(dirs) != 1:
        problems.append(f"domene {len(dirs)} flow'ų (laukta 1)")
    if "flow 1: atmestas — nvdec: prašoma 51,84, užimta 51,84, leidžiama 51,85 Mpix/s" not in e:
        problems.append("nėra flow 1 atmetimo eilutės su skaičiais")
    if "flow 2: atmestas — nvdec" not in e:
        problems.append("nėra flow 2 atmetimo eilutės")
    return F.result(f"{tid}[{sub}]", "FAIL" if problems else "PASS", why="; ".join(problems) or f"exit {rc}, admission {adm}")


def overload_partial(out):
    tid = "overload_partial"
    res = []
    gpu = gpu0()
    if gpu is None:
        return [F.result(tid, "SKIP", why="nvidia-smi nerado GPU")]
    if (why := I.ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    v = F.vcommon("gpu", "cuda-linear")
    res.append(dec_partial(out, tid, "dec", gpu, BASE_PORT + 20, ["--admission", "enforce"]))
    # enc NVENC: sesijų riba 2 (Mpix/s neriboja), --flows 3 → trečias atmestas dėl nvenc_sessions (įėjimų nėra — exit 4 po laukimo)
    cap = write_capacity(pathlib.Path(out) / tid / "enc" / f"capacity-{gpu['uuid']}.json", gpu, nvdec=1000.0, nvenc=10000.0, sessions=2)
    d = F.fresh_domain("ovl-enc")
    p = F.FProc(out, f"{tid}/enc", "enc", d, "--flows", 3, "--port-base", BASE_PORT + 30, "--duration", 3, *v, "--device-uuid",
                gpu["uuid"], "--admission", "enforce", "--capacity-file", cap, "--input-wait-ms", 500)
    rc = p.wait(timeout=60)
    if rc is None:
        p.term()
    e = err_text(p)
    s = p.summary() or {}
    adm = s.get("admission") or {}
    problems = []
    if "flow 2: atmestas — nvenc_sessions: prašoma 1, užimta 2, leidžiama 2 sesijų" not in e:
        problems.append("nėra nvenc_sessions atmetimo eilutės")
    if adm.get("admitted") != 2 or adm.get("refused") != 1:
        problems.append(f"admission={adm}")
    res.append(F.result(f"{tid}[enc]", "FAIL" if problems else "PASS", why="; ".join(problems) or f"exit {rc}, admission {adm}"))
    return res


def overload_default(out):
    """Numatytoji --admission auto (planas K13): su talpos failu veikia kaip enforce — be jokios --admission parinkties
    netelpantys srautai atmetami (exit 5). Be talpos failo — warn: tikrina overload_no_capacity[warn]."""
    tid = "overload_default"
    gpu = gpu0()
    if gpu is None:
        return [F.result(tid, "SKIP", why="nvidia-smi nerado GPU")]
    if (why := I.ensure_clips()):
        return [F.result(tid, "FAIL", why=why)]
    return [dec_partial(out, tid, "capacity", gpu, BASE_PORT + 40, [])]


TESTS = {"overload_enforce_synthetic": overload_enforce_synthetic, "overload_no_capacity": overload_no_capacity,
         "overload_partial": overload_partial, "overload_default": overload_default}
