# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Vaikų paleidimas, perkrovimai ir tvarkingas sustabdymas (OK-2 k. 4–5, T2).

* Paleidimo tvarka — `procs` tvarka (graph.commands: ingest → logo | gfx + composite → enc); vaikas paleidžiamas, kai
  jo Proc.wait_for flow'ai jau domene (kodėl — graph.py); prieš grandinę — `preflight` komandos (kiekviena iki pabaigos;
  ne 0 → exit 2). Taip vėliau įjungiama `gpu-check` (Task 5), nekeičiant ciklo.
* Vaikas nukrito (signalas arba exit kodas) → paleidžiamas iš naujo su tais pačiais argv ir env iš karto. Kritimai
  skaičiuojami kiekvienam vaikui slenkančiame lange `restart.window_s`: leidžiama `restart.max` perkrovimų; (max + 1)-as
  kritimas lange → visi stabdomi, exit 2, stderr „<vardas> nukrito N kartus per W s“.
* Paleidimo klaida — ne kritimas: exit 1 (naudojimas) visada, exit 2 / 5 (paleidimas, priėmimo kontrolė), kol vaikas dar
  nepasiruošęs (ne visi jo Proc.flows atsirado; be flow'ų — kol veikia < STARTUP_GRACE_S) → exit 2 be perkrovimo (perkrovimas
  nepadėtų: blogas argumentas, nėra GPU, įėjimas nepristato duomenų per input_wait_s).
* SIGTERM / SIGINT → SIGTERM vaikams atvirkštine tvarka, po STOP_KILL_S — SIGKILL likusiems, exit 0. Antras signalas —
  SIGKILL iškart.

Vaikų stdout + stderr → prižiūrėtojo stderr eilutėmis (vaikų žurnalai jau be paslapčių — redact_url pačiuose vaikuose).
"""
import collections
import ctypes
import datetime
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Optional, Sequence

from .config import Plan
from .graph import Proc, display
from .redact import redact_url

STOP_KILL_S = 4.0        # SIGTERM → SIGKILL (docker stop laukia 10 s; visa grandinė turi baigtis ≤ 5 s)
STARTUP_GRACE_S = 10.0   # vaikas be flow'ų laikomas pasiruošusiu po tiek s
READY_TIMEOUT_S = 30.0   # laukiami flow'ai (Proc.wait_for) turi atsirasti per input_wait_s + tiek s
POLL_S = 0.05
TAIL_LINES = 20
FATAL_ALWAYS = (1,)
FATAL_AT_START = (2, 5)
INGEST_NO_DATA = "nėra duomenų per --input-wait-s"  # ingest demux klaidos tekstas (src/ingest/demux.cpp)


def log(msg: str) -> None:
    """Prižiūrėtojo eilutė stderr su UTC laiku (kaip vaikų „[vramflow <laikas>Z]“ eilutės)."""
    t = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    print(f"[vramflow-run {t}Z] {msg}", file=sys.stderr, flush=True)


try:  # prctl surandamas iš anksto: vaike po fork (preexec_fn) — tik vienas C kvietimas, be bibliotekų įkėlimo
    _PRCTL = ctypes.CDLL(None).prctl
except (OSError, AttributeError):
    _PRCTL = None


def _pdeathsig():
    """Vaikas gauna SIGTERM, jei prižiūrėtojas žūva (kill -9): grandinė nelieka be prižiūrėtojo. Tik Linux; kitur — nieko.
    Signalas siejamas su kuriančia gija — visi vaikai kuriami pagrindinėje gijoje."""
    if _PRCTL is not None:
        _PRCTL(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG


class Child:
    def __init__(self, proc: Proc):
        self.proc = proc
        self.popen: Optional[subprocess.Popen] = None
        self.started = 0.0
        self.crashes = collections.deque()
        self.tail = collections.deque(maxlen=TAIL_LINES)
        self.reader: Optional[threading.Thread] = None
        self.starts = 0

    @property
    def name(self):
        return self.proc.name

    def start(self):
        # Atskira procesų grupė: terminalo Ctrl+C pasiekia tik prižiūrėtoją, o jis stabdo vaikus nustatyta tvarka.
        self.popen = subprocess.Popen(self.proc.argv, env=self.proc.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, process_group=0, preexec_fn=_pdeathsig)
        self.started = time.monotonic()
        self.starts += 1
        self.tail.clear()
        self.reader = threading.Thread(target=self._pump, args=(self.popen.stdout,), daemon=True)
        self.reader.start()
        log(f"{self.name} paleistas (pid {self.popen.pid}{', ' + str(self.starts) + '-as kartas' if self.starts > 1 else ''})")

    def _pump(self, f):
        for raw in iter(f.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\n")
            self.tail.append(line)
            try:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
            except (OSError, ValueError):
                pass
        f.close()

    def ready(self, domain: str) -> bool:
        if not self.proc.flows:
            return time.monotonic() - self.started >= STARTUP_GRACE_S
        return all((pathlib.Path(domain) / f"{fid}.mxl-flow" / "flow_def.json").exists() for fid in self.proc.flows)

    def alive(self):
        return self.popen is not None and self.popen.poll() is None

    def signal(self, sig):
        if self.alive():
            try:
                self.popen.send_signal(sig)
            except ProcessLookupError:
                pass

    def join_reader(self, timeout=1.0):
        if self.reader is not None:
            self.reader.join(timeout)

    def last_line(self):
        for line in reversed(self.tail):
            if line.strip():
                return line.strip()
        return ""


def _describe(rc: int) -> str:
    if rc < 0:
        try:
            return f"signalas {signal.Signals(-rc).name}"
        except ValueError:
            return f"signalas {-rc}"
    return f"exit {rc}"


def _fmt_s(v) -> str:
    return f"{v:g}"


class Supervisor:
    def __init__(self, plan: Plan, procs: Sequence[Proc], domain: str, preflight: Sequence[Proc] = ()):
        self.plan = plan
        self.children = [Child(p) for p in procs]
        self.preflight = list(preflight)
        self.domain = domain
        self.stop_sig = 0
        self.kill_now = False
        self.ready = {c.name: False for c in self.children}

    def _on_signal(self, signum, _frame):
        if self.stop_sig:
            self.kill_now = True  # antras signalas — nebelaukiam
        self.stop_sig = signum

    def _run_preflight(self) -> int:
        for p in self.preflight:
            log(f"patikra: {display(p)}")
            r = subprocess.run(p.argv, env=p.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out = r.stdout.decode("utf-8", "replace")
            if out:
                sys.stderr.write(out if out.endswith("\n") else out + "\n")
            if r.returncode != 0:
                log(f"patikra {p.name} nepavyko ({_describe(r.returncode)}) — grandinė nepaleidžiama")
                return 2
            if self.stop_sig:
                return 0
        return 0

    def _stop_all(self, reason: str) -> None:
        log(f"stabdoma ({reason}): SIGTERM vaikams atvirkštine tvarka")
        for c in reversed(self.children):
            c.signal(signal.SIGTERM)
        end = time.monotonic() + STOP_KILL_S
        while time.monotonic() < end and not self.kill_now and any(c.alive() for c in self.children):
            time.sleep(POLL_S)
        for c in reversed(self.children):
            if c.alive():
                log(f"{c.name} nesibaigė per {_fmt_s(STOP_KILL_S)} s — SIGKILL")
                c.signal(signal.SIGKILL)
        for c in self.children:
            if c.popen is not None:
                c.popen.wait()
            c.join_reader()

    def _fatal_start(self, c: Child, rc: int) -> str:
        """Paleidimo klaidos pranešimas (be paslapčių: vaikų tekstai jau išvalyti, įėjimai — redact_url)."""
        last = c.last_line()
        if c.name == "ingest" and INGEST_NO_DATA in "\n".join(c.tail):
            ins = ", ".join(f"inputs[{k}] {redact_url(it.url)}" for k, it in enumerate(self.plan.inputs))
            return (f"įėjimas nepristato duomenų per input_wait_s = {_fmt_s(self.plan.input_wait_s)} s ({ins}); "
                    f"ingest: {last}")
        return f"{c.name} nepasileido ({_describe(rc)}): {last}"

    def _on_exit(self, c: Child, rc: int) -> Optional[str]:
        """Vaikas baigėsi. Grąžina mirtinos klaidos tekstą arba None (perkrautas)."""
        now = time.monotonic()
        up = now - c.started
        if rc in FATAL_ALWAYS or (rc in FATAL_AT_START and not self.ready[c.name]):
            return self._fatal_start(c, rc)
        r = self.plan.restart
        c.crashes.append(now)
        while c.crashes and now - c.crashes[0] > r.window_s:
            c.crashes.popleft()
        n = len(c.crashes)
        if n > r.max:
            log(f"{c.name} nukrito ({_describe(rc)}, veikė {up:.1f} s) — perkrovimų riba restart.max = {r.max} išnaudota")
            return f"{c.name} nukrito {n} kartus per {_fmt_s(r.window_s)} s"
        log(f"{c.name} nukrito ({_describe(rc)}, veikė {up:.1f} s) — perkraunamas ({n}/{r.max} per {_fmt_s(r.window_s)} s)")
        self.ready[c.name] = False
        c.start()
        return None

    def _flows_exist(self, flows) -> bool:
        return all((pathlib.Path(self.domain) / f"{fid}.mxl-flow" / "flow_def.json").exists() for fid in flows)

    def run(self) -> int:
        old = {s: signal.signal(s, self._on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            if (rc := self._run_preflight()) != 0 or self.stop_sig:
                return rc
            log(f"grandinė: {' → '.join(c.name for c in self.children)}; srautų {len(self.plan.inputs)}; domenas {self.domain}")
            # Paleidimo vartai (Proc.wait_for, graph.py): vaikas paleidžiamas, kai visi jo laukiami flow'ai jau domene.
            fatal = None
            t_start = time.monotonic()
            while not self.stop_sig and fatal is None:
                for c in self.children:
                    if c.popen is None and self._flows_exist(c.proc.wait_for):
                        c.start()
                for c in self.children:
                    if c.popen is None:
                        continue
                    rc = c.popen.poll()
                    if rc is not None:
                        c.join_reader(0.5)  # paskutinės eilutės (priežastis) — į tail prieš sprendimą
                        fatal = self._on_exit(c, rc)
                        if fatal:
                            break
                    elif not self.ready[c.name] and c.ready(self.domain):
                        self.ready[c.name] = True
                waiting = [c.name for c in self.children if c.popen is None]
                limit = self.plan.input_wait_s + READY_TIMEOUT_S
                if fatal is None and waiting and time.monotonic() - t_start > limit:
                    fatal = f"{', '.join(waiting)} nepaleistas: laukiami flow'ai neatsirado per {_fmt_s(limit)} s"
                if fatal is None and not self.stop_sig:
                    time.sleep(POLL_S)
            if fatal is not None:
                log(fatal)
                self._stop_all("klaida")
                log(f"baigta, exit 2: {fatal}")
                return 2
            self._stop_all(signal.Signals(self.stop_sig).name)
            log("baigta, exit 0")
            return 0
        finally:
            if any(c.alive() for c in self.children):  # išimtis cikle — vaikai nelieka be prižiūrėtojo
                self._stop_all("išimtis")
            for s, h in old.items():
                signal.signal(s, h)


def run(plan: Plan, procs: Sequence[Proc], domain: Optional[str] = None, preflight: Sequence[Proc] = ()) -> int:
    """Paleidžia grandinę ir prižiūri ją iki signalo ar mirtinos klaidos. Grąžina exit kodą (0 — sustabdyta signalu,
    2 — klaida). Grįžta tik tada, kai visi vaikai baigėsi (domeną po to galima šalinti)."""
    if domain is None:  # numatyta — iš vaikų argv (graph.commands visiems duoda tą patį --domain)
        argv = procs[0].argv
        domain = argv[argv.index("--domain") + 1]
    return Supervisor(plan, procs, domain, preflight).run()
