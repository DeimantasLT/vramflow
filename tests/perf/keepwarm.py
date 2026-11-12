# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""GPU „šildymas“ harness'ams: `vramflow-func keepwarm` fone, kol vyksta GPU variantai (H, G).

Kodėl: nešiojamo kompiuterio GPU tuščiąja eiga nuslenka į P8, o tada bet kurio proceso platus GPU darbas (flow'ų kūrimas,
kito proceso startas / pabaiga) sustabdo VISŲ procesų GPU darbą 50–350 ms — laiko testai krenta ne dėl tikrinamo kodo
(docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md). Su nuolatine lengva apkrova — 0 tokių pliūpsnių.

Taisyklė: `want(gpu)` prieš kiekvieną variantą — GPU variantui (impl gpu: H, G) keepwarm paleidžiamas (jei dar neveikia) ir
palaukiama SETTLE_S, kol pakils taktai; CPU variantui (C) — sustabdomas. Numatyta ĮJUNGTA; `VRAMFLOW_GPU_KEEPWARM=0`
išjungia. `on()` — ar šiuo metu veikia (rašoma į kiekvieną rezultatą). `suspended()` — kontekstas, kuriame keepwarm
nepaleidžiamas (INFO matavimams „be keepwarm“).

Gyvavimas: keepwarm paleidžiamas su `--parent-pid <harness'o pid>` — jis baigiasi pats, kai harness'o nebėra (ir SIGKILL,
timeout, CI atveju: PR_SET_PDEATHSIG + patikra kiekviename cikle). Harness'o pusėje — atexit, SIGTERM / SIGINT / SIGHUP
handler'iai (sustabdo keepwarm, tada numatytasis signalo elgesys) ir try / finally kviečiančiuose skriptuose.
"""
import atexit
import contextlib
import os
import signal
import subprocess
import threading
import time

ENABLED = os.environ.get("VRAMFLOW_GPU_KEEPWARM", "1") != "0"
SETTLE_S = 2.0  # P8 → P3 perėjimas trunka ≤ ~0,35 s; atsarga, kad testas prasidėtų jau pakilusiais taktais


class KeepWarm:
    def __init__(self, func_bin, env=None, log_dir=None):
        self.func_bin, self.env, self.log_dir = func_bin, env, log_dir
        self.p = None
        self.starts = 0
        self.off = False  # suspended(): keepwarm nepaleidžiamas
        atexit.register(self.stop)  # ir po harness'o išimties keepwarm nelieka veikti
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                prev = signal.getsignal(sig)
                signal.signal(sig, lambda s, f, prev=prev: self._on_signal(s, f, prev))

    def _on_signal(self, sig, frame, prev):
        """Signalas harness'ui: keepwarm sustabdomas, tada — ankstesnis / numatytasis elgesys (procesas baigiasi kaip be mūsų)."""
        self.stop()
        if callable(prev):
            prev(sig, frame)
        elif prev != signal.SIG_IGN:
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

    def on(self):
        return self.p is not None and self.p.poll() is None

    @contextlib.contextmanager
    def suspended(self):
        """Kontekstas be keepwarm (INFO matavimui „be keepwarm“); po jo — vėl pagal want()."""
        was, self.off = self.off, True
        self.stop()
        try:
            yield
        finally:
            self.off = was

    def want(self, gpu):
        if gpu and ENABLED and not self.off:
            self.start()
        else:
            self.stop()

    def start(self):
        if self.on():
            return
        err = subprocess.DEVNULL
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            err = open(os.path.join(str(self.log_dir), f"keepwarm-{self.starts}.err"), "w")
        self.p = subprocess.Popen([self.func_bin, "keepwarm", "--duration", "0", "--parent-pid", str(os.getpid())],
                                  stdout=subprocess.DEVNULL, stderr=err, env=self.env)
        self.starts += 1
        time.sleep(SETTLE_S)
        if self.p.poll() is not None:  # neprasidėjo (nėra GPU / binaro) — testai tęsiasi be jo, rezultatai tai parodo
            self.p = None

    def stop(self):
        if self.p is None:
            return
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.p = None
