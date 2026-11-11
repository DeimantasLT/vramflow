# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""MXL domenai testų harness'ams (planas §P6, sprendimas K8).

Kiekvienas paleidimas gauna savo domeną (/dev/shm/vf-<tag>-<8 hex>), todėl du paleidimai su ta pačia žyma vienas kito
netrina. Domenas trinamas tik tada, kai jo nuomos kataloge (`<domenas>.vramflow/`, šalia domeno) nėra gyvų savininkų:
kiekvienas vramflow-func procesas laiko `<run_id>.lease` su flock(LOCK_EX) (branduolys atlaisvina ir po kill -9).
Tas pats algoritmas kaip src/runtime/domain_lease.cpp.
"""
import errno
import fcntl
import os
import pathlib
import secrets
import shutil

SHM = pathlib.Path(os.environ.get("VF_DOMAIN_ROOT", "/dev/shm"))


def lease_dir(path) -> pathlib.Path:
    p = str(path).rstrip("/") or "/"
    return pathlib.Path(p + ".vramflow")


def make_domain(tag: str) -> pathlib.Path:
    """Naujas tuščias domenas /dev/shm/vf-<tag>-<8 hex> (mkdir be exist_ok: vardas niekada nesidalija)."""
    while True:
        d = SHM / f"vf-{tag}-{secrets.token_hex(4)}"
        try:
            d.mkdir(parents=False)
            return d
        except FileExistsError:
            continue


def _probe(path: pathlib.Path) -> int:
    """1 — gyvas savininkas, 0 — negyvas (failas pašalintas), -1 — nėra / klaida."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return -1
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            return 1 if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN) else -1
        try:
            os.unlink(path)  # užraktą gavom — savininkas negyvas
        except OSError:
            pass
        return 0
    finally:
        os.close(fd)


def live_owners(path) -> list:
    """Gyvų savininkų run_id (surūšiuoti); negyvų nuomos failai pašalinami."""
    ld = lease_dir(path)
    if not ld.is_dir():
        return []
    return sorted(f.stem for f in ld.glob("*.lease") if not f.name.startswith(".") and _probe(f) == 1)


def remove_domain(path) -> bool:
    """Pašalina domeną ir jo nuomos katalogą. False ir nieko netrina, jei yra gyvų savininkų."""
    if live_owners(path):
        return False
    shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(lease_dir(path), ignore_errors=True)
    return not pathlib.Path(path).exists()
