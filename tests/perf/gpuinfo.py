# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""nvidia-smi kelio paieška Python harness'ui (ta pati tvarka kaip scripts/linux/lib/tools.sh: find_nvidia_smi).

Pirma PATH (`shutil.which`), WSL2 kelias tik kaip atsarginis; jei nėra nei vieno — grąžinamas „nvidia-smi“, kad
subprocess'as pats pranešytų FileNotFoundError (kaip anksčiau). Atsarginis kelias perrašomas VRAMFLOW_WSL_NVIDIA_SMI.
"""
import os
import shutil

WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


def nvidia_smi() -> str:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    fallback = os.environ.get("VRAMFLOW_WSL_NVIDIA_SMI", WSL_NVIDIA_SMI)
    if os.access(fallback, os.X_OK):
        return fallback
    return "nvidia-smi"
