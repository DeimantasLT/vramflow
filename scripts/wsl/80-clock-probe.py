#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/80-clock-probe.py (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/80-clock-probe.py nebenaudojamas; naudokite scripts/linux/80-clock-probe.py" >&2
exec python3 "$(dirname "$0")/../linux/80-clock-probe.py" "$@"
