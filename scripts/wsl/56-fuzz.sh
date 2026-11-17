#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/56-fuzz.sh (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/56-fuzz.sh nebenaudojamas; naudokite scripts/linux/56-fuzz.sh" >&2
exec bash "$(dirname "$0")/../linux/56-fuzz.sh" "$@"
