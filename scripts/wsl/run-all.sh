#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/run-all.sh (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/run-all.sh nebenaudojamas; naudokite scripts/linux/run-all.sh" >&2
exec bash "$(dirname "$0")/../linux/run-all.sh" "$@"
