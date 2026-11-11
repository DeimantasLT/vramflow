#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/50-sanitizers.sh (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/50-sanitizers.sh nebenaudojamas; naudokite scripts/linux/50-sanitizers.sh" >&2
exec bash "$(dirname "$0")/../linux/50-sanitizers.sh" "$@"
