#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/lib/pair.sh (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/lib/pair.sh nebenaudojamas; naudokite scripts/linux/lib/pair.sh" >&2
exec bash "$(dirname "$0")/../../linux/lib/pair.sh" "$@"
