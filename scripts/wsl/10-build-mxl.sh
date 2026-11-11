#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/10-build-mxl.sh (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/10-build-mxl.sh nebenaudojamas; naudokite scripts/linux/10-build-mxl.sh" >&2
exec bash "$(dirname "$0")/../linux/10-build-mxl.sh" "$@"
