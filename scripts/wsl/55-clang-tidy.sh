#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/55-clang-tidy.sh (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/55-clang-tidy.sh nebenaudojamas; naudokite scripts/linux/55-clang-tidy.sh" >&2
exec bash "$(dirname "$0")/../linux/55-clang-tidy.sh" "$@"
