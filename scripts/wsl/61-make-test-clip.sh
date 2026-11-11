#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/61-make-test-clip.sh (palikta vienam leidimui).
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/61-make-test-clip.sh nebenaudojamas; naudokite scripts/linux/61-make-test-clip.sh" >&2
exec bash "$(dirname "$0")/../linux/61-make-test-clip.sh" "$@"
