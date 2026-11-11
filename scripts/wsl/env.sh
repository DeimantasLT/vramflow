# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Nukreipimas į scripts/linux/env.sh (palikta vienam leidimui); šis failas tik įkeliamas per source.
[[ "${VRAMFLOW_SCRIPT_DEPRECATION:-}" == 1 ]] && echo "scripts/wsl/env.sh nebenaudojamas; naudokite scripts/linux/env.sh" >&2
source "$(dirname "${BASH_SOURCE[0]}")/../linux/env.sh"
