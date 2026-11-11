# U1 — `cuda-cross-container-test.sh`: non-interactive `sudo`

Target: dmf-mxl/mxl PR #653 (`thomastrue/mxl:feature/cuda-grain-payloads`, tested at `6dddc91d92f5`).
Form: comment on #653 with the patch (precedent: reviewer fixes integrated as commits), or a PR to the author's branch.

## Problem (found while running the PR test suite)

`examples/cuda-cross-container-test.sh` falls back to `sudo docker info` when `docker info` fails. When the
test suite runs as a non-root user without a working docker daemon and without passwordless sudo, `sudo`
waits for a password. The Catch2 test `CUDA payload : Reader in another container` then hangs until the
ctest timeout (600 s in our run) instead of being skipped (the script's documented "exit 2 = no docker").

## Fix

Use `sudo -n` (non-interactive): it fails immediately when a password would be required, so the script
reaches its existing `exit 2` path and the test reports SKIP.

## Evidence (WSL2 Ubuntu 24.04, non-root user `mxltest`, no docker in PATH, stdin = /dev/null)

| Script | Exit | Wall time | Output |
|---|---|---|---|
| before (PR head) | 124 (killed by `timeout 20`) | 20.02 s | — (sudo waiting for password) |
| after (`sudo -n`) | 2 | 0.03 s | `mxl-cuda-cross-container: docker is not available` |

Full ctest before the fix (as non-root): test 132 `Reader in another container` → `***Timeout 600.18 sec`;
all other 134 tests pass (see `evidence/20261110T194604Z/20-upstream/` in the vramflow repo).

## Commit message (to be committed by the author with `--signoff`)

```
examples: use non-interactive sudo in cuda-cross-container-test.sh

When docker is unavailable to the current user, the script falls back to
`sudo docker info`. Without passwordless sudo this waits for a password and
the "Reader in another container" test hangs until the ctest timeout instead
of being skipped. Use `sudo -n` so the script reaches its "docker is not
available" exit status 2 immediately.
```

Apply: `git am`-style is not used (plain diff) — `git apply 0001-cuda-cross-container-test-sudo-n.patch`,
then `git commit -s` with the message above.
