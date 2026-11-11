# Upstream contributions to dmf-mxl/mxl (drafts)

Prepared by Deimantas Kiršinas on top of PR #653 (`thomastrue/mxl:feature/cuda-grain-payloads` @ `6dddc91d92f5`).
Each folder holds a plain diff against that SHA, the evidence, and the commit message. **Commits must be made by the
author** (`git apply <patch>` then `git commit -s`) — DCO `Signed-off-by` is required on every commit, and the author
must understand every line (AGENTS.md). Nothing here has been pushed or posted.

| # | What | Type | Evidence |
|---|---|---|---|
| U1 | `examples/cuda-cross-container-test.sh`: `sudo -n` so the container test is SKIPPED, not hung, for non-root users without docker | fix (2 lines) | before: hangs (killed at 20 s); after: exit 2 in 0.03 s — `U1-sudo-noninteractive/README.md` |
| U2 | Catch2 test: a synchronization group with a member whose payload lives in a payload backend (`test-file`) waits for the slow backend member and times out for a missing grain | test (runs in CI without GPU) | `U2-…/ctest.txt` — PASS |
| U3 | Catch2 test: partial grains (slice-by-slice commit, `minValidSlices` reads, `TOO_EARLY` for one slice more, reopen keeps progress) with a payload backend | test (runs in CI without GPU) | `U3-…/ctest.txt` — PASS |
| U4 | Catch2 tests: `video/v210a` (fill + key planes) with `cuda-linear`: plane count and sizes, both planes reach the reader, partial grains on both planes | test (needs a CUDA device, SKIP without) | `U4-…/ctest.txt` — PASS |

All three together on PR head: **full ctest 136/136 passed** as a non-root user (WSL2 Ubuntu 24.04, Clang 19, CUDA 13.2,
RTX 4070 Laptop), except `Reader on a peer GPU` (skipped: one GPU) and `Reader in another container` (excluded: no docker) —
`full-suite-with-U1-U3.txt`. `clang-format-19 --dry-run -Werror` is clean for the changed files.

Still to do before posting: build and run with GCC as well (MXL CI runs GCC and Clang); Doxygen check of new helpers.

Suggested route (see `docs/architektura/2026-11-11-gpu-vs-cpu-ir-upstream.md` §3): post U1–U3 as a comment on #653 with
the patches and measurements (precedent: reviewer findings with patches were integrated into the PR), together with the
GPU-vs-CPU evidence package. Larger items (device-storage conformance suite, `mxl-info` storage line, installing
`CudaLinearStorage.h`) go through the Requirements Council first.
