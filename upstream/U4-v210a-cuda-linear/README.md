<!-- SPDX-FileCopyrightText: 2026 Deimantas Kiršinas -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# U4 — Catch2 tests: `video/v210a` (fill + key) with the `cuda-linear` payload backend

Target: dmf-mxl/mxl PR #653 (`thomastrue/mxl:feature/cuda-grain-payloads`, tested at `6dddc91d92f5`).
Form: comment on #653 with the patch, like U2/U3. **Draft, not posted.** No bug was found: the tests pass on the PR head
as written, so this is a conformance test, not a fix.

## What it covers

`mxlGrainStorageLayout.planeCount` is documented as "v210a has 2 (fill, then key)", but no test used a `v210a` flow with a
payload backend. Three tests in `lib/tests/cuda/test_cuda_payload.cpp` (they use the existing `data/v210a_flow.json`):

| Test | Checks |
|---|---|
| `CUDA payload : v210a has a fill plane and a key plane` | `planeCount == 2`; plane sizes `5120·1080` and `2560·1080`; pitches = `sliceSizes[0..1]`; `planes[1].offset == planes[0].size`; `grainSize` = sum; `totalSlices` = lines |
| `CUDA payload : v210a fill and key planes reach the reader` | pseudo random bytes written to both planes in device memory arrive byte for byte at a reader (0 differing bytes) |
| `CUDA payload : v210a partial grains cover both planes` | commits of 270 slices: reader with `minValidSlices == validSlices` gets OK and the first N lines of fill **and** key are byte-identical to what was written; `validSlices + 1` gives `MXL_ERR_OUT_OF_RANGE_TOO_EARLY` |

Not tested: that rows beyond `validSlices` are invisible (the reader returns the whole slot).

(The vramflow repo test `tests/gpu/test_v210a_cuda_linear.cpp` runs the same checks on a 1080i50 field flow; both pass.)

## Evidence

`ctest.txt`: 9 `CUDA payload` tests, 8 passed + 1 skipped (`Reader on a peer GPU`: one GPU), as non-root user, WSL2 Ubuntu 24.04,
Clang 19, CUDA 13.2, RTX 4070 Laptop. `clang-format-19 --dry-run -Werror` is clean. The patch applies with `git apply --check`
to the unmodified `6dddc91d92f5` tree. Built in a scratch copy (`/root/src/mxl-u4`); the original tree and install were not touched.

## Apply

```
git apply upstream/U4-v210a-cuda-linear/0001-tests-v210a-cuda-linear.patch
git commit -s    # DCO sign-off by the author
```

`test_cuda_payload.append.cpp` is the same code as the appended block, for review.

## Suggested commit message

```
Test v210a with the cuda-linear payload backend

Add tests that a v210a flow in cuda-linear storage has two planes (fill, then key)
with the sizes and pitches of the flow definition, that both planes written in device
memory reach a reader unchanged, and that after partial grain commits a reader with
minValidSlices sees the committed lines of both planes, and one slice more is too early.
```
