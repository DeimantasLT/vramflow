<!-- SPDX-FileCopyrightText: 2026 Deimantas Kiršinas -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# U6 — `cuda-linear` backend leaves the caller's thread in the primary CUDA context

Target: dmf-mxl/mxl PR #653 (`thomastrue/mxl:feature/cuda-grain-payloads`, observed at `6dddc91d92f5`).
Form: issue / comment on #653 with the repro below. **Draft, not posted. Nothing has been sent anywhere.**

## Facts

- `lib/payload-backends/cuda-linear/CudaLinearBackend.cpp:404-415` (writer side, allocation path): retains the device's
  primary context, calls `cuCtxSetCurrent(primary)` and then `cudaSetDevice(ordinal)`. The thread's previous current
  context is not saved and not restored. `CudaLinearBackend.cpp:1149` calls `cudaSetDevice` on another path as well.
- Effect: an application that created its own context (`cuCtxCreate`, common with NVENC/NVDEC SDK and FFmpeg-style code)
  and made it current finds the **primary** context current on that thread after `mxlCreateFlowWriter` with
  `{"payload":{"backend":"cuda-linear",…}}`. Its next driver-API calls (allocations, launches, NVENC sessions bound to its
  context) silently go to the wrong context or fail with `CUDA_ERROR_INVALID_CONTEXT` / invalid handle.
- Not checked: the reader side (`mxlCreateFlowReader` with a `cuda-linear` flow) and `mxlReleaseFlowWriter`.

## Minimal repro (`repro.cpp`)

Creates its own context, then a `video/v210` writer with `cuda-linear`, and compares the current context before and after.

```
clang++-19 -std=c++20 -I$MXL/include -I$CUDA/include repro.cpp -o u6-repro -L$MXL/lib -lmxl -lcuda -Wl,-rpath,$MXL/lib
./u6-repro
```

Output on WSL2 Ubuntu 24.04, CUDA 13.2, driver 596.49, RTX 4070 Laptop, MXL at `6dddc91d92f5` (exit code 1):

```
before mxlCreateFlowWriter: current == own context: yes
mxlCreateFlowWriter status 0; after: current == own context: no, == primary: yes
```

## Suggested change (for discussion)

Save the current context at the start of the backend functions that switch it (`cuCtxGetCurrent`) and restore it before
returning (`cuCtxSetCurrent(saved)`, also on error paths), or use `cuCtxPushCurrent(primary)` / `cuCtxPopCurrent`. Document
that the backend's own work always runs in the device's primary context.

## Workaround in vramflow

`libvramflow-gfx` saves and restores the calling thread's current context around every call that may touch CUDA
(`CtxGuard` in `src/gfx/gfx_writer.cpp`; test `gfx_writer: kviečiančiojo CUDA kontekstas atstatomas po bibliotekos
kvietimų`), and states the rule in `include/vramflow/gfx.h` ("CUDA FRAMES").
