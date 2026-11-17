<!-- SPDX-FileCopyrightText: 2026 Deimantas Kiršinas -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# U7 — `cuda-linear` backend returns from flow creation while its `cudaMemset` is still running

Target: dmf-mxl/mxl PR #653 (`thomastrue/mxl:feature/cuda-grain-payloads`, observed at `6dddc91d92f5`).
Form: issue / comment on #653 with the repro below. **Draft, not posted. Nothing has been sent anywhere.**

## Facts

- `lib/payload-backends/cuda-linear/CudaLinearBackend.cpp:1277` (`prepare`, writer side): after `cuMemCreate` / `cuMemMap`
  the backend zeroes the whole ring with `cudaMemset(base, 0, bytes)` and returns. Nothing waits for it. For device memory
  `cudaMemset` is asynchronous with respect to the host (CUDA runtime API reference, "Memory Management").
- Measured: right after `mxlCreateFlowWriter` returns, `cudaStreamQuery(nullptr)` is `cudaErrorNotReady` (600) — 20 runs
  out of 20 with the default 200 ms history (10 grains, 53 MiB) and 20 out of 20 with a 2 s history (100 grains, 527 MiB).
- Consequence 1, observed: a writer that is released shortly after it was created leaves GPU work on memory that is being
  unmapped. Under `compute-sanitizer --tool memcheck` a test that creates a `cuda-linear` flow, commits a few grain headers
  and releases it reported `Internal Sanitizer Error: The Sanitizer failed to handle a hardware exception` in 3 runs out
  of 5; in a longer test binary the CUDA context was left with a sticky error and the next `cudaMemset` in `prepare` failed
  with error 999. With one `cudaDeviceSynchronize()` after the flow creation: 0 runs out of 6.
- Consequence 2, **not observed**: an application that writes into the ring from a kernel on a non-blocking stream right
  after creation is not ordered after the zeroing, so its data could be overwritten with zeros. A test that writes 64 KiB at
  three places of a 527 MiB ring immediately after creation and reads them back found no lost bytes in 8 runs.
- Not checked: native Linux (everything here is WSL2), other drivers, the reader side.

## Minimal repro (`repro.cpp`)

Creates a `video/v210` 1080p50 writer with `cuda-linear` and asks whether the default stream is idle.

```
clang++-19 -std=c++20 -I$MXL/include -I$CUDA/include repro.cpp -o u7-repro -L$MXL/lib -lmxl -L$CUDA/lib64 -lcudart \
  -Wl,-rpath,$MXL/lib -Wl,-rpath,$CUDA/lib64
./u7-repro          # default 200 ms history
./u7-repro 2000     # 2 s history
```

Output on WSL2 Ubuntu 24.04, CUDA 13.2, driver 596.49, RTX 4070 Laptop, MXL at `6dddc91d92f5` (exit code 1):

```
history 200 ms (10 grains, 53 MiB); mxlCreateFlowWriter status 0; default stream right after: 600 (cudaErrorNotReady)
after cudaStreamSynchronize(default stream) = 0: 0 (cudaSuccess)
history 2000 ms (100 grains, 527 MiB); mxlCreateFlowWriter status 0; default stream right after: 600 (cudaErrorNotReady)
after cudaStreamSynchronize(default stream) = 0: 0 (cudaSuccess)
```

## Suggested change (for discussion)

Wait for the zeroing before `prepare` returns, for example `cudaStreamSynchronize(nullptr)` right after the `cudaMemset`
(and return its status). The cost is paid once per flow, at creation. Then the documented guarantee "readers of grains that
were never written see zeros" also holds for a reader that attaches immediately.

## Workaround in vramflow

`VideoOut::create` (`src/func/flows.cpp`) calls `cudaStreamSynchronize(nullptr)` for `cuda-linear` flows before it returns.
Test: `flows: cuda-linear VideoOut::create grįžta tik baigus žiedo nulinimą` in `tests/gpu/test_flows.cpp` (failed 6 of 6
before, passes 6 of 6 after). `compute-sanitizer --tool memcheck` on the same test binary: 3 clean runs out of 3 after.
