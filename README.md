# VRAMFlow

**GPU-resident media functions on [MXL](https://github.com/dmf-mxl/mxl) — and a test bench that proves they behave
exactly like MXL on the CPU.**

[Lietuviškai → README.lt.md](README.lt.md)

MXL (Media eXchange Layer, EBU / Linux Foundation) moves uncompressed video, audio and data between media functions
through shared-memory ring buffers on the host. Pull request
[dmf-mxl/mxl#653](https://github.com/dmf-mxl/mxl/pull/653) adds a `cuda-linear` payload backend, so a flow's grains can
live in GPU memory instead. This repository answers three questions about that backend:

1. **Is it correct?** Same grain indices, timestamps, slices, sync-group and partial-grain behaviour as host MXL, with
   bit-exact payloads.
2. **Is it stable?** Long runs, sanitizers, clock steps, start/stop, many flows.
3. **What is it worth?** How many full broadcast flows fit on one GPU compared with the CPU, for the same work.

It also contains draft upstream patches (tests and a fix) for PR #653, with evidence.

## What is in here

| Path | What it is |
|---|---|
| `src/mxlio/`, `src/gpu/` | `vramflow-verify`: writer/reader that checks every grain (pattern, CRC32C, index, timing) on host and cuda-linear flows |
| `src/media/` | Media kernels for 1080i50 v210: logo burn-in, NV12↔v210 weave/split, test pattern generator. CUDA (`media_kernels.cu`), CPU (`cpu/`) and an independent reference (`media_ref.cpp`) |
| `src/func/` | `vramflow-func`: real media functions on MXL flows — `gen`, `logo`, `dec` (H.264 TS/UDP → MXL, NVDEC or software), `enc` (MXL → H.264 + AAC TS, NVENC or x264), `verify` |
| `tests/` | Unit tests (Catch2), GPU kernel tests, functional tests, capacity harness (`perf/capacity.py`), comparison report (`perf/compare_report.py`) |
| `scripts/wsl/` | Build and run scripts (WSL2 or native Ubuntu 24.04) |
| `upstream/` | Draft patches for dmf-mxl/mxl PR #653 (U1–U3), each with evidence |
| `docs/architektura/` | Design documents (Lithuanian): verification design, GPU media functions, time and synchronization, GPU-vs-CPU method |

## Three variants of every function

| Variant | Processing | Flow storage | Meaning |
|---|---|---|---|
| **C** | CPU (own C++ kernels, software x264 / H.264 decoder) | host | what you run today |
| **H** | GPU | host (MXL v1.1 + copies to and from the GPU) | GPU without PR #653 |
| **G** | GPU | `cuda-linear` (PR #653) | grains stay on the GPU |

The C kernels are bit-exact with the GPU kernels (both are tested against the same independent reference), so the
comparison measures the same work.

## Results (RTX 4070 Laptop 8 GB, WSL2 Ubuntu 24.04 — indicative)

Format: 1080i50 (grain = field, 50 grains/s, v210), 60 s per level, all outputs verified.
**PASS strict** = 0 lost fields and 0 correctness errors. **PASS tol** = 0 correctness errors and ≤ 0.2 % lost
fields (separates single WSL2/Windows scheduling hiccups from a real capacity limit). N\* = largest N for which this
and every smaller level passed.

| Function | C (CPU) N\* | G (GPU, cuda-linear) N\* | CPU cores per flow, C → G |
|---|---|---|---|
| Logo burn-in, MXL → MXL | 16 | ≥ 64 (limit not reached) | 0.057 → 0.005 |
| H.264 decode, TS/UDP → MXL | 8 | 24 (16 strict) | ~0.55 → ~0.04 |
| Full chain: TS → decode → logo → encode H.264 + AAC → TS | 4 | ≥ 12 (limit not reached; see note) | ~2.0 → ~0.33 |

Full tables, GPU load and notes: [`docs/rezultatai/2026-11-11-gpu-vs-cpu.md`](docs/rezultatai/2026-11-11-gpu-vs-cpu.md).

Other measured facts:

- **GPU load (full chain, G):** at N=12 the CUDA cores are ~10 % busy, NVENC 64 %, NVDEC 50 %, ~34 W. The limit is
  the single NVENC engine of this laptop GPU, not CUDA compute; most of the GPU is still free.
- **Clocks matter (see note):** at very light load (chain N=2) the laptop GPU drops its clock from 1605 to 210 MHz
  every ~6 s and real-time deadlines are missed (18–23 % loss, reproduced 3×); from N=4 the clock stays up and all
  levels pass. Real-time GPU media functions need locked clocks (`nvidia-smi -pm 1` and `-lgc` on servers).
- **Power profile:** the decode and chain campaigns ran with the Windows "Whisper" (quiet, power-limited) profile.

- **Transport:** end-to-end grain latency p50 1.2 ms on cuda-linear vs 2.9 ms on host flows (N=1); host flows with
  GPU processing (variant H) collapse at N=32, cuda-linear does not.
- **Correctness:** NVDEC output bit-exact against the software decoder (1510/1510 sampled fields live); logo output
  bit-exact against the reference on GPU and CPU.
- **Upstream:** PR #653 full ctest **136/136** with patches U1–U3 (non-root; two tests skipped: one GPU, no docker).

Why indicative: WSL2 on Windows adds GPU scheduling (WDDM) and clock effects. The method and tools are made to be
rerun on native Linux servers, where the numbers become authoritative.

Detailed tables are produced by `tests/perf/compare_report.py` from the raw JSON of each run.

## Requirements

- Ubuntu 24.04 (native or WSL2), NVIDIA GPU with NVENC/NVDEC, driver with CUDA 13.x support
- CUDA Toolkit 13.x, Clang 19 (or GCC 13+), CMake ≥ 3.28, Ninja, vcpkg (for MXL)
- FFmpeg development libraries (`libavformat-dev libavcodec-dev libavutil-dev`), `ffmpeg` CLI, Python 3
- MXL from PR #653 at commit `6dddc91d92f5` (the scripts check this SHA)
- Time: kernel TAI offset must be 37 s (`python3 -c "import time;print(time.clock_gettime(time.CLOCK_TAI)-time.time())"`).
  On WSL2 see `docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md`.

## Build

```bash
# 1. MXL with the cuda-linear backend (PR #653)
git clone https://github.com/dmf-mxl/mxl ~/src/mxl-pr653
cd ~/src/mxl-pr653 && git fetch origin pull/653/head && git checkout 6dddc91d92f56c3c9d90fa089c95d8f262bc2c92
cd -
bash scripts/wsl/10-build-mxl.sh

# 2. This repository (verification tool, media functions, tests)
VERIFY_BUILD=$HOME/build/vramflow/func bash scripts/wsl/30-build-verify.sh -DMXLV_BUILD_FUNC=ON
```

Paths and versions are set in `scripts/wsl/env.sh` (`MXL_SRC`, `CUDA_HOME`, `VCPKG_ROOT`, …); override them with
environment variables.

## Use

```bash
# Everything at once: probes, upstream tests, unit/GPU/functional tests, transport scale, report
bash scripts/wsl/run-all.sh --soak skip

# Synthetic 1080i50 test clip (x264 interlaced) and its reference CRCs, for dec/chain
bash scripts/wsl/61-make-test-clip.sh

# One capacity level: 8 logo flows on cuda-linear, 60 s
python3 tests/perf/capacity.py ~/evidence/try --func logo --n 8 --seconds 60 --impl gpu --storage cuda-linear

# Same work on the CPU
python3 tests/perf/capacity.py ~/evidence/try --func logo --n 8 --seconds 60 --impl cpu

# GPU vs CPU campaign (variants alternate per level) and the report
bash scripts/wsl/71-compare.sh ~/evidence/cmp-logo logo 1,2,4,8,16,32,64
bash scripts/wsl/71-compare.sh ~/evidence/cmp-dec dec 1,2,4,8,12,16,24
bash scripts/wsl/71-compare.sh ~/evidence/cmp-chain chain 1,2,4,6,8,10,12
python3 tests/perf/compare_report.py report.md ~/evidence/cmp-logo ~/evidence/cmp-dec ~/evidence/cmp-chain
```

`vramflow-func <subcommand> --help` lists the options of each function (`gen`, `logo`, `verify`, `enc`, `dec`, `clip`,
`ref-crc`). Every run writes `summary-<function>.json` with counters (lost, late, invalid, verified fields) and a
latency histogram; `capacity.py` turns them into PASS/FAIL and CPU / VRAM per flow.

Only synthetic test media are used: generated patterns and a generated test clip.

## Upstream patches

See [`upstream/README.md`](upstream/README.md). U1 fixes a hang of the container test for non-root users; U2 and U3 add
Catch2 tests for synchronization groups and partial grains with payload-backend storage. They are drafts: the author
applies them and commits with `git commit -s` (DCO) before proposing them to dmf-mxl/mxl.

## Status and limits

- Results so far come from one laptop GPU under WSL2. A native Linux rerun is the next step.
- Variant H is implemented for logo and transport, not yet for dec/enc/chain.
- The 2-hour soak and the full sanitizer run on the media functions are not done yet.
- NVENC on this GPU encodes progressive frames (no field encoding); x264 encodes true interlaced (MBAFF, top field first).

## License and credit

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
You may use, change, distribute and sell this work, also in closed products. When you redistribute it or a derivative
work, keep the NOTICE file: it names the author. If this work helps a product, a standard or a paper, please credit
**"VRAMFlow by Deimantas Kiršinas"** with a link to this repository.

Binaries linked with an FFmpeg build that includes libx264 fall under the GPL when distributed (see NOTICE).

Developed with the help of AI coding assistants (Claude). The author reviewed the design, ran every measurement on real
hardware and is responsible for the content.

MXL is a project of the EBU and the Linux Foundation; this repository is independent and not endorsed by them.
