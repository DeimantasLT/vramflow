# VRAMFlow

**Keep MXL media flows in GPU memory and process them there, with the same timing and behaviour as MXL on the CPU.**

[Lietuviškai → README.lt.md](README.lt.md)

[MXL](https://github.com/dmf-mxl/mxl) (Media eXchange Layer, EBU / Linux Foundation) lets media functions exchange
uncompressed video, audio and data through shared memory. Normally that memory is host RAM. With the `cuda-linear`
backend from [dmf-mxl/mxl#653](https://github.com/dmf-mxl/mxl/pull/653), a flow can live in GPU memory (VRAM)
instead, so a GPU function can hand frames to the next GPU function without copying them through the CPU.

VRAMFlow gives you:

| Part | What it does |
|---|---|
| `vramflow-func` | Ready-made media functions on MXL flows: test generator, logo burn-in, H.264 decode, H.264 + AAC encode, alpha graphics compositor, output checker. Each runs on the CPU or the GPU. |
| `libvramflow-gfx` | C library: turn RGBA frames from any program into a standard MXL graphics flow (`video/v210a`, fill + key). |
| `vramflow-verify` | Test writer and reader that check every grain (content, index, timing) on host or GPU flows. |
| Tests and tools | Unit, GPU and functional tests; a capacity harness that measures how many flows fit on a machine. |

Everything is C++ / CUDA. Python is used only by the test and measurement scripts.

---

## 1. Basic ideas (two minutes)

| Term | Meaning |
|---|---|
| **Domain** | A folder in shared memory, e.g. `/dev/shm/my-domain`. All writers and readers that use the same folder see each other's flows. |
| **Flow** | One stream, for example one video signal. It has a UUID and a description (`flow_def.json`). |
| **Grain** | One unit of a flow. Here: one video **field** (1080i50 = 50 fields per second). Each grain has an index derived from TAI time, so every process agrees which grain belongs to which moment. |
| **Writer / reader** | A writer creates a flow and commits grains; any number of readers read them. When the last writer stops, the flow disappears. |
| **Storage** | Where the grains live: `host` (RAM, standard MXL) or `cuda-linear` (GPU memory, PR #653). Writers and readers use the same API for both. |
| **v210 / v210a** | 10-bit 4:2:2 video. `v210a` adds a key (alpha) plane, so a graphics flow carries both picture and transparency. |

Every function in VRAMFlow runs in one of three ways. You choose with two options:

| Variant | `--impl` | `--storage` | When to use |
|---|---|---|---|
| **C** — CPU | `cpu` | `host` | No GPU, or to compare with today's setup |
| **H** — GPU, RAM flows | `gpu` | `host` | GPU processing, but other programs need the flows in RAM |
| **G** — GPU, GPU flows | `gpu` | `cuda-linear` | Best: frames stay on the GPU between functions |

The three variants produce **bit-identical** output; all are tested against one independent reference.

---

## 2. Requirements

- Ubuntu 24.04 (native or WSL2)
- For the GPU variants: NVIDIA GPU with NVENC/NVDEC, driver with CUDA 13.x, CUDA Toolkit 13.x
- Clang 19 (or GCC 13+), CMake ≥ 3.28, Ninja, vcpkg (needed to build MXL)
- `libavformat-dev libavcodec-dev libavutil-dev`, `ffmpeg`, Python 3 (tests only)
- System clock with the correct TAI offset (37 s). Check:
  `python3 -c "import time;print(time.clock_gettime(time.CLOCK_TAI)-time.time())"` must print `37.0`.
  On WSL2 see [`docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md`](docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md).

## 3. Build

```bash
# 1. MXL with the GPU backend (PR #653, pinned commit)
git clone https://github.com/dmf-mxl/mxl ~/src/mxl-pr653
cd ~/src/mxl-pr653
git fetch origin pull/653/head
git checkout 6dddc91d92f56c3c9d90fa089c95d8f262bc2c92
cd -

# 2. Build and install MXL (into ~/src/mxl-pr653/install/Linux-Clang-Release)
bash scripts/linux/10-build-mxl.sh

# 3. Build VRAMFlow
VERIFY_BUILD=$HOME/build/vramflow bash scripts/linux/30-build-verify.sh -DMXLV_BUILD_FUNC=ON
```

Paths and versions (`MXL_SRC`, `CUDA_HOME`, `VCPKG_ROOT`, …) are in `scripts/linux/env.sh`; override any of them with an
environment variable. The build puts `vramflow-func`, `vramflow-verify` and `libvramflow-gfx.so` in `$VERIFY_BUILD`.

Before running anything, in every shell:

```bash
export LD_LIBRARY_PATH=$HOME/src/mxl-pr653/install/Linux-Clang-Release/lib:$HOME/build/vramflow
export PATH=$HOME/build/vramflow:$HOME/src/mxl-pr653/install/Linux-Clang-Release/bin:$PATH
```

---

## 4. Quick start

Each example uses its own domain and removes it at the end. Run the lines of one example in one shell.

### 4.1 First writer and reader (RAM)

```bash
D=/dev/shm/vf-hello; mkdir -p $D
vramflow-verify writer --domain $D --storage host --duration 10 --out /tmp/vf-hello &
vramflow-verify reader --domain $D --storage host --duration 8  --out /tmp/vf-hello
wait; rm -rf $D
```

The writer creates one 1080p50 v210 flow and writes a known pattern into every grain. The reader checks every grain
(content, index, timing) and writes `summary-reader.json` to `--out`. Exit code `0` means every grain was correct.

### 4.2 The same flow in GPU memory

Change only the storage:

```bash
D=/dev/shm/vf-hello; mkdir -p $D
vramflow-verify writer --domain $D --storage cuda-linear --duration 10 --out /tmp/vf-gpu &
vramflow-verify reader --domain $D --storage cuda-linear --duration 8  --out /tmp/vf-gpu
wait; rm -rf $D
```

Grains now live in VRAM; the reader gets device pointers and checks them on the GPU. With more than one GPU, choose
one with `--device-uuid GPU-…` (`nvidia-smi -L` lists the UUIDs).

### 4.3 A media function: logo burn-in

Three processes: `gen` writes the input flow, `logo` reads it and writes a new flow with a logo, `verify` checks the
output against the CPU reference.

```bash
D=/dev/shm/vf-logo; mkdir -p $D
V="--impl gpu --storage cuda-linear"        # without a GPU: V="--impl cpu --storage host"
vramflow-func gen    --domain $D --flows 1 --duration 30 $V --out /tmp/vf-logo &
vramflow-func logo   --domain $D --flows 1 --duration 25 $V --out /tmp/vf-logo &
vramflow-func verify --domain $D --flows 1 --duration 20 --mode logo $V --out /tmp/vf-logo
wait; rm -rf $D
```

`--flows 4` runs four independent chains in the same processes.

### 4.4 Graphics with transparency over video

`gfxgen` creates two graphics flows (a small logo-sized layer and a full-screen layer), `composite` puts them over the
video from `gen`, `verify` checks the result.

```bash
D=/dev/shm/vf-gfx; mkdir -p $D
V="--impl gpu --storage cuda-linear"
L="--layers 2 --layer-xy 1488,64 --layer-xy 0,0 --opacity 1 --opacity 0.5 --base gen --out-format v210"
vramflow-func gen       --domain $D --flows 1 --duration 40 $V --out /tmp/vf-gfx &
vramflow-func gfxgen    --domain $D --flows 1 --layers 2 --layer-geom logo,full --mode pace --duration 40 $V --out /tmp/vf-gfx &
vramflow-func composite --domain $D --flows 1 $L --duration 35 $V --out /tmp/vf-gfx &
vramflow-func verify    --domain $D --flows 1 --mode composite $L --layer-geom logo,full --gfx-mode pace \
                        --duration 25 $V --out /tmp/vf-gfx
wait; rm -rf $D
```

### 4.5 Look inside a domain

While an example is running, MXL's own tool lists the flows, their position and latency:

```bash
mxl-info -d /dev/shm/vf-gfx -l
```

---

## 5. Use it from your own program

### 5.1 MXL writer and reader in GPU memory (C / C++)

The MXL API is the same for RAM and VRAM. Only two things change: the writer passes a storage option, and you get the
grain address with a helper that returns a CUDA device pointer.

```cpp
#include <mxl/mxl.h>
#include <mxl/flow.h>
#include <mxl/time.h>
#include "CudaLinearStorage.h"   // from ~/src/mxl-pr653/lib/payload-backends/cuda-linear

mxlInstance inst = mxlCreateInstance("/dev/shm/my-domain", "");

// ---- Writer ------------------------------------------------------------------
const char* opts = R"({"payload":{"backend":"cuda-linear","deviceUuid":"GPU-xxxxxxxx-…"}})";  // RAM: nullptr
mxlFlowWriter w; mxlFlowConfigInfo cfg{}; bool created = false;
mxlCreateFlowWriter(inst, flow_def_json, opts, &w, &cfg, &created);

mxlGrainStorageLayout layout{};
mxlFlowWriterGetStorageLayout(w, &layout);
std::vector<mxlGrainStorage> slots(layout.slotCount);
mxlFlowWriterMapSlots(w, layout.slotCount, slots.data());

uint64_t index = mxlGetCurrentIndex(&cfg.common.grainRate);
for (;; ++index) {
    mxlGrainInfo info{}; uint32_t slot = 0;
    mxlFlowWriterOpenGrainSlot(w, index, &info, &slot);
    void* dst = mxlCudaLinearDevicePointer(slots[slot]);   // RAM: slots[slot].host.pointer
    my_kernel<<<grid, block, 0, stream>>>(dst /*, … */);   // write the field on the GPU
    cudaStreamSynchronize(stream);
    info.validSlices = info.totalSlices;                   // whole grain written: required, see rules below
    mxlFlowWriterCommitGrain(w, &info);
    // … wait until the next field time (TAI)
}

// ---- Reader (another process) ---------------------------------------------------
mxlFlowReader r;
mxlCreateFlowReader(inst, "flow-uuid", "", &r);
mxlFlowReaderGetStorageLayout(r, &layout);
slots.resize(layout.slotCount);
mxlFlowReaderMapSlots(r, layout.slotCount, slots.data());

mxlGrainInfo info{}; uint32_t slot = 0;
mxlFlowReaderGetGrainSlot(r, index, MXL_GRAIN_VALID_SLICES_ALL, 20'000'000 /* ns timeout */, &info, &slot);
const void* src = mxlCudaLinearDevicePointer(slots[slot]);  // read it with your kernel

mxlReleaseFlowReader(inst, r);
mxlReleaseFlowWriter(inst, w);
mxlDestroyInstance(inst);
```

The snippet leaves out error handling. Complete, checked versions: writer
[`src/mxlio/verify_writer.cpp`](src/mxlio/verify_writer.cpp), reader
[`src/mxlio/verify_reader.cpp`](src/mxlio/verify_reader.cpp). The reader learns the storage type from
`layout.storageType`, so the same reader code handles RAM and VRAM flows; see
[`src/mxlio/slots.hpp`](src/mxlio/slots.hpp).

Rules worth knowing:

- Writer and reader must use the same GPU (same `deviceUuid`). Moving VRAM flows between machines is not covered here.
- Grains are addressed by index = time. A reader should stay a little behind the writer (a few milliseconds) and ask
  for the index it wants, not "the next one".
- **Set `info.validSlices = info.totalSlices` before `mxlFlowWriterCommitGrain`** when the whole grain is written.
  Readers that ask for all slices (`MXL_GRAIN_VALID_SLICES_ALL`, e.g. MXL's `mxl-gst-sink`) otherwise wait forever
  (`MXL_ERR_OUT_OF_RANGE_TOO_EARLY`).
- **Commit every index.** A missing index can stall readers that step through the flow one index at a time
  (`mxl-gst-sink` retries the same index). When you have no content for an index, commit the grain with
  `MXL_GRAIN_FLAG_INVALID` instead of skipping it; do not invent a picture.
- A grain flagged `MXL_GRAIN_FLAG_INVALID` reads as OK; check `info.flags` and `info.index` (MXL does not check that
  the index you got is the one you asked for).

### 5.2 Graphics from any program: `libvramflow-gfx`

Your renderer (an HTML/OGraf page, a game engine, a script) produces RGBA or BGRA frames. The library converts them
into a standard `video/v210a` flow (fill + key, straight alpha) in RAM or GPU memory.

```c
#include <vramflow/gfx.h>

mxlInstance inst = mxlCreateInstance("/dev/shm/my-domain", "");

vfgfx_flow_params fp = VFGFX_FLOW_PARAMS_INIT;   /* or width/height/scan/frame_rate_* for any raster */
fp.flow_id = "5fbec3b1-1b0f-417d-9059-8b94a47197ed";
fp.format = "1080i50";
char def[2048];
vfgfx_flow_def(&fp, def, sizeof def, NULL);       /* video/v210a flow definition JSON */

vfgfx_writer g;
vfgfx_writer_create(inst, def, NULL /* RAM; or {"payload":{"backend":"cuda-linear","deviceUuid":"GPU-…"}} */, &g);

vfgfx_frame f = VFGFX_FRAME_INIT;                 /* sets struct_size: never leave it out */
f.pixels = pixels;
f.width = 1920;
f.height = 1080;
f.stride = 1920 * 4;
f.pixel_format = VFGFX_PIXEL_BGRA8;               /* or VFGFX_PIXEL_RGBA8 */
f.alpha = VFGFX_ALPHA_STRAIGHT;
f.mem = VFGFX_MEM_HOST;                           /* or VFGFX_MEM_CUDA (+ f.cuda_stream) */

/* Option A — the library keeps the pace: it commits the current frame every field. */
vfgfx_start_pacing(g, 0);
vfgfx_submit(g, &f);                 /* new picture from the next field (cancels scheduled frames) */
uint64_t at = (vfgfx_current_index(g) + 4) & ~(uint64_t)1;  /* interlaced: a frame starts on an even (top) field */
vfgfx_submit_at(g, at, &f);          /* or exactly from field `at` on (frame accurate); time: vfgfx_index_time_ns */
/* … */
vfgfx_stop_pacing(g);

/* Option B — you decide every grain:  vfgfx_write(g, index, &f); */

vfgfx_stats st = VFGFX_STATS_INIT;
vfgfx_get_stats(g, &st);                          /* committed, late_skips, no_frame, next_pacing_index, … */
vfgfx_writer_destroy(g);
mxlDestroyInstance(inst);
```

Functions return `0` or an error code; `vfgfx_last_error(g)` gives the message of the last failed call (the pacing
thread has its own, `vfgfx_pacing_error(g)`). The full contract (versioning and ABI policy, threads, memory, timing,
grain indices, CUDA stream ordering) is the comment at the top of [`include/vramflow/gfx.h`](include/vramflow/gfx.h).
The library is at release 0.2.0 (`vfgfx_version()`). The SONAME carries the ABI epoch, not the release:
`libvramflow-gfx.so.1`, with symbols versioned `VFGFX_1`; releases that only add functions, struct tail fields, flags or
enum values keep it, so a program built against epoch 1 keeps working with later epoch-1 releases.
Link with `-lvramflow-gfx -lmxl`. Thanks to the C ABI the library also works from other languages
(see `tests/gfx_ctypes_smoke.py`).

A graphics flow behaves like any other flow. If a source stops or restarts, the compositor does not invent a picture:
that output grain is not committed (readers see a gap, as with a stopped video writer), and output continues when the
source returns.

### 5.3 Example: an OGraf graphic over video

[`examples/ograf/`](examples/ograf/) renders an [EBU OGraf](https://github.com/ebu/ograf) graphic (an HTML web component)
with a headless browser with transparent background, feeds the RGBA frames to `libvramflow-gfx` (`vfgfx_submit`, pacing
mode) and puts the resulting `v210a` flow over a `gen` video with `composite`. It uses only the public interfaces.
`ograf_feed` also accepts raw RGBA/BGRA frames on stdin, so any renderer that can write frames to a pipe (for example
GStreamer `wpesrc`) can replace the browser. The README there lists what was checked, the commands, and what was and
was not run (the live render used the already installed Windows Edge, about 4 fps; the official EBU sample graphics were
not downloaded).

```bash
SOURCE=edge bash examples/ograf/run-demo.sh /tmp/vf-ograf        # or SOURCE=synthetic (no renderer needed)
```

---

## 6. Command reference

All options: `vramflow-func <command> --help`, `vramflow-verify <command> --help`.

### Options shared by all `vramflow-func` commands

| Option | Meaning |
|---|---|
| `--domain PATH` | MXL domain (required) |
| `--flows N` | Number of parallel flows (1–128) |
| `--flow-base B` | First flow number (to run several independent groups in one domain) |
| `--impl gpu\|cpu`, `--storage cuda-linear\|host` | Variant, see section 1 |
| `--device-uuid GPU-…` | GPU to use (default: device 0) |
| `--duration S`, `--warmup S` | Run time (default 30 s; `0` — run until `SIGINT` / `SIGTERM` for `gen`, `logo`, `composite`, `enc`, `dec`); warm-up seconds are left out of the statistics |
| `--out DIR` | Where `summary-<command>.json` is written |
| `--format F` | Flow format, default `1080i50` (see *Formats* below) |
| `--run-id ID` | Name of this run in the domain lease (see *Domain ownership*); default `<command>-<pid>-<8 hex>` |

### Commands

| Command | Input → output | Main options |
|---|---|---|
| `gen` | — → N video (+ audio) flows, v210 in `--format` (default 1080i50), moving test pattern | `--complexity 0–3`, `--audio` |
| `logo` | N video flows → N video flows with a logo | `--logo builtin\|file.png`, `--logo-x`, `--logo-y`, `--deadline-ms` |
| `dec` | H.264 MPEG-TS over UDP → N video flows (NVDEC or software) | `--port-base`, `--playout-frames`, `--chroma-siting auto\|field\|frame` |
| `enc` | N video (+ audio) flows → H.264 + AAC MPEG-TS over UDP (NVENC or x264) | `--encoder nvenc\|x264`, `--vbitrate`, `--abitrate`, `--port-base` |
| `gfxgen` | — → N × K synthetic graphics flows (`v210a`) | `--layers K`, `--layer-geom logo,full,…\|WxH`, `--mode pace\|manual` |
| `composite` | base video + K graphics flows → `v210` or `v210a` | `--layers`, `--layer-xy X,Y` (×K), `--opacity 0–1` (×K), `--base gen\|none`, `--out-format`, `--layer-flow K:UUID` |
| `verify` | checks the output of `gen`, `logo`, `dec` or `composite` against the CPU reference | `--mode`, `--every N` (check every N-th field) |
| `clip`, `ref-crc` | helpers for the decode test clip | `--format` |
| `split-raw` | raw NV12 frames → v210 frames with the reference (chroma PSNR tool) | `--format`, `--chroma frame\|field` |
| `keepwarm` | — (not an MXL function): keeps the GPU out of its idle power state with a light periodic fill, see *GPU clocks* in section 8 | `--size-mb` (64), `--interval-ms` (5), `--duration` (0 — until stopped) |

`composite --base none --out-format v210a` builds a graphics-only layer; `--layer-flow K:UUID` takes another flow
(for example another compositor's output) as layer K, so compositors can be chained. The base flow is `flow_id(--in-role, n)`; a
`v210a` (or other) base flow produced by another tool can be selected with `--in-role` / `--flow-base`. Interlaced inputs must be
`interlaced_tff`; `interlaced_bff` is rejected.

### Formats

Every flow has an explicit format: raster, scan and frame rate (`--format`, the same value for every function in a chain).

| Format | Grain | Grain rate | Grain size (v210) |
|---|---|---|---|
| `1080i50` (default), `1080i59.94` | field (top field first; even index = top field) | 50/1, 60000/1001 | 2 764 800 B |
| `1080p25`, `1080p50`, `1080p29.97`, `1080p59.94` | frame | 25/1, 50/1, 30000/1001, 60000/1001 | 5 529 600 B |
| `720p25`, `720p50`, `720p29.97`, `720p59.94` | frame | as above | 2 488 320 B |
| `2160p25`, `2160p50`, `2160p29.97`, `2160p59.94` | frame | as above | 22 118 400 B |

- In the MXL flow definition `grain_rate` is the *frame* rate (25/1 for 1080i50); MXL runs interlaced flows at twice that.
  A reader checks that the flow's grain rate matches the expected format and names both rates if not.
- Progressive 4:2:0 → 4:2:2 uses frame chroma (type-0 siting, `split_nv12_progressive`); interlaced uses field chroma.
  `dec` picks the path per decoded frame from its interlaced flag, so a progressive frame in an interlaced flow is still
  converted with frame chroma (`dec --chroma-siting auto`). Our own `enc --encoder nvenc` cannot field-encode on this GPU:
  its 1080i streams are flagged progressive but carry field chroma, so decode them with `dec --chroma-siting field`
  (x264 streams are flagged correctly). Against FFmpeg's bilinear swscale (`tests/perf/chroma_psnr.py`) the progressive
  path is bit-exact.
- Audio samples per grain are exact integers: `[sample_begin(i), sample_begin(i + 1))` — 960 per 1080i50 field,
  1920 per 1080p25 frame, 800 / 801 at 59.94.
- Tested end to end: 1080i50, 1080p25 and 1080p50 for every function (`gen`, `logo`, `verify`, `enc`, `dec`, `gfxgen`,
  `composite`), 720p50 for `logo`. Other listed formats pass the unit and kernel tests (720p, 2160p, 59.94) but have no
  end-to-end test yet. `libvramflow-gfx` accepts every listed rate. BFF and other rates are rejected with a message.
- The MXL ring keeps 200 ms by default (10 fields of 1080i50, only 5 frames of 1080p25); set
  `urn:x-mxl:option:history_duration/v1.0` in the domain's `options.json` for a deeper ring.

### Flow IDs

`vramflow-func` uses fixed, readable flow IDs so the functions find each other without configuration:
`4d584c47-00RR-4000-8000-NNNNNNNNNNNN`, where `RR` is the role (`01` gen, `02` logo, `03` audio, `04` dec, `05`
graphics, `06` compositor) and `N…` is the flow number in hex (graphics: `flow × 8 + layer`).

### Start order, waits and restarts

Functions can be started in any order. A function that reads flows (`logo`, `composite`, `verify`, `enc`) waits for its
inputs and prints `laukiu flow <id>, N s` to stderr every 5 s while it waits.

| Option (readers) | Meaning |
|---|---|
| `--input-wait-ms N` | `0` (default): wait for the inputs without a limit. `N > 0`: give up after `N` ms with exit code `4` and a message that names the missing flows and the limit |
| `--on-format-change exit\|wait` | An input came back with another format or storage: `exit` (default) stops with exit code `6` and names both formats; `wait` keeps waiting until it comes back with the original one |

- A reader attaches only to a flow that has a live writer and already produces grains. Old files left by a killed
  producer are ignored.
- If a producer disappears (killed, or its flow replaced), the reader notices within about one grain, keeps running and
  counts every index it could not produce as `missing_fields` (`missing_frames` in `enc`; `missing` in `verify`). When
  the producer comes back with the same flow ID, the reader attaches again by itself (`reattached` per flow; `reopened`
  in `composite`) and continues. Indices of the gap are not written to its own output yet (a reader downstream sees a gap);
  marking them as INVALID grains is planned with the gap policy (plan item P5).
- A restarted producer removes the stale flows of its killed predecessor (`mxlGarbageCollectFlows`) before it creates its
  own, so a restart with the same flow IDs works. A flow that still has a live writer is never removed.
- `SIGINT` / `SIGTERM` while waiting for inputs ends the function cleanly with exit code `0`.

### Domain ownership

Every `vramflow-func` process holds a lease file `<domain>.vramflow/<run-id>.lease` **next to** the domain folder (the
MXL domain itself stays clean: `mxl-info` and MXL garbage collection never see it). The lease is an `flock`, so the kernel
releases it even after `kill -9`. The lease lists the output flows of that run.

- `--run-id ID` (all commands) names the run; the default is `<command>-<pid>-<8 hex>`. It is written to
  `summary-<command>.json` (`run_id`).
- Creating an output flow that already has a live writer fails (exit code `2`) and the message names the owner's run ID.
- The functions never delete a domain. Test harnesses create a fresh domain per run (`tests/perf/domain.py`:
  `make_domain(tag)` → `/dev/shm/vf-<tag>-<8 hex>`) and delete it only when no live owner is left (`remove_domain`
  returns `False` and deletes nothing otherwise). Two runs with the same tag therefore never delete each other's flows.
- `flock` works on `tmpfs` (`/dev/shm`) and local file systems, not on network file systems.
- **Keep the domain path short: at most 48 characters for GPU (`cuda-linear`) flows.** A reader of a GPU flow connects
  to the Unix socket `<domain>/<flow-id>.mxl-flow/payload.sock`, and a socket path cannot be longer than 107 characters.
  With a longer domain path the writer still creates the flow, but no reader in another process can attach to it.
  `vramflow-func` therefore refuses such a path at start (exit code `1`, the message gives the length and the limit), and
  `make_domain` shortens a long tag. Flows in RAM (`--storage host`) have no such limit.
- While a reader waits, its `laukiu flow …` line also says why a flow that exists was not accepted (another format, the
  reader could not open it), so a wait never looks like a missing flow when it is something else.

### Exit code and summary

Exit code `0` means everything was correct. `1` — wrong options or configuration, `2` — setup failed (CUDA, MXL, an output
flow owned by another run) or an output kept failing (3 open / commit errors in a row on one output; the message names the
output), `3` — verification errors, `4` — something was lost (late, missing, a single output error, inputs did not appear
within `--input-wait-ms`), `6` — an input came back with another format (`--on-format-change exit`). Every run writes
`summary-<command>.json` with counters (fields written, read, lost, late, invalid, verification errors), a latency histogram,
the run ID and the software / GPU versions.

Counters of the functions on the shared tick engine (`gen`, `logo`, `composite`, `enc`, `dec`) that need a word:

| Counter | Meaning |
|---|---|
| `output_errors` (per flow) | Opening or committing an output grain failed; the index has no grain. Counted as a loss (exit `4`); 3 in a row on one output end the run with exit `2` |
| `clock_backward` (process) | The output index was already written — the clock went back (MXL refuses to write an index twice). Not an error, not a loss |
| `late_fields` → `missing_fields` | An input that stopped delivering (alive but stuck, e.g. `SIGSTOP`) costs 3 `late_fields` (the engine waits for it up to the deadline 3 times), after that it is *stalled* and every further index is `missing_fields` (no wait at all) until it delivers again |
| `read_errors` (per flow) | Replaces the old `read_status_<code>` keys: any unexpected read status of an input. The status code itself is no longer in the summary |
| `reopen_mismatch` (`composite`) | The input that came back with another format (with `--on-format-change exit`, exit `6`) |
| `lateness_max_us` (per flow) | Largest commit lateness after `T(i)` of this flow after warm-up, µs (`enc`: frame queued). The `out_lateness` histogram covers the whole tick |
| hist `proc_ms` / `wait_ms` / `held_ms` | `proc_ms` — processing only (kernels, sync, commit; summed over the phases of one index); `wait_ms` — tick open → first processing; `held_ms` — only for an index processed in more than one phase: time spent waiting for late inputs between phases |
| `enc`: `audio_underrun_frames` | The frame's audio was not there when the frame was encoded (silence is encoded instead). Audio is a *wanted* input: the engine waits for it at most until the first cut `a(i)` and never on the engine thread, so one stream without audio does not delay the others |
| `dec`: `clock_overrun_frames`, `clock_overflow_drops` | Frames dropped because a backward clock step of `S` made the tick wait: at most `ceil(S / frame)` frames, once, at the first frame tick after the step. Everything else above the buffer limit is `overrun_frames` (exit `4`)

---

## 7. Tests and measurements

```bash
bash scripts/linux/run-all.sh --soak skip      # everything: probes, upstream tests, unit / GPU / functional tests, report
bash scripts/linux/61-make-test-clip.sh        # synthetic 1080i50 H.264 clip for the dec / chain tests

# One capacity level: 8 logo flows on GPU flows, 60 s
python3 tests/perf/capacity.py ~/evidence/try --func logo --n 8 --seconds 60 --impl gpu --storage cuda-linear

# GPU vs CPU campaign and its report
bash scripts/linux/71-compare.sh ~/evidence/cmp-logo logo 1,2,4,8,16,32,64
python3 tests/perf/compare_report.py report.md ~/evidence/cmp-logo
```

Only synthetic test media are used (generated patterns and a generated clip).

## 8. Results so far

RTX 4070 Laptop 8 GB, WSL2 Ubuntu 24.04, 1080i50, 60 s per level, every output verified. N\* = the largest number of
flows for which this level and all smaller ones passed (0 correctness errors, ≤ 0.2 % lost fields). Indicative: a
native Linux rerun will give the authoritative numbers.

| Function | CPU (C) N\* | GPU with GPU flows (G) N\* | CPU cores per flow, C → G |
|---|---|---|---|
| Logo burn-in, MXL → MXL | 16 | ≥ 64 (limit not reached) | 0.057 → 0.005 |
| H.264 decode, TS/UDP → MXL | 8 | 24 (16 strict) | ~0.55 → ~0.04 |
| Full chain: TS → decode → logo → H.264 + AAC → TS | 4 | ≥ 12 (limit not reached) | ~2.0 → ~0.33 |

- Full chain at N=12: CUDA cores ~10 % busy, NVENC 64 %, NVDEC 50 %, ~34 W. The single NVENC engine of this laptop
  GPU is the limit, not compute.
- Grain latency p50 1.2 ms on GPU flows vs 2.9 ms on RAM flows. GPU processing with RAM flows (H) collapses at 32 logo
  flows; with GPU flows (G) it does not.
- **GPU clocks.** When a laptop GPU (WSL2 / WDDM) idles in power state P8 and any process starts a burst of wide GPU work
  (creating `cuda-linear` flows, a large `cudaMemset`, a full-GPU kernel, another function starting or stopping), the
  P8 → P3 step stops the GPU work of **all** processes for 50–350 ms: live functions miss real-time deadlines and skip
  indices. Measured: 35 of 70 such events stalled a 7-flow `gen` without a constant light GPU load (worst 368 ms, 290
  indices skipped), 0 of 70 with `vramflow-func keepwarm` running (worst 22 ms). Its cost: GPU busy 19.5 %, +8.5 W (11.3 W instead
  of 2.8 W at idle). Operational rule: on laptops and development machines run `vramflow-func keepwarm` (defaults:
  64 MB every 5 ms; `--duration 0` until stopped) while anything is live; the test harnesses (`functional.py`,
  `capacity.py`) start it for GPU variants by themselves (`VRAMFLOW_GPU_KEEPWARM=0` turns this off) and record it in every
  result. On servers lock the GPU clocks (`nvidia-smi -pm 1`, `-lgc`) — **not verified there yet**: rerun
  `tests/perf/gpu_event_hiccup.py` on the server before relying on it. Details:
  [`docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md`](docs/rezultatai/2026-11-12-gpu-ivykiu-trukdziai.md).

Full tables and notes: [`docs/rezultatai/2026-11-11-gpu-vs-cpu.md`](docs/rezultatai/2026-11-11-gpu-vs-cpu.md).

Graphics compositor (`composite`), 1 base + K layers → `v210`, 60 s per level, deadline 30 ms, same N\* rule (tolerant / strict):

| Layers K | C — CPU | H — GPU, RAM flows | G — GPU, GPU flows |
|---|---|---|---|
| 1 (logo 384×216) | 16 / 2 | 4 / 1 | 4 / 4 |
| 2 (logo + full-screen) | 8 / 4 | 4 / 0 | 2 / 0 |
| 4 (logo + 3 × full-screen) | 4 / 4 | 4 / 2 | 0 / 0 |

Read this table with care: the Windows power profile was "Whisper"; during the G runs the GPU clock averaged only
210–690 MHz; every level was measured once (no repeats); G levels sometimes fail at low N while larger N pass, so N\* (all
smaller levels pass) is pessimistic for G (the largest passing N was 32, 32 and 16 for K = 1, 2, 4; K = 4, N = 32 hit the
8 GB VRAM limit). Whether the clock explains the failures is a hypothesis that was not tested. The first G measurement was
invalid (a `gfxgen` bug, fixed) and was repeated. Details, latencies and VRAM model:
[`docs/rezultatai/2026-11-11-composite.md`](docs/rezultatai/2026-11-11-composite.md).

## 9. Repository layout

| Path | Contents |
|---|---|
| `include/vramflow/gfx.h` | Public C API of `libvramflow-gfx` |
| `src/func/` | `vramflow-func` commands |
| `src/gfx/` | Graphics: RGBA → v210a and compositing (CUDA, CPU, independent reference) |
| `src/media/` | Video kernels: logo, NV12 ↔ v210, test pattern (CUDA, CPU, reference) |
| `src/mxlio/`, `src/gpu/` | `vramflow-verify` writer and reader |
| `examples/ograf/` | OGraf graphic → `libvramflow-gfx` → `composite` example |
| `tests/` | `unit/`, `gpu/` (Catch2), `functional/`, `perf/` (capacity and reports) |
| `scripts/linux/` | Build and run scripts (native Linux and WSL2); `scripts/wsl/` keeps one-line redirects to them for one release |
| `upstream/` | Draft patches for dmf-mxl/mxl PR #653 (U1–U4) with evidence |
| `docs/` | Design notes, requirements, plans and results (in Lithuanian) |

## 10. Upstream patches

[`upstream/README.md`](upstream/README.md): U1 fixes a hang of a container test for non-root users; U2 and U3 add
tests for synchronization groups and partial grains with GPU storage; U4 adds tests for `video/v210a` on
`cuda-linear`. With U1–U3 the PR #653 test suite passes 136/136. They are drafts; the author submits them with
`git commit -s` (DCO).

## 11. Limits

- Measured on one laptop GPU under WSL2 so far; native Linux servers are next.
- Variant H exists for logo, compositor and transport; not yet for decode / encode / chain.
- No 2-hour soak and no full sanitizer run of the media functions yet.
- NVENC on this GPU encodes progressive frames; x264 encodes true interlaced (MBAFF, top field first).
- VRAM flows work between processes on the same machine and GPU; other machines are outside this project.

## 12. License and credit

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). You may use, change, distribute and sell this work,
also in closed products. When you redistribute it or a derivative, keep the NOTICE file: it names the author. If this
work helps a product, a standard or a paper, please credit **"VRAMFlow by Deimantas Kiršinas"** with a link to this
repository.

Binaries linked with an FFmpeg build that includes libx264 fall under the GPL when distributed (see NOTICE).

Developed with the help of AI coding assistants (Claude). The author reviewed the design, ran every measurement on real
hardware and is responsible for the content.

MXL is a project of the EBU and the Linux Foundation; this repository is independent and not endorsed by them.
