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
| `vramflow-func` | Ready-made media functions on MXL flows: test generator, real-source ingest with audio (MPEG-TS over UDP / RTP / SRT, HLS), logo burn-in, H.264 decode, H.264 + AAC encode, alpha graphics compositor, output checker. Each runs on the CPU or the GPU. |
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

### 4.5 Ingest a real source with audio

`ingest` turns a live MPEG-TS source (UDP, RTP, SRT or HLS) into one video flow (v210, in GPU memory) and one audio flow
(48 kHz float32) per source, both placed by the source's own timestamps. A local synthetic source:

```bash
bash scripts/linux/62-make-av-clip.sh ~/media-test/av          # synthetic H.264 + AAC clips with A/V sync markers (once)
D=/dev/shm/vf-ing; mkdir -p $D
python3 tests/perf/ts_burst.py ~/media-test/av/av_1080i50.ts --port 20000 --duration 45 &   # local UDP sender, paced by PCR
vramflow-func ingest --domain $D --input udp://127.0.0.1:20000 --duration 40 --out /tmp/vf-ing &
vramflow-func verify --domain $D --mode ingest --flows 1 --audio 1 --duration 35 --out /tmp/vf-ing
wait; rm -rf $D
```

The format is taken from the stream (`--format auto`, here 1080i50). `verify --mode ingest` reads both MXL flows and reports,
for every sync pulse, the offset between the audio impulse and the video frame it belongs to (`summary-verify-ingest.json`:
`av_max_abs_us`, and the `ingest` array with every pulse); in this example every offset is `0`.

### 4.6 Look inside a domain

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
- **An INVALID grain carries black, not the slot's old picture.** The ring slot of an INVALID grain still holds the
  picture of a previous lap (about 200 ms old), and a reader that ignores the flag would replay that loop for the whole
  outage. VRAMFlow writers therefore write black into the slot as well (`v210`: Y 64 / Cb Cr 512; `v210a`: also key 0,
  transparent) — once per ring slot per outage: a slot that already holds black from this outage is only flagged again,
  and a slot overwritten by real content loses the mark. Audio flows have no flag: a missing audio block is committed
  as silence (`gen`: `gap_silence` per flow).
- **Not every INVALID grain is black.** Only grains a VRAMFlow writer *commits* as gaps carry black (they have
  `validSlices == totalSlices`). Indices a writer never commits — the engine or the library pacer woke up more than a
  grain late and skipped them (`late_skips`, `clock_skips`; they are not filled while the writer is already late,
  decision K-T6) — are marked INVALID by MXL itself when the next index is opened, with `validSlices == 0` and the
  **old** slot payload. So always honour the flag; `verify` counts the first kind as `invalid_black` /
  `invalid_not_black` (the latter is an error) and the second as `invalid_skipped` (not an error).
- A grain flagged `MXL_GRAIN_FLAG_INVALID` reads as OK; check `info.flags` and `info.index` (MXL does not check that
  the index you got is the one you asked for).
- **An INVALID grain means "no content for this index", not an error of the flow.** A reader that is itself a writer
  applies its own gap policy to its output for that index (VRAMFlow functions: `--on-loss`, see section 6) and goes on;
  it never copies the stale payload of an INVALID grain. It is the upstream's gap, not the reader's loss: VRAMFlow
  functions count it as `gap_upstream`, and it does not change their exit code. A reader blocked on an index wakes up as soon as the INVALID
  grain is committed — it does not wait for the next grain.

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
The library is at release 0.3.0 (`vfgfx_version()`; changes in the *CHANGES* section of `gfx.h`). The SONAME carries the ABI epoch, not the release:
`libvramflow-gfx.so.1`, with symbols versioned `VFGFX_1`; releases that only add functions, struct tail fields, flags or
enum values keep it, so a program built against epoch 1 keeps working with later epoch-1 releases.
Link with `-lvramflow-gfx -lmxl`. Thanks to the C ABI the library also works from other languages
(see `tests/gfx_ctypes_smoke.py`).

A paced writer commits every index: until the first `vfgfx_submit` / `vfgfx_submit_at` frame is current, each grain is
committed with `MXL_GRAIN_FLAG_INVALID` and a transparent-black payload (written once per ring slot; counted in
`stats.no_frame`, not in `stats.committed`); from the first frame on, the current frame repeats until it is replaced.
This is a behaviour change of release 0.3 (before, nothing was committed until the first frame): `stats.last_committed_index`
is no longer `VFGFX_INDEX_NONE` before the first frame, and `vfgfx_write` after `vfgfx_stop_pacing` needs an index above
it. `stats.last_frame_index` (new in 0.3, same struct size) is the last index committed *with a frame*. See *CHANGES* and
*GAPS* in `gfx.h`.

**A failed graphics layer never takes the programme off air.** In `composite` only the base is required; every layer is
optional. A layer that is INVALID (for example a paced writer before its first frame), not attached yet, stalled, gone,
late at the unit's deadline, or rejected (geometry, storage or rate do not fit; checked when it first attaches) is fully
transparent for that index: the base passes through bit-exactly — the output equals the reference without that layer.
Each such index is counted per output as `layer_absent` (and `layer_absent_<k>`), the change of state is an operator event
(`layer_absent` / `layer_present`), and it is never a loss of the output. A layer that comes back is used again from the
next index (interlaced: from the next frame boundary, an even index). With `--base none` and every layer absent the output
is transparent black (key 0), not a gap. Only a missing or INVALID **base** makes the output index a gap, committed
according to `--on-loss` (default: an INVALID grain).

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
| `--duration S`, `--warmup S` | Run time. Default `0` — a service: run until `SIGINT` / `SIGTERM` (`gen`, `gfxgen`, `logo`, `composite`, `enc`, `dec`; `verify` keeps 30 s). `S > 0` — a measurement run. Warm-up seconds are left out of the statistics |
| `--strict-exit auto\|on\|off` | Exit code policy (see *Exit code and summary*); `auto` (default) is strict for `--duration S > 0` and lenient in service mode |
| `--drain-ms N` | On stop: how long worker queues (`enc`) may drain, default 500; a second signal stops at once |
| `--rt-priority N`, `--cpu-affinity LIST` | Engine thread only: `SCHED_FIFO` priority 1–99 and CPU list such as `2` or `0,2-3` (default: unchanged). Without the privilege the run continues with a warning |
| `--out DIR` | Where `summary-<command>.json` is written |
| `--format F` | Flow format, default `1080i50` (see *Formats* below) |
| `--run-id ID` | Name of this run in the domain lease (see *Domain ownership*); default `<command>-<pid>-<8 hex>` |
| `--admission auto\|off\|warn\|enforce` | Admission control against the measured GPU capacity (see *Capacity and admission*). `auto` (default): `enforce` when a capacity file is found, `warn` when there is none; `enforce`: flows that do not fit are not started (exit code `5`); `warn`: a flow over capacity starts with a warning line; `off`: no check (manual experiments and capacity measurement). Not applied to `verify` and `--impl cpu` |
| `--capacity-file PATH` | Capacity file from `tests/perf/capacity_file.py`, or a directory with `capacity-<GPU UUID>.json`; default: the `VRAMFLOW_CAPACITY` environment variable |
| `--admission-headroom H` | Limit = capacity × `H` for Mpix/s and VRAM, default `0.85`; NVENC sessions are a hard driver count and get no headroom |

Options of the MXL video writers (`gen`, `logo`, `composite`, `dec`) — every index is committed (gap policy K3):

| Option | Meaning |
|---|---|
| `--on-loss invalid\|freeze\|black` | An index without content (input gone or late, decoder underrun): `invalid` (default) commits an INVALID grain — nothing is invented, a reader wakes up at once; `freeze` repeats the last good grain of the same field parity for at most `--hold-ms`, then black; `black` commits a black grain at once (`v210a`: transparent). Without any good grain yet, `freeze` is black. Interlaced: with `freeze` / `black` content returns only on a frame boundary (even index) |
| `--hold-ms N` | `freeze`: how long to repeat, default 1000 |

Timing of the readers (`logo`, `composite`, `enc`): an input that is not there by the deadline `D` after `T(i)` makes
that index a gap. `--deadline-ms 0` (default) means **`D = 3·P/4`** of the grain period `P` (for `enc`: `3·P/2`, see
below): 15 ms at 50 grains/s
(1080i50, 1080p50, 720p50), 30 ms at 25p, 12.5 ms at 59.94 Hz; a value above `2P − G` is clamped with a warning. The
summary states the effective value (`deadline_us`, proc). WSL2 measurement campaigns (`tests/perf/capacity.py`) and
the functional harness (`tests/functional/functional.py`: `gen`, `logo`, `enc`, `composite`) pass `--deadline-ms 30`
explicitly, because GPU scheduling hiccups of 17–30 ms are common there; the tests that pin the default timing (isolation
gates, `ffmt_logo`) pass `--deadline-ms 0` and attribute an index the upstream itself delivered late to the upstream. `gen --deadline-ms` sets only the threshold of its own "late commit" counter and stderr line
(default: the same `3·P/4`, i.e. what a reader with the default deadline would miss).

`enc` has its own default: a frame's audio is complete only at `T(j) + P` (a real-time audio writer finishes the block
at the end of its period), so with `D = 3·P/4` all real-time audio would be lost. Until the edge timing contract exists,
`enc --deadline-ms 0` means **`D = 3·P/2`** (30 ms at 50 grains/s, 60 ms at 25p) and the audio input is *expected* at
`T + P`: audio arriving there is on time (not counted late every frame), and the engine does not hold the first cut of
the other streams waiting for it.

Per flow, the summary counts the committed gaps by action: `gap_invalid`, `gap_frozen`, `gap_black` — the gaps this
function caused because an input it needed was missing or late. Gaps that are not this function's loss are counted by
cause instead, whatever the action: `gap_startup` (a required input never attached yet), `gap_upstream` (a required input
delivered an INVALID grain — the upstream had nothing for that index; `enc`: `gap_upstream_frames`) and `gap_align`
(interlaced `freeze` / `black`: content came back on an odd index and is held to the next frame boundary). A loss is only
what a function failed to deliver although its inputs were valid; the strict exit code `4` and `any_loss` follow losses
only. Indices the engine itself skipped because it woke up late (`late_skips`) are
not filled: MXL marks them invalid when the next index is opened (their audio blocks, which have no flag, are committed as
silence and counted as `gap_silence`). Cost per gap (1080i50 grain, 2.76 MB): `invalid` — on the first lap of an outage
one black copy into the slot (host `memcpy` ≈ 110 µs p50, 210 µs p99; GPU D2D ≈ 26 µs p50, 110 µs p99), on later laps of the same
outage only the flag (≈ 0.2 µs); `freeze` / `black` one copy every gap — host `memcpy` ≈ 80–100 µs, GPU D2D ≈ 27 µs.
Every policy except `leave` keeps one black grain per raster in the flow's memory (`invalid` too, for the INVALID
payload).

### Commands

| Command | Input → output | Main options |
|---|---|---|
| `gen` | — → N video (+ audio) flows, v210 in `--format` (default 1080i50), moving test pattern | `--complexity 0–3`, `--audio` |
| `logo` | N video flows → N video flows with a logo | `--logo builtin\|file.png`, `--logo-x`, `--logo-y`, `--deadline-ms` |
| `ingest` | Real sources: MPEG-TS over UDP / RTP / SRT or HLS → per source one video flow (v210) + one audio flow (48 kHz float32), placed by the source PTS | `--input URL` (repeat), `--decoder nvdec\|cuvid\|sw`, `--latency-ms`, `--decode-lead-ms`, `--format auto` — see *Ingest* below |
| `dec` | H.264 MPEG-TS over UDP → N video flows (NVDEC or software); kept for the test clip, real sources use `ingest` | `--port-base`, `--playout-frames`, `--chroma-siting auto\|field\|frame` |
| `enc` | N video (+ audio) flows → H.264 + AAC MPEG-TS over UDP (NVENC or x264) | `--encoder nvenc\|x264`, `--vbitrate`, `--abitrate`, `--port-base` |
| `gfxgen` | — → N × K synthetic graphics flows (`v210a`) | `--layers K`, `--layer-geom logo,full,…\|WxH`, `--mode pace\|manual` |
| `composite` | base video + K graphics flows → `v210` or `v210a` | `--layers`, `--layer-xy X,Y` (×K), `--opacity 0–1` (×K), `--base gen\|none`, `--out-format`, `--layer-flow K:UUID` |
| `verify` | checks the output of `gen`, `logo`, `dec` or `composite` against the CPU reference; `--mode ingest` measures A/V sync and marker continuity of `ingest` flows | `--mode`, `--every N` (check every N-th field), `--ref`, `--audio 1` |
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
graphics, `06` compositor, `07` ingest video, `08` ingest audio) and `N…` is the flow number in hex (graphics: `flow × 8 +
layer`; ingest: source position + `--flow-base`). The audio flow of an ingest source shares the video flow's NMOS group hint.

### Start order, waits and restarts

Functions can be started in any order. While an input has never appeared, a reader prints `laukiu flow <id>, N s` to
stderr every 5 s.

- `logo` starts **immediately** by default (`--start immediate`): it creates its output flows at once (format from
  `--format`, storage from `--impl` / `--storage`), the engine ticks, and every output index is committed according to
  `--on-loss` until the input appears (default: INVALID grains). These start-up gaps are counted per flow as
  `gap_startup` and are not losses. An input that appears with another storage than expected is rejected like a format
  change (`--on-format-change`). `--start wait-all` keeps the old behaviour: wait for all inputs, then create the outputs.
- `composite`, `enc` and `verify` wait for their inputs before they start (`composite`: only for its base flows — the
  output raster is the base's; layers are optional and attach whenever they appear; `enc` has no MXL outputs).

| Option (readers) | Meaning |
|---|---|
| `--start immediate\|wait-all` | `logo` only, see above; default `immediate` |
| `--input-wait-ms N` | `0` (default): wait for the inputs without a limit. `N > 0`: give up after `N` ms with exit code `4` and a message that names the missing flows and the limit (also with `--start immediate`, where the outputs already exist) |
| `--on-format-change exit\|wait` | An input came back with another format or storage: `exit` (default) stops with exit code `6` and names both formats; `wait` keeps waiting until it comes back with the original one |

- A reader attaches only to a flow that has a live writer and already produces grains. Old files left by a killed
  producer are ignored.
- If a producer disappears (killed, or its flow replaced), the reader notices within about one grain, keeps running and
  counts every index it could not produce as `missing_fields` (`missing_frames` in `enc`; `missing` in `verify`). When
  the producer comes back with the same flow ID, the reader attaches again by itself (`reattached` per flow; `reopened`
  in `composite`) and continues. Every index of the gap is still committed to its own output according to `--on-loss`
  (default: INVALID grains, counted as `gap_invalid`), so a reader downstream never waits on a missing index.
- Attaching (creating the reader and, for `cuda-linear`, mapping the GPU payload — up to ~18 ms) and releasing a detached
  reader run on a helper thread of the input, never inside a tick: an input that appears late or returns costs the
  running outputs nothing; the engine adopts the prepared reader between ticks.
- A restarted producer removes the stale flows of its killed predecessor (`mxlGarbageCollectFlows`) before it creates its
  own, so a restart with the same flow IDs works. A flow that still has a live writer is never removed.
- `SIGINT` / `SIGTERM` while waiting for inputs ends the function cleanly with exit code `0`.
- A composite **layer** that comes back with another format is not attached; it stays transparent (`layer_absent`),
  one operator event and `layer_format_mismatch` per new reason, and the programme continues. Only the **base** follows
  `--on-format-change`.
- A layer is waited for only until the first cut a(i) (`T + E + ro + Wr`), never to the final cut, so layers never
  delay the programme or its advertised bound. A layer that disappeared is used again only after
  `--layer-return-frames` consecutive frames on time (default 12), and only from a frame boundary.

### Ingest (real sources with audio)

`vramflow-func ingest --domain D --input URL [--input URL …]` — one process, any number of sources. Transports are the
libavformat protocols of the system FFmpeg: `udp://`, `rtp://` (MPEG-TS payload, PT 33), `srt://` (if FFmpeg has it; otherwise
the start fails with "srt protokolas nepalaikomas"), `http(s)://…` (HLS, recognised by content; or MPEG-TS over HTTP), or a
local MPEG-TS file.

| Option | Meaning |
|---|---|
| `--input URL` | A source (repeat for more). Source `k` writes flows `07`/`08` number `--flow-base + k` |
| `--format auto\|F` | `auto` (default): taken from the stream before any flow is created; `F`: the stream must match |
| `--decoder nvdec\|cuvid\|sw` | `nvdec` (default): libavcodec `h264` / `hevc` with the CUDA hwaccel; `cuvid`: `h264_cuvid` (frames come out 4–8 frames after their packet); `sw`: software. `--impl cpu` always uses `sw` and writes host flows |
| `--latency-ms N` | A frame is shown `N` ms after its packet arrived (default 1000). HLS: at least 3 segments |
| `--decode-lead-ms N` | A packet enters the decoder no earlier than `N` ms before its time (default 400): bounds the decoded frames kept in VRAM |
| `--audio 0\|1`, `--audio-channels N` | Audio flow on / off (default 1), channel count in MXL (default 2; mono is upmixed, extra channels dropped) |
| `--input-wait-s S` | How long to wait for the first data at start (default: no limit); after that exit `2`, the message names the URL and the reason, and no flow is created |
| `--input-silence-ms N` | A source silent for longer reconnects with back-off 1, 2, 4 … ≤ 30 s (default 2000; HLS: at least 3 segments) |
| `--max-comp-ppm X` | Audio drift compensation limit (default 1000 = ±0.1 %, inaudible) |
| `--hls-max-target-s S` | Largest accepted HLS `#EXT-X-TARGETDURATION` (default 30); a longer one rejects that source with a clear message |
| `--hls-max-playlist B` | Largest accepted HLS playlist in bytes (default 1048576); a larger one ends that session |
| `--hls-same-host` | HLS: segments, keys, maps and variants only from the top-level playlist's host:port |
| `--libav-watchdog-s S` | libavformat call watchdog margin (default 30, 5–3600): a call stuck that much past its own bound, or without any progress for `S` s, ends the process with exit `7` (see *libavformat redirect-cache hang*) |
| `--max-queue-mb N` | Packet queue cap per queue, MB; each source has two queues (video and audio), so up to 2 × N MB per source (default 0 = (latency + 2.4 s) × 80 Mbit/s per queue); oldest packets dropped and counted |
| `--srt-passphrase-from env:NAME\|file:PATH` | SRT passphrase from an environment variable or a file, not from the command line; 10–79 characters without `&`, `#`, `?`, `%`, `+` or spaces (libavformat turns `+` into a space) |
| `--max-audio-ring-mb N` | Audio ring limit per source (default 256 MB; the ring grows with the latency, i.e. with the HLS segment) |
| `--on-loss`, `--hold-ms` | Video index without a frame — as for the other writers |

**Security rules (URLs and playlists).** The HLS guard below is defence in depth around libavformat's HTTP/HLS client, not
a sandbox: libavformat still parses the responses and follows redirects itself, so configure HLS sources only for known
servers that you operate or trust. The `--input` URL is trusted, but its scheme must be `udp`, `rtp`, `srt`, `http`,
`https` or a local file path; anything else (`file:`, `tcp:`, `concat:`, `data:`, `crypto+…` …) is a usage error (exit `1`).
Every libavformat open, including nested ones (HLS segments, RTP → UDP, HTTP → TCP/TLS), gets an explicit
`protocol_whitelist` and `format_whitelist` derived from that scheme (local file: `file` / `mpegts` only); HLS segment
demuxers get the same list without `hls`. An `http(s)` source is first requested by ingest itself: HLS is recognised by
its content (`#EXTM3U`), not by the `.m3u8` suffix, and for HLS libavformat is handed exactly that response (no second
request for the top-level playlist). A source whose first response is not M3U8 is requested a second time by the guarded
open (so a plain MPEG-TS source is requested twice per session) and opened as MPEG-TS only (`format_whitelist mpegts`),
with a single open per session; an M3U8 answer on that second request is refused. Every response libavformat opens
is inspected (first 4 KiB) without re-requesting it: the inspected bytes are served from memory and reading continues on
the same connection; a size query is answered from the response headers only and never falls back to seeking (a
`Content-Length` of 2^63 or more cannot trigger further ranged requests). An M3U8 answer is accepted only from a URL known to be a playlist (the top-level URL and its final
location after redirects, the variants and renditions of the checked master playlist, and reloads of those); a segment,
key or init section that answers with a playlist is refused. Every such playlist (variants, reloads) is read completely
(at most `--hls-max-playlist` bytes, default 1 MiB), checked completely and then given to libavformat as the same bytes,
parsed the way libavformat parses it (lines end at LF, CR, CR LF or NUL; attributes quoted or bare): a master playlist is
allowed only at the top (no nesting, no loops), at most 15 variants, 10 000 entries and 4 `#EXT-X-MAP` lines per playlist,
`#EXT-X-TARGETDURATION` ≤ `--hls-max-target-s`, and every URL it names (variants, segments, `URI` of keys, maps and
renditions) must resolve to `http`/`https` with `://` directly after the scheme (`http:host/…` is refused everywhere,
also as `--input`: libavformat would take the host from the path). Any `#EXT-X-MEDIA` line, with or without `URI`, makes a
playlist master level: at most 32 in the top-level playlist, none in variants or reloads. libavformat keeps every
`#EXT-X-MAP` init section and every `#EXT-X-MEDIA` rendition until the session ends, so init sections are also capped at
16 MiB of estimated memory per session. libavformat copies each segment's resolved URL and, while an AES-128 key is
active, the resolved key URL (up to 4 KiB each): that estimate is capped at 16 MiB per playlist and 64 MiB for all
playlists of a session together (a URL listed in several master entries counts once per entry: libavformat keeps a separate
playlist for each `#EXT-X-STREAM-INF` and each `#EXT-X-MEDIA` with `URI`) (during a reload libavformat briefly holds the previous list too). A violation ends only that session (the source reconnects
with back-off; the other sources are not affected). HTTP reads time out after 10 s, and a reconnecting `http(s)` source
gives up an open after `--input-wait-s` (default 30 s). Packet queues are capped in bytes (`--max-queue-mb`, default
(latency + 2.4 s) × 80 Mbit/s per queue; each source has two queues, video and audio); at the cap the oldest media
packets are dropped and counted (`queue_drops`, `queue_overflows`, one log line per episode), control markers are never
dropped. User info and all query values (`passphrase=…`, tokens) are replaced by `***` in log lines, error messages, the
summary and the stats file, including libavformat's own log lines (a quoted URL is redacted up to its closing quote). Not
redacted: a query value that contains a space in an unquoted URL, and secrets carried in the URL path instead of the
query. To keep the SRT passphrase out of `/proc/*/cmdline`, pass it with `--srt-passphrase-from env:NAME` or
`--srt-passphrase-from file:/path` (added to `srt://` inputs without `passphrase=` in memory only; `+` is refused because
libavformat turns it into a space).

**Server-side request forgery (SSRF) caveat.** An HLS playlist decides which hosts the process connects to: by default it
can point segments, keys and variants at any `http(s)` host and port the machine can reach, including loopback services
and cloud metadata addresses (e.g. `169.254.169.254`). Any segment, key or playlist URL can also answer with an HTTP
redirect; libavformat follows up to 8 redirects inside one open and FFmpeg 6.1 has no option to turn that off. The
responses are only parsed as playlists or media, never returned, but the requests are made. `--hls-same-host` restricts
every URL to the host:port of the top-level playlist (after its redirects; the authority is parsed exactly as
libavformat's `av_url_split` does, the host compared case-insensitively as text, without a trailing dot, default port
80 / 443 made explicit; only the exact lower-case schemes `http`, `https`, `crypto+http(s)` have an authority at all —
libavformat switches to TLS only for a lower-case `https`, so a redirect to `HTTPS://host/` is plain HTTP to port 80 and is
refused). The restriction fails closed: if the host:port of the top-level playlist cannot be determined (empty host, or a
host longer than 1023 bytes), the source is refused instead of running unrestricted. Independently of `--hls-same-host`,
the final URL of the top-level `http(s)` response after its redirects must start with exactly `http://` or `https://`
(libavformat follows a redirect to any scheme, e.g. `ftp://`, and still connects to its host:port with HTTP); otherwise
the source is refused. With `--hls-same-host` the playlist text is checked, every URL is checked again just before
it is opened, and after the open the final URL after redirects is compared before a single byte is read — a mismatch
ends the session and the redirected response is never used. What it cannot prevent: the redirected requests themselves
(libavformat follows them inside the open, before the guard sees the final location; their bodies are not used), host
names that point elsewhere than they read (DNS rebinding, another name or IP address of the same machine — only the text
of the host is compared), and for AES-128 segments (`crypto+http…`) the final URL is not visible, so there only the URL
before the open is checked. URL handling is modelled on libavformat built without `HAVE_DOS_PATHS`; Windows is not a
target platform. libavformat's `https` does not verify certificates by default
(`tls_verify=0`), so HTTPS protects against passive eavesdropping only. Use HLS sources you trust, or run ingest where it
cannot reach sensitive internal addresses. Two more limits inside libavformat: the number of HTTP response header lines is
not limited and `Icy-*` header values are concatenated without a cap (`http.c` n6.1:877-894; `Set-Cookie` values are kept
too), so a hostile server can make one open grow in memory until the open ends — bounded only by the open deadline
(`--input-wait-s`; for reconnects 30 s by default) and the 10 s read timeout per stalled read, not by size. URLs refused before
the open (wrong scheme, `http:` without `//`, another host with `--hls-same-host`) do not count toward the open-rate limit
(50 opens per second); their number per session is bounded by the playlist (at most 10 000 entries) and libavformat's reload
limit (`max_reload` 3, `hls.c` n6.1:1500-1503, 2568-2569).

**libavformat redirect-cache hang (FFmpeg ≤ 8.0) and the libav watchdog.** In FFmpeg 6.1 `http.c` a cached redirect jumps
back to the start of the open without counting redirects and without checking the interrupt callback (`http.c` n6.1:362-373),
and `301` / `308` answers are cached without expiry (425-431). A server that answers `301 Location: <the same URL>` (or
A → B → A) therefore makes that open spin at 100 % CPU without any I/O: the 10 s read timeout, `--input-wait-s` and the stop
request never fire, and FFmpeg 6.1 has no option to switch the cache off. Every `http(s)` open can hit it (top-level playlist,
variants, reloads, segments, keys, MPEG-TS over HTTP). FFmpeg 6.1 is affected (checked in its source); FFmpeg up to 8.0 has
the same code path, and FFmpeg master fixes it by applying `max_redirects` on the cached path as well (not re-checked here). A thread inside libavformat cannot be cancelled safely, and such a thread would also block
shutdown (the process joins every source thread). Ingest therefore watches every libavformat call of every source: if one
call (open, probe, read, close) stays inside libavformat longer than the bound ingest's interrupt callback enforces for it
(`--input-wait-s` while opening, the silence limit since the last packet while reading) plus `--libav-watchdog-s` (default
30 s), or libavformat stops calling the interrupt callback for `--libav-watchdog-s` (legitimate waits inside libavformat call it
repeatedly: every read attempt, the HLS reload wait every 100 ms), the process logs the source and the call and exits at once with code `7` — also while it is
shutting down. Nothing else is cleaned up: no summary is written and the flows stay as after a crash (the next start of the
same domain collects them). Run ingest under a supervisor that restarts it (e.g. systemd `Restart=on-failure`); all sources of
the process restart together. The margin covers blocking that the interrupt cannot cut short, mainly a DNS lookup (glibc: up
to 5 s × 2 attempts per name server); raise `--libav-watchdog-s` where name resolution is slower.

**Timing model in plain words.** Every packet is stamped once, when the demuxer reads it: *output time = source timestamp
+ one offset per source*. The offset is fixed by the first video packet so that it is shown `--latency-ms` after it arrived,
and it is the same for video and audio of that source. A frame therefore lands on the MXL index of its own timestamp, not
on the index of the moment it happened to arrive: bursty transports (HLS segments, a sender that stalls 300 ms) do not
move the picture against the sound. Packets wait compressed in RAM and are handed to the decoder `--decode-lead-ms` before
their time; the tick of index `i` takes exactly the frame of `i`. If the source clock runs faster or slower than ours,
the sound follows it sample by sample (a resampler changes the speed by at most `--max-comp-ppm`, no audible jump) and the
picture repeats or drops one whole frame (an interlaced field pair) whenever it is half a frame away from the sound — like
a broadcast frame synchroniser, so lip sync stays within one field. A timestamp jump (sender restart, loop splice, PTS wrap)
is handled per source: a wrap is invisible, a splice continues without a gap, a source that returns after an outage is
locked again. A frame that can no longer be decoded in time is not decoded: the decoder skips to the next key frame that
can still make it, so an overloaded decoder loses single GOPs instead of falling behind forever.

- One demuxer per source: video and audio come from the same libavformat context and the same clock.
- The format of every source is detected before any flow is created (`--format auto`): H.264 or HEVC, 8-bit 4:2:0,
  one of the formats in *Formats*. Field order is read from the first IDR of the stream (libavformat's guess from a
  mid-stream start is not reliable). 4:2:2, 10-bit, BFF and unknown sizes or rates are rejected at start with the format
  named (exit `2`).
- A resolution change in the middle of a stream (or a reconnect that comes back with another raster or scan) stops that
  source with a clear message and releases its flows; the other sources continue; the process ends with exit code `6`. A
  reconnect whose probe is incomplete (no video yet, unknown size) is retried with back-off; a different probed frame rate
  is only logged (frame indices follow the decoded frames' timestamps).
- Clock drift: the drift estimate comes from the windowed minima of the arrival margin; outside a dead band (default 50 ms)
  a slow feedback term (τ = 600 s, at most 90 % of `--max-comp-ppm`) steers the margin back, so a residual error does not
  accumulate into a relock. Video slips are applied in presentation order (one slip = exactly one frame on B-frame
  streams). HLS: no slope tracking (segment arrival phase is not the clock), margin feedback with a dead band of one
  segment.
- A relock (splice, new lock after an outage, latency change beyond the guard) is logged with its reason and size and
  counted (`relock_events`, live `relocks`); for a step of half a frame or more, everything of the old timeline not yet
  shown is dropped for video and audio at the same point, and audio switches with a 5 ms fade.
- Decoding starts at an IDR, a recovery-point SEI or an I slice (open GOP, periodic intra refresh); discarded packets are
  counted (`probe_scan_dropped_pkts`, `entry_dropped_pkts`), a stream without an entry point is reported. Field-coded
  H.264 second fields without timestamps are kept (frame time from the first field). If NVDEC cannot open or stops
  decoding, the source falls back to the software decoder (`decoder_fallbacks`); an audio stream that is missing or cannot
  be decoded after a reconnect is written as silence (`audio_absent`) and retried on the next session.
- Limit: the source clock is locked on the first video packet, not on the PCR. Audio headroom is therefore the latency minus
  the encoder's video VBV delay; for a stream with a VBV delay close to or above `--latency-ms` (default 1000) raise the
  latency accordingly.
- Live statistics: with `--stats-file` every line carries the per-source counters under `flows[n].fn` (`pkts_video`,
  `frames_decoded`, `late_frames`, `missing_frames`, `decode_errors`, `audio_silent_samples`, `input_reconnects`,
  `drift_slips`, `relocks`, `decoder_fallbacks`, …).
- Sources of different rate families (50 Hz and 59.94 Hz) need separate processes; 50 Hz and 25p mix in one process.
- Per-source counters (summary `flows[k]`): `fields_ok`, `frames_decoded`, `late_frames` (decoded after its time, dropped),
  `missing_frames`, `repeat_frames` / `slip_drop_frames` (frame synchroniser), `skip_to_key`, `drift_slips`, `relock_cont`
  (splice), `relock_fresh` (new lock), `wraps`, `audio_resyncs`, `audio_late_samples`, `input_reconnects`,
  `drift_ppm_abs_x1000`, `startup_frames` (before the first frame: the latency, not a loss).
- Edge contract: the outputs advertise the source commit bound (`ro + P/4 + G`) and `content-delay-ns` = the latency.
- Variant H (GPU with host flows) is not offered by `ingest`; use G or C.

### Chain timing (edge contract)

Every VRAMFlow output flow advertises in its flow definition when its grains are committed at the latest:

| Tag | Value (decimal ns, one string) |
|---|---|
| `urn:x-vramflow:tag:commit-bound-ns` | upper bound of the commit time of grain `i` after `T(i)` — a **bound**, not a measurement |
| `urn:x-vramflow:tag:content-delay-ns` | how far the content is behind its origin (only grows with `--input-offset`) |

A function reads the tags of its inputs when it starts and opens each tick at `T(i) + E + ro`, where `E` is the largest
input bound (one phase per process). Its own bound is `E + ro + Wr + G` (50 Hz: `+ 0.5 + 5 + 3` ms; sources `gen`,
`dec`, `gfxgen`: `ro + P/4 + G` = 8 ms). So each stage finds its inputs already committed, nobody is late by design,
and the latency adds up visibly — 50 Hz chain `gen → logo → logo → composite → enc`: 8, 16.5, 25, 33.5 and 42 ms.
The values are in the summary (`proc`: `expect_us`, `phase_us`, `commit_bound_us`, `content_delay_us`,
`inputs_tagged`, `inputs_untagged`, `ring_lag_us`, `ring_safe_us`) and in one stderr line at start.

| Option (readers) | Meaning |
|---|---|
| `--input-expect-ms X` | commit bound of an input **without** VRAMFlow tags (foreign writer); a tag wins. Default `0` for video, one grain period for audio (a real-time audio writer finishes the samples of grain `i` at `T(i) + P`) |
| `--input-offset K` | read grain `i + K` (`K ≤ 0`) — e.g. when a delay does not fit the ring; adds `−K · P` to `content-delay-ns` |
| `--in-flow-base B` | first input flow number (default `--flow-base`), for chains in one domain (`logo → logo`) |
| `--deadline-ms D` | last cut, counted from `T(i) + E` (default `3·P/4`) |
| `--allow-bad-clock` | start even if the kernel TAI offset is not 37 s (otherwise exit 2: indices would not match other processes) |

- Tags are read at start. An input that appears only later (started after its reader) counts as untagged; restart the
  reader to pick up its tag, or start chains top-down.
- The accumulated delay must fit the ring of the input: a reader refuses to start (exit 2) when `E + D − K·P` exceeds
  `(slots − 2) · P` of the domain history (default 200 ms = 10 grains at 50 Hz; safe up to 160 ms) and warns above half
  of it. Raise the history before the flows are created: `<domain>/options.json` with
  `{"urn:x-mxl:option:history_duration/v1.0": 500000000}`.
- `enc` reads video and audio of a bundle aligned: with different `content-delay-ns` the earlier member is read later by
  the difference (video in grains, audio in samples).
- Foreign MXL tools do not know the tags. `mxl-gst-sink` reads 40 ms behind (`--read-delay`); a flow whose advertised
  bound is above that (deep chains) needs a larger `--read-delay`.

### Interop with MXL's own tools (measured 2026-11-13)

Details: `docs/rezultatai/2026-11-13-saveika-su-mxl-irankiais.md`; tests `tests/functional/interop.py` (need Xvfb and `xwd`,
otherwise SKIP).

- `mxl-info` reads every VRAMFlow flow (host and cuda-linear, all formats, audio, v210a); the extra tags are accepted.
- `mxl-gst-sink` (host, progressive only) and `mxl-gst-cuda-sink` (cuda-linear `video/v210` only) play `gen` and `logo`
  outputs. During an upstream outage they skip INVALID grains and keep the last good picture (`--on-loss invalid`), show
  black with `--on-loss black`, freeze then black with `freeze`, and resume at once. Without committed gaps
  (`--on-loss leave` or a dead writer read directly) `mxl-gst-sink` spins on `TOO EARLY` (about a million log lines in
  3 s) and `mxl-gst-cuda-sink` exits when the flow is recreated — so commit every index.
- `mxl-gst-testsrc` (a foreign writer, progressive only, no VRAMFlow tags) → `logo --input-expect-ms X` works.

### Capacity and admission

A GPU function knows how much the GPU can serve only from a measurement: `PROFILE=knee scripts/linux/70-capacity.sh [dir]`
finds the knee per function (`tests/perf/capacity.py --n auto`, 3 repeats) and then runs
`tests/perf/capacity_file.py <campaign dir>`, which writes
`capacity-<GPU UUID>.json` (schema 1: NVDEC and NVENC Mpix/s with `bound`, the NVENC session count when the driver limit set
the knee, measured VRAM per flow for each function, GPU UUID and driver version). With `--capacity-file` each process
reserves what its flows need before it creates any output flow — demand in Mpix/s per engine (1080i50 / 1080p25 =
51.84 Mpix/s per flow), NVENC sessions and VRAM bytes — in a ledger next to the domain (`<domain>.vramflow/*.adm`, one
file per process held with `flock`, so a `kill -9` frees the reservation at once; read, decide and write run under one lock,
so two processes cannot overbook). Flows are admitted in order: with `enforce` the first flow that does not fit and every
flow after it are refused, their flows never appear in the domain, and the message names the engine and the numbers, e.g.
`flow 4: atmestas — nvdec: prašoma 51,84, užimta 207,36, leidžiama 229,50 Mpix/s`. If none is admitted the process exits
`5` at once; otherwise the admitted flows run and the process exits `5` at the end (the summary has `admission`: mode,
requested, admitted, refused, notes).

| Function | Checked |
|---|---|
| `ingest`, `dec` | NVDEC Mpix/s (not for `--decoder sw`) and VRAM: measured `functions.<ingest\|dec>.vram_per_flow_bytes` (+ the per-process offset), else an estimate (NV12 surfaces + v210 output ring) |
| `enc --encoder nvenc` | NVENC Mpix/s, NVENC sessions (1 per flow) and VRAM if `functions.enc-nvenc-<tune>` has it |
| `enc --encoder x264`, `gen`, `gfxgen`, `logo`, `composite` | VRAM only, if the capacity file has `functions.<tag>` (`enc-x264-<preset>`, `composite-k<K>`, …); otherwise not checked (one line says so) |

- A capacity value with `bound: lower_bound` (the real capacity is larger) is still used as the limit, with a warning.
- A file for another GPU (UUID differs) is not used. A different driver version is a warning with both versions; the
  running version comes from NVML.
- The default is `auto` (plan decision K13): with a usable capacity file it behaves as `enforce`, so streams already on air
  are protected as soon as the GPU has been measured; without one it behaves as `warn` and starts with one warning line.
  An explicit `enforce` without a usable capacity file exits `1` (`talpa neišmatuota, paleiskite 70-capacity …`). The
  capacity harness itself runs every function with `--admission off`, so a re-measurement is not capped by the old file.
- Limits: the ledger only sees VRAMFlow processes in the same domain on the same GPU (not other GPU users, not other
  domains); VRAM is checked against the total GPU memory × headroom, not against the free memory at start.

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
- **The lease folder is only as open as the domain.** `/dev/shm` is world-writable, so another local user could create
  `<domain>.vramflow` first or plant a link or a FIFO in it. `vramflow-func` therefore opens the folder without following
  links and refuses it (exit code `2`) when it is not a folder, when its owner is neither this process nor the owner of the
  domain folder, or when a foreign folder grants more than the domain folder does. Files in it are created with
  `O_EXCL | O_NOFOLLOW`, and other processes' files are opened without following links and only if they are regular files.
  The folder gets the permissions of the domain folder (files: the same without the execute bits), independent of `umask`:
  a domain shared with a group (`0770`) gives a shared lease folder, a private domain (`0700`) a private one. A domain that
  is writable by everyone gets no protection from this — do not use one on a shared machine.
- **One domain, one lease folder.** The folder name is built from the canonical domain path (links resolved, `/./` and a
  trailing `/` removed), so two processes that reach the same domain by different paths share one lease folder and one
  admission ledger.
- **Keep the domain path short: at most 48 characters for GPU (`cuda-linear`) flows.** A reader of a GPU flow connects
  to the Unix socket `<domain>/<flow-id>.mxl-flow/payload.sock`, and a socket path cannot be longer than 107 characters.
  With a longer domain path the writer still creates the flow, but no reader in another process can attach to it.
  `vramflow-func` therefore refuses such a path at start (exit code `1`, the message gives the length and the limit), and
  `make_domain` shortens a long tag. Flows in RAM (`--storage host`) have no such limit.
- While a reader waits, its `laukiu flow …` line also says why a flow that exists was not accepted (another format, the
  reader could not open it), so a wait never looks like a missing flow when it is something else.

### Exit code and summary

Exit codes of the functions on the shared tick engine (`gen`, `logo`, `composite`, `enc`, `dec`, `ingest`). Strict — `--strict-exit on`,
or `auto` with `--duration S > 0` (test and measurement runs); lenient — `off`, or `auto` in service mode (`--duration 0`):

| Situation | strict | lenient |
|---|---:|---:|
| Clean end (duration over or `SIGINT` / `SIGTERM`), nothing lost | 0 | 0 |
| Something was lost (gap, late, skipped index, lost input grain, single output error) | 4 | 0 |
| One unit failed (3 open / commit errors in a row on its output): only it stops, its output flow is released, the other units keep running | 2 | 0 |
| All units failed | 2 | 2 |
| Wrong options or configuration | 1 | 1 |
| Setup or run-time failure (CUDA, MXL, an output flow owned by another run) | 2 | 2 |
| Inputs did not appear within `--input-wait-ms` | 4 | 4 |
| An input came back with another format (`--on-format-change exit`) | 6 | 6 |
| Stopped while still waiting for inputs | 0 | 0 |
| `ingest` only: a libavformat call is stuck (see *libavformat redirect-cache hang*); immediate exit, no summary | 7 | 7 |
| Admission control refused flows (`--admission enforce`, or the default `auto` when a capacity file is found): none admitted — at once; some admitted — they run, exit at the end (an error code above wins) | 5 | 5 |
| An explicit `--admission enforce` without a usable capacity file (the default `auto` only warns) | 1 | 1 |

`verify` exits `3` on verification errors. In service mode losses are reported in the summary only: `any_loss` and
`strict_exit` (top level) and the counters; failed units — `failed_units` (proc) and `unit_failed` (per flow). A second
`SIGINT` / `SIGTERM` skips draining (`--drain-ms`) and exits at once, after releasing the output flows. Every run writes
`summary-<command>.json` with counters (fields written, read, lost, late, invalid, verification errors), a latency histogram,
the run ID and the software / GPU versions.

Counters of the functions on the shared tick engine (`gen`, `logo`, `composite`, `enc`, `dec`) that need a word:

| Counter | Meaning |
|---|---|
| `output_errors` (per flow) | Opening or committing an output grain failed; the index has no grain. Counted as a loss (exit `4` when strict); 3 in a row on one output fail that unit alone (`unit_failed`, output flow released; exit `2` when strict or when all units failed). An open refused with `MXL_ERR_INVALID_ARG` is "already written" (clock stepped back) only for an index this output already wrote; otherwise it is an output error |
| `clock_backward` (process) | The output index was already written — the clock went back (MXL refuses to write an index twice). Not an error, not a loss |
| `late_fields` → `missing_fields` | An input that stopped delivering (alive but stuck, e.g. `SIGSTOP`) costs 3 `late_fields` (the engine waits for it up to the deadline 3 times), after that it is *stalled* and every further index is `missing_fields` (no wait at all) until it delivers again |
| `logo`: `gap_late_commits`, `gap_late_commit_min_us` / `_max_us` | Gaps committed at the deadline because an input was late (not because it was gone): how many, and how late after `T(i)` the gap grain was committed: no later than `D` plus the wake-up of the blocking wait (measured on WSL2 over 2072 gaps at `D` = 15 ms: p50 15.3, p99 16.3, max 16.6 ms; the extra time is the futex wake-up, p99 1.2 ms, not processing, p99 0.05 ms) |
| `read_errors` (per flow) | Replaces the old `read_status_<code>` keys: any unexpected read status of an input. The status code itself is no longer in the summary |
| `reopen_mismatch` (`composite`) | The input that came back with another format (with `--on-format-change exit`, exit `6`) |
| `composite`: `layer_absent`, `layer_absent_<k>` | Output indices composed without a layer (INVALID, not attached, stalled, gone, late or rejected): the layer was transparent and the base passed through. Not a loss |
| `lateness_max_us` (per flow) | Largest commit lateness after `T(i)` of this flow after warm-up, µs (`enc`: frame queued). The `out_lateness` histogram covers the whole tick |
| hist `proc_ms` / `wait_ms` / `held_ms` | `proc_ms` — processing only (kernels, sync, commit; summed over the phases of one index); `wait_ms` — tick open → first processing; `held_ms` — only for an index processed in more than one phase: time spent waiting for late inputs between phases |
| `enc`: `audio_underrun_frames` | The frame's audio was not there when the frame was encoded (silence is encoded instead). Audio is a *wanted* input, expected at `T + P`: the engine waits for it at most until the frame's deadline `D` (default `3·P/2`) and the first cut `a(i)` of the other streams never waits for it, so one stream with late or no audio does not delay the others |
| `dec`: `clock_overrun_frames`, `clock_overflow_drops` | Frames dropped because a backward clock step of `S` made the tick wait: at most `ceil(S / frame)` frames, once, at the first frame tick after the step. Everything else above the buffer limit is `overrun_frames` (exit `4`)

---

## 7. Tests and measurements

```bash
bash scripts/linux/run-all.sh --soak skip      # everything: probes, upstream tests, unit / GPU / functional tests, report
bash scripts/linux/61-make-test-clip.sh        # synthetic 1080i50 H.264 clip for the dec / chain tests
bash scripts/linux/62-make-av-clip.sh          # synthetic A/V clips with sync markers for the ingest tests

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

- Full chain at N=12: CUDA cores ~10 % busy, NVENC 63 %, NVDEC 51 %, ~40 W. Neither engine is saturated; the 2026-11-17
  rerun below fails N=12 on late frames, cause not yet found (an earlier version of this README wrongly named NVENC).

**Capacity knee with repeats (2026-11-17, plan P8).** Same machine, driver 596.49, G only, 30 s per level, a level
counts only if **all 3 repeats** pass; found by `capacity.py --n auto` (doubling, then bisection). VRAM per flow is the
least-squares slope of ΔVRAM(N) over passing levels.

| Function | Knee N | What set it | VRAM per flow (R²) | Engines at the knee |
|---|---|---|---|---|
| Logo burn-in | ≥ 64 | search limit | 56.0 MB (1.000) | SM ~39 % |
| NVENC low-latency encode | 12 | **driver NVENC session limit** | 126.2 MB (1.000) | NVENC 64 % |
| H.264 decode (NVDEC) | 22 | NVDEC throughput | 182.2 MB (0.997) | NVDEC ~91 % |
| Full chain | 11 | late frames at 12, cause not found | 284.3 MB (0.999) | NVENC 59 %, NVDEC 46 % |
| Compositor, K = 2 | ≥ 32 | search limit | 119.1 MB (1.000) | SM ~37 %, 1.1 GHz |

- The GeForce driver allows 12 concurrent NVENC sessions; the 13th fails with `OpenEncodeSessionEx failed: incompatible
  client key` and takes the whole `enc` process down. `capacity.py` tags such levels `driver_limit = nvenc_sessions`.
  Admission control has to treat encoder sessions as a resource of their own.
- The compositor at K = 2 went from N\* 2 (table below, GPU clock 210–690 MHz, no repeats) to ≥ 32 with `keepwarm`
  (550–1112 MHz): the earlier G failures were the GPU clock, not the code.

Details: [`docs/rezultatai/2026-11-17-p8-talpa.md`](docs/rezultatai/2026-11-17-p8-talpa.md).
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

Ingest (`vramflow-func ingest`), synthetic H.264 + AAC MPEG-TS over loopback UDP (paced by PCR), 30 s per level, every source
checked by `verify --mode ingest` (≥ 99.8 % grains, 0 late frames, every sync pulse within one field / frame):

| | 1080i50 | 1080p25 |
|---|---|---|
| NVDEC, GPU flows (G): N\* | ≥ 16 (ladder end) | ≥ 16 (ladder end) |
| Software, CPU flows (C): N\* | ≥ 8 (ladder end) | ≥ 8 (ladder end) |
| VRAM per source (G, slope 1 → 16) | 135 MiB | 135 MiB |
| CPU cores per source, G / C | 0.10 / 0.26 | 0.10 / 0.25 |
| A/V offset, every pulse (all levels) | max 0.52 ms | max 0.0 ms |
| Commit after T(i), p99 | ≤ 2.8 ms (G), ≤ 4.8 ms (C) | ≤ 2.5 ms (G), ≤ 6.1 ms (C) |

Functional tests (`tests/functional/ingest.py`, synthetic only): A/V 0.0 ms at every pulse for 1080i50 and 1080p25 on NVDEC
and software, over UDP, RTP, SRT and HLS; 2 s bursts for 120 s without drift (Δ A/V 0.0 ms, no slip, no relock); a source
0.5 % fast for 120 s: 13 whole-frame slips, A/V ≤ 18.8 ms, no audible break; PTS wrap and loop splices without index gaps;
progressive chroma bit-exact against FFmpeg bilinear. Details and what is not verified:
[`docs/rezultatai/2026-11-14-ingest.md`](docs/rezultatai/2026-11-14-ingest.md).

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
- `ingest`: tested with synthetic sources over loopback only (no real broadcast source, no lossy network); HEVC only with one
  synthetic 1080p25 clip; UHD and the 59.94 family not run; variant H not offered; sources of different rate families need
  separate processes.

## 12. License and credit

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). You may use, change, distribute and sell this work,
also in closed products. When you redistribute it or a derivative, keep the NOTICE file: it names the author. If this
work helps a product, a standard or a paper, please credit **"VRAMFlow by Deimantas Kiršinas"** with a link to this
repository.

Binaries linked with an FFmpeg build that includes libx264 fall under the GPL when distributed (see NOTICE).

Developed with the help of AI coding assistants (Claude). The author reviewed the design, ran every measurement on real
hardware and is responsible for the content.

MXL is a project of the EBU and the Linux Foundation; this repository is independent and not endorsed by them.
