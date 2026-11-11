# Make VRAMFlow production-grade — lessons from a real-world field test

Source: author's brief, 2026-11-11. The field test ran on an RTX 3090 workstation (Ubuntu 26.04, sm_86) with live
public broadcast sources; the reference prototype from that test lives outside this repository. The plan derived from
this brief must stay machine-independent: it improves VRAMFlow in general, not one machine's setup.

Until now VRAMFlow was tested only with synthetic 1080i50 test patterns. It was then driven with real live broadcast
sources (H.264 HLS, progressive 1080p25, separate audio, up to 30–32 parallel streams on one RTX 3090). Everything
below is what that exposed.

## 1. What we learned

### Timing and sync
1. **Grains must be placed by media time, not arrival time.** `dec` plays frames out in arrival order through a
   jitter queue whose depth floats (0.5–1.5 s). Two flows from one source (video + audio) then drift apart even though
   MXL itself aligns them perfectly by index. Rule: `MXL index = source PTS + one offset per source`, offset fixed once
   and shared by every flow of that source. MXL only guarantees alignment of *indices*, not of *content*.
2. **One demuxer per source.** Audio and video of one programme must come from the same demux/clock, otherwise
   bursty transports (segmented, separate playlists) make them arrive seconds apart.
3. **Decoders need lead time.** Hardware decoders (NVDEC via h264_cuvid) emit frames 4–8 frames after the packet;
   packets must be fed ~1 s ahead of presentation (by DTS), frames committed at their presentation time.
4. **Overload must not collapse.** When the decoder is short of throughput, every stream falls behind; late packets
   are still decoded and thrown away, so the engine never catches up and *all* outputs go dark. Deadline-aware
   behaviour fixes it: if a frame cannot make its deadline, skip to the next keyframe that can, flush, resume.

### Formats
5. **1080i50 is hard-coded everywhere** (`kHd1080i`, grain rate 50/1 checks, field = grain). Real sources are often
   progressive (1080p25/50, 720p50, UHD). Progressive content forced through the interlaced path is converted wrongly:
   `split_nv12` takes field chroma from alternate 4:2:0 rows, which for a progressive source halves and shifts chroma
   (measured: U/V 50/54 dB vs FFmpeg; a progressive-aware kernel gives 62/66 dB, luma identical).
6. Chroma siting and progressive/interlaced handling must be explicit per flow, not assumed.

### Writer/reader contract and interoperability
7. Host writers must set `validSlices = totalSlices` before commit; otherwise reference readers (`mxl-gst-sink`,
   reading all slices) wait forever (`TOO_EARLY`). The README example omitted it.
8. A missing index in a progressive flow stalls `mxl-gst-sink` (it retries one index forever). Writers must commit
   every index (repeat/black/invalid-flagged) or readers must be documented and tested to step over gaps.
9. VRAMFlow was never tested against MXL's own tools as consumers. Reference tools: `mxl-gst-sink` (host, progressive
   only), `mxl-gst-cuda-sink` (CUDA v210 video only), `mxl-info`.

### Capacity and resources
10. **The limit is the fixed-function engine, not CUDA.** RTX 3090: ~800 fps 1080p H.264 NVDEC ≈ 30 streams stable,
    32 at the edge, ≥ 48 collapse — while CUDA SM stays at 10–13 %, CPU ~20 %. ~125 MB VRAM per decoded stream
    (decoder surfaces + two 10-grain v210 flows). Capacity planning must measure per engine (NVDEC, NVENC, SM,
    copy engines, VRAM per flow), not "GPU %".
11. There is no admission control: nothing stops a user from starting more streams than the engine can serve.

### Operations
12. Functions wait a fixed 15 s for input flows; real startup (network sources, many decoders) takes longer → silent
    exit. Start order must not matter; flows may appear, disappear and come back.
13. Two runs sharing one domain path: one run's cleanup deleted the other's flows. Domains need ownership/isolation.
14. Statistics are only written at exit (`summary-*.json`); during a run there is no live view of lateness, drops,
    per-engine load. Field debugging needed ad-hoc tools.
15. Display/monitoring of VRAM flows always went through host copies (20 MB/frame D2H for a 3440×1440 canvas);
    there is no zero-copy GPU display path.

### Build and quality
16. `CMAKE_CUDA_ARCHITECTURES` default sat after `project()`, so builds silently targeted sm_75 + JIT instead of the
    real GPU (`native`, set before `project()`, fixes it).
17. Compilers were hard-coded (clang-19); scripts live in `scripts/wsl/` although they also serve native Linux
    (`CC_BIN`/`CXX_BIN` overrides fix the first part).
18. Catch2 test `gfx_writer: progresyvus 382x40 …` fails 15–17 of 20 runs on unmodified main (flaky, root cause
    unknown).
19. All published performance numbers came from a laptop GPU with sm_75 JIT code — re-measure natively.

## 2. What to build, in priority order

Each item: a feature branch, tests for every behaviour, before/after numbers where performance is involved.

- **P1 — Timing model in the library.** Shared "source clock" helper: PTS (wrap/discontinuity handling) → MXL index
  with one offset per source, re-lock on discontinuity, used by every ingest-type function. Unit tests: wrap,
  discontinuity, two flows of one source stay index-aligned, clock step.
- **P2 — Format-agnostic flows.** Replace fixed `kHd1080i` + 50/1 with a raster/format description per flow
  (progressive frame grains and interlaced field grains, 25/50/59.94, 720p/1080/UHD). Add `split_nv12_progressive`
  (CUDA + CPU reference); choose by stream field flags. Tests: CPU-reference bit-exactness, chroma PSNR vs FFmpeg
  ≥ 60 dB, every function accepts at least 1080i50, 1080p25, 1080p50.
- **P3 — Real ingest function with audio** (`vramflow-func ingest`): TS over UDP/SRT/RTP and HLS in, one demuxer,
  NVDEC (or software) video + audio decode, PTS-locked video and audio flows. Keep `dec` for the test clip. Tests: A/V
  offset at the output ≤ 1 field with a synthetic clip carrying sync markers; bursty input causes no drift.
- **P4 — Deadline-aware overload behaviour + admission control.** Skip-to-next-keyframe when a frame cannot meet its
  deadline; capacity estimation per engine (measured headroom) and refusal/degradation beyond it. Test: start 2×
  capacity — first N keep 0 late frames, the rest are refused or reported, no global collapse.
- **P5 — Writer/reader contract and interop tests.** Helpers that always set `validSlices`, never leave index gaps,
  documented reader gap handling. Functional tests with `mxl-gst-sink`, `mxl-gst-cuda-sink`, `mxl-info` as consumers.
  README rules.
- **P6 — Robust start/stop.** Configurable or unlimited input wait, start in any order, re-attach when a flow
  disappears and returns, per-run domain ownership. Tests: consumers before producers, kill/restart a producer.
- **P7 — Live observability.** Periodic stats while running (JSON lines or HTTP/Prometheus; NMOS IS-12/BCP-008
  later): per flow ok/late/lost, latency p50/p99, per engine NVDEC/NVENC/SM/VRAM via NVML. Keep exit summaries.
- **P8 — Capacity harness v2.** Per-engine utilisation and VRAM per flow, automatic knee, decode comparisons
  (h264_cuvid vs FFmpeg nvdec hwaccel vs Video Codec SDK, surface counts, video clock). Re-measure README results
  natively on sm_86/sm_89 builds.
- **P9 — GPU monitoring/display path.** VRAM-native multiviewer (scaled grid, per-flow alarms: black, freeze,
  silence, audio levels), zero-copy presentation (CUDA–GL/Vulkan interop), optional MXL output flow of the canvas.
- **P10 — Quality debt.** Root cause of the flaky `gfx_writer 382x40` test (never skip or loosen it without asking);
  platform-neutral scripts instead of `scripts/wsl`; build fixes (CUDA architectures, compiler overrides).

## 3. Constraints

- Conventional Commits in English with trailer `Assisted-by: <tool> (<model>)`; small commits; draft PRs
  ≤ ~400 changed lines each. (Branch/push policy for this repository: decided by the author per change.)
- Ask first: new/changed dependencies (exact pin, ≥ 30 days old), suppressing warnings/tests, CI/CD, deleting or
  rewriting tests, anything destructive, stopping other GPU jobs on the machine.
- New user-visible functionality (P3, P4, P9 at least) needs a user story first: Lithuanian, Given/When/Then,
  ≥ 1 positive + ≥ 1 negative criterion, approved by the author.
- Synthetic or public test media only; no production systems, no personal data.
- Inside MXL everything stays uncompressed (v210/v210a 10-bit, float32 audio); only inputs/outputs may be compressed.
- Code comments in Lithuanian like the rest of the repo; README/ADR in English. 0 compiler warnings.

## 4. Done means

- 0 warnings, all tests pass (flaky one reported unless P10 fixed it), new tests for every behaviour change.
- For every performance change: before/after table (per-engine utilisation, fps/streams, VRAM per flow, CPU,
  latency p50/p99), measured natively.
- Docs updated (README rules, format support, timing model, capacity numbers).
- Per item: PR text; report Done / Verified (numbers) / Caveats / Next.
