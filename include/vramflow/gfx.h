/* SPDX-FileCopyrightText: 2026 Deimantas Kiršinas */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VRAMFLOW_GFX_H
#define VRAMFLOW_GFX_H
/*
 * libvramflow-gfx — public C API. This comment is the contract.
 *
 * The library turns 8-bit RGBA/BGRA frames into a standard MXL `video/v210a` flow (v210 fill + 10-bit key, straight
 * alpha, BT.709 video range) in host memory or in GPU memory (MXL `cuda-linear` payload backend). The output is
 * bit-exact with the repository's independent reference (src/gfx/gfx_ref.*) for every supported input.
 *
 * VERSIONING AND ABI
 *  - Compile time: VFGFX_VERSION_MAJOR / _MINOR / _PATCH (and VFGFX_VERSION, VFGFX_VERSION_STRING) describe the header.
 *    Run time: vfgfx_version() / vfgfx_version_string() describe the loaded library.
 *  - SONAME policy. The SONAME carries an ABI epoch, VFGFX_ABI_VERSION, independent of the release version:
 *    libvramflow-gfx.so.1 (release 0.1 used .so.0). The epoch changes ONLY on an incompatible change (a removed or
 *    changed function, a changed meaning or offset of an existing field). Additive changes keep it: new functions, new
 *    fields at the end of a struct (struct_size tells the library what the caller has), new flags and new enum values
 *    (an older library rejects unknown ones with VFGFX_ERR_UNSUPPORTED). So a program built against any release of
 *    epoch 1 runs with any later release of epoch 1; vfgfx_version() tells which release is loaded.
 *  - Symbol versions: every exported function carries the version node of the release that added it; the epoch-1
 *    baseline is VFGFX_1 (`nm -D --with-symbol-versions` shows vfgfx_*@@VFGFX_1). Functions added later go into new
 *    nodes (VFGFX_1.1, …), so a program that needs a newer function fails at load time with a clear loader message
 *    instead of at the call.
 *  - Only `vfgfx_*` symbols are exported; the library carries its own private, statically linked CUDA runtime.
 *  - Extensible structs. Every struct that crosses the API starts with `uint32_t struct_size` (the caller sets it to
 *    sizeof the struct as the caller's header declares it; the *_INIT macros and *_init() functions do this) followed by
 *    `uint32_t flags`, and ends with reserved space. For input structs the library
 *      - rejects struct_size below the minimum it needs or above 4096 (VFGFX_ERR_STRUCT_SIZE);
 *      - accepts a larger struct from a newer header only if every byte it does not know is zero, otherwise
 *        VFGFX_ERR_UNSUPPORTED (a field it cannot honour is never silently ignored);
 *      - rejects unknown flag bits and non-zero reserved fields (VFGFX_ERR_UNSUPPORTED).
 *    For the output struct (vfgfx_stats) the library fills the fields it knows within the caller's struct_size and
 *    zeroes the rest; struct_size and flags are not modified.
 *
 * CHANGES (observable behaviour; the ABI epoch stays 1, SONAME libvramflow-gfx.so.1, symbols unchanged)
 *  0.3.0
 *   - A paced writer commits EVERY index from vfgfx_start_pacing on: until the first frame is current each grain is an
 *     INVALID grain (see GAPS). Observable for a 0.2 client: stats.last_committed_index is no longer
 *     VFGFX_INDEX_NONE before the first frame, and vfgfx_write after vfgfx_stop_pacing needs an index above the last
 *     INVALID grain the thread committed (not only above the last frame). stats.no_frame counts those grains.
 *   - INVALID grains carry a payload: transparent black (fill Y 64 / Cb Cr 512, key 0), written once per ring slot,
 *     so a reader that ignores the flag never sees stale or random slot memory. Content grains are unchanged.
 *   - New field vfgfx_stats.last_frame_index (in former reserved space; sizeof(vfgfx_stats) unchanged): the last index
 *     committed WITH a frame, VFGFX_INDEX_NONE before the first one; last_committed_index also counts INVALID grains.
 *  0.2.0 — ABI epoch 1 (SONAME .so.1, symbol node VFGFX_1), extensible structs, vfgfx_submit_at, BGRA, CUDA frames.
 *
 * WRITER LIFETIME
 *  vfgfx_writer_create() creates the MXL flow (from a flow definition JSON, see vfgfx_flow_def()) and its writer on the
 *  given mxlInstance; vfgfx_writer_destroy() stops pacing, releases the writer and frees everything. The instance must
 *  outlive the writer (destroy writers before mxlDestroyInstance). Creating a flow that already exists in the domain
 *  fails with VFGFX_ERR_STATE.
 *
 * TWO MODES (mutually exclusive at any moment)
 *  - Manual: vfgfx_write(w, index, frame) converts and commits grain `index` now. Interlaced flows: grain = field of
 *    parity index % 2 — even index = top field = frame rows 0, 2, 4…; progressive flows: grain = frame, parity ignored.
 *  - Paced: vfgfx_start_pacing(w, offset_ns) starts a library thread that commits one grain per grain period of the
 *    flow (TAI clock; interlaced: 2 x frame rate). The content of grain i is the frame that is current at i:
 *      * vfgfx_submit(w, frame) means "this from now on": `frame` becomes the current content from the next grain the
 *        thread processes and every scheduled (vfgfx_submit_at) frame is cancelled;
 *      * vfgfx_submit_at(w, first_index, frame) makes `frame` current from grain `first_index` on (frame accurate).
 *    A paced writer commits EVERY index (gap policy "invalid", see GAPS below): until the first frame exists each grain
 *    is committed with MXL_GRAIN_FLAG_INVALID (no content is invented). While pacing runs, the thread is the only user
 *    of the MXL writer and vfgfx_write fails with VFGFX_ERR_STATE. vfgfx_submit / vfgfx_submit_at are also allowed
 *    before vfgfx_start_pacing (the frames are then ready when pacing starts, and no INVALID grain is written).
 *
 * GAPS (paced mode; behaviour change in 0.3: before, nothing was committed until the first frame)
 *  - Index without a frame (before the first vfgfx_submit / vfgfx_submit_at becomes current): the thread opens the grain,
 *    sets MXL_GRAIN_FLAG_INVALID and validSlices = totalSlices and commits it at the index's commit time. Its payload is
 *    transparent black (fill Y 64 / Cb Cr 512, key 0), written into a ring slot only once while that slot holds no frame
 *    (cost: one grain copy per slot on the first lap, nothing after). A reader blocked on that index wakes up at once
 *    with MXL_STATUS_OK + the INVALID flag (it does not wait for the next commit). Counted in stats.no_frame; NOT in
 *    stats.committed / vfgfx_committed (content grains only); stats.last_committed_index includes them (a later
 *    vfgfx_write must use a higher index), stats.last_frame_index does not.
 *  - Once a frame is current, every grain repeats it until another frame replaces it — static graphics, not a gap.
 *  - Indices skipped because the thread woke up late (stats.late_skips) are not filled: MXL marks them invalid itself
 *    when the next index is opened. The policy is fixed ("invalid"); freeze / black are policies of the media
 *    functions (vramflow-func --on-loss), not of this library.
 *
 * GRAIN INDICES AND TIMING (paced mode)
 *  - Index i is committed at its commit time T(i) + offset_ns, where T(i) = vfgfx_index_time_ns(w, i) is the TAI time of
 *    grain i (mxlIndexToTimestamp at the flow grain rate). offset_ns > 0 commits later, < 0 earlier (ahead of time).
 *  - Guaranteed: indices are committed in strictly increasing order, never twice (also across stop/start, and never an
 *    index at or below one already written with vfgfx_write), and a grain is never committed before its commit time.
 *    The first index after vfgfx_start_pacing is the smallest index whose commit time has not passed (and that is
 *    above every committed index); no index whose commit time is still ahead is skipped.
 *  - Not guaranteed: every index. If the thread wakes up late, the index it was waiting for is still committed (late),
 *    then the indices whose time has already passed are skipped (stats.late_skips) — no catch-up burst. If the TAI
 *    clock steps backwards (detected against CLOCK_MONOTONIC_RAW, steps > 5 ms), the thread waits until time catches
 *    up (stats.clock_backward). A distant commit time caused by offset_ns or by vfgfx_write ahead is not a clock step.
 *  - stats.next_pacing_index: the index the pacing thread commits (or skips) next; when not pacing, the index pacing
 *    would start with if started now with the last offset_ns (0 before the first start). No fixed relation to
 *    vfgfx_current_index is promised; schedule by time with vfgfx_index_time_ns when that matters.
 *  - vfgfx_submit_at(w, first_index, frame):
 *      * accepted when stats.next_pacing_index <= first_index <= vfgfx_current_index(w) + VFGFX_SUBMIT_AT_MAX_LEAD_S x
 *        grain rate; earlier -> VFGFX_ERR_TOO_LATE, later -> VFGFX_ERR_TOO_EARLY (nothing is queued in either case);
 *        first_index >= next_pacing_index + 2 leaves a margin of at least one grain period;
 *      * interlaced flows: first_index must be even (a frame starts on its top field), else VFGFX_ERR_INVALID_ARG;
 *      * up to stats.queue_capacity (at least VFGFX_SUBMIT_AT_MIN_QUEUE) FUTURE frames may be pending; one more fails
 *        with VFGFX_ERR_QUEUE_FULL. A frame for an index that is already pending replaces the pending one
 *        (stats.scheduled_replaced);
 *      * a pending frame becomes current exactly at grain first_index if the pacing thread processes that grain. It is
 *        never shown before first_index;
 *      * past entries: as soon as first_index < next_pacing_index (pacing stopped over it, the thread skipped it after
 *        a late wake-up, or it was queued before vfgfx_start_pacing and its time passed), the entry no longer occupies a
 *        queue slot: of all such entries only the newest (highest first_index) survives, as the current content
 *        (stats.scheduled_late); the older ones count as replaced;
 *      * the queue survives vfgfx_stop_pacing / vfgfx_start_pacing (past entries are collapsed as above at start);
 *      * vfgfx_submit cancels all pending frames; vfgfx_cancel_scheduled cancels them and keeps the current content
 *        (both count the cancelled frames in stats.scheduled_replaced);
 *      * memory: each pending frame holds one prepared frame in the flow's memory (GPU for cuda-linear): the grain size
 *        x 2 fields for interlaced flows, x 1 for progressive (1080i or 1080p: 8.3 MB, 2160p: 33.2 MB). Buffers
 *        are allocated on demand, one per frame actually pending (plus 2 always), and kept until vfgfx_writer_destroy.
 *  - vfgfx_current_index(w): MXL's current index at the flow grain rate, mxlTimestampToIndex(rate, now) — equal to
 *    mxlGetCurrentIndex for that rate. MXL rounds to the NEAREST index: the result is the index i with
 *    T(i) - P/2 <= now < T(i) + P/2 (P = grain period), so T(i) may still be up to half a period in the future.
 *    Independent of offset_ns.
 *
 * FRAMES (vfgfx_frame)
 *  - pixel_format: VFGFX_PIXEL_RGBA8 (bytes R, G, B, A) or VFGFX_PIXEL_BGRA8 (bytes B, G, R, A — Chromium/browsers,
 *    GStreamer BGRA). Both give identical output for the same picture. Other values -> VFGFX_ERR_UNSUPPORTED.
 *  - width/height must equal the flow raster (VFGFX_ERR_GEOMETRY); stride >= width x 4 bytes; CUDA frames: pointer and
 *    stride multiples of 4 bytes; host frames into a cuda-linear flow: stride multiple of 4 bytes.
 *  - alpha: VFGFX_ALPHA_STRAIGHT or VFGFX_ALPHA_PREMULTIPLIED; mem: VFGFX_MEM_HOST or VFGFX_MEM_CUDA (device pointer
 *    on the device the writer uses). Other values (e.g. from a newer header) -> VFGFX_ERR_UNSUPPORTED.
 *  - Ownership: calls are synchronous. When vfgfx_write / vfgfx_submit / vfgfx_submit_at return, the library has
 *    finished reading the pixels; the buffer may be reused or freed immediately. The library keeps no pointer to it.
 *  - Cost: matching pairs (host frame -> host flow, CUDA frame -> cuda-linear flow) are one conversion straight into
 *    the destination. Crossed pairs add a copy (host -> cuda-linear: H2D of the frame; CUDA -> host flow: D2H). In paced
 *    mode the conversion is paid once per submit, not per grain (the thread only copies the prepared field).
 *
 * CUDA FRAMES: ORDERING WITH THE CALLER'S GPU WORK
 *  The library has a private CUDA runtime, so it cannot see the caller's streams implicitly. Before reading a CUDA frame
 *  it orders itself after the caller's work in one of three ways:
 *   1. cuda_stream == NULL and no VFGFX_FRAME_CALLER_SYNCED (default): cudaDeviceSynchronize() on the writer's device
 *      (primary context) — waits for ALL GPU work in that context, including unrelated streams. Always safe; costly when
 *      the process keeps the GPU busy (a 100 ms kernel on another stream delays the call by up to 100 ms).
 *   2. cuda_stream = the caller's cudaStream_t (as void*; cudaStreamLegacy / cudaStreamPerThread handles are accepted):
 *      the library records an event on that stream and makes its own stream wait for it, so it reads the frame after
 *      exactly the work queued on that stream before the call; unrelated GPU work is not waited for and the caller's
 *      stream is not blocked. The stream must belong to the writer's device (primary context).
 *   3. flags |= VFGFX_FRAME_CALLER_SYNCED: the caller guarantees the frame is complete (e.g. cudaStreamSynchronize before
 *      the call); the library does not wait.
 *  Measured cost of vfgfx_write (1080i50 cuda-linear flow, CUDA frame, one field; RTX 4070 Laptop, WSL2; tool
 *  tests/gpu/gfx_so_cuda_check.cu --bench against the shipped .so): idle GPU 0.74 / 0.74 / 0.62 ms for paths 1 / 2 / 3;
 *  with a 100 ms kernel running on another stream 99.6 / 0.55 / 0.57 ms.
 *  cuda_stream together with VFGFX_FRAME_CALLER_SYNCED, or cuda_stream on a host frame -> VFGFX_ERR_INVALID_ARG.
 *  Note: a "synchronous" cudaMemcpy from pageable host memory may return before the DMA has finished; use 1 or 2 then.
 *  Devices: a cuda-linear flow uses the device named by "deviceUuid" in writer_options_json (REQUIRED on multi-GPU
 *  machines: the caller's cudaSetDevice is invisible to the private runtime; without it device 0 is used). A host
 *  flow downloads CUDA frames on the "deviceUuid" device (or device 0). UUID format "GPU-xxxxxxxx-…", any case.
 *  A host flow fed only host frames never touches CUDA (works without a GPU).
 *  The caller's current CUDA context is preserved: MXL's cuda-linear backend and the library's runtime make the
 *  device's PRIMARY context current while they work (cuCtxSetCurrent / cudaSetDevice); every call that may touch CUDA
 *  in the calling thread (create with a cuda-linear option, write / submit / submit_at with a CUDA frame or on a
 *  writer that uses CUDA, destroy) saves the thread's current context and restores it before returning. The library's
 *  own work always runs in the primary context, so CUDA frames and the caller's cuda_stream must belong to the primary
 *  context of the writer's device (memory of a cuCtxCreate context is not valid there).
 *
 * SUPPORTED FLOWS
 *  `video/v210a`, any raster with even width (interlaced: even height), scan progressive or `interlaced_tff`
 *  (`interlaced_bff` is rejected), frame rates 25, 50, 30000/1001, 60000/1001 progressive; 25, 30000/1001 interlaced.
 *
 * ERRORS
 *  Functions return VFGFX_OK (0) or an error code. MXL status codes (mxlStatus, currently 1..1032) are passed through
 *  unchanged; VFGFX_ERR_* codes are >= 2000, so the two ranges never overlap. No C++ exception crosses the API
 *  (VFGFX_ERR_INTERNAL). Error classes: VFGFX_ERR_INVALID_ARG = NULL pointers and contradictory fields;
 *  VFGFX_ERR_UNSUPPORTED = a value this library does not know (unknown enum value of alpha / mem / pixel_format /
 *  scan, unknown flag, non-zero reserved or unknown trailing field) — typically from a newer header. Messages are in
 *  English and meant for logs; do not parse them.
 *   - vfgfx_last_error(w): message of the most recent failed call on w (any caller thread); cleared (returns "") by the
 *     next successful call on w that changes state (write, submit, submit_at, cancel_scheduled, start/stop pacing).
 *     The read-only queries (vfgfx_get_stats, vfgfx_get_format, vfgfx_committed, vfgfx_current_index,
 *     vfgfx_index_time_ns, vfgfx_version*) never set or clear it and do not count in stats.caller_errors, so a
 *     monitoring thread does not hide the caller's error. Writer-wide, so with concurrent callers it may describe
 *     another thread's call — decide on return codes.
 *   - vfgfx_pacing_error(w): most recent error of the pacing thread (open/copy/commit failure, or the exception that
 *     ended the thread); cleared by the next successful commit of the pacing thread. If the thread ended on an exception
 *     the mode stays "running" (vfgfx_write -> VFGFX_ERR_STATE) until vfgfx_stop_pacing.
 *   - vfgfx_last_error(NULL): this thread's most recent vfgfx_writer_create / vfgfx_flow_def error; cleared by the next
 *     successful one of these calls in this thread.
 *  The returned pointer is never NULL and stays valid until the next vfgfx_last_error / vfgfx_pacing_error call in the
 *  same thread.
 *
 * THREADING
 *  For one writer, vfgfx_write, vfgfx_submit, vfgfx_submit_at, start/stop pacing and the queries may be called from
 *  several threads (internally serialized). vfgfx_writer_destroy must not race with other calls on the same writer.
 *  Different writers are independent.
 */
#include <stddef.h>
#include <stdint.h>

#include <mxl/mxl.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Version --------------------------------------------------------------------------------------------------------- */
#define VFGFX_VERSION_MAJOR 0
#define VFGFX_VERSION_MINOR 3
#define VFGFX_VERSION_PATCH 0
#define VFGFX_MAKE_VERSION(major, minor, patch) (((uint32_t)(major) << 16) | ((uint32_t)(minor) << 8) | (uint32_t)(patch))
#define VFGFX_VERSION VFGFX_MAKE_VERSION(VFGFX_VERSION_MAJOR, VFGFX_VERSION_MINOR, VFGFX_VERSION_PATCH)
#define VFGFX_VERSION_STRING "0.3.0"
#define VFGFX_ABI_VERSION 1 /* ABI epoch: SONAME libvramflow-gfx.so.1, symbol version node VFGFX_1 */

/* --- Error codes ------------------------------------------------------------------------------------------------------ */
#define VFGFX_OK 0
#define VFGFX_ERR_INVALID_ARG 2001      /* NULL pointer, contradictory fields, bad JSON, odd submit_at index (interlaced) */
#define VFGFX_ERR_GEOMETRY 2002         /* frame size/stride/alignment does not fit the flow; unsupported raster/rate */
#define VFGFX_ERR_CUDA 2003             /* CUDA runtime error (the message has the CUDA error string) */
#define VFGFX_ERR_STATE 2004            /* call not allowed in this mode; flow already exists */
#define VFGFX_ERR_INTERNAL 2005         /* unexpected internal exception (never propagated across the API) */
#define VFGFX_ERR_STRUCT_SIZE 2006      /* struct_size smaller than required or larger than 4096 */
#define VFGFX_ERR_UNSUPPORTED 2007      /* unknown enum value / flag, non-zero reserved field or unknown trailing bytes */
#define VFGFX_ERR_TOO_LATE 2008         /* vfgfx_submit_at: first_index < next_pacing_index */
#define VFGFX_ERR_TOO_EARLY 2009        /* vfgfx_submit_at: first_index too far in the future */
#define VFGFX_ERR_QUEUE_FULL 2010       /* vfgfx_submit_at: queue_capacity frames already pending */
#define VFGFX_ERR_BUFFER_TOO_SMALL 2011 /* vfgfx_flow_def: buf_size < *required */

#define VFGFX_INDEX_NONE UINT64_MAX   /* "no index" (nothing committed yet, NULL writer) */
#define VFGFX_SUBMIT_AT_MIN_QUEUE 4   /* the vfgfx_submit_at queue holds at least this many pending frames */
#define VFGFX_SUBMIT_AT_MAX_LEAD_S 10 /* vfgfx_submit_at accepts first_index at most this many seconds ahead */

/* --- Types ------------------------------------------------------------------------------------------------------------ */
typedef struct vfgfx_writer_s* vfgfx_writer;

typedef enum { VFGFX_ALPHA_STRAIGHT = 0, VFGFX_ALPHA_PREMULTIPLIED = 1 } vfgfx_alpha;
typedef enum { VFGFX_MEM_HOST = 0, VFGFX_MEM_CUDA = 1 } vfgfx_mem;
/* Pixel format layout: bits 0..7 = channel order (0 = RGBA, 1 = BGRA), bits 8..15 = bytes per channel - 1 (0 = 8 bit).
 * 16-bit formats, when added, will be 0x0100 | order (RGBA16 = 0x0100, BGRA16 = 0x0101) with stride >= width x 8. */
typedef enum { VFGFX_PIXEL_RGBA8 = 0x0000, VFGFX_PIXEL_BGRA8 = 0x0001 } vfgfx_pixel_format;
typedef enum { VFGFX_SCAN_PROGRESSIVE = 0, VFGFX_SCAN_INTERLACED_TFF = 1 } vfgfx_scan;

/* vfgfx_frame.flags */
#define VFGFX_FRAME_CALLER_SYNCED 0x1u /* CUDA frame already complete: the library does not wait (see CUDA FRAMES) */

/* One input frame. Initialise with VFGFX_FRAME_INIT or vfgfx_frame_init(), then set the fields. */
typedef struct vfgfx_frame {
    uint32_t struct_size;  /* sizeof(vfgfx_frame) */
    uint32_t flags;        /* VFGFX_FRAME_* */
    const void* pixels;    /* first byte of row 0 (host or CUDA device pointer, see mem) */
    uint32_t width;        /* pixels; must equal the flow width */
    uint32_t height;       /* rows of the full frame (interlaced: both fields); must equal the flow height */
    uint32_t stride;       /* bytes from one row to the next, >= width x 4 */
    uint32_t pixel_format; /* vfgfx_pixel_format */
    uint32_t alpha;        /* vfgfx_alpha */
    uint32_t mem;          /* vfgfx_mem */
    void* cuda_stream;     /* optional cudaStream_t whose prior work produces the frame (CUDA frames only), or NULL */
    uint64_t reserved[4];  /* must be zero */
} vfgfx_frame;

#define VFGFX_FRAME_INIT                                                                                                  \
    { (uint32_t)sizeof(vfgfx_frame), 0u, NULL, 0u, 0u, 0u, VFGFX_PIXEL_RGBA8, VFGFX_ALPHA_STRAIGHT, VFGFX_MEM_HOST, NULL, \
      {0u, 0u, 0u, 0u} }

/* Writer counters and state (vfgfx_get_stats). Counters are cumulative since vfgfx_writer_create. The fields are read
 * one by one while the writer runs: they are NOT an atomic snapshot (e.g. committed may already include a grain that
 * last_committed_index does not show yet). */
typedef struct vfgfx_stats {
    uint32_t struct_size;          /* in: sizeof(vfgfx_stats) */
    uint32_t flags;                /* in: must be 0 */
    uint64_t committed;            /* grains committed with content, both modes (== vfgfx_committed; INVALID: no_frame) */
    uint64_t written;              /* of which by vfgfx_write */
    uint64_t paced;                /* of which by the pacing thread */
    uint64_t submitted;            /* frames accepted by vfgfx_submit */
    uint64_t scheduled;            /* frames accepted by vfgfx_submit_at */
    uint64_t scheduled_late;       /* scheduled frames that became current after their first_index */
    uint64_t scheduled_replaced;   /* scheduled frames that never became current (replaced or cancelled) */
    uint64_t late_skips;           /* grains skipped because the pacing thread was late */
    uint64_t no_frame;             /* grain periods without any frame: committed as INVALID grains (see GAPS) */
    uint64_t clock_backward;       /* backward clock steps seen by the pacing thread */
    uint64_t pacing_errors;        /* pacing thread open/copy/commit failures and exceptions */
    uint64_t caller_errors;        /* failed calls on this writer */
    uint64_t last_committed_index; /* index of the most recent commit (content or INVALID), VFGFX_INDEX_NONE if none */
    uint64_t next_pacing_index;    /* smallest first_index vfgfx_submit_at accepts right now */
    uint32_t pacing;               /* 1 while paced mode is active (between start and stop) */
    uint32_t queue_depth;          /* vfgfx_submit_at frames pending */
    uint32_t queue_capacity;       /* maximum pending frames (>= VFGFX_SUBMIT_AT_MIN_QUEUE) */
    uint32_t reserved0;
    uint64_t last_frame_index;     /* 0.3: index of the most recent commit WITH a frame (not INVALID), VFGFX_INDEX_NONE if none */
    uint64_t reserved[7];
} vfgfx_stats;

#define VFGFX_STATS_INIT                                                                                        \
    { (uint32_t)sizeof(vfgfx_stats), 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, \
      {0u, 0u, 0u, 0u, 0u, 0u, 0u} }

/* Flow definition parameters (vfgfx_flow_def): either `format`, or width/height/scan/frame rate. */
typedef struct vfgfx_flow_params {
    uint32_t struct_size;   /* sizeof(vfgfx_flow_params) */
    uint32_t flags;         /* must be 0 */
    const char* flow_id;    /* required: UUID "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" (hex digits) */
    const char* label;      /* optional, NULL -> "vramflow-gfx"; also the description; <= 256 bytes (JSON-escaped) */
    const char* group_hint; /* optional NMOS grouphint "<group>:<role>" (<= 256 bytes), NULL ->
                               "vramflow-gfx-<flow_id>:Video" */
    const char* format;     /* optional broadcast name ("1080i50", "1080p25", "720p59.94", "2160p50", …); when set,
                               width, height, scan and frame_rate_* must be 0 */
    uint32_t width;         /* pixels, even */
    uint32_t height;        /* rows of the full frame (even when interlaced) */
    uint32_t scan;          /* vfgfx_scan */
    uint32_t reserved0;
    int64_t frame_rate_num; /* FRAMES per second: 1080i50 -> 25/1, 59.94p -> 60000/1001 */
    int64_t frame_rate_den;
    uint64_t reserved[4];
} vfgfx_flow_params;

#define VFGFX_FLOW_PARAMS_INIT \
    { (uint32_t)sizeof(vfgfx_flow_params), 0u, NULL, NULL, NULL, NULL, 0u, 0u, VFGFX_SCAN_PROGRESSIVE, 0u, 0, 0, {0u, 0u, 0u, 0u} }

static inline vfgfx_frame vfgfx_frame_init(void) {
    vfgfx_frame f = VFGFX_FRAME_INIT;
    return f;
}
static inline vfgfx_stats vfgfx_stats_init(void) {
    vfgfx_stats s = VFGFX_STATS_INIT;
    return s;
}
static inline vfgfx_flow_params vfgfx_flow_params_init(void) {
    vfgfx_flow_params p = VFGFX_FLOW_PARAMS_INIT;
    return p;
}

/* --- Functions -------------------------------------------------------------------------------------------------------- */
/* Version of the loaded library: VFGFX_MAKE_VERSION(major, minor, patch) and "major.minor.patch". */
uint32_t vfgfx_version(void);
const char* vfgfx_version_string(void);

/* Builds the `video/v210a` MXL flow definition JSON into buf (NUL-terminated). *required (if not NULL) receives the
 * size needed including the NUL. buf == NULL with buf_size == 0 only queries the size (returns VFGFX_OK). buf_size too
 * small -> VFGFX_ERR_BUFFER_TOO_SMALL (buf gets "" if buf_size > 0). Validates like vfgfx_writer_create (raster, scan,
 * rate); errors -> vfgfx_last_error(NULL). */
int vfgfx_flow_def(const vfgfx_flow_params* p, char* buf, size_t buf_size, size_t* required);

/* Creates the flow described by flow_def_json (`video/v210a`) and a writer for it. writer_options_json: MXL writer
 * options (NULL or "" = host memory; {"payload":{"backend":"cuda-linear","deviceUuid":"GPU-…"}} = GPU memory). On
 * error *out = NULL and vfgfx_last_error(NULL) has the message. */
int vfgfx_writer_create(mxlInstance inst, const char* flow_def_json, const char* writer_options_json, vfgfx_writer* out);
/* Stops pacing, releases the MXL writer and frees all resources. NULL is a no-op. */
void vfgfx_writer_destroy(vfgfx_writer w);

/* Manual mode: converts f and commits grain `index` (interlaced: field index % 2). VFGFX_ERR_STATE while pacing. */
int vfgfx_write(vfgfx_writer w, uint64_t index, const vfgfx_frame* f);

/* Paced mode. start: VFGFX_ERR_STATE if already running. stop: joins the thread; no-op when not running. */
int vfgfx_start_pacing(vfgfx_writer w, int64_t offset_ns);
int vfgfx_stop_pacing(vfgfx_writer w);
/* Makes f the current content from the next grain the pacing thread processes. */
int vfgfx_submit(vfgfx_writer w, const vfgfx_frame* f);
/* Makes f the content from grain first_index on (accepted range and queue: see GRAIN INDICES AND TIMING). */
int vfgfx_submit_at(vfgfx_writer w, uint64_t first_index, const vfgfx_frame* f);

/* Cancels every pending vfgfx_submit_at frame (counted in stats.scheduled_replaced); the current content stays. */
int vfgfx_cancel_scheduled(vfgfx_writer w);

/* Index of the grain period containing now (TAI) at the flow grain rate; VFGFX_INDEX_NONE for NULL. */
uint64_t vfgfx_current_index(vfgfx_writer w);
/* TAI time (ns) of grain `index` at the flow grain rate (mxlIndexToTimestamp); 0 for NULL. */
uint64_t vfgfx_index_time_ns(vfgfx_writer w, uint64_t index);
/* Grains committed with content so far (both modes; paced INVALID grains are stats.no_frame); 0 for NULL. */
uint64_t vfgfx_committed(vfgfx_writer w);
/* Fills *out; out->struct_size and out->flags are set by the caller (VFGFX_STATS_INIT / vfgfx_stats_init()). */
int vfgfx_get_stats(vfgfx_writer w, vfgfx_stats* out);

/* Fills *out with the flow's geometry: width, height, scan, frame_rate_num/den (frames per second) and flow_id (owned by
 * the writer, valid until vfgfx_writer_destroy); label, group_hint and format are set to NULL. out->struct_size and
 * out->flags (0) are set by the caller; fields beyond the caller's struct_size are not touched, a larger struct's
 * unknown tail is zeroed. Read-only: does not change vfgfx_last_error. */
int vfgfx_get_format(vfgfx_writer w, vfgfx_flow_params* out);

/* Error messages (see ERRORS). Never NULL; "" when there is no error. */
const char* vfgfx_last_error(vfgfx_writer w);
const char* vfgfx_pacing_error(vfgfx_writer w);

#ifdef __cplusplus
}
#endif
#endif /* VRAMFLOW_GFX_H */
