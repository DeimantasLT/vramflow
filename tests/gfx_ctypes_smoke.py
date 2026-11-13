#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""libvramflow-gfx ctypes dūmų testas (ABI 0.3): versija, vfgfx_flow_def -> vfgfx_writer_create (host), vienas RGBA ir
vienas BGRA kadras per vfgfx_write, vfgfx_get_stats (committed == written == 2), klaidos išvalymas, sunaikinimas.
Naudojimas: gfx_ctypes_smoke.py <libvramflow-gfx.so> [libmxl.so] (libmxl.so keliu galima perduoti ir per MXL_LIB; kitaip
ieškoma pagal pavadinimą). Sėkmės atveju spausdina `OK`."""
import ctypes
import os
import shutil
import sys
import uuid

VFGFX_ALPHA_STRAIGHT, VFGFX_MEM_HOST = 0, 0
VFGFX_PIXEL_RGBA8, VFGFX_PIXEL_BGRA8 = 0, 1
VFGFX_SCAN_INTERLACED_TFF = 1
VFGFX_ERR_STRUCT_SIZE = 2006
W, H = 128, 72  # interlaced 128x72 v210a: grain'as = laukas 128x36


class Frame(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("pixels", ctypes.c_void_p),
                ("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("stride", ctypes.c_uint32),
                ("pixel_format", ctypes.c_uint32), ("alpha", ctypes.c_uint32), ("mem", ctypes.c_uint32),
                ("cuda_stream", ctypes.c_void_p), ("reserved", ctypes.c_uint64 * 4)]


class Stats(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("flags", ctypes.c_uint32)] + \
        [(n, ctypes.c_uint64) for n in ("committed", "written", "paced", "submitted", "scheduled", "scheduled_late",
                                         "scheduled_replaced", "late_skips", "no_frame", "clock_backward",
                                         "pacing_errors", "caller_errors", "last_committed_index",
                                         "next_pacing_index")] + \
        [(n, ctypes.c_uint32) for n in ("pacing", "queue_depth", "queue_capacity", "reserved0")] + \
        [("last_frame_index", ctypes.c_uint64), ("reserved", ctypes.c_uint64 * 7)]


class FlowParams(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("flow_id", ctypes.c_char_p),
                ("label", ctypes.c_char_p), ("group_hint", ctypes.c_char_p), ("format", ctypes.c_char_p),
                ("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("scan", ctypes.c_uint32),
                ("reserved0", ctypes.c_uint32), ("frame_rate_num", ctypes.c_int64), ("frame_rate_den", ctypes.c_int64),
                ("reserved", ctypes.c_uint64 * 4)]


class Rational(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_int64), ("denominator", ctypes.c_int64)]


def main():
    if len(sys.argv) < 2:
        print("naudojimas: gfx_ctypes_smoke.py <libvramflow-gfx.so> [libmxl.so]", file=sys.stderr)
        return 2
    mxl_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MXL_LIB", "libmxl.so.1")
    mxl = ctypes.CDLL(mxl_path, mode=ctypes.RTLD_GLOBAL)  # pirmas: ta pati libmxl pernaudojama
    gfx = ctypes.CDLL(os.path.abspath(sys.argv[1]))

    mxl.mxlCreateInstance.restype = ctypes.c_void_p
    mxl.mxlCreateInstance.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    mxl.mxlDestroyInstance.argtypes = [ctypes.c_void_p]
    mxl.mxlGetCurrentIndex.restype = ctypes.c_uint64
    mxl.mxlGetCurrentIndex.argtypes = [ctypes.POINTER(Rational)]

    gfx.vfgfx_version.restype = ctypes.c_uint32
    gfx.vfgfx_version_string.restype = ctypes.c_char_p
    gfx.vfgfx_flow_def.argtypes = [ctypes.POINTER(FlowParams), ctypes.c_char_p, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_size_t)]
    gfx.vfgfx_writer_create.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    gfx.vfgfx_write.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(Frame)]
    gfx.vfgfx_committed.restype = ctypes.c_uint64
    gfx.vfgfx_committed.argtypes = [ctypes.c_void_p]
    gfx.vfgfx_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(Stats)]
    gfx.vfgfx_last_error.restype = ctypes.c_char_p
    gfx.vfgfx_last_error.argtypes = [ctypes.c_void_p]
    gfx.vfgfx_writer_destroy.argtypes = [ctypes.c_void_p]

    # ABI išdėstymas užfiksuotas (kaip gfx_c_check.c): dydžiai ir poslinkiai be tarpų.
    assert (ctypes.sizeof(Frame), ctypes.sizeof(Stats), ctypes.sizeof(FlowParams)) == (80, 200, 104)
    # 0.3: last_frame_index buvusioje rezervuotoje vietoje (136), reserved — nuo 144 (dydis nepakitęs)
    assert (Frame.cuda_stream.offset, Frame.reserved.offset, Stats.pacing.offset, Stats.last_frame_index.offset, Stats.reserved.offset,
            FlowParams.width.offset, FlowParams.frame_rate_num.offset, FlowParams.reserved.offset) == (40, 48, 120, 136, 144, 40, 56, 72)

    version = gfx.vfgfx_version()
    vstr = gfx.vfgfx_version_string().decode()
    assert vstr == "%d.%d.%d" % (version >> 16, (version >> 8) & 255, version & 255), (version, vstr)
    assert version >> 8 == 0x0003, "laukta 0.3.x, gauta %s" % vstr

    domain = "/dev/shm/vf-gfx-smoke-%d-%s" % (os.getpid(), uuid.uuid4().hex[:8])
    os.makedirs(domain)
    inst = None
    writer = ctypes.c_void_p()
    try:
        inst = mxl.mxlCreateInstance(domain.encode(), b"")
        assert inst, "mxlCreateInstance nepavyko"
        fp = FlowParams(struct_size=ctypes.sizeof(FlowParams), flow_id=str(uuid.uuid4()).encode(),
                        label=b"gfx ctypes smoke", width=W, height=H, scan=VFGFX_SCAN_INTERLACED_TFF,
                        frame_rate_num=25, frame_rate_den=1)
        need = ctypes.c_size_t()
        assert gfx.vfgfx_flow_def(ctypes.byref(fp), None, 0, ctypes.byref(need)) == 0, gfx.vfgfx_last_error(None)
        buf = ctypes.create_string_buffer(need.value)
        assert gfx.vfgfx_flow_def(ctypes.byref(fp), buf, need.value, None) == 0, gfx.vfgfx_last_error(None)
        rc = gfx.vfgfx_writer_create(inst, buf.value, None, ctypes.byref(writer))
        assert rc == 0, "vfgfx_writer_create=%d: %s" % (rc, gfx.vfgfx_last_error(None))
        rgba = (ctypes.c_uint8 * (W * H * 4))(*([200, 100, 50, 255] * (W * H)))
        bgra = (ctypes.c_uint8 * (W * H * 4))(*([50, 100, 200, 255] * (W * H)))
        idx = mxl.mxlGetCurrentIndex(ctypes.byref(Rational(50, 1)))
        bad = Frame(struct_size=4, pixels=ctypes.cast(rgba, ctypes.c_void_p), width=W, height=H, stride=W * 4)
        rc = gfx.vfgfx_write(writer, idx, ctypes.byref(bad))
        assert rc == VFGFX_ERR_STRUCT_SIZE and gfx.vfgfx_last_error(writer), rc
        for k, (px, fmt) in enumerate(((rgba, VFGFX_PIXEL_RGBA8), (bgra, VFGFX_PIXEL_BGRA8))):
            frame = Frame(struct_size=ctypes.sizeof(Frame), pixels=ctypes.cast(px, ctypes.c_void_p), width=W, height=H,
                          stride=W * 4, pixel_format=fmt, alpha=VFGFX_ALPHA_STRAIGHT, mem=VFGFX_MEM_HOST)
            rc = gfx.vfgfx_write(writer, idx + k, ctypes.byref(frame))
            assert rc == 0, "vfgfx_write=%d: %s" % (rc, gfx.vfgfx_last_error(writer))
        assert gfx.vfgfx_last_error(writer) == b"", "sėkmė turi išvalyti klaidą"
        st = Stats(struct_size=ctypes.sizeof(Stats))
        assert gfx.vfgfx_get_stats(writer, ctypes.byref(st)) == 0
        assert st.committed == 2 and st.written == 2 and st.caller_errors == 1, (st.committed, st.written, st.caller_errors)
        assert st.last_committed_index == idx + 1, st.last_committed_index
        assert st.last_frame_index == idx + 1, st.last_frame_index  # 0.3: paskutinis su kadru (vfgfx_write — visada kadras)
        n = gfx.vfgfx_committed(writer)
        assert n == 2, "vfgfx_committed=%d, tikėtasi 2" % n
    finally:
        if writer.value:
            gfx.vfgfx_writer_destroy(writer)
        if inst:
            mxl.mxlDestroyInstance(inst)
        shutil.rmtree(domain, ignore_errors=True)
    print("OK", vstr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
