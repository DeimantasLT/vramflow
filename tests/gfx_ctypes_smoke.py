#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""libvramflow-gfx ctypes dūmų testas: sukuria MXL instanciją ir host writer'į, įrašo vieną kadrą, tikrina
vfgfx_committed == 1, sunaikina. Naudojimas: gfx_ctypes_smoke.py <libvramflow-gfx.so> [libmxl.so]
(libmxl.so keliu galima perduoti ir per MXL_LIB; kitaip ieškoma pagal pavadinimą). Sėkmės atveju spausdina `OK`."""
import ctypes
import os
import shutil
import sys
import uuid

VFGFX_ALPHA_STRAIGHT, VFGFX_MEM_HOST = 0, 0
W, H = 128, 72  # interlaced 128x72 v210a: grain'as = laukas 128x36


class Frame(ctypes.Structure):
    _fields_ = [("rgba", ctypes.c_void_p), ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("stride", ctypes.c_uint32), ("alpha", ctypes.c_int), ("mem", ctypes.c_int)]


class Rational(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_int64), ("denominator", ctypes.c_int64)]


def flow_def(flow_id):
    comps = ",".join(
        '{"name":"%s","width":%d,"height":%d,"bit_depth":10}' % (n, w, H) for n, w in (("Y", W), ("Cb", W // 2), ("Cr", W // 2)))
    return ('{"id":"%s","description":"gfx ctypes smoke","label":"gfx ctypes smoke",'
            '"tags":{"urn:x-nmos:tag:grouphint/v1.0":["vramflow-gfx-smoke:Video"]},'
            '"format":"urn:x-nmos:format:video","parents":[],"media_type":"video/v210a",'
            '"grain_rate":{"numerator":25,"denominator":1},"frame_width":%d,"frame_height":%d,'
            '"interlace_mode":"interlaced_tff","colorspace":"BT709","components":[%s]}' % (flow_id, W, H, comps))


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

    gfx.vfgfx_writer_create.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    gfx.vfgfx_write.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(Frame)]
    gfx.vfgfx_committed.restype = ctypes.c_uint64
    gfx.vfgfx_committed.argtypes = [ctypes.c_void_p]
    gfx.vfgfx_last_error.restype = ctypes.c_char_p
    gfx.vfgfx_last_error.argtypes = [ctypes.c_void_p]
    gfx.vfgfx_writer_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    domain = "/dev/shm/vf-gfx-smoke-%d-%s" % (os.getpid(), uuid.uuid4().hex[:8])
    os.makedirs(domain)
    inst = None
    writer = ctypes.c_void_p()
    try:
        inst = mxl.mxlCreateInstance(domain.encode(), b"")
        assert inst, "mxlCreateInstance nepavyko"
        flow_id = str(uuid.uuid4())
        rc = gfx.vfgfx_writer_create(inst, flow_def(flow_id).encode(), None, ctypes.byref(writer))
        assert rc == 0, "vfgfx_writer_create=%d: %s" % (rc, gfx.vfgfx_last_error(None))
        rgba = (ctypes.c_uint8 * (W * H * 4))(*([200, 100, 50, 255] * (W * H)))
        frame = Frame(ctypes.cast(rgba, ctypes.c_void_p), W, H, W * 4, VFGFX_ALPHA_STRAIGHT, VFGFX_MEM_HOST)
        idx = mxl.mxlGetCurrentIndex(ctypes.byref(Rational(50, 1)))
        rc = gfx.vfgfx_write(writer, idx, ctypes.byref(frame))
        assert rc == 0, "vfgfx_write=%d: %s" % (rc, gfx.vfgfx_last_error(writer))
        n = gfx.vfgfx_committed(writer)
        assert n == 1, "vfgfx_committed=%d, tikėtasi 1" % n
    finally:
        if writer.value:
            gfx.vfgfx_writer_destroy(writer, inst)
        if inst:
            mxl.mxlDestroyInstance(inst)
        shutil.rmtree(domain, ignore_errors=True)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
