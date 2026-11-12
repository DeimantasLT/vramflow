/* SPDX-FileCopyrightText: 2026 Deimantas Kiršinas */
/* SPDX-License-Identifier: Apache-2.0 */
/* include/vramflow/gfx.h privalo kompiliuotis kaip C (-std=c11 -pedantic -Werror) ir susilinkuoti su libvramflow-gfx.so:
 * versija (antraštė == .so), _INIT makrokomandos ir *_init() funkcijos, klaidų keliai be writer'io, vfgfx_flow_def. */
#include <stdio.h>
#include <string.h>

#include <stddef.h>
#include <vramflow/gfx.h>

/* ABI išdėstymas užfiksuotas (be tarpų; tas pats tikrina test_gfx_writer.cpp ir gfx_ctypes_smoke.py). */
_Static_assert(sizeof(vfgfx_frame) == 80, "vfgfx_frame size");
_Static_assert(offsetof(vfgfx_frame, pixels) == 8 && offsetof(vfgfx_frame, width) == 16 && offsetof(vfgfx_frame, mem) == 36 &&
                   offsetof(vfgfx_frame, cuda_stream) == 40 && offsetof(vfgfx_frame, reserved) == 48,
               "vfgfx_frame layout");
_Static_assert(sizeof(vfgfx_stats) == 200, "vfgfx_stats size");
_Static_assert(offsetof(vfgfx_stats, committed) == 8 && offsetof(vfgfx_stats, next_pacing_index) == 112 &&
                   offsetof(vfgfx_stats, pacing) == 120 && offsetof(vfgfx_stats, reserved0) == 132 &&
                   offsetof(vfgfx_stats, reserved) == 136,
               "vfgfx_stats layout");
_Static_assert(sizeof(vfgfx_flow_params) == 104, "vfgfx_flow_params size");
_Static_assert(offsetof(vfgfx_flow_params, flow_id) == 8 && offsetof(vfgfx_flow_params, format) == 32 &&
                   offsetof(vfgfx_flow_params, width) == 40 && offsetof(vfgfx_flow_params, frame_rate_num) == 56 &&
                   offsetof(vfgfx_flow_params, reserved) == 72,
               "vfgfx_flow_params layout");

int main(void) {
    vfgfx_writer w = NULL;
    vfgfx_frame f = VFGFX_FRAME_INIT;
    vfgfx_frame g = vfgfx_frame_init();
    vfgfx_stats st = VFGFX_STATS_INIT;
    vfgfx_flow_params p = vfgfx_flow_params_init();
    char buf[2048];
    size_t need = 0;
    if (vfgfx_version() != VFGFX_VERSION || strcmp(vfgfx_version_string(), VFGFX_VERSION_STRING) != 0) return 10;
    if (f.struct_size != sizeof(vfgfx_frame) || g.struct_size != sizeof(vfgfx_frame) || memcmp(&f, &g, sizeof f) != 0) return 11;
    if (st.struct_size != sizeof(vfgfx_stats) || p.struct_size != sizeof(vfgfx_flow_params)) return 12;
    if (f.pixel_format != VFGFX_PIXEL_RGBA8 || f.alpha != VFGFX_ALPHA_STRAIGHT || f.mem != VFGFX_MEM_HOST) return 13;
    if (vfgfx_writer_create(NULL, NULL, NULL, &w) != VFGFX_ERR_INVALID_ARG || w != NULL) return 1;
    if (vfgfx_last_error(NULL)[0] == '\0') return 2;
    if (vfgfx_write(NULL, 0, &f) != VFGFX_ERR_INVALID_ARG || vfgfx_submit(NULL, &f) != VFGFX_ERR_INVALID_ARG) return 3;
    if (vfgfx_submit_at(NULL, 0, &f) != VFGFX_ERR_INVALID_ARG || vfgfx_get_stats(NULL, &st) != VFGFX_ERR_INVALID_ARG) return 4;
    if (vfgfx_committed(NULL) != 0 || vfgfx_current_index(NULL) != VFGFX_INDEX_NONE) return 5;
    if (vfgfx_pacing_error(NULL)[0] != '\0') return 6;
    p.flow_id = "7a210a00-0000-4000-8000-0000000000c1";
    p.format = "1080i50";
    if (vfgfx_flow_def(&p, NULL, 0, &need) != VFGFX_OK || need < 100 || need > sizeof buf) return 7;
    if (vfgfx_flow_def(&p, buf, sizeof buf, NULL) != VFGFX_OK || strstr(buf, "video/v210a") == NULL) return 8;
    if (vfgfx_last_error(NULL)[0] != '\0') return 9; /* sėkmė išvalė */
    vfgfx_writer_destroy(NULL);
    puts("OK");
    return 0;
}
