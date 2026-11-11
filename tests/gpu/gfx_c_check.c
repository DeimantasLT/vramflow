/* SPDX-FileCopyrightText: 2026 Deimantas Kiršinas */
/* SPDX-License-Identifier: Apache-2.0 */
/* include/vramflow/gfx.h privalo kompiliuotis kaip C (-std=c11 -pedantic -Werror) ir susilinkuoti su libvramflow-gfx.so. */
#include <stdio.h>
#include <string.h>

#include <vramflow/gfx.h>

int main(void) {
    vfgfx_writer w = NULL;
    vfgfx_frame f;
    memset(&f, 0, sizeof f);
    f.alpha = VFGFX_ALPHA_STRAIGHT;
    f.mem = VFGFX_MEM_HOST;
    if (vfgfx_writer_create(NULL, NULL, NULL, &w) != 2001 || w != NULL) return 1;
    if (vfgfx_last_error(NULL)[0] == '\0') return 2;
    if (vfgfx_write(NULL, 0, &f) != 2001 || vfgfx_submit(NULL, &f) != 2001) return 3;
    if (vfgfx_committed(NULL) != 0) return 4;
    vfgfx_writer_destroy(NULL, NULL);
    puts("OK");
    return 0;
}
