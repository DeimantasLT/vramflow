/* SPDX-FileCopyrightText: 2026 Deimantas Kiršinas */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VRAMFLOW_GFX_H
#define VRAMFLOW_GFX_H
/*
 * libvramflow-gfx: bet kuri programa gali rašyti RGBA kadrus į standartinį MXL `video/v210a` flow'ą
 * (fill v210 + key, straight alfa, video diapazonas; host arba cuda-linear saugykla).
 *
 * Du režimai (vienu metu negalimi):
 *  - rankinis: vfgfx_write(index, kadras) rašo grain'ą `index` (interlaced rastre laukas index%2 — paritetas 0 =
 *    lyginis indeksas = viršutinis laukas = kadro eilutės 0, 2, 4…; progresyviame rastre paritetas ignoruojamas);
 *  - ritmo: vfgfx_start_pacing(offset_ns) paleidžia bibliotekos giją, kuri kas lauką (TAI laikas, 50/1) commit'ina
 *    PASKUTINĮ vfgfx_submit kadrą. Iki pirmo submit grain'ų nėra. Kol ritmo gija veikia, ji vienintelė liečia MXL writer'į.
 *
 * Kadro atmintis ir kaina: abi atminties rūšys (VFGFX_MEM_HOST / VFGFX_MEM_CUDA) veikia su abiem saugyklomis.
 * Atitinkanti pora (host→host, CUDA→cuda-linear) — vienas konvertavimas (CPU ciklas arba kernel'is) tiesiai į
 * paskirties atmintį. Kryžminė pora kainuoja kopiją: host kadras → cuda-linear = H2D (~8 MB 1080p) + kernel'is;
 * CUDA kadras → host saugykla = D2H + CPU konvertavimas. Ritmo režime ši kaina mokama tik vfgfx_submit metu, ne
 * kiekvieną lauką. CUDA kadras turi priklausyti tam pačiam įrenginiui kaip cuda-linear saugykla (deviceUuid iš
 * writer_options_json; kitaip — kviečiančios gijos dabartiniam), rodyklė ir stride — kartotiniai 4 B.
 *
 * Gijų sauga: vienas vfgfx_writer — vienu metu viena kviečianti gija (submit leidžiamas kviesti iš kelių gijų,
 * jis serializuojamas). Laiko žymės: indeksai iš TAI (mxlGetTime), grain'ų ritmas 50/1.
 */
#include <stdint.h>

#include <mxl/mxl.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vfgfx_writer_s* vfgfx_writer;
typedef enum { VFGFX_ALPHA_STRAIGHT = 0, VFGFX_ALPHA_PREMULTIPLIED = 1 } vfgfx_alpha;
typedef enum { VFGFX_MEM_HOST = 0, VFGFX_MEM_CUDA = 1 } vfgfx_mem;
typedef struct { const void* rgba; uint32_t width, height, stride; vfgfx_alpha alpha; vfgfx_mem mem; } vfgfx_frame;
/* 0 = OK; 1..1023 = mxlStatus; >= 2000 = VFGFX_ERR_* (INVALID_ARG 2001, GEOMETRY 2002, CUDA 2003, STATE 2004) */
int vfgfx_writer_create(mxlInstance inst, const char* flow_def_json, const char* writer_options_json, vfgfx_writer* out);
int vfgfx_write(vfgfx_writer w, uint64_t index, const vfgfx_frame* f);   /* rankinis: grain index (laukas index%2) */
int vfgfx_start_pacing(vfgfx_writer w, int64_t offset_ns);                 /* ritmo gija, kas laukas */
int vfgfx_submit(vfgfx_writer w, const vfgfx_frame* f);                    /* ritmo režimas: naujas kadras */
int vfgfx_stop_pacing(vfgfx_writer w);
uint64_t vfgfx_committed(vfgfx_writer w);                                  /* commit'intų grain'ų skaičius */
const char* vfgfx_last_error(vfgfx_writer w);
void vfgfx_writer_destroy(vfgfx_writer w, mxlInstance inst);

#ifdef __cplusplus
}
#endif
#endif /* VRAMFLOW_GFX_H */
