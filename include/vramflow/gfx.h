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
 * writer_options_json). deviceUuid PRIVALOMAS daugiau nei vieno GPU kompiuteriuose: biblioteka turi savo (privatų) CUDA
 * runtime, todėl kviečiančiojo cudaSetDevice jai nematomas; be deviceUuid naudojamas įrenginys 0. UUID — "GPU-xxxxxxxx-…",
 * raidžių dydis nesvarbus, rodyklė ir stride — kartotiniai 4 B.
 *
 * CUDA kadro tvarka: numatytai biblioteka pati palaukia, kol baigsis visi GPU darbai tame kontekste, ir tik tada skaito
 * f->rgba — pusiau įrašytas kadras (pvz. po „sinchroninio“ cudaMemcpy H2D iš pageable atminties, kuris gali grįžti prieš DMA
 * pabaigą) neįmanomas. Kviečiantysis, kuris pats sinchronizuoja (cudaStreamSynchronize prieš kvietimą), gali tą laukimą
 * išjungti: vfgfx_set_caller_synced(w, 1). Kitas CUDA kontekstas nei pirminis (cuCtxCreate) — kviečiančiojo atsakomybė.
 * Iškvietimas sinchroninis:
 * grįžus, f->rgba buferį (host ar CUDA) galima iškart naudoti iš naujo / atlaisvinti — biblioteka jo neišsaugo.
 *
 * Ritmo laikas: vfgfx_start_pacing(offset_ns) — grain indeksas i commit'inamas laiku T(i) + offset_ns, kur T(i) — indekso
 * pradžios TAI laikas (50/1). offset_ns > 0 = vėliau, < 0 = anksčiau (iš anksto). Pirmas indeksas — pirmas, kurio laikas
 * dar nepraėjo; sustabdžius ir vėl paleidus, jau commit'inti indeksai nekartojami. Jei gija pavėluoja, vėluojantis indeksas
 * vis tiek įrašomas, o praleisti laukai nerašomi (jokio „catch-up“ srauto).
 *
 * Interlace: palaikomas TIK `interlaced_tff` (lyginis grain indeksas = viršutinis laukas) ir progresyvus (`progressive` arba
 * nenurodytas) rastras; `interlaced_bff` ir kitos interlace_mode reikšmės atmetamos (vfgfx_writer_create -> INVALID_ARG).
 *
 * ABI: SOVERSION 0 — iki 1.0 struktūrų išdėstymas (vfgfx_frame, vfgfx_pacing_stats…) gali keistis; stabilumo pažado dar nėra.
 *
 * Klaidos: funkcijos grąžina 0 arba kodą (VFGFX_ERR_* žemiau, 1..1023 — mxlStatus). C++ išimtys neperženg ribos
 * (VFGFX_ERR_INTERNAL). Jei ritmo gijoje įvyko išimtis, gija baigiasi, bet režimas lieka „veikiantis“ (vfgfx_write -> STATE)
 * iki vfgfx_stop_pacing. vfgfx_last_error(w): paskutinė kviečiančiosios pusės (bet kurios gijos) klaida, bendra visam writer'iui (jei jos nėra — paskutinė ritmo
 * gijos klaida); vfgfx_last_error(NULL): ŠIOS gijos paskutinė vfgfx_writer_create klaida. Rodyklė galioja iki kito
 * vfgfx_last_error iškvietimo toje pačioje gijoje; niekada NULL.
 *
 * Gijų sauga: vienas vfgfx_writer — vfgfx_write / vfgfx_submit / start / stop gali būti kviečiami iš kelių gijų (vidinės
 * užraktai juos serializuoja), tačiau vfgfx_write ritmo metu grąžina VFGFX_ERR_STATE. Laiko žymės: indeksai iš TAI
 * (mxlGetTime), grain'ų ritmas 50/1.
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
/* 0 = OK; 1..1023 = mxlStatus; >= 2000 = VFGFX_ERR_* (INVALID_ARG 2001, GEOMETRY 2002, CUDA 2003, STATE 2004, INTERNAL 2005) */
#define VFGFX_ERR_INVALID_ARG 2001
#define VFGFX_ERR_GEOMETRY 2002
#define VFGFX_ERR_CUDA 2003
#define VFGFX_ERR_STATE 2004 /* taip pat: flow jau egzistuoja (created == false) */
#define VFGFX_ERR_INTERNAL 2005 /* netikėta vidinė išimtis (niekada neperduodama per C ABI ribą) */
int vfgfx_writer_create(mxlInstance inst, const char* flow_def_json, const char* writer_options_json, vfgfx_writer* out);
int vfgfx_write(vfgfx_writer w, uint64_t index, const vfgfx_frame* f);   /* rankinis: grain index (laukas index%2) */
int vfgfx_start_pacing(vfgfx_writer w, int64_t offset_ns);                 /* ritmo gija, kas laukas */
int vfgfx_submit(vfgfx_writer w, const vfgfx_frame* f);                    /* ritmo režimas: naujas kadras */
int vfgfx_stop_pacing(vfgfx_writer w);
uint64_t vfgfx_committed(vfgfx_writer w);                                  /* commit'intų grain'ų skaičius */
/* CUDA kadrų sinchronizavimas (R20). Numatyta (0): prieš skaitydama CUDA kadrą biblioteka palaukia, kol baigsis VISI
 * GPU darbai tame CUDA kontekste (cudaDeviceSynchronize pirminiame kontekste), todėl pusiau įrašytas kadras neįmanomas,
 * net jei kviečiantysis pamiršo sinchronizuoti. Kaina — laukimas ir kitų stream'ų darbų. synced = 1: kviečiantysis
 * garantuoja, kad kadras jau įrašytas (cudaStreamSynchronize prieš kvietimą), ir biblioteka nelaukia. */
int vfgfx_set_caller_synced(vfgfx_writer w, int synced);
const char* vfgfx_last_error(vfgfx_writer w);
void vfgfx_writer_destroy(vfgfx_writer w, mxlInstance inst);

#ifdef __cplusplus
}
#endif
#endif /* VRAMFLOW_GFX_H */
