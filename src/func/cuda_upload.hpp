// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Host -> device kadro įkėlimas, kuris grįžta tik kai duomenys JAU įrenginyje.
//
// cudaMemcpy H2D iš pageable atminties grįžta, kai buferis nukopijuotas į staging, o DMA į įrenginį gali dar vykti (CUDA
// „API synchronization behavior“). Tvarka garantuojama tik tam pačiam (legacy default) srautui; libvramflow-gfx kadrą skaito
// savo non-blocking srautu (ir savo privačiu cudart), todėl be sinchronizavimo konvertuotų dalinai įkeltą kadrą (gfx.h:
// kviečiantysis privalo būti BAIGĘS rašyti į f->rgba). Išmatuota: capacity G N >= 2 — lyginis gfxgen laukas neteisingas.
#include <cuda_runtime_api.h>

#include <cstddef>

namespace mxlv::func {

inline cudaError_t upload_h2d(void* dst, const void* src, std::size_t bytes) {
    const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    return e != cudaSuccess ? e : cudaStreamSynchronize(nullptr);  // default stream: tas pats, kuriame cudaMemcpy
}

}  // namespace mxlv::func
