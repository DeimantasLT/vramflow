// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Slot'o adresas: cuda-linear → device pointer; host → host pointer (tada GPU dirba su scratch buferiu).
#include <mxl/flow.h>

#include <cstring>
#include <string>

#include "CudaLinearStorage.h"

namespace mxlv {

inline bool is_cuda_layout(const mxlGrainStorageLayout& l) {
    return std::strncmp(l.storageType, MXL_CUDA_LINEAR_STORAGE_TYPE, sizeof l.storageType) == 0;
}
inline bool is_host_layout(const mxlGrainStorageLayout& l) {
    return std::strncmp(l.storageType, MXL_PAYLOAD_STORAGE_HOST_POINTER, sizeof l.storageType) == 0;
}
inline void* slot_ptr(const mxlGrainStorageLayout& l, const mxlGrainStorage& s) {
    return is_cuda_layout(l) ? mxlCudaLinearDevicePointer(s) : s.host.pointer;
}
inline std::string storage_name(const mxlGrainStorageLayout& l) {
    return std::string(l.storageType, strnlen(l.storageType, sizeof l.storageType));
}

}  // namespace mxlv
