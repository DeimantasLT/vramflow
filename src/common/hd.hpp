// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// __host__ __device__ žymė, kad tas pats kodas kompiliuotųsi ir nvcc, ir paprastu C++ kompiliatoriumi.
#if defined(__CUDACC__)
#    define MXLV_HD __host__ __device__ __forceinline__
#else
#    define MXLV_HD inline
#endif
