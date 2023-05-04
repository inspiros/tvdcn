#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); i += (blockDim.x * gridDim.x))

inline unsigned int GET_THREADS(
        const float FRACTION = 1.0) {
#ifdef __HIP_PLATFORM_HCC__
    return (unsigned int) (256 * FRACTION);
#endif
    if (at::cuda::getCurrentDeviceProperties()->major >= 6) {
        return (unsigned int) (1024 * FRACTION);
    }
    return (unsigned int) (512 * FRACTION);
}

inline unsigned int GET_BLOCKS(
        const unsigned int THREADS,
        const unsigned int N) {
    unsigned int kMaxGridNum = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
    return std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
}

// Temporarily counter latest MSVC update that causes incompatibility with CUDA
#if (_MSC_VER >= 1928)
#define floor floorf
#define ceil ceilf
#endif
