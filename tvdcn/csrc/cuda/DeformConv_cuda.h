#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <THC/THCAtomics.cuh>

constexpr int kMaxParallelImgs = 32;

inline unsigned int GET_THREADS(
        const float FRACTION = 1.0
        ) {
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

static int get_greatest_divisor_below_bound(int n, int bound) {
    for (int k = std::min(n, bound); k > 1; --k) {
        if (n % k == 0) {
            return k;
        }
    }
    return 1;
}

// Temporarily counter latest MSVC update that causes incompatibility with CUDA
#if (_MSC_VER >= 1928)
#define floor floorf
#endif
