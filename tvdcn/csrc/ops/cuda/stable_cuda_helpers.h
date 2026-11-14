#pragma once
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/cuda/Atomic.h>
#include <algorithm>

#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t) \
for (index_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); i += (blockDim.x * gridDim.x))

inline unsigned int GET_THREADS(const float fraction = 1.0) {
    return static_cast<unsigned int>(512 * fraction);
}
inline unsigned int GET_BLOCKS(unsigned int threads, int64_t n) {
    int device, max_grid;
    STD_TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
    STD_TORCH_CHECK(cudaDeviceGetAttribute(&max_grid, cudaDevAttrMaxGridDimX, device) == cudaSuccess,
                    "cudaDeviceGetAttribute failed");
    return static_cast<unsigned int>(std::min<int64_t>(max_grid, (n + threads - 1) / threads));
}
namespace tvdcn::ops::stable_compat {
inline cudaStream_t current_cuda_stream() {
    void* stream = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
        torch::stable::accelerator::getCurrentDeviceIndex(), &stream));
    return static_cast<cudaStream_t>(stream);
}
}
