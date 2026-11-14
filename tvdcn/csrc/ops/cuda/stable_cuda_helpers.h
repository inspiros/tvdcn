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

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
#include <cuda_bf16.h>
#endif

namespace tvdcn::ops::cuda {
// Atomic add into [begin, end). For 16-bit types this issues one paired (32-bit) atomic,
// adding zero to the neighbouring element, which is much faster than a scalar 16-bit
// atomic (same technique as ATen's fastAtomicAdd). The neighbour may lie outside the
// accessor's view only if it is inside [begin, end), so callers pass the bounds of the
// whole contiguous allocation they write to.
template<typename scalar_t>
__device__ __forceinline__ void fast_atomic_add(scalar_t *address, const scalar_t *begin, const scalar_t *end,
                                                const scalar_t val) {
    if constexpr (std::is_same_v<scalar_t, c10::Half>) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        __half *addr = reinterpret_cast<__half *>(address);
        const __half hval = static_cast<__half>(val);
        const bool low = (reinterpret_cast<std::uintptr_t>(addr) % sizeof(__half2)) == 0;
        if (low && address + 1 < end) {
            atomicAdd(reinterpret_cast<__half2 *>(addr), __halves2half2(hval, __ushort_as_half(0)));
        } else if (!low && address - 1 >= begin) {
            atomicAdd(reinterpret_cast<__half2 *>(addr - 1), __halves2half2(__ushort_as_half(0), hval));
        } else {
            gpuAtomicAdd(address, val);
        }
#else
        gpuAtomicAdd(address, val);
#endif
    } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        __nv_bfloat16 *addr = reinterpret_cast<__nv_bfloat16 *>(address);
        const __nv_bfloat16 bval = __float2bfloat16(static_cast<float>(val));
        const __nv_bfloat16 zero = __ushort_as_bfloat16(0);
        const bool low = (reinterpret_cast<std::uintptr_t>(addr) % sizeof(__nv_bfloat162)) == 0;
        if (low && address + 1 < end) {
            atomicAdd(reinterpret_cast<__nv_bfloat162 *>(addr), __halves2bfloat162(bval, zero));
        } else if (!low && address - 1 >= begin) {
            atomicAdd(reinterpret_cast<__nv_bfloat162 *>(addr - 1), __halves2bfloat162(zero, bval));
        } else {
            gpuAtomicAdd(address, val);
        }
#else
        gpuAtomicAdd(address, val);
#endif
    } else {
        gpuAtomicAdd(address, val);
    }
}
}
