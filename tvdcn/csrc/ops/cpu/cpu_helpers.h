#pragma once
#include <cmath>

#define CPU_1D_KERNEL_LOOP(i, n) CPU_1D_KERNEL_LOOP_T(i, n, int)

#define CPU_1D_KERNEL_LOOP_T(i, n, index_t)                     \
    for (index_t i = 0; i < n; ++i)

#if defined(_MSC_VER)
#define __forceinline__ __forceinline
#elif defined(__GNUC__) && !defined(__clang__)
#define __forceinline__ __attribute__((always_inline)) inline
#else
#define __forceinline__ inline
#endif

template<typename scalar_t>
inline scalar_t min(scalar_t left, scalar_t right) {
    return std::min(left, right);
}

template<typename scalar_t>
inline scalar_t max(scalar_t left, scalar_t right) {
    return std::max(left, right);
}
