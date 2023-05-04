#pragma once
#include <cmath>

#define CPU_1D_KERNEL_LOOP(i, n)                                \
    for (int i = 0; i < n; ++i)

template<typename scalar_t>
inline scalar_t min(scalar_t left, scalar_t right) {
    return std::min(left, right);
}

template<typename scalar_t>
inline scalar_t max(scalar_t left, scalar_t right) {
    return std::max(left, right);
}
