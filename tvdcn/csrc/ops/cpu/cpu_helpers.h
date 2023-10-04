#pragma once

#include <cmath>
#include <type_traits>

#ifdef AT_PARALLEL_OPENMP
#ifndef OMP_NUM_THREADS
#define OMP_NUM_THREADS 32
#endif

#if defined(_MSC_VER)
#define _PRAGMA_OMP_PARALLEL_FOR __pragma(omp parallel for num_threads(OMP_NUM_THREADS))
#else
#define _PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for num_threads(OMP_NUM_THREADS)")
#endif
#else
#define _PRAGMA_OMP_PARALLEL_FOR
#endif

// regular for loop
#define CPU_1D_KERNEL_LOOP_BETWEEN_T(i, start, end, index_t) \
for (index_t i = start; i < end; ++i)

#define CPU_1D_KERNEL_LOOP_T(i, n, index_t) \
CPU_1D_KERNEL_LOOP_BETWEEN_T(i, 0, n, index_t)

#define CPU_1D_KERNEL_LOOP_BETWEEN(i, start, end) \
CPU_1D_KERNEL_LOOP_BETWEEN_T(i, start, end,       \
    std::remove_cv_t<std::remove_reference_t<std::common_type_t<decltype(start), decltype(between)>>>)

#define CPU_1D_KERNEL_LOOP(i, n) \
CPU_1D_KERNEL_LOOP_T(i, n, std::remove_cv_t<std::remove_reference_t<decltype(n)>>)

// openmp parallel for loop
#define CPU_1D_PARALLEL_KERNEL_LOOP_BETWEEN_T(i, start, end, index_t) \
_PRAGMA_OMP_PARALLEL_FOR                                     \
CPU_1D_KERNEL_LOOP_BETWEEN_T(i, start, end, index_t)

#define CPU_1D_PARALLEL_KERNEL_LOOP_T(i, n, index_t) \
CPU_1D_PARALLEL_KERNEL_LOOP_BETWEEN_T(i, 0, n, index_t)

#define CPU_1D_PARALLEL_KERNEL_LOOP_BETWEEN(i, start, end) \
CPU_1D_PARALLEL_KERNEL_LOOP_BETWEEN_T(i, start, end,       \
    std::remove_cv_t<std::remove_reference_t<std::common_type_t<decltype(start), decltype(between)>>>)

#define CPU_1D_PARALLEL_KERNEL_LOOP(i, n) \
CPU_1D_PARALLEL_KERNEL_LOOP_T(i, n, std::remove_cv_t<std::remove_reference_t<decltype(n)>>)

// inline
#if defined(_MSC_VER)
#define __forceinline__ __forceinline
#elif defined(__GNUC__) && !defined(__clang__)
#define __forceinline__ __attribute__((always_inline)) inline
#else
#define __forceinline__ inline
#endif

using std::min;
using std::max;
using std::clamp;
using std::ceil;
using std::floor;
