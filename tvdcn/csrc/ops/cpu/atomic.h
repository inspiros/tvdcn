#pragma once

#include "cpu_helpers.h"

template<typename T>
__forceinline__ void cpuAtomicAdd(T* address, T val) {
#ifdef AT_PARALLEL_OPENMP
#pragma omp atomic
#endif
    *address += val;
}
