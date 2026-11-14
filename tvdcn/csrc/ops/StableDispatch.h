#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <cstdlib>

#define TVDCN_PRIVATE_OPTION(NAME, VAL, ...) \
    if (!VAL) {                              \
        static const bool NAME = false;      \
        __VA_ARGS__();                       \
    } else {                                 \
        static const bool NAME = true;       \
        __VA_ARGS__();                       \
    }

#define TVDCN_PRIVATE_OPTIONS2(NAME1, VAL1, NAME2, VAL2, ...) \
    if (!VAL1 && !VAL2) {                                     \
        static const bool NAME1 = false, NAME2 = false;       \
        __VA_ARGS__();                                        \
    }                                                         \
    else if (!VAL1 && VAL2) {                                 \
        static const bool NAME1 = false, NAME2 = true;        \
        __VA_ARGS__();                                        \
    }                                                         \
    else if (VAL1 && !VAL2) {                                 \
        static const bool NAME1 = true, NAME2 = false;        \
        __VA_ARGS__();                                        \
    }                                                         \
    else {                                                    \
        static const bool NAME1 = true, NAME2 = true;         \
        __VA_ARGS__();                                        \
    }

#define TVDCN_DISPATCH_CONDITION(ARG1, ...) \
        TVDCN_PRIVATE_OPTION(ARG1, ARG1, __VA_ARGS__)

#define TVDCN_DISPATCH_CONDITION2(ARG1, ARG2, ...) \
        TVDCN_PRIVATE_OPTIONS2(ARG1, ARG1, ARG2, ARG2, __VA_ARGS__)

// index type
namespace tvdcn::ops {
inline bool force_int64_index() {
    static const bool value = [] {
        const char *env = std::getenv("TVDCN_FORCE_INT64_INDEX");
        return env != nullptr && env[0] == '1';
    }();
    return value;
}
}  // namespace tvdcn::ops

#define TVDCN_DISPATCH_INDEX_TYPE(N_KERNELS, ...) \
    if (((int64_t)N_KERNELS) >= (int64_t(1) << 30)) { \
        using index_t = int64_t;                  \
        __VA_ARGS__();                            \
    }                                             \
    else {                                        \
        using index_t = int;                      \
        __VA_ARGS__();                            \
    }

#define TVDCN_DISPATCH_INDEX_TYPE_CPU(N_KERNELS, ...) \
    using index_t = int64_t;                          \
    __VA_ARGS__();                                    \

// 32-bit indices when every index (and the grid-stride loop variable) stays below 2^30;
// pass stable_compat::index_bound(n_kernels, {tensors...}) as N_KERNELS.
// TVDCN_FORCE_INT64_INDEX=1 (read once per process) forces the 64-bit path so tests can
// cover it without allocating 2^30-element tensors.
#define TVDCN_DISPATCH_INDEX_TYPE_CUDA(N_KERNELS, ...) \
    if (((int64_t)N_KERNELS) >= (int64_t(1) << 30) ||  \
        ::tvdcn::ops::force_int64_index()) {           \
        using index_t = int64_t;                       \
        __VA_ARGS__();                                 \
    }                                                  \
    else {                                             \
        using index_t = int;                           \
        __VA_ARGS__();                                 \
    }

#define TVDCN_DISPATCH_INDEX_TYPE_DEVICE(N_KERNELS, DEVICE, ...) \
C10_CONCATENATE(TVDCN_DISPATCH_INDEX_TYPE_, DEVICE)(N_KERNELS, __VA_ARGS__)
