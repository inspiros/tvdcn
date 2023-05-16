#pragma once

#define TVDCN_PRIVATE_OPTION(NAME, VAL, ...)     \
  if (!VAL) {                                    \
    static const bool NAME = false;              \
    __VA_ARGS__();                               \
  } else {                                       \
    static const bool NAME = true;               \
    __VA_ARGS__();                               \
  }

#define TVDCN_PRIVATE_OPTIONS2(NAME1, VAL1, NAME2, VAL2, ...)     \
  if (!VAL1 && !VAL2) {                                           \
    static const bool NAME1 = false, NAME2 = false;               \
    __VA_ARGS__();                                                \
  }                                                               \
  else if (!VAL1 && VAL2) {                                       \
    static const bool NAME1 = false, NAME2 = true;                \
    __VA_ARGS__();                                                \
  }                                                               \
  else if (VAL1 && !VAL2) {                                       \
    static const bool NAME1 = true, NAME2 = false;                \
    __VA_ARGS__();                                                \
  }                                                               \
  else {                                                          \
    static const bool NAME1 = true, NAME2 = true;                 \
    __VA_ARGS__();                                                \
  }

#define TVDCN_DISPATCH_CONDITION(ARG1, ...)     \
    TVDCN_PRIVATE_OPTION(ARG1, ARG1, __VA_ARGS__)

#define TVDCN_DISPATCH_CONDITION2(ARG1, ARG2, ...)     \
    TVDCN_PRIVATE_OPTIONS2(ARG1, ARG1, ARG2, ARG2, __VA_ARGS__)

// index type
#define TVDCN_DISPATCH_INDEX_TYPE(N_KERNELS, ...)     \
  if (((int64_t)N_KERNELS) > (1 << 31)) {             \
    using index_t = int64_t;                          \
    __VA_ARGS__();                                    \
  }                                                   \
  else {                                              \
    using index_t = int;                              \
    __VA_ARGS__();                                    \
  }

#define TVDCN_DISPATCH_INDEX_TYPE2(N_KERNELS1, N_KERNELS2, ...)                     \
  if (((int64_t)N_KERNELS1) > (1 << 31) || ((int64_t)N_KERNELS2) > (1 << 31)) {     \
    using index_t = int64_t;                                                        \
    __VA_ARGS__();                                                                  \
  }                                                                                 \
  else {                                                                            \
    using index_t = int;                                                            \
    __VA_ARGS__();                                                                  \
  }
