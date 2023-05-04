#pragma once

#define TVDCN_PRIVATE_OPTION(NAME, VAL, ...)       \
  if (!VAL) {                                      \
    static const bool NAME = false;                \
    return __VA_ARGS__();                          \
  } else {                                         \
    static const bool NAME = true;                 \
    return __VA_ARGS__();                          \
  }

#define TVDCN_PRIVATE_OPTIONS2(NAME1, VAL1, NAME2, VAL2, ...)   \
  if (!VAL1 && !VAL2) {                                         \
    static const bool NAME1 = false, NAME2 = false;             \
    return __VA_ARGS__();                                       \
  }                                                             \
  else if (VAL1 && !VAL2) {                                     \
    static const bool NAME1 = false, NAME2 = true;              \
    return __VA_ARGS__();                                       \
  }                                                             \
  else if (!VAL1 && VAL2) {                                     \
    static const bool NAME1 = true, NAME2 = false;              \
    return __VA_ARGS__();                                       \
  }                                                             \
  else if (VAL1 && VAL2) {                                      \
    static const bool NAME1 = true, NAME2 = true;               \
    return __VA_ARGS__();                                       \
  }

#define TVDCN_PRIVATE_OPTIONS3(NAME1, VAL1, NAME2, VAL2, NAME3, VAL3, ...)   \
  if (!VAL1 && !VAL2 && !VAL3) {                                             \
    static const bool NAME1 = false, NAME2 = false, NAME3 = false;           \
    return __VA_ARGS__();                                                    \
  }                                                                          \
  else if (!VAL1 && !VAL2 && VAL3) {                                         \
    static const bool NAME1 = false, NAME2 = false, NAME3 = true;            \
    return __VA_ARGS__();                                                    \
  }                                                                          \
  else if (!VAL1 && VAL2 && !VAL3) {                                         \
    static const bool NAME1 = false, NAME2 = true, NAME3 = false;            \
    return __VA_ARGS__();                                                    \
  }                                                                          \
  else if (!VAL1 && VAL2 && VAL3) {                                          \
    static const bool NAME1 = false, NAME2 = true, NAME3 = true;             \
    return __VA_ARGS__();                                                    \
  }                                                                          \
  else if (VAL1 && !VAL2 && !VAL3) {                                         \
    static const bool NAME1 = true, NAME2 = false, NAME3 = false;            \
    return __VA_ARGS__();                                                    \
  }                                                                          \
  else if (VAL1 && !VAL2 && VAL3) {                                          \
    static const bool NAME1 = true, NAME2 = false, NAME3 = true;             \
    return __VA_ARGS__();                                                    \
  }                                                                          \
  else if (VAL1 && VAL2 && !VAL3) {                                          \
    static const bool NAME1 = true, NAME2 = true, NAME3 = false;             \
    return __VA_ARGS__();                                                    \
  }                                                                          \
  else if (VAL1 && VAL2 && VAL3) {                                           \
    static const bool NAME1 = true, NAME2 = true, NAME3 = true;              \
    return __VA_ARGS__();                                                    \
  }

#define TVDCN_DISPATCH_CONDITION(ARG1, ...)                 \
  [&] {                                                     \
    TVDCN_PRIVATE_OPTION(ARG1, ARG1, __VA_ARGS__)           \
  }()

#define TVDCN_DISPATCH_CONDITION2(ARG1, ARG2, ...)                  \
  [&] {                                                             \
    TVDCN_PRIVATE_OPTIONS2(ARG1, ARG1, ARG2, ARG2, __VA_ARGS__)     \
  }()

#define TVDCN_DISPATCH_CONDITION3(ARG1, ARG2, ARG3, ...)                        \
  [&] {                                                                         \
    TVDCN_PRIVATE_OPTIONS3(ARG1, ARG1, ARG2, ARG2, ARG3, ARG3, __VA_ARGS__)     \
  }()
