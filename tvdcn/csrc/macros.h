#pragma once

#if defined(_WIN32) && !defined(TVDCN_BUILD_STATIC_LIBS)
#if defined(tvdcn_EXPORTS)
#define TVDCN_API __declspec(dllexport)
#else
#define TVDCN_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define TVDCN_API __attribute__((visibility("default")))
#else
#define TVDCN_API
#endif
