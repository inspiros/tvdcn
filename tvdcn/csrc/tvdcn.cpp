#ifdef USE_PYTHON
#include <Python.h>
#endif // USE_PYTHON

#include <torch/script.h>
#include "tvdcn.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
#ifdef USE_PYTHON
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    // extension.py will run on load
    return nullptr;
}
#endif // USE_PYTHON
#endif // _WIN32

namespace tvdcn {
    int64_t cuda_version() {
#ifdef WITH_CUDA
        return CUDA_VERSION;
#else
        return -1;
#endif
    }

    std::string cuda_arch_flags() {
#ifdef WITH_CUDA
#ifdef CUDA_ARCH_FLAGS
        static const char *flags = C10_STRINGIZE(CUDA_ARCH_FLAGS);
        return flags;
#elifdef TORCH_CUDA_ARCH_LIST
        static const char *flags = C10_STRINGIZE(TORCH_CUDA_ARCH_LIST);
        return flags;
#else
        // TODO: this is just a work around.
        return std::to_string(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
#endif
#else
        return {};
#endif
    }

    TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
        m.def("_cuda_version", &cuda_version);
        m.def("_cuda_arch_flags", &cuda_arch_flags);
    }
}
