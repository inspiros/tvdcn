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
}

static auto registry =
        torch::RegisterOperators()
                .op("tvdcn::_cuda_version", &tvdcn::cuda_version)
                .op("tvdcn::deform_conv1d", &tvdcn::ops::deform_conv1d)
                .op("tvdcn::deform_conv2d", &tvdcn::ops::deform_conv2d)
                .op("tvdcn::deform_conv3d", &tvdcn::ops::deform_conv3d)
                .op("tvdcn::deform_conv_transpose1d", &tvdcn::ops::deform_conv_transpose1d)
                .op("tvdcn::deform_conv_transpose2d", &tvdcn::ops::deform_conv_transpose2d)
                .op("tvdcn::deform_conv_transpose3d", &tvdcn::ops::deform_conv_transpose3d);
