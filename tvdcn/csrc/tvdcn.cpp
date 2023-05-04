#include <Python.h>
#include <torch/script.h>
#include "tvdcn.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    // extension.py will run on load
    return nullptr;
}
#endif

int64_t cuda_version() {
#ifdef WITH_CUDA
    return CUDA_VERSION;
#else
    return -1;
#endif
}

static auto registry =
        torch::RegisterOperators()
                .op("tvdcn::_cuda_version", &cuda_version)
                .op("tvdcn::deform_conv1d", &deform_conv1d)
                .op("tvdcn::deform_conv2d", &deform_conv2d)
                .op("tvdcn::deform_conv3d", &deform_conv3d)
                .op("tvdcn::deform_conv_transpose1d", &deform_conv_transpose1d)
                .op("tvdcn::deform_conv_transpose2d", &deform_conv_transpose2d)
                .op("tvdcn::deform_conv_transpose3d", &deform_conv_transpose3d);
