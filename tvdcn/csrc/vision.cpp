#include <Python.h>
#include <torch/script.h>
#include "DeformConv.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    // extension.py will run on load
    return NULL;
}
#endif

static auto registry =
    torch::RegisterOperators()
        // pairwise metrics
        .op("tvdcn::deform_conv1d", &deform_conv1d)
        .op("tvdcn::deform_conv2d", &deform_conv2d)
        .op("tvdcn::deform_conv3d", &deform_conv3d)
        ;
