// CUDA metadata belongs to the optional module, so _C never needs CUDA libraries.
#ifdef USE_PYTHON
#include <Python.h>
#endif
#include <torch/library.h>
#include <cuda.h>
#include <string>

#if defined(_WIN32) && defined(USE_PYTHON)
PyMODINIT_FUNC PyInit__C_cuda(void) {
    return nullptr;
}
#endif

namespace {
int64_t cuda_build_version() {
    return CUDA_VERSION;
}

std::string cuda_build_arch_flags() {
    return TVDCN_CUDA_ARCH_FLAGS;
}
}

TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
    m.def("_cuda_build_version", &cuda_build_version);
    m.def("_cuda_build_arch_flags", &cuda_build_arch_flags);
}
