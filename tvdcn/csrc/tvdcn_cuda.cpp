// CUDA metadata belongs to the optional module, so _C never needs CUDA libraries.
#ifdef USE_PYTHON
#include <Python.h>
#endif
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <cuda.h>
#include "tvdcn_cuda_arch_flags.h"
#include <string>

#if defined(_WIN32) && defined(USE_PYTHON)
PyMODINIT_FUNC PyInit__C_cuda(void) {
    return nullptr;
}
#endif

namespace {
// extension.py loads _C first. Publish metadata through its stable dispatcher
// op rather than a cross-DSO C++ symbol (torch loads these libraries locally).
struct RegisterCudaMetadata {
    RegisterCudaMetadata() {
        StableIValue stack[] = {
            torch::stable::detail::from(int64_t{CUDA_VERSION}),
            torch::stable::detail::from(std::string(TVDCN_CUDA_ARCH_FLAGS))};
        STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
            "tvdcn::_set_cuda_metadata", "", stack, TORCH_ABI_VERSION));
    }
} register_cuda_metadata;
}
