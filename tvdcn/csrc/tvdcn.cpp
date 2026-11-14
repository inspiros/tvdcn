#ifdef USE_PYTHON
#include <Python.h>
#endif // USE_PYTHON

#include <torch/script.h>
#include "tvdcn.h"

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
        auto op = c10::Dispatcher::singleton().findSchema({"tvdcn::_cuda_build_version", ""});
        return op ? op->typed<int64_t()>().call() : -1;
    }

    std::string cuda_arch_flags() {
        auto op = c10::Dispatcher::singleton().findSchema({"tvdcn::_cuda_build_arch_flags", ""});
        return op ? op->typed<std::string()>().call() : std::string{};
    }

    TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
        m.def("_cuda_version", &cuda_version);
        m.def("_cuda_arch_flags", &cuda_arch_flags);
    }
}
