#ifdef USE_PYTHON
#include <Python.h>
#endif
#include <torch/csrc/stable/library.h>
#include "macros.h"
#include <string>

#if defined(_WIN32) && defined(USE_PYTHON)
PyMODINIT_FUNC PyInit__C(void) {
    return nullptr;
}
#endif

namespace tvdcn {
namespace {
int64_t build_version = -1;
std::string build_arch_flags;
void set_cuda_metadata(int64_t version, const std::string& arch_flags) {
    build_version = version;
    build_arch_flags = arch_flags;
}
}
// Defaults remain usable when the optional CUDA module is absent or unloadable.
TVDCN_API int64_t cuda_version() { return build_version; }
TVDCN_API std::string cuda_arch_flags() { return build_arch_flags; }

STABLE_TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
    m.def("_cuda_version() -> int");
    m.def("_cuda_arch_flags() -> str");
    m.def("_set_cuda_metadata(int version, str arch_flags) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(tvdcn, CompositeExplicitAutograd, m) {
    m.impl("_cuda_version", TORCH_BOX(&cuda_version));
    m.impl("_cuda_arch_flags", TORCH_BOX(&cuda_arch_flags));
    m.impl("_set_cuda_metadata", TORCH_BOX(&set_cuda_metadata));
}
}
