#include <Python.h>
#include <torch/csrc/stable/library.h>

namespace {
int64_t increment(int64_t value) {
    return value + 1;
}
}

STABLE_TORCH_LIBRARY(tvdcn_packaging_test, m) {
    m.def("increment(int value) -> int");
    m.impl("increment", TORCH_BOX(&increment));
}

// Exercise actual limited-API CPython imports as well as the torch C shim.
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_stable_ext", nullptr, -1, nullptr,
};
PyMODINIT_FUNC PyInit__stable_ext() {
    return PyModule_Create(&module);
}
