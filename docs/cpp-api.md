# C++ API

The headers in [tvdcn/csrc/ops](../tvdcn/csrc/ops) provide header-inline `tvdcn::ops::deform_conv1d/2d/3d` and `deform_conv_transpose1d/2d/3d` wrappers. They take `torch::stable::Tensor` values, optional tensors and spatial array arguments, and dispatch to the registered tvdcn operators. Include the appropriate `ops/deform_conv2d.h` header and make the tvdcn native library available and loaded before calling the wrapper. Python-side autograd and autocast registrations apply when loaded.

The [C++ smoke test](../tests/packaging/cpp_api/smoke.cpp) shows all six wrappers and stable tensor construction. Its [packaging test](../tests/packaging/test_cpp_api.py) demonstrates compiling and loading against the installed extension.
