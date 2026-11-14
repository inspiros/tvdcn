#include <torch/csrc/stable/library.h>

namespace tvdcn::ops {
        STABLE_TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
            m.def((
                          "tvdcn::deform_conv2d(Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor"));
            m.def((
                          "tvdcn::_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"));
        }
}
