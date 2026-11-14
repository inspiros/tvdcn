#include <torch/csrc/stable/library.h>

namespace tvdcn::ops {
        STABLE_TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
            m.def((
                          "tvdcn::deform_conv_transpose3d(Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, "
                          "SymInt[3] dilation=1, SymInt groups=1) -> Tensor"));
            m.def((
                          "tvdcn::_deform_conv_transpose3d_backward(Tensor grad, Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, "
                          "SymInt[3] dilation=1, SymInt groups=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"));
        }
}
