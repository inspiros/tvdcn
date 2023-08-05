#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace tvdcn {
    namespace ops {
        at::Tensor deform_conv_transpose2d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset,
                const at::optional<at::Tensor> &mask,
                const at::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                at::IntArrayRef output_padding,
                at::IntArrayRef dilation,
                int64_t groups) {
            C10_LOG_API_USAGE_ONCE("tvdcn.csrc.ops.deform_conv_transpose.deform_conv_transpose2d")
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("tvdcn::deform_conv_transpose2d", "")
                    .typed<decltype(deform_conv_transpose2d)>();
            return op.call(
                    input,
                    weight,
                    offset,
                    mask,
                    bias,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    groups);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv_transpose2d_backward(
                    const at::Tensor &grad_out,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef output_padding,
                    at::IntArrayRef dilation,
                    int64_t groups) {
                static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("tvdcn::_deform_conv_transpose2d_backward", "")
                        .typed<decltype(_deform_conv_transpose2d_backward)>();
                return op.call(
                        grad_out,
                        input,
                        weight,
                        offset,
                        mask,
                        bias,
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups);
            }
        }

        TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "tvdcn::deform_conv_transpose2d(Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "int[2] stride=1, int[2] padding=0, int[2] output_padding=0, "
                          "int[2] dilation=1, int groups=1) -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "tvdcn::_deform_conv_transpose2d_backward(Tensor grad, Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "int[2] stride=1, int[2] padding=0, int[2] output_padding=0, "
                          "int[2] dilation=1, int groups=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"));
        }
    }
}
