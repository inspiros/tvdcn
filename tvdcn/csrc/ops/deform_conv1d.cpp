#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace tvdcn {
    namespace ops {
        at::Tensor deform_conv1d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset,
                const at::optional<at::Tensor> &mask,
                const at::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                at::IntArrayRef dilation,
                int64_t groups) {
            C10_LOG_API_USAGE_ONCE("tvdcn.csrc.ops.deform_conv.deform_conv1d")
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("tvdcn::deform_conv1d", "")
                    .typed<decltype(deform_conv1d)>();
            return op.call(
                    input,
                    weight,
                    offset,
                    mask,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups);
        }

        at::Tensor deform_conv1d_symint(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset,
                const at::optional<at::Tensor> &mask,
                const at::optional<at::Tensor> &bias,
                at::SymIntArrayRef stride,
                at::SymIntArrayRef padding,
                at::SymIntArrayRef dilation,
                c10::SymInt groups) {
            C10_LOG_API_USAGE_ONCE("tvdcn.csrc.ops.deform_conv.deform_conv1d")
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("tvdcn::deform_conv1d", "")
                    .typed<decltype(deform_conv1d_symint)>();
            return op.call(
                    input,
                    weight,
                    offset,
                    mask,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv1d_backward(
                    const at::Tensor &grad_out,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef dilation,
                    int64_t groups) {
                static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("tvdcn::_deform_conv1d_backward", "")
                        .typed<decltype(_deform_conv1d_backward)>();
                return op.call(
                        grad_out,
                        input,
                        weight,
                        offset,
                        mask,
                        bias,
                        stride,
                        padding,
                        dilation,
                        groups);
            }

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv1d_backward_symint(
                    const at::Tensor &grad_out,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::SymIntArrayRef stride,
                    at::SymIntArrayRef padding,
                    at::SymIntArrayRef dilation,
                    c10::SymInt groups) {
                static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("tvdcn::_deform_conv1d_backward", "")
                        .typed<decltype(_deform_conv1d_backward_symint)>();
                return op.call(
                        grad_out,
                        input,
                        weight,
                        offset,
                        mask,
                        bias,
                        stride,
                        padding,
                        dilation,
                        groups);
            }
        }

        TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "tvdcn::deform_conv1d(Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] dilation=1, SymInt groups=1) -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "tvdcn::_deform_conv1d_backward(Tensor grad, Tensor input, Tensor weight, "
                          "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                          "SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] dilation=1, SymInt groups=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"));
        }
    }
}
