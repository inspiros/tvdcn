#include <torch/autograd.h>
#include <torch/types.h>

#include "../deform_conv2d.h"

namespace tvdcn {
    namespace ops {
        namespace {
            class DeformConv2dFunction
                    : public torch::autograd::Function<DeformConv2dFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &input,
                        const torch::autograd::Variable &weight,
                        const at::optional<torch::autograd::Variable> &offset,
                        const at::optional<torch::autograd::Variable> &mask,
                        const at::optional<torch::autograd::Variable> &bias,
                        at::SymIntArrayRef stride,
                        at::SymIntArrayRef padding,
                        at::SymIntArrayRef dilation,
                        c10::SymInt groups) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = deform_conv2d_symint(
                            input,
                            weight,
                            offset,
                            mask,
                            bias,
                            stride,
                            padding,
                            dilation,
                            groups);

                    ctx->save_for_backward({input,
                                            weight,
                                            offset.value_or(at::Tensor()),
                                            mask.value_or(at::Tensor()),
                                            bias.value_or(at::Tensor())});
                    ctx->saved_data["stride"] = stride.vec();
                    ctx->saved_data["padding"] = padding.vec();
                    ctx->saved_data["dilation"] = dilation.vec();
                    ctx->saved_data["groups"] = groups;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto input = saved[0];
                    auto weight = saved[1];
                    auto offset = saved[2];
                    auto mask = saved[3];
                    auto bias = saved[4];

                    auto stride = ctx->saved_data["stride"].toSymIntVector();
                    auto padding = ctx->saved_data["padding"].toSymIntVector();
                    auto dilation = ctx->saved_data["dilation"].toSymIntVector();
                    auto groups = ctx->saved_data["groups"].toSymInt();

                    auto grads = detail::_deform_conv2d_backward_symint(
                            grad_output[0],
                            input,
                            weight,
                            offset,
                            mask,
                            bias,
                            stride,
                            padding,
                            dilation,
                            groups);

                    auto grad_input = std::get<0>(grads);
                    auto grad_weight = std::get<1>(grads);
                    auto grad_offset = std::get<2>(grads);
                    auto grad_mask = std::get<3>(grads);
                    auto grad_bias = std::get<4>(grads);

                    return {
                            grad_input,
                            grad_weight,
                            grad_offset,
                            grad_mask,
                            grad_bias,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            // TODO: Update when torchvision guys found an easier way
            class DeformConv2dBackwardFunction
                    : public torch::autograd::Function<DeformConv2dBackwardFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &grad,
                        const torch::autograd::Variable &input,
                        const torch::autograd::Variable &weight,
                        const at::optional<torch::autograd::Variable> &offset,
                        const at::optional<torch::autograd::Variable> &mask,
                        const at::optional<torch::autograd::Variable> &bias,
                        at::SymIntArrayRef stride,
                        at::SymIntArrayRef padding,
                        at::SymIntArrayRef dilation,
                        c10::SymInt groups) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto grads = detail::_deform_conv2d_backward_symint(
                            grad,
                            input,
                            weight,
                            offset,
                            mask,
                            bias,
                            stride,
                            padding,
                            dilation,
                            groups);

                    auto grad_input = std::get<0>(grads);
                    auto grad_weight = std::get<1>(grads);
                    auto grad_offset = std::get<2>(grads);
                    auto grad_mask = std::get<3>(grads);
                    auto grad_bias = std::get<4>(grads);

                    return {
                        grad_input,
                        grad_weight,
                        grad_offset,
                        grad_mask,
                        grad_bias
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    TORCH_CHECK(0, "double backwards on deform_conv2d not supported");
                }
            };

            at::Tensor deform_conv2d_autograd(
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::SymIntArrayRef stride,
                    at::SymIntArrayRef padding,
                    at::SymIntArrayRef dilation,
                    c10::SymInt groups) {
                auto result = DeformConv2dFunction::apply(
                        input,
                        weight,
                        offset,
                        mask,
                        bias,
                        stride,
                        padding,
                        dilation,
                        groups);
                return result;
            }

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            deform_conv2d_backward_autograd(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::SymIntArrayRef stride,
                    at::SymIntArrayRef padding,
                    at::SymIntArrayRef dilation,
                    c10::SymInt groups) {
                auto result = DeformConv2dBackwardFunction::apply(
                        grad,
                        input,
                        weight,
                        offset,
                        mask,
                        bias,
                        stride,
                        padding,
                        dilation,
                        groups);
                return std::make_tuple(result[0], result[1], result[2], result[3], result[4]);
            }
        }

        TORCH_LIBRARY_IMPL(tvdcn, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("tvdcn::deform_conv2d"),
                    TORCH_FN(deform_conv2d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("tvdcn::_deform_conv2d_backward"),
                    TORCH_FN(deform_conv2d_backward_autograd));
        }
    }
}
