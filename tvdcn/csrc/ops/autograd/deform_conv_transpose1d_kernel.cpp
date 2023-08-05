#include <torch/autograd.h>
#include <torch/types.h>

#include "../deform_conv_transpose.h"

namespace tvdcn {
    namespace ops {
        namespace {
            class DeformConvTranspose1dFunction
                    : public torch::autograd::Function<DeformConvTranspose1dFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &input,
                        const torch::autograd::Variable &weight,
                        const at::optional<torch::autograd::Variable> &offset,
                        const at::optional<torch::autograd::Variable> &mask,
                        const at::optional<torch::autograd::Variable> &bias,
                        at::IntArrayRef stride,
                        at::IntArrayRef padding,
                        at::IntArrayRef output_padding,
                        at::IntArrayRef dilation,
                        int64_t groups) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = deform_conv_transpose1d(
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

                    ctx->save_for_backward({input,
                                            weight,
                                            offset.value_or(at::Tensor()),
                                            mask.value_or(at::Tensor()),
                                            bias.value_or(at::Tensor())});
                    ctx->saved_data["stride"] = stride.vec();
                    ctx->saved_data["padding"] = padding.vec();
                    ctx->saved_data["output_padding"] = output_padding.vec();
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

                    auto stride = ctx->saved_data["stride"].toIntVector();
                    auto padding = ctx->saved_data["padding"].toIntVector();
                    auto output_padding = ctx->saved_data["output_padding"].toIntVector();
                    auto dilation = ctx->saved_data["dilation"].toIntVector();
                    auto groups = ctx->saved_data["groups"].toInt();

                    auto grads = detail::_deform_conv_transpose1d_backward(
                            grad_output[0],
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
                            torch::autograd::Variable(),
                    };
                }
            };

            at::Tensor deform_conv_transpose1d_autograd(
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
                auto result = DeformConvTranspose1dFunction::apply(
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
                return result;
            }
        }

        TORCH_LIBRARY_IMPL(tvdcn, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("tvdcn::deform_conv_transpose1d"),
                    TORCH_FN(deform_conv_transpose1d_autograd));
        }
    }
}
