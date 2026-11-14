#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <optional>
#include <vector>
#include "../macros.h"

namespace tvdcn::ops {
// Calls the registered operator, including Python autograd/autocast when loaded.
inline torch::stable::Tensor deform_conv_transpose1d(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& weight,
    const std::optional<torch::stable::Tensor>& offset = {},
    const std::optional<torch::stable::Tensor>& mask = {},
    const std::optional<torch::stable::Tensor>& bias = {},
    torch::headeronly::IntHeaderOnlyArrayRef stride = {1},
    torch::headeronly::IntHeaderOnlyArrayRef padding = {0},
    torch::headeronly::IntHeaderOnlyArrayRef output_padding = {0},
    torch::headeronly::IntHeaderOnlyArrayRef dilation = {1},
    int64_t groups = 1) {
    StableIValue stack[] = {
        torch::stable::detail::from(input), torch::stable::detail::from(weight),
        torch::stable::detail::from(offset), torch::stable::detail::from(mask),
        torch::stable::detail::from(bias), torch::stable::detail::from(stride),
        torch::stable::detail::from(padding),
        torch::stable::detail::from(output_padding), torch::stable::detail::from(dilation),
        torch::stable::detail::from(groups)};
    STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
        "tvdcn::deform_conv_transpose1d", "", stack, TORCH_ABI_VERSION));
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}
} // namespace tvdcn::ops
