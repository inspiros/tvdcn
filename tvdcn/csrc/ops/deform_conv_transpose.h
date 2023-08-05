#pragma once

#include <ATen/Tensor.h>

#include "../macros.h"

namespace tvdcn {
    namespace ops {
        TVDCN_API at::Tensor deform_conv_transpose1d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset = {},
                const at::optional<at::Tensor> &mask = {},
                const at::optional<at::Tensor> &bias = {},
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                at::IntArrayRef output_padding = 0,
                at::IntArrayRef dilation = 1,
                int64_t groups = 1);

        TVDCN_API at::Tensor deform_conv_transpose2d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset = {},
                const at::optional<at::Tensor> &mask = {},
                const at::optional<at::Tensor> &bias = {},
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                at::IntArrayRef output_padding = 0,
                at::IntArrayRef dilation = 1,
                int64_t groups = 1);

        TVDCN_API at::Tensor deform_conv_transpose3d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset = {},
                const at::optional<at::Tensor> &mask = {},
                const at::optional<at::Tensor> &bias = {},
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                at::IntArrayRef output_padding = 0,
                at::IntArrayRef dilation = 1,
                int64_t groups = 1);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv_transpose1d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef output_padding,
                    at::IntArrayRef dilation,
                    int64_t groups);

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv_transpose2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef output_padding,
                    at::IntArrayRef dilation,
                    int64_t groups);

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv_transpose3d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef output_padding,
                    at::IntArrayRef dilation,
                    int64_t groups);
        }
    }
}
