#pragma once

#include <ATen/Tensor.h>

#include "../macros.h"

namespace tvdcn {
    namespace ops {
        TVDCN_API at::Tensor deform_conv2d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset = {},
                const at::optional<at::Tensor> &mask = {},
                const at::optional<at::Tensor> &bias = {},
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                at::IntArrayRef dilation = 1,
                int64_t groups = 1);

        TVDCN_API at::Tensor deform_conv2d_symint(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset = {},
                const at::optional<at::Tensor> &mask = {},
                const at::optional<at::Tensor> &bias = {},
                at::SymIntArrayRef stride = c10::SymInt(1),
                at::SymIntArrayRef padding = c10::SymInt(0),
                at::SymIntArrayRef dilation = c10::SymInt(1),
                c10::SymInt groups = 1);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef dilation,
                    int64_t groups);

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv2d_backward_symint(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::SymIntArrayRef stride,
                    at::SymIntArrayRef padding,
                    at::SymIntArrayRef dilation,
                    c10::SymInt groups);
        }
    }
}
