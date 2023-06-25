#pragma once

#include <ATen/Tensor.h>
#include "../macros.h"

namespace tvdcn {
    namespace ops {
        at::Tensor deform_conv1d_forward(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &bias,
                int stride,
                int padding,
                int dilation,
                int groups,
                int offset_groups,
                int mask_groups,
                bool deformable,
                bool modulated);

        at::Tensor deform_conv2d_forward(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &bias,
                const std::pair<int, int> &stride,
                const std::pair<int, int> &padding,
                const std::pair<int, int> &dilation,
                int groups,
                int offset_groups,
                int mask_groups,
                bool deformable,
                bool modulated);

        at::Tensor deform_conv3d_forward(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &bias,
                const std::tuple<int, int, int> &stride,
                const std::tuple<int, int, int> &padding,
                const std::tuple<int, int, int> &dilation,
                int groups,
                int offset_groups,
                int mask_groups,
                bool deformable,
                bool modulated);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv1d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const at::Tensor &bias,
                    int stride,
                    int padding,
                    int dilation,
                    int groups,
                    int offset_groups,
                    int mask_groups,
                    bool deformable,
                    bool modulated);

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const at::Tensor &bias,
                    const std::pair<int, int> &stride,
                    const std::pair<int, int> &padding,
                    const std::pair<int, int> &dilation,
                    int groups,
                    int offset_groups,
                    int mask_groups,
                    bool deformable,
                    bool modulated);

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv3d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const at::Tensor &bias,
                    const std::tuple<int, int, int> &stride,
                    const std::tuple<int, int, int> &padding,
                    const std::tuple<int, int, int> &dilation,
                    int groups,
                    int offset_groups,
                    int mask_groups,
                    bool deformable,
                    bool modulated);
        }

        VISION_API at::Tensor deform_conv1d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &bias,
                int64_t stride,
                int64_t pad,
                int64_t dilation,
                int64_t groups,
                int64_t offset_groups,
                int64_t mask_groups,
                bool deformable,
                bool modulated);

        VISION_API at::Tensor deform_conv2d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &bias,
                int64_t stride_h,
                int64_t stride_w,
                int64_t pad_h,
                int64_t pad_w,
                int64_t dilation_h,
                int64_t dilation_w,
                int64_t groups,
                int64_t offset_groups,
                int64_t mask_groups,
                bool deformable,
                bool modulated);

        VISION_API at::Tensor deform_conv3d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &bias,
                int64_t stride_d,
                int64_t stride_h,
                int64_t stride_w,
                int64_t pad_d,
                int64_t pad_h,
                int64_t pad_w,
                int64_t dilation_d,
                int64_t dilation_h,
                int64_t dilation_w,
                int64_t groups,
                int64_t offset_groups,
                int64_t mask_groups,
                bool deformable,
                bool modulated);
    }
}
