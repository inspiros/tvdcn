#pragma once

#include <ATen/ATen.h>

namespace tvdcn {
    namespace ops {
        void arr2col(
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                int64_t in_channels,
                int64_t width,
                int64_t weight_w,
                int64_t stride_w,
                int64_t pad_w,
                int64_t dilation_w,
                int64_t out_w,
                int64_t batch_sz,
                int64_t offset_groups,
                int64_t mask_groups,
                bool deformable,
                bool modulated,
                at::Tensor &columns);

        void col2arr(
                const at::Tensor &columns,
                const at::Tensor &offset,
                const at::Tensor &mask,
                int64_t in_channels,
                int64_t width,
                int64_t weight_w,
                int64_t stride_w,
                int64_t pad_w,
                int64_t dilation_w,
                int64_t out_w,
                int64_t batch_sz,
                int64_t offset_groups,
                int64_t mask_groups,
                bool deformable,
                bool modulated,
                at::Tensor &grad_input);

        void deform_conv1d_compute_grad_offset(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                int64_t in_channels,
                int64_t width,
                int64_t weight_w,
                int64_t stride_w,
                int64_t pad_w,
                int64_t dilation_w,
                int64_t out_w,
                int64_t batch_sz,
                int64_t offset_groups,
                int64_t mask_groups,
                bool deformable,
                bool modulated,
                at::Tensor &grad_offset);

        void deform_conv1d_compute_grad_mask(
                const at::Tensor &columns,
                const at::Tensor &input,
                const at::Tensor &offset,
                int64_t in_channels,
                int64_t width,
                int64_t weight_w,
                int64_t stride_w,
                int64_t pad_w,
                int64_t dilation_w,
                int64_t out_w,
                int64_t batch_sz,
                int64_t offset_groups,
                int64_t mask_groups,
                bool deformable,
                bool modulated,
                at::Tensor &grad_mask);
    }
}
