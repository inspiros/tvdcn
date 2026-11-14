#pragma once

#include "../StableABICompat.h"

namespace tvdcn {
    namespace ops {
        namespace cuda {
            void arr2col(
                    const torch::stable::Tensor &input,
                    const torch::stable::Tensor &offset,
                    const torch::stable::Tensor &mask,
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
                    torch::stable::Tensor &columns);

            void col2arr(
                    const torch::stable::Tensor &columns,
                    const torch::stable::Tensor &offset,
                    const torch::stable::Tensor &mask,
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
                    torch::stable::Tensor &grad_input);

            void deform_conv1d_compute_grad_offset(
                    const torch::stable::Tensor &columns,
                    const torch::stable::Tensor &input,
                    const torch::stable::Tensor &offset,
                    const torch::stable::Tensor &mask,
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
                    torch::stable::Tensor &grad_offset);

            void deform_conv1d_compute_grad_mask(
                    const torch::stable::Tensor &columns,
                    const torch::stable::Tensor &input,
                    const torch::stable::Tensor &offset,
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
                    torch::stable::Tensor &grad_mask);
        }
    }
}
