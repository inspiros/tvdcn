/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

// modified from
// https://github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/deform_conv2d_kernel.cpp

#include "../StableABICompat.h"
#include <torch/csrc/stable/library.h>

#include "deform_conv3d_common_kernels.h"

namespace tvdcn {
    namespace ops {
        using namespace cuda;

        namespace {
            torch::stable::Tensor deform_conv3d_forward_kernel(
                    const torch::stable::Tensor &input,
                    const torch::stable::Tensor &weight,
                    const std::optional<torch::stable::Tensor> &offset,
                    const std::optional<torch::stable::Tensor> &mask,
                    const std::optional<torch::stable::Tensor> &bias,
                    torch::headeronly::IntHeaderOnlyArrayRef stride,
                    torch::headeronly::IntHeaderOnlyArrayRef padding,
                    torch::headeronly::IntHeaderOnlyArrayRef dilation,
                    int64_t groups) {
                bool unbatched = input.dim() == 4;
                bool deformable = offset.has_value() && offset->defined();
                bool modulated = mask.has_value() && mask->defined();
                bool with_bias = bias.has_value() && bias->defined();

                stable_compat::check_tensors(input, weight, offset, mask, bias, true);

                auto input_c = torch::stable::contiguous(input);
                auto weight_c = torch::stable::contiguous(weight);
                auto offset_c = torch::stable::contiguous((deformable ? offset.value() : torch::stable::new_zeros(input, {0})));
                auto mask_c = torch::stable::contiguous((modulated ? mask.value() : torch::stable::new_zeros(input, {0})));
                auto bias_c = torch::stable::contiguous((with_bias ? bias.value() : torch::stable::new_zeros(input, {0})));

                STD_TORCH_CHECK(unbatched || input_c.dim() == 5,
                            "Expected 4D (unbatched) or 5D (batched) input to deform_conv3d, "
                            "but got input of rank: ",
                            input_c.dim())
                STD_TORCH_CHECK(weight_c.dim() == 5,
                            "Expected 5D weight")
                STD_TORCH_CHECK(!deformable ||
                            (unbatched && offset_c.dim() == 4) ||
                            (!unbatched && offset_c.dim() == 5),
                            "Expected 4D (unbatched) or 5D (batched) offset, "
                            "but got offset of rank: ",
                            offset_c.dim())
                STD_TORCH_CHECK(!modulated ||
                            (unbatched && mask_c.dim() == 4) ||
                            (!unbatched && mask_c.dim() == 5),
                            "Expected 4D (unbatched) or 5D (batched) mask, "
                            "but got mask of rank: ",
                            mask_c.dim())
                if (unbatched) {
                    input_c = torch::stable::unsqueeze(input_c, 0);
                    offset_c = torch::stable::unsqueeze(offset_c, 0);
                    mask_c = torch::stable::unsqueeze(mask_c, 0);
                }

                int64_t batch_sz = input_c.size(0);
                int64_t in_channels = input_c.size(1);
                int64_t in_d = input_c.size(2);
                int64_t in_h = input_c.size(3);
                int64_t in_w = input_c.size(4);

                int64_t n_parallel_imgs = stable_compat::get_greatest_divisor_below_bound(batch_sz, stable_compat::kMaxParallelImgs);

                int64_t out_channels = weight_c.size(0);
                int64_t weight_d = weight_c.size(2);
                int64_t weight_h = weight_c.size(3);
                int64_t weight_w = weight_c.size(4);

                auto stride_c = stable_compat::ntuple<3>(stride);
                int64_t stride_d = stride_c[0];
                int64_t stride_h = stride_c[1];
                int64_t stride_w = stride_c[2];

                auto padding_c = stable_compat::ntuple<3>(padding);
                int64_t pad_d = padding_c[0];
                int64_t pad_h = padding_c[1];
                int64_t pad_w = padding_c[2];

                auto dilation_c = stable_compat::ntuple<3>(dilation);
                int64_t dilation_d = dilation_c[0];
                int64_t dilation_h = dilation_c[1];
                int64_t dilation_w = dilation_c[2];

                int64_t out_d = (in_d + 2 * pad_d - (dilation_d * (weight_d - 1) + 1)) / stride_d + 1;
                int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
                int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

                STD_TORCH_CHECK(
                        weight_d > 0 && weight_h > 0 && weight_w > 0,
                        "weight_d: ",
                        weight_d,
                        " weight_h: ",
                        weight_h,
                        " weight_w: ",
                        weight_w)
                STD_TORCH_CHECK(
                        stride_d > 0 && stride_h > 0 && stride_w > 0,
                        " stride_d: ",
                        stride_d,
                        "stride_h: ",
                        stride_h,
                        " stride_w: ",
                        stride_w)
                STD_TORCH_CHECK(
                        pad_d >= 0 && pad_h >= 0 && pad_w >= 0,
                        "pad_d: ",
                        pad_d,
                        " pad_h: ",
                        pad_h,
                        " pad_w: ",
                        pad_w)
                STD_TORCH_CHECK(
                        dilation_d > 0 && dilation_h > 0 && dilation_w > 0,
                        "dilation_d: ",
                        dilation_d,
                        " dilation_h: ",
                        dilation_h,
                        " dilation_w: ",
                        dilation_w)

                STD_TORCH_CHECK(weight_c.size(1) * groups == in_channels)
                STD_TORCH_CHECK(weight_c.size(0) % groups == 0)

                int64_t offset_groups = deformable ? offset_c.size(1) / (3 * weight_d * weight_h * weight_w) : 0;
                int64_t mask_groups = modulated ? mask_c.size(1) / (weight_d * weight_h * weight_w) : 0;

                STD_TORCH_CHECK(
                        !deformable || offset_c.size(0) == input_c.size(0),
                        "invalid batch size of offset")
                STD_TORCH_CHECK(
                        !deformable || offset_groups > 0,
                        "The shape of the offset tensor at dimension 1 is not valid. It should "
                        "be a multiple of 3 * weight.size(2) * weight.size(3) * weight.size(4).\nGot offset.size(1)=",
                        offset_c.size(1),
                        ", while 3 * weight.size(2) * weight.size(3) * weight.size(3)=",
                        3 * weight_d * weight_h * weight_w)
                STD_TORCH_CHECK(!deformable || input_c.size(1) % offset_groups == 0)
                STD_TORCH_CHECK(
                        !deformable || offset_c.size(1) == offset_groups * 3 * weight_d * weight_h * weight_w,
                        "offset.shape[1] is not valid. got: ",
                        offset_c.size(1),
                        " expected: ",
                        offset_groups * 3 * weight_d * weight_h * weight_w)
                STD_TORCH_CHECK(
                        !deformable || (offset_c.size(2) == out_d &&
                                        offset_c.size(3) == out_h &&
                                        offset_c.size(4) == out_w),
                        "offset output dims: (",
                        offset_c.size(2),
                        ", ",
                        offset_c.size(3),
                        ", ",
                        offset_c.size(4),
                        ") - ",
                        "computed output dims: (",
                        out_d,
                        ", ",
                        out_h,
                        ", ",
                        out_w,
                        ")")

                STD_TORCH_CHECK(
                        !modulated || mask_c.size(0) == input_c.size(0),
                        "invalid batch size of mask")
                STD_TORCH_CHECK(
                        !modulated || mask_groups > 0,
                        "The shape of the mask tensor at dimension 1 is not valid. It should "
                        "be a multiple of weight.size(2) * weight.size(3) * weight.size(4).\nGot mask.size(1)=",
                        mask_c.size(1),
                        ", while weight.size(2) * weight.size(3) * weight.size(4)=",
                        weight_d * weight_h * weight_w)
                STD_TORCH_CHECK(!modulated || input_c.size(1) % mask_groups == 0)
                STD_TORCH_CHECK(
                        !modulated || mask_c.size(1) == mask_groups * weight_d * weight_h * weight_w,
                        "mask.shape[1] is not valid. got: ",
                        mask_c.size(1),
                        " expected: ",
                        mask_groups * weight_d * weight_h * weight_w)
                STD_TORCH_CHECK(
                        !modulated || (mask_c.size(2) == out_d &&
                                       mask_c.size(3) == out_h &&
                                       mask_c.size(4) == out_w),
                        "mask output dims: (",
                        mask_c.size(2),
                        ", ",
                        mask_c.size(3),
                        ", ",
                        mask_c.size(4),
                        ") - ",
                        "computed output dims: (",
                        out_d,
                        ", ",
                        out_h,
                        ", ",
                        out_w,
                        ")")

                STD_TORCH_CHECK(
                        out_d > 0 && out_h > 0 && out_w > 0,
                        "Calculated output size too small - out_d: ",
                        out_d,
                        " out_h: ",
                        out_h,
                        " out_w: ",
                        out_w)

                auto output = torch::stable::new_zeros(input_c, {batch_sz, out_channels, out_d, out_h, out_w});

                // Separate batches into blocks
                input_c = torch::stable::view(input_c, {batch_sz / n_parallel_imgs,
                                        n_parallel_imgs,
                                        in_channels,
                                        in_d,
                                        in_h,
                                        in_w});
                if (deformable)
                    offset_c = torch::stable::view(offset_c, {batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              offset_groups,
                                              weight_d,
                                              weight_h,
                                              weight_w,
                                              3,
                                              out_d,
                                              out_h,
                                              out_w});
                else
                    offset_c = torch::stable::view(offset_c, {batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              0, 0, 0, 0, 0, 0, 0, 0});
                if (modulated)
                    mask_c = torch::stable::view(mask_c, {batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          mask_groups,
                                          weight_d,
                                          weight_h,
                                          weight_w,
                                          out_d,
                                          out_h,
                                          out_w});
                else
                    mask_c = torch::stable::view(mask_c, {batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          0, 0, 0, 0, 0, 0, 0});

                output = torch::stable::view(output, {batch_sz / n_parallel_imgs,
                                      n_parallel_imgs,
                                      out_channels,
                                      out_d,
                                      out_h,
                                      out_w});
                auto output_buf = torch::stable::new_zeros(output, {batch_sz / n_parallel_imgs,
                         out_channels,
                         n_parallel_imgs,
                         out_d,
                         out_h,
                         out_w});

                // Separate channels into convolution groups
                output_buf = torch::stable::view(output_buf, {output_buf.size(0),
                                              groups,
                                              output_buf.size(1) / groups,
                                              output_buf.size(2),
                                              output_buf.size(3),
                                              output_buf.size(4),
                                              output_buf.size(5)});
                weight_c = torch::stable::view(weight_c, {groups,
                                          weight_c.size(0) / groups,
                                          weight_c.size(1),
                                          weight_c.size(2),
                                          weight_c.size(3),
                                          weight_c.size(4)});

                // Sample points and perform convolution
                auto columns = torch::stable::new_empty(input_c, {groups,
                                          in_channels * weight_d * weight_h * weight_w / groups,
                                          n_parallel_imgs * out_d * out_h * out_w});
                auto columns_view = torch::stable::view(columns, {in_channels,
                                                  weight_d,
                                                  weight_h,
                                                  weight_w,
                                                  n_parallel_imgs,
                                                  out_d,
                                                  out_h,
                                                  out_w});
                for (int64_t b = 0; b < batch_sz / n_parallel_imgs; b++) {
                    vol2col(
                            torch::stable::select(input_c, 0, b),
                            torch::stable::select(offset_c, 0, b),
                            torch::stable::select(mask_c, 0, b),
                            in_channels,
                            in_d,
                            in_h,
                            in_w,
                            weight_d,
                            weight_h,
                            weight_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            columns_view);

                    for (int64_t g = 0; g < groups; g++) {
                        stable_compat::addmm_<true>(torch::stable::flatten(torch::stable::select(torch::stable::select(output_buf, 0, b), 0, g), 1), torch::stable::flatten(torch::stable::select(weight_c, 0, g), 1), torch::stable::select(columns, 0, g));
                    }
                }

                output_buf = torch::stable::view(output_buf, {batch_sz / n_parallel_imgs,
                                              out_channels,
                                              n_parallel_imgs,
                                              out_d,
                                              out_h,
                                              out_w});
                output_buf = torch::stable::transpose(output_buf, 1, 2);
                torch::stable::copy_(output, output_buf);
                output = torch::stable::view(output, {batch_sz, out_channels, out_d, out_h, out_w});
                if (with_bias)
                    stable_compat::add_<true>(output, torch::stable::view(bias_c, {1, out_channels, 1, 1, 1}));
                if (unbatched)
                    output = torch::stable::squeeze(output, 0);

                return output;
            }

            std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
            deform_conv3d_backward_kernel(
                    const torch::stable::Tensor &grad_out,
                    const torch::stable::Tensor &input,
                    const torch::stable::Tensor &weight,
                    const std::optional<torch::stable::Tensor> &offset,
                    const std::optional<torch::stable::Tensor> &mask,
                    const std::optional<torch::stable::Tensor> &bias,
                    torch::headeronly::IntHeaderOnlyArrayRef stride,
                    torch::headeronly::IntHeaderOnlyArrayRef padding,
                    torch::headeronly::IntHeaderOnlyArrayRef dilation,
                    int64_t groups) {
                bool unbatched = input.dim() == 4;
                bool deformable = offset.has_value() && offset->defined();
                bool modulated = mask.has_value() && mask->defined();
                bool with_bias = bias.has_value() && bias->defined();

                auto grad_out_c = torch::stable::contiguous(grad_out);
                auto input_c = torch::stable::contiguous(input);
                auto weight_c = torch::stable::contiguous(weight);
                auto offset_c = torch::stable::contiguous((deformable ? offset.value() : torch::stable::new_zeros(input, {0})));
                auto mask_c = torch::stable::contiguous((modulated ? mask.value() : torch::stable::new_zeros(input, {0})));
                auto bias_c = torch::stable::contiguous((with_bias ? bias.value() : torch::stable::new_zeros(input, {0})));

                auto grad_input = torch::stable::new_zeros(input_c, input_c.sizes());
                auto grad_weight = torch::stable::new_zeros(weight_c, weight_c.sizes());
                auto grad_offset = torch::stable::new_zeros(offset_c, offset_c.sizes());
                auto grad_mask = torch::stable::new_zeros(mask_c, mask_c.sizes());
                auto grad_bias = with_bias ? torch::stable::sum(grad_out_c,
                        unbatched ? torch::headeronly::IntHeaderOnlyArrayRef({1, 2, 3})
                                  : torch::headeronly::IntHeaderOnlyArrayRef({0, 2, 3, 4})) : torch::stable::new_zeros(input, {0});

                if (unbatched) {
                    grad_out_c = torch::stable::unsqueeze(grad_out_c, 0);
                    input_c = torch::stable::unsqueeze(input_c, 0);
                    grad_input = torch::stable::unsqueeze(grad_input, 0);
                    offset_c = torch::stable::unsqueeze(offset_c, 0);
                    grad_offset = torch::stable::unsqueeze(grad_offset, 0);
                    mask_c = torch::stable::unsqueeze(mask_c, 0);
                    grad_mask = torch::stable::unsqueeze(grad_mask, 0);
                }

                int64_t batch_sz = input_c.size(0);
                int64_t in_channels = input_c.size(1);
                int64_t in_d = input_c.size(2);
                int64_t in_h = input_c.size(3);
                int64_t in_w = input_c.size(4);

                int64_t n_parallel_imgs = stable_compat::get_greatest_divisor_below_bound(batch_sz, stable_compat::kMaxParallelImgs);

                int64_t out_channels = weight_c.size(0);
                int64_t weight_d = weight_c.size(2);
                int64_t weight_h = weight_c.size(3);
                int64_t weight_w = weight_c.size(4);

                auto stride_c = stable_compat::ntuple<3>(stride);
                int64_t stride_d = stride_c[0];
                int64_t stride_h = stride_c[1];
                int64_t stride_w = stride_c[2];

                auto padding_c = stable_compat::ntuple<3>(padding);
                int64_t pad_d = padding_c[0];
                int64_t pad_h = padding_c[1];
                int64_t pad_w = padding_c[2];

                auto dilation_c = stable_compat::ntuple<3>(dilation);
                int64_t dilation_d = dilation_c[0];
                int64_t dilation_h = dilation_c[1];
                int64_t dilation_w = dilation_c[2];

                int64_t out_d = (in_d + 2 * pad_d - (dilation_d * (weight_d - 1) + 1)) / stride_d + 1;
                int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
                int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

                int64_t offset_groups = deformable ? offset_c.size(1) / (3 * weight_d * weight_h * weight_w) : 0;
                int64_t mask_groups = modulated ? mask_c.size(1) / (weight_d * weight_h * weight_w) : 0;

                // Separate into blocks
                input_c = torch::stable::view(input_c, {batch_sz / n_parallel_imgs,
                                        n_parallel_imgs,
                                        in_channels,
                                        in_d,
                                        in_h,
                                        in_w});
                grad_input = torch::stable::view(grad_input, input_c.sizes());
                if (deformable)
                    offset_c = torch::stable::view(offset_c, {batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              offset_groups,
                                              weight_d,
                                              weight_h,
                                              weight_w,
                                              3,
                                              out_d,
                                              out_h,
                                              out_w});
                else
                    offset_c = torch::stable::view(offset_c, {batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              0, 0, 0, 0, 0, 0, 0, 0});
                grad_offset = torch::stable::view(grad_offset, offset_c.sizes());
                if (modulated)
                    mask_c = torch::stable::view(mask_c, {batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          mask_groups,
                                          weight_d,
                                          weight_h,
                                          weight_w,
                                          out_d,
                                          out_h,
                                          out_w});
                else
                    mask_c = torch::stable::view(mask_c, {batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          0, 0, 0, 0, 0, 0, 0});
                grad_mask = torch::stable::view(grad_mask, mask_c.sizes());

                // Separate channels into convolution groups
                grad_out_c = torch::stable::contiguous(torch::stable::permute(torch::stable::view(grad_out_c, {batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              groups,
                                              out_channels / groups,
                                              out_d,
                                              out_h,
                                              out_w}), {0, 2, 3, 1, 4, 5, 6}));

                weight_c = torch::stable::view(weight_c, {groups,
                                          out_channels / groups,
                                          in_channels / groups,
                                          weight_d,
                                          weight_h,
                                          weight_w});
                grad_weight = torch::stable::view(grad_weight, weight_c.sizes());

                auto columns = torch::stable::new_empty(input_c, {groups,
                                          in_channels * weight_d * weight_h * weight_w / groups,
                                          n_parallel_imgs * out_d * out_h * out_w});
                auto columns_view = torch::stable::view(columns, {in_channels,
                                                  weight_d,
                                                  weight_h,
                                                  weight_w,
                                                  n_parallel_imgs,
                                                  out_d,
                                                  out_h,
                                                  out_w});
                for (int64_t b = 0; b < batch_sz / n_parallel_imgs; b++) {
                    torch::stable::zero_(columns);
                    for (int64_t g = 0; g < groups; g++) {
                        stable_compat::addmm_<true>(torch::stable::select(columns, 0, g), torch::stable::transpose(torch::stable::flatten(torch::stable::select(weight_c, 0, g), 1), 0, 1), torch::stable::flatten(torch::stable::select(torch::stable::select(grad_out_c, 0, b), 0, g), 1));
                    }

                    auto grad_offset_b = torch::stable::select(grad_offset, 0, b);
                    deform_conv3d_compute_grad_offset(
                            columns_view,
                            torch::stable::select(input_c, 0, b),
                            torch::stable::select(offset_c, 0, b),
                            torch::stable::select(mask_c, 0, b),
                            in_channels,
                            in_d,
                            in_h,
                            in_w,
                            weight_d,
                            weight_h,
                            weight_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            grad_offset_b);

                    auto grad_mask_b = torch::stable::select(grad_mask, 0, b);
                    deform_conv3d_compute_grad_mask(
                            columns_view,
                            torch::stable::select(input_c, 0, b),
                            torch::stable::select(offset_c, 0, b),
                            in_channels,
                            in_d,
                            in_h,
                            in_w,
                            weight_d,
                            weight_h,
                            weight_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            grad_mask_b);

                    auto grad_input_b = torch::stable::select(grad_input, 0, b);
                    col2vol(
                            columns_view,
                            torch::stable::select(offset_c, 0, b),
                            torch::stable::select(mask_c, 0, b),
                            in_channels,
                            in_d,
                            in_h,
                            in_w,
                            weight_d,
                            weight_h,
                            weight_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            grad_input_b);

                    vol2col(
                            torch::stable::select(input_c, 0, b),
                            torch::stable::select(offset_c, 0, b),
                            torch::stable::select(mask_c, 0, b),
                            in_channels,
                            in_d,
                            in_h,
                            in_w,
                            weight_d,
                            weight_h,
                            weight_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            out_d,
                            out_h,
                            out_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            columns_view);

                    for (int64_t g = 0; g < groups; g++) {
                        stable_compat::addmm_<true>(torch::stable::flatten(torch::stable::select(grad_weight, 0, g), 1), torch::stable::flatten(torch::stable::select(torch::stable::select(grad_out_c, 0, b), 0, g), 1), torch::stable::transpose(torch::stable::select(columns, 0, g), 1, 0));
                    }
                }

                grad_input = torch::stable::view(grad_input, input.sizes());
                grad_weight = torch::stable::view(grad_weight, weight.sizes());
                grad_offset = deformable ? torch::stable::view(grad_offset, offset.value().sizes()) : torch::stable::new_zeros(input, {0});
                grad_mask = modulated ? torch::stable::view(grad_mask, mask.value().sizes()) : torch::stable::new_zeros(input, {0});
                // The views above already restore the original unbatched ranks.
                // Squeezing again would remove a singleton input channel.

                return std::make_tuple(grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
            }
        }

        STABLE_TORCH_LIBRARY_IMPL(tvdcn, CUDA, m) {
            m.impl(
                    "deform_conv3d",
                    TORCH_BOX(&deform_conv3d_forward_kernel));
            m.impl(
                    "_deform_conv3d_backward",
                    TORCH_BOX(&deform_conv3d_backward_kernel));
        }
    }
}
