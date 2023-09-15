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

#include <ATen/ATen.h>
#include <torch/library.h>

#include "deform_conv3d_common_kernels.h"
#include "../utils/parallel_helpers.h"
#include "../utils/tensor_utils.h"

namespace tvdcn {
    namespace ops {
        using namespace cuda;

        namespace {
            at::Tensor deform_conv3d_forward_kernel(
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::optional<at::Tensor> &offset,
                    const at::optional<at::Tensor> &mask,
                    const at::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef dilation,
                    int64_t groups) {
                bool unbatched = input.ndimension() == 4;
                bool deformable = offset.has_value() && offset->defined();
                bool modulated = mask.has_value() && mask->defined();
                bool with_bias = bias.has_value() && bias->defined();

                at::CheckedFrom c = "deform_conv3d_forward_kernel";
                std::vector<at::TensorArg> args;
                args.reserve(5);
                args.emplace_back(at::TensorArg(input, "input", 1));
                args.emplace_back(at::TensorArg(weight, "weight", 2));
                if (deformable)
                    args.emplace_back(at::TensorArg(offset.value(), "offset", 3));
                if (modulated)
                    args.emplace_back(at::TensorArg(mask.value(), "mask", 4));
                if (with_bias)
                    args.emplace_back(at::TensorArg(bias.value(), "bias", 5));
                at::checkAllSameGPUExceptUndefined(c, args);
                at::checkAllSameTypeExceptUndefined(c, args);

                auto input_c = input.contiguous();
                auto weight_c = weight.contiguous();
                auto offset_c = (deformable ? offset.value() : input.new_zeros(0)).contiguous();
                auto mask_c = (modulated ? mask.value() : input.new_zeros(0)).contiguous();
                auto bias_c = (with_bias ? bias.value() : input.new_zeros(0)).contiguous();

                TORCH_CHECK(unbatched || input_c.ndimension() == 5,
                            "Expected 4D (unbatched) or 5D (batched) input to deform_conv3d, "
                            "but got input of size: ",
                            input_c.sizes())
                TORCH_CHECK(weight_c.ndimension() == 5,
                            "Expected 5D weight")
                TORCH_CHECK(!deformable ||
                            (unbatched && offset_c.ndimension() == 4) ||
                            (!unbatched && offset_c.ndimension() == 5),
                            "Expected 4D (unbatched) or 5D (batched) offset, "
                            "but got offset of size: ",
                            offset_c.sizes())
                TORCH_CHECK(!modulated ||
                            (unbatched && mask_c.ndimension() == 4) ||
                            (!unbatched && mask_c.ndimension() == 5),
                            "Expected 4D (unbatched) or 5D (batched) mask, "
                            "but got mask of size: ",
                            mask_c.sizes())
                if (unbatched) {
                    input_c = input_c.unsqueeze(0);
                    offset_c = offset_c.unsqueeze(0);
                    mask_c = mask_c.unsqueeze(0);
                }

                int64_t batch_sz = input_c.size(0);
                int64_t in_channels = input_c.size(1);
                int64_t in_d = input_c.size(2);
                int64_t in_h = input_c.size(3);
                int64_t in_w = input_c.size(4);

                int64_t n_parallel_imgs = get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

                int64_t out_channels = weight_c.size(0);
                int64_t weight_d = weight_c.size(2);
                int64_t weight_h = weight_c.size(3);
                int64_t weight_w = weight_c.size(4);

                int64_t stride_d = stride[0];
                int64_t stride_h = stride[1];
                int64_t stride_w = stride[2];

                int64_t pad_d = padding[0];
                int64_t pad_h = padding[1];
                int64_t pad_w = padding[2];

                int64_t dilation_d = dilation[0];
                int64_t dilation_h = dilation[1];
                int64_t dilation_w = dilation[2];

                int64_t out_d = (in_d + 2 * pad_d - (dilation_d * (weight_d - 1) + 1)) / stride_d + 1;
                int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
                int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

                TORCH_CHECK(
                        weight_d > 0 && weight_h > 0 && weight_w > 0,
                        "weight_d: ",
                        weight_d,
                        " weight_h: ",
                        weight_h,
                        " weight_w: ",
                        weight_w)
                TORCH_CHECK(
                        stride_d > 0 && stride_h > 0 && stride_w > 0,
                        " stride_d: ",
                        stride_d,
                        "stride_h: ",
                        stride_h,
                        " stride_w: ",
                        stride_w)
                TORCH_CHECK(
                        pad_d >= 0 && pad_h >= 0 && pad_w >= 0,
                        "pad_d: ",
                        pad_d,
                        " pad_h: ",
                        pad_h,
                        " pad_w: ",
                        pad_w)
                TORCH_CHECK(
                        dilation_d > 0 && dilation_h > 0 && dilation_w > 0,
                        "dilation_d: ",
                        dilation_d,
                        " dilation_h: ",
                        dilation_h,
                        " dilation_w: ",
                        dilation_w)

                TORCH_CHECK(weight_c.size(1) * groups == in_channels)
                TORCH_CHECK(weight_c.size(0) % groups == 0)

                int64_t offset_groups = deformable ? offset_c.size(1) / (3 * weight_d * weight_h * weight_w) : 0;
                int64_t mask_groups = modulated ? mask_c.size(1) / (weight_d * weight_h * weight_w) : 0;

                TORCH_CHECK(
                        !deformable || offset_c.size(0) == input_c.size(0),
                        "invalid batch size of offset")
                TORCH_CHECK(
                        !deformable || offset_groups > 0,
                        "The shape of the offset tensor at dimension 1 is not valid. It should "
                        "be a multiple of 3 * weight.size(2) * weight.size(3) * weight.size(4).\nGot offset.size(1)=",
                        offset_c.size(1),
                        ", while 3 * weight.size(2) * weight.size(3) * weight.size(3)=",
                        3 * weight_d * weight_h * weight_w)
                TORCH_CHECK(!deformable || input_c.size(1) % offset_groups == 0)
                TORCH_CHECK(
                        !deformable || offset_c.size(1) == offset_groups * 3 * weight_d * weight_h * weight_w,
                        "offset.shape[1] is not valid. got: ",
                        offset_c.size(1),
                        " expected: ",
                        offset_groups * 3 * weight_d * weight_h * weight_w)
                TORCH_CHECK(
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

                TORCH_CHECK(
                        !modulated || mask_c.size(0) == input_c.size(0),
                        "invalid batch size of mask")
                TORCH_CHECK(
                        !modulated || mask_groups > 0,
                        "The shape of the mask tensor at dimension 1 is not valid. It should "
                        "be a multiple of weight.size(2) * weight.size(3) * weight.size(4).\nGot mask.size(1)=",
                        mask_c.size(1),
                        ", while weight.size(2) * weight.size(3) * weight.size(4)=",
                        weight_d * weight_h * weight_w)
                TORCH_CHECK(!modulated || input_c.size(1) % mask_groups == 0)
                TORCH_CHECK(
                        !modulated || mask_c.size(1) == mask_groups * weight_d * weight_h * weight_w,
                        "mask.shape[1] is not valid. got: ",
                        mask_c.size(1),
                        " expected: ",
                        mask_groups * weight_d * weight_h * weight_w)
                TORCH_CHECK(
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

                TORCH_CHECK(
                        out_d > 0 && out_h > 0 && out_w > 0,
                        "Calculated output size too small - out_d: ",
                        out_d,
                        " out_h: ",
                        out_h,
                        " out_w: ",
                        out_w)

                auto output = at::zeros({batch_sz, out_channels, out_d, out_h, out_w}, input_c.options());

                // Separate batches into blocks
                input_c = input_c.view({batch_sz / n_parallel_imgs,
                                        n_parallel_imgs,
                                        in_channels,
                                        in_d,
                                        in_h,
                                        in_w});
                if (deformable)
                    offset_c = offset_c.view({batch_sz / n_parallel_imgs,
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
                    offset_c = offset_c.view({batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              0, 0, 0, 0, 0, 0, 0, 0});
                if (modulated)
                    mask_c = mask_c.view({batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          mask_groups,
                                          weight_d,
                                          weight_h,
                                          weight_w,
                                          out_d,
                                          out_h,
                                          out_w});
                else
                    mask_c = mask_c.view({batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          0, 0, 0, 0, 0, 0, 0});

                output = output.view({batch_sz / n_parallel_imgs,
                                      n_parallel_imgs,
                                      out_channels,
                                      out_d,
                                      out_h,
                                      out_w});
                auto output_buf = at::zeros(
                        {batch_sz / n_parallel_imgs,
                         out_channels,
                         n_parallel_imgs,
                         out_d,
                         out_h,
                         out_w},
                        output.options());

                // Separate channels into convolution groups
                output_buf = output_buf.view({output_buf.size(0),
                                              groups,
                                              output_buf.size(1) / groups,
                                              output_buf.size(2),
                                              output_buf.size(3),
                                              output_buf.size(4),
                                              output_buf.size(5)});
                weight_c = weight_c.view({groups,
                                          weight_c.size(0) / groups,
                                          weight_c.size(1),
                                          weight_c.size(2),
                                          weight_c.size(3),
                                          weight_c.size(4)});

                // Sample points and perform convolution
                auto columns = at::empty({groups,
                                          in_channels * weight_d * weight_h * weight_w / groups,
                                          n_parallel_imgs * out_d * out_h * out_w},
                                         input_c.options());
                auto columns_view = columns.view({in_channels,
                                                  weight_d,
                                                  weight_h,
                                                  weight_w,
                                                  n_parallel_imgs,
                                                  out_d,
                                                  out_h,
                                                  out_w});
                for (int64_t b = 0; b < batch_sz / n_parallel_imgs; b++) {
                    vol2col(
                            input_c[b],
                            offset_c[b],
                            mask_c[b],
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
                        output_buf[b][g].flatten(1).addmm_(weight_c[g].flatten(1), columns[g]);
                    }
                }

                output_buf = output_buf.view({batch_sz / n_parallel_imgs,
                                              out_channels,
                                              n_parallel_imgs,
                                              out_d,
                                              out_h,
                                              out_w});
                output_buf.transpose_(1, 2);
                output.copy_(output_buf);
                output = output.view({batch_sz, out_channels, out_d, out_h, out_w});
                if (with_bias)
                    output.add_(bias_c.view({1, out_channels, 1, 1, 1}));
                if (unbatched)
                    output.squeeze_(0);

                return output;
            }

            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            deform_conv3d_backward_kernel(
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
                bool unbatched = input.ndimension() == 4;
                bool deformable = offset.has_value() && offset->defined();
                bool modulated = mask.has_value() && mask->defined();
                bool with_bias = bias.has_value() && bias->defined();

                auto grad_out_c = grad_out.contiguous();
                auto input_c = input.contiguous();
                auto weight_c = weight.contiguous();
                auto offset_c = (deformable ? offset.value() : input.new_zeros(0)).contiguous();
                auto mask_c = (modulated ? mask.value() : input.new_zeros(0)).contiguous();
                auto bias_c = (with_bias ? bias.value() : input.new_zeros(0)).contiguous();

                auto grad_input = at::zeros_like(input_c);
                auto grad_weight = at::zeros_like(weight_c);
                auto grad_offset = at::zeros_like(offset_c);
                auto grad_mask = at::zeros_like(mask_c);
                auto grad_bias = with_bias ? grad_out_c.sum(
                        unbatched ? at::IntArrayRef({1, 2, 3})
                                  : at::IntArrayRef({0, 2, 3, 4})) : at::Tensor();

                if (unbatched) {
                    grad_out_c = grad_out_c.unsqueeze(0);
                    input_c = input_c.unsqueeze(0);
                    grad_input.unsqueeze_(0);
                    offset_c = offset_c.unsqueeze(0);
                    grad_offset.unsqueeze_(0);
                    mask_c = mask_c.unsqueeze(0);
                    grad_mask.unsqueeze_(0);
                }

                int64_t batch_sz = input_c.size(0);
                int64_t in_channels = input_c.size(1);
                int64_t in_d = input_c.size(2);
                int64_t in_h = input_c.size(3);
                int64_t in_w = input_c.size(4);

                int64_t n_parallel_imgs = get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

                int64_t out_channels = weight_c.size(0);
                int64_t weight_d = weight_c.size(2);
                int64_t weight_h = weight_c.size(3);
                int64_t weight_w = weight_c.size(4);

                int64_t stride_d = stride[0];
                int64_t stride_h = stride[1];
                int64_t stride_w = stride[2];

                int64_t pad_d = padding[0];
                int64_t pad_h = padding[1];
                int64_t pad_w = padding[2];

                int64_t dilation_d = dilation[0];
                int64_t dilation_h = dilation[1];
                int64_t dilation_w = dilation[2];

                int64_t out_d = (in_d + 2 * pad_d - (dilation_d * (weight_d - 1) + 1)) / stride_d + 1;
                int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
                int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

                int64_t offset_groups = deformable ? offset_c.size(1) / (3 * weight_d * weight_h * weight_w) : 0;
                int64_t mask_groups = modulated ? mask_c.size(1) / (weight_d * weight_h * weight_w) : 0;

                // Separate into blocks
                input_c = input_c.view({batch_sz / n_parallel_imgs,
                                        n_parallel_imgs,
                                        in_channels,
                                        in_d,
                                        in_h,
                                        in_w});
                grad_input = grad_input.view_as(input_c);
                if (deformable)
                    offset_c = offset_c.view({batch_sz / n_parallel_imgs,
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
                    offset_c = offset_c.view({batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              0, 0, 0, 0, 0, 0, 0, 0});
                grad_offset = grad_offset.view_as(offset_c);
                if (modulated)
                    mask_c = mask_c.view({batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          mask_groups,
                                          weight_d,
                                          weight_h,
                                          weight_w,
                                          out_d,
                                          out_h,
                                          out_w});
                else
                    mask_c = mask_c.view({batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          0, 0, 0, 0, 0, 0, 0});
                grad_mask = grad_mask.view_as(mask_c);

                // Separate channels into convolution groups
                grad_out_c = grad_out_c.view({batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              groups,
                                              out_channels / groups,
                                              out_d,
                                              out_h,
                                              out_w}).permute({0, 2, 3, 1, 4, 5, 6}).contiguous();

                weight_c = weight_c.view({groups,
                                          out_channels / groups,
                                          in_channels / groups,
                                          weight_d,
                                          weight_h,
                                          weight_w});
                grad_weight = grad_weight.view_as(weight_c);

                auto columns = at::empty({groups,
                                          in_channels * weight_d * weight_h * weight_w / groups,
                                          n_parallel_imgs * out_d * out_h * out_w},
                                         input_c.options());
                auto columns_view = columns.view({in_channels,
                                                  weight_d,
                                                  weight_h,
                                                  weight_w,
                                                  n_parallel_imgs,
                                                  out_d,
                                                  out_h,
                                                  out_w});
                for (int64_t b = 0; b < batch_sz / n_parallel_imgs; b++) {
                    columns.zero_();
                    for (int64_t g = 0; g < groups; g++) {
                        columns[g].addmm_(weight_c[g].flatten(1).transpose(0, 1), grad_out_c[b][g].flatten(1));
                    }

                    auto grad_offset_b = grad_offset[b];
                    deform_conv3d_compute_grad_offset(
                            columns_view,
                            input_c[b],
                            offset_c[b],
                            mask_c[b],
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

                    auto grad_mask_b = grad_mask[b];
                    deform_conv3d_compute_grad_mask(
                            columns_view,
                            input_c[b],
                            offset_c[b],
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

                    auto grad_input_b = grad_input[b];
                    col2vol(
                            columns_view,
                            offset_c[b],
                            mask_c[b],
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
                            input_c[b],
                            offset_c[b],
                            mask_c[b],
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
                        grad_weight[g].flatten(1).addmm_(grad_out_c[b][g].flatten(1), columns[g].transpose(1, 0));
                    }
                }

                grad_input = grad_input.view_as(input);
                grad_weight = grad_weight.view_as(weight);
                grad_offset = deformable ? grad_offset.view_as(offset.value()) : at::Tensor();
                grad_mask = modulated ? grad_mask.view_as(mask.value()) : at::Tensor();
                if (unbatched) {
                    grad_input.squeeze_(0);
                    if (deformable)
                        grad_offset.squeeze_(0);
                    if (modulated)
                        grad_mask.squeeze_(0);
                }

                return std::make_tuple(grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
            }
        }

        TORCH_LIBRARY_IMPL(tvdcn, CUDA, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("tvdcn::deform_conv3d"),
                    TORCH_FN(deform_conv3d_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("tvdcn::_deform_conv3d_backward"),
                    TORCH_FN(deform_conv3d_backward_kernel));
        }
    }
}
