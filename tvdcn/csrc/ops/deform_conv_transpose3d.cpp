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

#include <torch/autograd.h>
#include "utils/parallel_helpers.h"
#include "dispatch/deform_conv3d_kernels.h"

namespace tvdcn {
    namespace ops {
        at::Tensor deform_conv_transpose3d_forward(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &bias,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                at::IntArrayRef output_padding,
                at::IntArrayRef dilation,
                int64_t groups,
                bool deformable,
                bool modulated) {
            at::CheckedFrom c = "deform_conv_transpose3d_forward";
            auto args = {
                    at::TensorArg(input, "input", 1),
                    at::TensorArg(weight, "weight", 2),
                    at::TensorArg(offset, "offset", 3),
                    at::TensorArg(mask, "mask", 4),
                    at::TensorArg(bias, "bias", 5)};
            at::checkAllSameType(c, args);
            if (input.device().is_cuda())
                at::checkAllSameGPU(c, args);

            at::Tensor input_c = input.contiguous();
            at::Tensor weight_c = weight.contiguous();
            at::Tensor offset_c = offset.contiguous();
            at::Tensor mask_c = mask.contiguous();
            at::Tensor bias_c = bias.contiguous();

            TORCH_CHECK(input_c.ndimension() == 5)
            TORCH_CHECK(weight_c.ndimension() == 5)
            TORCH_CHECK(!deformable || offset_c.ndimension() == 5)
            TORCH_CHECK(!modulated || mask_c.ndimension() == 5)

            int64_t batch_sz = input_c.size(0);
            int64_t in_channels = input_c.size(1);
            int64_t in_d = input_c.size(2);
            int64_t in_h = input_c.size(3);
            int64_t in_w = input_c.size(4);

            int64_t n_parallel_imgs = get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

            int64_t out_channels = weight_c.size(1) * groups;
            int64_t weight_d = weight_c.size(2);
            int64_t weight_h = weight_c.size(3);
            int64_t weight_w = weight_c.size(4);

            int64_t stride_d = stride[0];
            int64_t stride_h = stride[1];
            int64_t stride_w = stride[2];

            int64_t pad_d = padding[0];
            int64_t pad_h = padding[1];
            int64_t pad_w = padding[2];

            int64_t out_pad_d = output_padding[0];
            int64_t out_pad_h = output_padding[1];
            int64_t out_pad_w = output_padding[2];

            int64_t dilation_d = dilation[0];
            int64_t dilation_h = dilation[1];
            int64_t dilation_w = dilation[2];

            int64_t out_d = (in_d - 1) * stride_d - 2 * pad_d + dilation_d * (weight_d - 1) + 1 + out_pad_d;
            int64_t out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (weight_h - 1) + 1 + out_pad_h;
            int64_t out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (weight_w - 1) + 1 + out_pad_w;

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
            TORCH_CHECK(pad_d >= 0 && pad_h >= 0 && pad_w >= 0, "pad_d: ", pad_d, " pad_h: ", pad_h, " pad_w: ", pad_w)
            TORCH_CHECK(dilation_d > 0 && dilation_h > 0 && dilation_w > 0, "dilation_d: ", dilation_d, " dilation_h: ",
                        dilation_h, " dilation_w: ", dilation_w)

            TORCH_CHECK(weight_c.size(1) * groups == out_channels)
            TORCH_CHECK(weight_c.size(0) % groups == 0)

            int64_t offset_groups = offset.size(1) / (3 * weight_d * weight_h * weight_w);
            int64_t mask_groups = mask.size(1) / (weight_d * weight_h * weight_w);

            TORCH_CHECK(!deformable || offset_groups > 0,
                        "The shape of the offset tensor at dimension 1 is not valid. It should "
                        "be a multiple of 3 * weight.size(2) * weight.size(3) * weight.size(4).\nGot offset.size(1)=",
                        offset.size(1),
                        ", while 3 * weight.size(2) * weight.size(3) * weight.size(3)=",
                        3 * weight_d * weight_h * weight_w)
            TORCH_CHECK(!deformable || input_c.size(1) % offset_groups == 0)

            TORCH_CHECK(!modulated || mask_groups > 0,
                        "The shape of the mask tensor at dimension 1 is not valid. It should "
                        "be a multiple of weight.size(2) * weight.size(3) * weight.size(4).\nGot mask.size(1)=",
                        mask.size(1),
                        ", while weight.size(2) * weight.size(3) * weight.size(4)=",
                        weight_d * weight_h * weight_w)
            TORCH_CHECK(!modulated || input_c.size(1) % mask_groups == 0)

            TORCH_CHECK(
                    (!deformable || offset_c.size(1) == offset_groups * 3 * weight_d * weight_h * weight_w),
                    "offset.shape[1] is not valid. got: ",
                    offset_c.size(1),
                    " expected: ",
                    offset_groups * 3 * weight_d * weight_h * weight_w)
            TORCH_CHECK(
                    (offset_c.size(0) == input_c.size(0)), "invalid batch size of offset")
            TORCH_CHECK(
                    (!deformable || (offset_c.size(2) == in_d &&
                                     offset_c.size(3) == in_h &&
                                     offset_c.size(4) == in_w)),
                    "offset input dims: (",
                    offset_c.size(2),
                    ", ",
                    offset_c.size(3),
                    ", ",
                    offset_c.size(4),
                    ") - ",
                    "input dims: (",
                    in_d,
                    ", ",
                    in_h,
                    ", ",
                    in_w,
                    ")")

            TORCH_CHECK(
                    (!modulated || mask_c.size(1) == mask_groups * weight_d * weight_h * weight_w),
                    "mask.shape[1] is not valid. got: ",
                    mask_c.size(1),
                    " expected: ",
                    mask_groups * weight_d * weight_h * weight_w)
            TORCH_CHECK(
                    (mask_c.size(0) == input_c.size(0)), "invalid batch size of mask")
            TORCH_CHECK(
                    (!modulated || (mask_c.size(2) == in_d &&
                                    mask_c.size(3) == in_h &&
                                    mask_c.size(4) == in_w)),
                    "mask input dims: (",
                    mask_c.size(2),
                    ", ",
                    mask_c.size(3),
                    ", ",
                    mask_c.size(4),
                    ") - ",
                    "input dims: (",
                    in_d,
                    ", ",
                    in_h,
                    ", ",
                    in_w,
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
            output = output.view({batch_sz / n_parallel_imgs,
                                  n_parallel_imgs,
                                  out_channels,
                                  out_d,
                                  out_h,
                                  out_w});
            if (deformable)
                offset_c = offset_c.view({batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          offset_groups,
                                          weight_d,
                                          weight_h,
                                          weight_w,
                                          3,
                                          in_d,
                                          in_h,
                                          in_w});
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
                                      in_d,
                                      in_h,
                                      in_w});
            else
                mask_c = mask_c.view({batch_sz / n_parallel_imgs,
                                      n_parallel_imgs,
                                      0, 0, 0, 0, 0, 0, 0});

            // Separate channels into convolution groups
            input_c = input_c.view({batch_sz / n_parallel_imgs,
                                    n_parallel_imgs,
                                    groups,
                                    in_channels / groups,
                                    in_d,
                                    in_h,
                                    in_w}).permute({0, 2, 3, 1, 4, 5, 6}).contiguous();
            weight_c = weight_c.view({groups,
                                      weight_c.size(0) / groups,
                                      weight_c.size(1),
                                      weight_c.size(2),
                                      weight_c.size(3),
                                      weight_c.size(4)});

            // Sample points and perform convolution
            auto columns = at::empty({groups,
                                      out_channels * weight_d * weight_h * weight_w / groups,
                                      n_parallel_imgs * in_d * in_h * in_w},
                                     input_c.options());
            auto columns_view = columns.view({out_channels,
                                              weight_d,
                                              weight_h,
                                              weight_w,
                                              n_parallel_imgs,
                                              in_d,
                                              in_h,
                                              in_w});
            for (int64_t b = 0; b < batch_sz / n_parallel_imgs; b++) {
                columns.zero_();
                for (int64_t g = 0; g < groups; g++) {
                    columns[g].addmm_(weight_c[g].flatten(1).transpose(0, 1), input_c[b][g].flatten(1));
                }

                auto output_b = output[b];
                col2vol(
                        columns_view,
                        offset_c[b],
                        mask_c[b],
                        out_channels,
                        out_d,
                        out_h,
                        out_w,
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
                        in_d,
                        in_h,
                        in_w,
                        n_parallel_imgs,
                        offset_groups,
                        mask_groups,
                        deformable,
                        modulated,
                        output_b);
            }

            output = output.view({batch_sz, out_channels, out_d, out_h, out_w});

            return output + bias_c.view({1, out_channels, 1, 1, 1});
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
            _deform_conv_transpose3d_backward(
                    const at::Tensor &grad_out,
                    const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const at::Tensor &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef output_padding,
                    at::IntArrayRef dilation,
                    int64_t groups,
                    bool deformable,
                    bool modulated) {
                at::Tensor grad_out_c = grad_out.contiguous();
                at::Tensor input_c = input.contiguous();
                at::Tensor weight_c = weight.contiguous();
                at::Tensor offset_c = offset.contiguous();
                at::Tensor mask_c = mask.contiguous();
                at::Tensor bias_c = bias.contiguous();

                int64_t batch_sz = input_c.size(0);
                int64_t in_channels = input_c.size(1);
                int64_t in_d = input_c.size(2);
                int64_t in_h = input_c.size(3);
                int64_t in_w = input_c.size(4);

                int64_t n_parallel_imgs = get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

                int64_t out_channels = weight_c.size(1) * groups;
                int64_t weight_d = weight_c.size(2);
                int64_t weight_h = weight_c.size(3);
                int64_t weight_w = weight_c.size(4);

                int64_t stride_d = stride[0];
                int64_t stride_h = stride[1];
                int64_t stride_w = stride[2];

                int64_t pad_d = padding[0];
                int64_t pad_h = padding[1];
                int64_t pad_w = padding[2];

                int64_t out_pad_d = output_padding[0];
                int64_t out_pad_h = output_padding[1];
                int64_t out_pad_w = output_padding[2];

                int64_t dilation_d = dilation[0];
                int64_t dilation_h = dilation[1];
                int64_t dilation_w = dilation[2];

                int64_t out_d = (in_d - 1) * stride_d - 2 * pad_d + dilation_d * (weight_d - 1) + 1 + out_pad_d;
                int64_t out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (weight_h - 1) + 1 + out_pad_h;
                int64_t out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (weight_w - 1) + 1 + out_pad_w;

                int64_t offset_groups = offset.size(1) / (3 * weight_d * weight_h * weight_w);
                int64_t mask_groups = mask.size(1) / (weight_d * weight_h * weight_w);

                auto grad_input = at::zeros_like(input_c);
                auto grad_weight = at::zeros_like(weight_c);
                auto grad_offset = at::zeros_like(offset_c);
                auto grad_mask = at::zeros_like(mask_c);
                auto grad_bias = at::ones_like(bias_c);

                // Separate into blocks
                grad_out_c = grad_out_c.view({batch_sz / n_parallel_imgs,
                                              n_parallel_imgs,
                                              out_channels,
                                              out_d,
                                              out_h,
                                              out_w});

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
                                              in_d,
                                              in_h,
                                              in_w});
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
                                          in_d,
                                          in_h,
                                          in_w});
                else
                    mask_c = mask_c.view({batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          0, 0, 0, 0, 0, 0, 0});
                grad_mask = grad_mask.view_as(mask_c);

                auto grad_input_buf = at::zeros({batch_sz / n_parallel_imgs,
                                                 in_channels,
                                                 n_parallel_imgs,
                                                 in_d,
                                                 in_h,
                                                 in_w}, input_c.options());

                // Separate channels into convolution groups
                input_c = input_c.view({batch_sz / n_parallel_imgs,
                                        n_parallel_imgs,
                                        groups,
                                        in_channels / groups,
                                        in_d,
                                        in_h,
                                        in_w}).permute({0, 2, 3, 1, 4, 5, 6}).contiguous();

                grad_input_buf = grad_input_buf.view({grad_input_buf.size(0),
                                                      groups,
                                                      grad_input_buf.size(1) / groups,
                                                      grad_input_buf.size(2),
                                                      grad_input_buf.size(3),
                                                      grad_input_buf.size(4),
                                                      grad_input_buf.size(5)});

                weight_c = weight_c.view({groups,
                                          in_channels / groups,
                                          out_channels / groups,
                                          weight_d,
                                          weight_h,
                                          weight_w});
                grad_weight = grad_weight.view_as(weight_c);

                auto columns = at::empty({groups,
                                          out_channels * weight_d * weight_h * weight_w / groups,
                                          n_parallel_imgs * in_d * in_h * in_w},
                                         input_c.options());
                auto columns_view = columns.view({out_channels,
                                                  weight_d,
                                                  weight_h,
                                                  weight_w,
                                                  n_parallel_imgs,
                                                  in_d,
                                                  in_h,
                                                  in_w});
                for (int64_t b = 0; b < batch_sz / n_parallel_imgs; b++) {
                    columns.zero_();
                    for (int64_t g = 0; g < groups; g++) {
                        columns[g].addmm_(weight_c[g].flatten(1).transpose(0, 1), input_c[b][g].flatten(1));
                    }

                    auto grad_offset_b = grad_offset[b];
                    deform_conv3d_compute_grad_offset(
                            columns_view,
                            grad_out_c[b],
                            offset_c[b],
                            mask_c[b],
                            out_channels,
                            out_d,
                            out_h,
                            out_w,
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
                            in_d,
                            in_h,
                            in_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            grad_offset_b);

                    auto grad_mask_b = grad_mask[b];
                    deform_conv3d_compute_grad_mask(
                            columns_view,
                            grad_out_c[b],
                            offset_c[b],
                            out_channels,
                            out_d,
                            out_h,
                            out_w,
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
                            in_d,
                            in_h,
                            in_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            grad_mask_b);

                    vol2col(
                            grad_out_c[b],
                            offset_c[b],
                            mask_c[b],
                            out_channels,
                            out_d,
                            out_h,
                            out_w,
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
                            in_d,
                            in_h,
                            in_w,
                            n_parallel_imgs,
                            offset_groups,
                            mask_groups,
                            deformable,
                            modulated,
                            columns_view);

                    for (int64_t g = 0; g < groups; g++) {
                        grad_input_buf[b][g].flatten(1).addmm_(weight_c[g].flatten(1), columns[g]);
                        grad_weight[g].flatten(1).addmm_(input_c[b][g].flatten(1), columns[g].transpose(1, 0));
                    }
                }

                grad_input_buf = grad_input_buf.view({batch_sz / n_parallel_imgs,
                                                      in_channels,
                                                      n_parallel_imgs,
                                                      in_d,
                                                      in_h,
                                                      in_w}).transpose_(1, 2);
                grad_input.copy_(grad_input_buf);

                grad_input = grad_input.view_as(input);
                grad_weight = grad_weight.view_as(weight);
                grad_offset = grad_offset.view_as(offset);
                grad_mask = grad_mask.view_as(mask);
                grad_bias *= grad_out.sum(at::IntArrayRef({0, 2, 3, 4}));

                return std::make_tuple(grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
            }
        }

        class DeformConvTranspose3dFunction
                : public torch::autograd::Function<DeformConvTranspose3dFunction> {
        public:
            static torch::autograd::variable_list forward(
                    torch::autograd::AutogradContext *ctx,
                    const torch::autograd::Variable &input,
                    const torch::autograd::Variable &weight,
                    const torch::autograd::Variable &offset,
                    const torch::autograd::Variable &mask,
                    const torch::autograd::Variable &bias,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    at::IntArrayRef output_padding,
                    at::IntArrayRef dilation,
                    int64_t groups,
                    bool deformable,
                    bool modulated) {
                at::AutoDispatchBelowADInplaceOrView g;
                auto output = deform_conv_transpose3d_forward(
                        input,
                        weight,
                        offset,
                        mask,
                        bias,
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups,
                        deformable,
                        modulated);

                ctx->save_for_backward({input, weight, offset, mask, bias});
                ctx->saved_data["stride"] = stride.vec();
                ctx->saved_data["padding"] = padding.vec();
                ctx->saved_data["output_padding"] = output_padding.vec();
                ctx->saved_data["dilation"] = dilation.vec();
                ctx->saved_data["groups"] = groups;
                ctx->saved_data["deformable"] = deformable;
                ctx->saved_data["modulated"] = modulated;

                return {
                        output,
                };
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
                auto deformable = ctx->saved_data["deformable"].toBool();
                auto modulated = ctx->saved_data["modulated"].toBool();

                auto grads = detail::_deform_conv_transpose3d_backward(
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
                        groups,
                        deformable,
                        modulated);
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
                        torch::autograd::Variable(),
                        torch::autograd::Variable(),
                };
            }
        };

        at::Tensor deform_conv_transpose3d(
                const at::Tensor &input,
                const at::Tensor &weight,
                const at::optional<at::Tensor> &offset = {},
                const at::optional<at::Tensor> &mask = {},
                const at::optional<at::Tensor> &bias = {},
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                at::IntArrayRef output_padding = 0,
                at::IntArrayRef dilation = 1,
                int64_t groups = 1) {
            C10_LOG_API_USAGE_ONCE("tvdcn.csrc.ops.deform_conv_transpose.deform_conv_transpose3d");
            auto result = DeformConvTranspose3dFunction::apply(
                    input,
                    weight,
                    offset.value_or(at::zeros({input.size(0), 0}, input.options().requires_grad(false))),
                    mask.value_or(at::zeros({input.size(0), 0}, input.options().requires_grad(false))),
                    bias.value_or(at::zeros({weight.size(0)}, input.options().requires_grad(false))),
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    groups,
                    offset.has_value(),
                    mask.has_value());
            return result[0];
        }

        TORCH_LIBRARY_FRAGMENT(tvdcn, m) {
            m.def("tvdcn::deform_conv_transpose3d(Tensor input, Tensor weight, "
                  "Tensor? offset=None, Tensor? mask=None, Tensor? bias=None, "
                  "int[3] stride=1, int[3] padding=0, int[3] output_padding=0, "
                  "int[3] dilation=1, int groups=1) -> Tensor",
                  &deform_conv_transpose3d);
        }
    }
}
