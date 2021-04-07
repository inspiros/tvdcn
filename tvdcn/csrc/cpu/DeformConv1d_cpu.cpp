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
// https://github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/DeformConv_cpu.cpp

#include "DeformConv_cpu.h"

template<typename scalar_t>
static scalar_t linear_interpolate(
        const scalar_t *im_data,
        const int width,
        scalar_t x) {
    if (x <= -1 || width <= x) {
        return 0;
    }

    int x_l = floor(x);
    int x_h = x_l + 1;

    bool valid_x_l = x_l >= 0;
    bool valid_x_h = x_h < width;

    scalar_t dx_h = x - x_l;
    scalar_t dx_l = 1 - dx_h;

    scalar_t val = 0;
    if (valid_x_l) val += dx_l * im_data[x_l];
    if (valid_x_h) val += dx_h * im_data[x_h];

    return val;
}

template<typename scalar_t>
static scalar_t deform_conv1d_get_coordinate_weight(
        const scalar_t *im_data,
        const int width,
        scalar_t x) {
    int x_l = floor(x);
    int x_h = x_l + 1;

    bool valid_x_l = 0 <= x_l && x_l < width;
    bool valid_x_h = 0 <= x_h && x_h < width;

    scalar_t dx_h = 1;
    scalar_t dx_l = -1;

    scalar_t val = 0;
    if (valid_x_l) val += dx_l * im_data[x_l];
    if (valid_x_h) val += dx_h * im_data[x_h];

    return val;
}

template<typename scalar_t>
static void deform_conv1d_im2col_kernel(
        const int n,
        const scalar_t *input,
        const scalar_t *offset,
        const scalar_t *mask,
        const int width,
        const int weight_w,
        const int pad_w,
        const int stride_w,
        const int dilation_w,
        const int out_w,
        const int batch_sz,
        const int in_channels,
        const int n_offset_grps,
        const bool modulated,
        scalar_t *columns) {
    for (int index = 0; index != n; ++index) {
        const int out_x = index % out_w;
        const int out_b = (index / out_w) % batch_sz;
        const int in_c = index / (out_w * batch_sz);
        const int out_c = in_c * weight_w;

        int c_per_offset_grp = in_channels / n_offset_grps;
        const int grp_idx = in_c / c_per_offset_grp;

        auto columns_ptr = columns +
                           (out_c * (batch_sz * out_w) + out_b * out_w + out_x);

        auto input_ptr = input +
                         (out_b * (in_channels * width) + in_c * width);

        auto offset_ptr = offset +
                          (out_b * n_offset_grps + grp_idx) * weight_w * out_w;

        auto mask_ptr = mask +
                        (out_b * n_offset_grps + grp_idx) * weight_w * out_w;

        for (int i = 0; i < weight_w; ++i) {
            const scalar_t offset_w =
                    offset_ptr[i * out_w + out_x];
            const scalar_t x = (out_x * stride_w - pad_w) + i * dilation_w + offset_w;
            const scalar_t val = linear_interpolate(input_ptr, width, x);

            const int mask_idx = (i * out_w + out_x);
            const scalar_t mask_weight = modulated ? mask_ptr[mask_idx] : static_cast<scalar_t>(1);

            *columns_ptr = val * mask_weight;
            columns_ptr += batch_sz * out_w;
        }
    }
}

static void deform_conv1d_im2col(
        const at::Tensor &input,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const int in_channels,
        const int width,
        const int weight_w,
        const int pad_w,
        const int stride_w,
        const int dilation_w,
        const int out_w,
        const int parallel_imgs,
        const int deformable_group,
        const bool modulated,
        at::Tensor &data_col) {
    int num_kernels = in_channels * out_w * parallel_imgs;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "deform_conv1d_im2col", ([&] {
        deform_conv1d_im2col_kernel(
                num_kernels,
                input.data_ptr<scalar_t>(),
                offset.data_ptr<scalar_t>(),
                mask.data_ptr<scalar_t>(),
                width,
                weight_w,
                pad_w,
                stride_w,
                dilation_w,
                out_w,
                parallel_imgs,
                in_channels,
                deformable_group,
                modulated,
                data_col.data_ptr<scalar_t>());
    }));
}

at::Tensor DeformConv1d_forward_cpu(
        const at::Tensor &input_param,
        const at::Tensor &weight_param,
        const at::Tensor &offset_param,
        const at::Tensor &mask_param,
        const at::Tensor &bias,
        const int stride_w,
        const int pad_w,
        const int dilation_w,
        const int n_weight_grps,
        const int n_offset_grps,
        const bool modulated) {
    at::Tensor input = input_param;
    at::Tensor weight = weight_param;
    at::Tensor offset = offset_param;
    at::Tensor mask = mask_param;

    TORCH_CHECK(input.ndimension() == 3);
    TORCH_CHECK(offset.ndimension() == 3);
    TORCH_CHECK(!modulated || mask.ndimension() == 3);
    TORCH_CHECK(weight.ndimension() == 3);
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(offset.is_contiguous());
    TORCH_CHECK(weight.is_contiguous());
    TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");

    int batch_sz = input.size(0);
    int in_channels = input.size(1);
    int in_w = input.size(2);

    int n_parallel_imgs =
            get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

    // Unpack shapes and args
    int out_channels = weight.size(0);
    int weight_w = weight.size(2);

    int out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

    TORCH_CHECK(
            weight_w > 0,
            " weight_w: ",
            weight_w);
    TORCH_CHECK(
            stride_w > 0,
            " stride_w: ",
            stride_w);
    TORCH_CHECK(pad_w >= 0, " pad_w: ", pad_w);
    TORCH_CHECK(dilation_w > 0, " dilation_w: ", dilation_w);

    TORCH_CHECK(weight.size(1) * n_weight_grps == input.size(1));
    TORCH_CHECK(weight.size(0) % n_weight_grps == 0);
    TORCH_CHECK(input.size(1) % n_offset_grps == 0);

    TORCH_CHECK(
            (offset.size(1) == n_offset_grps * weight_w),
            "offset.shape[1] is not valid: got: ",
            offset.size(1),
            " expected: ",
            n_offset_grps * weight_w);
    TORCH_CHECK(
            (offset.size(0) == input.size(0)), "invalid batch size of offset");
    TORCH_CHECK(
            (offset.size(2) == out_w),
            "offset output dims: (",
            offset.size(2),
            ") - ",
            "computed output dims: (",
            out_w,
            ")");

    TORCH_CHECK(
            (!modulated || mask.size(1) == n_offset_grps * weight_w),
            "mask.shape[1] is not valid: got: ",
            mask.size(1),
            " expected: ",
            n_offset_grps * weight_w);
    TORCH_CHECK(
            (mask.size(0) == input.size(0)), "invalid batch size of mask");
    TORCH_CHECK(
            (!modulated || mask.size(2) == out_w),
            "mask output dims: (",
            mask.size(2),
            ") - ",
            "computed output dims: (",
            out_w,
            ")");

    TORCH_CHECK(
            out_w > 0,
            "Calculated output size too small - out_w: ",
            out_w);

    auto out = at::zeros({batch_sz, out_channels, out_w}, input.options());

    // Separate batches into blocks
    out = out.view({batch_sz / n_parallel_imgs,
                    n_parallel_imgs,
                    out_channels,
                    out_w});
    input = input.view(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_w});
    offset = offset.view({batch_sz / n_parallel_imgs,
                          n_parallel_imgs,
                          n_offset_grps * weight_w,
                          out_w});
    if (modulated)
        mask = mask.view({batch_sz / n_parallel_imgs,
                          n_parallel_imgs,
                          n_offset_grps * weight_w,
                          out_w});
    auto out_buf = at::zeros(
            {batch_sz / n_parallel_imgs,
             out_channels,
             n_parallel_imgs,
             out_w},
            out.options());

    // Separate channels into convolution groups
    out_buf = out_buf.view({out_buf.size(0),
                            n_weight_grps,
                            out_buf.size(1) / n_weight_grps,
                            out_buf.size(2),
                            out_buf.size(3)});
    weight = weight.view({n_weight_grps,
                          weight.size(0) / n_weight_grps,
                          weight.size(1),
                          weight.size(2)});

    // Sample points and perform convolution
    auto columns = at::zeros(
            {in_channels * weight_w, n_parallel_imgs * out_w},
            input.options());
    columns = columns.view(
            {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
        deform_conv1d_im2col(
                input[b],
                offset[b],
                mask[b],
                in_channels,
                in_w,
                weight_w,
                pad_w,
                stride_w,
                dilation_w,
                out_w,
                n_parallel_imgs,
                n_offset_grps,
                modulated,
                columns);

        for (int g = 0; g < n_weight_grps; g++) {
            out_buf[b][g] = out_buf[b][g]
                    .flatten(1)
                    .addmm_(weight[g].flatten(1), columns[g])
                    .view_as(out_buf[b][g]);
        }
    }

    out_buf = out_buf.view({batch_sz / n_parallel_imgs,
                            out_channels,
                            n_parallel_imgs,
                            out_w});
    out_buf.transpose_(1, 2);
    out.copy_(out_buf);
    out = out.view({batch_sz, out_channels, out_w});

    return out + bias.view({1, out_channels, 1});
}

template<typename scalar_t>
static void deform_conv1d_col2im_kernel(
        const int n,
        const scalar_t *col,
        const scalar_t *offset,
        const scalar_t *mask,
        const int channels,
        const int width,
        const int weight_w,
        const int pad_w,
        const int stride_w,
        const int dilation_w,
        const int out_w,
        const int batch_sz,
        const int n_offset_grps,
        const bool modulated,
        scalar_t *grad_im) {
    for (int index = 0; index != n; ++index) {
        const int out_x = index % out_w;
        const int b = (index / out_w) % batch_sz;
        const int i = (index / (out_w * batch_sz)) % weight_w;
        const int c = index / (out_w * batch_sz * weight_w);

        int c_per_offset_grp = channels / n_offset_grps;
        const int offset_grp = c / c_per_offset_grp;

        auto offset_ptr = offset +
                          (b * n_offset_grps + offset_grp) * weight_w * out_w;
        auto mask_ptr = mask +
                        (b * n_offset_grps + offset_grp) * weight_w * out_w;
        const int offset_w_ptr = i * out_w + out_x;
        const scalar_t offset_w = offset_ptr[offset_w_ptr];
        const scalar_t x = (out_x * stride_w - pad_w) + i * dilation_w + offset_w;
        const int mask_idx = (i * out_w + out_x);
        const scalar_t mask_weight = modulated ? mask_ptr[mask_idx] : static_cast<scalar_t>(1);

        for (int dx = -1; dx <= 1; dx++) {
            int xp = int(x) + dx;
            if (0 <= xp && xp < width &&
                std::abs(x - xp) < 1) {
                int grad_pos = (b * channels + c) * width + xp;
                scalar_t weight = (1 - std::abs(x - xp));
                grad_im[grad_pos] += weight * col[index] * mask_weight;
            }
        }
    }
}

static void deform_conv1d_compute_grad_input(
        const at::Tensor &columns,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const int channels,
        const int width,
        const int weight_w,
        const int pad_w,
        const int stride_w,
        const int dilation_w,
        const int out_w,
        const int parallel_imgs,
        const int n_offset_grps,
        const bool modulated,
        at::Tensor &grad_im) {
    int num_kernels =
            channels * weight_w * out_w * parallel_imgs;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            columns.scalar_type(), "deform_conv1d_col2im", ([&] {
        deform_conv1d_col2im_kernel(
                num_kernels,
                columns.data_ptr<scalar_t>(),
                offset.data_ptr<scalar_t>(),
                mask.data_ptr<scalar_t>(),
                channels,
                width,
                weight_w,
                pad_w,
                stride_w,
                dilation_w,
                out_w,
                parallel_imgs,
                n_offset_grps,
                modulated,
                grad_im.data_ptr<scalar_t>());
    }));
}

template<typename scalar_t>
static void deform_conv1d_col2im_coord_kernel(
        const int n,
        const scalar_t *col,
        const scalar_t *im,
        const scalar_t *offset,
        const scalar_t *mask,
        const int channels,
        const int width,
        const int weight_w,
        const int pad_w,
        const int stride_w,
        const int dilation_w,
        const int out_w,
        const int batch_sz,
        const int offset_channels,
        const int n_offset_grps,
        const bool modulated,
        scalar_t *grad_offset,
        scalar_t *grad_mask) {
    for (int index = 0; index != n; ++index) {
        scalar_t val = 0, mval = 0;
        int w = index % out_w;
        int c = (index / out_w) % offset_channels;
        int b = index / (out_w * offset_channels);

        const int offset_grp = c / weight_w;
        const int col_step = weight_w;

        int c_per_offset_grp = channels / n_offset_grps;

        auto col_ptr = col +
                       offset_grp * c_per_offset_grp * weight_w * batch_sz * out_w;
        auto im_ptr = im +
                      (b * n_offset_grps + offset_grp) * c_per_offset_grp * width;
        auto offset_ptr = offset +
                          (b * n_offset_grps + offset_grp) * weight_w * out_w;
        auto mask_ptr = mask +
                        (b * n_offset_grps + offset_grp) * weight_w * out_w;

        const int offset_c = c - offset_grp * weight_w;

        const int c_bound = c_per_offset_grp * weight_w;
        for (int col_c = offset_c; col_c < c_bound; col_c += col_step) {
            const int col_pos = ((col_c * batch_sz + b) * out_w) + w;

            int out_x = col_pos % out_w;
            int i = (col_pos / (out_w * batch_sz)) % weight_w;

            const int offset_w_idx = (i * out_w + out_x);
            const scalar_t offset_w = offset_ptr[offset_w_idx];

            scalar_t x = (out_x * stride_w - pad_w) + i * dilation_w + offset_w;

            const int mask_idx = (i * out_w + out_x);
            const scalar_t mask_weight = modulated ? mask_ptr[mask_idx] : static_cast<scalar_t>(1);

            const scalar_t weight =
                    deform_conv1d_get_coordinate_weight(im_ptr, width, x);
            val += weight * col_ptr[col_pos] * mask_weight;
            mval += col_ptr[col_pos] * linear_interpolate(im_ptr, width, x);
            im_ptr += width;
        }

        grad_offset[index] = val;
        if (modulated) {
            const int grad_mask_idx =
                    ((b * n_offset_grps + offset_grp) * weight_w + offset_c) * out_w + w;
            grad_mask[grad_mask_idx] = mval;
        }
    }
}

static void deform_conv1d_compute_grad_offset_mask(
        const at::Tensor &columns,
        const at::Tensor &input,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const int channels,
        const int width,
        const int weight_w,
        const int pad_w,
        const int stride_w,
        const int dilation_w,
        const int out_w,
        const int parallel_imgs,
        const int n_offset_grps,
        const bool modulated,
        at::Tensor &grad_offset,
        at::Tensor &grad_mask) {
    int num_kernels =
            out_w * weight_w * n_offset_grps * parallel_imgs;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            columns.scalar_type(), "deform_conv1d_col2im_coord", ([&] {
        deform_conv1d_col2im_coord_kernel(
                num_kernels,
                columns.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                offset.data_ptr<scalar_t>(),
                mask.data_ptr<scalar_t>(),
                channels,
                width,
                weight_w,
                pad_w,
                stride_w,
                dilation_w,
                out_w,
                parallel_imgs,
                weight_w * n_offset_grps,
                n_offset_grps,
                modulated,
                grad_offset.data_ptr<scalar_t>(),
                grad_mask.data_ptr<scalar_t>());
    }));
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> deform_conv1d_backward_inputs_cpu(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor grad_out,
        const int stride_w,
        const int pad_w,
        const int dilation_w,
        const int n_weight_grps,
        const int n_offset_grps,
        const int n_parallel_imgs,
        const bool modulated) {
    int batch_sz = input.size(0);
    int in_channels = input.size(1);
    int in_w = input.size(2);

    int out_channels = weight.size(0);
    int weight_w = weight.size(2);

    int out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);
    auto columns = at::empty(
            {in_channels * weight_w, n_parallel_imgs * out_w},
            input.options());

    // Separate into blocks
    grad_input = grad_input.reshape(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_w});
    input = input.reshape(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_w});
    grad_offset = grad_offset.reshape({batch_sz / n_parallel_imgs,
                                       n_parallel_imgs,
                                       n_offset_grps * weight_w,
                                       out_w});
    offset = offset.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * weight_w,
                             out_w});
    if (modulated) {
        grad_mask = grad_mask.reshape({batch_sz / n_parallel_imgs,
                                       n_parallel_imgs,
                                       n_offset_grps * weight_w,
                                       out_w});
        mask = mask.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * weight_w,
                             out_w});
    }

    grad_out = grad_out
            .reshape({batch_sz / n_parallel_imgs,
                      n_parallel_imgs,
                      n_weight_grps,
                      out_channels / n_weight_grps,
                      out_w})
            .permute({0, 2, 3, 1, 4});

    weight = weight.reshape({n_weight_grps,
                             weight.size(0) / n_weight_grps,
                             weight.size(1),
                             weight.size(2)});

    columns = columns.view(
            {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});

    for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
        columns.zero_();
        // Separate into weight groups
        for (int g = 0; g < n_weight_grps; g++) {
            columns[g] = columns[g].addmm_(
                    weight[g].flatten(1).transpose(0, 1), grad_out[b][g].flatten(1));
        }

        auto grad_input_b = grad_input[b];
        auto grad_offset_b = grad_offset[b];
        auto grad_mask_b = grad_mask[b];
        deform_conv1d_compute_grad_offset_mask(
                columns,
                input[b],
                offset[b],
                mask[b],
                in_channels,
                in_w,
                weight_w,
                pad_w,
                stride_w,
                dilation_w,
                out_w,
                n_parallel_imgs,
                n_offset_grps,
                modulated,
                grad_offset_b,
                grad_mask_b);

        deform_conv1d_compute_grad_input(
                columns,
                offset[b],
                mask[b],
                in_channels,
                in_w,
                weight_w,
                pad_w,
                stride_w,
                dilation_w,
                out_w,
                n_parallel_imgs,
                n_offset_grps,
                modulated,
                grad_input_b);
    }

    grad_input = grad_input.view({batch_sz, in_channels, in_w});
    grad_offset = grad_offset.view(
            {batch_sz, n_offset_grps * weight_w, out_w});
    if (modulated) {
        grad_mask = grad_mask.view(
                {batch_sz, n_offset_grps * weight_w, out_w});
    }

    return std::make_tuple(grad_input, grad_offset, grad_mask);
}

static at::Tensor deform_conv1d_backward_parameters_cpu(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor grad_out,
        const int stride_w,
        const int pad_w,
        const int dilation_w,
        const int n_weight_grps,
        const int n_offset_grps,
        const int n_parallel_imgs,
        const bool modulated) {
    int batch_sz = input.size(0);
    int in_channels = input.size(1);
    int in_w = input.size(2);

    int out_channels = weight.size(0);
    int weight_w = weight.size(2);

    int out_w = grad_out.size(2);

    auto grad_weight = at::zeros_like(weight);

    at::Tensor grad_out_buf = grad_out
            .reshape({batch_sz / n_parallel_imgs,
                      n_parallel_imgs,
                      n_weight_grps,
                      out_channels / n_weight_grps,
                      out_w})
            .permute({0, 2, 3, 1, 4})
            .contiguous();

    input = input.reshape(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_w});
    offset = offset.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * weight_w,
                             out_w});
    if (modulated) {
        mask = mask.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * weight_w,
                             out_w});
    }

    grad_weight = grad_weight.view({n_weight_grps,
                                    grad_weight.size(0) / n_weight_grps,
                                    grad_weight.size(1),
                                    grad_weight.size(2)});

    auto columns = at::empty(
            {n_weight_grps,
             in_channels * weight_w / n_weight_grps,
             n_parallel_imgs * out_w},
            input.options());

    for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
        deform_conv1d_im2col(
                input[b],
                offset[b],
                mask[b],
                in_channels,
                in_w,
                weight_w,
                pad_w,
                stride_w,
                dilation_w,
                out_w,
                n_parallel_imgs,
                n_offset_grps,
                modulated,
                columns);

        for (int g = 0; g < n_weight_grps; g++) {
            grad_weight[g] =
                    grad_weight[g]
                            .flatten(1)
                            .addmm_(
                                    grad_out_buf[b][g].flatten(1), columns[g].transpose(1, 0))
                            .view_as(grad_weight[g]);
        }
    }

    grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                    grad_weight.size(2),
                                    grad_weight.size(3)});
    return grad_weight;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
DeformConv1d_backward_cpu(
        const at::Tensor &grad_out,
        const at::Tensor &input,
        const at::Tensor &weight,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const at::Tensor &bias,
        const int stride_w,
        const int pad_w,
        const int dilation_w,
        const int n_weight_grps,
        const int n_offset_grps,
        const bool modulated) {
    at::Tensor grad_out_c = grad_out.contiguous();
    at::Tensor input_c = input.contiguous();
    at::Tensor weight_c = weight.contiguous();
    at::Tensor offset_c = offset.contiguous();
    at::Tensor mask_c = mask.contiguous();
    at::Tensor bias_c = bias.contiguous();

    int batch_sz = input_c.size(0);
    int n_parallel_imgs =
            get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

    auto grad_input_and_offset_and_mask = deform_conv1d_backward_inputs_cpu(
            input_c,
            weight_c,
            offset_c,
            mask_c,
            grad_out_c,
            stride_w,
            pad_w,
            dilation_w,
            n_weight_grps,
            n_offset_grps,
            n_parallel_imgs,
            modulated);

    auto grad_input = std::get<0>(grad_input_and_offset_and_mask);
    auto grad_offset = std::get<1>(grad_input_and_offset_and_mask);
    auto grad_mask = std::get<2>(grad_input_and_offset_and_mask);

    auto grad_weight = deform_conv1d_backward_parameters_cpu(
            input_c,
            weight_c,
            offset_c,
            mask_c,
            grad_out_c,
            stride_w,
            pad_w,
            dilation_w,
            n_weight_grps,
            n_offset_grps,
            n_parallel_imgs,
            modulated);

    auto grad_bias = at::ones_like(bias_c) * grad_out_c.sum(c10::IntArrayRef({0, 2}));

    return std::make_tuple(grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
}
