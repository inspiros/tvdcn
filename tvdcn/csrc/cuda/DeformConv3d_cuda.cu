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
// https://github.com/pytorch/vision/blob/master/torchvision/csrc/cuda/DeformConv_cuda.cpp

#include "cuda_helpers.h"
#include "DeformConv_cuda.h"

constexpr float threadsFraction = 0.5;

template<typename scalar_t>
__device__ scalar_t trilinear_interpolate(
        const scalar_t *im_data,
        const int depth,
        const int height,
        const int width,
        scalar_t z,
        scalar_t y,
        scalar_t x) {
    if (z <= -1 || depth <= z || y <= -1 || height <= y || x <= -1 || width <= x) {
        return 0;
    }

    int z_l = floor(z);
    int z_h = z_l + 1;
    int y_l = floor(y);
    int y_h = y_l + 1;
    int x_l = floor(x);
    int x_h = x_l + 1;

    bool valid_z_l = z_l >= 0;
    bool valid_z_h = z_h < depth;
    bool valid_y_l = y_l >= 0;
    bool valid_y_h = y_h < height;
    bool valid_x_l = x_l >= 0;
    bool valid_x_h = x_h < width;

    scalar_t dz_h = z - z_l;
    scalar_t dy_h = y - y_l;
    scalar_t dx_h = x - x_l;
    scalar_t dz_l = 1 - dz_h;
    scalar_t dy_l = 1 - dy_h;
    scalar_t dx_l = 1 - dx_h;

    scalar_t val = 0;
    if (valid_z_l && valid_y_l && valid_x_l)
        val += dz_l * dy_l * dx_l * im_data[z_l * (width * height) + y_l * width + x_l];
    if (valid_z_l && valid_y_l && valid_x_h)
        val += dz_l * dy_l * dx_h * im_data[z_l * (width * height) + y_l * width + x_h];
    if (valid_z_l && valid_y_h && valid_x_l)
        val += dz_l * dy_h * dx_l * im_data[z_l * (width * height) + y_h * width + x_l];
    if (valid_z_l && valid_y_h && valid_x_h)
        val += dz_l * dy_h * dx_h * im_data[z_l * (width * height) + y_h * width + x_h];
    if (valid_z_h && valid_y_l && valid_x_l)
        val += dz_h * dy_l * dx_l * im_data[z_h * (width * height) + y_l * width + x_l];
    if (valid_z_h && valid_y_l && valid_x_h)
        val += dz_h * dy_l * dx_h * im_data[z_h * (width * height) + y_l * width + x_h];
    if (valid_z_h && valid_y_h && valid_x_l)
        val += dz_h * dy_h * dx_l * im_data[z_h * (width * height) + y_h * width + x_l];
    if (valid_z_h && valid_y_h && valid_x_h)
        val += dz_h * dy_h * dx_h * im_data[z_h * (width * height) + y_h * width + x_h];

    return val;
}

template<typename scalar_t>
__device__ scalar_t deform_conv3d_get_coordinate_weight(
        const scalar_t *im_data,
        const int depth,
        const int height,
        const int width,
        scalar_t z,
        scalar_t y,
        scalar_t x,
        const int direction) {
    int z_l = floor(z);
    int y_l = floor(y);
    int x_l = floor(x);
    int z_h = z_l + 1;
    int y_h = y_l + 1;
    int x_h = x_l + 1;

    bool valid_z_l = 0 <= z_l && z_l < depth;
    bool valid_z_h = 0 <= z_h && z_h < depth;
    bool valid_y_l = 0 <= y_l && y_l < height;
    bool valid_y_h = 0 <= y_h && y_h < height;
    bool valid_x_l = 0 <= x_l && x_l < width;
    bool valid_x_h = 0 <= x_h && x_h < width;

    scalar_t dz_h = (direction == 0) ? static_cast<scalar_t>(1) : z - z_l;
    scalar_t dy_h = (direction == 1) ? static_cast<scalar_t>(1) : y - y_l;
    scalar_t dx_h = (direction == 2) ? static_cast<scalar_t>(1) : x - x_l;
    scalar_t dz_l = (direction == 0) ? static_cast<scalar_t>(-1) : 1 - dz_h;
    scalar_t dy_l = (direction == 1) ? static_cast<scalar_t>(-1) : 1 - dy_h;
    scalar_t dx_l = (direction == 2) ? static_cast<scalar_t>(-1) : 1 - dx_h;

    scalar_t val = 0;
    if (valid_z_l && valid_y_l && valid_x_l)
        val += dz_l * dy_l * dx_l * im_data[z_l * (width * height) + y_l * width + x_l];
    if (valid_z_l && valid_y_l && valid_x_h)
        val += dz_l * dy_l * dx_h * im_data[z_l * (width * height) + y_l * width + x_h];
    if (valid_z_l && valid_y_h && valid_x_l)
        val += dz_l * dy_h * dx_l * im_data[z_l * (width * height) + y_h * width + x_l];
    if (valid_z_l && valid_y_h && valid_x_h)
        val += dz_l * dy_h * dx_h * im_data[z_l * (width * height) + y_h * width + x_h];
    if (valid_z_h && valid_y_l && valid_x_l)
        val += dz_h * dy_l * dx_l * im_data[z_h * (width * height) + y_l * width + x_l];
    if (valid_z_h && valid_y_l && valid_x_h)
        val += dz_h * dy_l * dx_h * im_data[z_h * (width * height) + y_l * width + x_h];
    if (valid_z_h && valid_y_h && valid_x_l)
        val += dz_h * dy_h * dx_l * im_data[z_h * (width * height) + y_h * width + x_l];
    if (valid_z_h && valid_y_h && valid_x_h)
        val += dz_h * dy_h * dx_h * im_data[z_h * (width * height) + y_h * width + x_h];

    return val;
}

template<typename scalar_t>
__global__ void deform_conv3d_im2col_gpu_kernel(
        const int n,
        const scalar_t *input_ptr,
        const scalar_t *offset_ptr,
        const scalar_t *mask_ptr,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int out_d,
        const int out_h,
        const int out_w,
        const int batch_sz,
        const int in_channels,
        const int n_offset_grps,
        const bool modulated,
        scalar_t *columns_ptr) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        const int out_x = index % out_w;
        const int out_y = (index / out_w) % out_h;
        const int out_z = (index / (out_w * out_h)) % out_d;
        const int out_b = (index / (out_w * out_h * out_d)) % batch_sz;
        const int in_c = index / (out_w * out_h * out_d * batch_sz);
        const int out_c = in_c * weight_d * weight_h * weight_w;

        int c_per_offset_grp = in_channels / n_offset_grps;
        const int grp_idx = in_c / c_per_offset_grp;

        columns_ptr +=
                (out_c * (batch_sz * out_d * out_h * out_w) + out_b * (out_d * out_h * out_w) +
                 out_z * (out_h * out_w) + out_y * out_w + out_x);

        input_ptr +=
                (out_b * (in_channels * depth * height * width) + in_c * (depth * height * width));

        offset_ptr +=
                (out_b * n_offset_grps + grp_idx) * 3 * weight_d * weight_h * weight_w * out_d * out_h * out_w;

        mask_ptr +=
                (out_b * n_offset_grps + grp_idx) * weight_d * weight_h * weight_w * out_d * out_h * out_w;

        for (int i = 0; i < weight_d; ++i) {
            for (int j = 0; j < weight_h; ++j) {
                for (int k = 0; k < weight_w; ++k) {
                    const int offset_idx = 3 * (i * (weight_h * weight_w) + j * weight_w + k);
                    const scalar_t offset_d =
                            offset_ptr[offset_idx * (out_d * out_h * out_w) + out_z * (out_h * out_w) + out_y * out_w +
                                       out_x];
                    const scalar_t offset_h =
                            offset_ptr[(offset_idx + 1) * (out_d * out_h * out_w) + out_z * (out_h * out_w) +
                                       out_y * out_w + out_x];
                    const scalar_t offset_w =
                            offset_ptr[(offset_idx + 2) * (out_d * out_h * out_w) + out_z * (out_h * out_w) +
                                       out_y * out_w + out_x];
                    const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
                    const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
                    const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;
                    const scalar_t val = trilinear_interpolate(input_ptr, depth, height, width, z, y, x);

                    const int mask_idx =
                            (((i * weight_h * weight_w + j * weight_w + k) * out_d + out_z) * out_h + out_y) * out_w +
                            out_x;
                    const scalar_t mask_weight = modulated ? mask_ptr[mask_idx] : static_cast<scalar_t>(1);

                    *columns_ptr = val * mask_weight;
                    columns_ptr += batch_sz * out_d * out_h * out_w;
                }
            }
        }
    }
}

static void deform_conv3d_im2col(
        const at::Tensor &input,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const int in_channels,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int out_d,
        const int out_h,
        const int out_w,
        const int parallel_imgs,
        const int deformable_group,
        const bool modulated,
        at::Tensor data_col) {
    int num_kernels = in_channels * out_d * out_h * out_w * parallel_imgs;

    const unsigned int threads = GET_THREADS(threadsFraction);
    const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "deform_conv3d_im2col_gpu", ([&] {
                deform_conv3d_im2col_gpu_kernel<<<blocks, threads>>>(
                        num_kernels,
                        input.data_ptr<scalar_t>(),
                        offset.data_ptr<scalar_t>(),
                        mask.data_ptr<scalar_t>(),
                        depth,
                        height,
                        width,
                        weight_d,
                        weight_h,
                        weight_w,
                        pad_d,
                        pad_h,
                        pad_w,
                        stride_d,
                        stride_h,
                        stride_w,
                        dilation_d,
                        dilation_h,
                        dilation_w,
                        out_d,
                        out_h,
                        out_w,
                        parallel_imgs,
                        in_channels,
                        deformable_group,
                        modulated,
                        data_col.data_ptr<scalar_t>());
            }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in deform_conv3d_im2col: %s\n", cudaGetErrorString(err));
    }
}

at::Tensor DeformConv3d_forward_cuda(
        const at::Tensor &input_param,
        const at::Tensor &weight_param,
        const at::Tensor &offset_param,
        const at::Tensor &mask_param,
        const at::Tensor &bias,
        const std::tuple<int, int, int> &stride,
        const std::tuple<int, int, int> &pad,
        const std::tuple<int, int, int> &dilation,
        const int n_weight_grps,
        const int n_offset_grps,
        const bool modulated) {
    at::Tensor input = input_param;
    at::Tensor offset = offset_param;
    at::Tensor mask = mask_param;
    at::Tensor weight = weight_param;

    TORCH_CHECK(input.ndimension() == 5);
    TORCH_CHECK(offset.ndimension() == 5);
    TORCH_CHECK(!modulated || mask.ndimension() == 5);
    TORCH_CHECK(weight.ndimension() == 5);
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(offset.is_contiguous());
    TORCH_CHECK(weight.is_contiguous());
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");

    at::DeviceGuard guard(input.device());

    int batch_sz = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int n_parallel_imgs =
            get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

    int out_channels = weight.size(0);
    int weight_d = weight.size(2);
    int weight_h = weight.size(3);
    int weight_w = weight.size(4);

    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    int pad_d = std::get<0>(pad);
    int pad_h = std::get<1>(pad);
    int pad_w = std::get<2>(pad);

    int dilation_d = std::get<0>(dilation);
    int dilation_h = std::get<1>(dilation);
    int dilation_w = std::get<2>(dilation);

    int out_d = (in_d + 2 * pad_d - (dilation_d * (weight_d - 1) + 1)) / stride_d + 1;
    int out_h = (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

    TORCH_CHECK(
            weight_d > 0 && weight_h > 0 && weight_w > 0,
            "weight_d: ",
            weight_d,
            " weight_h: ",
            weight_h,
            " weight_w: ",
            weight_w);
    TORCH_CHECK(
            stride_d > 0 && stride_h > 0 && stride_w > 0,
            " stride_d: ",
            stride_d,
            "stride_h: ",
            stride_h,
            " stride_w: ",
            stride_w);
    TORCH_CHECK(pad_d >= 0 && pad_h >= 0 && pad_w >= 0, "pad_d: ", pad_d, " pad_h: ", pad_h, " pad_w: ", pad_w);
    TORCH_CHECK(dilation_d > 0 && dilation_h > 0 && dilation_w > 0, "dilation_d: ", dilation_d, " dilation_h: ",
                dilation_h, " dilation_w: ", dilation_w);

    TORCH_CHECK(weight.size(1) * n_weight_grps == input.size(1));
    TORCH_CHECK(weight.size(0) % n_weight_grps == 0);
    TORCH_CHECK(input.size(1) % n_offset_grps == 0);

    TORCH_CHECK(
            (offset.size(1) == n_offset_grps * 3 * weight_d * weight_h * weight_w),
            "offset.shape[1] is not valid: got: ",
            offset.size(1),
            " expected: ",
            n_offset_grps * 3 * weight_d * weight_h * weight_w);
    TORCH_CHECK(
            (offset.size(0) == input.size(0)), "invalid batch size of offset");
    TORCH_CHECK(
            (offset.size(2) == out_d && offset.size(3) == out_h && offset.size(4) == out_w),
            "offset output dims: (",
            offset.size(2),
            ", ",
            offset.size(3),
            ", ",
            offset.size(4),
            ") - ",
            "computed output dims: (",
            out_d,
            ", ",
            out_h,
            ", ",
            out_w,
            ")");

    TORCH_CHECK(
            (!modulated || mask.size(1) == n_offset_grps * weight_d * weight_h * weight_w),
            "mask.shape[1] is not valid: got: ",
            mask.size(1),
            " expected: ",
            n_offset_grps * weight_d * weight_h * weight_w);
    TORCH_CHECK(
            (mask.size(0) == input.size(0)), "invalid batch size of mask");
    TORCH_CHECK(
            (!modulated || (mask.size(2) == out_d && mask.size(3) == out_h && mask.size(4) == out_w)),
            "mask output dims: (",
            mask.size(2),
            ", ",
            mask.size(3),
            ", ",
            mask.size(4),
            ") - ",
            "computed output dims: (",
            out_d,
            ", ",
            out_h,
            ", ",
            out_w,
            ")");

    TORCH_CHECK(
            out_d > 0 && out_h > 0 && out_w > 0,
            "Calculated output size too small - out_d: ",
            out_d,
            " out_h: ",
            out_h,
            " out_w: ",
            out_w);

    auto out = at::zeros({batch_sz, out_channels, out_d, out_h, out_w}, input.options());

    // Separate batches into blocks
    out = out.view({batch_sz / n_parallel_imgs,
                    n_parallel_imgs,
                    out_channels,
                    out_d,
                    out_h,
                    out_w});
    input = input.view(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_d, in_h, in_w});
    offset = offset.view({batch_sz / n_parallel_imgs,
                          n_parallel_imgs,
                          n_offset_grps * 3 * weight_d * weight_h * weight_w,
                          out_d,
                          out_h,
                          out_w});
    if (modulated)
        mask = mask.view({batch_sz / n_parallel_imgs,
                          n_parallel_imgs,
                          n_offset_grps * weight_d * weight_h * weight_w,
                          out_d,
                          out_h,
                          out_w});
    auto out_buf = at::zeros(
            {batch_sz / n_parallel_imgs,
             out_channels,
             n_parallel_imgs,
             out_d,
             out_h,
             out_w},
            out.options());

    // Separate channels into convolution groups
    out_buf = out_buf.view({out_buf.size(0),
                            n_weight_grps,
                            out_buf.size(1) / n_weight_grps,
                            out_buf.size(2),
                            out_buf.size(3),
                            out_buf.size(4),
                            out_buf.size(5)});
    weight = weight.view({n_weight_grps,
                          weight.size(0) / n_weight_grps,
                          weight.size(1),
                          weight.size(2),
                          weight.size(3),
                          weight.size(4)});

    // Sample points and perform convolution
    auto columns = at::zeros(
            {in_channels * weight_d * weight_h * weight_w, n_parallel_imgs * out_d * out_h * out_w},
            input.options());
    columns = columns.view(
            {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
        deform_conv3d_im2col(
                input[b],
                offset[b],
                mask[b],
                in_channels,
                in_d,
                in_h,
                in_w,
                weight_d,
                weight_h,
                weight_w,
                pad_d,
                pad_h,
                pad_w,
                stride_d,
                stride_h,
                stride_w,
                dilation_d,
                dilation_h,
                dilation_w,
                out_d,
                out_h,
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
                            out_d,
                            out_h,
                            out_w});
    out_buf.transpose_(1, 2);
    out.copy_(out_buf);
    out = out.view({batch_sz, out_channels, out_d, out_h, out_w});

    return out + bias.view({1, out_channels, 1, 1, 1});
}

template<typename scalar_t>
__global__ void deform_conv3d_col2im_gpu_kernel(
        const int n,
        const scalar_t *col,
        const scalar_t *offset_ptr,
        const scalar_t *mask_ptr,
        const int channels,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int out_d,
        const int out_h,
        const int out_w,
        const int batch_sz,
        const int n_offset_grps,
        const bool modulated,
        scalar_t *grad_im) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        const int out_x = index % out_w;
        const int out_y = (index / out_w) % out_h;
        const int out_z = (index / (out_w * out_h)) % out_d;
        const int b = (index / (out_w * out_h * out_d)) % batch_sz;
        const int k = (index / (out_w * out_h * out_d * batch_sz)) % weight_w;
        const int j = (index / (out_w * out_h * out_d * batch_sz * weight_w)) % weight_h;
        const int i = (index / (out_w * out_h * out_d * batch_sz * weight_w * weight_h)) % weight_d;
        const int c = index / (out_w * out_h * out_d * batch_sz * weight_w * weight_h * weight_d);

        int c_per_offset_grp = channels / n_offset_grps;
        const int offset_grp = c / c_per_offset_grp;

        offset_ptr +=
                (b * n_offset_grps + offset_grp) * 3 * weight_h * weight_w * weight_d * out_h * out_w * out_d;
        mask_ptr +=
                (b * n_offset_grps + offset_grp) * weight_d * weight_h * weight_w * out_d * out_h * out_w;

        const int offset_idx = 3 * (i * weight_h * weight_w + j * weight_w + k);
        const int offset_d_ptr =
                ((offset_idx * out_d + out_z) * out_h + out_y) * out_w + out_x;
        const int offset_h_ptr =
                (((offset_idx + 1) * out_d + out_z) * out_h + out_y) * out_w + out_x;
        const int offset_w_ptr =
                (((offset_idx + 2) * out_d + out_z) * out_h + out_y) * out_w + out_x;
        const scalar_t offset_d = offset_ptr[offset_d_ptr];
        const scalar_t offset_h = offset_ptr[offset_h_ptr];
        const scalar_t offset_w = offset_ptr[offset_w_ptr];
        const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
        const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
        const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;
        const int mask_idx =
                (((i * weight_h * weight_w + j * weight_w + k) * out_d + out_z) * out_h + out_y) * out_w + out_x;
        const scalar_t mask_weight = modulated ? mask_ptr[mask_idx] : static_cast<scalar_t>(1);

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int zp = int(z) + dz;
                    int yp = int(y) + dy;
                    int xp = int(x) + dx;
                    if (0 <= zp && zp < depth && 0 <= yp && yp < height && 0 <= xp && xp < width &&
                        std::abs(z - zp) < 1 && std::abs(y - yp) < 1 && std::abs(x - xp) < 1) {
                        int grad_pos = (((b * channels + c) * depth + zp) * height + yp) * width + xp;
                        scalar_t weight = (1 - std::abs(z - zp)) * (1 - std::abs(y - yp)) * (1 - std::abs(x - xp));
                        atomicAdd(grad_im + grad_pos, weight * col[index] * mask_weight);
                    }
                }
            }
        }
    }
}

static void deform_conv3d_compute_grad_input(
        const at::Tensor &columns,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const int channels,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int out_d,
        const int out_h,
        const int out_w,
        const int parallel_imgs,
        const int n_offset_grps,
        const bool modulated,
        at::Tensor grad_im) {
    int num_kernels =
            channels * weight_d * weight_h * weight_w * out_d * out_h * out_w * parallel_imgs;

    const unsigned int threads = GET_THREADS(threadsFraction);
    const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            columns.scalar_type(), "deform_conv3d_col2im_gpu", ([&] {
                deform_conv3d_col2im_gpu_kernel<<<blocks, threads>>>(
                        num_kernels,
                        columns.data_ptr<scalar_t>(),
                        offset.data_ptr<scalar_t>(),
                        mask.data_ptr<scalar_t>(),
                        channels,
                        depth,
                        height,
                        width,
                        weight_d,
                        weight_h,
                        weight_w,
                        pad_d,
                        pad_h,
                        pad_w,
                        stride_d,
                        stride_h,
                        stride_w,
                        dilation_d,
                        dilation_h,
                        dilation_w,
                        out_d,
                        out_h,
                        out_w,
                        parallel_imgs,
                        n_offset_grps,
                        modulated,
                        grad_im.data_ptr<scalar_t>());
            }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in deform_conv3d_compute_grad_input: %s\n", cudaGetErrorString(err));
    }
}

template<typename scalar_t>
__global__ void deform_conv3d_col2im_coord_gpu_kernel(
        const int n,
        const scalar_t *col_ptr,
        const scalar_t *im_ptr,
        const scalar_t *offset_ptr,
        const scalar_t *mask_ptr,
        const int channels,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int out_d,
        const int out_h,
        const int out_w,
        const int batch_sz,
        const int offset_channels,
        const int n_offset_grps,
        const bool modulated,
        scalar_t *grad_offset,
        scalar_t *grad_mask) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        scalar_t val = 0, mval = 0;
        int w = index % out_w;
        int h = (index / out_w) % out_h;
        int d = (index / (out_w * out_h)) % out_d;
        int c = (index / (out_w * out_h * out_d)) % offset_channels;
        int b = index / (out_w * out_h * out_d * offset_channels);

        const int offset_grp = c / (3 * weight_d * weight_h * weight_w);
        const int col_step = weight_d * weight_h * weight_w;

        int c_per_offset_grp = channels / n_offset_grps;

        col_ptr +=
                offset_grp * c_per_offset_grp * weight_d * weight_h * weight_w * batch_sz * out_d * out_h * out_w;
        im_ptr +=
                (b * n_offset_grps + offset_grp) * c_per_offset_grp * depth * height * width;
        offset_ptr +=
                (b * n_offset_grps + offset_grp) * 3 * weight_d * weight_h * weight_w * out_d * out_h * out_w;
        mask_ptr +=
                (b * n_offset_grps + offset_grp) * weight_d * weight_h * weight_w * out_d * out_h * out_w;

        const int offset_c = c - offset_grp * 3 * weight_d * weight_h * weight_w;
        const int direction = offset_c % 3;

        const int c_bound = c_per_offset_grp * weight_d * weight_h * weight_w;
        for (int col_c = (offset_c / 3); col_c < c_bound; col_c += col_step) {
            const int col_pos = (((col_c * batch_sz + b) * out_d + d) * out_h + h) * out_w + w;

            int out_x = col_pos % out_w;
            int out_y = (col_pos / out_w) % out_h;
            int out_z = (col_pos / (out_w * out_h)) % out_d;
            int k = (col_pos / (out_w * out_h * out_d * batch_sz)) % weight_w;
            int j = (col_pos / (out_w * out_h * out_d * batch_sz * weight_w)) % weight_h;
            int i = (col_pos / (out_w * out_h * out_d * batch_sz * weight_w * weight_h)) % weight_d;

            const int offset_idx = 3 * (i * weight_h * weight_w + j * weight_w + k);
            const int offset_d_idx =
                    ((offset_idx * out_d + out_z) * out_h + out_y) * out_w + out_x;
            const int offset_h_idx =
                    (((offset_idx + 1) * out_d + out_z) * out_h + out_y) * out_w + out_x;
            const int offset_w_idx =
                    (((offset_idx + 2) * out_d + out_z) * out_h + out_y) * out_w + out_x;
            const scalar_t offset_d = offset_ptr[offset_d_idx];
            const scalar_t offset_h = offset_ptr[offset_h_idx];
            const scalar_t offset_w = offset_ptr[offset_w_idx];

            scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
            scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
            scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

            const int mask_idx =
                    (((i * weight_h * weight_w + j * weight_w + k) * out_d + out_z) * out_h + out_y) * out_w + out_x;
            const scalar_t mask_weight = modulated ? mask_ptr[mask_idx] : static_cast<scalar_t>(1);

            const scalar_t weight =
                    deform_conv3d_get_coordinate_weight(im_ptr, depth, height, width, z, y, x, direction);
            val += weight * col_ptr[col_pos] * mask_weight;
            mval += col_ptr[col_pos] * trilinear_interpolate(im_ptr, depth, height, width, z, y, x);
            im_ptr += depth * height * width;
        }

        grad_offset[index] = val;
        if (modulated && direction == 0) {
            const int grad_mask_idx =
                    ((((b * n_offset_grps + offset_grp) * weight_d * weight_h * weight_w + offset_c / 3) * out_d + d) *
                     out_h + h) * out_w + w;
            grad_mask[grad_mask_idx] = mval;
        }
    }
}

static void deform_conv3d_compute_grad_offset_mask(
        const at::Tensor &columns,
        const at::Tensor &input,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const int channels,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int out_d,
        const int out_h,
        const int out_w,
        const int parallel_imgs,
        const int n_offset_grps,
        const bool modulated,
        at::Tensor &grad_offset,
        at::Tensor &grad_mask) {
    int num_kernels =
            out_d * out_h * out_w * 3 * weight_d * weight_h * weight_w * n_offset_grps * parallel_imgs;

    const unsigned int threads = GET_THREADS(threadsFraction);
    const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            columns.scalar_type(), "deform_conv3d_col2im_coord_gpu", ([&] {
                deform_conv3d_col2im_coord_gpu_kernel<<<blocks, threads>>>(
                        num_kernels,
                        columns.data_ptr<scalar_t>(),
                        input.data_ptr<scalar_t>(),
                        offset.data_ptr<scalar_t>(),
                        mask.data_ptr<scalar_t>(),
                        channels,
                        depth,
                        height,
                        width,
                        weight_d,
                        weight_h,
                        weight_w,
                        pad_d,
                        pad_h,
                        pad_w,
                        stride_d,
                        stride_h,
                        stride_w,
                        dilation_d,
                        dilation_h,
                        dilation_w,
                        out_d,
                        out_h,
                        out_w,
                        parallel_imgs,
                        3 * weight_d * weight_h * weight_w * n_offset_grps,
                        n_offset_grps,
                        modulated,
                        grad_offset.data_ptr<scalar_t>(),
                        grad_mask.data_ptr<scalar_t>());
            }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in deform_conv3d_compute_grad_offset_mask: %s\n", cudaGetErrorString(err));
    }
}

static std::tuple <at::Tensor, at::Tensor, at::Tensor> deform_conv3d_backward_inputs_cuda(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor grad_out,
        const std::tuple<int, int, int> &stride,
        const std::tuple<int, int, int> &pad,
        const std::tuple<int, int, int> &dilation,
        const int n_weight_grps,
        const int n_offset_grps,
        const int n_parallel_imgs,
        const bool modulated) {
    at::DeviceGuard guard(input.device());

    int batch_sz = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int out_channels = weight.size(0);
    int weight_d = weight.size(2);
    int weight_h = weight.size(3);
    int weight_w = weight.size(4);

    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    int pad_d = std::get<0>(pad);
    int pad_h = std::get<1>(pad);
    int pad_w = std::get<2>(pad);

    int dilation_d = std::get<0>(dilation);
    int dilation_h = std::get<1>(dilation);
    int dilation_w = std::get<2>(dilation);

    int out_d = (in_d + 2 * pad_d - (dilation_d * (weight_d - 1) + 1)) / stride_d + 1;
    int out_h = (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);
    auto columns = at::empty(
            {in_channels * weight_d * weight_h * weight_w, n_parallel_imgs * out_d * out_h * out_w},
            input.options());

    // Separate into blocks
    grad_input = grad_input.reshape(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_d, in_h, in_w});
    input = input.reshape(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_d, in_h, in_w});
    grad_offset = grad_offset.reshape({batch_sz / n_parallel_imgs,
                                       n_parallel_imgs,
                                       n_offset_grps * 3 * weight_d * weight_h * weight_w,
                                       out_d,
                                       out_h,
                                       out_w});
    offset = offset.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * 3 * weight_d * weight_h * weight_w,
                             out_d,
                             out_h,
                             out_w});
    if (modulated) {
        grad_mask = grad_mask.reshape({batch_sz / n_parallel_imgs,
                                       n_parallel_imgs,
                                       n_offset_grps * weight_d * weight_h * weight_w,
                                       out_d,
                                       out_h,
                                       out_w});
        mask = mask.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * weight_d * weight_h * weight_w,
                             out_d,
                             out_h,
                             out_w});
    }

    grad_out = grad_out
            .reshape({batch_sz / n_parallel_imgs,
                      n_parallel_imgs,
                      n_weight_grps,
                      out_channels / n_weight_grps,
                      out_d,
                      out_h,
                      out_w})
            .permute({0, 2, 3, 1, 4, 5, 6});

    weight = weight.reshape({n_weight_grps,
                             weight.size(0) / n_weight_grps,
                             weight.size(1),
                             weight.size(2),
                             weight.size(3),
                             weight.size(4)});

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
        deform_conv3d_compute_grad_offset_mask(
                columns,
                input[b],
                offset[b],
                mask[b],
                in_channels,
                in_d,
                in_h,
                in_w,
                weight_d,
                weight_h,
                weight_w,
                pad_d,
                pad_h,
                pad_w,
                stride_d,
                stride_h,
                stride_w,
                dilation_d,
                dilation_h,
                dilation_w,
                out_d,
                out_h,
                out_w,
                n_parallel_imgs,
                n_offset_grps,
                modulated,
                grad_offset_b,
                grad_mask_b);

        deform_conv3d_compute_grad_input(
                columns,
                offset[b],
                mask[b],
                in_channels,
                in_d,
                in_h,
                in_w,
                weight_d,
                weight_h,
                weight_w,
                pad_d,
                pad_h,
                pad_w,
                stride_d,
                stride_h,
                stride_w,
                dilation_d,
                dilation_h,
                dilation_w,
                out_d,
                out_h,
                out_w,
                n_parallel_imgs,
                n_offset_grps,
                modulated,
                grad_input_b);
    }

    grad_input = grad_input.view({batch_sz, in_channels, in_d, in_h, in_w});
    grad_offset = grad_offset.view(
            {batch_sz, n_offset_grps * 3 * weight_d * weight_h * weight_w, out_d, out_h, out_w});
    if (modulated) {
        grad_mask = grad_mask.view(
                {batch_sz, n_offset_grps * weight_d * weight_h * weight_w, out_d, out_h, out_w});
    }

    return std::make_tuple(grad_input, grad_offset, grad_mask);
}

static at::Tensor deform_conv3d_backward_parameters_cuda(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor grad_out,
        const std::tuple<int, int, int> &stride,
        const std::tuple<int, int, int> &pad,
        const std::tuple<int, int, int> &dilation,
        const int n_weight_grps,
        const int n_offset_grps,
        const int n_parallel_imgs,
        const bool modulated) {
    at::DeviceGuard guard(input.device());

    int batch_sz = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int out_channels = weight.size(0);
    int weight_d = weight.size(2);
    int weight_h = weight.size(3);
    int weight_w = weight.size(4);

    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    int pad_d = std::get<0>(pad);
    int pad_h = std::get<1>(pad);
    int pad_w = std::get<2>(pad);

    int dilation_d = std::get<0>(dilation);
    int dilation_h = std::get<1>(dilation);
    int dilation_w = std::get<2>(dilation);

    int out_d = grad_out.size(2);
    int out_h = grad_out.size(3);
    int out_w = grad_out.size(4);

    auto grad_weight = at::zeros_like(weight);

    at::Tensor grad_out_buf = grad_out
            .reshape({batch_sz / n_parallel_imgs,
                      n_parallel_imgs,
                      n_weight_grps,
                      out_channels / n_weight_grps,
                      out_d,
                      out_h,
                      out_w})
            .permute({0, 2, 3, 1, 4, 5, 6})
            .contiguous();

    input = input.reshape(
            {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_d, in_h, in_w});
    offset = offset.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * 3 * weight_d * weight_h * weight_w,
                             out_d,
                             out_h,
                             out_w});
    if (modulated) {
        mask = mask.reshape({batch_sz / n_parallel_imgs,
                             n_parallel_imgs,
                             n_offset_grps * weight_d * weight_h * weight_w,
                             out_d,
                             out_h,
                             out_w});
    }

    grad_weight = grad_weight.view({n_weight_grps,
                                    grad_weight.size(0) / n_weight_grps,
                                    grad_weight.size(1),
                                    grad_weight.size(2),
                                    grad_weight.size(3),
                                    grad_weight.size(4)});

    auto columns = at::empty(
            {n_weight_grps,
             in_channels * weight_d * weight_w * weight_h / n_weight_grps,
             n_parallel_imgs * out_d * out_h * out_w},
            input.options());

    for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
        deform_conv3d_im2col(
                input[b],
                offset[b],
                mask[b],
                in_channels,
                in_d,
                in_h,
                in_w,
                weight_d,
                weight_h,
                weight_w,
                pad_d,
                pad_h,
                pad_w,
                stride_d,
                stride_h,
                stride_w,
                dilation_d,
                dilation_h,
                dilation_w,
                out_d,
                out_h,
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
                                    grad_weight.size(3),
                                    grad_weight.size(4),
                                    grad_weight.size(5)});
    return grad_weight;
}

std::tuple <at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
DeformConv3d_backward_cuda(
        const at::Tensor &grad_out,
        const at::Tensor &input,
        const at::Tensor &weight,
        const at::Tensor &offset,
        const at::Tensor &mask,
        const at::Tensor &bias,
        const std::tuple<int, int, int> &stride,
        const std::tuple<int, int, int> &pad,
        const std::tuple<int, int, int> &dilation,
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

    auto grad_input_and_offset_and_mask = deform_conv3d_backward_inputs_cuda(
            input_c,
            weight_c,
            offset_c,
            mask_c,
            grad_out_c,
            stride,
            pad,
            dilation,
            n_weight_grps,
            n_offset_grps,
            n_parallel_imgs,
            modulated);

    auto grad_input = std::get<0>(grad_input_and_offset_and_mask);
    auto grad_offset = std::get<1>(grad_input_and_offset_and_mask);
    auto grad_mask = std::get<2>(grad_input_and_offset_and_mask);

    auto grad_weight = deform_conv3d_backward_parameters_cuda(
            input_c,
            weight_c,
            offset_c,
            mask_c,
            grad_out_c,
            stride,
            pad,
            dilation,
            n_weight_grps,
            n_offset_grps,
            n_parallel_imgs,
            modulated);

    auto grad_bias = at::ones_like(bias_c) * grad_out_c.sum({0, 2, 3, 4});

    return std::make_tuple(grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
}
