#pragma once

#include <ATen/ATen.h>

void arr2col_cuda(
        const at::Tensor &input,
        const at::Tensor &offset,
        const at::Tensor &mask,
        int in_channels,
        int width,
        int weight_w,
        int pad_w,
        int stride_w,
        int dilation_w,
        int out_w,
        int batch_sz,
        int n_offset_grps,
        int n_mask_grps,
        bool deformable,
        bool modulated,
        at::Tensor &columns);

void col2arr_cuda(
        const at::Tensor &columns,
        const at::Tensor &offset,
        const at::Tensor &mask,
        int in_channels,
        int width,
        int weight_w,
        int pad_w,
        int stride_w,
        int dilation_w,
        int out_w,
        int batch_sz,
        int n_offset_grps,
        int n_mask_grps,
        bool deformable,
        bool modulated,
        at::Tensor &grad_input);

void deform_conv1d_compute_grad_offset_cuda(
        const at::Tensor &columns,
        const at::Tensor &input,
        const at::Tensor &offset,
        const at::Tensor &mask,
        int in_channels,
        int width,
        int weight_w,
        int pad_w,
        int stride_w,
        int dilation_w,
        int out_w,
        int batch_sz,
        int n_offset_grps,
        int n_mask_grps,
        bool deformable,
        bool modulated,
        at::Tensor &grad_offset);

void deform_conv1d_compute_grad_mask_cuda(
        const at::Tensor &columns,
        const at::Tensor &input,
        const at::Tensor &offset,
        int in_channels,
        int width,
        int weight_w,
        int pad_w,
        int stride_w,
        int dilation_w,
        int out_w,
        int batch_sz,
        int n_offset_grps,
        int n_mask_grps,
        bool deformable,
        bool modulated,
        at::Tensor &grad_mask);