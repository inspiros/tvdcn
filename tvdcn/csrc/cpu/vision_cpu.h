#pragma once
#include <torch/extension.h>

at::Tensor DeformConv1d_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int stride,
    int pad,
    int dilation,
    int groups,
    int deformable_groups,
    bool modulated);

at::Tensor DeformConv2d_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    const std::pair<int, int>& stride,
    const std::pair<int, int>& pad,
    const std::pair<int, int>& dilation,
    int groups,
    int deformable_groups,
    bool modulated);

at::Tensor DeformConv3d_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    const std::tuple<int, int, int>& stride,
    const std::tuple<int, int, int>& pad,
    const std::tuple<int, int, int>& dilation,
    int groups,
    int deformable_groups,
    bool modulated);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
DeformConv1d_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int stride,
    int pad,
    int dilation,
    int groups,
    int deformable_groups,
    bool modulated);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
DeformConv2d_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    const std::pair<int, int>& stride,
    const std::pair<int, int>& pad,
    const std::pair<int, int>& dilation,
    int groups,
    int deformable_groups,
    bool modulated);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
DeformConv3d_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    const std::tuple<int, int, int>& stride,
    const std::tuple<int, int, int>& pad,
    const std::tuple<int, int, int>& dilation,
    int groups,
    int deformable_groups,
    bool modulated);
