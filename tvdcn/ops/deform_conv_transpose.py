from typing import List, Tuple, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple

from .activations import (
    MaskSigmoid, MaskSoftmax1d, MaskSoftmax2d, MaskSoftmax3d,
)
from .deform_conv import _DeformConvNd
from .._types import _IntTuple, _Activation
from ..extension import _assert_has_ops
from ..utils import _log_api_usage_once

__all__ = [
    'deform_conv_transpose1d',
    'deform_conv_transpose2d',
    'deform_conv_transpose3d',
    'DeformConvTranspose1d',
    'DeformConvTranspose2d',
    'DeformConvTranspose3d',
    'PackedDeformConvTranspose1d',
    'PackedDeformConvTranspose2d',
    'PackedDeformConvTranspose3d',
]


def deform_conv_transpose1d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1) -> Tensor:
    r"""
    Performs 1D Transposed Deformable Convolution.

    Arguments:
        input (Tensor[batch_size, in_channels, in_width]): input tensor
        weight (Tensor[in_channels, out_channels // groups, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, offset_groups * kernel_width, in_width]): offsets
            to be applied for each position in the convolution kernel. Default: None
        mask (Tensor[batch_size, mask_groups * kernel_width, in_width]): modulation
            masks to be multiplied with each output of convolution kernel. Default: None
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int]): padding of zeroes around the input. Default: 0
        output_padding (int or Tuple[int]): additional size added to one side
            of each dimension in the output shape. Default: 0
        dilation (int or Tuple[int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_w]): result of convolution

    Examples:
        >>> input = torch.rand(4, 3, 8)
        >>> kw = 3
        >>> weight = torch.rand(3, 5, kw)
        >>> # offset and mask should have the same spatial size as the input.
        >>> offset = torch.rand(4, kw, 8)
        >>> mask = torch.rand(4, kw, 8).sigmoid()
        >>> out = deform_conv_transpose1d(input, weight, offset, mask)
        >>> print(out.shape)
        Output:
        >>>  torch.Size([4, 5, 10])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv_transpose1d)
    _assert_has_ops()

    return torch.ops.tvdcn.deform_conv_transpose1d(
        input, weight, offset, mask, bias,
        stride, padding, output_padding, dilation, groups)


def deform_conv_transpose2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1) -> Tensor:
    r"""
    Performs 2D Transposed Deformable Convolution.

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        weight (Tensor[in_channels, out_channels // groups, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            in_height, in_width]): offsets to be applied for each position in the
            convolution kernel. Default: None
        mask (Tensor[batch_size, mask_groups * kernel_height * kernel_width,
            in_height, in_width]): modulation masks to be multiplied with each output
            of convolution kernel. Default: None
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): padding of zeroes around the input. Default: 0
        output_padding (int or Tuple[int, int]): additional size added to one side
            of each dimension in the output shape. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w]): result of convolution

    Examples:
        >>> input = torch.rand(4, 3, 8, 8)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(3, 5, kh, kw)
        >>> # offset and mask should have the same spatial size as the input.
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8).sigmoid()
        >>> out = deform_conv_transpose2d(input, weight, offset, mask)
        >>> print(out.shape)
        Output:
        >>>  torch.Size([4, 5, 10, 10])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv_transpose2d)
    _assert_has_ops()

    return torch.ops.tvdcn.deform_conv_transpose2d(
        input, weight, offset, mask, bias,
        stride, padding, output_padding, dilation, groups)


def deform_conv_transpose3d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1) -> Tensor:
    r"""
    Performs 3D Transposed Deformable Convolution.

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width, in_depth]): input tensor
        weight (Tensor[in_channels, out_channels // groups, kernel_depth, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, 3 * offset_groups * kernel_depth * kernel_height * kernel_width,
            in_depth, in_height, in_width]): offsets to be applied for each position in the
            convolution kernel. Default: None
        mask (Tensor[batch_size, mask_groups * kernel_depth * kernel_height * kernel_width,
            in_depth, in_height, in_width]): modulation masks to be multiplied with each output
            of convolution kernel. Default: None
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int, int]): height/width of padding of zeroes around
            each image. Default: 0
        output_padding (int or Tuple[int, int, int]): additional size added to one side
            of each dimension in the output shape. Default: 0
        dilation (int or Tuple[int, int, int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_d, out_h, out_w]): result of convolution

    Examples:
        >>> input = torch.rand(4, 3, 8, 8, 8)
        >>> kd, kh, kw = 3, 3, 3
        >>> weight = torch.rand(3, 5, kd, kh, kw)
        >>> # offset and mask should have the same spatial size as the input.
        >>> offset = torch.rand(4, 3 * kd * kh * kw, 8, 8, 8)
        >>> mask = torch.rand(4, kd * kh * kw, 8, 8, 8).sigmoid()
        >>> out = deform_conv_transpose3d(input, weight, offset, mask)
        >>> print(out.shape)
        Output:
        >>> torch.Size([4, 5, 10, 10, 10])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv_transpose3d)
    _assert_has_ops()

    return torch.ops.tvdcn.deform_conv_transpose3d(
        input, weight, offset, mask, bias,
        stride, padding, output_padding, dilation, groups)


################################################################################
# Modules
################################################################################
# noinspection PyMethodOverriding
class _DeformConvTransposeNd(_DeformConvNd):
    """
    Base class for DeformConvTranspose
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _IntTuple,
                 stride: _IntTuple,
                 padding: Union[str, _IntTuple],
                 dilation: _IntTuple,
                 transposed: bool,
                 output_padding: Union[str, _IntTuple],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode, **factory_kwargs)

    def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
                        stride: List[int], padding: List[int], kernel_size: List[int],
                        num_spatial_dims: int, dilation: Optional[List[int]] = None) -> List[int]:
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
                    .format(num_spatial_dims, input.dim(), num_spatial_dims,
                            num_non_spatial_dims + num_spatial_dims, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def _conv_transpose_forward(self,
                                input: Tensor,
                                weight: Tensor,
                                offset: Optional[Tensor],
                                mask: Optional[Tensor],
                                bias: Optional[Tensor],
                                output_size: Optional[List[int]] = None) -> Tensor:
        raise NotImplementedError

    def forward(self,
                input: Tensor,
                offset: Optional[Tensor],
                mask: Optional[Tensor] = None,
                output_size: Optional[List[int]] = None) -> Tensor:
        return self._conv_transpose_forward(input, self.weight, offset, mask, self.bias, output_size)


class DeformConvTranspose1d(_DeformConvTransposeNd):
    """
    See :func:`deform_conv_transpose1d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 output_padding: Union[str, _size_1_t] = 0,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)

    def _conv_transpose_forward(self,
                                input: Tensor,
                                weight: Tensor,
                                offset: Optional[Tensor],
                                mask: Optional[Tensor],
                                bias: Optional[Tensor],
                                output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for DeformConvTranspose1d.')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]
        return deform_conv_transpose1d(input, weight, offset, mask, bias,
                                       self.stride, self.padding,
                                       (output_padding[0],),
                                       self.dilation, self.groups)


class DeformConvTranspose2d(_DeformConvTransposeNd):
    """
    See :func:`deform_conv_transpose2d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 output_padding: Union[str, _size_2_t] = 0,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)

    def _conv_transpose_forward(self,
                                input: Tensor,
                                weight: Tensor,
                                offset: Optional[Tensor],
                                mask: Optional[Tensor],
                                bias: Optional[Tensor],
                                output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for DeformConvTranspose2d.')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        return deform_conv_transpose2d(input, weight, offset, mask, bias,
                                       self.stride, self.padding,  # type: ignore[arg-type]
                                       (output_padding[0], output_padding[1]),
                                       self.dilation, self.groups)  # type: ignore[arg-type]


class DeformConvTranspose3d(_DeformConvTransposeNd):
    """
    See :func:`deform_conv_transpose3d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: Union[str, _size_3_t] = 0,
                 dilation: _size_3_t = 1,
                 output_padding: Union[str, _size_3_t] = 0,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)

    def _conv_transpose_forward(self,
                                input: Tensor,
                                weight: Tensor,
                                offset: Optional[Tensor],
                                mask: Optional[Tensor],
                                bias: Optional[Tensor],
                                output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for DeformConvTranspose3d.')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 3
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        return deform_conv_transpose3d(input, weight, offset, mask, bias,
                                       self.stride, self.padding,  # type: ignore[arg-type]
                                       (output_padding[0], output_padding[1], output_padding[2]),
                                       self.dilation, self.groups)  # type: ignore[arg-type]


################################################################################
# Packed Modules
################################################################################
# noinspection PyMethodOverriding
class PackedDeformConvTranspose1d(DeformConvTranspose1d):
    """
    Packed version of :class:`DeformConvTranspose1d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 output_padding: Union[str, _size_1_t] = 0,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 generator_bias: bool = False,
                 deformable: bool = True,
                 modulated: bool = False,
                 offset_activation: Union[str, _Activation] = None,
                 mask_activation: Union[str, _Activation] = 'sigmoid',
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            output_padding, groups, bias, padding_mode, device, dtype)

        if in_channels % offset_groups != 0:
            raise ValueError('in_channels must be divisible by offset_groups')
        if out_channels % offset_groups != 0:
            raise ValueError('out_channels must be divisible by offset_groups')

        if in_channels % mask_groups != 0:
            raise ValueError('in_channels must be divisible by mask_groups')
        if out_channels % mask_groups != 0:
            raise ValueError('out_channels must be divisible by mask_groups')

        if isinstance(offset_activation, str):
            raise ValueError('currently, no activation is supported for offset')
        if isinstance(mask_activation, str):
            if mask_activation == 'sigmoid':
                mask_activation = MaskSigmoid(scale=2.)
            elif mask_activation == 'softmax':
                mask_activation = MaskSoftmax1d(self.kernel_size)  # type: ignore[arg-type]
            else:
                raise ValueError('only \"sigmoid\" and \"softmax\" activations are supported for mask')

        self.offset_groups = offset_groups
        self.mask_groups = mask_groups

        self.deformable = deformable
        self.modulated = modulated

        if self.deformable:
            self.offset_generator = nn.Conv1d(
                self.in_channels,
                self.kernel_size[0] * self.offset_groups,
                kernel_size=1,
                groups=self.offset_groups,
                bias=generator_bias,
                device=device,
                dtype=dtype)
        else:
            self.register_module('offset_generator', None)

        if self.modulated:
            self.mask_generator = nn.Conv1d(
                self.in_channels,
                self.kernel_size[0] * self.mask_groups,
                kernel_size=1,
                groups=self.mask_groups,
                bias=generator_bias,
                device=device,
                dtype=dtype)
        else:
            self.register_module('mask_generator', None)

        self.offset_activation = offset_activation
        self.mask_activation = mask_activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not hasattr(self, 'modulated'):
            return
        super().reset_parameters()
        if self.offset_generator is not None:
            init.zeros_(self.offset_generator.weight)
            if self.offset_generator.bias is not None:
                init.zeros_(self.offset_generator.bias)
        if self.mask_generator is not None:
            init.zeros_(self.mask_generator.weight)
            if self.mask_generator.bias is not None:
                init.zeros_(self.mask_generator.bias)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.deformable and self.offset_generator is not None:
            offset = self.offset_generator(input)
            if self.offset_activation is not None:
                offset = self.offset_activation(offset)
        else:
            offset = None

        if self.modulated and self.mask_generator is not None:
            mask = self.mask_generator(input)
            if self.mask_activation is not None:
                mask = self.mask_activation(mask)
        else:
            mask = None

        return self._conv_transpose_forward(input, self.weight, offset, mask, self.bias, output_size)


# noinspection PyMethodOverriding
class PackedDeformConvTranspose2d(DeformConvTranspose2d):
    """
    Packed version of :class:`DeformConvTranspose2d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 output_padding: Union[str, _size_2_t] = 0,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 generator_bias: bool = False,
                 deformable: bool = True,
                 modulated: bool = False,
                 offset_activation: Union[str, _Activation] = None,
                 mask_activation: Union[str, _Activation] = 'sigmoid',
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            output_padding, groups, bias, padding_mode, device, dtype)

        if in_channels % offset_groups != 0:
            raise ValueError('in_channels must be divisible by offset_groups')
        if out_channels % offset_groups != 0:
            raise ValueError('out_channels must be divisible by offset_groups')

        if in_channels % mask_groups != 0:
            raise ValueError('in_channels must be divisible by mask_groups')
        if out_channels % mask_groups != 0:
            raise ValueError('out_channels must be divisible by mask_groups')

        if isinstance(offset_activation, str):
            raise ValueError('currently, no activation is supported for offset')
        if isinstance(mask_activation, str):
            if mask_activation == 'sigmoid':
                mask_activation = MaskSigmoid(scale=2.)
            elif mask_activation == 'softmax':
                mask_activation = MaskSoftmax2d(self.kernel_size)  # type: ignore[arg-type]
            else:
                raise ValueError('only \"sigmoid\" and \"softmax\" activations are supported for mask')

        self.offset_groups = offset_groups
        self.mask_groups = mask_groups

        self.deformable = deformable
        self.modulated = modulated

        if self.deformable:
            self.offset_generator = nn.Conv2d(
                self.in_channels,
                2 * self.kernel_size[0] * self.kernel_size[1] * self.offset_groups,
                kernel_size=1,
                groups=self.offset_groups,
                bias=generator_bias,
                device=device,
                dtype=dtype)
        else:
            self.register_module('offset_generator', None)

        if self.modulated:
            self.mask_generator = nn.Conv2d(
                self.in_channels,
                self.kernel_size[0] * self.kernel_size[1] * self.mask_groups,
                kernel_size=1,
                groups=self.mask_groups,
                bias=generator_bias,
                device=device,
                dtype=dtype)
        else:
            self.register_module('mask_generator', None)

        self.offset_activation = offset_activation
        self.mask_activation = mask_activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not hasattr(self, 'modulated'):
            return
        super().reset_parameters()
        if self.offset_generator is not None:
            init.zeros_(self.offset_generator.weight)
            if self.offset_generator.bias is not None:
                init.zeros_(self.offset_generator.bias)
        if self.mask_generator is not None:
            init.zeros_(self.mask_generator.weight)
            if self.mask_generator.bias is not None:
                init.zeros_(self.mask_generator.bias)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.deformable and self.offset_generator is not None:
            offset = self.offset_generator(input)
            if self.offset_activation is not None:
                offset = self.offset_activation(offset)
        else:
            offset = None

        if self.modulated and self.mask_generator is not None:
            mask = self.mask_generator(input)
            if self.mask_activation is not None:
                mask = self.mask_activation(mask)
        else:
            mask = None

        return self._conv_transpose_forward(input, self.weight, offset, mask, self.bias, output_size)


# noinspection PyMethodOverriding
class PackedDeformConvTranspose3d(DeformConvTranspose3d):
    """
    Packed version of :class:`DeformConvTranspose3d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: Union[str, _size_3_t] = 0,
                 dilation: _size_3_t = 1,
                 output_padding: Union[str, _size_3_t] = 0,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 generator_bias: bool = False,
                 deformable: bool = True,
                 modulated: bool = False,
                 offset_activation: Union[str, _Activation] = None,
                 mask_activation: Union[str, _Activation] = 'sigmoid',
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            output_padding, groups, bias, padding_mode, device, dtype)

        if in_channels % offset_groups != 0:
            raise ValueError('in_channels must be divisible by offset_groups')
        if out_channels % offset_groups != 0:
            raise ValueError('out_channels must be divisible by offset_groups')

        if in_channels % mask_groups != 0:
            raise ValueError('in_channels must be divisible by mask_groups')
        if out_channels % mask_groups != 0:
            raise ValueError('out_channels must be divisible by mask_groups')

        if isinstance(offset_activation, str):
            raise ValueError('currently, no activation is supported for offset')
        if isinstance(mask_activation, str):
            if mask_activation == 'sigmoid':
                mask_activation = MaskSigmoid(scale=2.)
            elif mask_activation == 'softmax':
                mask_activation = MaskSoftmax3d(self.kernel_size)  # type: ignore[arg-type]
            else:
                raise ValueError('only \"sigmoid\" and \"softmax\" activations are supported for mask')

        self.offset_groups = offset_groups
        self.mask_groups = mask_groups

        self.deformable = deformable
        self.modulated = modulated

        if self.deformable:
            self.offset_generator = nn.Conv3d(
                self.in_channels,
                3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.offset_groups,
                kernel_size=1,
                groups=self.offset_groups,
                bias=generator_bias,
                device=device,
                dtype=dtype)
        else:
            self.register_module('offset_generator', None)

        if self.modulated:
            self.mask_generator = nn.Conv3d(
                self.in_channels,
                self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.mask_groups,
                kernel_size=1,
                groups=self.mask_groups,
                bias=generator_bias,
                device=device,
                dtype=dtype)
        else:
            self.register_module('mask_generator', None)

        self.offset_activation = offset_activation
        self.mask_activation = mask_activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not hasattr(self, 'modulated'):
            return
        super().reset_parameters()
        if self.offset_generator is not None:
            init.zeros_(self.offset_generator.weight)
            if self.offset_generator.bias is not None:
                init.zeros_(self.offset_generator.bias)
        if self.mask_generator is not None:
            init.zeros_(self.mask_generator.weight)
            if self.mask_generator.bias is not None:
                init.zeros_(self.mask_generator.bias)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.deformable and self.offset_generator is not None:
            offset = self.offset_generator(input)
            if self.offset_activation is not None:
                offset = self.offset_activation(offset)
        else:
            offset = None

        if self.modulated and self.mask_generator is not None:
            mask = self.mask_generator(input)
            if self.mask_activation is not None:
                mask = self.mask_activation(mask)
        else:
            mask = None

        return self._conv_transpose_forward(input, self.weight, offset, mask, self.bias, output_size)
