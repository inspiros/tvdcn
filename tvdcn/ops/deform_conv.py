from typing import Callable, Union, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import init
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple

from .activations import (
    MaskSigmoid, MaskSoftmax1d, MaskSoftmax2d, MaskSoftmax3d,
)
from ..extension import _assert_has_ops
from ..utils import _log_api_usage_once

__all__ = [
    'deform_conv1d',
    'deform_conv2d',
    'deform_conv3d',
    'DeformConv1d',
    'DeformConv2d',
    'DeformConv3d',
    'PackedDeformConv1d',
    'PackedDeformConv2d',
    'PackedDeformConv3d',
]


def deform_conv1d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1) -> Tensor:
    r"""
    Performs 1D Deformable Convolution.

    Arguments:
        input (Tensor[batch_size, in_channels, in_width]): input tensor
        weight (Tensor[out_channels, in_channels // groups, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, offset_groups * kernel_width, out_width]): offsets
            to be applied for each position in the convolution kernel. Default: None
        mask (Tensor[batch_size, mask_groups * kernel_width, out_width]): modulation
            masks to be multiplied with each output of convolution kernel. Default: None
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int]): padding of zeroes around the input. Default: 0
        dilation (int or Tuple[int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_w]): result of convolution

    Examples:
        >>> input = torch.rand(4, 3, 10)
        >>> kw = 3
        >>> weight = torch.rand(5, 3, kw)
        >>> # offset and mask should have the same spatial size as the output.
        >>> # In this case, for an input of 10, stride of 1 and kernel size of 3,
        >>> # without padding, the output size is 8.
        >>> offset = torch.rand(4, kw, 8)
        >>> mask = torch.rand(4, kw, 8).sigmoid()
        >>> out = deform_conv1d(input, weight, offset, mask)
        >>> print(out.shape)
        Output:
        >>>  torch.Size([4, 5, 8])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv1d)
    _assert_has_ops()

    return torch.ops.tvdcn.deform_conv1d(
        input, weight, offset, mask, bias,
        stride, padding, dilation, groups)


def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1) -> Tensor:
    r"""
    Performs 2D Deformable Convolution.

    References:
        https://arxiv.org/abs/1811.11168
        https://arxiv.org/abs/1703.06211
        https://arxiv.org/abs/2211.05778

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel. Default: None
        mask (Tensor[batch_size, mask_groups * kernel_height * kernel_width,
            out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel. Default: None
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): padding of zeroes around the input. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w]): result of convolution

    Examples:
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output.
        >>> # In this case, for an input of 10, stride of 1 and kernel size of 3,
        >>> # without padding, the output size is 8.
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8).sigmoid()
        >>> out = deform_conv2d(input, weight, offset, mask)
        >>> print(out.shape)
        Output:
        >>>  torch.Size([4, 5, 8, 8])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv2d)
    _assert_has_ops()

    return torch.ops.tvdcn.deform_conv2d(
        input, weight, offset, mask, bias,
        stride, padding, dilation, groups)


def deform_conv3d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1) -> Tensor:
    r"""
    Performs 3D Deformable Convolution.

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width, in_depth]): input tensor
        weight (Tensor[out_channels, in_channels // groups, kernel_depth, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, 3 * offset_groups * kernel_depth * kernel_height * kernel_width,
            out_depth, out_height, out_width]): offsets to be applied for each position in the
            convolution kernel. Default: None
        mask (Tensor[batch_size, mask_groups * kernel_depth * kernel_height * kernel_width,
            out_depth, out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel. Default: None
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int, int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_d, out_h, out_w]): result of convolution

    Examples:
        >>> input = torch.rand(4, 3, 10, 10, 10)
        >>> kd, kh, kw = 3, 3, 3
        >>> weight = torch.rand(5, 3, kd, kh, kw)
        >>> # offset and mask should have the same spatial size as the output.
        >>> # In this case, for an input of 10, stride of 1 and kernel size of 3,
        >>> # without padding, the output size is 8.
        >>> offset = torch.rand(4, 3 * kd * kh * kw, 8, 8, 8)
        >>> mask = torch.rand(4, kd * kh * kw, 8, 8, 8).sigmoid()
        >>> out = deform_conv3d(input, weight, offset, mask)
        >>> print(out.shape)
        Output:
        >>> torch.Size([4, 5, 8, 8, 8])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv3d)
    _assert_has_ops()

    return torch.ops.tvdcn.deform_conv3d(
        input, weight, offset, mask, bias,
        stride, padding, dilation, groups)


################################################################################
# Modules
################################################################################
# noinspection PyMethodOverriding
class _DeformConvNd(_ConvNd):
    """
    Base class for DeformConv
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 stride: _size_any_t,
                 padding: Union[str, _size_any_t],
                 dilation: _size_any_t,
                 transposed: bool,
                 output_padding: _size_any_t,
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

    def _conv_forward(self,
                      input: Tensor,
                      weight: Tensor,
                      offset: Optional[Tensor],
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor, offset: Optional[Tensor], mask: Optional[Tensor] = None) -> Tensor:
        return self._conv_forward(input, self.weight, offset, mask, self.bias)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if hasattr(self, 'offset_groups') and self.offset_groups != 1:
            s += ', offset_groups={offset_groups}'
        if hasattr(self, 'mask_groups') and self.mask_groups != 1:
            s += ', mask_groups={mask_groups}'
        if self.bias is None:
            s += ', bias=False'
        if hasattr(self, 'deformable') and not self.deformable:
            s += ', deformable=False'
        if hasattr(self, 'modulated') and not self.modulated:
            s += ', modulated=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class DeformConv1d(_DeformConvNd):
    """
    See :func:`deform_conv1d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self,
                      input: Tensor,
                      weight: Tensor,
                      offset: Optional[Tensor],
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]) -> Tensor:
        if self.padding_mode != 'zeros':
            return deform_conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 weight, offset, mask, bias, self.stride,
                                 (0,), self.dilation, self.groups)
        return deform_conv1d(input, weight, offset, mask, bias,
                             self.stride, self.padding, self.dilation, self.groups)


class DeformConv2d(_DeformConvNd):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self,
                      input: Tensor,
                      weight: Tensor,
                      offset: Optional[Tensor],
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]) -> Tensor:
        if self.padding_mode != 'zeros':
            return deform_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 weight, offset, mask, bias, self.stride,  # type: ignore[arg-type]
                                 (0, 0), self.dilation, self.groups)  # type: ignore[arg-type]
        return deform_conv2d(input, weight, offset, mask, bias,
                             self.stride, self.padding, self.dilation, self.groups)  # type: ignore[arg-type]


class DeformConv3d(_DeformConvNd):
    """
    See :func:`deform_conv3d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: Union[str, _size_3_t] = 0,
                 dilation: _size_3_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _triple(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self,
                      input: Tensor,
                      weight: Tensor,
                      offset: Optional[Tensor],
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]) -> Tensor:
        if self.padding_mode != 'zeros':
            return deform_conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 weight, offset, mask, bias, self.stride,  # type: ignore[arg-type]
                                 (0, 0, 0), self.dilation, self.groups)  # type: ignore[arg-type]
        return deform_conv3d(input, weight, offset, mask, bias,
                             self.stride, self.padding, self.dilation, self.groups)  # type: ignore[arg-type]


################################################################################
# Packed Modules
################################################################################
# noinspection PyMethodOverriding
class PackedDeformConv1d(DeformConv1d):
    """
    Packed version of :class:`DeformConv1d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 generator_bias: bool = False,
                 deformable: bool = True,
                 modulated: bool = False,
                 offset_activation: Optional[Union[str, Callable]] = None,
                 mask_activation: Optional[Union[str, Callable]] = 'sigmoid',
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

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
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
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
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
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

    def forward(self, input: Tensor) -> Tensor:
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

        return self._conv_forward(input, self.weight, offset, mask, self.bias)


# noinspection PyMethodOverriding
class PackedDeformConv2d(DeformConv2d):
    """
    Packed version of :class:`DeformConv2d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 generator_bias: bool = False,
                 deformable: bool = True,
                 modulated: bool = False,
                 offset_activation: Optional[Union[str, Callable]] = None,
                 mask_activation: Optional[Union[str, Callable]] = 'sigmoid',
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

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
                kernel_size=self.kernel_size,  # type: ignore[arg-type]
                stride=self.stride,  # type: ignore[arg-type]
                padding=self.padding,
                dilation=self.dilation,  # type: ignore[arg-type]
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
                kernel_size=self.kernel_size,  # type: ignore[arg-type]
                stride=self.stride,  # type: ignore[arg-type]
                padding=self.padding,
                dilation=self.dilation,  # type: ignore[arg-type]
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

    def forward(self, input: Tensor) -> Tensor:
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

        return self._conv_forward(input, self.weight, offset, mask, self.bias)


# noinspection PyMethodOverriding
class PackedDeformConv3d(DeformConv3d):
    """
    Packed version of :class:`DeformConv3d`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: Union[str, _size_3_t] = 0,
                 dilation: _size_3_t = 1,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 generator_bias: bool = False,
                 deformable: bool = True,
                 modulated: bool = False,
                 offset_activation: Optional[Union[str, Callable]] = None,
                 mask_activation: Optional[Union[str, Callable]] = 'sigmoid',
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

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
                kernel_size=self.kernel_size,  # type: ignore[arg-type]
                stride=self.stride,  # type: ignore[arg-type]
                padding=self.padding,
                dilation=self.dilation,  # type: ignore[arg-type]
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
                kernel_size=self.kernel_size,  # type: ignore[arg-type]
                stride=self.stride,  # type: ignore[arg-type]
                padding=self.padding,
                dilation=self.dilation,  # type: ignore[arg-type]
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

    def forward(self, input: Tensor) -> Tensor:
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

        return self._conv_forward(input, self.weight, offset, mask, self.bias)
