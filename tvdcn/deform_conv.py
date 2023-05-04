import math

import torch
from torch import nn, Tensor
from torch.jit.annotations import Optional, Tuple, Union
from torch.nn import init
from torch.nn.modules.utils import _single, _pair, _triple, _ntuple
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
from ._types import IntTuple

import torch.nn.functional as F

from .extension import _assert_has_ops

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
        stride: IntTuple = 1,
        padding: IntTuple = 0,
        dilation: IntTuple = 1,
        groups: int = 1) -> Tensor:
    r"""
    Performs 1D version of Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs 1D version of Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel. Default: None
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
            out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel. Default: None
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w]): result of convolution
    Examples::
        >>> input = torch.rand(1, 3, 10)
        >>> kw = 3
        >>> weight = torch.rand(5, 3, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(5, kw, 8)
        >>> mask = torch.rand(5, kw, 8).sigmoid()
        >>> out = deform_conv1d(input, weight, offset, mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([1, 5, 8])
    """

    _assert_has_ops()
    out_channels = weight.shape[0]

    deformable = offset is not None
    modulated = mask is not None

    if offset is None:
        offset = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride = _single(stride)
    pad = _single(padding)
    dil = _single(dilation)

    weight_w = weight.shape[-1]
    _, n_in_channels, in_w = input.shape

    n_offset_grps = offset.shape[1] // weight_w
    n_mask_grps = mask.shape[1] // weight_w
    n_weight_grps = n_in_channels // weight.shape[1]

    assert n_weight_grps == groups
    if deformable and n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of weight.size[2].\n"
            "Got offset.shape[1]={}, while weight.size[2]={}".format(
                offset.shape[1], weight_w))
    if modulated and n_mask_grps == 0:
        raise RuntimeError(
            "the shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of weight.size[2].\n"
            "Got mask.shape[1]={}, while weight.size[2]={}".format(
                mask.shape[1], weight_w))

    return torch.ops.tvdcn.deform_conv1d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride[0],
        pad[0],
        dil[0],
        n_weight_grps,
        n_offset_grps,
        n_mask_grps,
        deformable,
        modulated)


def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: IntTuple = (1, 1),
        padding: IntTuple = (0, 0),
        dilation: IntTuple = (1, 1),
        groups: int = 1) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

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
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w]): result of convolution
    Examples::
        >>> input = torch.rand(1, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(5, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(5, kh * kw, 8, 8).sigmoid()
        >>> kernel_offset = torch.randn(5, 2, 8, 8).sigmoid()
        >>> out = deform_conv2d(input, weight, offset, mask, kernel_offset)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([1, 5, 8, 8])
    """

    _assert_has_ops()
    out_channels = weight.shape[0]

    deformable = offset is not None
    modulated = mask is not None

    if offset is None:
        offset = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weight_h, weight_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (2 * weight_h * weight_w)
    n_mask_grps = mask.shape[1] // (weight_h * weight_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    assert n_weight_grps == groups
    if deformable and n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                offset.shape[1], 2 * weight_h * weight_w))
    if modulated and n_mask_grps == 0:
        raise RuntimeError(
            "the shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of weight.size[2] * weight.size[3].\n"
            "Got mask.shape[1]={}, while weight.size[2] * weight.size[3]={}".format(
                mask.shape[1], weight_h * weight_w))

    return torch.ops.tvdcn.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        n_mask_grps,
        deformable,
        modulated)


def deform_conv3d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: IntTuple = (1, 1, 1),
        padding: IntTuple = (0, 0, 0),
        dilation: IntTuple = (1, 1, 1),
        groups: int = 1) -> Tensor:
    r"""
    Performs 3D version of Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs 3D version of Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width, in_depth]): input tensor
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        offset (Tensor[batch_size, 3 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        mask (Tensor[batch_size, 3 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel.
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int, int]): the spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w, out_d]): result of convolution
    Examples::
        >>> input = torch.rand(1, 3, 10, 10, 10)
        >>> kd, kh, kw = 3, 3, 3
        >>> weight = torch.rand(5, 3, kd, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(5, 3 * kd * kh * kw, 8, 8, 8)
        >>> mask = torch.rand(5, kd * kh * kw, 8, 8, 8)
        >>> out = deform_conv3d(input, weight, offset, mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([1, 5, 8, 8, 8])
    """

    _assert_has_ops()
    out_channels = weight.shape[0]

    deformable = offset is not None
    modulated = mask is not None

    if offset is None:
        offset = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_d, stride_h, stride_w = _triple(stride)
    pad_d, pad_h, pad_w = _triple(padding)
    dil_d, dil_h, dil_w = _triple(dilation)
    weight_d, weight_h, weight_w = weight.shape[-3:]
    _, n_in_channels, in_d, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (3 * weight_d * weight_h * weight_w)
    n_mask_grps = mask.shape[1] // (weight_d * weight_h * weight_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    assert n_weight_grps == groups
    if deformable and n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 3 * weight.size[2] * weight.size[3] * weight.size[4].\n"
            "Got offset.shape[1]={}, while 3 * weight.size[2] * weight.size[3] * weight.size[4]={}".format(
                offset.shape[1], 3 * weight_d * weight_h * weight_w))
    if modulated and n_mask_grps == 0:
        raise RuntimeError(
            "the shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of weight.size[2] * weight.size[3] * weight.size[4].\n"
            "Got offset.shape[1]={}, while weight.size[2] * weight.size[3] * weight.size[4]={}".format(
                mask.shape[1], weight_d * weight_h * weight_w))

    return torch.ops.tvdcn.deform_conv3d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        n_mask_grps,
        deformable,
        modulated)


# noinspection PyMethodOverriding
class _DeformConvNd(_ConvNd):
    """
    Base class for DeformConv
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntTuple,
                 stride: IntTuple,
                 padding: Union[IntTuple, str],
                 dilation: IntTuple,
                 transposed: bool,
                 output_padding: Union[IntTuple, str],
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
                      offset: Tensor,
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]):
        raise NotImplementedError

    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self._conv_forward(input, self.weight, offset, mask, self.bias)

    def __repr__(self) -> str:
        num_spatial_dims = self.weight.ndim - 2
        n_tuple = _ntuple(num_spatial_dims)
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != n_tuple(0) else ''
        s += ', dilation={dilation}' if self.dilation != n_tuple(1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class DeformConv1d(_DeformConvNd):
    """
    See :func:`deform_conv1d`
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntTuple,
                 stride: IntTuple = 1,
                 padding: Union[IntTuple, str] = 0,
                 dilation: IntTuple = 1,
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
                      offset: Tensor,
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return deform_conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 weight, offset, mask, bias, self.stride,
                                 _single(0), self.dilation, self.groups)
        return deform_conv1d(input, weight, offset, mask, bias,
                             self.stride, self.padding, self.dilation, self.groups)


class DeformConv2d(_DeformConvNd):
    """
    See :func:`deform_conv2d`
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntTuple,
                 stride: IntTuple = 1,
                 padding: Union[IntTuple, str] = 0,
                 dilation: IntTuple = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
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
                      offset: Tensor,
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return deform_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 weight, offset, mask, bias, self.stride,
                                 _single(0), self.dilation, self.groups)
        return deform_conv2d(input, weight, offset, mask, bias,
                             self.stride, self.padding, self.dilation, self.groups)


class DeformConv3d(_DeformConvNd):
    """
    See :func:`deform_conv3d`
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntTuple,
                 stride: IntTuple = 1,
                 padding: Union[IntTuple, str] = 0,
                 dilation: IntTuple = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
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
                      offset: Tensor,
                      mask: Optional[Tensor],
                      bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return deform_conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 weight, offset, mask, bias, self.stride,
                                 _single(0), self.dilation, self.groups)
        return deform_conv3d(input, weight, offset, mask, bias,
                             self.stride, self.padding, self.dilation, self.groups)


# noinspection PyMethodOverriding
class PackedDeformConv1d(DeformConv1d):
    """
    See DeformConv1d
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntTuple,
                 stride: IntTuple = 1,
                 padding: Union[IntTuple, str] = 0,
                 dilation: IntTuple = 1,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 deformable: bool = True,
                 modulated: bool = False,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
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

        self.offset_groups = offset_groups
        self.mask_groups = mask_groups
        self.deformable = deformable
        self.modulated = modulated

        self.conv_offset = nn.Conv1d(
            self.in_channels,
            self.kernel_size[0] * self.offset_groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True)

        self.conv_mask = nn.Conv1d(
            self.in_channels,
            self.kernel_size[0] * self.mask_groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True) if self.modulated else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not hasattr(self, 'modulated'):
            return
        super().reset_parameters()
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        if self.modulated:
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        """
        Arguments:
            input (Tensor[batch_size, in_channels, in_width]): input tensor
        """
        offset = self.conv_offset(input)
        mask = self.conv_mask(input).sigmoid() if self.modulated else None
        return super().forward(input, offset, mask)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != 0 else ''
        s += ', dilation={dilation}' if self.dilation != 1 else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', offset_groups={offset_groups}' if self.offset_groups != 1 else ''
        s += ', mask_groups={mask_groups}' if self.mask_groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ', modulated=True' if self.modulated else ''
        s += ')'
        return s.format(**self.__dict__)


# noinspection PyMethodOverriding
class PackedDeformConv2d(DeformConv2d):
    """
    See DeformConv2d
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntTuple,
                 stride: IntTuple = 1,
                 padding: Union[IntTuple, str] = 0,
                 dilation: IntTuple = 1,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 deformable: bool = True,
                 modulated: bool = False,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
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

        self.offset_groups = offset_groups
        self.mask_groups = mask_groups
        self.deformable = deformable
        self.modulated = modulated

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            2 * self.kernel_size[0] * self.kernel_size[1] * self.offset_groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True) if self.deformable else None

        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.kernel_size[0] * self.kernel_size[1] * self.mask_groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True) if self.modulated else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not hasattr(self, 'modulated'):
            return
        super(PackedDeformConv2d, self).reset_parameters()
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        if self.conv_mask is not None:
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        """
        Arguments:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        offset = self.conv_offset(input) if self.deformable else None
        mask = self.conv_mask(input).sigmoid() if self.modulated else None
        return super().forward(input, offset, mask)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', offset_groups={offset_groups}' if self.offset_groups != 1 else ''
        s += ', mask_groups={mask_groups}' if self.mask_groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ', modulated=True' if self.modulated else ''
        s += ')'
        return s.format(**self.__dict__)


# noinspection PyMethodOverriding
class PackedDeformConv3d(DeformConv3d):
    """
    See :class:`DeformConv3d`
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntTuple,
                 stride: IntTuple = 1,
                 padding: Union[IntTuple, str] = 0,
                 dilation: IntTuple = 1,
                 groups: int = 1,
                 offset_groups: int = 1,
                 mask_groups: int = 1,
                 bias: bool = True,
                 deformable: bool = True,
                 modulated: bool = False,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
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

        self.offset_groups = offset_groups
        self.mask_groups = mask_groups
        self.deformable = deformable
        self.modulated = modulated

        self.conv_offset = nn.Conv3d(
            self.in_channels,
            3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.offset_groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)

        self.conv_mask = nn.Conv3d(
            self.in_channels,
            self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.mask_groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True) if self.modulated else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not hasattr(self, 'modulated'):
            return
        super().reset_parameters()
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        if self.conv_mask is not None:
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        """
        Arguments:
            input (Tensor[batch_size, in_channels, in_height, in_width, in_depth]): input tensor
        """
        offset = self.conv_offset(input) if self.deformable else None
        mask = self.conv_mask(input).sigmoid() if self.modulated else None
        return super().forward(input, offset, mask)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', offset_groups={offset_groups}' if self.offset_groups != 1 else ''
        s += ', mask_groups={mask_groups}' if self.mask_groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ', modulated=True' if self.modulated else ''
        s += ')'
        return s.format(**self.__dict__)
