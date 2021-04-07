import math

import torch
from torch import nn, Tensor
from torch.jit.annotations import Optional, Tuple
from torch.nn import init
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from torchvision.extension import _assert_has_ops

__all__ = ['deform_conv1d',
           'deform_conv2d',
           'deform_conv3d',
           'DeformConv1d',
           'PackedDeformConv1d',
           'DeformConv2d',
           'PackedDeformConv2d',
           'DeformConv3d',
           'PackedDeformConv3d']


def deform_conv1d(
        input: Tensor,
        weight: Tensor,
        offset: Tensor,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1
) -> Tensor:
    r"""
    Performs 1D Deformable Convolution v2, described in
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
            convolution kernel.
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
            out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel.
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int]): the spacing between kernel elements. Default: 1
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

    modulated = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride = _single(stride)
    pad = _single(padding)
    dil = _single(dilation)

    weights_w = weight.shape[-1]
    _, n_in_channels, in_w = input.shape

    n_offset_grps = offset.shape[1] // weights_w
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of weight.size[2].\n"
            "Got offset.shape[1]={}, while weight.size[2]={}".format(
                offset.shape[1], weights_w))

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
        modulated)


def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Tensor,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1)
) -> Tensor:
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
            convolution kernel.
        mask (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel.
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
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
        >>> out = deform_conv2d(input, weight, offset, mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([1, 5, 8, 8])
    """

    _assert_has_ops()
    out_channels = weight.shape[0]

    modulated = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                offset.shape[1], 2 * weights_h * weights_w))

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
        modulated)


def deform_conv3d(
        input: Tensor,
        weight: Tensor,
        offset: Tensor,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1)
) -> Tensor:
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

    modulated = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_d, stride_h, stride_w = _triple(stride)
    pad_d, pad_h, pad_w = _triple(padding)
    dil_d, dil_h, dil_w = _triple(dilation)
    weights_d, weights_h, weights_w = weight.shape[-3:]
    _, n_in_channels, in_d, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (3 * weights_d * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 3 * weight.size[2] * weight.size[3] * weight.size[4].\n"
            "Got offset.shape[1]={}, while 3 * weight.size[2] * weight.size[3] * weight.size[4]={}".format(
                offset.shape[1], 3 * weights_d * weights_h * weights_w))

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
        modulated)


class DeformConv1d(nn.Module):
    """
    See deform_conv1d
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ):
        super(DeformConv1d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0]), requires_grad=True)

        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_width]): input tensor
            offset (Tensor[batch_size, 1 * offset_groups * kernel_width, out_width]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_width, out_width]):
                masks to be applied for each position in the convolution kernel.
        """
        return deform_conv1d(input, self.weight, offset, mask, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != 0 else ''
        s += ', dilation={dilation}' if self.dilation != 1 else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class PackedDeformConv1d(DeformConv1d):
    """
    See DeformConv1d
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            modulated: bool = False
    ):
        super(PackedDeformConv1d, self).__init__(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 groups,
                                                 bias)
        self.modulated = modulated

        self.conv_offset = nn.Conv1d(
            self.in_channels,
            self.kernel_size[0],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)

        self.conv_mask = nn.Conv1d(
            self.in_channels,
            self.kernel_size[0],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True) if self.modulated else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super(PackedDeformConv1d, self).reset_parameters()

        if hasattr(self, 'modulated'):
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
        return deform_conv1d(input, self.weight, offset, mask, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != 0 else ''
        s += ', dilation={dilation}' if self.dilation != 1 else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ', modulated=True' if self.modulated else ''
        s += ')'
        return s.format(**self.__dict__)


class DeformConv2d(nn.Module):
    """
    See deform_conv2d
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ):
        super(DeformConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0], self.kernel_size[1]), requires_grad=True)

        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
                out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
                out_height, out_width]): masks to be applied for each position in the
                convolution kernel.
        """
        return deform_conv2d(input, self.weight, offset, mask, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class PackedDeformConv2d(DeformConv2d):
    """
    See DeformConv2d
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            modulated: bool = False
    ):
        super(PackedDeformConv2d, self).__init__(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 groups,
                                                 bias)
        self.modulated = modulated

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)

        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True) if self.modulated else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super(PackedDeformConv2d, self).reset_parameters()

        if hasattr(self, 'modulated'):
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
        offset = self.conv_offset(input)
        mask = self.conv_mask(input).sigmoid() if self.modulated else None
        return deform_conv2d(input, self.weight, offset, mask, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ', modulated=True' if self.modulated else ''
        s += ')'
        return s.format(**self.__dict__)


class DeformConv3d(nn.Module):
    """
    See deform_conv3d
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ):
        super(DeformConv3d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]),
                                requires_grad=True)

        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_depth in_height, in_width]): input tensor
            offset (Tensor[batch_size, 3 * offset_groups * kernel_depth * kernel_height * kernel_width,
                out_depth, out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_depth * kernel_height * kernel_width,
                out_depth, out_height, out_width]): masks to be applied for each position in the
                convolution kernel.
        """
        return deform_conv3d(input, self.weight, offset, mask, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class PackedDeformConv3d(DeformConv3d):
    """
    See DeformConv3d
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            modulated: bool = False
    ):
        super(PackedDeformConv3d, self).__init__(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 groups,
                                                 bias)
        self.modulated = modulated

        self.conv_offset = nn.Conv3d(
            self.in_channels,
            3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)

        self.conv_mask = nn.Conv3d(
            self.in_channels,
            self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True) if self.modulated else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super(PackedDeformConv3d, self).reset_parameters()

        if hasattr(self, 'modulated'):
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
        offset = self.conv_offset(input)
        mask = self.conv_mask(input).sigmoid() if self.modulated else None
        return deform_conv3d(input, self.weight, offset, mask, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ', modulated=True' if self.modulated else ''
        s += ')'
        return s.format(**self.__dict__)
