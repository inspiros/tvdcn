import torch
import torch.nn as nn
from torch import Tensor
from torch.jit.annotations import Tuple, Optional
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple

from .._types import _IntTuple

__all__ = [
    'mask_softmax1d',
    'mask_softmax2d',
    'mask_softmax3d',
    'MaskSoftmax1d',
    'MaskSoftmax2d',
    'MaskSoftmax3d',
]


def mask_softmax1d(mask: Tensor,
                   kernel_size: Tuple[int]) -> Tensor:
    r"""
    Performs 1D Mask Softmax Normalization.

    Arguments:
        mask (Tensor[batch_size, mask_groups * kernel_width, out_width]): modulation
            masks to be multiplied with each output of convolution kernel.
        kernel_size (int or Tuple[int]): convolution kernel size.
    """
    batch_size, _, out_width = mask.size()

    weight_w, = _single(kernel_size)
    mask_groups = mask.size(1) // weight_w

    if mask_groups == 0:
        raise RuntimeError(
            "The shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of kernel_size[0].\n"
            "Got mask.shape[1]={}, while kernel_size[0]={}".format(
                mask.shape[1], weight_w))

    mask = mask.view(batch_size,
                     mask_groups,
                     weight_w,
                     out_width)
    mask = torch.softmax(mask, dim=2)
    mask = mask.view(batch_size,
                     mask_groups * weight_w,
                     out_width)
    return mask


def mask_softmax2d(mask: Tensor,
                   kernel_size: Tuple[int, int]) -> Tensor:
    r"""
    Performs 2D Mask Softmax Normalization.

    References:
        https://arxiv.org/abs/2211.05778

    Arguments:
        mask (Tensor[batch_size, mask_groups * kernel_height * kernel_width,
            out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel.
        kernel_size (int or Tuple[int, int]): convolution kernel size.
    """
    batch_size, _, out_height, out_width = mask.size()

    weight_h, weight_w = _pair(kernel_size)
    mask_groups = mask.size(1) // (weight_h * weight_w)

    if mask_groups == 0:
        raise RuntimeError(
            "The shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of kernel_size[0] * kernel_size[1].\n"
            "Got mask.shape[1]={}, while kernel_size[0] * kernel_size[1]={}".format(
                mask.shape[1], weight_h * weight_w))

    mask = mask.view(batch_size,
                     mask_groups,
                     weight_h * weight_w,
                     out_height,
                     out_width)
    mask = torch.softmax(mask, dim=2)
    mask = mask.view(batch_size,
                     mask_groups * weight_h * weight_w,
                     out_height,
                     out_width)
    return mask


def mask_softmax3d(mask: Tensor,
                   kernel_size: Tuple[int, int, int]) -> Tensor:
    r"""
    Performs 3D Mask Softmax Normalization.

    Arguments:
        mask (Tensor[batch_size, mask_groups * kernel_depth * kernel_height * kernel_width,
            out_depth, out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel. Default: None
        kernel_size (int or Tuple[int, int, int]): convolution kernel size.
    """
    batch_size, _, out_depth, out_height, out_width = mask.size()

    weight_d, weight_h, weight_w = _triple(kernel_size)
    mask_groups = mask.size(1) // (weight_d * weight_h * weight_w)

    if mask_groups == 0:
        raise RuntimeError(
            "The shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of kernel_size[0] * kernel_size[1] * kernel_size[2].\n"
            "Got mask.shape[1]={}, while kernel_size[0] * kernel_size[1] * kernel_size[2]={}".format(
                mask.shape[1], weight_d * weight_h * weight_w))

    mask = mask.view(batch_size,
                     mask_groups,
                     weight_d * weight_h * weight_w,
                     out_depth,
                     out_height,
                     out_width)
    mask = torch.softmax(mask, dim=2)
    mask = mask.view(batch_size,
                     mask_groups * weight_d * weight_h * weight_w,
                     out_depth,
                     out_height,
                     out_width)
    return mask


################################################################################
# Modules
################################################################################
class _MaskSoftmaxNd(nn.Module):
    """
    Base class for MaskSoftmax
    """

    def __init__(self, kernel_size: _IntTuple):
        super().__init__()
        self.kernel_size: Tuple[int, ...] = kernel_size

    def forward(self, mask: Tensor, kernel_size: _IntTuple = None) -> Tensor:
        raise NotImplementedError


class MaskSoftmax1d(_MaskSoftmaxNd):
    """
    See :func:`mask_softmax1d`
    """

    def __init__(self, kernel_size: _size_1_t) -> None:
        kernel_size = _single(kernel_size)
        super().__init__(kernel_size)

    def forward(self, mask: Tensor, kernel_size: Optional[Tuple[int]] = None) -> Tensor:
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        return mask_softmax1d(mask, kernel_size)


class MaskSoftmax2d(_MaskSoftmaxNd):
    """
    See :func:`mask_softmax2d`
    """

    def __init__(self, kernel_size: _size_2_t) -> None:
        kernel_size = _pair(kernel_size)
        super().__init__(kernel_size)

    def forward(self, mask: Tensor, kernel_size: Optional[Tuple[int, int]] = None) -> Tensor:
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        return mask_softmax2d(mask, kernel_size)  # type: ignore[arg-type]


class MaskSoftmax3d(_MaskSoftmaxNd):
    """
    See :func:`mask_softmax3d`
    """

    def __init__(self, kernel_size: _size_3_t) -> None:
        kernel_size = _triple(kernel_size)
        super().__init__(kernel_size)

    def forward(self, mask: Tensor, kernel_size: Optional[Tuple[int, int, int]] = None) -> Tensor:
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        return mask_softmax3d(mask, kernel_size)  # type: ignore[arg-type]
