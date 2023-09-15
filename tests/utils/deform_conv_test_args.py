import math

import torch
import torch.nn as nn

__all__ = [
    'DeformConvTestArgs'
]


def _decontiguous(x: torch.Tensor):
    perm_ind = torch.arange(x.ndim)
    while perm_ind.allclose(torch.arange(x.ndim)):
        perm_ind = torch.randperm(x.ndim)
    deperm_ind = torch.argsort(perm_ind)
    return x.permute(*perm_ind.tolist()).contiguous().permute(*deperm_ind.tolist())


class DeformConvTestArgs(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs.get('dim', 2)
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.batch_size = kwargs.get('batch_size', 1)
        self.batched = self.batch_size > 0
        self._batch_size = (self.batch_size,) if self.batched else ()
        self.dtype = kwargs.get('dtype', torch.float64)
        self.transposed = kwargs.get('transposed', False)
        contiguous = kwargs.get('contiguous', True)

        self.in_channels = kwargs.get('in_channels', 4)
        self.out_channels = kwargs.get('out_channels', 2)
        self.groups = kwargs.get('groups', 1)
        self.offset_groups = kwargs.get('offset_groups', 2)
        self.mask_groups = kwargs.get('mask_groups', 1)

        self.stride = tuple(range(1, 1 + self.dim))
        self.padding = tuple(range(0, 0 + self.dim))
        self.output_padding = tuple(range(0, 0 + self.dim)) if self.transposed else (0,) * self.dim
        self.dilation = tuple(range(1, 1 + self.dim))

        # self.stride = (1,) * self.dim
        # self.padding = (0,) * self.dim
        # self.output_padding = (0,) * self.dim
        # self.dilation = (1,) * self.dim

        self.kernel_size = tuple(range(2, 2 + self.dim))
        self.in_size = tuple(range(5, 5 + self.dim))
        self.out_size = [
            (self.in_size[_] + 2 * self.padding[_] - (
                    self.dilation[_] * (self.kernel_size[_] - 1) + 1))
            // self.stride[_] + 1 for _ in range(self.dim)]

        # input param
        self.input = nn.Parameter(
            torch.rand(*self._batch_size, self.in_channels,
                       *(self.in_size if not self.transposed else self.out_size),
                       device=self.device, dtype=self.dtype))
        # conv params
        if not self.transposed:
            self.weight = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels // self.groups,
                            *self.kernel_size,
                            device=self.device, dtype=self.dtype))
        else:
            self.weight = nn.Parameter(
                torch.empty(self.in_channels, self.out_channels // self.groups,
                            *self.kernel_size,
                            device=self.device, dtype=self.dtype))
        self.bias = nn.Parameter(
            torch.empty(self.out_channels,
                        device=self.device, dtype=self.dtype))
        # deformable conv params
        self.offset = nn.Parameter(
            torch.empty(*self._batch_size, self.offset_groups * self.dim * math.prod(self.kernel_size),
                        *self.out_size,
                        device=self.device, dtype=self.dtype))
        self.mask = nn.Parameter(
            torch.empty(*self._batch_size, self.mask_groups * math.prod(self.kernel_size), *self.out_size,
                        device=self.device, dtype=self.dtype))
        self.reset_parameters()
        self.set_contiguous(contiguous)

        if self.transposed:
            self.in_size, self.out_size = self.out_size, self.in_size

    def reset_parameters(self):
        torch.nn.init.uniform_(self.input)
        torch.nn.init.uniform_(self.weight)
        torch.nn.init.uniform_(self.offset)
        torch.nn.init.uniform_(self.mask)
        torch.nn.init.uniform_(self.bias)

    def contiguous(self):
        return self.input.is_contiguous()

    def set_contiguous(self, cond: bool = True):
        if cond:
            self.input = self.input.contiguous()
            self.weight = self.weight.contiguous()
            self.offset = self.offset.contiguous()
            self.mask = self.mask.contiguous()
        else:
            self.input = nn.Parameter(_decontiguous(self.input))
            self.weight = nn.Parameter(_decontiguous(self.weight))
            self.offset = nn.Parameter(_decontiguous(self.offset))
            self.mask = nn.Parameter(_decontiguous(self.mask))

    def equivalent_torch_layer(self):
        if self.dim == 1:
            cls = torch.nn.Conv1d if not self.transposed else torch.nn.ConvTranspose1d
        elif self.dim == 2:
            cls = torch.nn.Conv2d if not self.transposed else torch.nn.ConvTranspose2d
        else:
            cls = torch.nn.Conv3d if not self.transposed else torch.nn.ConvTranspose3d
        params = dict(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding,
                      output_padding=self.output_padding,
                      dilation=self.dilation,
                      bias=True,
                      groups=self.groups,
                      device=self.device,
                      dtype=self.dtype)
        if not self.transposed:
            params.pop('output_padding')
        layer = cls(**params)
        layer.weight.data = self.weight.data
        layer.bias.data = self.bias.data
        return layer

    @property
    def expected_cols_size(self, n_parallel_imgs=1):
        return torch.Size([self.in_channels if not self.transposed else
                           self.out_channels * math.prod(self.kernel_size),
                           n_parallel_imgs * math.prod(self.expected_output_size[2:])])

    @property
    def expected_output_size(self):
        out_size = (self.out_size[_] + self.output_padding[_] for _ in range(self.dim))
        return torch.Size([*self._batch_size, self.out_channels, *out_size])

    @property
    def tol(self):
        return 2e-3 if self.dtype is torch.half else 1e-5

    def __repr__(self):
        s = f"{self.__class__.__name__}(spatial_dim={self.dim}"
        s += "\n- conv_params:"
        s += f"\n\t{'in_channels':<15} = {self.in_channels}"
        s += f"\n\t{'out_channels':<15} = {self.out_channels}"
        s += f"\n\t{'kernel_size':<15} = {self.kernel_size}"
        s += f"\n\t{'stride':<15} = {self.stride}"
        s += f"\n\t{'padding':<15} = {self.padding}"
        s += f"\n\t{'dilation':<15} = {self.dilation}"
        s += f"\n\t{'weight_groups':<15} = {self.groups}"
        s += f"\n\t{'offset_groups':<15} = {self.offset_groups}"
        s += f"\n\t{'mask_groups':<15} = {self.mask_groups}"
        s += f"\n\t{'transposed':<15} = {self.transposed}"
        s += "\n- input_params:"
        s += f"\n\t{'batch_size':<15} = {self.batch_size}"
        s += f"\n\t{'in_size':<15} = {self.in_size}"
        s += f"\n\t{'out_size':<15} = {self.out_size}"
        s += "\n- input_shapes:"
        s += f"\n\t{'input':<15} = {self.input.shape}"
        s += f"\n\t{'weight':<15} = {self.weight.shape}"
        s += f"\n\t{'offset':<15} = {self.offset.shape}"
        s += f"\n\t{'mask':<15} = {self.mask.shape}"
        s += f"\n\t{'bias':<15} = {self.bias.shape}"
        s += "\n- output_shapes:"
        s += f"\n\t{'expected_cols':<15} = {self.expected_cols_size}"
        s += f"\n\t{'expected_output':<15} = {self.expected_output_size}"
        s += "\n- factory_params:"
        s += f"\n\t{'device':<15} = {self.device}"
        s += f"\n\t{'dtype':<15} = {self.dtype}"
        s += f"\n\t{'contiguous':<15} = {self.contiguous()}"
        s += "\n)"
        return s
