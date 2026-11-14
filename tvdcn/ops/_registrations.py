"""Autograd, autocast and fake implementations for the stable operator families."""
import torch


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:5])
    ctx.geometry = inputs[5:]


def _fake_backward(grad, input, weight, offset=None, mask=None, bias=None,
                   stride=None, padding=None, dilation=None, groups=1):
    # The native schema has five Tensor outputs. Missing optional gradients are
    # represented by defined empty tensors and discarded by _backward.
    return tuple(t.new_empty(t.shape) if t is not None else input.new_empty((0,))
                 for t in (input, weight, offset, mask, bias))


_autocast_keys = (torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCPU)
                  | torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCUDA))
_autocast_lib = torch.library.Library('tvdcn', 'IMPL')


def _cast(t):
    if t is not None and t.is_floating_point() and t.dtype != torch.float64:
        return t.float()
    return t


def _register_family(dim):
    name = f'deform_conv{dim}d'
    if not hasattr(torch.ops.tvdcn, name):
        return
    forward_op = getattr(torch.ops.tvdcn, name)
    backward_op = getattr(torch.ops.tvdcn, f'_{name}_backward')
    ones, zeros = (1,) * dim, (0,) * dim

    def _backward(ctx, grad):
        tensors = ctx.saved_tensors
        grads = backward_op(grad, *tensors, *ctx.geometry)
        return tuple(g if t is not None and i < len(ctx.needs_input_grad)
                     and ctx.needs_input_grad[i] else None
                     for i, (g, t) in enumerate(zip(grads, tensors))) + (None,) * 4

    def _no_double_backward(ctx, *grads):
        raise RuntimeError(f'double backwards on {name} not supported')

    def _fake_forward(input, weight, offset=None, mask=None, bias=None,
                      stride=ones, padding=zeros, dilation=ones, groups=1):
        torch._check(input.dim() in (dim + 1, dim + 2),
                     lambda: f'Expected {dim + 1}D or {dim + 2}D input')
        torch._check(weight.dim() == dim + 2, lambda: f'Expected {dim + 2}D weight')
        out = tuple((input.shape[-dim + i] + 2 * padding[i]
                     - dilation[i] * (weight.shape[2 + i] - 1) - 1) // stride[i] + 1
                    for i in range(dim))
        for size in out:
            torch._check(size > 0, lambda: 'Calculated output size too small')
        shape = (*input.shape[:-dim - 1], weight.shape[0], *out)
        return input.new_empty(shape)

    def _autocast(input, weight, offset=None, mask=None, bias=None,
                  stride=ones, padding=zeros, dilation=ones, groups=1):
        with torch._C._ExcludeDispatchKeyGuard(_autocast_keys):
            return forward_op(*(_cast(t) for t in (input, weight, offset, mask, bias)),
                              stride, padding, dilation, groups).to(input.dtype)

    torch.library.register_autograd(f'tvdcn::{name}', _backward,
                                    setup_context=_setup_context)
    torch.library.register_autograd(f'tvdcn::_{name}_backward', _no_double_backward)
    torch.library.register_fake(f'tvdcn::{name}', _fake_forward)
    torch.library.register_fake(f'tvdcn::_{name}_backward', _fake_backward)
    for key in ('AutocastCPU', 'AutocastCUDA'):
        _autocast_lib.impl(name, _autocast, key)


for _dim in (1, 2, 3):
    _register_family(_dim)
