"""Autograd, autocast and fake implementations for the stable operator families."""
import torch


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:5])
    ctx.geometry = inputs[5:]


def _backward(ctx, grad):
    tensors = ctx.saved_tensors
    grads = torch.ops.tvdcn._deform_conv2d_backward(grad, *tensors, *ctx.geometry)
    return tuple(g if t is not None and i < len(ctx.needs_input_grad)
                 and ctx.needs_input_grad[i] else None
                 for i, (g, t) in enumerate(zip(grads, tensors))) + (None,) * 4


def _no_double_backward(ctx, *grads):
    raise RuntimeError('double backwards on deform_conv2d not supported')


def _fake_forward(input, weight, offset=None, mask=None, bias=None,
                  stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    torch._check(input.dim() in (3, 4), lambda: 'Expected 3D or 4D input')
    torch._check(weight.dim() == 4, lambda: 'Expected 4D weight')
    out = tuple((input.shape[-2 + i] + 2 * padding[i]
                 - dilation[i] * (weight.shape[2 + i] - 1) - 1) // stride[i] + 1
                for i in range(2))
    for size in out:
        torch._check(size > 0, lambda: 'Calculated output size too small')
    shape = (*input.shape[:-3], weight.shape[0], *out)
    return input.new_empty(shape)


def _fake_backward(grad, input, weight, offset=None, mask=None, bias=None,
                   stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    # The native schema has five Tensor outputs. Missing optional gradients are
    # represented by defined empty tensors and discarded by _backward.
    return tuple(t.new_empty(t.shape) if t is not None else input.new_empty((0,))
                 for t in (input, weight, offset, mask, bias))


_autocast_keys = (torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCPU)
                  | torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCUDA))


def _cast(t):
    if t is not None and t.is_floating_point() and t.dtype != torch.float64:
        return t.float()
    return t


def _autocast(input, weight, offset=None, mask=None, bias=None,
              stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    with torch._C._ExcludeDispatchKeyGuard(_autocast_keys):
        return torch.ops.tvdcn.deform_conv2d(
            *(_cast(t) for t in (input, weight, offset, mask, bias)),
            stride, padding, dilation, groups).to(input.dtype)


if hasattr(torch.ops.tvdcn, 'deform_conv2d'):
    torch.library.register_autograd('tvdcn::deform_conv2d', _backward,
                                    setup_context=_setup_context)
    torch.library.register_autograd('tvdcn::_deform_conv2d_backward', _no_double_backward)
    torch.library.register_fake('tvdcn::deform_conv2d', _fake_forward)
    torch.library.register_fake('tvdcn::_deform_conv2d_backward', _fake_backward)
    _autocast_lib = torch.library.Library('tvdcn', 'IMPL')
    for _key in ('AutocastCPU', 'AutocastCUDA'):
        _autocast_lib.impl('deform_conv2d', _autocast, _key)
