"""Small deterministic cases shared by pytest and the legacy fixture generator."""
import math

import torch
import tvdcn

# Cover independent optional tensors and group partitions without a full product.
PROFILES = {
    'plain': dict(offset=False, mask=False),
    'offset': dict(offset=True, mask=False),
    'mask': dict(offset=False, mask=True),
    'both': dict(offset=True, mask=True),
    'grouped': dict(offset=True, mask=True, groups=2, offset_groups=2),
    'geometry': dict(offset=True, mask=True, mask_groups=2, geometry=True),
}
# Legacy torch 2.14 backward cannot process undefined optional tensors. Capture
# all five gradients on these paths; optional paths remain covered by live tests.
GOLDEN_PROFILES = ('both', 'grouped', 'geometry')


def make_case(dim, transposed, profile='both', *, device='cpu', dtype=torch.float64,
              small=False):
    spec = PROFILES[profile]
    groups = spec.get('groups', 1)
    kernel = (2,) * dim
    spatial = (2 if small else 3,) * dim
    stride = tuple(2 if spec.get('geometry') and i % 2 == 0 else 1 for i in range(dim))
    dilation = tuple(2 if spec.get('geometry') else 1 for _ in range(dim))
    padding = tuple(1 if spec.get('geometry') else 0 for _ in range(dim))
    output_padding = tuple(s - 1 if transposed else 0 for s in stride)
    if transposed:
        out = tuple((n - 1) * s - 2 * p + d * (k - 1) + op + 1
                    for n, s, p, d, k, op in zip(spatial, stride, padding, dilation, kernel, output_padding))
        deform_spatial = spatial
    else:
        out = tuple((n + 2 * p - d * (k - 1) - 1) // s + 1
                    for n, s, p, d, k in zip(spatial, stride, padding, dilation, kernel))
        deform_spatial = out
    # Always generate on CPU, so CPU and CUDA consume exactly the same data.
    rng = torch.Generator().manual_seed(1729 + dim + 10 * transposed)

    def rand(shape, *, offset=False):
        value = torch.rand(shape, generator=rng, dtype=torch.float64)
        value = value * 0.3 + 0.1 if offset else value - 0.5
        return value.to(device=device, dtype=dtype).requires_grad_()

    tensors = dict(
        input=rand((1, 2, *spatial)),
        weight=rand((2, 2 // groups, *kernel)),
        offset=rand((1, spec.get('offset_groups', 1) * dim * math.prod(kernel), *deform_spatial),
                    offset=True) if spec['offset'] else None,
        mask=rand((1, spec.get('mask_groups', 1) * math.prod(kernel), *deform_spatial)) if spec['mask'] else None,
        bias=rand((2,)) if spec.get('bias', True) else None,
    )
    kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    if transposed:
        kwargs['output_padding'] = output_padding
    return dict(dim=dim, transposed=transposed, profile=profile, tensors=tensors,
                kwargs=kwargs, output_shape=(1, 2, *out))


def operator(case):
    return getattr(tvdcn, f"deform_conv{'_transpose' if case['transposed'] else ''}{case['dim']}d")


def forward(case, tensors=None):
    return operator(case)(**(case['tensors'] if tensors is None else tensors), **case['kwargs'])


def output_and_grads(case):
    output = forward(case)
    # A nonuniform upstream gradient catches spatial/channel indexing mistakes.
    probe = torch.linspace(-0.7, 0.9, output.numel(), dtype=output.dtype,
                           device=output.device).reshape_as(output)
    active = {name: tensor for name, tensor in case['tensors'].items() if tensor is not None}
    grads = torch.autograd.grad(output, tuple(active.values()), probe)
    return output, dict(zip(active, grads))
