from pathlib import Path

import pytest
import torch

from .utils.cases import GOLDEN_PROFILES, output_and_grads

DATA = Path(__file__).parent / 'data'


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])
@pytest.mark.parametrize('profile', GOLDEN_PROFILES)
def test_golden(dim, transposed, profile, device):
    # Fixed file names deliberately make missing fixtures fail instead of collecting zero tests.
    path = DATA / f"golden_{'transpose' if transposed else 'conv'}{dim}d.pt"
    fixture = torch.load(path, map_location='cpu', weights_only=True)
    assert fixture['metadata']['format_version'] == 1
    case = next(case for case in fixture['cases'] if case['profile'] == profile)
    expected_output, expected_grads = case['output'], case['grads']
    case['tensors'] = {name: value.to(device).detach().requires_grad_() if value is not None else None
                       for name, value in case['tensors'].items()}
    actual_output, actual_grads = output_and_grads(case)
    tolerance = dict(rtol=1e-10, atol=1e-12 if device == 'cpu' else 1e-9)
    torch.testing.assert_close(actual_output.cpu(), expected_output, **tolerance)
    assert actual_grads.keys() == expected_grads.keys()
    for name, actual in actual_grads.items():
        torch.testing.assert_close(actual.cpu(), expected_grads[name], **tolerance)
