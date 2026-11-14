"""Cover the CUDA kernels' 64-bit index path without 2^30-element tensors."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
def test_golden_with_int64_indices(device):
    if device != 'cuda':
        pytest.skip('CUDA-only dispatch path')
    tests_dir = Path(__file__).resolve().parent
    # The override is read once per process, so run the golden tests in a fresh one.
    env = {**os.environ, 'TVDCN_FORCE_INT64_INDEX': '1',
           'PYTHONPATH': os.pathsep.join(p for p in sys.path if p)}
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', str(tests_dir / 'test_golden.py'),
         '--device', 'cuda', '-q', '-p', 'no:cacheprovider'],
        cwd=tests_dir.parent, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert ' passed' in result.stdout and 'skipped' not in result.stdout.splitlines()[-1]
