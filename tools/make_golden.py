#!/usr/bin/env python3
"""Capture the legacy extension before porting it. Run from any directory.

    python tools/make_golden.py

Fixtures contain CPU float64 inputs, outputs, and all present parameter gradients.
Regenerate deliberately on the legacy revision, never as part of running pytest.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import tvdcn
from tests.utils.cases import GOLDEN_PROFILES, make_case, output_and_grads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tests' / 'data')
    args = parser.parse_args()
    if not tvdcn.has_ops():
        parser.error('Build the legacy extension first: python setup.py build_ext --inplace')
    torch.set_num_threads(1)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata = dict(format_version=1, revision=revision, torch=str(torch.__version__),
                    tvdcn=str(tvdcn.__version__), dtype='float64', device='cpu',
                    description='Legacy native kernels; C++20 build flag and torch pin relaxed only')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for dim in (1, 2, 3):
        for transposed in (False, True):
            cases = []
            for profile in GOLDEN_PROFILES:
                case = make_case(dim, transposed, profile)
                output, grads = output_and_grads(case)
                case['tensors'] = {name: value.detach() if value is not None else None
                                   for name, value in case['tensors'].items()}
                case.update(output=output.detach(), grads={name: value.detach() for name, value in grads.items()})
                cases.append(case)
            path = args.output_dir / f"golden_{'transpose' if transposed else 'conv'}{dim}d.pt"
            torch.save(dict(metadata=metadata, cases=cases), path)
            total += path.stat().st_size
            print(f'{path.name}: {len(cases)} cases, {path.stat().st_size} bytes')
    if total >= 2_000_000:
        raise RuntimeError(f'Fixtures exceed the 2 MB budget: {total} bytes')
    print(f'Total: {total} bytes; source {revision}, torch {torch.__version__}')


if __name__ == '__main__':
    main()
