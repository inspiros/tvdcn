#!/usr/bin/env python3
"""Audit Linux extensions (or every .so in a wheel) without loading them.

Requires binutils' nm, readelf and c++filt. Checks both defined and undefined
*dynamic* symbols: hidden/header-only implementations do not affect the ABI.
The CPython allowlist is pinned to the wheel's minimum Python (3.10), rather
than whichever interpreter happens to run this script.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


CPYTHON_ABI = frozenset(
    line for line in Path(__file__).with_name('cpython310_abi.txt').read_text().splitlines()
    if line and not line.startswith('#')
)


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, **kwargs)


def inspect_module(path):
    violations = defaultdict(set)
    dynamic = run('readelf', '-d', str(path))
    for dependency in re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic):
        if 'libtorch_python' in dependency:
            violations['forbidden dependency'].add(dependency)

    symbols = {}
    for mode, label in (('--defined-only', 'defined'), ('--undefined-only', 'undefined')):
        output = run('nm', '-D', mode, '--format=posix', '--no-demangle', str(path))
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 2:
                symbols[fields[0].split('@')[0]] = label
    mangled = sorted(name for name in symbols if name.startswith('_Z'))
    demangled = run('c++filt', input='\n'.join(mangled) + '\n').splitlines() if mangled else []
    for name, readable in zip(mangled, demangled):
        # Stable/headeronly definitions are deliberately hidden by torch's
        # headers. A C++ torch symbol reaching this table is an ABI dependency,
        # including vtables, const methods and templates (not just _ZN2at...).
        if re.search(r'\b(?:at|c10|torch)::', readable):
            violations['unstable torch C++ symbol'].add(
                f'{symbols[name]} {name} ({readable})')
    for name, label in symbols.items():
        if name.startswith(('Py', '_Py')) and name not in CPYTHON_ABI:
            if label == 'defined' and name.startswith('PyInit_'):
                continue
            violations['non-limited CPython symbol (cp310)'].add(f'{label} {name}')
    return violations


def audit(path, label, verbose):
    violations = inspect_module(path)
    if not violations:
        print(f'PASS {label}: no unstable dynamic ABI dependencies')
        return True
    print(f'FAIL {label}')
    for category, names in sorted(violations.items()):
        print(f'  {category}: {len(names)} violation(s)')
        ordered = sorted(names)
        for name in ordered if verbose else ordered[:12]:
            print(f'    {name}')
        if not verbose and len(ordered) > 12:
            print(f'    ... {len(ordered) - 12} more (use --verbose for all symbols)')
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, metavar='wheel-or-.so')
    parser.add_argument('--verbose', action='store_true', help='List every violating symbol')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        print('SKIP: stable ABI inspection currently supports Linux only')
        return 0
    passed = True
    for path in args.inputs:
        try:
            if path.suffix == '.whl':
                with zipfile.ZipFile(path) as wheel, tempfile.TemporaryDirectory() as directory:
                    modules = [name for name in wheel.namelist() if name.endswith('.so')]
                    if not modules:
                        raise ValueError('wheel contains no extension modules (.so)')
                    for index, name in enumerate(modules):
                        target = Path(directory) / f'{index}.so'
                        target.write_bytes(wheel.read(name))
                        passed = audit(target, f'{path}!{name}', args.verbose) and passed
            else:
                passed = audit(path, str(path), args.verbose) and passed
        except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            print(f'ERROR {path}: {error}', file=sys.stderr)
            if isinstance(error, subprocess.CalledProcessError):
                print(error.output, file=sys.stderr)
            passed = False
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
