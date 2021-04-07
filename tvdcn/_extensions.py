import os
import importlib.machinery
import torch

__all__ = ['_HAS_OPS']

_HAS_OPS = False

try:
    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_C")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)
    _HAS_OPS = True
except (ImportError, OSError) as e:
    pass
