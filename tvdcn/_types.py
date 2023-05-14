import torch.nn as nn
from torch.jit.annotations import Any, Union, Tuple

_IntTuple = Union[int, Tuple[int, ...]]
_Activation = Union[nn.Module, Any]
