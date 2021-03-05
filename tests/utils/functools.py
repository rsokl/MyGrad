from collections import UserDict
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, Iterable, Sequence, Tuple, Union

import numpy as np

import mygrad as mg
from mygrad.operation_base import _NoValue

Real = Union[float, int]
NotTensor = Union[Real, Sequence[Real], np.ndarray]


def _make_read_only(item):
    if isinstance(item, np.ndarray):
        item.flags["WRITEABLE"] = False
    elif isinstance(item, mg.Tensor):
        item.data.flags["WRITEABLE"] = False
    return


def add_constant_passthrough(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        kwargs.pop("constant", None)
        return func(*args, **kwargs)

    return wrapped


class SmartSignature(UserDict):
    """Used to populate mygrad ops
    - ignores any arguments with _NoValue assigned
    - supports * unpacking for args
    - supports ** unpacking for kwargs
    - supports getitem/setitem for kwargs
    """

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def __len__(self):
        return len(self.args)

    def __init__(self, *args, **kwargs):
        self.args = tuple(item for item in args if item is not _NoValue)
        self.kwargs = {k: v for k, v in kwargs.items() if v is not _NoValue}

    def __repr__(self) -> str:
        return (
            ", ".join(repr(a) for a in self.args)
            + " "
            + " ".join(f"{k}: {v}" for k, v in self.kwargs.items())
        )

    def keys(self) -> Iterable[str]:
        return self.kwargs.keys()

    def __iter__(self):
        return iter(self.args)

    def __getitem__(self, key: str):
        return self.kwargs[key]

    def __setitem__(self, key: str, value):
        if value is not _NoValue:
            self.kwargs[key] = value

    def args_as_no_mygrad(self) -> Tuple[NotTensor, ...]:
        return tuple(
            mg.asarray(item) if isinstance(item, mg.Tensor) else item
            for item in self.args
        )

    def tensors_only(self) -> Tuple[mg.Tensor, ...]:
        return tuple(a for a in self.args if isinstance(a, mg.Tensor))

    def make_array_based_args_read_only(self):
        map(_make_read_only, self.args)
        map(_make_read_only, self.kwargs.values())

    def copy(self):
        return SmartSignature(
            *(deepcopy(x) for x in self.args), **deepcopy(self.kwargs)
        )
