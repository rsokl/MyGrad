from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union

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

@dataclass
class MinimalArgs:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            ", ".join(repr(a) for a in self.args)
            + " "
            + " ".join(f"{k}: {v}" for k, v in self.kwargs.items())
        )

    def __getitem__(self, key: str):
        return self.kwargs[key]

    def __setitem__(self, key: str, value):
        self.kwargs[key] = value

    def args_as_no_mygrad(self) -> Tuple[NotTensor, ...]:
        return tuple(
            mg.asarray(item) if isinstance(item, mg.Tensor) else item
            for item in self.args
        )

    def tensors_only(self, filter=lambda x: True) -> Tuple[mg.Tensor, ...]:
        return tuple(a for a in self.args if isinstance(a, mg.Tensor) and filter(a))

    def make_arrays_read_only(self):
        map(_make_read_only, self.args)
        map(_make_read_only, self.kwargs.values())


def populate_args(*args, **kwargs) -> MinimalArgs:
    return MinimalArgs(
        args=tuple(item for item in args if item is not _NoValue),
        kwargs={k: v for k, v in kwargs.items() if v is not _NoValue},
    )
