from typing import Any, Dict, NamedTuple, Tuple, Union

import numpy as np

import mygrad as mg
from mygrad.operation_base import _NoValue


def is_float_arr(arr: Union[np.ndarray, mg.Tensor]) -> bool:
    return issubclass(arr.dtype.type, np.floating)


class MinimalArgs(NamedTuple):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


def populate_args(*args, **kwargs) -> MinimalArgs:
    return MinimalArgs(
        args=tuple(item for item in args if item is not _NoValue),
        kwargs={k: v for k, v in kwargs.items() if v is not _NoValue},
    )
