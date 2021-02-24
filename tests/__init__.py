from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np

import mygrad
from mygrad._utils.op_creators import MyGradBinaryUfunc, MyGradUnaryUfunc
from mygrad.operation_base import _NoValue
from mygrad.typing import ArrayLike


def is_float_arr(arr: Union[np.ndarray, mygrad.Tensor]) -> bool:
    return issubclass(arr.dtype.type, np.floating)


class MinimalArgs(NamedTuple):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


def populate_args(*args, **kwargs) -> MinimalArgs:
    return MinimalArgs(
        args=tuple(item for item in args if item is not _NoValue),
        kwargs={k: v for k, v in kwargs.items() if v is not _NoValue},
    )


_public_ufunc_names: List[str] = []
_ufuncs: List[Union[MyGradUnaryUfunc, MyGradBinaryUfunc]] = []

for _name in sorted(
    public_name for public_name in dir(mygrad) if not public_name.startswith("_")
):
    attr = getattr(mygrad, _name)
    if isinstance(attr, (MyGradUnaryUfunc, MyGradBinaryUfunc)):
        _ufuncs.append(attr)
        _public_ufunc_names.append(_name)

public_ufunc_names = tuple(_public_ufunc_names)
ufuncs = tuple(_ufuncs)


