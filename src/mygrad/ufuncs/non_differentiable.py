from functools import wraps
from typing import Union

import numpy as np

from ..tensor_base import Tensor, asarray


def _constant_only_passthrough(ufunc: np.ufunc):
    def _as_constant_array(tensor: Union[Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(tensor, Tensor):
            if tensor.constant is False:
                raise ValueError(
                    f"{repr(ufunc)} cannot involve non-constant mygrad tensors."
                )
            return tensor.data
        return tensor

    @wraps(ufunc)
    def wrap(*inputs, **kwargs):
        if "out" in kwargs:
            kwargs["out"] = _as_constant_array(kwargs["out"])
        return ufunc(*(_as_constant_array(t) for t in inputs), **kwargs)

    return wrap


def _boolean_out_passthrough(ufunc: np.ufunc):
    @wraps(ufunc)
    def wrap(*inputs, **kwargs):
        if "out" in kwargs and kwargs["out"] is not None:
            kwargs["out"] = asarray(kwargs["out"])
        return ufunc(*(asarray(t) for t in inputs), **kwargs)

    return wrap
