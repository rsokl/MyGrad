import numpy as np

from mygrad.tensor_base import Tensor

from .ops import Where

__all__ = ["where"]


def where(condition, x=None, y=None, constant=False):
    if x is None and y is None:
        return np.where(condition)

    return Tensor._op(
        Where, x, y, op_kwargs=dict(condition=condition), constant=constant
    )
