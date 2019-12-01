import numpy as np

from mygrad.tensor_base import Tensor

from .ops import Where

__all__ = ["where"]


class _UniqueIdentifier:
    def __init__(self, identifier):
        self.identifier = identifier

    def __repr__(self):
        return self.identifier


not_set = _UniqueIdentifier("not_set")


def where(condition, x=not_set, y=not_set, constant=False):
    if x is not_set and y is not_set:
        if isinstance(condition, Tensor):
            condition = condition.data
        return np.where(condition)

    if x is not_set or y is not_set:
        raise ValueError("either both or neither of x and y should be given")

    return Tensor._op(
        Where, x, y, op_kwargs=dict(condition=condition), constant=constant
    )
