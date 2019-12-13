from mygrad.tensor_base import Tensor

from .ops import Repeat

__all__ = ["repeat"]


def repeat(a, repeats, axis=None, constant=False):
    return Tensor._op(Repeat, a, op_args=(repeats, axis), constant=constant)
