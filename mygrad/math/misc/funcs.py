from .ops import Sqrt, Cbrt, Abs
from mygrad.tensor_base import Tensor


__all__ = ["abs",
           "cbrt",
           "sqrt"]


def abs(a):
    """ f(a)-> |a|

        The derivative at a == 0 returns nan"""
    return Tensor._op(Abs, a)


def sqrt(a):
    return Tensor._op(Sqrt, a)


def cbrt(a):
    return Tensor._op(Cbrt, a)
