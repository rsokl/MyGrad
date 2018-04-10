from .ops import Sqrt, Cbrt, Abs, Maximum, Minimum
from mygrad.tensor_base import Tensor


__all__ = ["abs",
           "cbrt",
           "sqrt",
           "maximum",
           "minimum"]


def abs(a):
    """ f(a)-> |a|

        The derivative at a == 0 returns nan"""
    return Tensor._op(Abs, a)


def sqrt(a):
    return Tensor._op(Sqrt, a)


def cbrt(a):
    return Tensor._op(Cbrt, a)


def maximum(a, b):
    """ Element-wise maximum of array elements.

        The gradient does not exist where a == b; we use a
        value of 0 here."""
    return Tensor._op(Maximum, a, b)


def minimum(a, b):
    """ Element-wise minimum of array elements.

        The gradient does not exist where a == b; we use a
        value of 0 here."""
    return Tensor._op(Minimum, a, b)

