from .ops import Sqrt, Cbrt, Abs, Maximum, Minimum
from mygrad.tensor_base import Tensor


__all__ = ["abs",
           "cbrt",
           "sqrt",
           "maximum",
           "minimum"]


def abs(a, constant=False):
    """ f(a)-> |a|

        The derivative at a == 0 returns nan"""
    return Tensor._op(Abs, a, constant=constant)


def sqrt(a, constant=False):
    return Tensor._op(Sqrt, a, constant=constant)


def cbrt(a, constant=False):
    return Tensor._op(Cbrt, a, constant=constant)


def maximum(a, b, constant=False):
    """ Element-wise maximum of array elements.

        The gradient does not exist where a == b; we use a
        value of 0 here."""
    return Tensor._op(Maximum, a, b, constant=constant)


def minimum(a, b, constant=False):
    """ Element-wise minimum of array elements.

        The gradient does not exist where a == b; we use a
        value of 0 here."""
    return Tensor._op(Minimum, a, b, constant=constant)

