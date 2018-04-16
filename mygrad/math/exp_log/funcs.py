from .ops import *

from mygrad.tensor_base import Tensor

__all__ = ["exp",
           "expm1",
           "logaddexp",
           "logaddexp2",
           "log",
           "log2",
           "log10",
           "log1p"]


def exp(a, constant=False):
    return Tensor._op(Exp, a, constant=constant)


def expm1(a, constant=False):
    return Tensor._op(Expm1, a, constant=constant)


def logaddexp(a, b, constant=False):
    return Tensor._op(Logaddexp, a, b, constant=constant)


def logaddexp2(a, b, constant=False):
    return Tensor._op(Logaddexp2, a, b, constant=constant)


def log(a, constant=False):
    return Tensor._op(Log, a, constant=constant)


def log2(a, constant=False):
    return Tensor._op(Log2, a, constant=constant)


def log10(a, constant=False):
    return Tensor._op(Log10, a, constant=constant)


def log1p(a, constant=False):
    return Tensor._op(Log1p, a, constant=constant)
