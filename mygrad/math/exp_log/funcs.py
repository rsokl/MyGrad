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


def exp(a):
    return Tensor._op(Exp, a)


def expm1(a):
    return Tensor._op(Expm1, a)


def logaddexp(a, b):
    return Tensor._op(Logaddexp, a, b)


def logaddexp2(a, b):
    return Tensor._op(Logaddexp2, a, b)


def log(a):
    return Tensor._op(Log, a)


def log2(a):
    return Tensor._op(Log2, a)


def log10(a):
    return Tensor._op(Log10, a)


def log1p(a):
    return Tensor._op(Log1p, a)
