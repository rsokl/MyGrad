from .ops import *
from mygrad.tensor_base import Tensor

__all__ = ["sin",
           "cos",
           "tan",
           "cot",
           "csc",
           "sec",
           "arccos",
           "arccsc",
           "arcsin",
           "arctan",
           "arcsec",
           "arccot"]


def sin(a):
    return Tensor._op(Sin, a)


def cos(a):
    return Tensor._op(Cos, a)


def tan(a):
    return Tensor._op(Tan, a)


def cot(a):
    return Tensor._op(Cot, a)


def csc(a):
    return Tensor._op(Csc, a)


def sec(a):
    return Tensor._op(Sec, a)


def arccos(a):
    return Tensor._op(Arccos, a)


def arccsc(a):
    return Tensor._op(Arccsc, a)


def arccot(a):
    return Tensor._op(Arccot, a)


def arcsin(a):
    return Tensor._op(Arcsin, a)


def arctan(a):
    return Tensor._op(Arctan, a)


def arcsec(a):
    return Tensor._op(Arcsec, a)