from .ops import *
from mygrad.tensor_base import Tensor

__all__ = ["sin",
           "sinc",
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


def sin(a, constant=False):
    return Tensor._op(Sin, a, constant=constant)


def sinc(a, constant=False):
    return Tensor._op(Sinc, a, constant=constant)


def cos(a, constant=False):
    return Tensor._op(Cos, a, constant=constant)


def tan(a, constant=False):
    return Tensor._op(Tan, a, constant=constant)


def cot(a, constant=False):
    return Tensor._op(Cot, a, constant=constant)


def csc(a, constant=False):
    return Tensor._op(Csc, a, constant=constant)


def sec(a, constant=False):
    return Tensor._op(Sec, a, constant=constant)


def arccos(a, constant=False):
    return Tensor._op(Arccos, a, constant=constant)


def arccsc(a, constant=False):
    return Tensor._op(Arccsc, a, constant=constant)


def arccot(a, constant=False):
    return Tensor._op(Arccot, a, constant=constant)


def arcsin(a):
    return Tensor._op(Arcsin, a)


def arctan(a):
    return Tensor._op(Arctan, a)


def arcsec(a):
    return Tensor._op(Arcsec, a)