from .ops import *
from mygrad.tensor_base import Tensor

__all__ = ["arccosh",
           "arccoth",
           "arccsch",
           "arcsinh",
           "arctanh",
           "cosh",
           "coth",
           "csch",
           "sech",
           "sinh",
           "tanh"]


def arccosh(a, constant=False):
    return Tensor._op(Arccosh, a, constant=constant)


def arccoth(a, constant=False):
    return Tensor._op(Arccoth, a, constant=constant)


def arccsch(a, constant=False):
    return Tensor._op(Arccsch, a, constant=constant)


def arcsinh(a, constant=False):
    return Tensor._op(Arcsinh, a, constant=constant)


def arctanh(a, constant=False):
    return Tensor._op(Arctanh, a, constant=constant)


def cosh(a, constant=False):
    return Tensor._op(Cosh, a, constant=constant)


def coth(a, constant=False):
    return Tensor._op(Coth, a, constant=constant)


def csch(a, constant=False):
    return Tensor._op(Csch, a, constant=constant)


def sech(a, constant=False):
    return Tensor._op(Sech, a, constant=constant)


def sinh(a, constant=False):
    return Tensor._op(Sinh, a, constant=constant)


def tanh(a, constant=False):
    return Tensor._op(Tanh, a, constant=constant)
