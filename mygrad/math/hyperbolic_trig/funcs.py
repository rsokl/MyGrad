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


def arccosh(a):
    return Tensor._op(Arccosh, a)


def arccoth(a):
    return Tensor._op(Arccoth, a)


def arccsch(a):
    return Tensor._op(Arccsch, a)


def arcsinh(a):
    return Tensor._op(Arcsinh, a)


def arctanh(a):
    return Tensor._op(Arctanh, a)


def cosh(a):
    return Tensor._op(Cosh, a)


def coth(a):
    return Tensor._op(Coth, a)


def csch(a):
    return Tensor._op(Csch, a)


def sech(a):
    return Tensor._op(Sech, a)


def sinh(a):
    return Tensor._op(Sinh, a)


def tanh(a):
    return Tensor._op(Tanh, a)
