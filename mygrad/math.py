from .tensor_base import Tensor
from .operations import *

__all__ = ["abs",
           "arccos",
           "arccosh",
           "arccot",
           "arccoth",
           "arccsc",
           "arccsch",
           "arcsec",
           "arcsin",
           "arcsinh",
           "arctan",
           "arctanh",
           "cbrt",
           "cos",
           "cosh",
           "cot",
           "coth",
           "csc",
           "csch",
           "max",
           "mean"
           "min",
           "sec",
           "sech",
           "sin",
           "sinh",
           "sqrt",
           "sum",
           "tan",
           "tanh"]


def abs(a):
    """ f(a)-> |a|

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Abs, a)


def cbrt(a):
    """ f(a)-> cbrt(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cbrt, a)


def sqrt(a):
    """ f(a)-> sqrt(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sqrt, a)