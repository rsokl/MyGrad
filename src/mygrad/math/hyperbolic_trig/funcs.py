from mygrad.tensor_base import Tensor

from .ops import *

__all__ = [
    "arccosh",
    "arccoth",
    "arccsch",
    "arcsinh",
    "arctanh",
    "cosh",
    "coth",
    "csch",
    "sech",
    "sinh",
    "tanh",
]


def arccosh(a, constant=False):
    """``f(a) -> arccosh(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arccosh, a, constant=constant)


def arccoth(a, constant=False):
    """``f(a) -> arccoth(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arccoth, a, constant=constant)


def arccsch(a, constant=False):
    """``f(a) -> arccsch(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arccsch, a, constant=constant)


def arcsinh(a, constant=False):
    """``f(a) -> arcsinh(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arcsinh, a, constant=constant)


def arctanh(a, constant=False):
    """``f(a) -> arctanh(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arctanh, a, constant=constant)


def cosh(a, constant=False):
    """``f(a) -> cosh(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Cosh, a, constant=constant)


def coth(a, constant=False):
    """``f(a) -> coth(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Coth, a, constant=constant)


def csch(a, constant=False):
    """``f(a) -> csch(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Csch, a, constant=constant)


def sech(a, constant=False):
    """``f(a) -> sech(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Sech, a, constant=constant)


def sinh(a, constant=False):
    """``f(a) -> sinh(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Sinh, a, constant=constant)


def tanh(x, constant=False):
    """``f(x) -> tanh(x)``

    Parameters
    ----------
    x : array_like
        tanh is applied element-wise to ``x``

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import softmax, sigmoid, tanh
    >>> x = mg.linspace(-5, 5, 10)
    >>> tanh(x)
    Tensor([-0.9999092 , -0.99916247, -0.99229794, -0.93110961, -0.5046724 ,
             0.5046724 ,  0.93110961,  0.99229794,  0.99916247,  0.9999092 ])
    """
    return Tensor._op(Tanh, x, constant=constant)
