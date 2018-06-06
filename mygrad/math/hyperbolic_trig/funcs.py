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
    """ ``f(a) -> arccosh(a)``

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
    """ ``f(a) -> arccoth(a)``

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
    """ ``f(a) -> arccsch(a)``

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
    """ ``f(a) -> arcsinh(a)``

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
    """ ``f(a) -> arctanh(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not backpropagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arctanh, a, constant=constant)


def cosh(a, constant=False):
    """ ``f(a) -> cosh(a)``

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
    """ ``f(a) -> coth(a)``

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
    """ ``f(a) -> csch(a)``

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
    """ ``f(a) -> sech(a)``

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
    """ ``f(a) -> sinh(a)``

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


def tanh(a, constant=False):
    """ ``f(a) -> tanh(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not backpropagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Tanh, a, constant=constant)
