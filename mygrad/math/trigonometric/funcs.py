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
           "arccot",
           "arctan2"]


def sin(a, constant=False):
    """ ``f(a) -> sin(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sin, a, constant=constant)


def sinc(a, constant=False):
    """ ``f(a) -> sinc(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sinc, a, constant=constant)


def cos(a, constant=False):
    """ ``f(a) -> cos(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cos, a, constant=constant)


def tan(a, constant=False):
    """ ``f(a) -> tan(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Tan, a, constant=constant)


def cot(a, constant=False):
    """ ``f(a) -> cot(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cot, a, constant=constant)


def csc(a, constant=False):
    """ ``f(a) -> csc(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Csc, a, constant=constant)


def sec(a, constant=False):
    """ ``f(a) -> sec(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sec, a, constant=constant)


def arccos(a, constant=False):
    """ ``f(a) -> arccos(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccos, a, constant=constant)


def arccsc(a, constant=False):
    """ ``f(a) -> arccsc(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccsc, a, constant=constant)


def arccot(a, constant=False):
    """ ``f(a) -> arccot(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccot, a, constant=constant)


def arcsin(a):
    """ ``f(a) -> arctanh(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arcsin, a)


def arctan(a):
    """ ``f(a) -> arctan(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arctan, a)


def arcsec(a):
    """ ``f(a) -> arcsec(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arcsec, a)

def arctan2(a, b):
    """ ``f(a, b) -> arctan(a/b)``

        Parameters
        ----------
        a : array_like
        b : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arctan2, a, b)