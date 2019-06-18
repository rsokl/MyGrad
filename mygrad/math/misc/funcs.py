from mygrad.tensor_base import Tensor

from .ops import Abs, Cbrt, Maximum, Minimum, Sqrt

__all__ = ["abs", "absolute", "cbrt", "sqrt", "maximum", "minimum"]


def abs(a, constant=False):
    """ ``f(a) -> abs(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor

        Notes
        -----
        The derivative at a == 0 returns nan"""
    return Tensor._op(Abs, a, constant=constant)


absolute = abs


def sqrt(a, constant=False):
    """ ``f(a) -> sqrt(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not backpropagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sqrt, a, constant=constant)


def cbrt(a, constant=False):
    """ ``f(a) -> cbrt(a)``

        Parameters
        ----------
        a : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not backpropagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cbrt, a, constant=constant)


def maximum(a, b, constant=False):
    """ Element-wise maximum of array elements.

        Parameters
        ----------
        a : array_like

        b : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not backpropagate a gradient)

        Returns
        -------
        mygrad.Tensor

        Notes
        -----
        The gradient does not exist where a == b; we use a
        value of 0 here."""
    return Tensor._op(Maximum, a, b, constant=constant)


def minimum(a, b, constant=False):
    """ Element-wise minimum of array elements.

        Parameters
        ----------
        a : array_like

        b : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not backpropagate a gradient)

        Returns
        -------
        mygrad.Tensor

        Notes
        -----
        The gradient does not exist where a == b; we use a
        value of 0 here."""
    return Tensor._op(Minimum, a, b, constant=constant)
