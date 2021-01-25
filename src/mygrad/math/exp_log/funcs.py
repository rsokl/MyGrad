from mygrad.tensor_base import Tensor

from .ops import Exp, Exp2, Expm1, Log, Log1p, Log2, Log10, Logaddexp, Logaddexp2

__all__ = [
    "exp",
    "exp2",
    "expm1",
    "logaddexp",
    "logaddexp2",
    "log",
    "log2",
    "log10",
    "log1p",
]


def exp(a, constant=False):
    """``f(a) -> exp(a)``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Exp, a, constant=constant)


def exp2(a, constant=False):
    """``f(a) -> 2^a``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Exp2, a, constant=constant)


def expm1(a, constant=False):
    """``f(a) -> exp(a) - 1``

    The inverse of ``logp1``.

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=True)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    This function provides greater precision than ``exp(x) - 1`` for
    small values of ``x``."""
    return Tensor._op(Expm1, a, constant=constant)


def logaddexp(a, b, constant=False):
    """``f(a, b) -> log(exp(a) + exp(b))``

    Parameters
    ----------
    a : array_like

    b : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    This function is useful
    in statistics where the calculated probabilities of events may
    be so small as to exceed the range of normal floating point
    numbers. In such cases the logarithm of the calculated
    probability is stored. This function allows adding probabilities
    stored in such a fashion."""
    return Tensor._op(Logaddexp, a, b, constant=constant)


def logaddexp2(a, b, constant=False):
    """``f(a, b) -> log_2(2 ** a + 2 ** b)``

    Utilizes base-2 log.

    Parameters
    ----------
    a : array_like

    b : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    This function is useful
    in statistics where the calculated probabilities of events may
    be so small as to exceed the range of normal floating point
    numbers. In such cases the logarithm of the calculated
    probability is stored. This function allows adding probabilities
    stored in such a fashion."""
    return Tensor._op(Logaddexp2, a, b, constant=constant)


def log(a, constant=False):
    """``f(a) -> log(a)``

    Natural logarithm

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=True)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    This function provides greater precision than ``exp(x) - 1`` for
    small values of ``x``."""
    return Tensor._op(Log, a, constant=constant)


def log2(a, constant=False):
    """``f(a) -> log2(a)``

    Base-2 logarithm

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=True)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Log2, a, constant=constant)


def log10(a, constant=False):
    """``f(a) -> log10(a)``

    Base-10 logarithm

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=True)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Log10, a, constant=constant)


def log1p(a, constant=False):
    """f(a) -> log(1 + a)

    The inverse of ``expm1``.

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=True)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    For real-valued input, log1p is accurate also
    for ``x`` so small that ``1 + x == 1`` in floating-point
    accuracy."""
    return Tensor._op(Log1p, a, constant=constant)
