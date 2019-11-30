from mygrad.tensor_base import Tensor

from .ops import Abs, Cbrt, Maximum, Minimum, Sqrt

__all__ = ["abs", "absolute", "cbrt", "clip", "sqrt", "maximum", "minimum"]


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


def clip(a, a_min, a_max, constant=False):
    """ Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Equivalent to `mg.maximum(a_min, mg.minimum(a, a_max))``.

    No check is performed to ensure ``a_min < a_max``.

    This docstring was adapted from that of `numpy.clip`

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.

    a_min : Optional[float, array_like]
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.

    a_max : Optional[float, array_like]
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`. If `a_min` or `a_max` are array_like, then the three
        arrays will be broadcasted to match their shapes.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    Tensor
        A tensor with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.arange(10)
    >>> mg.clip(a, 1, 8)
    Tensor([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a
    Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mg.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
    Tensor([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])"""
    if a_min is None and a_max is None:
        raise ValueError("`a_min` and `a_max` cannot both be set to `None`")

    if a_min is not None:
        a = maximum(a_min, a, constant=constant)

    if a_max is not None:
        a = minimum(a_max, a, constant=constant)

    return a
