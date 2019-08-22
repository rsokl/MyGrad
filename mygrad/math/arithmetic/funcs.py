from mygrad.tensor_base import Tensor

from .ops import (
    Add,
    AddSequence,
    Divide,
    Multiply,
    MultiplySequence,
    Negative,
    Positive,
    Power,
    Reciprocal,
    Square,
    Subtract,
)

__all__ = [
    "add",
    "add_sequence",
    "divide",
    "multiply",
    "multiply_sequence",
    "negative",
    "positive",
    "power",
    "reciprocal",
    "square",
    "subtract",
]


def add(a, b, constant=False):
    """ ``f(a, b) -> a + b``

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

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.add(1.0, 4.0)
    Tensor(5.0)
    >>> x1 = mg.arange(9.0).reshape((3, 3))
    >>> x2 = mg.arange(3.0)
    >>> mg.add(x1, x2)  # equivalent to `x1 + x2`
    Tensor([[  0.,   2.,   4.],
            [  3.,   5.,   7.],
            [  6.,   8.,  10.]])"""
    return Tensor._op(Add, a, b, constant=constant)


def subtract(a, b, constant=False):
    """ ``f(a, b) -> a - b``

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

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.subtract(1.0, 4.0, constant=True)  # resulting tensor will not back-propagate a gradient
    Tensor(5.0)
    >>> x1 = mg.arange(9.0).reshape((3, 3))
    >>> x2 = mg.arange(3.0)
    >>> mg.subtract(x2, x1)  # equivalent to `x2 - x1`
    Tensor([[  0.,   0.,   0.],
            [  3.,   3.,   3.],
            [  6.,   6.,  6.]])
    """
    return Tensor._op(Subtract, a, b, constant=constant)


def divide(a, b, constant=False):
    """ ``f(a, b) -> a / b``

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
    return Tensor._op(Divide, a, b, constant=constant)


def square(a, constant=False):
    """ ``f(a) -> a ** 2``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Square, a, constant=constant)


def reciprocal(a, constant=False):
    """ ``f(a) -> 1 / a``

    Parameters
    ----------
    a : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Reciprocal, a, constant=constant)


def power(a, b, constant=False):
    """ ``f(a, b) -> a ** b``

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
    return Tensor._op(Power, a, b, constant=constant)


def multiply(a, b, constant=False):
    """ ``f(a, b) -> a * b``

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
    return Tensor._op(Multiply, a, b, constant=constant)


def multiply_sequence(*variables, constant=False):
    """  ``f(a, b, ...) -> a * b * ...``

    Multiply a sequence of N tensors.

    Parameters
    ----------
    variables : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    It is more efficient to back-propagate through this
    function than it is through a computational graph
    with N-1 corresponding multiplication operations."""
    if len(variables) < 2:
        raise ValueError(
            "`multiply_sequence` requires at least two inputs, got {} inputs".format(
                len(variables)
            )
        )
    return Tensor._op(MultiplySequence, *variables, constant=constant)


def add_sequence(*variables, constant=False):
    """ ``f(a, b, ...) -> a + b + ...``

    Add a sequence of N tensors.

    Parameters
    ----------
    variables : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    It is more efficient to back-propagate through this
    function than it is through a computational graph
    with N-1 corresponding addition operations."""
    if len(variables) < 2:
        raise ValueError(
            "`add_sequence` requires at least two inputs, got {} inputs".format(
                len(variables)
            )
        )
    return Tensor._op(AddSequence, *variables, constant=constant)


def positive(a, where=True, constant=False):
    """ ``f(a) -> +a``

    Creates a new tensor.

    Parameters
    ----------
    a : array_like

    where : numpy.ndarray
        Accepts a boolean array which is broadcast together
        with the operand(s). Values of True indicate to calculate
        the function at that position, values of False indicate
        to leave the value in the output alone.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Positive, a, op_kwargs=(dict(where=where)), constant=constant)


def negative(a, where=True, constant=False):
    """ ``f(a) -> -a``

    Parameters
    ----------
    a : array_like

    where : numpy.ndarray
        Accepts a boolean array which is broadcast together
        with the operand(s). Values of True indicate to calculate
        the function at that position, values of False indicate
        to leave the value in the output alone.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

"""
    return Tensor._op(Negative, a, op_kwargs=(dict(where=where)), constant=constant)
