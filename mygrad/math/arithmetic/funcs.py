from .ops import Add, Subtract, Power, Square, Divide, Multiply, MultiplySequence, AddSequence
from .ops import Positive, Negative, Reciprocal
from mygrad.tensor_base import Tensor


__all__ = ["add",
           "subtract",
           "power",
           "divide",
           "square",
           "reciprocal",
           "multiply",
           "multiply_sequence",
           "add_sequence",
           "positive",
           "negative"]


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
        mygrad.Tensor"""
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
        mygrad.Tensor"""
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
    """ Add a sequence of N tensors.

        ``f(a, b, ...) -> a + b + ...``

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
    return Tensor._op(MultiplySequence, *variables, constant=constant)


def add_sequence(*variables, constant=False):
    """ Multiply a sequence of N tensors.

        ``f(a, b, ...) -> a * b * ...``

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
    return Tensor._op(AddSequence, *variables, constant=constant)


def positive(a, where=True, constant=False):
    """ ``f(a) -> +a``

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
        mygrad.Tensor"""
    return Tensor._op(Negative, a, op_kwargs=(dict(where=where)), constant=constant)
