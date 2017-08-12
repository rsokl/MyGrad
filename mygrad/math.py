from .tensor_base import Tensor
from .operations import *

__all__ = ["add",
           "add_sequence",
           "divide",
           "log",
           "multiply",
           "multiply_sequence",
           "power",
           "subtract"]

def add(a, b):
    """ f(a, b) -> a + b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Add, a, b)


def add_sequence(*variables):
    """ f(a, b, ...) -> a + b + ...

        Parameters
        ----------
        variables : Sequence[Union[tensor-like, Number]]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(AddSequence, *variables)


def divide(a, b):
    """ f(a, b) -> a / b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Divide, a, b)


def log(a):
    """ f(a)-> log(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Log, a)


def multiply(a, b):
    """ f(a, b) -> a * b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Multiply, a, b)


def multiply_sequence(*variables):
    """ f(a, b, ...) -> a * b * ...

        Parameters
        ----------
        variables : Sequence[Union[tensor-like, Number]]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(MultiplySequence, *variables)


def power(a, b):
    """ f(a, b) -> a ** b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Power, a, b)


def subtract(a, b):
    """ f(a, b) -> a - b
        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]
        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Subtract, a, b)

def dot(x, w):
    """ Calculates dot product between two tensors of shape (N)

        Parameters
        ----------
        x : Union[mygrad.Tensor, array_like], shape=(N)

        w : Union[mygrad.Tensor, array_like], shape=(D)

        Returns
        -------
        Tensor, shape=(1)
            The result of the dot product `x` with `w`

        Notes
        -----
        This is a "scalar-only" operation, meaning that back propagation through
        this layer assumes that a scalar (i.e. a 0-dimensional tensor) will invoke
        `tensor.backward()` for the computational graph. This is standard for a
        neural network, which terminates in a scalar loss."""
    return Tensor._op(Dot, x, w)
