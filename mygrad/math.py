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


