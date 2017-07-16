from .tensor_base import Tensor
from .operations import *

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


def subtract(a, b):
    #pass
    # YOUR CODE HERE
    return Tensor._op(Subtract, a, b)