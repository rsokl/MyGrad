from .ops import Add, Subtract, Power, Divide, Multiply, MultiplySequence, AddSequence
from .ops import Positive, Negative, Reciprocal
from mygrad.tensor_base import Tensor


__all__ = ["add",
           "subtract",
           "power",
           "divide",
           "reciprocal",
           "multiply",
           "multiply_sequence",
           "add_sequence",
           "positive",
           "negative"]


def add(a, b):
    return Tensor._op(Add, a, b)


def subtract(a, b):
    return Tensor._op(Subtract, a, b)


def divide(a, b):
    return Tensor._op(Divide, a, b)


def reciprocal(a):
    return Tensor._op(Reciprocal, a)


def power(a, b):
    return Tensor._op(Power, a, b)


def multiply(a, b):
    return Tensor._op(Multiply, a, b)


def multiply_sequence(*variables):
    return Tensor._op(MultiplySequence, *variables)


def add_sequence(*variables):
    return Tensor._op(AddSequence, *variables)


def positive(a, where=True):
    return Tensor._op(Positive, a, op_kwargs=(dict(where=where)))


def negative(a, where=True):
    return Tensor._op(Negative, a, op_kwargs=(dict(where=where)))
