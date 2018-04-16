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


def add(a, b, constant=False):
    return Tensor._op(Add, a, b, constant=constant)


def subtract(a, b, constant=False):
    return Tensor._op(Subtract, a, b, constant=constant)


def divide(a, b, constant=False):
    return Tensor._op(Divide, a, b, constant=constant)


def reciprocal(a, constant=False):
    return Tensor._op(Reciprocal, a, constant=constant)


def power(a, b, constant=False):
    return Tensor._op(Power, a, b, constant=constant)


def multiply(a, b, constant=False):
    return Tensor._op(Multiply, a, b, constant=constant)


def multiply_sequence(*variables, constant=False):
    return Tensor._op(MultiplySequence, *variables, constant=constant)


def add_sequence(*variables, constant=False):
    return Tensor._op(AddSequence, *variables, constant=constant)


def positive(a, where=True, constant=False):
    return Tensor._op(Positive, a, op_kwargs=(dict(where=where)), constant=constant)


def negative(a, where=True, constant=False):
    return Tensor._op(Negative, a, op_kwargs=(dict(where=where)), constant=constant)
