from abc import ABC
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor

from mygrad.operation_base import BinaryUfunc, BroadcastableOp, UnaryUfunc

__all__ = [
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Reciprocal",
    "Power",
    "Square",
    "Positive",
    "Negative",
    "AddSequence",
    "MultiplySequence",
]

# Binary Ops


class Add(BinaryUfunc):
    numpy_ufunc = np.add

    def backward_var(self, grad, index, **kwargs):
        return grad


class Subtract(BinaryUfunc):
    numpy_ufunc = np.subtract

    def backward_var(self, grad, index, **kwargs):
        if index == 0:
            return grad
        else:
            return -grad


class Multiply(BinaryUfunc):
    numpy_ufunc = np.multiply

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            return grad * b.data
        elif index == 1:  # backprop through b
            return grad * a.data


class Divide(BinaryUfunc):
    numpy_ufunc = np.divide

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            return grad / b.data
        else:  # broadcast through b
            return -grad * a.data / (b.data ** 2)


class Power(BinaryUfunc):
    numpy_ufunc = np.power

    def backward_var(self, grad, index, **kwargs):
        x, y = (i.data for i in self.variables)

        if index == 0:
            return grad * y * (x ** np.where(y, (y - 1), 1))
        else:
            return grad * (x ** y) * np.log(np.where(x, x, 1))


# Unary Ops


class Reciprocal(UnaryUfunc):
    numpy_ufunc = np.reciprocal

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return -grad * np.reciprocal(a.data ** 2)


class Square(UnaryUfunc):
    numpy_ufunc = np.square

    def backward_var(self, grad, index, **kwargs):
        grad = 2 * grad
        grad *= self.variables[index].data
        return grad


class Positive(UnaryUfunc):
    numpy_ufunc = np.positive

    def backward_var(self, grad, index, **kwargs):
        return np.positive(grad, where=self.where)


class Negative(UnaryUfunc):
    numpy_ufunc = np.negative

    def backward_var(self, grad, index, **kwargs):
        return np.negative(grad, where=self.where)


class AddSequence(BroadcastableOp):
    """Performs f(a, b, ..., z) = a + b + ... + z"""

    def __call__(self, *input_vars):
        assert len(input_vars) > 1, "`add_sequence` requires at least two operands"
        self.variables = input_vars
        out = sum(var.data for var in input_vars)
        return out

    def backward_var(self, grad, index, **kwargs):
        return grad


class MultiplySequence(BroadcastableOp):
    """ Performs f(a, b, ..., z) = a * b * ... * z"""

    def __call__(self, *input_vars):
        self.variables = input_vars
        assert 2 <= len(self.variables)

        out = reduce(lambda x, y: x * y, (var.data for var in input_vars))
        self._iszero = np.any(out == 0)
        return out

    def backward(self, grad, *, graph, **kwargs):
        """Back-propagates the gradient through all of the operation's inputs. This needs to be updated
        by an operation if that operation takes more than 2 Tensor arguments."""
        if not self._iszero:
            self._product = grad * reduce(
                lambda x, y: x * y, (var.data for n, var in enumerate(self.variables))
            )
        else:
            self._product = None
        super().backward(grad, graph=graph, **kwargs)

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        if not self._iszero:
            return self._product / var.data
        else:
            return grad * reduce(
                lambda x, y: x * y,
                (var.data for n, var in enumerate(self.variables) if n != index),
            )
