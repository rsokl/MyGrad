from .multivar_operations import BroadcastableOp
from functools import reduce
import numpy as np

__all__ = ["MultiplySequence"]


class MultiplySequence(BroadcastableOp):
    """ Performs f(a, b, ..., z) = a * b * ... * z"""
    def __call__(self, *input_vars):
        assert len(input_vars) > 1, "`multiply_sequence` requires at least two operands"
        self.variables = input_vars
        out = reduce(lambda x, y: x*y, (var.data for var in input_vars))
        self._iszero = np.any(out == 0)
        return out

    def backward(self, grad, **kwargs):
        """ Back-propagates the gradient through all of the operation's inputs. This needs to be updated
            by an operation if that operation takes more than 2 Tensor arguments."""
        if not self._iszero:
            self._product = grad * reduce(lambda x, y: x*y, (var.data for n, var in enumerate(self.variables)))
        else:
            self._product = None

        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index, **kwargs)

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        if not self._iszero:
            grad = self._product / var.data
        else:
            grad = grad * reduce(lambda x, y: x*y, (var.data for n, var in enumerate(self.variables) if n != index))
        var.backward(grad, **kwargs)

