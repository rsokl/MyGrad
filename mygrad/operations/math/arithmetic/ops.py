from functools import reduce
import numpy as np
from mygrad.operations.multivar_operations import BroadcastableOp, Operation

__all__ = ["Add",
           "Subtract",
           "Multiply",
           "Divide",
           "Reciprocal",
           "Power",
           "Positive",
           "Negative",
           "AddSequence",
           "MultiplySequence"]


class Add(BroadcastableOp):
    def __call__(self, a, b):
        """ Performs 'add' forward-pass: f(a,b) -> a + b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """

        self.variables = (a, b)
        out = a.data + b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad, **kwargs)


class Subtract(BroadcastableOp):
    def __call__(self, a, b):
        """ f(a,b) -> a - b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """
        self.variables = (a, b)
        out = a.data - b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            a.backward(grad, **kwargs)
        else:
            b.backward(-grad, **kwargs)


class Multiply(BroadcastableOp):
    def __call__(self, a, b):
        """ Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor"""
        self.variables = (a, b)
        out = a.data * b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            a.backward(grad * b.data, **kwargs)
        elif index == 1:  # backprop through b
            b.backward(grad * a.data, **kwargs)


class Divide(BroadcastableOp):
    def __call__(self, a, b):
        """ f(a, b) -> a / b"""
        self.variables = (a, b)
        out = a.data / b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            a.backward(grad / b.data, **kwargs)
        else:           # broadcast through b
            b.backward(- grad * a.data / (b.data ** 2), **kwargs)


class Reciprocal(BroadcastableOp):
    def __call__(self, a, b):
        """ f(a) -> 1 / a"""
        self.variables = (a, )
        return np.reciprocal(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(-grad * np.reciprocal(a.data ** 2), **kwargs)


class Power(BroadcastableOp):
    def __call__(self, a, b):
        """ f(a, b) -> a ** b

            Parameters
            ----------
            a: mygrad.Tensor
            b: mygrad.Tensor"""
        self.variables = (a, b)
        out = a.data ** b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            grad = grad * b.data * (a.data ** (b.data - 1))
            a.backward(grad, **kwargs)

        else:
            grad = np.nan_to_num(grad * (a.data ** b.data) * np.log(a.data))
            b.backward(grad, **kwargs)


class Positive(Operation):
    """ f(a) = +a """
    def __call__(self, a, where=True):
        """ Parameters
            ----------
            a: mygrad.Tensor

            where : array_like, optional
                Values of True indicate to calculate the ufunc at that position,
                values of False indicate to leave the value in the output alone."""
        self.variables = (a,)
        self.conf = dict(where=where)
        return np.positive(a.data, where=where)

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(np.positive(grad, **self.conf), **kwargs)


class Negative(Operation):
    """ f(a) = -a """
    def __call__(self, a, where=True):
        """ Parameters
            ----------
            a : mygrad.Tensor

            where : array_like, optional
                Values of True indicate to calculate the ufunc at that position,
                values of False indicate to leave the value in the output alone."""
        self.variables = (a,)
        self.conf = dict(where=where)
        return np.negative(a.data, where=where)

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(np.negative(grad, **self.conf), **kwargs)


class AddSequence(BroadcastableOp):
    """ Performs f(a, b, ..., z) = a + b + ... + z"""
    def __call__(self, *input_vars):
        assert len(input_vars) > 1, "`add_sequence` requires at least two operands"
        self.variables = input_vars
        out = sum(var.data for var in input_vars)
        return out

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad, **kwargs)


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