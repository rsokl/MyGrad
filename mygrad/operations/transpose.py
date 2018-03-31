from mygrad.operations.operation_base import Operation
import numpy as np


class Tensor_Transpose_Property(Operation):
    def __call__(self, a):
        """ Same as a.transpose(), except that a is returned if
            a.ndim < 2.

            Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return a.data.T

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad.T, **kwargs)


class Transpose(Operation):
    def __call__(self, a, axes=None):
        self.variables = (a,)
        if axes is not None:
            self.axes = tuple(axis % a.ndim for axis in axes)
        else:
            self.axes = tuple(range(a.ndim)[::-1])
        return a.data.transpose(axes)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        if a.ndim > 1:
            grad = grad.transpose(np.argsort(self.axes))
        a.backward(grad)
