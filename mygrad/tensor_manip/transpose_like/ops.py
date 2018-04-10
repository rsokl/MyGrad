from mygrad.operation_base import Operation
import numpy as np

__all__ = ["Tensor_Transpose_Property",
           "Transpose",
           "MoveAxis"]


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
        return np.transpose(a.data, axes)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        if a.ndim > 1:
            grad = grad.transpose(np.argsort(self.axes))
        a.backward(grad)


class MoveAxis(Operation):
    def __call__(self, a, source, destination):
        self.variables = (a,)
        self.source = source
        self.destination = destination
        return np.moveaxis(a, source, destination)

    def backward_var(self, grad, index, **kwargs):
        if not index == 0:
            raise IndexError
        self.variables[index].backward(np.moveaxis(grad, self.destination, self.source),
                                       **kwargs)


class SwapAxes(Operation):
    def __call__(self, a, axis1, axis2):
        self.variables = (a,)
        self.axis1 = axis1
        self.axis2 = axis2
        return np.swapaxes(a.data, axis1, axis2)

    def backward_var(self, grad, index, **kwargs):
        if not index == 0:
            raise IndexError
        self.variables[index].backward(np.swapaxes(grad, self.axis2, self.axis1),
                                       **kwargs)
