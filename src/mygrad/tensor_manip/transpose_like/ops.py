import numpy as np

from mygrad.operation_base import Operation

__all__ = ["Tensor_Transpose_Property", "Transpose", "MoveAxis", "Roll", "SwapAxes"]


class Tensor_Transpose_Property(Operation):
    can_return_view = True

    def __call__(self, a):
        """ Same as a.transpose(), except that a is returned if
            a.ndim < 2.

            Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return a.data.T

    def backward_var(self, grad, index, **kwargs):
        return grad.T


class Transpose(Operation):
    can_return_view = True

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
        return grad


class MoveAxis(Operation):
    can_return_view = True

    def __call__(self, a, source, destination):
        self.variables = (a,)
        self.source = source
        self.destination = destination
        return np.moveaxis(a.data, source, destination)

    def backward_var(self, grad, index, **kwargs):
        if not index == 0:  # pragma: no cover
            raise IndexError
        return np.moveaxis(grad, self.destination, self.source)


class SwapAxes(Operation):
    can_return_view = True

    def __call__(self, a, axis1, axis2):
        self.variables = (a,)
        self.axis1 = axis1
        self.axis2 = axis2
        return np.swapaxes(a.data, axis1, axis2)

    def backward_var(self, grad, index, **kwargs):
        if not index == 0:  # pragma: no cover
            raise IndexError
        return np.swapaxes(grad, self.axis2, self.axis1)


class Roll(Operation):
    def __call__(self, a, shift, axis):
        self.variables = (a,)
        self.shift = shift
        self.axis = axis
        return np.roll(a.data, shift=shift, axis=axis)

    def backward_var(self, grad, index, **kwargs):
        if not index == 0:  # pragma: no cover
            raise IndexError
        rev_shift = (
            -self.shift
            if not hasattr(self.shift, "__iter__")
            else tuple(-i for i in self.shift)
        )
        return np.roll(grad, axis=self.axis, shift=rev_shift)
