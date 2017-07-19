from ..operations import Operation
import numpy as np

class Tensor_Transpose_Property(Operation):
    def __call__(self, a):
        """ Same as a.transpose(), except that a is returned if
            a.ndim < 2."""
        self.a = a
        return self.a.data.T

    def backward_a(self, grad):
        self.a.backward(grad.T)


class Transpose(Operation):
    def __call__(self, a, axes=None):
        self.a = a
        if axes is not None:
            self.axes = tuple(axis % a.ndim for axis in axes)
        else:
            self.axes = tuple(range(a.ndim)[::-1])
        return self.a.data.transpose(axes)

    def backward_a(self, grad):
        if self.a.ndim > 1:
            grad = grad.transpose(np.argsort(self.axes))
        self.a.backward(grad)