from .operation_base import Operation
import numpy as np


__all__ = ["Sum"]


class Sum(Operation):

    def __call__(self, a, axis=None, keepdims=False):
        self.a = a

        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)
        self.axis = axis

        self.keepdims = keepdims
        out = a.data.sum(axis=axis, keepdims=keepdims)
        self.outshape = out.shape if isinstance(out, np.ndarray) else None
        return out

    def backward_a(self, grad):
        if self.outshape is None:
            self.a.backward(np.full(self.a.shape, grad, dtype=float))
            return None

        if self.keepdims:
            self.a.backward(grad * np.ones_like(self.a.data, dtype=float))
            return None

        index = [slice(None) for i in range(self.a.ndim)]
        for i in self.axis:
            index[i] = np.newaxis
        self.a.backward(grad[index] * np.ones_like(self.a.data, dtype=float))