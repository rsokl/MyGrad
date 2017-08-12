from .operation_base import Operation
import numpy as np


class Dot(Operation):
    scalar_only = True

    def __call__(self, a, b):
        assert a.ndim == 1 and b.ndim == 1
        self.a = a
        self.b = b
        return np.dot(a.data, b.data)

    def backward_a(self, grad):
        self.a.backward(grad * self.b)

    def backward_b(self, grad):
        self.b.backward(grad * self.a)
