from .operation_base import Operation
import numpy as np


class Abs(Operation):
    def __call__(self, a):
        self.a = a
        return np.abs(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * np.piecewise(a.data, [a.data < 0, a.data >= 0], [lambda data: -a.data, lambda data: a.data]))
