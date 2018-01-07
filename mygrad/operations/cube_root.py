from .operation_base import Operation
import numpy as np


class Cbrt(Operation):
    def __call__(self, a):
        self.a = a
        return np.cbrt(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (3 * np.cbrt(a.data ** 2)))
