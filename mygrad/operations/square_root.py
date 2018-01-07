from .operation_base import Operation
import numpy as np


class Sqrt(Operation):
    def __call__(self, a):
        self.a = a
        return np.sqrt(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (2 * np.sqrt(self.a.data)))
