from .operation_base import Operation
import numpy as np

__all__ = ["Exp"]

class Exp(Operation):
    def __call__(self, a):
        self.a = a
        return np.exp(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * np.exp(self.a.data))
