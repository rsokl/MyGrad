from .operation_base import Operation
import numpy as np


class Log(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a <= 0):
            raise ValueError("Invalid log-domain value")
        return np.log(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / self.a.data)
