from .operation_base import Operation
import numpy as np

__all__ = ["Log",
           "Log2",
           "Log10"]


class Log(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a <= 0):
            raise ValueError("Invalid log-domain value")
        return np.log(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / self.a.data)


class Log2(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a <= 0):
            raise ValueError("Invalid log-domain value")
        return np.log2(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (self.a.data * np.log(2)))


class Log10(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a <= 0):
            raise ValueError("Invalid log-domain value")
        return np.log10(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (self.a.data * np.log(10)))
