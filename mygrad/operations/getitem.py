from .operation_base import Operation
import numpy as np


class GetItem(Operation):
    def __call__(self, a, index):
        self.a = a
        self.index = index
        out = self.a.data[index]
        self.shape = out.shape if isinstance(out, np.ndarray) else None
        return out

    def backward_a(self, grad):
        out = np.zeros_like(self.a.data)
        out[self.index] = grad
        self.a.backward(out)

