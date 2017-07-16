from .operation_base import BroadcastableOp
import numpy as np

class Divide(BroadcastableOp):
    def __call__(self, a, b):
        if np.any(b == 0):
            raise ZeroDivisionError
        self.a = a
        self.b = b
        out = a.data / b.data
        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        self.a.backward(super(Divide, self).backward_a(grad / self.b.data))

    def backward_b(self, grad):
        self.b.backward(super(Divide, self).backward_b(- grad * self.a.data / (self.b.data ** 2)))