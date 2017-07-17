from .operation_base import BroadcastableOp
import numpy as np


class Power(BroadcastableOp):
    def __call__(self, a, b):
        self.a = a
        self.b = b
        out = a.data ** b.data
        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        grad = super(Power, self).backward_a(grad * self.b.data * (self.a.data ** (self.b.data - 1)))
        self.a.backward(grad)

    def backward_b(self, grad):
        grad = np.nan_to_num(grad * (self.a.data ** self.b.data) * np.log(self.a.data))
        grad = super(Power, self).backward_b(grad)
        self.b.backward(grad)
