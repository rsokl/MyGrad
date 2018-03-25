from .multivar_operations import MultiVarBroadcastableOp
import numpy as np

class Divide(MultiVarBroadcastableOp):
    def __call__(self, a, b):
        """ f(a, b) -> a / b"""
        if np.any(b == 0):
            raise ZeroDivisionError
        self.variables = (a, b)
        out = a.data / b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            a.backward(grad / b.data, **kwargs)
        else:           # broadcast through b
            b.backward(- grad * a.data / (b.data ** 2), **kwargs)
