from mygrad.operations.multivar_operations import MultiVarBroadcastableOp
import numpy as np


class Power(MultiVarBroadcastableOp):
    def __call__(self, a, b):
        self.variables = (a, b)
        out = a.data ** b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            grad = grad * b.data * (a.data ** (b.data - 1))
            a.backward(grad, **kwargs)

        else:
            grad = np.nan_to_num(grad * (a.data ** b.data) * np.log(a.data))
            b.backward(grad, **kwargs)

