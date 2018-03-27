from .multivar_operations import Operation
import numpy as np


class Cbrt(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.cbrt(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (3 * np.cbrt(a.data ** 2)), **kwargs)

