from mygrad.operations.multivar_operations import MultiVarOperation
import numpy as np


class Sqrt(MultiVarOperation):
    def __call__(self, a):
        """ f(a) = sqrt(a)

            Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return np.sqrt(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (2 * np.sqrt(a.data)), **kwargs)

