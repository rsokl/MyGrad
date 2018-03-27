from .multivar_operations import Operation
import numpy as np

__all__ = ["Exp"]


class Exp(Operation):
    def __call__(self, a):
        """ f(a) -> exp(a)

            Parameters
            ----------
            a : mygrad.Tensor

            Returns
            -------
            numpy.ndarray"""
        self.variables = (a,)
        return np.exp(a.data)

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        var.backward(grad * np.exp(var.data))

