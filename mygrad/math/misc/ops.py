from mygrad.operation_base import Operation
import numpy as np


__all__ = ["Abs",
           "Sqrt",
           "Cbrt"]


class Abs(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.abs(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * np.piecewise(a.data, [a.data < 0, a.data == 0, a.data > 0], [-1, np.nan, 1]), **kwargs)


class Sqrt(Operation):
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



class Cbrt(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.cbrt(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (3 * np.cbrt(a.data ** 2)), **kwargs)