from mygrad.operations.multivar_operations import Operation
import numpy as np

__all__ = ["Log",
           "Log2",
           "Log10"]


class Log(Operation):
    def __call__(self, a):
        self.variables = (a,)
        if np.any(a <= 0):
            raise ValueError("Invalid log-domain value")
        return np.log(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / a.data, **kwargs)


class Log2(Operation):
    def __call__(self, a):
        self.variables = (a,)
        if np.any(a <= 0):
            raise ValueError("Invalid log-domain value")
        return np.log2(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (a.data * np.log(2)), **kwargs)


class Log10(Operation):
    def __call__(self, a):
        self.variables = (a,)
        if np.any(a <= 0):
            raise ValueError("Invalid log-domain value")
        return np.log10(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (a.data * np.log(10)), **kwargs)
