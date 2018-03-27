from mygrad.operations.multivar_operations import Operation
import numpy as np

__all__ = ["Sin",
           "Cos",
           "Tan",
           "Csc",
           "Sec",
           "Cot"]


class Sin(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.sin(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * np.cos(a.data), **kwargs)


class Cos(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.cos(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * -np.sin(a.data), **kwargs)


class Tan(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.tan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / np.cos(a.data)**2, **kwargs)


class Csc(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.sin(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * -np.cos(a.data) / np.sin(a.data)**2, **kwargs)


class Sec(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.cos(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * np.sin(a.data) / np.cos(a.data)**2, **kwargs)


class Cot(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.tan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(-grad / np.sin(a.data)**2, **kwargs)
