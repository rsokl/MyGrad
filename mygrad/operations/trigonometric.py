from mygrad.operations.multivar_operations import MultiVarOperation
import numpy as np

__all__ = ["Sin",
           "Cos",
           "Tan",
           "Csc",
           "Sec",
           "Cot"]


class Sin(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.sin(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * np.cos(a.data))


class Cos(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.cos(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * -np.sin(a.data))


class Tan(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.tan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / np.cos(a.data)**2)


class Csc(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.sin(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * -np.cos(a.data) / np.sin(a.data)**2)


class Sec(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.cos(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * np.sin(a.data) / np.cos(a.data)**2)


class Cot(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.tan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(-grad / np.sin(a.data)**2)
