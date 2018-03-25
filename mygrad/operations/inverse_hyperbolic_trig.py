from mygrad.operations.multivar_operations import MultiVarOperation
import numpy as np

__all__ = ["Arcsinh",
           "Arccosh",
           "Arctanh",
           "Arccsch",
           "Arccoth"]


class Arcsinh(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.arcsinh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / np.sqrt(1 + a.data ** 2), **kwargs)


class Arccosh(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.arccosh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / np.sqrt(a.data ** 2 - 1), **kwargs)


class Arctanh(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.arctanh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (1 - a.data ** 2), **kwargs)


class Arccsch(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.arcsinh(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(-grad / (np.abs(a.data) * np.sqrt(1 + a.data ** 2)), **kwargs)


class Arccoth(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.arctanh(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (1 - a.data ** 2), **kwargs)
