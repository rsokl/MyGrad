from mygrad.operations.operation_base import Operation
import numpy as np

__all__ = ["Sinh",
           "Cosh",
           "Tanh",
           "Csch",
           "Sech",
           "Coth",
           ""
           "Arcsinh",
           "Arccosh",
           "Arctanh",
           "Arccsch",
           "Arccoth"]


class Sinh(Operation):
    """ f(a) -> sinh(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.sinh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * np.cosh(a.data), **kwargs)


class Cosh(Operation):
    """ f(a) -> cosh(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.cosh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * np.sinh(a.data), **kwargs)


class Tanh(Operation):
    """ f(a) -> tanh(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.tanh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * (1 - np.tanh(a.data) ** 2), **kwargs)


class Csch(Operation):
    """ f(a) -> csch(a) """
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.sinh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * -np.cosh(a.data) / np.sinh(a.data)**2)


class Sech(Operation):
    """ f(a) -> sech(a) """
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.cosh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * -np.sinh(a.data) / np.cosh(a.data)**2, **kwargs)


class Coth(Operation):
    """ f(a) -> coth(a) """
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.tanh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad * -1 / np.sinh(a.data)**2, **kwargs)


class Arcsinh(Operation):
    """ f(a) -> arcsinh(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.arcsinh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / np.sqrt(1 + a.data ** 2), **kwargs)


class Arccosh(Operation):
    """ f(a) -> arccosh(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.arccosh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / np.sqrt(a.data ** 2 - 1), **kwargs)


class Arctanh(Operation):
    """ f(a) -> arctanh(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.arctanh(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (1 - a.data ** 2), **kwargs)


class Arccsch(Operation):
    """ f(a) -> arccsch(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.arcsinh(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(-grad / (np.abs(a.data) * np.sqrt(1 + a.data ** 2)), **kwargs)


class Arccoth(Operation):
    """ f(a) -> arccoth(a) """
    def __call__(self, a):
        self.variables = (a,)
        return np.arctanh(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (1 - a.data ** 2), **kwargs)
