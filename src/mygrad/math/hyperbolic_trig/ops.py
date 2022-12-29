import numpy as np

from mygrad.operation_base import Operation, UnaryUfunc

__all__ = [
    "Sinh",
    "Cosh",
    "Tanh",
    "Csch",
    "Sech",
    "Coth",
    "" "Arcsinh",
    "Arccosh",
    "Arctanh",
    "Arccsch",
    "Arccoth",
]


class Sinh(UnaryUfunc):
    numpy_ufunc = np.sinh

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * np.cosh(a.data)


class Cosh(UnaryUfunc):
    numpy_ufunc = np.cosh

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * np.sinh(a.data)


class Tanh(UnaryUfunc):
    numpy_ufunc = np.tanh

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * (1 - np.tanh(a.data) ** 2)


class Csch(Operation):
    """f(a) -> csch(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.sinh(a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * -np.cosh(a.data) / np.sinh(a.data) ** 2


class Sech(Operation):
    """f(a) -> sech(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.cosh(a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * -np.sinh(a.data) / np.cosh(a.data) ** 2


class Coth(Operation):
    """f(a) -> coth(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.tanh(a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * -1 / np.sinh(a.data) ** 2


class Arcsinh(UnaryUfunc):
    numpy_ufunc = np.arcsinh

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / np.sqrt(1 + a.data**2)


class Arccosh(UnaryUfunc):
    numpy_ufunc = np.arccosh

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / np.sqrt(a.data**2 - 1)


class Arctanh(UnaryUfunc):
    numpy_ufunc = np.arctanh

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / (1 - a.data**2)


class Arccsch(Operation):
    """f(a) -> arccsch(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return np.arcsinh(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return -grad / (np.abs(a.data) * np.sqrt(1 + a.data**2))


class Arccoth(Operation):
    """f(a) -> arccoth(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return np.arctanh(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / (1 - a.data**2)
