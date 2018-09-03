from mygrad.operation_base import Operation
import numpy as np

__all__ = ["Sin",
           "Sinc",
           "Cos",
           "Tan",
           "Csc",
           "Sec",
           "Cot",
           "Arcsin",
           "Arccos",
           "Arctan",
           "Arccsc",
           "Arcsec",
           "Arccot"]


class Sin(Operation):
    """ f(a) -> sin(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.sin(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad * np.cos(a.data)


def _dsinc(x):
    x = x * np.pi
    return (x * np.cos(x) - np.sin(x)) / x ** 2


class Sinc(Operation):
    """ f(a) -> sin(pi*a)/(pi*a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.sinc(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        x = a.data
        return np.pi * grad * np.piecewise(x, [x == 0, x != 0], [np.zeros_like, _dsinc])


class Cos(Operation):
    """ f(a) -> cos(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.cos(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad * -np.sin(a.data)


class Tan(Operation):
    """ f(a) -> tan(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.tan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad / np.cos(a.data)**2


class Csc(Operation):
    """ f(a) -> csc(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.sin(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad * -np.cos(a.data) / np.sin(a.data)**2


class Sec(Operation):
    """ f(a) -> sec(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.cos(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad * np.sin(a.data) / np.cos(a.data)**2


class Cot(Operation):
    """ f(a) -> cot(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.tan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return -grad / np.sin(a.data)**2

class Arcsin(Operation):
    """ f(a) -> arcsin(a)"""
    def __call__(self, a):
        self.variables = (a,)
        if np.any(1 < a) or np.any(a < -1):
            raise ValueError("Invalid arcsin-domain value")
        return np.arcsin(a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arcsin / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        return np.select([np.abs(a.data) != 1], [grad / np.sqrt(1 - a.data ** 2)])


class Arccos(Operation):
    """ f(a) -> arccos(a)"""
    def __call__(self, a):
        self.variables = (a,)
        if np.any(1 < a) or np.any(a < -1):
            raise ValueError("Invalid arccos-domain value")
        return np.arccos(a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arccos / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        return np.select([np.abs(a.data) != 1], [-grad / np.sqrt(1 - a.data ** 2)])


class Arctan(Operation):
    """ f(a) -> arctan(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.arctan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad / (1 + a.data ** 2)


class Arccsc(Operation):
    """ f(a) -> arccsc(a)"""
    def __call__(self, a):
        self.variables = (a,)
        if np.any(-1 < a) and np.any(a < 1):
            raise ValueError("Invalid arccsc-domain value")
        return np.arcsin(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arccsc / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        return np.select([np.abs(a.data) != 1], [-grad / (np.abs(a.data) * np.sqrt(a.data ** 2 - 1))])


class Arcsec(Operation):
    """ f(a) -> arcsec(a)"""
    def __call__(self, a):
        self.variables = (a,)
        if np.any(-1 < a) and np.any(a < 1):
            raise ValueError("Invalid arcsec-domain value")
        return np.arccos(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arcsec / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        return np.select([np.abs(a.data) != 1], [grad / (np.abs(a.data) * np.sqrt(a.data ** 2 - 1))])


class Arccot(Operation):
    """ f(a) -> arccot(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.piecewise(a.data, [a.data == 0, a.data != 0], [np.pi / 2, lambda x: np.arctan(1 / x)])

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return -grad / (1 + a.data ** 2)