from mygrad.operations.multivar_operations import MultiVarOperation
import numpy as np

__all__ = ["Arcsin",
           "Arccos",
           "Arctan",
           "Arccsc",
           "Arcsec",
           "Arccot"]


class Arcsin(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        if np.any(1 < a) or np.any(a < -1):
            raise ValueError("Invalid arcsin-domain value")
        return np.arcsin(a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arcsin / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        a.backward(np.select([np.abs(a.data) != 1], [grad / np.sqrt(1 - a.data ** 2)]), **kwargs)


class Arccos(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        if np.any(1 < a) or np.any(a < -1):
            raise ValueError("Invalid arccos-domain value")
        return np.arccos(a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arccos / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        a.backward(np.select([np.abs(a.data) != 1], [-grad / np.sqrt(1 - a.data ** 2)]), **kwargs)


class Arctan(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.arctan(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (1 + a.data ** 2), **kwargs)


class Arccsc(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        if np.any(-1 < a) and np.any(a < 1):
            raise ValueError("Invalid arccsc-domain value")
        return np.arcsin(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arccsc / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        a.backward(np.select([np.abs(a.data) != 1], [-grad / (np.abs(a.data) * np.sqrt(a.data ** 2 - 1))]), **kwargs)


class Arcsec(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        if np.any(-1 < a) and np.any(a < 1):
            raise ValueError("Invalid arcsec-domain value")
        return np.arccos(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arcsec / dx at x = -1, 1 returns 0, not NaN
        a = self.variables[index]
        a.backward(np.select([np.abs(a.data) != 1], [grad / (np.abs(a.data) * np.sqrt(a.data ** 2 - 1))]), **kwargs)


class Arccot(MultiVarOperation):
    def __call__(self, a):
        self.variables = (a,)
        return np.piecewise(a.data, [a.data == 0, a.data != 0], [np.pi / 2, lambda x: np.arctan(1 / x)])

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(-grad / (1 + a.data ** 2), **kwargs)
