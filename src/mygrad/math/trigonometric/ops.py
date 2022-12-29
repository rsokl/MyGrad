import numpy as np

from mygrad.operation_base import BinaryUfunc, Operation, UnaryUfunc

__all__ = [
    "Sin",
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
    "Arccot",
    "Arctan2",
]


class Sin(UnaryUfunc):
    numpy_ufunc = np.sin

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * np.cos(a.data)


def _dsinc(x):
    x = x * np.pi
    return (x * np.cos(x) - np.sin(x)) / x**2


class Sinc(Operation):
    """f(a) -> sin(pi*a)/(pi*a)"""

    def __call__(self, a):
        self.variables = (a,)
        return np.sinc(a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        x = a.data

        # TODO: use tiny
        near_0 = np.isclose(x, 0, atol=1e-20)
        return (
            np.pi * grad * np.piecewise(x, [near_0, ~near_0], [np.zeros_like, _dsinc])
        )


class Cos(UnaryUfunc):
    numpy_ufunc = np.cos

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * -np.sin(a.data)


class Tan(UnaryUfunc):
    numpy_ufunc = np.tan

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / np.cos(a.data) ** 2


class Csc(Operation):
    """f(a) -> csc(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.sin(a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * -np.cos(a.data) / np.sin(a.data) ** 2


class Sec(Operation):
    """f(a) -> sec(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.cos(a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * np.sin(a.data) / np.cos(a.data) ** 2


class Cot(Operation):
    """f(a) -> cot(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return 1 / np.tan(a.data)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return -grad / np.sin(a.data) ** 2


class Arcsin(UnaryUfunc):
    numpy_ufunc = np.arcsin

    def backward_var(self, grad, index, **kwargs):
        # d arcsin / dx at x = -1, 1 returns 0, not NaN
        (a,) = self.variables
        return np.select([np.abs(a.data) != 1], [grad / np.sqrt(1 - a.data**2)])


class Arccos(UnaryUfunc):
    numpy_ufunc = np.arccos

    def backward_var(self, grad, index, **kwargs):
        # d arccos / dx at x = -1, 1 returns 0, not NaN
        (a,) = self.variables
        return np.select([np.abs(a.data) != 1], [-grad / np.sqrt(1 - a.data**2)])


class Arctan(UnaryUfunc):
    numpy_ufunc = np.arctan

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / (1 + a.data**2)


class Arccsc(Operation):
    """f(a) -> arccsc(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return np.arcsin(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arccsc / dx at x = -1, 1 returns 0, not NaN
        (a,) = self.variables
        return np.select(
            [np.abs(a.data) != 1], [-grad / (np.abs(a.data) * np.sqrt(a.data**2 - 1))]
        )


class Arcsec(Operation):
    """f(a) -> arcsec(a)"""

    def __call__(self, a):
        self.variables = (a,)
        return np.arccos(1 / a.data)

    def backward_var(self, grad, index, **kwargs):
        # d arcsec / dx at x = -1, 1 returns 0, not NaN
        (a,) = self.variables
        return np.select(
            [np.abs(a.data) != 1], [grad / (np.abs(a.data) * np.sqrt(a.data**2 - 1))]
        )


class Arccot(Operation):
    """f(a) -> arccot(a)"""

    def __call__(self, a):
        self.variables = (a,)
        arr = a.data
        is_zero = arr == 0
        return np.piecewise(
            arr,
            [is_zero, np.logical_not(is_zero)],
            [np.pi / 2, lambda x: np.arctan(1 / x)],
        )

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return -grad / (1 + a.data**2)


class Arctan2(BinaryUfunc):
    numpy_ufunc = np.arctan2

    def __init__(self):
        super().__init__()
        self.cached_denom = None

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if self.cached_denom is None:
            self.cached_denom = a.data**2 + b.data**2

        if index == 0:
            return grad * b.data / self.cached_denom
        else:
            return -1.0 * grad * a.data / self.cached_denom
