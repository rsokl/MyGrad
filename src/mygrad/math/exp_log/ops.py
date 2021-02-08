import numpy as np

from mygrad.operation_base import BinaryUfunc, UnaryUfunc

__all__ = [
    "Exp",
    "Exp2",
    "Expm1",
    "Log",
    "Log2",
    "Log10",
    "Log1p",
    "Logaddexp",
    "Logaddexp2",
]


class Exp(UnaryUfunc):
    numpy_ufunc = np.exp

    def backward_var(self, grad, index, **kwargs):
        return grad * np.exp(self.variables[index].data)


class Exp2(UnaryUfunc):
    numpy_ufunc = np.exp2

    def backward_var(self, grad, index, **kwargs):
        return grad * np.exp2(self.variables[index].data) * np.log(2)


class Expm1(UnaryUfunc):
    numpy_ufunc = np.expm1

    def backward_var(self, grad, index, **kwargs):
        return grad * np.exp(self.variables[index].data)


class Logaddexp(BinaryUfunc):
    numpy_ufunc = np.logaddexp

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            return grad / (1 + np.exp(b.data - a.data))
        elif index == 1:
            return grad / (1 + np.exp(a.data - b.data))
        else:  # pragma: no cover
            raise IndexError(f"Back-propagation through tensor-{index}")


class Logaddexp2(BinaryUfunc):
    numpy_ufunc = np.logaddexp2

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            return grad / (1 + 2 ** (b.data - a.data))
        elif index == 1:
            return grad / (1 + 2 ** (a.data - b.data))
        else:  # pragma: no cover
            raise IndexError(f"Back-propagation through tensor-{index}")


class Log(UnaryUfunc):
    numpy_ufunc = np.log

    def backward_var(self, grad, index, **kwargs):
        return grad / self.variables[index].data


class Log2(UnaryUfunc):
    numpy_ufunc = np.log2

    def backward_var(self, grad, index, **kwargs):
        return grad / (self.variables[index].data * np.log(2))


class Log10(UnaryUfunc):
    numpy_ufunc = np.log10

    def backward_var(self, grad, index, **kwargs):
        return grad / (self.variables[index].data * np.log(10))


class Log1p(UnaryUfunc):
    numpy_ufunc = np.log1p

    def backward_var(self, grad, index, **kwargs):
        return grad / (1 + self.variables[index].data)
