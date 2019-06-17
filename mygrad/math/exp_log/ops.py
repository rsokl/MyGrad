import numpy as np

from mygrad.operation_base import BroadcastableOp, Operation

__all__ = ["Exp", "Exp2", "Expm1",
           "Log", "Log2", "Log10", "Log1p",
           "Logaddexp", "Logaddexp2"]


class Exp(Operation):
    def __call__(self, a):
        """ f(a) -> exp(a)

            Parameters
            ----------
            a : mygrad.Tensor

            Returns
            -------
            numpy.ndarray"""
        self.variables = (a,)
        return np.exp(a.data)

    def backward_var(self, grad, index, **kwargs):
        return grad * np.exp(self.variables[index].data)

class Exp2(Operation):
    def __call__(self, a):
        """ f(a) -> 2^a

            Parameters
            ----------
            a : mygrad.Tensor

            Returns
            -------
            numpy.ndarray"""
        self.variables = (a,)
        return np.exp2(a.data)
    
    def backward_var(self, grad, index, **kwargs):
        return grad * np.exp2(self.variables[index].data) * np.log(2)

class Expm1(Operation):
    """ f(a) -> exp(a) - 1

        This function provides greater precision than exp(x) - 1
        for small values of x."""

    def __call__(self, a):
        self.variables = (a,)
        return np.expm1(a.data)

    def backward_var(self, grad, index, **kwargs):
        return grad * np.exp(self.variables[index].data)


class Logaddexp(BroadcastableOp):
    """f(a,b) -> log(exp(a) + exp(b))"""
    def __call__(self, a, b):
        self.variables = (a, b)
        out = np.logaddexp(a.data, b.data)
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            return grad / (1 + np.exp(b.data - a.data))
        elif index == 1:
            return grad / (1 + np.exp(a.data - b.data))
        else:
            raise IndexError


class Logaddexp2(BroadcastableOp):
    """f(a,b) -> log2(exp(a) + exp(b))"""
    def __call__(self, a, b):
        self.variables = (a, b)
        out = np.logaddexp2(a.data, b.data)
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            return grad / (1 + 2 ** (b.data - a.data))
        elif index == 1:
            return grad / (1 + 2 ** (a.data - b.data))
        else:
            raise IndexError


class Log(Operation):
    """ f(a) -> ln(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.log(a.data)

    def backward_var(self, grad, index, **kwargs):
        return grad / self.variables[index].data


class Log2(Operation):
    def __call__(self, a):
        """ f(a) -> log2(a)"""
        self.variables = (a,)
        return np.log2(a.data)

    def backward_var(self, grad, index, **kwargs):
        return grad / (self.variables[index].data * np.log(2))


class Log10(Operation):
    """ f(a) -> log10(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.log10(a.data)

    def backward_var(self, grad, index, **kwargs):
        return grad / (self.variables[index].data * np.log(10))


class Log1p(Operation):
    """ f(a) -> ln(1 + a)

        log1p is accurate for x so small that 1 + x == 1 in
        floating-point accuracy."""
    def __call__(self, a):
        self.variables = (a,)
        return np.log1p(a.data)

    def backward_var(self, grad, index, **kwargs):
        return grad / (1 + self.variables[index].data)
